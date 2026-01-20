from tqdm import tqdm
import torch
import torchvision.utils as vutils
from .fm_model import ModelIF
from omegaconf import ListConfig
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
from data.utils import get_trajectory_from_speeds_and_yaw_rates_batch, RunningNorm


def text_on_image(images, texts):
    # image is a tensor of shape (B, C, H, W)
    # texts is a list of strings of length B
    for i in range(images.shape[0]):
        image = images[i]
        text = texts[i]

        # Convert the image tensor to a PIL Image
        image = PIL.Image.fromarray((((image + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        # Draw the text on the image
        draw = PIL.ImageDraw.Draw(image)
        draw.text((10, 10), text, fill="red", font_size=20)

        # Convert the image back to a tensor
        image = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0) * 2 - 1  # scale back to [-1, 1]
        images[i] = image

    return images


class ModelIFSteering(ModelIF):

    def __init__(self, *, num_context_frames, steering_drop=(0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self.steering_drop = steering_drop
        self.num_context_frames = num_context_frames

    def get_context_and_target(self, x):
        if hasattr(self.vit, 'num_frames'):
            assert x.size(1) == self.vit.num_frames, f"Input tensor must have {self.vit.num_frames} frames, but got {x.size(1)} frames."
            num_frames = self.vit.num_frames
        else:
            num_frames = x.size(1)
        num_pred_frames = self.num_pred_frames
        context = x[:,:num_frames-num_pred_frames].clone()
        target = x[:,num_frames-num_pred_frames:]
        assert context.shape[1] == num_frames - num_pred_frames, f"Context shape {context.shape} does not match expected {num_frames - num_pred_frames}"
        assert target.shape[1] == num_pred_frames, f"Target shape {target.shape} does not match expected {num_pred_frames}"
        return context, target

    def get_input(self, batch, k):
        if type(batch) == dict:
            x = batch['images']
            frame_rate = batch['frame_rate']
            if 'steering' in batch:
                steering = batch['steering'][:, -self.num_pred_frames]   # only take the last steering value before the target frames
            else:
                steering = None
        else:
            x = batch
            frame_rate = None
            steering = None
        assert len(x.shape) == 5, 'input must be 5D tensor'
        
        if k == 'images':
            return x, frame_rate
        return x, frame_rate, steering

    def drop_steering(self, steering):
        # Randomly drop some steering values
        # sample for each dimension of steering
        mask = torch.cat([torch.rand(steering.shape[0], 1) < self.steering_drop[i] for i in range(steering.shape[1])], dim=1)
        mask = mask.to(steering.device)
        steering[mask] = float('nan')
        return steering

    def log_losses(self, loss, loss_recon, loss_sem):
        # log loss for each predicted frame
        for frame_idx in range(self.num_pred_frames):
            self.log(f"train/loss_frame_{frame_idx}", loss[:, frame_idx].mean().item(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
        self.log("train/loss", loss.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_recon", loss_recon.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_sem", loss_sem.mean().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        images, frame_rate, steering = self.get_input(batch, 'steering')

        if steering is not None and self.steering_drop:
            steering = self.drop_steering(steering)

        x = self.encode_frames(images)
        context, target = self.get_context_and_target(x)

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)
        
        pred = self.vit(target_t, context, t, frame_rate=frame_rate, steering=steering)
        
        # -dxt/dt
        target = self.A(t) * target + self.B(t) * noise
        loss = ((pred.float() - target.float()) ** 2)
        loss_recon, loss_sem = torch.split(loss, loss.size(2)//2, dim=2)  # dim 2 is the channel dimension
        
        self.log_losses(loss, loss_recon, loss_sem)

        if loss.isnan().any():
            raise ValueError("loss contains nans")
        
        return loss.mean()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None, steering=None, return_sample=False):
        self.ema_vit.eval()
        self.vit.eval()
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device
        
        if not latent:
            context = self.encode_frames(images)
        else:
            context = images.clone()

        if frame_rate is None:
            frame_rate = torch.full_like( torch.ones((num_samples,)), 5, device=device)
        
        num_pred_frames = self.num_pred_frames
        
        input_h, input_w = self.vit.input_size[0], self.vit.input_size[1] if isinstance(self.vit.input_size, (list, tuple, ListConfig)) else self.vit.input_size
        target_t = torch.randn(num_samples, num_pred_frames, self.vit.in_channels, input_h, input_w, device=device)
        
        t_steps = torch.linspace(1, 0, NFE + 1, device=device)

        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i].repeat(target_t.shape[0])
                neg_v = net(target_t, context, t=t * self.timescale, frame_rate=frame_rate, steering=steering)
                dt = t_steps[i] - t_steps[i+1] 
                dw = torch.randn(target_t.size()).to(target_t.device) * torch.sqrt(dt)
                diffusion = dt
                target_t  = target_t + neg_v * dt + eta *  torch.sqrt(2 * diffusion) * dw
                
        if return_sample:
            images = self.decode_frames(target_t.clone())
            return target_t, images
        else:
            return target_t
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        images, frame_rate, steering = self.get_input(batch, 'steering')
        N = min(5, images.size(0))
        images = images[:N]
        if self.use_precomputed_training_inputs and images.shape[-2:] == self.vit.input_size:
            images = self.decode_frames(images)
        
        # write steering values on last frame
        if steering is not None:
            steering_strings = [", ".join([f"{s[i]:.2f}" for i in range(s.shape[0])]) for s in steering[:N].cpu().numpy()]
            images[:,-1] = text_on_image(images[:,-1], steering_strings)
        
        steering = steering[:N] if steering is not None else None
        frame_rate = frame_rate[:N] if frame_rate is not None else None
        b, f, e, h, w = images.size()

        l_visual_recon = [images[:,f] for f in range(images.size(1))]
        l_visual_recon_ema = [images[:,f] for f in range(images.size(1))]

        images = images[:,:-self.num_pred_frames] if f > 1 else None

        # sample
        samples = self.sample(images, eta=0.0, NFE=30, sample_with_ema=False, num_samples=N, frame_rate=frame_rate, steering=steering, return_sample=True)[1]
        samples = samples[:N]

        for frame_idx in range(samples.size(1)):
            l_visual_recon.append(samples[:, frame_idx])

        l_visual_recon = torch.cat(l_visual_recon, dim=0)
        chunks = torch.chunk(l_visual_recon, 2 + 2, dim=0)
        reco_sample = torch.cat(chunks, 0)
        reco_sample = vutils.make_grid(reco_sample, nrow=N, padding=2, normalize=False,)
        
        # sample
        samples_ema = self.sample(images, eta=0.0, NFE=30, sample_with_ema=True, num_samples=N, steering=steering, return_sample=True)[1]
        samples_ema = samples_ema[:N]

        for frame_idx in range(samples_ema.size(1)):
            l_visual_recon_ema.append(samples_ema[:, frame_idx])

        l_visual_recon_ema = torch.cat(l_visual_recon_ema, dim=0)
        chunks_ema = torch.chunk(l_visual_recon_ema, 2 + 2, dim=0)
        reco_sample_ema = torch.cat(chunks_ema, 0)
        reco_sample_ema = vutils.make_grid(reco_sample_ema, nrow=N, padding=2, normalize=False)

        ret = {"sampled": reco_sample, "sampled_ema": reco_sample_ema}
        self.vit.train()
        return ret

    def roll_out(self, data_batch, num_gen_frames, latent_input=True, eta=0.0, NFE=20, sample_with_ema=True):
        # raise NotImplementedError("Need to adapt to the new input format, which passes the whole data_batch")
        assert not latent_input, "Latent input is not supported for this model class."
        
        frame_rate = data_batch['frame_rate']
        images = data_batch['images']
        steering = data_batch['steering']
        
        x_c = self.encode_frames(images)
        x_all = [x_c.clone()]

        for idx in tqdm(range(num_gen_frames)):
            # for steering we take the first frame after context
            steering_idx = self.num_context_frames + idx
            x_last_t = self.sample(images=x_c, steering=steering[:, steering_idx], latent=True,
                                   eta=eta, NFE=NFE, sample_with_ema=sample_with_ema,
                                   num_samples=x_c.size(0), frame_rate=frame_rate)
            x_all.append(x_last_t)
            # update conditioning autoregressively
            x_c = torch.cat([x_c[:, self.num_pred_frames:], x_last_t], dim=1)
        
        x_all = torch.cat(x_all, dim=1)
        samples = self.decode_frames(x_all)

        return x_all, samples
    
    

class ModelIFGoalPointCond(ModelIFSteering):
    """
    Provide a goal point as additional conditioning to the model, computed from the odometry trajectory.
    Predict the the frame(s) leading to the goal point.
    Optionally also predict the trajectory itself as an auxiliary task.
    """
    def __init__(self, *, goal_distance_frames, orientation_conditioning=False, **kwargs):
        super().__init__(**kwargs)
        # assert not np.sum(self.steering_drop), "steering_drop not supported for ModelIFGoalPointCond yet."
        self.goal_distance_frames = goal_distance_frames
        self.orientation_conditioning = orientation_conditioning

    def get_input(self, batch, k):
        """
        We need all the frames to predict, but only the very last trajectory point (goal point) for conditioning.
        """
        if type(batch) == dict:
            x = batch['images'][:, :self.num_context_frames + self.num_pred_frames]
            frame_rate = batch['frame_rate']
            # check that all frame rates are the same, pick the first one
            assert (frame_rate == frame_rate[0]).all(), "All frame rates in the batch must be the same."
            if 'steering' in batch:
                # convert speed/yawrate to trajectory
                trajectory_heading = get_trajectory_from_speeds_and_yaw_rates_batch(
                    speeds=batch['steering'][:, :, 0],
                    yaw_rates=batch['steering'][:, :, 1],
                    dt=1.0/frame_rate
                )
                # only take the steering value corresponding to the last predicted frame, i.e. the goal point
                steering = trajectory_heading[:, self.num_context_frames+self.goal_distance_frames-1]
                if not self.orientation_conditioning:
                    steering = steering[:, :2] # only the (x, y) coordinates, not the heading
            else:
                steering = None
        else:
            raise ValueError("Input batch must be a dict containing 'images' and optionally 'steering'.")
        assert len(x.shape) == 5, 'input must be 5D tensor'
        
        if k == 'images':
            return x, frame_rate
        
        return x, frame_rate, steering

    def roll_out(self, data_batch, num_gen_frames, latent_input=True, eta=0.0, NFE=20, sample_with_ema=True):
        assert not latent_input, "Latent input is not supported for this model class."
        frame_rate = data_batch['frame_rate']
        images = data_batch['images']
        speeds_yawrates = data_batch['steering']
        
        # encode the first num_context_frames frames
        x_c = self.encode_frames(images[:, :self.num_context_frames])
        x_all = [x_c.clone()]
        goals_all = []

        # num_steps should be such that we predict *at least* num_pred_frames frames, i.e. round up
        num_steps = num_gen_frames // self.num_pred_frames + (1 if num_gen_frames % self.num_pred_frames > 0 else 0)
        for idx in tqdm(range(num_steps)):
            current_frame_idx = idx * self.num_pred_frames + self.num_context_frames - 1
            goal_idx = current_frame_idx + self.goal_distance_frames
            # at every iteration we must:
            # - update the image context by appending the predicted frames and selecting the last num_context_frames frames
            # - update the goal point by selecting the trajectory point corresponding to the goal index, re-centered to the ego position
            centered_traj = get_trajectory_from_speeds_and_yaw_rates_batch(
                speeds=speeds_yawrates[:, current_frame_idx:goal_idx+1, 0],
                yaw_rates=speeds_yawrates[:, current_frame_idx:goal_idx+1, 1],
                dt=1.0/frame_rate
            )
            goal = centered_traj[:, -1, :2] if not self.orientation_conditioning else centered_traj[:, -1, :]
            x_last_t = self.sample(images=x_c, steering=goal, latent=True,
                                   eta=eta, NFE=NFE, sample_with_ema=sample_with_ema,
                                   num_samples=x_c.size(0), frame_rate=frame_rate)
            x_all.append(x_last_t)
            goals_all.append(goal)
            # update conditioning autoregressively
            x_c = torch.cat([x_c[:, self.num_pred_frames:], x_last_t], dim=1)
        
        x_all = torch.cat(x_all, dim=1)[:, :self.num_context_frames+num_gen_frames]  # in case we generated too many frames, cut the excess
        samples = self.decode_frames(x_all)

        return {
            "latent": x_all.cpu(), 
            "images": samples.cpu(), 
            "goals": torch.stack(goals_all, dim=1).cpu()  # (B, num_steps, 2 or 3)
        }
    
    