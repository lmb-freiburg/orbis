import os
import math
from copy import deepcopy
from collections import OrderedDict

import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm


from util import instantiate_from_config

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
        
def init_ema_model(model):
    ema_model = deepcopy(model)
    requires_grad(ema_model, False)
    update_ema(ema_model, model, decay=0)
    ema_model.eval()
    return ema_model


class Model(pl.LightningModule):
    """
    Base model class for the flow matching version of the world model.
    
    The methods training_step and validation_step follow PyTorch Lightning conventions.
    The class additionally implements the following methods:
    - sample: to generate new samples from the model (from start to end), given a set of input frames.
    - roll_out: to generate a sequence of samples autoregressively, given a set of starting input frames and a number of frames to generate.
    """
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, sigma_min=1e-5, timescale=1.0, enc_scale=4, warmup_steps=5000, min_lr_multiplier=0.1):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        
        # Training parameters
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        
        # denoising backbone
        self.vit = self.build_generator(generator_config)
        
        # Tokenizer
        self.ae = self.build_tokenizer(tokenizer_config)

        # Loss and Optimizer
        self.enc_scale = enc_scale

        # EMA
        self.ema_vit = init_ema_model(self.vit)
        self.sigma_min = sigma_min
        self.timescale = timescale
        
        
    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)


    def build_tokenizer(self, tokenizer_config):
        """
        Instantiate the tokenizer model from the config.
        """
        
        tokenizer_folder = os.path.expandvars(tokenizer_config.folder)
        ckpt_path = tokenizer_config.ckpt_path if tokenizer_config.ckpt_path else "checkpoints/last.ckpt"
        # Load config and create
        tokenizer_config = OmegaConf.load(os.path.join(tokenizer_folder, "config.yaml"))
        model = instantiate_from_config(tokenizer_config.model)
        # Load checkpoint
        checkpoint = torch.load(os.path.join(tokenizer_folder, ckpt_path), map_location="cpu", weights_only=True)["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        return model
    
    def build_generator(self, generator_config):
        """
        Instantiate the denoising backbone model from the config.
        """
        model = instantiate_from_config(generator_config)
        return model
    

    def get_warmup_scheduler(self, optimizer, warmup_steps=1, min_lr_multiplier=1.0):
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            else:
                progress = (min(step, total_steps) - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr_multiplier) * cosine_decay + min_lr_multiplier
                return decayed      
        
        return LambdaLR(optimizer, lr_lambda)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.vit.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        return  [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def get_input(self, batch, k):
        if type(batch) == dict:
            x = batch[k]
            frame_rate = batch['frame_rate']
        else:
            x = batch
            frame_rate = None
        assert len(x.shape) == 5, 'input must be 5D tensor'
        return x, frame_rate
    
    
    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    @torch.no_grad()
    def encode_frames(self, images):
        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1
        x = self.ae.encode(images)['continuous']
        x = x * (self.enc_scale)
        x = rearrange(x, '(b f) e h w -> b f e h w',b=b, f=f)
        return x
    
    @torch.no_grad()
    def decode_frames(self, x):
        frame = x[:,0] 
        frame = frame / (self.enc_scale)

        frame = self.ae.post_quant_conv(frame)
        samples = (self.ae.decoder(frame)).unsqueeze(1)
        for idx in range(1, x.shape[1]):
            frame = x[:,idx] 
            frame = frame / (self.enc_scale)
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            samples = torch.cat([samples, (frame).unsqueeze(1)], dim=1)
        return samples
  
    
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')

        x = self.encode_frames(images)

        b, f, e, h, w = x.size()
        
        if f == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:,:-1].clone()
            target = x[:,-1]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)
        
        target_t = target_t.unsqueeze(1)
        
        pred = self.vit(target_t, context, t, frame_rate=frame_rate)
        
        # -dxt/dt
        target = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - target.float()) ** 2)
        
        loss = loss.mean()
            
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA at the end of each epoch as fallback"""
        update_ema(self.ema_vit, self.vit)

    def validation_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        # VQGAN encoding to img tokens
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()
        
        if f == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:,:-1].clone()
            target = x[:,-1]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)
        
        target_t = target_t.unsqueeze(1)
        
        pred = self.vit(target_t, context, t, frame_rate=frame_rate)
        
        # -dxt/dt
        target = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - target.float()) ** 2)

        loss = loss.mean()
            
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)


    def roll_out(self, x_0, num_gen_frames=25, latent_input=True, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8):
        b, f = x_0.size(0), x_0.size(1)
        if latent_input:
            x_c = x_0.clone()
        else:
            x_c = self.encode_frames(x_0)

        x_all = x_c.clone()
        samples = []
        for idx in tqdm(range(num_gen_frames), desc="Rolling out frames", leave=False):
            x_last, sample = self.sample(images=x_c, latent=True, eta=eta, NFE=NFE, sample_with_ema=sample_with_ema, num_samples=num_samples)
            
            x_all = torch.cat([x_all, x_last.unsqueeze(1)], dim=1)
            x_c = torch.cat([x_c[:, 1:], x_last.unsqueeze(1)], dim=1)
            samples.append(sample)
        
        samples = torch.cat(samples, dim=1)

        return x_all, samples
    
    @torch.no_grad()
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None):
        self.ema_vit.eval()
        self.vit.eval()
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device
        
        if images is not None:
            b, f, e, h, w = images.size()
            if not latent:
                context = self.encode_frames(images)
                b, f, e, h, w = context.size()
            else:
                context = images.clone()
        else:
            context = None

        if frame_rate is None:
            frame_rate = torch.full_like( torch.ones((num_samples,)), 5, device=device)
            
        input_h, input_w = self.vit.input_size[0], self.vit.input_size[1] if isinstance(self.vit.input_size, (list, tuple, ListConfig)) else self.vit.input_size
        target_t = torch.randn(num_samples, 1, self.vit.in_channels, input_h, input_w, device=device)
        
        t_steps = torch.linspace(1, 0, NFE + 1, device=device)

        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i].repeat(target_t.shape[0])
                neg_v = net(target_t, context, t=t * self.timescale, frame_rate=frame_rate)
                dt = t_steps[i] - t_steps[i+1] 
                dw = torch.randn(target_t.size()).to(target_t.device) * torch.sqrt(dt)
                diffusion = dt
                target_t  = target_t + neg_v * dt + eta *  torch.sqrt(2 * diffusion) * dw
        last_frame = target_t.clone()

        images = self.decode_frames(last_frame)
        self.vit.train()
        return target_t.squeeze(1), images

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        images, frame_rate = self.get_input(batch, 'images')
        N = min(5, images.size(0))
        images = images[:N]
        frame_rate = frame_rate[:N] if frame_rate is not None else None
        b, f, e, h, w = images.size()

        l_visual_recon = [images[:,f] for f in range(images.size(1))]
        l_visual_recon_ema = [images[:,f] for f in range(images.size(1))]
        
        
        images = images[:,:-1] if f > 1 else None

        # sample
        samples = self.sample(images, eta=0.0, NFE=30, sample_with_ema=False, num_samples=N, frame_rate=frame_rate)[1]
        
        # Only keep N first generated frame
        samples = samples[:N, 0]

        l_visual_recon.append(samples)

        l_visual_recon = torch.cat(l_visual_recon, dim=0)
        chunks = torch.chunk(l_visual_recon, 2 + 2, dim=0)
        sampled = torch.cat(chunks, 0)
        sampled = vutils.make_grid(sampled, nrow=N, padding=2, normalize=False,)
        
        # sample
        samples_ema = self.sample(images, eta=0.0, NFE=30, sample_with_ema=True, num_samples=N)[1]
        # Only keep N first generated frame
        samples_ema = samples_ema[:N, 0]

        l_visual_recon_ema.append(samples_ema)

        l_visual_recon_ema = torch.cat(l_visual_recon_ema, dim=0)
        chunks_ema = torch.chunk(l_visual_recon_ema, 2 + 2, dim=0)
        sampled_ema = torch.cat(chunks_ema, 0)
        sampled_ema = vutils.make_grid(sampled_ema, nrow=N, padding=2, normalize=False)

        log["ema_sampled"] = sampled_ema
        log["sampled"] = sampled
        self.vit.train()
        return log
    

class ModelIF(Model):
    """
    Flow matching model for token factorization latents (IF: Image Folder)
    """
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, 
                 sigma_min=1e-5, timescale=1.0, enc_scale=1.89066, enc_scale_dino=3.45062, warmup_steps=5000, min_lr_multiplier=0.1):
        super().__init__(
            tokenizer_config=tokenizer_config,
            generator_config=generator_config,
            adjust_lr_to_batch_size=adjust_lr_to_batch_size,
            sigma_min=sigma_min,
            timescale=timescale,
            enc_scale=enc_scale,
            warmup_steps=warmup_steps,
            min_lr_multiplier=min_lr_multiplier
        )
        self.enc_scale_dino = enc_scale_dino
        
    
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')

        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        
        if f == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:,:-1].clone()
            target = x[:,-1]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)
        
        target_t = target_t.unsqueeze(1)
        
        pred = self.vit(target_t, context, t, frame_rate=frame_rate)
        
        # -dxt/dt
        target = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - target.float()) ** 2)
        
        loss_recon = loss[:,:loss.size(1)//2].mean()
        loss_sem = loss[:,loss.size(1)//2:].mean()
        loss = loss.mean()
            
        self.log("train/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_recon", loss_recon.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_sem", loss_sem.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss 



    def validation_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()
        
        if f == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:,:-1].clone()
            target = x[:,-1]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)
        
        target_t = target_t.unsqueeze(1)
        
        
        pred = self.vit(target_t, context, t, frame_rate=frame_rate)
        
        # -dxt/dt
        target = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - target.float()) ** 2)

        loss_recon = loss[:,:loss.size(1)//2].mean()
        loss_sem = loss[:,loss.size(1)//2:].mean()
        loss = loss.mean()
            
        self.log("val/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_recon", loss_recon.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_sem", loss_sem.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)


        
    @torch.no_grad()
    def encode_frames(self, images):
        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            context=None
            f = 1
        x = self.ae.encode(images)["continuous"]
        x0 = x[0] * self.enc_scale
        x1 = x[1] * self.enc_scale_dino
        x = torch.cat([x0, x1], dim=1)
        x = rearrange(x, '(b f) e h w -> b f e h w',b=b, f=f)
        return x
    
    @torch.no_grad()
    def decode_frames(self, x):
        frame = x[:,0] 
        half = frame.size(1) // 2
        frame = torch.cat([
            frame[:, :half] / self.enc_scale,
            frame[:, half:] / self.enc_scale_dino
        ], dim=1)
        frame = self.ae.post_quant_conv(frame)
        samples = (self.ae.decoder(frame)).unsqueeze(1)
        for idx in range(1, x.shape[1]):
            frame = x[:,idx] 
            half = frame.size(1) // 2
            frame = torch.cat([
                frame[:, :half] / self.enc_scale,
                frame[:, half:] / self.enc_scale_dino
            ], dim=1)
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            samples = torch.cat([samples, (frame).unsqueeze(1)], dim=1)
        return samples
