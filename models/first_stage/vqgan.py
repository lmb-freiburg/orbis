import math
import random
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config

class VQModel(pl.LightningModule):
    """
    VQGAN model: vector-quantized autoencoder with adversarial training.
    """
    
    def __init__(
        self,
        encoder_config,
        decoder_config,
        quantizer_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        distill_model_type="VIT_DINOv2",  # Options: VIT_DINO, CNN, VIT_DINOv2, etc.
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__()

        ignore_keys = ignore_keys or []
        self.automatic_optimization = False
        self.grad_acc_steps = grad_acc_steps
        self.monitor = monitor
        self.distill_model_type = distill_model_type
        self.cont_ratio_trainig = cont_ratio_trainig
        self.only_decoder=only_decoder
        self.min_lr_multiplier = min_lr_multiplier
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Decoder uses encoder params if none provided
        if not hasattr(decoder_config, "params"):
            decoder_config.params = encoder_config.params

        # Instantiate core components
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        self.quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)

        self.if_distill_loss = False if loss_config.params.get('distill_loss_weight', 0.0) == 0.0 else True
        
        # Image and patch size
        self.image_size = encoder_config.params["resolution"]
        self.patch_size = encoder_config.params["patch_size"]

        self._init_distill_model(distill_model_type, encoder_config, decoder_config, quantizer_config)

    def _init_distill_model(self, distill_type, encoder_cfg, decoder_cfg, quantizer_cfg):
        image_size = encoder_cfg.params["resolution"]
        patch_size = encoder_cfg.params["patch_size"]
        q_e_dim = quantizer_cfg.params["e_dim"]
        z_channels = decoder_cfg.params["z_channels"]

        def conv1x1(in_c, out_c): return nn.Conv2d(in_c, out_c, 1)

        if distill_type == "VIT_DINO":
            self.distill = timm.create_model("timm/vit_base_patch16_224.dino", img_size=image_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2":
            img_size = self._compute_scaled_size(image_size, patch_size)
            self.distill = timm.create_model("timm/vit_base_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2g":
            img_size = int(image_size * 14 / patch_size)
            self.distill = timm.create_model("timm/vit_giant_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, 1536)
        elif distill_type == "VIT_DINOv2_large":
            img_size = int(image_size * 14 / patch_size)
            self.distill = timm.create_model("timm/vit_large_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2_large_reg4":
            img_size = int(image_size * 14 / patch_size)
            self.distill = timm.create_model("timm/vit_large_patch14_reg4_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "SAM_VIT":
            self.distill = timm.create_model("samvit_large_patch16.sa1b", pretrained=True)
            self.post_quant_conv_distill = nn.Identity()
        elif distill_type == "SAM_VIT_w_conv":
            self.distill = timm.create_model("samvit_large_patch16.sa1b", pretrained=True)
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)

        elif distill_type == "depth_anything_VIT_L14":
            self.distill = timm.create_model("vit_large_patch14_dinov2.lvd142m", img_size=224, pretrained=False)
            state_dict = torch.load("./pretrained_models/depth_anything_vitl14.pth")
            state_dict = {k.replace("pretrained.", "", 1): v for k, v in state_dict.items()}
            state_dict["pos_embed"] = resample_abs_pos_embed(state_dict["pos_embed"], new_size=(16, 16))
            self.distill.load_state_dict(state_dict, strict=False)
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)


    @staticmethod
    def _compute_scaled_size(image_size, patch_size):
        if isinstance(image_size, int):
            return [image_size * 14 // patch_size] * 2
        return [image_size[0] * 14 // patch_size, image_size[1] * 14 // patch_size]
    
    def get_input(self, batch):
        x = batch
        return x.float()

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        ret = self.quantize(h)
        ret["continuous"] = h
        return ret
        
    def decode(self, quant):
        distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        return self.decoder(quant2), distill_conv_out
    
    def forward(self, input):
        encoded = self.encode(input)
        if torch.rand(1) > self.cont_ratio_trainig:
            dec, distill_conv_out = self.decode(encoded["quantized"])
        else:
            dec, distill_conv_out = self.decode(encoded["continuous"])
        return dec, (encoded['quantization_loss'], encoded['entropy_loss']), distill_conv_out
    
    def forward_se(self, input):
        random_scale = [random.choice(self.scale_equivariance[0]), random.choice(self.scale_equivariance[1])]
        downscale_factor = [1/random_scale[0], 1/random_scale[1]]
        encoded = self.encode(input)
        if torch.rand(1) > self.cont_ratio_trainig:
            dec, distill_conv_out = self.decode(encoded["quantized"])
            quant_se = F.interpolate(encoded["quantized"], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(quant_se)[0]
        else:
            dec, distill_conv_out = self.decode(encoded["continuous"])
            latents_se = F.interpolate(encoded["continuous"], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(latents_se)[0]

        input_se = F.interpolate(input, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        decs = [dec, dec_se]
        inputs = [input, input_se]
        return inputs, decs, (encoded['quantization_loss'], encoded['entropy_loss']), distill_conv_out

    def distill_loss(self, distill_output, decoder_distill_output):
        #print(f'DINO loss calculation')
        if 'VIT' in self.distill_model_type:
            if 'reg4' in self.distill_model_type:
                distill_output = distill_output[:, 5:, :] # [CLS, Register*4, Embeddings]
            elif 'reg4' not in self.distill_model_type and 'DINO' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :] # uncomment for DINOv1 
            elif 'depth_anything' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :]
            elif self.distill_model_type == 'SAM_VIT':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                distill_output = F.normalize(distill_output, p=2, dim=2) # without post_conv layer
            elif self.distill_model_type == 'SAM_VIT_w_conv':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                # without L2 normalization
            distill_output = distill_output.permute(0, 2, 1).contiguous()
        
        elif self.distill_model_type == 'CNN':
            distill_output = distill_output.view(distill_output.shape[0], distill_output.shape[1], -1)
        decoder_distill_output = decoder_distill_output.view(decoder_distill_output.shape[0], decoder_distill_output.shape[1], -1)
        #print (f'distill_output.shape: {distill_output.shape} decoder_distill_output.shape: {decoder_distill_output.shape}')
        cos_similarity = F.cosine_similarity(decoder_distill_output, distill_output, dim=1)
        cosine_loss = 1 - cos_similarity
        distill_loss = cosine_loss.mean()
        return distill_loss

    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr) * cosine_decay + min_lr
                return decayed
            #return 1.0        
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.min_lr_multiplier)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, decoder_distill_output = self(x)

        distill_loss = self.distill_loss(self.get_distill_gt(x), decoder_distill_output) if self.if_distill_loss else torch.tensor(0.0, device=x.device)
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/distill_loss", distill_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        _, log_dict_disc = self.loss(qloss, distill_loss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return self.log_dict

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()
        
        if self.only_decoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.quantize.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv_distill.parameters():
                param.requires_grad = False

        x = self.get_input(batch)
        
        if self.scale_equivariance:
            xs, xrec, qloss, decoder_distill_output = self.forward_se(x)
        else:
            xrec, qloss, decoder_distill_output = self(x)
            xs = x

        distill_loss = self.distill_loss(self.get_distill_gt(x), decoder_distill_output) if self.if_distill_loss else torch.tensor(0.0, device=x.device)

        optimizer_idx = 1
        discloss, log_dict_disc = self.loss(qloss, distill_loss, xs, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, xs, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/distill_loss", distill_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    
    def get_distill_gt(self, x):
        with torch.no_grad():
            if 'VIT' in self.distill_model_type:
                # resize image x to 224x224
                if 'VIT_DINOv2' in self.distill_model_type or 'depth_anything' in self.distill_model_type:
                    image_size = (self.image_size*14//self.patch_size, self.image_size*14//self.patch_size) if isinstance(self.image_size, int) else (self.image_size[0]*14//self.patch_size, self.image_size[1]*14//self.patch_size)
                    x_224 = F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)
                    distill_output = self.distill.forward_features(x_224)
                else: # for VIT-DINOv1, VIT-SAM models
                    distill_output = self.distill.forward_features(x)

            elif self.distill_model_type == 'CNN':
                distill_output = self.distill(x)
        return distill_output
    
    
    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQModelIF(VQModel):
    """
    VQGAN model with token factorization (IF: ImageFolder)
    """
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 cont_ratio_trainig= 0.0,
                 ignore_keys=[],
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                 distill_model_type='VIT_DINOv2', # 'VIT_DINO' or 'CNN' or VIT_DINOv2, VIT_DINOv2_large_reg4, SAM_VIT
                 min_lr_multiplier=0.1,
                 only_decoder=False,
                 scale_equivariance=[]
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, cont_ratio_trainig, ignore_keys, 
                         monitor, 
                         entropy_loss_weight_scheduler_config, 
                         distill_model_type, min_lr_multiplier, only_decoder, scale_equivariance)
    
        self.encoder2 = instantiate_from_config(encoder_config)
        self.post_quant_conv = torch.nn.Conv2d(quantizer_config.params['e_dim']*2, decoder_config.params["z_channels"], 1)
        self.quant_conv2 = torch.nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params['e_dim'], 1)
        self.quantize2 = instantiate_from_config(quantizer_config)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.encoder2.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.min_lr_multiplier)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)

        h2 = self.encoder2(x)
        h2 = self.quant_conv2(h2)
        if self.encoder_normalize_embedding:
            h2 = F.normalize(h2, p=2, dim=1)

        quant = self.quantize(h)
        quant2 = self.quantize2(h2)
        
        quant_loss = quant['quantization_loss'] + quant2['quantization_loss']
        entropy_loss = quant['entropy_loss'] + quant2['entropy_loss'] if quant['entropy_loss'] is not None and quant2['entropy_loss'] is not None else None
        
        ret = {
            "quantized": (quant["quantized"], quant2["quantized"]),
            "quantization_loss": quant_loss,
            "entropy_loss": entropy_loss,
            "indices": (quant["indices"], quant2["indices"]),
            "continuous": (h, h2)
        }
        return ret
    

    def decode(self, quant):
        if isinstance(quant, tuple):
            quant_rec = quant[0]
            quant_sem = quant[1]
        else:
            print('Error: quant should be a tuple')
        distill_conv_out = self.post_quant_conv_distill(quant_sem)
        quant_cat = torch.cat((quant_rec, quant_sem), dim=1)
        quant = self.post_quant_conv(quant_cat)
        return self.decoder(quant), distill_conv_out
    
    def decode_code(self, code_b):
        code_b_rec, code_b_sem = code_b
        quant_b_rec = self.quantize.get_codebook_entry(code_b_rec, (-1, code_b_rec.size(1), code_b_rec.size(2), self.quantize.e_dim))
        quant_b_sem = self.quantize2.get_codebook_entry(code_b_sem, (-1, code_b_sem.size(1), code_b_sem.size(2), self.quantize.e_dim))
        quant_b = (quant_b_rec, quant_b_sem)
        dec = self.decode(quant_b)
        return dec
    
    def forward_se(self, input):
        random_scale = [random.choice(self.scale_equivariance[0]), random.choice(self.scale_equivariance[1])]
        downscale_factor = [1/random_scale[0], 1/random_scale[1]]
        encoded = self.encode(input)
        quantized = encoded["quantized"]
        continuous = encoded["continuous"]
        if torch.rand(1) > self.cont_ratio_trainig:
            dec, distill_conv_out = self.decode(quantized)
            quant_se = F.interpolate(quantized[0], scale_factor=downscale_factor, mode='bilinear', align_corners=False), \
                       F.interpolate(quantized[1], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(quant_se)[0]
        else:
            dec, distill_conv_out = self.decode(continuous)
            latents_se =  F.interpolate(continuous[0], scale_factor=downscale_factor, mode='bilinear', align_corners=False), \
                          F.interpolate(continuous[1], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(latents_se)[0]

        input_se = F.interpolate(input, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        decs = [dec, dec_se]
        inputs = [input, input_se]
        return inputs, decs, (encoded["quantization_loss"], encoded["entropy_loss"]), distill_conv_out