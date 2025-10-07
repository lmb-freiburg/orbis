import torch
import timm
from torch import nn


from typing import Tuple, Union

class Encoder(nn.Module):
    def __init__(
        self, 
        resolution: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        patch_size: int = 16,
        z_channels: int = 768,
        e_dim: int = 8,
        normalize_embedding: bool = True,
        # **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = resolution
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        self.e_dim = e_dim
        
        self.init_transformer(pretrained_encoder)

    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_encoder == 'base':
            pretrained_encoder_model = 'timm/vit_base_patch16_224'
        elif pretrained_encoder == 'large':
            pretrained_encoder_model = 'timm/vit_large_patch16_224'
       

        self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        """Initialize weights of target_model with weights from source_model."""
        with torch.no_grad():
            for target_param, source_param in zip(self.encoder.parameters(), pretrained_model.parameters()):
                target_param.data.copy_(source_param.data)

        # Clean up
        del pretrained_model
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img)[:,1:]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h