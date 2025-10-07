# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
# Modifications for STDiT:
# - Renamed/extended DiT -> STDiT (spatio-temporal DiT) architecture.
# - Adjusted attention blocks and conditioning to support temporal dims.
# --------------------------------------------------------

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from omegaconf import ListConfig

from ..swin.swin_free_aspect_ratio import SwinTransformerBlock


def modulate(x, shift, scale):
    if len(x.shape) == 3:
        # x: [B, N, C], shift: [B, C], scale: [B, C]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == 4:
        # x: [B, T, N, C], shift: [B, C], scale: [B, C]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3D or 4D tensor.")

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
   
#################################################################################
#                   Sine/Cosine Frequency Embedding Functions                  #
#################################################################################

class FrequencyEncoder:
    def __init__(self, embed_dim, freq_min=1, freq_max=5):
        """
        Deterministic frequency encoder with fixed normalization.

        Args:
            embed_dim (int): Dimensionality of the token embeddings.
            freq_min (float): Minimum frequency value.
            freq_max (float): Maximum frequency value.
        """
        self.embed_dim = embed_dim
        self.freq_min = freq_min
        self.freq_max = freq_max

    def encode(self, frequencies):
        """
        Encodes frequencies into embeddings using sine-cosine features.

        Args:
            frequencies (torch.Tensor): Tensor of shape (batch_size,) containing frequencies.

        Returns:
            torch.Tensor: Encoded frequency embeddings of shape (batch_size, embed_dim).
        """
        batch_size = frequencies.size(0)

        # Fixed normalization: Scale frequencies to [0, 1]
        normalized_freq = (frequencies - self.freq_min) / (self.freq_max - self.freq_min)

        # Generate positional features using sine and cosine
        positions = torch.arange(0, self.embed_dim, dtype=torch.float32, device=frequencies.device)
        scaling_factors = 1 / (10000 ** (2 * (positions // 2) / self.embed_dim))
        frequency_features = normalized_freq.unsqueeze(1) * scaling_factors  # Shape: (batch_size, embed_dim)

        # Apply sine to even indices and cosine to odd indices
        encoded_freq = torch.zeros(batch_size, self.embed_dim, device=frequencies.device)
        encoded_freq[:, 0::2] = torch.sin(frequency_features[:, 0::2])  # Sine for even indices
        encoded_freq[:, 1::2] = torch.cos(frequency_features[:, 1::2])  # Cosine for odd indices

        return encoded_freq


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
   """
   grid_size: int of the grid height and width
   return:
   pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
   """
   grid_h = np.arange(grid_size[0], dtype=np.float32)
   grid_w = np.arange(grid_size[1], dtype=np.float32)
   grid = np.meshgrid(grid_w, grid_h)  # here w goes first
   grid = np.stack(grid, axis=0)


   grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
   pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
   if cls_token and extra_tokens > 0:
       pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
   return pos_embed




def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
   assert embed_dim % 2 == 0


   # use half of dimensions to encode grid_h
   emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
   emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)


   emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
   return emb




def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
   """
   embed_dim: output dimension for each position
   pos: a list of positions to be encoded: size (M,)
   out: (M, D)
   """
   assert embed_dim % 2 == 0
   omega = np.arange(embed_dim // 2, dtype=np.float64)
   omega /= embed_dim / 2.
   omega = 1. / 10000**omega  # (D/2,)


   pos = pos.reshape(-1)  # (M,)
   out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product


   emb_sin = np.sin(out) # (M, D/2)
   emb_cos = np.cos(out) # (M, D/2)


   emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
   return emb



#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def _expand(self, param, x):
        """
        param: [B, C]
        x:     [B*R, ...]  (R = F for spatial path, R = N for temporal-MLP)
        returns [B*R, C] without doing any full copies
        """
        B, C = param.shape
        rep = x.shape[0] // B
        # -> [B, 1, C] -> [B, rep, C] (no-copy) -> [B*rep, C]
        return param.view(B, 1, C).expand(-1, rep, -1).reshape(B * rep, C)

    def _apply_mod(self, x, norm, block, shift, scale, gate):
        shift_e = self._expand(shift, x)
        scale_e = self._expand(scale, x)
        gate_e  = self._expand(gate,  x)
        return x + gate_e.unsqueeze(1) * block(modulate(norm(x), shift_e, scale_e))

    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=1)

        # — spatial attention path —
        x = rearrange(x, 'b f n d -> (b f) n d', b=B, f=F)
        x = self._apply_mod(x, self.norm1, self.space_attn,
                             shift_msa,   scale_msa,   gate_msa)
        x = self._apply_mod(x, self.norm2, self.space_mlp,
                             shift_mlp_s, scale_mlp_s, gate_mlp_s)

        # — temporal attention path —
        x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
        x = x + self.time_attn(x)   # same as before
        x = self._apply_mod(x, self.norm3, self.time_mlp,
                             shift_mlp_t, scale_mlp_t, gate_mlp_t)

        # restore
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        return x
    
    
class SwinSTDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = SwinTransformerBlock(
                hidden_size,
                input_resolution=input_shape,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
                mlp_ratio=mlp_ratio,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                )
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def _expand(self, param, x):
        """
        param: [B, C]
        x:     [B*R, ...]  (R = F for spatial path, R = N for temporal-MLP)
        returns [B*R, C] without doing any full copies
        """
        B, C = param.shape
        rep = x.shape[0] // B
        # -> [B, 1, C] -> [B, rep, C] (no-copy) -> [B*rep, C]
        return param.view(B, 1, C).expand(-1, rep, -1).reshape(B * rep, C)

    def _apply_mod(self, x, norm, block, shift, scale, gate):
        shift_e = self._expand(shift, x)
        scale_e = self._expand(scale, x)
        gate_e  = self._expand(gate,  x)
        return x + gate_e.unsqueeze(1) * block(modulate(norm(x), shift_e, scale_e))

    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=1)

        # — spatial attention path —
        x = rearrange(x, 'b f n d -> (b f) n d', b=B, f=F)
        x = self._apply_mod(x, self.norm1, self.space_attn,
                             shift_msa,   scale_msa,   gate_msa)
        x = self._apply_mod(x, self.norm2, self.space_mlp,
                             shift_mlp_s, scale_mlp_s, gate_mlp_s)

        # — temporal attention path —
        x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
        x = x + self.time_attn(x)
        x = self._apply_mod(x, self.norm3, self.time_mlp,
                             shift_mlp_t, scale_mlp_t, gate_mlp_t)

        # restore
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class DiT(nn.Module):
   """
   Diffusion model with a Transformer backbone.
   """
   def __init__(
       self,
       input_size=16,
       patch_size=2,
       in_channels=32,
       hidden_size=1152,
       depth=28,
       num_heads=16,
       mlp_ratio=4.0,
       max_num_frames=6,
       dropout=0.0,
       ctx_noise_aug_ratio=0.1,
       ctx_noise_aug_prob = 0.5,
       drop_ctx_rate=0.2,
       frequency_range=(2, 15),
       learn_sigma=False,
   ):
       super().__init__()
       self.input_size= input_size if isinstance(input_size, (list, tuple, ListConfig)) else [input_size, input_size]
       self.in_channels = in_channels
       self.out_channels = in_channels * 2 if learn_sigma else in_channels
       self.patch_size = patch_size
       self.num_heads = num_heads
       self.ctx_noise_aug_ratio = ctx_noise_aug_ratio
       self.ctx_noise_aug_prob = ctx_noise_aug_prob
       self.drop_ctx_rate = drop_ctx_rate


       self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
      
       self.num_patches = self.x_embedder.num_patches


       self.t_embedder = TimestepEmbedder(hidden_size)


       self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)


       self.blocks = nn.ModuleList([
           DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout) for _ in range(depth)
       ])       
      
       self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
       self.max_num_frames = max_num_frames
       self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0., 0.02)


       self.frame_rate_encoder = FrequencyEncoder(hidden_size, freq_min=frequency_range[0], freq_max=frequency_range[1])


       self.initialize_weights()


   def initialize_weights(self):
       # Initialize transformer layers:
       def _basic_init(module):
           if isinstance(module, nn.Linear):
               torch.nn.init.xavier_uniform_(module.weight)
               if module.bias is not None:
                   nn.init.constant_(module.bias, 0)
       self.apply(_basic_init)


       # Initialize (and freeze) pos_embed by sin-cos embedding:
       pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size], cls_token=False, extra_tokens=0)
       self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


       w = self.x_embedder.proj.weight.data
       nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
       nn.init.constant_(self.x_embedder.proj.bias, 0)


       # Initialize timestep embedding MLP:
       nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
       nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


       # Zero-out adaLN modulation layers in DiT blocks:
       for block in self.blocks:
           nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
           nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


       # Zero-out output layers:
       nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
       nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
       nn.init.constant_(self.final_layer.linear.weight, 0)
       nn.init.constant_(self.final_layer.linear.bias, 0)


   def unpatchify(self, x):
       """
       x: (N, T, patch_size**2 * C)
       imgs: (N, H, W, C)
       """
       c = self.out_channels
       p = self.x_embedder.patch_size[0]
       h = self.x_embedder.grid_size[0]
       w = self.x_embedder.grid_size[1]


       x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
       x = torch.einsum('bfhwpqc->bfchpwq', x)
       imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, w * p))
       return imgs


   def get_condition_embeddings(self, t):
       """
       Get the condition embeddings for the given timesteps.
       t: (N,) tensor of diffusion timesteps
       returns: (N, D) tensor of condition embeddings
       """
       return self.t_embedder(t)
  
   def preprocess_inputs(self, target, context, t, frame_rate):
       b, f_target = target.size()[:2]
       f_context = context.size(1)
      
       if self.training:
           # Drop the context frame
           if torch.rand(1, device=target.device)<self.drop_ctx_rate:
               context = None
               f_context = 0
           elif torch.rand(1, device=target.device) < self.ctx_noise_aug_prob:
               # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
               mask = (t >= self.ctx_noise_aug_ratio)
               aug_noise = torch.randn_like(context)
               context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio


       frame_embeddings = self.frame_rate_encoder.encode(frame_rate)
       frame_embeddings = frame_embeddings.unsqueeze(1).unsqueeze(1).to(target.device)
       x = torch.cat((context, target), dim=1) if context is not None else target
       x = rearrange(x, 'b f c h w -> (b f) c h w')
       x = self.x_embedder(x) + self.pos_embed.to(x.device)
       x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
       x = x + self.frame_emb[:, self.max_num_frames-(f_target+f_context):].to(x.device) + frame_embeddings
       return x


   def postprocess_outputs(self, out):
       return self.unpatchify(out)


   def forward(self, target, context, t, frame_rate):
       """
       Forward pass of DiT.
       x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
       t: (N,) tensor of diffusion timesteps
       y: (N,) tensor of class labels
       """
      
       num_frames_ctx = context.size(1)
       num_frames_pred = target.size(1)
      
       c = self.get_condition_embeddings(t)
      
       x = self.preprocess_inputs(target, context, t, frame_rate)
      
       x = rearrange(x,  'b f hw c -> b (f hw) c')
       for block in self.blocks:
           x = block(x, c)
       x = rearrange(x,  'b (f hw) c -> b f hw c', f=(num_frames_ctx+num_frames_pred))[:,-num_frames_pred:]
       out = self.final_layer(x, c)
                
       out = self.postprocess_outputs(out)
       return out



class STDiT(DiT):
    """Spatiotemporal DiT"""
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, frequency_range=frequency_range, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout) for _ in range(depth)
            ])


    def forward(self, target, context, t, frame_rate):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        f_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)  # (B, F, N, D)

        for block in self.blocks:
            x = block(x, c)

        out = self.final_layer(x[:,-f_pred:], c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        return out
    

class SwinSTDiT(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, frequency_range=frequency_range, **kwargs)
        self.blocks = nn.ModuleList([
                SwinSTDiTBlock(hidden_size=hidden_size, num_heads=num_heads, input_shape=input_size, layer_idx=layer_idx, mlp_ratio=mlp_ratio, window_size=window_size, dropout_rate=dropout) for layer_idx in range(depth)
            ])
