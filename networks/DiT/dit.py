# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import numpy as np


from timm.layers.mlp import SwiGLU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn


from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from omegaconf import ListConfig

from ..swin.swin_free_aspect_ratio import SwinTransformerBlock, SwinAttention


def get_norm_layer(norm_layer):
    if isinstance(norm_layer, str):
        if norm_layer == 'layer_norm':
            return nn.LayerNorm
        elif norm_layer == 'rms_norm':
            return nn.RMSNorm
        else:
            raise ValueError(f"Unsupported norm layer: {norm_layer}")
    return norm_layer


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
#                                 Cross Attention                               #
#################################################################################


class CrossAttention(nn.Module):
   """
   Cross-attention mechanism that computes attention between target (x) and context (y).
   Args:
       dim (int): Input dimension.
       num_heads (int): Number of attention heads.
       qkv_bias (bool): Whether to include bias terms.
       attn_drop (float): Dropout rate for attention weights.
       proj_drop (float): Dropout rate after projection.
   """
   def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm):
       super(CrossAttention, self).__init__()
       self.num_heads = num_heads
       self.head_dim = dim // num_heads
       self.scale = self.head_dim ** -0.5


       self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.proj = nn.Linear(dim, dim)
       self.attn_drop = nn.Dropout(attn_drop)
       self.proj_drop = nn.Dropout(proj_drop)
       self.qk_norm = qk_norm
       if qk_norm:
           self.q_norm = norm_layer(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)
           self.k_norm = norm_layer(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)


   def forward(self, x, y):
       B, N, C = x.shape
       B, M, _ = y.shape


       q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
       k = self.k_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
       v = self.v_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
       if self.qk_norm:
           q, k = self.q_norm(q), self.k_norm(k)
       attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
       attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
       return self.proj_drop(self.proj(attn_output))








class STBlock(nn.Module):
   # Used for temporal compression in context
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, 
                norm_layer=nn.LayerNorm,
                mlp_block='mlp',
                act_layer=lambda: nn.GELU(approximate="tanh"), 
                **block_kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)    

        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=norm_layer, **block_kwargs)
    
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=norm_layer, **block_kwargs)
        
        if mlp_block == 'mlp':
            self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer, norm_layer=norm_layer, drop=0)
            self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu': 
            self.space_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
            self.time_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
        else:
            raise NotImplementedError(f"mlp_block {mlp_block} not implemented")


   def forward(self, x):
       B, F, N, D = x.shape


       # Spatial attention
       x = rearrange(x, 'b f n d -> (b f) n d')
       x = x + self.space_attn(self.norm1(x))
       x = x + self.space_mlp(self.norm2(x))


       # Temporal attention
       x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
       x = x + self.time_attn(self.norm3(x))
       x = x + self.time_mlp(self.norm4(x))


       # Restore original shape
       x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)


       return x


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
# TODO : try:  # self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=ffn_bias)
class DiTBlock(nn.Module):
   """
   A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu':
            self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=True)
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

   def forward(self, x, c):
       shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
       x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
       x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
       return x



class CDiTBlock(nn.Module):
   """
   A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm_layer, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu':
            self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=True)


   def forward(self, x, c, x_cond):
       shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
       x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
       x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
       x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
       x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
       return x
  



class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, modulate_time_attn=False, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
            self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
        elif mlp_block == 'swiglu': 
            self.space_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
            self.time_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
        else:
            raise NotImplementedError(f"mlp_block {mlp_block} not implemented")
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.causal_time_attn = causal_time_attn
        self.modulate_time_attn = modulate_time_attn
        
        if modulate_time_attn:
            self.adaLN_time_attn_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
            self.norm_time_attn = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
            # initialize
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].bias, 0)
        else:
            self.norm_time_attn = nn.Identity()


    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
        shift_mlp_s, scale_mlp_s, gate_mlp_s,
        shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
        x_ = self.space_attn(x_modulated)
        x_ = rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)
        x = x + gate_msa.unsqueeze(1).unsqueeze(1) * x_

        x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)

        # — temporal attention path —
        if self.modulate_time_attn:
            shift_mta, scale_mta, gate_mta = self.adaLN_time_attn_modulation(c).chunk(3, dim=1)
        else:
            shift_mta, scale_mta, gate_mta = torch.zeros_like(shift_mlp_t), torch.zeros_like(scale_mlp_t), torch.ones_like(gate_mlp_t)
        x_modulated = modulate(self.norm_time_attn(x), shift_mta, scale_mta)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d', b=B, f=F, n=N)
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x_ = self.time_attn(x_modulated, attn_mask=time_attn_mask)
        x_ = rearrange(x_, '(b n) f d -> b f n d', b=B, n=N, f=F)
        x = x + gate_mta.unsqueeze(1).unsqueeze(1) * x_
        
        x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
        x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated)  
        
        return x

  
class STDiTBlock_tmpadaLN(nn.Module):
   """
   A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, **block_kwargs):
       raise NotImplementedError("This is a deprecated version of STDiTBlock with temporary adaLN for time attention. Use STDiTBlock instead.")
       super().__init__()
       self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
       self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
       self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       mlp_hidden_dim = int(hidden_size * mlp_ratio)
       approx_gelu = lambda: nn.GELU(approximate="tanh")
       self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
       self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
       self.adaLN_modulation = nn.Sequential(
           nn.SiLU(),
           nn.Linear(hidden_size, 12 * hidden_size, bias=True)
       )
       self.causal_time_attn = causal_time_attn


   def forward(self, x, c):
       B, F, N, D = x.shape


       # chunk into 12 [B, C] vectors
       (shift_msa_s, scale_msa_s, gate_msa_s,
        shift_msa_t, scale_msa_t, gate_msa_t,
        shift_mlp_s, scale_mlp_s, gate_mlp_s,
        shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(12, dim=1)


       x_modulated = modulate(self.norm1(x), shift_msa_s, scale_msa_s)
       x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
       x_ = self.space_attn(x_modulated)
       x = x + gate_msa_s.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)


       x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
       x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)


       # — temporal attention path —
       x_modulated = modulate(self.norm3(x), shift_msa_t, scale_msa_t)
       x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d', b=B, f=F, n=N)
       time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
       x_ = self.time_attn(x_modulated, attn_mask=time_attn_mask)
       x = x + gate_msa_t.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b n) f d -> b f n d', b=B, f=F)
       x_modulated = modulate(self.norm4(x), shift_mlp_t, scale_mlp_t)
       x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated) 
       return x


class SwinSTDiTBlock(STDiTBlock):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, 
                 causal_time_attn=False, modulate_time_attn=False, 
                 norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate, 
                         causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, mlp_block=mlp_block)
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = SwinTransformerBlock(
                hidden_size,
                input_resolution=input_shape,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
                qk_norm=False,
                mlp_ratio=mlp_ratio,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                norm_layer=norm_layer,
                mlp_block=mlp_block,
                **block_kwargs
                )


class SwinSTDiTBlockNoExtraMLP(STDiTBlock):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__(hidden_size, num_heads, input_shape, layer_idx, mlp_ratio, dropout_rate, norm_layer=norm_layer, mlp_block=mlp_block, **block_kwargs)
        self.space_attn = SwinAttention(
            hidden_size,
            input_resolution=input_shape,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
            proj_drop=dropout_rate,
            attn_drop=dropout_rate,
            qk_norm=True,
            norm_layer=norm_layer,
            **block_kwargs,
        )




class FinalLayer(nn.Module):
   """
   The final layer of DiT.
   """
   def __init__(self, hidden_size, patch_size, out_channels, norm_layer=nn.LayerNorm, act_layer=nn.SiLU):
       super().__init__()
       self.norm_final = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
       self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
       self.adaLN_modulation = nn.Sequential(
           act_layer(),
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
        norm_layer=nn.LayerNorm,
        mlp_block='mlp',
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
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
        ])       
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_layer)
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

    def get_condition_embeddings(self, t):
        """
        Get the condition embeddings for the given timesteps.
        t: (N,) tensor of diffusion timesteps
        returns: (N, D) tensor of condition embeddings
        """
        return self.t_embedder(t)
    
    def preprocess_inputs(self, target, context, t, frame_rate):
        b, f_target = target.size()[:2]
        f_context = context.size(1) if context is not None else 0
        
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

    def forward(self, target, context, t, frame_rate, return_features=False):
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
            features = []
            for block in self.blocks:
                x = block(x, c)
                features.append(x) if return_features else None
            x = rearrange(x,  'b (f hw) c -> b f hw c', f=(num_frames_ctx+num_frames_pred))[:,-num_frames_pred:]
            out = self.final_layer(x, c)
                        
            out = self.postprocess_outputs(out)
            if return_features:
                return out, features
            return out


class CDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, norm_layer=nn.LayerNorm, mlp_block='mlp', **kwargs):
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, 
                         ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
            ])


    def forward(self, target, context, t, frame_rate, return_features=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        num_frames_ctx = context.size(1)
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)  # (B, F, N, D)


        target = rearrange(x[:,-num_frames_pred:],  'b f hw c -> b (f hw) c')
        ctx = rearrange(x[:,:-num_frames_pred],  'b f hw c -> b (f hw) c') if num_frames_ctx>1 else None

        features = []
        for block in self.blocks:
            target = block(target, c, ctx)                      # (N, T, D)
            features.append(target)

        target = rearrange(target,  'b (f hw) c -> b f hw c', f=(num_frames_pred))
        out = self.final_layer(target, c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        if return_features:
            return out, features
        return out



class STDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, 
                 dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), 
                 causal_time_attn=False, modulate_time_attn=False, norm_layer=nn.LayerNorm, mlp_block='mlp',
                 **kwargs):
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                         max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, 
                         norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, 
                           causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, 
                           norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
            ])

    def forward(self, target, context, t, frame_rate, return_features=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        f_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)  # (B, F, N, D)

        features = []
        for block in self.blocks:
            x = block(x, c)
            features.append(x) if return_features else None


        out = self.final_layer(x[:,-f_pred:], c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        if return_features:
            return out, features
        return out


class STDiTWithGlobalBlocks(STDiT):
    """
    A DiT with extra global attention blocks (DiT blocks).
    """
    def __init__(self, *args, global_block_indices, **kwargs):
        
        class DiTBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, x, c):
                B, F, N, D = x.shape
                x_ = rearrange(x, 'b f n d -> b (f n) d', b=B, f=F)
                x_ = self.block(x_, c)
                x_ = rearrange(x_, 'b (f n) d -> b f n d', b=B, f=F)
                return x_
        
        super().__init__(*args, **kwargs)
        self.global_block_indices = global_block_indices
        for idx in global_block_indices:
            dit_block = DiTBlock(hidden_size=kwargs.get('hidden_size',1152), num_heads=kwargs.get('num_heads',16),
                                 mlp_ratio=kwargs.get('mlp_ratio',4.0), dropout_rate=kwargs.get('dropout',0.0), 
                                 norm_layer=kwargs.get('norm_layer',nn.LayerNorm), mlp_block=kwargs.get('mlp_block','mlp'))
            self.blocks[idx] = DiTBlockWrapper(dit_block)
            
            
class STDiT_tmpadaLN(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), causal_time_attn=False, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, norm_layer=norm_layer, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock_tmpadaLN(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, causal_time_attn=causal_time_attn, norm_layer=norm_layer) for _ in range(depth)
            ])


class SwinSTDiT(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), norm_layer=nn.LayerNorm, mlp_block='mlp', **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                SwinSTDiTBlock(hidden_size=hidden_size, num_heads=num_heads, input_shape=input_size, layer_idx=layer_idx, mlp_ratio=mlp_ratio, window_size=window_size, dropout_rate=dropout, norm_layer=norm_layer) for layer_idx in range(depth)
            ])


class SwinSTDiTNoExtraMLP(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], 
                 dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), 
                 norm_layer='layer_norm', mlp_block='mlp', qk_norm=True, **kwargs):
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, 
                         num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, 
                         ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, 
                         norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        
        for block_idx, block in enumerate(self.blocks):
            block.space_attn = SwinAttention(
                hidden_size,
                input_resolution=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0) if (block_idx % 2 == 0) else [ws//2 for ws in window_size],
                proj_drop=dropout,
                attn_drop=dropout,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
            )
