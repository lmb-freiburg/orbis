import numpy as np
import torch
from torch import nn
from .dit import STDiT, STDiTBlock, SwinSTDiTNoExtraMLP, modulate
from einops import rearrange
from util import instantiate_from_config
# from .dit_multiframe_pred import STDiTMultiFramePred


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class SplitSteeringSpeedYawEmbedder(torch.nn.Module):
    def __init__(self, hidden_size, noise_amount=0.0):
        super().__init__()
        self.speed_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        nn.init.normal_(self.speed_embedder[0].weight, std=0.02)
        nn.init.normal_(self.speed_embedder[0].bias, std=0.02)
        nn.init.normal_(self.speed_embedder[2].weight, std=0.02)
        nn.init.normal_(self.speed_embedder[2].bias, std=0.02)
        self.yaw_rate_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        nn.init.normal_(self.yaw_rate_embedder[0].weight, std=0.02)
        nn.init.normal_(self.yaw_rate_embedder[0].bias, std=0.02)
        nn.init.normal_(self.yaw_rate_embedder[2].weight, std=0.02)
        nn.init.normal_(self.yaw_rate_embedder[2].bias, std=0.02)
        self.no_speed = nn.Parameter(torch.zeros(1, hidden_size))  # init to zero
        self.no_yaw_rate = nn.Parameter(torch.zeros(1, hidden_size))  # init to zero
        # random init of no_speed and no_yaw_rate embeddings
        nn.init.normal_(self.no_speed, std=0.02)
        nn.init.normal_(self.no_yaw_rate, std=0.02)
        self.noise_amount = noise_amount

    def forward(self, steering):
        b = steering.shape[0]
        
        # linear combination of speed and yaw rate with some noise added during training, keeping the same std
        if self.training and self.noise_amount > 0.0:
            noise_amount = torch.tensor(self.noise_amount, device=steering.device)
            steering = torch.sqrt((1.0 - noise_amount**2)) * steering + noise_amount * torch.randn_like(steering)

        speed, yaw_rate = steering[:, [0]], steering[:, [1]]
        speed_embedding = self.speed_embedder(torch.nan_to_num(speed, nan=0.0))
        yaw_rate_embedding = self.yaw_rate_embedder(torch.nan_to_num(yaw_rate, nan=0.0))
        # replace NaN values with no_speed and no_yaw_rate
        speed_embedding[torch.isnan(speed).squeeze(-1)] = self.no_speed.to(speed_embedding.dtype)
        yaw_rate_embedding[torch.isnan(yaw_rate).squeeze(-1)] = self.no_yaw_rate.to(yaw_rate_embedding.dtype)
        steering_embedding = speed_embedding + yaw_rate_embedding
        return steering_embedding


class LinearSteeringEmbedder(torch.nn.Module):
    """
    Generic linear embedding layer for steering signals.
    """
    def __init__(self, num_input_features, hidden_size):
        super().__init__()
        self.embedders = nn.ModuleList()
        self.no_value_embeddings = nn.ParameterList()
        for _i in range(num_input_features):
            embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
            nn.init.normal_(embedder[0].weight, std=0.02)
            nn.init.normal_(embedder[0].bias, std=0.02)
            nn.init.normal_(embedder[2].weight, std=0.02)
            nn.init.normal_(embedder[2].bias, std=0.02)
            self.embedders.append(embedder)
            no_value_embedding = nn.Parameter(torch.zeros(1, hidden_size))  # init to zero
            self.no_value_embeddings.append(no_value_embedding)

    def forward(self, steering):
        b, num_features = steering.shape
        assert num_features == len(self.embedders), f"Expected {len(self.embedders)} features, but got {num_features}"
        embeddings = []
        for i in range(num_features):
            feature = steering[:, [i]]
            embedding = self.embedders[i](feature)
            # replace NaN values with no_value_embedding
            embedding = torch.where(torch.isnan(embedding), self.no_value_embeddings[i].expand(b, -1), embedding)
            embeddings.append(embedding)
        steering_embedding = torch.stack(embeddings, dim=1).sum(dim=1)
        return steering_embedding


class STDiTSteering(STDiT):
    def __init__(self, *, steering_config, **kwargs):
        super().__init__(**kwargs)
        self.steering_embedder = instantiate_from_config(steering_config)
    
    def forward(self, x, t, frame_rate, steering):
        raise NotImplementedError("This method should be overridden in subclasses.")


class STDiTSteeringSimpleCond(STDiTSteering):    
    
    def get_condition_embeddings(self, t, steering):
        c = self.t_embedder(t) + self.steering_embedder(steering)
        return c
    
    def process_outputs(self, out):
        out = self.unpatchify(out).unsqueeze(1)
        return out
    
    def forward(self, target, context, t, frame_rate, steering):        
        num_frames_pred = target.size(1)

        c = self.get_condition_embeddings(t, steering)

        x = self.preprocess_inputs(target, context, t, frame_rate)

        for block in self.blocks:
            x = block(x, c)
        out = self.final_layer(x[:,-num_frames_pred:], c)

        out = self.postprocess_outputs(out)
        return out


class SwinSTDiTSteeringSimpleCond(STDiTSteeringSimpleCond, SwinSTDiTNoExtraMLP):
    def __init__(self, *, steering_config, **kwargs):
        SwinSTDiTNoExtraMLP.__init__(self, **kwargs)
        self.steering_embedder = instantiate_from_config(steering_config)

