import tqdm
import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torchvision.transforms.functional as TF
import copy
import torchinfo as tf
from timm.models.vision_transformer import PatchEmbed, Mlp
from rotary_embedding_torch import RotaryEmbedding

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data.mul_(self.beta).add_(current_params.data, alpha=1 - self.beta)

    def update_average(self, old, new):
        if old is None:
            return newa
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class t_embeddings(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, emmbedding_size=512, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emmbedding_size, bias=True),
            nn.SiLU(),
            nn.Linear(emmbedding_size, emmbedding_size, bias=True),
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
class LiniarEmmbedder(nn.Module):
    def __init__(self, text_size=512, hidden_size= 512, emmbedding_size=512):
        super().__init__()
        self.L1 = nn.Linear(text_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, emmbedding_size)
    def forward(self,text):
        x =  self.L1(text)
        return self.L2(x)




class MHA(nn.Module):
    def __init__(self, hidden_size, num_heads,drop_prob= 0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(dim=head_dim)

        self.attn_drop = nn.Dropout(drop_prob)
        self.proj_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
    
        # FIX: Make tensors contiguous in memory for FlashAttention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return self.proj(x)
class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MHA(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
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
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        text_size=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = t_embeddings(hidden_size)

        # Fixed typo: standardized to text_embedder
        self.text_embedder = LiniarEmmbedder(text_size=text_size, hidden_size=hidden_size, emmbedding_size=hidden_size)
        num_patches = self.x_embedder.num_patches

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 1. Basic Init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 3. Patch Embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 4. Condition Embedders
        # Removed y_embedder (doesn't exist).
        # Initializing text and conditional embedders instead.
        for module in [self.text_embedder]:
             nn.init.normal_(module.L1.weight, std=0.02)
             nn.init.normal_(module.L2.weight, std=0.02)

        # 5. Timestep Embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 6. AdaLN-Zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 7. Final Layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, text):
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.text_embedder(text)
        # Combine conditioning
        c =t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x