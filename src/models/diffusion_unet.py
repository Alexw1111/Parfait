import torch
import torch.nn as nn
import math
from einops import rearrange

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim=None, dropout=0.1):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))) if time_emb_dim else None
        self.block1 = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.block2 = nn.Sequential(nn.GroupNorm(8, dim_out), nn.SiLU(), nn.Dropout(dropout), nn.Conv1d(dim_out, dim_out, 3, padding=1))
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1") + h
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)

class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_channels: int, guide_channels: int, model_channels: int,
        context_dim: int, num_res_blocks: int = 2, channel_mults: tuple = (1, 2, 4)
    ):
        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        time_dim = model_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.init_conv = nn.Conv1d(input_channels + guide_channels, model_channels, 3, padding=1)
        dims = [model_channels, *map(lambda m: model_channels * m, channel_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.down_blocks.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Attention(dim_out),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.up_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.up_blocks.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Attention(dim_in),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1)
            ]))
        
        self.final_res_block = ResnetBlock(model_channels * 2, model_channels, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(model_channels, self.input_channels, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, style_context: torch.Tensor, guide_curve: torch.Tensor):
        x = torch.cat([x_t, guide_curve], dim=1)
        x = self.init_conv(x)
        r = x.clone()

        time_emb = self.time_mlp(t)
        context_emb = self.context_mlp(style_context)
        emb = time_emb + context_emb

        h = []
        for res1, res2, attn, downsample in self.down_blocks:
            x = res1(x, emb)
            h.append(x)
            x = res2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for res1, res2, attn, upsample in self.up_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = res1(x, emb)
            x = torch.cat((x, h.pop()), dim=1)
            x = res2(x, emb)
            x = attn(x)
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, emb)
        return self.final_conv(x)