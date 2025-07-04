
import torch
import torch.nn as nn

# 正余弦位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        exponents = 10000 ** (-exponents)
        t = t.float().unsqueeze(1)
        sinusoid = t * exponents
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)

# ResNet 块
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))

# 改进的 DDPM 模型
class ImprovedDDPM(nn.Module):
    def __init__(self, input_dim, time_embed_dim=32):
        super().__init__()
        self.time_embed = PositionalEncoding(time_embed_dim)
        total_input_dim = input_dim + time_embed_dim
        hidden_dim = 128
        self.input_layer = nn.Linear(total_input_dim, hidden_dim)
        self.resblock1 = ResNetBlock(hidden_dim)
        self.resblock2 = ResNetBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x_in = torch.cat([x, t_embed], dim=1)
        x = self.input_layer(x_in)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return self.output_layer(x)
