import torch
import torch.nn as nn
import torch.nn.functional as F

def time_embedding(timesteps: torch.Tensor, time_dim: int) -> torch.Tensor:
    """
    Sinusoidal time embeddings
    Args:
        timesteps: tensor of shape (batch_size,) - time steps
        time_dim: int - dimension of the time embedding
    Returns:
        tensor of shape (batch_size, time_dim) - time embeddings
    """
    denom = torch.arange(1, time_dim // 2 + 1, dtype=torch.float32)
    sin = torch.sin(timesteps[:, None] / 10000 ** (2 * (denom - 1) / time_dim))
    cos = torch.cos(timesteps[:, None] / 10000 ** (2 * (denom - 1) / time_dim))
    emb = torch.cat([sin, cos], dim=-1)

    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.dropout = nn.Dropout(dropout)

        self.gn1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.gn2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()


    def forward(self, x, t):
        out = x

        out = self.gn1(out)
        out = F.silu(out)
        out = self.conv1(out)
        
        out = out + self.time_embed(t)[:, :, None, None]

        out = self.gn2(out)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + self.residual(x)

        return out
    

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsamples from in_channels to out_channels

        Args:
            in_channels: int - number of input channels
            out_channels: int - number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        return self.conv(x)
    
        
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Downsamples from in_channels to out_channels

        Args:
            in_channels: int - number of input channels
            out_channels: int - number of output channels
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, t):
        batch_size, channels, height, width = x.shape
        q = self.q(x).view(batch_size, channels, -1)
        k = self.k(x).view(batch_size, channels, -1)
        v = self.v(x).view(batch_size, channels, -1)
        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = self.softmax(attn)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out.view(batch_size, channels, height, width)


class UNet(nn.Module):
    def __init__(self, img_channels):
        super().__init__()

        self.time_dim = 128

        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.img_channels = img_channels

        self.first_conv = nn.Conv2d(self.img_channels, 32, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.ModuleList()

        self.encoder.append(ResidualBlock(32, 32, self.time_dim))
        self.encoder.append(DownsampleBlock(32, 64))
        self.encoder.append(ResidualBlock(64, 64, self.time_dim))
        self.encoder.append(DownsampleBlock(64, 128))
        self.encoder.append(ResidualBlock(128, 128, self.time_dim))
        self.encoder.append(DownsampleBlock(128, 256))
        self.encoder.append(ResidualBlock(256, 256, self.time_dim))


        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResidualBlock(256, 256, self.time_dim))
        # self.bottleneck.append(SelfAttention(256))
        self.bottleneck.append(ResidualBlock(256, 256, self.time_dim))

        self.decoder = nn.ModuleList()
        self.decoder.append(ResidualBlock(256, 256, self.time_dim))
        self.decoder.append(UpsampleBlock(256, 128))
        self.decoder.append(ResidualBlock(256, 128, self.time_dim))
        self.decoder.append(UpsampleBlock(128, 64))
        self.decoder.append(ResidualBlock(128, 64, self.time_dim))
        self.decoder.append(UpsampleBlock(64, 32))
        self.decoder.append(ResidualBlock(64, 32, self.time_dim))

        self.final_norm = nn.GroupNorm(8, 32)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(32, self.img_channels, kernel_size=3, padding=1)


    def forward(self, x, t):
        
        out = self.first_conv(x)

        time_emb = time_embedding(t, self.time_dim).to(x.device)
        time_emb = self.time_proj(time_emb)


        skips = []
        for layer in self.encoder:
            if isinstance(layer, DownsampleBlock):
                skips.append(out)
            out = layer(out, time_emb)
        

        for layer in self.bottleneck:
            out = layer(out, time_emb)


        for layer in self.decoder:
            if isinstance(layer, UpsampleBlock):
                out = layer(out, time_emb)
                out = torch.cat([out, skips.pop()], dim=1)
            else:
                out = layer(out, time_emb) 

        out = self.final_norm(out)
        out = self.final_activation(out)
        out = self.final_conv(out)

        return out
