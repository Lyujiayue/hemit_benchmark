"""
Dual-Branch Pix2pix Generator (HEMIT Paper Method)

Architecture: Two parallel encoders (one for structure, one for texture/signal)
with a shared decoder. Each encoder processes H&E from different perspectives.

Reference: Bian et al., "HEMIT: H&E to Multiplex-immunohistochemistry Image Translation
with Dual-Branch Pix2pix Generator", arXiv:2403.18501, 2024.

Configuration notes:
- `--netG SwinTResnet` in original code
- `--lr 0.00003` (3e-5)
- `--lambda_L1 30`
- `--batch_size 2`
- `--n_epochs 50 --n_epochs_decay 30`
- Loss: L1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """Residual block with InstanceNorm."""

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=bias),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=bias),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ChannelAttention(nn.Module):
    """Channel attention module for dual-branch fusion."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class DualBranchBlock(nn.Module):
    """
    A single dual-branch processing block.
    Two parallel conv branches with channel attention for fusion.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        # Branch 1: Standard conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Depthwise + pointwise (texture branch)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=True),
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=1, bias=True),
            nn.InstanceNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )

        # Channel attention for fusion
        self.attention = ChannelAttention(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.attention(out)
        return out


class DualBranchEncoder(nn.Module):
    """
    Dual-branch encoder with progressive downsampling.
    Processes H&E input to extract multi-scale features.
    """

    def __init__(self, input_nc: int = 1, ngf: int = 64):
        super().__init__()

        # Initial conv
        self.inc = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Downsample blocks
        self.down1 = DualBranchBlock(ngf, ngf * 2, stride=2)     # 512
        self.down2 = DualBranchBlock(ngf * 2, ngf * 4, stride=2)   # 256
        self.down3 = DualBranchBlock(ngf * 4, ngf * 8, stride=2)  # 128
        self.down4 = DualBranchBlock(ngf * 8, ngf * 8, stride=2)   # 64

        # Residual blocks in latent space
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf * 8) for _ in range(6)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Returns (latent_features, skip_connections)"""
        skips = []

        x = self.inc(x)
        skips.append(x)

        x = self.down1(x)
        skips.append(x)

        x = self.down2(x)
        skips.append(x)

        x = self.down3(x)
        skips.append(x)

        x = self.down4(x)
        skips.append(x)

        x = self.res_blocks(x)

        return x, skips


class DualBranchDecoder(nn.Module):
    """
    Dual-branch decoder with skip connections.
    Upsamples latent features back to full resolution.
    (Fixed channel dimensions for skip connection concatenation)
    """

    def __init__(self, ngf: int = 64, output_nc: int = 3):
        super().__init__()

        # up1 receive 512 channel -> outputs 256 channel
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        # x(256) + skip4(512) = 768 up2 must receive 768 channel
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + ngf * 8, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )

        # x(128) + skip3(256) = 384 up3 must receive 384 channel
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + ngf * 4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # x(64) + skip2(128) = 192 up4 must receive 192 channel
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # x(64) + skip1(64) = 128 outc must receive 128 channel
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf + ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip1: torch.Tensor, # Receive Encoder skips[0] (64ch)
        skip2: torch.Tensor, # Receive Encoder skips[1] (128ch)
        skip3: torch.Tensor, # Receive Encoder skips[2] (256ch)
        skip4: torch.Tensor  # Receive Encoder skips[3] (512ch)
    ) -> torch.Tensor:
        
        x = self.up1(x)                     # 512 -> 256
        x = torch.cat([x, skip4], dim=1)    # 256 + 512 = 768

        x = self.up2(x)                     # 768 -> 128
        x = torch.cat([x, skip3], dim=1)    # 128 + 256 = 384

        x = self.up3(x)                     # 384 -> 64
        x = torch.cat([x, skip2], dim=1)    # 64 + 128 = 192

        x = self.up4(x)                     # 192 -> 64
        x = torch.cat([x, skip1], dim=1)    # 64 + 64 = 128

        x = self.outc(x)                    # 128 -> 3
        return x


class DualBranchGenerator(nn.Module):
    """
    Dual-Branch Pix2pix Generator for H&E to mIHC translation.

    Key features:
    - Two parallel encoder branches (structure + texture)
    - Channel attention for branch fusion
    - Residual blocks in latent space
    - Skip connections from both branches

    Args:
        input_nc: Number of input channels (1 for grayscale H&E)
        output_nc: Number of output channels (3 for DAPI, panCK, CD3)
        ngf: Number of generator filters (base)
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3, ngf: int = 64):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.encoder = DualBranchEncoder(input_nc, ngf)
        self.decoder = DualBranchDecoder(ngf, output_nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encoder(x)
        output = self.decoder(latent, skips[0], skips[1], skips[2], skips[3])
        return output

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dual_branch_generator(input_nc: int = 1, output_nc: int = 3, ngf: int = 64) -> DualBranchGenerator:
    """Factory function to create Dual-Branch generator."""
    return DualBranchGenerator(input_nc, output_nc, ngf)


# ---------------------------------------------------------------------------
# Dual-Branch Discriminator (same architecture as pix2pix PatchGAN)
# ---------------------------------------------------------------------------

class DualBranchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Dual-Branch pix2pix.
    Conditions on both input (H&E) and output (mIHC) images.
    """

    def __init__(self, input_nc: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, fake_img: torch.Tensor, input_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fake_img, input_img], dim=1)
        return self.model(x)
