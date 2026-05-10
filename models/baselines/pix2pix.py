"""
Pix2pix Baselines for HEMIT Benchmark

Implements:
- pix2pix_UNet: U-Net based generator with skip connections
- pix2pix_ResNet: ResNet-based generator (9-block)
- PatchGAN Discriminator

Reference: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

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


class DownBlock(nn.Module):
    """Downsampling block: Conv -> InstanceNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2,
                 padding: int = 1, norm: bool = True, bias: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose -> InstanceNorm -> ReLU/Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2,
                 padding: int = 1, norm: bool = True, bias: bool = True, dropout: bool = False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.block(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net Generator for pix2pix.

    Encoder-decoder with skip connections at each level.
    Input: (B, 1 or 3, 1024, 1024) H&E image
    Output: (B, 3, 1024, 1024) mIHC image
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3, ngf: int = 64):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 512
        self.enc2 = DownBlock(ngf, ngf * 2)  # 256
        self.enc3 = DownBlock(ngf * 2, ngf * 4)  # 128
        self.enc4 = DownBlock(ngf * 4, ngf * 8)  # 64
        self.enc5 = DownBlock(ngf * 8, ngf * 8)  # 32
        self.enc6 = DownBlock(ngf * 8, ngf * 8)  # 16
        self.enc7 = DownBlock(ngf * 8, ngf * 8)  # 8
        self.enc8 = DownBlock(ngf * 8, ngf * 8, norm=False)  # 4

        # Decoder with skip connections
        self.dec1 = UpBlock(ngf * 8, ngf * 8, dropout=True)  # 8
        self.dec2 = UpBlock(ngf * 16, ngf * 8, dropout=True)  # 16
        self.dec3 = UpBlock(ngf * 16, ngf * 8, dropout=True)  # 32
        self.dec4 = UpBlock(ngf * 16, ngf * 8)  # 64
        self.dec5 = UpBlock(ngf * 16, ngf * 4)  # 128
        self.dec6 = UpBlock(ngf * 8, ngf * 2)  # 256
        self.dec7 = UpBlock(ngf * 4, ngf)  # 512

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # 1024

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decode with skip connections
        d1 = self.dec1(e8, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        return self.final(d7)


class ResNetGenerator(nn.Module):
    """
    ResNet-based Generator for pix2pix (9 residual blocks).

    Reference: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution".
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3, n_blocks: int = 9, ngf: int = 64):
        super().__init__()
        assert n_blocks >= 0, "n_blocks must be non-negative"

        self.input_nc = input_nc
        self.output_nc = output_nc

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_ch = ngf
        for _ in range(2):
            mult = 2 ** (model.count(nn.Conv2d) + model.count(nn.ConvTranspose2d) // 2)
            out_ch = ngf * mult
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        # Residual blocks
        mult = 2 ** (model.count(nn.Conv2d) + model.count(nn.ConvTranspose2d) // 2)
        for _ in range(n_blocks):
            model.append(ResidualBlock(ngf * mult))

        # Upsampling
        for _ in range(2):
            mult = 2 ** (model.count(nn.Conv2d) + model.count(nn.ConvTranspose2d) // 2 - 2)
            out_ch = ngf * mult
            model += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SwinTResNetGenerator(nn.Module):
    """
    Swin Transformer + ResNet Hybrid Generator.

    Used in the HEMIT paper's dual-branch method.
    Swin-T encoder for global context, ResNet decoder for detail.
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3, ngf: int = 64):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        # Shallow feature extraction
        self.inc = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Downsampling encoder (ResNet-style)
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )

        # Transformer-like attention (simplified self-attention)
        self.attention = SelfAttentionBlock(ngf * 8)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf * 8) for _ in range(9)])

        # Upsampling decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        self.outc = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x3 = self.attention(x3)
        x3 = self.res_blocks(x3)

        x = self.up1(x3)
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up3(torch.cat([x, x1], dim=1))
        x = self.outc(torch.cat([x, x0], dim=1))

        return x


class SelfAttentionBlock(nn.Module):
    """Simplified self-attention for SwinTResNetGenerator."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with optional conditioning on input image.

    For multi-channel output (3 mIHC channels), the discriminator can be
    conditioned on the input H&E image (conditional GAN).
    """

    def __init__(
        self,
        input_nc: int = 4,  # 3 output channels + 1 input channel = 4
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer=nn.InstanceNorm2d
    ):
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
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input_tensor: torch.Tensor, cond_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_tensor: Generated image (B, output_nc, H, W)
            cond_tensor: Conditioning input image (B, input_nc, H, W)
        Returns:
            Patch-level predictions (B, 1, H', W')
        """
        if cond_tensor is not None:
            # Concatenate along channel dimension for conditional GAN
            x = torch.cat([input_tensor, cond_tensor], dim=1)
        else:
            x = input_tensor
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """1x1 Pixel-level discriminator."""

    def __init__(self, input_nc: int = 4, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1)
        )

    def forward(self, input_tensor: torch.Tensor, cond_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond_tensor is not None:
            x = torch.cat([input_tensor, cond_tensor], dim=1)
        else:
            x = input_tensor
        return self.model(x)


# ---------------------------------------------------------------------------
# Generator Factory
# ---------------------------------------------------------------------------

def create_generator(arch: str, input_nc: int = 1, output_nc: int = 3, **kwargs) -> nn.Module:
    """
    Factory function to create generators.

    Args:
        arch: One of 'unet', 'resnet', 'swintresnet'
        input_nc: Number of input channels
        output_nc: Number of output channels (3 for mIHC: DAPI, panCK, CD3)
    """
    arch = arch.lower()
    if arch in ('unet', 'unet64'):
        return UNetGenerator(input_nc, output_nc, ngf=kwargs.get('ngf', 64))
    elif arch in ('resnet', 'resnet9', 'resnet_9block'):
        return ResNetGenerator(input_nc, output_nc, n_blocks=9, ngf=kwargs.get('ngf', 64))
    elif arch == 'swintresnet':
        return SwinTResNetGenerator(input_nc, output_nc, ngf=kwargs.get('ngf', 64))
    else:
        raise ValueError(f"Unknown generator architecture: {arch}. Choose 'unet', 'resnet', or 'swintresnet'.")


def create_discriminator(
    input_nc: int = 4,
    ndf: int = 64,
    n_layers: int = 3,
    patchgan: bool = True
) -> nn.Module:
    """Factory function to create discriminators."""
    if patchgan:
        return NLayerDiscriminator(input_nc, ndf, n_layers)
    else:
        return PixelDiscriminator(input_nc, ndf)
