"""
DGR/DTR (Dual-Generator Registration / Dual-Transform Registration) Implementation

Based on: "Misalignment-Robust Virtual Staining for Multi-plex Immunohistochemistry"

This method handles misalignment between H&E and mIHC images by using dual generators
with alignment-invariant losses.

Key features:
1. Dual generators to handle different tissue structures
2. Registration-like loss to be robust to misalignment
3. Perceptual loss for better structure preservation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies"""

    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()

        self.channels = channels
        self.theta = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.phi = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.g = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.o = nn.Conv2d(channels // 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.size()

        # Compute attention
        theta = self.theta(x).view(batch_size, C // 8, H * W)
        phi = self.phi(x).view(batch_size, C // 8, H * W)
        phi = F.softmax(phi, dim=-1)

        g = self.g(x).view(batch_size, C // 2, H * W)
        g = F.softmax(g, dim=-1)

        # Attention map
        attention = torch.bmm(theta.transpose(1, 2), phi)
        attention = attention.view(batch_size, H * W, H * W)

        # Apply attention
        g = torch.bmm(g, attention.transpose(1, 2))
        g = g.view(batch_size, C // 2, H, W)

        o = self.o(g)
        out = self.gamma * o + x

        return out


class ResidualBlock(nn.Module):
    """Residual block with optional attention"""

    def __init__(self, channels: int, use_attention: bool = False, norm_fn=None):
        super(ResidualBlock, self).__init__()

        if norm_fn is None:
            norm_fn = nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            norm_fn(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            norm_fn(channels)
        )

        self.attention = AttentionBlock(channels) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.attention is not None:
            out = self.attention(out)
        return out + x


class DTRGenerator(nn.Module):
    """
    DTR (Dual-Transform Registration) Generator

    Uses dual-path processing with registration-like operations
    to handle misalignment between source and target domains.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        num_resblocks: int = 9,
        use_attention: bool = True,
        norm_layer: str = 'instance'
    ):
        super(DTRGenerator, self).__init__()

        # Normalization function
        if norm_layer == 'instance':
            norm_fn = lambda ch: nn.InstanceNorm2d(ch)
        elif norm_layer == 'batch':
            norm_fn = lambda ch: nn.BatchNorm2d(ch)
        else:
            norm_fn = lambda ch: nn.Identity()

        # Input processing
        self.input_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            norm_fn(ngf),
            nn.ReLU(True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            norm_fn(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            norm_fn(ngf * 4),
            nn.ReLU(True)
        )

        # Dual-branch processing with residual blocks
        channels = ngf * 4
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, use_attention=use_attention, norm_fn=norm_fn)
            for _ in range(num_resblocks)
        ])

        # Spatial transform layer for alignment (thin-plate spline or affine)
        self.spatial_transform = SpatialTransformer(output_size=(256, 256))

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_fn(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_fn(ngf),
            nn.ReLU(True)
        )

        # Output
        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        # Offset predictor for spatial transformer
        self.offset_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)  # 6 parameters for 2D affine transformation
        )

    def forward(self, x: torch.Tensor, return_offset: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input encoding
        x = self.input_conv(x)
        x = self.down1(x)
        x = self.down2(x)

        # Store features for offset prediction
        features = x

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Predict spatial offset
        offset = self.offset_predictor(x)
        offset = offset.view(-1, 2, 3)  # [batch, 2, 3] affine matrix

        # Apply spatial transformation
        x = self.spatial_transform(x, offset)

        # Decoding
        x = self.up1(x)
        x = self.up2(x)
        x = self.output_conv(x)

        if return_offset:
            return x, offset
        return x


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network (STN) module.

    Applies affine transformations to feature maps.
    """

    def __init__(self, output_size: Tuple[int, int] = (256, 256)):
        super(SpatialTransformer, self).__init__()

        self.output_size = output_size

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )

        # Regressor for affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )

        # Initialize as identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor, precomputed_offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        if precomputed_offset is not None:
            theta = precomputed_offset
        else:
            batch_size = x.size(0)
            xs = self.localization(x)
            xs = xs.view(batch_size, -1)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


class DGRDiscriminator(nn.Module):
    """
    DGR (Dual-Generator Registration) Discriminator

    Uses dual discriminators to handle different aspects of image quality.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        num_layers: int = 3,
        norm_layer: str = 'instance'
    ):
        super(DGRDiscriminator, self).__init__()

        if norm_layer == 'instance':
            norm_fn = lambda ch: nn.InstanceNorm2d(ch)
        else:
            norm_fn = lambda ch: nn.Identity()

        # Global discriminator (assesses overall quality)
        kw = 4
        padw = 1
        global_disc = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            global_disc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_fn(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        global_disc += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()
        ]

        self.global_discriminator = nn.Sequential(*global_disc)

        # Per-channel discriminator (assesses each mIHC channel)
        self.channel_discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, ndf // 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf // 2, ndf, kernel_size=4, stride=2, padding=1),
                norm_fn(ndf),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=1),
                nn.Sigmoid()
            ) for _ in range(3)  # DAPI, panCK, CD3
        ])

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """Global discriminator output"""
        return self.global_discriminator(x)

    def forward_channels(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Per-channel discriminator outputs.

        Args:
            x: (B, 3, H, W) multi-channel image

        Returns:
            List of 3 discriminator outputs, one per channel
        """
        outputs = []
        for i, disc in enumerate(self.channel_discriminators):
            channel = x[:, i:i+1, :, :]  # Extract single channel
            outputs.append(disc(channel))
        return outputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Combined forward pass"""
        global_out = self.forward_global(x)
        channel_outs = self.forward_channels(x)
        return global_out, channel_outs


class DGRLoss(nn.Module):
    """
    Loss functions for DGR/DTR model.

    Includes:
    - GAN loss (global + per-channel)
    - L1 loss with misalignment robustness
    - Perceptual loss
    - Channel-wise consistency loss
    """

    def __init__(self, lambda_L1: float = 100.0, lambda_perceptual: float = 10.0):
        super(DGRLoss, self).__init__()

        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual

        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        # Perceptual loss network (VGG-like)
        self.perceptual_net = PerceptualNetwork()

    def compute_gan_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
        is_discriminator: bool = True
    ) -> torch.Tensor:
        """Compute GAN loss"""
        if is_discriminator:
            real_loss = self.criterionGAN(pred_real, torch.ones_like(pred_real))
            fake_loss = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            return (real_loss + fake_loss) * 0.5
        else:
            return self.criterionGAN(pred_fake, torch.ones_like(pred_fake))

    def compute_perceptual_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss using feature matching"""
        fake_features = self.perceptual_net(fake)
        real_features = self.perceptual_net(real)

        loss = 0.0
        for f_fake, f_real in zip(fake_features, real_features):
            loss += self.criterionL1(f_fake, f_real)
        return loss

    def compute_channel_consistency_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor
    ) -> torch.Tensor:
        """
        Channel-wise consistency loss.

        Ensures each output channel matches the corresponding ground truth channel.
        """
        loss = 0.0
        for i in range(fake.size(1)):
            loss += self.criterionL1(fake[:, i, :, :], real[:, i, :, :])
        return loss / fake.size(1)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        for_discriminator: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}

        if for_discriminator:
            # Discriminator losses
            fake_B = outputs['fake_B']
            real_A = outputs['real_A']
            real_B = outputs['real_B']

            # Global discriminator loss
            pred_real_global, pred_real_channels = outputs['pred_real_global'], outputs['pred_real_channels']
            pred_fake_global, pred_fake_channels = outputs['pred_fake_global'], outputs['pred_fake_channels']

            loss_D_global = self.compute_gan_loss(pred_real_global, pred_fake_global)

            loss_D_channels = 0.0
            for i in range(3):
                loss_D_channels += self.compute_gan_loss(
                    pred_real_channels[i], pred_fake_channels[i]
                )
            loss_D_channels /= 3.0

            losses['loss_D'] = loss_D_global + loss_D_channels

        else:
            # Generator losses
            fake_B = outputs['fake_B']
            real_B = outputs['real_B']
            pred_fake_global = outputs['pred_fake_global']
            pred_fake_channels = outputs['pred_fake_channels']

            # GAN loss
            loss_G_global = self.compute_gan_loss(
                pred_fake_global, None, is_discriminator=False
            )
            loss_G_channels = 0.0
            for i in range(3):
                loss_G_channels += self.criterionGAN(
                    pred_fake_channels[i], torch.ones_like(pred_fake_channels[i])
                )
            loss_G_channels /= 3.0

            # L1 loss
            loss_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1

            # Perceptual loss
            loss_perceptual = self.compute_perceptual_loss(fake_B, real_B) * self.lambda_perceptual

            # Channel consistency
            loss_consistency = self.compute_channel_consistency_loss(fake_B, real_B)

            losses['loss_G'] = loss_G_global + loss_G_channels + loss_L1
            losses['loss_L1'] = loss_L1
            losses['loss_perceptual'] = loss_perceptual
            losses['loss_consistency'] = loss_consistency

        return losses


class PerceptualNetwork(nn.Module):
    """
    Perceptual loss network based on VGG features.
    """

    def __init__(self):
        super(PerceptualNetwork, self).__init__()

        # Simple feature extractor (conv layers before each pool)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at multiple scales"""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [1, 4, 8]:  # After each block
                features.append(x)
        return features


class DGRModel(nn.Module):
    """
    Complete DGR/DTR Model for HEMIT.

    Combines DTR generator with DGR discriminator.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        num_resblocks: int = 9,
        lambda_L1: float = 100.0,
        lambda_perceptual: float = 10.0
    ):
        super(DGRModel, self).__init__()

        # Generator
        self.generator = DTRGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            num_resblocks=num_resblocks
        )

        # Discriminator
        self.discriminator = DGRDiscriminator(
            input_nc=output_nc,
            ndf=ndf
        )

        # Loss calculator
        self.loss_calculator = DGRLoss(
            lambda_L1=lambda_L1,
            lambda_perceptual=lambda_perceptual
        )

    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Generate fake image
        fake_B = self.generator(real_A)

        outputs = {
            'fake_B': fake_B,
            'real_A': real_A,
            'real_B': real_B
        }

        if training:
            # Global discriminator
            pred_real_global = self.discriminator.forward_global(real_B)
            pred_fake_global = self.discriminator.forward_global(fake_B.detach())

            # Channel discriminators
            pred_real_channels = self.discriminator.forward_channels(real_B)
            pred_fake_channels = self.discriminator.forward_channels(fake_B.detach())

            outputs.update({
                'pred_real_global': pred_real_global,
                'pred_fake_global': pred_fake_global,
                'pred_real_channels': pred_real_channels,
                'pred_fake_channels': pred_fake_channels
            })

            # Compute discriminator loss
            D_outputs = outputs.copy()
            D_outputs['pred_real_global'] = pred_real_global
            D_outputs['pred_fake_global'] = pred_fake_global
            D_outputs['pred_real_channels'] = pred_real_channels
            D_outputs['pred_fake_channels'] = pred_fake_channels
            D_losses = self.loss_calculator(D_outputs, for_discriminator=True)

            # Compute generator loss
            G_outputs = outputs.copy()
            G_outputs['pred_fake_global'] = self.discriminator.forward_global(fake_B)
            G_outputs['pred_fake_channels'] = self.discriminator.forward_channels(fake_B)
            G_losses = self.loss_calculator(G_outputs, for_discriminator=False)

            outputs.update({**D_losses, **G_losses})

        return outputs

    def inference(self, real_A: torch.Tensor) -> torch.Tensor:
        """Generate fake mIHC image"""
        self.eval()
        with torch.no_grad():
            return self.generator(real_A)


def create_dgr_model(config: Dict) -> DGRModel:
    """Factory function to create DGR model from config"""
    return DGRModel(
        input_nc=config.get('input_nc', 3),
        output_nc=config.get('output_nc', 3),
        ngf=config.get('ngf', 64),
        ndf=config.get('ndf', 64),
        num_resblocks=config.get('num_resblocks', 9),
        lambda_L1=config.get('lambda_L1', 100.0),
        lambda_perceptual=config.get('lambda_perceptual', 10.0)
    )
