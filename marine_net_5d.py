"""
MARINE-Net Architecture 5D.

KEY ARCHITECTURAL SHIFT:
Spatially-Varying Neural Color LUT taking a 5D input: (x, y, R, G, B) mapping to (R', G', B').
Gradients for the CNN and INR are 100% decoupled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import math

try:
    from pytorch_msssim import SSIM
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


class ImplicitNeuralRepresentation(nn.Module):
    """
    Spatially-Varying Neural Color LUT.
    Maps (x, y, r, g, b) -> (r', g', b') using SIREN activations.
    """
    def __init__(self, hidden_dim: int = 64, out_features: int = 3, 
                 num_layers: int = 3, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
        
        # Positional encoding frequencies (only for spatial coords)
        self.water_frequencies = nn.Parameter(torch.randn(2, hidden_dim // 4) * 0.01)
        pe_dim = (hidden_dim // 4) * 2
        
        # Input dimension = PE(coords) + raw coords + raw RGB
        in_dim = pe_dim + 2 + 3
        
        # SIREN layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_features))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                num_input = layer.weight.shape[1]
                if i == 0:
                    layer.weight.uniform_(-1 / num_input, 1 / num_input)
                else:
                    w_std = math.sqrt(6 / num_input) / self.omega_0
                    layer.weight.uniform_(-w_std, w_std)
                if layer.bias is not None:
                    layer.bias.uniform_(-1 / num_input, 1 / num_input)
                    
    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        water_modulated = torch.matmul(coords, self.water_frequencies)
        water_modulated = torch.clamp(water_modulated, -5, 5)
        sin_enc = torch.sin(water_modulated)
        cos_enc = torch.cos(water_modulated)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)
        return torch.clamp(encoded, -1, 1)
    
    def forward(self, coords: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        # Encode only the spatial coordinates
        encoded_coords = self.positional_encoding(coords)
        
        # Concatenate: [PE_Coords, Raw_Coords, Raw_RGB]
        x = torch.cat([encoded_coords, coords, rgb], dim=-1)
        
        # SIREN forward pass
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.sin(self.omega_0 * x)
            x = torch.clamp(x, -10, 10)
            
        x = self.layers[-1](x)
        return torch.sigmoid(x)


class FrequencyAwareDecomposition(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.low_freq_conv = nn.Conv2d(3, channels // 4, 7, padding=3)
        self.mid_freq_conv = nn.Conv2d(3, channels // 4, 5, padding=2)
        self.high_freq_conv = nn.Conv2d(3, channels // 4, 3, padding=1)
        self.ultra_high_freq_conv = nn.Conv2d(3, channels // 4, 1)
        self.turbidity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(3, channels, 1), nn.ReLU(),
            nn.Conv2d(channels, 4, 1), nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        turbidity_weights = self.turbidity_estimator(x)
        features = torch.cat([
            self.low_freq_conv(x) * turbidity_weights[:, 0:1],
            self.mid_freq_conv(x) * turbidity_weights[:, 1:2],
            self.high_freq_conv(x) * turbidity_weights[:, 2:3],
            self.ultra_high_freq_conv(x) * turbidity_weights[:, 3:4]
        ], dim=1)
        return self.fusion(features), turbidity_weights


class AdaptivePhysicsGuidedAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.absorption_branch = nn.Sequential(nn.Conv2d(channels, channels // 2, 1), nn.ReLU(), nn.Conv2d(channels // 2, 3, 1))
        self.scattering_branch = nn.Sequential(nn.Conv2d(channels, channels // 2, 1), nn.ReLU(), nn.Conv2d(channels // 2, 1, 1))
        self.depth_estimator = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, padding=1), nn.ReLU(), nn.Conv2d(channels // 2, 1, 1), nn.Sigmoid())
        self.channel_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 8, 1), nn.ReLU(), nn.Conv2d(channels // 8, channels, 1), nn.Sigmoid())
        self.spatial_att = nn.Sequential(nn.Conv2d(channels, 1, 7, padding=3), nn.Sigmoid())
        
    def forward(self, features, image):
        absorption = torch.sigmoid(self.absorption_branch(features))
        scattering = torch.clamp(torch.sigmoid(self.scattering_branch(features)), 0.01, 0.99)
        depth = torch.clamp(self.depth_estimator(features), 0.01, 0.99)
        transmission = torch.clamp(torch.exp(-scattering * depth), 0.01, 1.0)
        absorbed_features = (features.unsqueeze(2) * (1 - absorption.unsqueeze(1) * depth.unsqueeze(1))).mean(dim=2)
        return (absorbed_features * self.channel_att(absorbed_features) * self.spatial_att(absorbed_features) * transmission) + features


class LightweightResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels)
        )
    def forward(self, x): return F.relu(x + self.conv(x))


class MARINENet(nn.Module):
    def __init__(self, base_channels: int = 80, num_blocks: int = 4):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, base_channels, 3, padding=1)
        self.freq_decompose = FrequencyAwareDecomposition(base_channels)
        
        # CNN Encoder
        self.encoder_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        channels = base_channels
        for i in range(3):
            self.encoder_blocks.append(nn.Sequential(*[LightweightResidualBlock(channels) for _ in range(num_blocks)]))
            if i < 2:
                self.downsample.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
                channels *= 2
                
        self.physics_attention = AdaptivePhysicsGuidedAttention(channels)
        
        # CNN Decoder
        self.decoder_blocks, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i in range(2):
            channels //= 2
            self.upsample.append(nn.ConvTranspose2d(channels * 2, channels, 3, stride=2, padding=1, output_padding=1))
            self.decoder_blocks.append(nn.Sequential(*[LightweightResidualBlock(channels) for _ in range(num_blocks)]))
            
        self.reconstruct = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, 3, 1), nn.Sigmoid()
        )
        
        # 100% INDEPENDENT INR BRANCH
        self.inr = ImplicitNeuralRepresentation(hidden_dim=64, num_layers=3)
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        
    def generate_coordinate_grid(self, h, w, device):
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        x = torch.clamp(x, 0, 1)
        
        # ============================================================
        # BRANCH 1: DISCRETE CNN
        # ============================================================
        features = self.initial_conv(x)
        freq_features, turbidity = self.freq_decompose(x)
        features = features + freq_features
        
        encoder_features = []
        for i, (blocks, down) in enumerate(zip(self.encoder_blocks[:-1], self.downsample)):
            features = blocks(features)
            encoder_features.append(features)
            features = down(features)
            
        features = self.encoder_blocks[-1](features)
        features = self.physics_attention(features, F.interpolate(x, size=features.shape[-2:], mode='bilinear', align_corners=False))
        
        for i, (up, blocks) in enumerate(zip(self.upsample, self.decoder_blocks)):
            features = up(features)
            if features.shape != encoder_features[-(i+1)].shape:
                features = F.interpolate(features, size=encoder_features[-(i+1)].shape[-2:], mode='bilinear', align_corners=False)
            features = features + encoder_features[-(i+1)]
            features = blocks(features)
            
        cnn_output = torch.clamp(self.reconstruct(features), 0, 1)
        
        # ============================================================
        # BRANCH 2: CONTINUOUS INR (100% Independent 5D Neural LUT)
        # ============================================================
        coords = self.generate_coordinate_grid(h, w, x.device).unsqueeze(0).expand(b, -1, -1)
        
        # Flatten the original degraded RGB image: [B, 3, H, W] -> [B, H*W, 3]
        rgb_flat = x.view(b, c, h * w).permute(0, 2, 1)
        
        # Fast Vectorized INR inference
        inr_pred = self.inr(coords, rgb_flat)
        inr_output = torch.clamp(inr_pred.view(b, h, w, 3).permute(0, 3, 1, 2), 0, 1)
        
        # ============================================================
        # LATE FUSION
        # ============================================================
        alpha_scalar = torch.sigmoid(self.fusion_weight)
        alpha = alpha_scalar.expand_as(cnn_output[:, :1])
        enhanced = torch.clamp(alpha_scalar * cnn_output + (1 - alpha_scalar) * inr_output, 0, 1)
        
        if torch.isnan(enhanced).any(): enhanced = cnn_output
        
        return {
            'enhanced': enhanced, 'cnn_output': cnn_output, 'inr_output': inr_output,
            'turbidity': turbidity, 'alpha': alpha, 'fusion_weight': alpha_scalar
        }


# ======================================================================
# Loss Functions
# ======================================================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]
            self.vgg = vgg.eval()
            for param in self.vgg.parameters(): param.requires_grad = False
            if torch.cuda.is_available(): self.vgg = self.vgg.cuda()
            self.use_vgg = True
        except Exception:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
            )
            for param in self.feature_extractor.parameters(): param.requires_grad = False
            if torch.cuda.is_available(): self.feature_extractor = self.feature_extractor.cuda()
            self.feature_extractor.eval()
            self.use_vgg = False
            
    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            pred, target = torch.clamp(pred.float(), 0, 1), torch.clamp(target.float(), 0, 1)
            if self.use_vgg:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
                pred_features = self.vgg((pred - mean) / std)
                target_features = self.vgg((target - mean) / std)
            else:
                pred_features, target_features = self.feature_extractor(pred), self.feature_extractor(target)
            loss = F.mse_loss(pred_features, target_features)
            if torch.isnan(loss) or torch.isinf(loss): loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        return loss

class MARINELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        if SSIM_AVAILABLE:
            self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)
        else:
            self.ssim_loss = None
        
    def gradient_loss(self, pred, target):
        def gradient(img):
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            return dx, dy
        pdx, pdy = gradient(pred); tdx, tdy = gradient(target)
        pdx, pdy = torch.clamp(pdx, -1.0, 1.0), torch.clamp(pdy, -1.0, 1.0)
        tdx, tdy = torch.clamp(tdx, -1.0, 1.0), torch.clamp(tdy, -1.0, 1.0)
        return (self.l1(pdx, tdx) + self.l1(pdy, tdy)) / 2
    
    def color_consistency_loss(self, pred, target):
        pm = pred.mean(dim=[2, 3], keepdim=True); tm = target.mean(dim=[2, 3], keepdim=True)
        ps = pred.std(dim=[2, 3], keepdim=True) + 1e-6; ts = target.std(dim=[2, 3], keepdim=True) + 1e-6
        return self.l2(pm, tm) + self.l2(ps, ts)
    
    def forward(self, outputs, target):
        enhanced   = torch.clamp(outputs['enhanced'], 0, 1)
        cnn_output = torch.clamp(outputs['cnn_output'], 0, 1)
        inr_output = torch.clamp(outputs['inr_output'], 0, 1)
        target     = torch.clamp(target, 0, 1)
        
        loss_l1     = self.l1(enhanced, target)
        loss_cnn_l1 = self.l1(cnn_output, target)
        loss_inr_l1 = self.l1(inr_output, target)
        
        loss_ssim = 1 - self.ssim_loss(enhanced, target) if self.ssim_loss is not None else torch.tensor(0.0, device=enhanced.device, requires_grad=True)
        loss_perceptual = self.perceptual(enhanced, target)
        loss_gradient   = self.gradient_loss(enhanced, target)
        loss_color      = self.color_consistency_loss(enhanced, target)
        
        total = (1.0 * loss_l1 + 0.3 * (loss_cnn_l1 + loss_inr_l1) + 0.5 * loss_ssim + 0.0 * loss_perceptual + 0.4 * loss_gradient + 0.2 * loss_color)
        if torch.isnan(total) or torch.isinf(total): total = loss_l1
        
        return {'total': total, 'l1': loss_l1, 'ssim': loss_ssim, 'perceptual': loss_perceptual, 'gradient': loss_gradient, 'color': loss_color}


def create_model(pretrained: bool = False) -> nn.Module:
    return MARINENet(base_channels=80, num_blocks=4)
