"""MARINE-Net Architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math

try:
    from pytorch_msssim import SSIM
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, in_features: int = 2, hidden_dim: int = 64, out_features: int = 3, 
                 num_layers: int = 3, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.water_frequencies = nn.Parameter(torch.randn(in_features, hidden_dim // 4) * 0.01)
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = nn.Linear(in_features + hidden_dim // 2, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, out_features)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            layers.append(layer)
            
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
                        
    def positional_encoding(self, coords):
        if coords.dim() != 2:
            coords = coords.reshape(-1, coords.shape[-1])
        
        water_modulated = torch.matmul(coords, self.water_frequencies)
        
        if water_modulated.dim() > 2:
            water_modulated = water_modulated.squeeze(0)
        
        water_modulated = torch.clamp(water_modulated, -5, 5)
        
        sin_enc = torch.sin(water_modulated)
        cos_enc = torch.cos(water_modulated)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)
        encoded = torch.clamp(encoded, -1, 1)
        
        if encoded.dim() > 2:
            encoded = encoded.squeeze(0)
        
        return encoded
    
    def forward(self, coords):
        if coords.dim() != 2:
            coord_dim = coords.shape[-1]
            coords = coords.reshape(-1, coord_dim)
        
        coords = torch.clamp(coords, -1, 1)
        encoded = self.positional_encoding(coords)
        
        if encoded.dim() != 2:
            encoded = encoded.reshape(-1, 32)
        if coords.dim() != 2:
            coords = coords.reshape(-1, 2)
        
        x = torch.cat([encoded, coords], dim=-1)
        
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
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 4, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        turbidity_weights = self.turbidity_estimator(x)
        
        low = self.low_freq_conv(x)
        mid = self.mid_freq_conv(x)
        high = self.high_freq_conv(x)
        ultra_high = self.ultra_high_freq_conv(x)
        
        features = torch.cat([
            low * turbidity_weights[:, 0:1],
            mid * turbidity_weights[:, 1:2],
            high * turbidity_weights[:, 2:3],
            ultra_high * turbidity_weights[:, 3:4]
        ], dim=1)
        
        return self.fusion(features), turbidity_weights


class AdaptivePhysicsGuidedAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        
        self.absorption_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 3, 1)
        )
        
        self.scattering_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 1)
        )
        
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        absorption = torch.sigmoid(self.absorption_branch(features))
        scattering = torch.sigmoid(self.scattering_branch(features))
        depth = self.depth_estimator(features)
        
        scattering = torch.clamp(scattering, 0.01, 0.99)
        depth = torch.clamp(depth, 0.01, 0.99)
        
        transmission = torch.exp(-scattering * depth)
        transmission = torch.clamp(transmission, 0.01, 1.0)
        
        absorbed_features = features.unsqueeze(2) * (1 - absorption.unsqueeze(1) * depth.unsqueeze(1))
        absorbed_features = absorbed_features.mean(dim=2)
        
        ca = self.channel_att(absorbed_features)
        sa = self.spatial_att(absorbed_features)
        
        attended = absorbed_features * ca * sa * transmission
        return attended + features


class LightweightResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x))


class MARINENet(nn.Module):
    def __init__(self, base_channels: int = 80, num_blocks: int = 4):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, base_channels, 3, padding=1)
        self.freq_decompose = FrequencyAwareDecomposition(base_channels)
        
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        channels = base_channels
        for i in range(3):
            blocks = nn.Sequential(*[
                LightweightResidualBlock(channels) for _ in range(num_blocks)
            ])
            self.encoder_blocks.append(blocks)
            
            if i < 2:
                self.downsample.append(
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
                )
                channels *= 2
        
        self.physics_attention = AdaptivePhysicsGuidedAttention(channels)
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in range(2):
            channels //= 2
            self.upsample.append(
                nn.ConvTranspose2d(channels * 2, channels, 3, stride=2, padding=1, output_padding=1)
            )
            blocks = nn.Sequential(*[
                LightweightResidualBlock(channels) for _ in range(num_blocks)
            ])
            self.decoder_blocks.append(blocks)
        
        self.inr = ImplicitNeuralRepresentation(
            in_features=2,
            hidden_dim=64,
            out_features=3,
            num_layers=3
        )
        
        self.reconstruct = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, 3, 1),
            nn.Sigmoid()
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.95))
        
    def generate_coordinate_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        x = torch.clamp(x, 0, 1)
        
        features = self.initial_conv(x)
        freq_features, turbidity = self.freq_decompose(x)
        features = features + freq_features
        
        encoder_features = []
        for i, (blocks, down) in enumerate(zip(self.encoder_blocks[:-1], self.downsample)):
            features = blocks(features)
            encoder_features.append(features)
            features = down(features)
        
        features = self.encoder_blocks[-1](features)
        features = self.physics_attention(features, 
                                         F.interpolate(x, size=features.shape[-2:], mode='bilinear', align_corners=False))
        
        for i, (up, blocks) in enumerate(zip(self.upsample, self.decoder_blocks)):
            features = up(features)
            if features.shape != encoder_features[-(i+1)].shape:
                features = F.interpolate(features, size=encoder_features[-(i+1)].shape[-2:], 
                                       mode='bilinear', align_corners=False)
            features = features + encoder_features[-(i+1)]
            features = blocks(features)
        
        cnn_output = torch.clamp(self.reconstruct(features), 0, 1)
        
        coords = self.generate_coordinate_grid(h, w, x.device)
        coords = coords.unsqueeze(0).expand(b, -1, -1)
        
        inr_output = []
        for i in range(b):
            inr_pred = self.inr(coords[i])
            inr_pred = inr_pred.reshape(h, w, 3).permute(2, 0, 1)
            inr_output.append(inr_pred)
            
        inr_output = torch.clamp(torch.stack(inr_output, dim=0), 0, 1)
        
        weight = torch.clamp(torch.sigmoid(self.fusion_weight), 0.7, 0.99)
        enhanced = torch.clamp(weight * cnn_output + (1 - weight) * inr_output, 0, 1)
        
        if torch.isnan(enhanced).any():
            enhanced = cnn_output
        
        return {
            'enhanced': enhanced,
            'cnn_output': cnn_output,
            'inr_output': inr_output,
            'turbidity': turbidity
        }


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]
            self.vgg = vgg.eval()
            
            for param in self.vgg.parameters():
                param.requires_grad = False
            
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
            
            self.use_vgg = True
        except Exception:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            )
            
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            
            if torch.cuda.is_available():
                self.feature_extractor = self.feature_extractor.cuda()
            
            self.feature_extractor.eval()
            self.use_vgg = False
            
    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            pred = torch.clamp(pred.float(), 0, 1)
            target = torch.clamp(target.float(), 0, 1)
            
            if self.use_vgg:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
                
                pred_features = self.vgg((pred - mean) / std)
                target_features = self.vgg((target - mean) / std)
            else:
                pred_features = self.feature_extractor(pred)
                target_features = self.feature_extractor(target)
            
            loss = F.mse_loss(pred_features, target_features)
            
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss


class MARINELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        
        if SSIM_AVAILABLE:
            self.ssim_loss = SSIM(
                data_range=1.0,
                size_average=True,
                channel=3,
                nonnegative_ssim=True
            )
            print("SSIM loss enabled.")
        else:
            self.ssim_loss = None
            print("SSIM loss disabled. Install pytorch-msssim.")
        
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def gradient(img):
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            return dx, dy
        
        pred_dx, pred_dy = gradient(pred)
        target_dx, target_dy = gradient(target)
        
        pred_dx = torch.clamp(pred_dx, -1.0, 1.0)
        pred_dy = torch.clamp(pred_dy, -1.0, 1.0)
        target_dx = torch.clamp(target_dx, -1.0, 1.0)
        target_dy = torch.clamp(target_dy, -1.0, 1.0)
        
        return (self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)) / 2
    
    def color_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=[2, 3], keepdim=True)
        target_mean = target.mean(dim=[2, 3], keepdim=True)
        
        pred_std = pred.std(dim=[2, 3], keepdim=True) + 1e-6
        target_std = target.std(dim=[2, 3], keepdim=True) + 1e-6
        
        return self.l2(pred_mean, target_mean) + self.l2(pred_std, target_std)
    
    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        enhanced = torch.clamp(outputs['enhanced'], 0, 1)
        cnn_output = torch.clamp(outputs['cnn_output'], 0, 1)
        inr_output = torch.clamp(outputs['inr_output'], 0, 1)
        target = torch.clamp(target, 0, 1)
        
        loss_l1 = self.l1(enhanced, target)
        loss_cnn_l1 = self.l1(cnn_output, target)
        loss_inr_l1 = self.l1(inr_output, target)
        
        if self.ssim_loss is not None:
            loss_ssim = 1 - self.ssim_loss(enhanced, target)
        else:
            loss_ssim = torch.tensor(0.0, device=enhanced.device, requires_grad=True)
        
        loss_perceptual = self.perceptual(enhanced, target)
        loss_gradient = self.gradient_loss(enhanced, target)
        loss_color = self.color_consistency_loss(enhanced, target)
        
        if torch.isnan(loss_l1) or torch.isinf(loss_l1):
            loss_l1 = torch.tensor(0.0, device=loss_l1.device, requires_grad=True)
        if torch.isnan(loss_ssim) or torch.isinf(loss_ssim):
            loss_ssim = torch.tensor(0.0, device=loss_ssim.device, requires_grad=True)
        if torch.isnan(loss_perceptual) or torch.isinf(loss_perceptual):
            loss_perceptual = torch.tensor(0.0, device=loss_perceptual.device, requires_grad=True)
        if torch.isnan(loss_gradient) or torch.isinf(loss_gradient):
            loss_gradient = torch.tensor(0.0, device=loss_gradient.device, requires_grad=True)
        if torch.isnan(loss_color) or torch.isinf(loss_color):
            loss_color = torch.tensor(0.0, device=loss_color.device, requires_grad=True)
        
        total_loss = (
            1.0 * loss_l1 +
            0.3 * (loss_cnn_l1 + loss_inr_l1) +
            0.5 * loss_ssim +
            0.0 * loss_perceptual +
            0.4 * loss_gradient +
            0.2 * loss_color
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = loss_l1
        
        return {
            'total': total_loss,
            'l1': loss_l1,
            'ssim': loss_ssim,
            'perceptual': loss_perceptual,
            'gradient': loss_gradient,
            'color': loss_color
        }


def create_model(pretrained: bool = False) -> nn.Module:
    model = MARINENet(base_channels=80, num_blocks=4)
    if pretrained:
        pass
    return model


def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_memory_usage(model: nn.Module, batch_size: int = 4, 
                         image_size: int = 256) -> float:
    input_size = batch_size * 3 * image_size * image_size * 4
    param_size = get_parameter_count(model) * 4
    activation_size = param_size * 2.5
    
    return (input_size + param_size + activation_size) / (1024 * 1024)


if __name__ == "__main__":
    model = create_model()
    
    print("MARINE-Net Model Statistics:")
    print(f"Total Parameters: {get_parameter_count(model):,}")
    print(f"Model Size: {get_parameter_count(model) * 4 / (1024*1024):.2f} MB")
    print(f"Estimated Memory (batch=4, size=256): {estimate_memory_usage(model, 4, 256):.2f} MB")
    
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nOutput shape: {output['enhanced'].shape}")