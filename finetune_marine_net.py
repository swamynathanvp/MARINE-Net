"""
MARINE-Net v3.0 Fine-tuning Script
Fine-tune the trained model with enhanced underwater-specific metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

from marine_net_model import create_model
from train_marine_net_kaggle import UnderwaterImageDataset


class EnhancedMARINELoss(nn.Module):
    """Enhanced loss focusing on underwater quality metrics"""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def color_constancy_loss(self, enhanced, target):
        """Encourage natural color balance"""
        # Calculate mean color per channel
        enhanced_mean = torch.mean(enhanced, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        
        return self.mse_loss(enhanced_mean, target_mean)
    
    def saturation_loss(self, enhanced, target):
        """Encourage proper color saturation"""
        # Calculate saturation (std of color channels)
        enhanced_sat = torch.std(enhanced, dim=1)
        target_sat = torch.std(target, dim=1)
        
        return self.mse_loss(enhanced_sat, target_sat)
    
    def edge_sharpness_loss(self, enhanced, target):
        """Encourage sharp edges using Sobel-like operator"""
        # Sobel kernels
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                               dtype=enhanced.dtype, device=enhanced.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                               dtype=enhanced.dtype, device=enhanced.device)
        
        # Expand for 3 channels
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)
        
        # Compute gradients
        enhanced_grad_x = torch.nn.functional.conv2d(enhanced, sobel_x, groups=3, padding=1)
        enhanced_grad_y = torch.nn.functional.conv2d(enhanced, sobel_y, groups=3, padding=1)
        enhanced_grad = torch.sqrt(enhanced_grad_x**2 + enhanced_grad_y**2 + 1e-8)
        
        target_grad_x = torch.nn.functional.conv2d(target, sobel_x, groups=3, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target, sobel_y, groups=3, padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return self.mse_loss(enhanced_grad, target_grad)
    
    def forward(self, enhanced, target):
        """Compute combined loss"""
        # Ensure both are tensors and have the same shape
        if isinstance(enhanced, dict):
            enhanced = enhanced['enhanced']
        
        # Ensure both are in valid range
        enhanced = torch.clamp(enhanced, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Base reconstruction loss
        l1 = self.l1_loss(enhanced, target)
        
        # Underwater-specific losses
        color_const = self.color_constancy_loss(enhanced, target)
        saturation = self.saturation_loss(enhanced, target)
        edge_sharp = self.edge_sharpness_loss(enhanced, target)
        
        # Weighted combination (tuned for UIQM/UCIQE)
        total = (
            1.0 * l1 +
            0.3 * color_const +
            0.3 * saturation +
            0.2 * edge_sharp
        )
        
        return {
            'total': total,
            'l1': l1,
            'color_constancy': color_const,
            'saturation': saturation,
            'edge_sharpness': edge_sharp
        }


class FineTuner:
    """Fine-tuning trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        self.model = create_model().to(self.device)
        checkpoint = torch.load(config['pretrained_path'])
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded pretrained model from {config['pretrained_path']}")
        
        # Enhanced loss
        self.criterion = EnhancedMARINELoss().to(self.device)
        
        # Lower learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-8
        )
        
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'FineTune Epoch {epoch}')
        
        for batch in pbar:
            input_img = batch['input'].to(self.device)
            target_img = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward - model returns dict with 'enhanced' key
            outputs = self.model(input_img)
            
            # Extract enhanced image from dictionary
            if isinstance(outputs, dict):
                enhanced = outputs['enhanced']
            else:
                enhanced = outputs
            
            loss_dict = self.criterion(enhanced, target_img)
            loss = loss_dict['total']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{loss_dict["l1"].item():.4f}',
                'cc': f'{loss_dict["color_constancy"].item():.4f}',
                'sat': f'{loss_dict["saturation"].item():.4f}',
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_img = batch['input'].to(self.device)
                target_img = batch['target'].to(self.device)
                
                outputs = self.model(input_img)
                
                # Extract enhanced image from dictionary
                if isinstance(outputs, dict):
                    enhanced = outputs['enhanced']
                else:
                    enhanced = outputs
                
                loss_dict = self.criterion(enhanced, target_img)
                total_loss += loss_dict['total'].item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        print(f"Starting fine-tuning for {self.config['num_epochs']} epochs")
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            self.scheduler.step()
            
            # Save best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                path = self.checkpoint_dir / 'best_finetuned.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, path)
                print(f"✓ Saved best model: val_loss={val_loss:.4f}")
            
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"LR={self.optimizer.param_groups[0]['lr']:.7f}")
        
        # Save final
        final_path = self.checkpoint_dir / 'marine_net_finetuned_final.pth'
        torch.save(self.model.state_dict(), final_path)
        print(f"\nFine-tuning complete! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/marine_net_v3_finetuned')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # 10x lower for fine-tuning
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    config = {
        'pretrained_path': args.pretrained_path,
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
    }
    
    # Create datasets
    train_dataset = UnderwaterImageDataset(
        root_dir=config['data_root'],
        split='train',
        image_size=config['image_size'],
        augment=True
    )
    
    val_dataset = UnderwaterImageDataset(
        root_dir=config['data_root'],
        split='val',
        image_size=config['image_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Fine-tune
    finetuner = FineTuner(config)
    finetuner.train(train_loader, val_loader)


if __name__ == '__main__':
    main()