"""MARINE-Net Training Script."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import json
import time
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from marine_net_model import create_model, MARINELoss


class UnderwaterImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256, augment: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        input_dir = self.root_dir / split / 'input'
        target_dir = self.root_dir / split / 'target'
        
        self.image_pairs = []
        if input_dir.exists() and target_dir.exists():
            input_images = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
            for input_path in input_images:
                target_path = target_dir / input_path.name
                if target_path.exists():
                    self.image_pairs.append((str(input_path), str(target_path)))
        
        print(f"Found {len(self.image_pairs)} image pairs for {split}")
        
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.image_pairs[idx]
        name = Path(input_path).name
        
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if self.augment:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)
            target_img = self.transform(target_img)
        else:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        return {'input': input_img, 'target': target_img, 'name': name}

class Trainer:
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = create_model().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")
        
        self.criterion = MARINELoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        self.warmup_epochs = 10
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'] - self.warmup_epochs,
            eta_min=1e-7
        )
        
        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        nan_count = 0
        
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            input_img = batch['input'].to(self.device)
            target_img = batch['target'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(input_img)
                    loss_dict = self.criterion(outputs, target_img)
                    loss = loss_dict['total'] / self.gradient_accumulation_steps
            else:
                outputs = self.model(input_img)
                loss_dict = self.criterion(outputs, target_img)
                loss = loss_dict['total'] / self.gradient_accumulation_steps
            
            if torch.isnan(loss):
                nan_count += 1
                continue
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'l1': f'{loss_dict["l1"].item():.4f}',
                'ssim': f'{loss_dict.get("ssim", torch.tensor(0.0)).item():.4f}',
                'lr': f'{self.get_lr():.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, nan_count
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch in pbar:
                input_img = batch['input'].to(self.device)
                target_img = batch['target'].to(self.device)
                
                outputs = self.model(input_img)
                loss_dict = self.criterion(outputs, target_img)
                loss = loss_dict['total']
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def apply_warmup(self, epoch):
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lr = self.config['learning_rate'] * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if epoch % self.config['save_frequency'] == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    
    def train(self, train_loader, val_loader, start_epoch=0):
        print(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            if epoch < self.warmup_epochs:
                self.apply_warmup(epoch)
            
            train_loss, nan_count = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"LR={self.get_lr():.6f}, Best={self.best_val_loss:.4f}, NaN count={nan_count}")
        
        final_path = self.checkpoint_dir / 'marine_net_final.pth'
        torch.save(self.model.state_dict(), final_path)
        print("Training completed.")


def main():
    parser = argparse.ArgumentParser(description='Train MARINE-Net')
    
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/marine_net_v3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--save_frequency', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    config = vars(args)
    
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
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    trainer = Trainer(config)
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)


if __name__ == '__main__':
    main()