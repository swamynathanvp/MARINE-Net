"""
MARINE-Net Multi-Seed Training
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
import argparse
import json

from train_marine_net import UnderwaterImageDataset, Trainer


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds strictly set to {seed}")


def main():
    parser = argparse.ArgumentParser(description='MARINE-Net Multi-Seed Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True, help='Random seed (42, 123, 2024)')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    
    # 32-bit stable settings (No AMP)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_frequency', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None, help='Force manual resume path')

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'./checkpoints/seed_{args.seed}'

    set_all_seeds(args.seed)

    config = {
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
        'use_amp': True,
        'checkpoint_dir': args.checkpoint_dir,
        'data_root': args.data_root,
        'save_frequency': args.save_frequency,
        'seed': args.seed,
    }

    print("=" * 60)
    print(f"MARINE-Net Multi-Seed Training (seed={args.seed}) - AUTO-RESUME ENABLED")
    print("=" * 60)

    train_dataset = UnderwaterImageDataset(
        root_dir=config['data_root'], split='train',
        image_size=config['image_size'], augment=True
    )
    val_dataset = UnderwaterImageDataset(
        root_dir=config['data_root'], split='val',
        image_size=config['image_size'], augment=False
    )

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        persistent_workers=config['num_workers'] > 0,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        persistent_workers=config['num_workers'] > 0,
    )

    trainer = Trainer(config)

   
    # AUTO-RESUME LOGIC

    start_epoch = 0
    latest_ckpt_path = Path(config['checkpoint_dir']) / 'latest_checkpoint.pth'
    
    if args.resume:
    
        start_epoch = trainer.load_checkpoint(args.resume)
    elif latest_ckpt_path.exists():
        
        print("\n[!] Previous session detected. Auto-resuming to survive Kaggle timeout...")
        start_epoch = trainer.load_checkpoint(str(latest_ckpt_path))

    # Train
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    # Save metadata
    seed_info = {
        'seed': args.seed,
        'num_epochs': args.num_epochs,
        'best_val_loss': trainer.best_val_loss,
        'checkpoint_dir': args.checkpoint_dir,
    }
    info_path = Path(args.checkpoint_dir) / 'seed_info.json'
    with open(info_path, 'w') as f:
        json.dump(seed_info, f, indent=2)


if __name__ == '__main__':
    main()