import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import csv
import time
import argparse
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from marine_net_model import create_model, MARINENet

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(target, pred, channel_axis=2, data_range=1.0)

def compute_entropy(image: np.ndarray) -> float:
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_uiqm(image: np.ndarray) -> float:
    if image.max() > 1.0:
        image = image / 255.0
    img_uint8 = (image * 255).astype(np.uint8)

    R = image[:, :, 0].astype(np.float64)
    G = image[:, :, 1].astype(np.float64)
    B = image[:, :, 2].astype(np.float64)
    RG = R - G
    YB = (R + G) / 2.0 - B
    uicm = -0.0268 * np.sqrt(np.mean(RG)**2 + np.mean(YB)**2) + \
            0.1586 * np.sqrt(np.std(RG)**2 + np.std(YB)**2)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    block_size = 8
    h, w = edge_magnitude.shape
    num_bh = max(1, h // block_size)
    num_bw = max(1, w // block_size)
    eme_sum = 0.0
    num_valid = 0
    
    for i in range(num_bh):
        for j in range(num_bw):
            block = edge_magnitude[i*block_size:(i+1)*block_size,
                                   j*block_size:(j+1)*block_size]
            if block.size == 0:
                continue
            bmax = np.max(block)
            bmin = np.min(block)
            if bmin > 0.01 and bmax > bmin:
                eme_sum += np.log(bmax / bmin)
                num_valid += 1
                
    uism = (2.0 / max(num_valid, 1)) * eme_sum

    gray_f = gray.astype(np.float64)
    plip_sum = 0.0
    num_valid_c = 0
    
    for i in range(num_bh):
        for j in range(num_bw):
            block = gray_f[i*block_size:(i+1)*block_size,
                           j*block_size:(j+1)*block_size]
            if block.size == 0:
                continue
            bmax = np.max(block)
            bmin = np.min(block)
            if bmax + bmin > 0:
                plip_sum += (bmax - bmin) / (bmax + bmin + 1e-7)
                num_valid_c += 1
                
    uiconm = plip_sum / max(num_valid_c, 1)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.3253 * uiconm
    return float(uiqm)

def compute_uciqe(image: np.ndarray) -> float:
    if image.max() <= 1.0:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
        
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    
    L = L.astype(np.float64) / 255.0 * 100.0
    a = a.astype(np.float64) - 128.0
    b = b.astype(np.float64) - 128.0
    
    chroma = np.sqrt(a**2 + b**2)
    sigma_c_norm = np.std(chroma) / 64.0
    L_sorted = np.sort(L.flatten())
    n = len(L_sorted)
    top_idx = max(1, int(0.01 * n))
    L_high = np.mean(L_sorted[-top_idx:])
    L_low = np.mean(L_sorted[:top_idx])
    
    con_l_norm = (L_high - L_low) / 100.0
    saturation = chroma / (L + 0.01)
    mu_s_norm = min(np.mean(saturation) / 1.5, 1.0)
    uciqe = 0.4680 * sigma_c_norm + 0.2745 * con_l_norm + 0.2576 * mu_s_norm
    
    return float(uciqe)

def init_pyiqa_metrics(device):
    try:
        import pyiqa
        niqe = pyiqa.create_metric('niqe', device=device)
        piqe = pyiqa.create_metric('piqe', device=device)
        return niqe, piqe
    except ImportError:
        return None, None

def compute_niqe_piqe(enhanced_tensor, niqe_metric, piqe_metric):
    niqe_val = float('nan')
    piqe_val = float('nan')
    
    if niqe_metric is not None:
        try:
            with torch.no_grad():
                niqe_val = niqe_metric(enhanced_tensor).item()
        except Exception:
            pass
            
    if piqe_metric is not None:
        try:
            with torch.no_grad():
                piqe_val = piqe_metric(enhanced_tensor).item()
        except Exception:
            pass
            
    return niqe_val, piqe_val

def evaluate(model_path: str, data_root: str, output_dir: str, device: str = 'cuda'):
    os.makedirs(output_dir, exist_ok=True)
    
    enhanced_dir = Path(output_dir) / 'enhanced_images'
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    model = create_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    raw_w = model.fusion_weight.item()
    alpha = torch.sigmoid(torch.tensor(raw_w)).item()
    alpha_clamped = max(0.7, min(0.99, alpha))
    total_params = sum(p.numel() for p in model.parameters())

    niqe_metric, piqe_metric = init_pyiqa_metrics(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    val_input_dir = Path(data_root) / 'val' / 'input'
    val_target_dir = Path(data_root) / 'val' / 'target'
    input_images = sorted(list(val_input_dir.glob('*.png')) + list(val_input_dir.glob('*.jpg')))

    all_results = []
    niqe_input_all = []
    piqe_input_all = []
    total_inference_time = 0.0

    for img_path in tqdm(input_images):
        target_path = val_target_dir / img_path.name
        if not target_path.exists():
            continue

        inp_img = Image.open(img_path).convert('RGB')
        tgt_img = Image.open(target_path).convert('RGB')
        inp_tensor = transform(inp_img).unsqueeze(0).to(device)
        tgt_tensor = transform(tgt_img).unsqueeze(0).to(device)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            outputs = model(inp_tensor)
            
        if device == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.time() - t0

        enhanced_tensor = outputs['enhanced'].clamp(0, 1)
        enhanced = enhanced_tensor[0].cpu().numpy().transpose(1, 2, 0)
        target_np = tgt_tensor[0].cpu().numpy().transpose(1, 2, 0)
        input_np = inp_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        enhanced = np.clip(enhanced, 0, 1)
        target_np = np.clip(target_np, 0, 1)
        
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        Image.fromarray(enhanced_uint8).save(enhanced_dir / img_path.name)

        psnr_val = compute_psnr(enhanced, target_np)
        ssim_val = compute_ssim(enhanced, target_np)
        uiqm_val = compute_uiqm(enhanced)
        uciqe_val = compute_uciqe(enhanced)
        entropy_val = compute_entropy(enhanced)
        
        niqe_val, piqe_val = compute_niqe_piqe(enhanced_tensor, niqe_metric, piqe_metric)
        niqe_inp, piqe_inp = compute_niqe_piqe(inp_tensor, niqe_metric, piqe_metric)
        
        niqe_input_all.append(niqe_inp)
        piqe_input_all.append(piqe_inp)

        result = {
            'name': img_path.name,
            'psnr': float(round(psnr_val, 4)),
            'ssim': float(round(ssim_val, 4)),
            'uiqm': float(round(uiqm_val, 4)),
            'uciqe': float(round(uciqe_val, 4)),
            'entropy': float(round(entropy_val, 4)),
            'niqe': float(round(niqe_val, 4)) if not np.isnan(niqe_val) else 'N/A',
            'piqe': float(round(piqe_val, 4)) if not np.isnan(piqe_val) else 'N/A',
            'niqe_input': float(round(niqe_inp, 4)) if not np.isnan(niqe_inp) else 'N/A',
            'piqe_input': float(round(piqe_inp, 4)) if not np.isnan(piqe_inp) else 'N/A',
            'mean_intensity': float(round(float(np.mean(input_np)), 4)),
        }
        all_results.append(result)

    psnr_values = [r['psnr'] for r in all_results]
    ssim_values = [r['ssim'] for r in all_results]
    uiqm_values = [r['uiqm'] for r in all_results]
    uciqe_values = [r['uciqe'] for r in all_results]
    entropy_values = [r['entropy'] for r in all_results]
    niqe_values = [r['niqe'] for r in all_results if r['niqe'] != 'N/A']
    piqe_values = [r['piqe'] for r in all_results if r['piqe'] != 'N/A']

    avg_time = total_inference_time / len(all_results)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    summary = {
        'num_images': len(all_results),
        'psnr_mean': float(round(np.mean(psnr_values), 2)),
        'psnr_std': float(round(np.std(psnr_values), 2)),
        'ssim_mean': float(round(np.mean(ssim_values), 4)),
        'ssim_std': float(round(np.std(ssim_values), 4)),
        'uiqm_mean': float(round(np.mean(uiqm_values), 2)),
        'uiqm_std': float(round(np.std(uiqm_values), 2)),
        'uciqe_mean': float(round(np.mean(uciqe_values), 4)),
        'uciqe_std': float(round(np.std(uciqe_values), 4)),
        'entropy_mean': float(round(np.mean(entropy_values), 2)),
        'entropy_std': float(round(np.std(entropy_values), 2)),
        'avg_inference_time_ms': float(round(avg_time * 1000, 1)),
        'fps_256': float(round(fps, 1)),
        'total_params': int(total_params),
        'learned_alpha': float(round(alpha_clamped, 4)),
    }

    if niqe_values:
        niqe_input_clean = [v for v in niqe_input_all if not np.isnan(v)]
        summary['niqe_enhanced_mean'] = float(round(np.mean(niqe_values), 4))
        summary['niqe_enhanced_std'] = float(round(np.std(niqe_values), 4))
        if niqe_input_clean:
            summary['niqe_input_mean'] = float(round(np.mean(niqe_input_clean), 4))
            summary['niqe_improvement'] = float(round(np.mean(niqe_input_clean) - np.mean(niqe_values), 4))
        summary['niqe_library'] = 'pyiqa (MATLAB-calibrated)'

    if piqe_values:
        piqe_input_clean = [v for v in piqe_input_all if not np.isnan(v)]
        summary['piqe_enhanced_mean'] = float(round(np.mean(piqe_values), 4))
        summary['piqe_enhanced_std'] = float(round(np.std(piqe_values), 4))
        if piqe_input_clean:
            summary['piqe_input_mean'] = float(round(np.mean(piqe_input_clean), 4))
            summary['piqe_improvement'] = float(round(np.mean(piqe_input_clean) - np.mean(piqe_values), 4))
        summary['piqe_library'] = 'pyiqa (MATLAB-calibrated)'

    summary_path = Path(output_dir) / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = Path(output_dir) / 'per_image_metrics.csv'
    all_results.sort(key=lambda x: x['name'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    failures_path = Path(output_dir) / 'failure_cases.json'
    all_results.sort(key=lambda x: x['psnr'])
    with open(failures_path, 'w') as f:
        json.dump(all_results[:10], f, indent=2)

    failures_img_dir = Path(output_dir) / 'failure_images'
    failures_img_dir.mkdir(parents=True, exist_ok=True)
    
    for r in all_results[:10]:
        img_name = r['name']
        src_path = enhanced_dir / img_name
        dst_path = failures_img_dir / img_name
        if src_path.exists():
            shutil.copy(src_path, dst_path)

    return summary, all_results

def compute_flops(device: str = 'cuda'):
    model = create_model().to(device)
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    try:
        from thop import profile, clever_format
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        return {'macs': float(macs), 'params': float(params), 'gflops': float(macs * 2 / 1e9)}
    except ImportError:
        total_flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                h_out = 256 // max(module.stride[0], 1)
                w_out = 256 // max(module.stride[1], 1)
                flops = 2 * module.in_channels * module.out_channels * \
                        module.kernel_size[0] * module.kernel_size[1] * \
                        h_out * w_out // module.groups
                total_flops += flops
            elif isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
        return {'gflops_estimated': float(total_flops / 1e9)}

def benchmark_inference(model_path: str, device: str = 'cuda', warmup: int = 10, runs: int = 50):
    model = create_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    results = {}
    for h, w in [(256, 256), (512, 512), (1024, 1024)]:
        inp = torch.randn(1, 3, h, w).to(device)
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(inp)

            if device == 'cuda':
                torch.cuda.synchronize()
            times = []
            
            with torch.no_grad():
                for _ in range(runs):
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model(inp)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)

            avg_ms = np.mean(times) * 1000
            fps = 1000 / avg_ms
            mem_mb = 0
            if device == 'cuda':
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats()

            results[f'{h}x{w}'] = {
                'latency_ms': float(round(avg_ms, 1)),
                'fps': float(round(fps, 1)),
                'memory_mb': float(round(mem_mb, 0))
            }
        except RuntimeError as e:
            results[f'{h}x{w}'] = {'error': 'OOM'}

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_flops', action='store_true')
    parser.add_argument('--skip_benchmark', action='store_true')

    args = parser.parse_args()

    summary, results = evaluate(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device
    )

    if not args.skip_flops:
        flops_result = compute_flops(args.device)
        summary_path = Path(args.output_dir) / 'evaluation_summary.json'
        summary.update({k: v for k, v in flops_result.items()})
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    if not args.skip_benchmark:
        bench_result = benchmark_inference(args.model_path, args.device)
        summary_path = Path(args.output_dir) / 'evaluation_summary.json'
        summary['inference_benchmark'] = bench_result
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)