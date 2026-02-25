"""
evaluate.py — Quality metrics for generated microstructures.

Metrics:
  - FID (Fréchet Inception Distance) via clean-fid
  - SSIM (structural similarity with real patches)
  - Power Spectral Density comparison (micro-texture fidelity)
  - Gram matrix distance (style/texture similarity)
  - Perceptual patch diversity score
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from scipy import stats
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def load_images(directory, patch_size=256, n_patches=500):
    """Load random grayscale patches from a directory of TIF/PNG images."""
    paths = list(Path(directory).glob('**/*.tif')) + \
            list(Path(directory).glob('**/*.png')) + \
            list(Path(directory).glob('**/*.tiff'))
    assert paths, f"No images in {directory}"

    patches = []
    transform = T.ToTensor()
    while len(patches) < n_patches:
        path = paths[np.random.randint(len(paths))]
        img = Image.open(path).convert('L')
        w, h = img.size
        if w < patch_size or h < patch_size:
            continue
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        patch = img.crop((x, y, x + patch_size, y + patch_size))
        patches.append(transform(patch))
    return torch.stack(patches)  # [N, 1, H, W]


def compute_psd(patches):
    """Compute mean 1D radially-averaged Power Spectral Density."""
    psds = []
    for p in patches:
        f = np.fft.fft2(p.squeeze().numpy())
        f_shifted = np.fft.fftshift(f)
        psd2d = np.abs(f_shifted) ** 2
        h, w = psd2d.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        radial_sum = np.bincount(R.ravel(), psd2d.ravel())
        radial_count = np.bincount(R.ravel())
        radial_mean = radial_sum / (radial_count + 1e-8)
        psds.append(radial_mean)
    max_len = min(p.shape[0] for p in psds)
    psds = np.stack([p[:max_len] for p in psds])
    return psds.mean(0), psds.std(0)


def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def extract_vgg_features(patches, device, layer=10):
    """Extract VGG19 features for texture comparison."""
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:layer].to(device)
    vgg.eval()
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    grams = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            batch = patches[i:i+bs].to(device)
            batch_rgb = batch.repeat(1, 3, 1, 1)
            batch_rgb = normalize(batch_rgb)
            feat = vgg(batch_rgb)
            grams.append(gram_matrix(feat).cpu())
    return torch.cat(grams)


def compute_ssim_batch(real_patches, fake_patches, n_pairs=200):
    """Compute mean SSIM between random real/fake pairs."""
    scores = []
    for _ in range(n_pairs):
        r = real_patches[np.random.randint(len(real_patches))].squeeze().numpy()
        f = fake_patches[np.random.randint(len(fake_patches))].squeeze().numpy()
        score = ssim(r, f, data_range=1.0)
        scores.append(score)
    return np.mean(scores), np.std(scores)


def try_fid(real_dir, fake_dir):
    """FID via clean-fid if available."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(real_dir, fake_dir, mode='clean', num_workers=4)
        return score
    except ImportError:
        print("  clean-fid not installed. Install with: pip install clean-fid")
        return None


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("Coating Microstructure Evaluation")
    print(f"{'='*50}")
    print(f"Real dir : {args.real_dir}")
    print(f"Fake dir : {args.fake_dir}")
    print(f"Patches  : {args.n_patches}")
    print()

    print("Loading real patches...")
    real = load_images(args.real_dir, patch_size=args.patch_size, n_patches=args.n_patches)
    print("Loading generated patches...")
    fake = load_images(args.fake_dir, patch_size=args.patch_size, n_patches=args.n_patches)

    results = {}

    # FID
    print("\n[1/4] Computing FID...")
    fid_score = try_fid(args.real_dir, args.fake_dir)
    if fid_score is not None:
        results['FID'] = fid_score
        print(f"  FID = {fid_score:.2f}  (lower is better; <50 is good, <20 is excellent)")

    # SSIM
    print("\n[2/4] Computing SSIM...")
    ssim_mean, ssim_std = compute_ssim_batch(real, fake)
    results['SSIM_mean'] = ssim_mean
    results['SSIM_std'] = ssim_std
    print(f"  SSIM = {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"  Note: Lower SSIM = more diverse (>0.1 expected for good diversity)")

    # PSD comparison
    print("\n[3/4] Computing Power Spectral Density...")
    real_psd, real_psd_std = compute_psd(real)
    fake_psd, fake_psd_std = compute_psd(fake)
    psd_corr, _ = stats.pearsonr(np.log1p(real_psd[:100]), np.log1p(fake_psd[:100]))
    results['PSD_correlation'] = psd_corr
    print(f"  PSD log-correlation = {psd_corr:.4f}  (closer to 1.0 = better texture match)")

    # Plot PSD
    fig, ax = plt.subplots(figsize=(8, 4))
    freqs = np.arange(len(real_psd))
    ax.semilogy(freqs[:100], real_psd[:100], 'b-', label='Real', linewidth=2)
    ax.fill_between(freqs[:100],
                    np.maximum(real_psd[:100] - real_psd_std[:100], 1e-8),
                    real_psd[:100] + real_psd_std[:100], alpha=0.2, color='b')
    ax.semilogy(freqs[:100], fake_psd[:100], 'r--', label='Generated', linewidth=2)
    ax.fill_between(freqs[:100],
                    np.maximum(fake_psd[:100] - fake_psd_std[:100], 1e-8),
                    fake_psd[:100] + fake_psd_std[:100], alpha=0.2, color='r')
    ax.set_xlabel('Spatial Frequency (cycles/patch)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Radially-Averaged Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'psd_comparison.png'), dpi=150)
    print(f"  Saved PSD plot → psd_comparison.png")

    # Gram matrix distance (texture)
    print("\n[4/4] Computing Gram matrix texture distance...")
    real_grams = extract_vgg_features(real, device)
    fake_grams = extract_vgg_features(fake, device)
    gram_dist = (real_grams.mean(0) - fake_grams.mean(0)).pow(2).mean().sqrt().item()
    results['Gram_distance'] = gram_dist
    print(f"  Gram distance = {gram_dist:.6f}  (lower is better)")

    # Summary
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:25s}: {v:.4f}")

    # Save results
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Coating Microstructure Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\nResults saved to {args.output_dir}/metrics.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', required=True, help='Directory of real images')
    parser.add_argument('--fake_dir', required=True, help='Directory of generated images')
    parser.add_argument('--output_dir', default='./eval_results')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--n_patches', type=int, default=500)
    args = parser.parse_args()
    main(args)
