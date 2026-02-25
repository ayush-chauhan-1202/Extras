"""
generate.py — Sample images from trained coating microstructure model.
Supports: random generation, interpolation, truncation sweep, tiled wide images.
"""

import argparse
import os
import math
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils

from train import Generator  # reuse architecture


def slerp(t, v0, v1):
    """Spherical linear interpolation between two latent vectors."""
    v0_n = v0 / v0.norm(dim=-1, keepdim=True)
    v1_n = v1 / v1.norm(dim=-1, keepdim=True)
    dot = (v0_n * v1_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
    omega = dot.acos()
    sin_omega = omega.sin()
    # Handle near-parallel case
    safe = sin_omega > 1e-6
    interp = (((1 - t) * omega).sin() / (sin_omega + 1e-8)) * v0 + \
             ((t * omega).sin() / (sin_omega + 1e-8)) * v1
    linear = (1 - t) * v0 + t * v1
    return torch.where(safe, interp, linear)


def load_model(ckpt_path, z_dim=512, w_dim=512, patch_size=512, device='cuda'):
    G = Generator(z_dim=z_dim, w_dim=w_dim, img_size=patch_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('G_ema', ckpt.get('G', ckpt))
    G.load_state_dict(state)
    G.eval()
    print(f"Loaded model from step {ckpt.get('step', 'unknown')}")
    return G


@torch.no_grad()
def generate_random(G, n, truncation, z_dim, device, batch_size=8):
    """Generate n random samples."""
    samples = []
    # Compute truncation latent (mean of 4096 mapping outputs)
    if truncation < 1.0:
        z_avg = torch.randn(4096, z_dim, device=device)
        w_avg = G.mapping(z_avg).mean(0, keepdim=True)
    else:
        w_avg = None

    for i in range(0, n, batch_size):
        bs = min(batch_size, n - i)
        z = torch.randn(bs, z_dim, device=device)
        imgs = G(z, truncation=truncation, truncation_latent=w_avg)
        samples.append(imgs.cpu())
    return torch.cat(samples)


@torch.no_grad()
def generate_wide(G, n_tiles, truncation, z_dim, patch_size, device):
    """
    Generate a wide image matching original 6:1 aspect ratio by tiling patches
    with a single latent (consistent texture across the strip).
    """
    if truncation < 1.0:
        z_avg = torch.randn(4096, z_dim, device=device)
        w_avg = G.mapping(z_avg).mean(0, keepdim=True)
    else:
        w_avg = None

    tiles = []
    for i in range(n_tiles):
        z = torch.randn(1, z_dim, device=device)
        img = G(z, truncation=truncation, truncation_latent=w_avg)
        tiles.append(img)
    strip = torch.cat(tiles, dim=-1)  # Concatenate along width
    return strip


@torch.no_grad()
def generate_interpolation(G, n_frames, truncation, z_dim, device):
    """Generate a latent space interpolation sequence."""
    z0 = torch.randn(1, z_dim, device=device)
    z1 = torch.randn(1, z_dim, device=device)
    ts = torch.linspace(0, 1, n_frames, device=device).unsqueeze(-1)
    zs = slerp(ts, z0.expand(n_frames, -1), z1.expand(n_frames, -1))

    if truncation < 1.0:
        z_avg = torch.randn(4096, z_dim, device=device)
        w_avg = G.mapping(z_avg).mean(0, keepdim=True)
    else:
        w_avg = None

    imgs = []
    for i in range(n_frames):
        img = G(zs[i:i+1], truncation=truncation, truncation_latent=w_avg)
        imgs.append(img.cpu())
    return torch.cat(imgs)


def to_pil(tensor):
    """Convert [-1,1] tensor [1,H,W] to uint8 PIL image."""
    img = (tensor.squeeze(0).squeeze(0).clamp(-1, 1) + 1) / 2
    return Image.fromarray((img.numpy() * 255).astype(np.uint8), mode='L')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    G = load_model(args.ckpt, args.z_dim, args.w_dim, args.patch_size, device)

    if args.mode == 'random':
        print(f"Generating {args.n} random images (truncation={args.truncation})...")
        samples = generate_random(G, args.n, args.truncation, args.z_dim, device)
        for i, s in enumerate(samples):
            to_pil(s).save(os.path.join(args.output_dir, f'gen_{i:04d}.tif'))
        grid = vutils.make_grid(samples, nrow=int(math.sqrt(args.n)),
                                normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, os.path.join(args.output_dir, 'grid.png'))
        print(f"Saved {args.n} images + grid.png")

    elif args.mode == 'wide':
        print(f"Generating wide strip ({args.n_tiles} tiles)...")
        strip = generate_wide(G, args.n_tiles, args.truncation, args.z_dim, args.patch_size, device)
        to_pil(strip).save(os.path.join(args.output_dir, 'wide_strip.tif'))
        print("Saved wide_strip.tif")

    elif args.mode == 'interpolate':
        print(f"Generating {args.n_frames}-frame interpolation...")
        frames = generate_interpolation(G, args.n_frames, args.truncation, args.z_dim, device)
        frame_dir = os.path.join(args.output_dir, 'interpolation')
        os.makedirs(frame_dir, exist_ok=True)
        for i, f in enumerate(frames):
            to_pil(f).save(os.path.join(frame_dir, f'frame_{i:04d}.tif'))
        print(f"Saved {args.n_frames} frames to {frame_dir}")

    elif args.mode == 'truncation_sweep':
        print("Generating truncation sweep (0.3 → 1.0)...")
        z = torch.randn(1, args.z_dim, device=device)
        z_avg_raw = torch.randn(4096, args.z_dim, device=device)
        w_avg = G.mapping(z_avg_raw).mean(0, keepdim=True)
        imgs = []
        for t in np.linspace(0.3, 1.0, 8):
            img = G(z, truncation=float(t), truncation_latent=w_avg)
            imgs.append(img.cpu())
        grid = vutils.make_grid(torch.cat(imgs), nrow=8, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, os.path.join(args.output_dir, 'truncation_sweep.png'))
        print("Saved truncation_sweep.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint (.pt)')
    parser.add_argument('--output_dir', default='./generated')
    parser.add_argument('--mode', choices=['random', 'wide', 'interpolate', 'truncation_sweep'],
                        default='random')
    parser.add_argument('--n', type=int, default=64, help='Number of random images')
    parser.add_argument('--n_tiles', type=int, default=6, help='Tiles for wide mode')
    parser.add_argument('--n_frames', type=int, default=30, help='Frames for interpolation')
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--w_dim', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=512)
    args = parser.parse_args()
    main(args)
