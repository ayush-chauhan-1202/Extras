"""
Coating Microstructure Generative Model
Architecture: StyleGAN2-ADA + Perceptual/Texture losses + Patch Discriminator
Designed for: 200 grayscale microstructure images, RTX 6000 ADA (48GB VRAM)
"""

import os
import argparse
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MicrostructureDataset(Dataset):
    """
    Loads 8-bit RGB TIFs, converts to grayscale, tiles into square patches.
    6:1 aspect ratio images → multiple 512x512 tiles per image with overlap.
    """
    def __init__(self, image_dir, patch_size=512, overlap=0.25, augment=True):
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment = augment

        exts = {'.tif', '.tiff', '.png', '.jpg'}
        self.image_paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        print(f"Found {len(self.image_paths)} images")

        # Extract all patches upfront (store as list of (path, y, x))
        self.patches = self._index_patches()
        print(f"Total patches: {len(self.patches)}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),                # [0,1]
            transforms.Normalize([0.5], [0.5]),   # [-1,1]
        ])

        if augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
            ])
        else:
            self.aug = None

    def _index_patches(self):
        patches = []
        stride = int(self.patch_size * (1 - self.overlap))
        for path in self.image_paths:
            img = Image.open(path).convert('L')
            w, h = img.size
            # Pad if smaller than patch_size
            if w < self.patch_size or h < self.patch_size:
                img = transforms.functional.pad(img, (
                    max(0, (self.patch_size - w) // 2),
                    max(0, (self.patch_size - h) // 2),
                    max(0, (self.patch_size - w + 1) // 2),
                    max(0, (self.patch_size - h + 1) // 2),
                ), padding_mode='reflect')
                w, h = img.size

            ys = list(range(0, h - self.patch_size + 1, stride)) or [0]
            xs = list(range(0, w - self.patch_size + 1, stride)) or [0]
            for y in ys:
                for x in xs:
                    patches.append((str(path), y, x))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        path, y, x = self.patches[idx]
        img = Image.open(path).convert('L')
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.aug:
            patch = self.aug(patch)

        return self.transform(patch)  # shape: [1, H, W]


# ─────────────────────────────────────────────
# StyleGAN2-ADA style Generator
# ─────────────────────────────────────────────

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale,
                       bias=self.bias * self.lr_mul if self.bias is not None else None)
        if self.activation == 'fused_lrelu':
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8, lr_mul=0.01):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(EqualLinear(z_dim if i == 0 else w_dim, w_dim,
                                       activation='fused_lrelu', lr_mul=lr_mul))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, w_dim=512, demodulate=True, upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.modulation = EqualLinear(w_dim, in_ch, bias=True)

    def forward(self, x, w):
        batch = x.shape[0]
        # Modulate
        style = self.modulation(w).view(batch, 1, self.in_ch, 1, 1)
        weight = self.scale * self.weight * style
        # Demodulate
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d
        # Reshape for grouped conv
        x = x.view(1, batch * self.in_ch, *x.shape[2:])
        weight = weight.view(batch * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = F.interpolate(x.view(batch, self.in_ch, *x.shape[2:]),
                              scale_factor=2, mode='bilinear', align_corners=False)
            x = x.view(1, batch * self.in_ch, *x.shape[2:])

        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        return out.view(batch, self.out_ch, *out.shape[2:])


class StyledBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim=512, upsample=True):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, w_dim, upsample=upsample)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, w_dim)
        self.noise_weight1 = nn.Parameter(torch.zeros(1))
        self.noise_weight2 = nn.Parameter(torch.zeros(1))
        self.bias1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        x = self.conv1(x, w)
        if noise is not None:
            x = x + self.noise_weight1 * torch.randn_like(x[:, :1])
        x = self.activate(x + self.bias1)

        x = self.conv2(x, w)
        if noise is not None:
            x = x + self.noise_weight2 * torch.randn_like(x[:, :1])
        x = self.activate(x + self.bias2)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_size=512, out_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim)

        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256, 128: 128, 256: 64, 512: 32
        }
        self.img_size = img_size
        self.log_size = int(math.log2(img_size))
        assert img_size in self.channels, f"img_size {img_size} not supported"

        # Constant input
        self.const = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))

        self.blocks = nn.ModuleList()
        in_ch = self.channels[4]
        for res in [8, 16, 32, 64, 128, 256, 512][:self.log_size - 1]:
            out_ch = self.channels[res]
            self.blocks.append(StyledBlock(in_ch, out_ch, w_dim, upsample=True))
            in_ch = out_ch

        self.to_rgb = ModulatedConv2d(in_ch, out_channels, 1, w_dim, demodulate=False)
        self.rgb_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, z, truncation=1.0, truncation_latent=None):
        w = self.mapping(z)
        if truncation < 1.0 and truncation_latent is not None:
            w = truncation_latent + truncation * (w - truncation_latent)

        x = self.const.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)

        rgb = self.to_rgb(x, w) + self.rgb_bias
        return torch.tanh(rgb)


# ─────────────────────────────────────────────
# Multi-scale Discriminator
# ─────────────────────────────────────────────

class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        y = x.view(g, -1, c, h, w).float()
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0) + 1e-8
        y = y.sqrt().mean([1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, h, w).to(x.dtype)
        return torch.cat([x, y], 1)


class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        if self.downsample:
            y = F.avg_pool2d(y, 2)
            x = F.avg_pool2d(self.skip(x), 2)
        else:
            x = self.skip(x)
        return (y + x) * self.scale


class Discriminator(nn.Module):
    def __init__(self, img_size=512, in_channels=1):
        super().__init__()
        channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256, 128: 128, 256: 64, 512: 32
        }
        log_size = int(math.log2(img_size))

        self.from_rgb = nn.Conv2d(in_channels, channels[img_size], 1)
        self.act = nn.LeakyReLU(0.2)

        self.blocks = nn.ModuleList()
        in_ch = channels[img_size]
        # Downsample from img_size down to 8x8; final_block handles 4x4
        for i in range(log_size - 2, 2, -1):
            out_ch = channels[2 ** i]
            self.blocks.append(DiscBlock(in_ch, out_ch, downsample=True))
            in_ch = out_ch
        # One final downsample block: 8x8 -> 4x4
        self.blocks.append(DiscBlock(in_ch, channels[4], downsample=True))
        in_ch = channels[4]

        self.final_block = nn.Sequential(
            MinibatchStdDev(),
            nn.Conv2d(in_ch + 1, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_ch, in_ch, 4),   # 4x4 -> 1x1
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_ch, 1)
        )

    def forward(self, x):
        x = self.act(self.from_rgb(x))
        for block in self.blocks:
            x = block(x)
        return self.final_block(x)


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """Multi-layer perceptual + Gram matrix (texture) loss via VGG19."""
    def __init__(self, layers=('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        layer_map = {
            'relu1_1': (0, 2), 'relu2_1': (2, 7),
            'relu3_1': (7, 12), 'relu4_1': (12, 21)
        }
        prev = 0
        for name in layers:
            start, end = layer_map[name]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        # x: [B, 1, H, W] in [-1,1] → [B, 3, H, W] normalized for VGG
        x = (x + 1) / 2  # [0,1]
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    @staticmethod
    def gram_matrix(feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
        return gram

    def forward(self, fake, real):
        fake_prep = self.preprocess(fake)
        real_prep = self.preprocess(real)

        perc_loss = 0.0
        style_loss = 0.0
        f_out, r_out = fake_prep, real_prep
        for slice_ in self.slices:
            f_out = slice_(f_out)
            r_out = slice_(r_out)
            perc_loss += F.l1_loss(f_out, r_out.detach())
            style_loss += F.l1_loss(self.gram_matrix(f_out),
                                    self.gram_matrix(r_out.detach()))
        return perc_loss, style_loss


class FFTTextureLoss(nn.Module):
    """Frequency-domain loss to preserve micro-texture patterns."""
    def forward(self, fake, real):
        def fft_mag(x):
            f = torch.fft.fft2(x)
            return torch.abs(torch.fft.fftshift(f))
        return F.l1_loss(fft_mag(fake), fft_mag(real))


def r1_penalty(real_pred, real_img):
    """R1 gradient penalty for discriminator regularization."""
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    return grad_real.pow(2).reshape(real_img.shape[0], -1).sum(1).mean()


# ─────────────────────────────────────────────
# ADA Augmentation
# ─────────────────────────────────────────────

class ADAugment(nn.Module):
    """
    Adaptive Discriminator Augmentation (ADA).
    Dynamically adjusts augmentation probability to prevent D from overfitting on 200 images.
    """
    def __init__(self, target_rt=0.6, initial_p=0.0, ada_interval=4, ada_kimg=100):
        super().__init__()
        self.p = initial_p
        self.target_rt = target_rt
        self.ada_interval = ada_interval
        self.ada_kimg = ada_kimg
        self.rt_accum = 0.0
        self.rt_count = 0

    def update(self, real_pred):
        # rt = fraction of real predictions that D thinks are fake (sign(D) < 0)
        rt = (real_pred.sign() < 0).float().mean().item()
        self.rt_accum += rt
        self.rt_count += 1

    def adjust_p(self, batch_size):
        if self.rt_count > 0:
            rt = self.rt_accum / self.rt_count
            adjustment = batch_size * self.ada_interval / (self.ada_kimg * 1000)
            if rt > self.target_rt:
                self.p = min(1.0, self.p + adjustment)
            else:
                self.p = max(0.0, self.p - adjustment)
            self.rt_accum = 0.0
            self.rt_count = 0

    def augment(self, imgs):
        p = self.p
        if p == 0.0:
            return imgs
        b = imgs.shape[0]
        # Apply augmentations probabilistically
        # Horizontal flip
        mask = torch.rand(b, 1, 1, 1, device=imgs.device) < p
        imgs = torch.where(mask, imgs.flip(-1), imgs)
        # Vertical flip
        mask = torch.rand(b, 1, 1, 1, device=imgs.device) < p
        imgs = torch.where(mask, imgs.flip(-2), imgs)
        # Brightness
        mask = (torch.rand(b, 1, 1, 1, device=imgs.device) < p)
        noise = torch.randn(b, 1, 1, 1, device=imgs.device) * 0.2
        imgs = torch.where(mask, (imgs + noise).clamp(-1, 1), imgs)
        # Contrast
        mask = (torch.rand(b, 1, 1, 1, device=imgs.device) < p)
        factor = (torch.rand(b, 1, 1, 1, device=imgs.device) * 0.4 + 0.8)
        imgs = torch.where(mask, (imgs * factor).clamp(-1, 1), imgs)
        return imgs

    def forward(self, imgs):
        return self.augment(imgs)


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, target_model):
        target_model.load_state_dict(self.shadow)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset
    dataset = MicrostructureDataset(args.data_dir, patch_size=args.patch_size,
                                     overlap=args.overlap, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Models
    G = Generator(z_dim=args.z_dim, w_dim=args.w_dim, img_size=args.patch_size).to(device)
    D = Discriminator(img_size=args.patch_size).to(device)
    G_ema = Generator(z_dim=args.z_dim, w_dim=args.w_dim, img_size=args.patch_size).to(device)
    G_ema.load_state_dict(G.state_dict())
    ema = EMAModel(G, decay=0.999)

    perceptual_loss = VGGPerceptualLoss().to(device)
    fft_loss = FFTTextureLoss().to(device)
    ada = ADAugment(target_rt=0.6, initial_p=0.0)

    # Optimizers (separate LR for mapping network via param groups)
    g_params = [
        {'params': G.mapping.parameters(), 'lr': args.lr * 0.01},
        {'params': list(G.blocks.parameters()) + [G.const] +
                   list(G.to_rgb.parameters()) + [G.rgb_bias], 'lr': args.lr}
    ]
    G_opt = torch.optim.Adam(g_params, betas=(0.0, 0.99), eps=1e-8)
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99), eps=1e-8)

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    # Resume
    start_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'latest.pt')
    if os.path.exists(ckpt_path) and not args.restart:
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        G_ema.load_state_dict(ckpt['G_ema'])
        G_opt.load_state_dict(ckpt['G_opt'])
        D_opt.load_state_dict(ckpt['D_opt'])
        ada.p = ckpt.get('ada_p', 0.0)
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step}, ADA p={ada.p:.3f}")

    # Fixed noise for visualization
    fixed_z = torch.randn(16, args.z_dim, device=device)

    # Training
    step = start_step
    loader_iter = iter(loader)

    # Lazy regularization intervals
    d_reg_every = 16
    g_reg_every = 8
    pl_mean = torch.zeros([], device=device)

    print(f"\nTraining for {args.total_steps} steps...")
    print(f"Dataset: {len(dataset)} patches | Batch: {args.batch_size}")

    while step < args.total_steps:
        try:
            real = next(loader_iter).to(device)
        except StopIteration:
            loader_iter = iter(loader)
            real = next(loader_iter).to(device)

        # ── Discriminator ──
        D_opt.zero_grad()
        real.requires_grad_(True)

        with autocast():
            real_aug = ada(real)
            real_pred = D(real_aug)

            z = torch.randn(args.batch_size, args.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            fake_aug = ada(fake.detach())
            fake_pred = D(fake_aug)

            # Non-saturating loss + R1 lazy regularization
            d_adv = F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()

            d_loss = d_adv

        scaler_D.scale(d_loss).backward()

        # Lazy R1 regularization
        if step % d_reg_every == 0:
            D_opt.zero_grad()
            real_for_r1 = real.detach().requires_grad_(True)
            with autocast():
                real_pred_r1 = D(real_for_r1)
            r1 = r1_penalty(real_pred_r1, real_for_r1)
            r1_loss = r1 * (args.r1_gamma / 2) * d_reg_every
            scaler_D.scale(r1_loss).backward()

        scaler_D.step(D_opt)
        scaler_D.update()

        ada.update(real_pred.detach())
        if step % ada.ada_interval == 0:
            ada.adjust_p(args.batch_size)

        # ── Generator ──
        G_opt.zero_grad()
        with autocast():
            z = torch.randn(args.batch_size, args.z_dim, device=device)
            fake = G(z)
            fake_aug = ada(fake)
            fake_pred = D(fake_aug)
            g_adv = F.softplus(-fake_pred).mean()

            # Perceptual + texture losses (compare fake vs random real batch)
            perc, style = perceptual_loss(fake, real.detach())
            freq = fft_loss(fake, real.detach())

            g_loss = (g_adv +
                      args.lambda_perc * perc +
                      args.lambda_style * style +
                      args.lambda_fft * freq)

        scaler_G.scale(g_loss).backward()
        scaler_G.step(G_opt)
        scaler_G.update()

        ema.update()
        step += 1

        # ── Logging ──
        if step % args.log_every == 0:
            print(f"Step {step:6d} | "
                  f"D={d_adv.item():.3f} | G={g_adv.item():.3f} | "
                  f"Perc={perc.item():.3f} | Style={style.item():.3f} | "
                  f"FFT={freq.item():.3f} | ADA_p={ada.p:.3f}")

        # ── Sample ──
        if step % args.sample_every == 0:
            G_ema.eval()
            ema.apply(G_ema)
            with torch.no_grad():
                samples = G_ema(fixed_z, truncation=0.7)
            grid = vutils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid,
                os.path.join(args.output_dir, f'sample_{step:06d}.png'))
            G_ema.train()

        # ── Checkpoint ──
        if step % args.save_every == 0:
            torch.save({
                'G': G.state_dict(), 'D': D.state_dict(),
                'G_ema': G_ema.state_dict(),
                'G_opt': G_opt.state_dict(), 'D_opt': D_opt.state_dict(),
                'step': step, 'ada_p': ada.p
            }, ckpt_path)
            torch.save({'G_ema': G_ema.state_dict(), 'step': step},
                       os.path.join(args.output_dir, f'ckpt_{step:06d}.pt'))
            print(f"  → Saved checkpoint at step {step}")

    print("Training complete!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Microstructure StyleGAN2-ADA Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./runs/microstructure')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--w_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--r1_gamma', type=float, default=10.0)
    parser.add_argument('--lambda_perc', type=float, default=0.1)
    parser.add_argument('--lambda_style', type=float, default=50.0)
    parser.add_argument('--lambda_fft', type=float, default=0.05)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--restart', action='store_true', help='Ignore existing checkpoints')
    args = parser.parse_args()

    train(args)
