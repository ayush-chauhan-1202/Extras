"""
==============================================================================
  STYLEGAN v2 FOR MEDICAL X-RAY SYNTHESIS  —  IMPROVED EDITION
  Target: Mac Mini M4 (24 GB RAM) | 93 grayscale 8-bit TIFF images
==============================================================================

WHAT IS NEW vs. the vanilla version (stylegan_xray.py):
  ┌─────────────────────────────────────────────────┬────────────────────────┐
  │ Improvement                                     │ Why it matters here    │
  ├─────────────────────────────────────────────────┼────────────────────────┤
  │ 1. Equalized Learning Rate (EqLR)               │ Stable gradient flow   │
  │ 2. Minibatch Standard Deviation (MinibatchStd)  │ Detects mode collapse  │
  │ 3. Adaptive Discriminator Augmentation (ADA)    │ CRITICAL for 93 images │
  │ 4. Path Length Regularisation (PLR)             │ Isotropic W→image map  │
  │ 5. SSIM Perceptual Loss                         │ Medical texture fidelity│
  │ 6. Style Mixing Regularisation                  │ Prevents W-space overfit│
  │ 7. Generator EMA                                │ Smoother inference      │
  │ 8. Truncation Trick (inference)                 │ Quality/diversity tradeoff│
  └─────────────────────────────────────────────────┴────────────────────────┘

  All vanilla components (MappingNetwork, AdaIN, SynthBlock, R1, etc.) are
  retained and upgraded in-place.  Every new piece is documented with:
    WHY    – the conceptual motivation
    HOW    – the mathematical formula
    EFFECT – what you'll see in practice

HARDWARE NOTE:
  MPS (Apple Metal) backend.  Falls back to CPU automatically.
==============================================================================
"""

# ------------------------------------------------------------------------------
# 0.  IMPORTS
# ------------------------------------------------------------------------------
import os, math, time, glob, json, random, copy
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

try:
    from torch_fidelity import calculate_metrics
    HAS_FIDELITY = True
except ImportError:
    HAS_FIDELITY = False
    print("[WARN] torch-fidelity not found. FID skipped.  pip install torch-fidelity")

# ------------------------------------------------------------------------------
# 1.  CONFIGURATION
# ------------------------------------------------------------------------------
CFG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "data_dir"          : "./xray_tifs",
    "image_size"        : 256,

    # ── Latent ────────────────────────────────────────────────────────────────
    "z_dim"             : 512,
    "w_dim"             : 512,

    # ── Architecture ──────────────────────────────────────────────────────────
    "map_layers"        : 8,
    "base_channels"     : 512,
    "max_channels"      : 512,

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size"        : 4,
    "num_epochs"        : 1000,
    "lr_G"              : 1e-4,
    "lr_D"              : 4e-4,
    "betas"             : (0.0, 0.99),

    # ── R1 regularisation ─────────────────────────────────────────────────────
    "r1_gamma"          : 10.0,
    "r1_interval"       : 16,

    # ── [NEW] Path Length Regularisation ──────────────────────────────────────
    "pl_weight"         : 2.0,    # overall scale of PLR loss term
    "pl_interval"       : 4,      # apply PLR every N G-steps (expensive)
    "pl_decay"          : 0.01,   # EMA decay for running mean path length
    # WHY: forces the mapping W -> image to be locally isotropic.
    # Small: PLR barely active.  Large: PLR dominates, G gets stiff.
    # 2.0 matches StyleGAN2 paper defaults.

    # ── [NEW] SSIM Perceptual Loss ─────────────────────────────────────────────
    "ssim_weight"       : 0.1,    # weight of SSIM component in G's total loss
    # WHY: pure adversarial loss only cares about fooling D, not about
    # preserving diagnostic structure (lung borders, vascular markings).
    # SSIM adds a structural similarity term that penalises G for losing
    # macro-texture relative to the nearest real image in the batch.
    # 0.1 keeps it as a soft regulariser rather than dominating.

    # ── [NEW] Adaptive Discriminator Augmentation (ADA) ───────────────────────
    "ada_target"        : 0.6,    # target r_t (D overfitting metric)
    "ada_interval"      : 4,      # update augmentation probability every N D-steps
    "ada_kimg"          : 500,    # speed of adaptation (lower = faster response)
    # r_t = E[sign(D(real))].  When r_t -> 1.0, D is overfitting to real
    # images and we increase augmentation.  We target r_t = 0.6.
    # ada_kimg: 500 means p adjusts by ~1/500k images per step.
    # Lower = more aggressive / responsive to overfitting.

    # ── [NEW] Style Mixing ────────────────────────────────────────────────────
    "style_mix_prob"    : 0.9,    # probability of mixing two w codes per batch
    # WHY: Without mixing, the mapping network can learn to store all
    # information about a sample in a single direction of W, effectively
    # re-entangling features.  By randomly mixing w1 (for coarse blocks)
    # with w2 (for fine blocks) we force the network to make each block
    # truly independent.

    # ── [NEW] Generator EMA ───────────────────────────────────────────────────
    "ema_beta"          : 0.999,  # EMA decay for generator weight averaging
    # WHY: The training generator's weights oscillate due to the adversarial
    # dynamics.  A slow exponential moving average smooths this oscillation
    # and typically produces better and more stable generated images.
    # All sample grids and FID are computed from the EMA generator.

    # ── Logging / saving ──────────────────────────────────────────────────────
    "sample_interval"   : 50,
    "ckpt_interval"     : 100,
    "fid_interval"      : 100,
    "n_fid_samples"     : 93,
    "out_dir"           : "./stylegan_v2_output",
    "seed"              : 42,
}

# ------------------------------------------------------------------------------
# 2.  DEVICE
# ------------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("[INFO] Apple MPS (Metal) backend.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[INFO] CUDA GPU.")
        return torch.device("cuda")
    print("[WARN] CPU only – will be slow.")
    return torch.device("cpu")

# ------------------------------------------------------------------------------
# 3.  DATASET  (unchanged from v1, with augmentation disabled at dataset level
#               because ADA now handles augmentation inside the training loop)
# ------------------------------------------------------------------------------
class XRayDataset(Dataset):
    """
    Loads 8-bit grayscale TIFFs -> [-1, 1] tensors.

    NOTE on augmentation:
      In v1 we applied RandomHorizontalFlip in the Dataset.
      In v2 augmentation is applied to *discriminator* inputs only via ADA
      (see AdaptiveAugment below).  The Dataset still applies the flip as
      a cheap free doubling of effective dataset size before ADA kicks in.
    """
    def __init__(self, data_dir, image_size, augment=True):
        self.paths = sorted(
            glob.glob(os.path.join(data_dir, "*.tif")) +
            glob.glob(os.path.join(data_dir, "*.tiff")))
        if not self.paths:
            raise FileNotFoundError(
                f"No .tif/.tiff files in '{data_dir}'.  "
                f"Set CFG['data_dir'] to your image folder.")
        print(f"[DATA] {len(self.paths)} images in '{data_dir}'.")

        tfm = [T.Resize((image_size, image_size),
                        interpolation=T.InterpolationMode.BILINEAR)]
        if augment:
            tfm.append(T.RandomHorizontalFlip(p=0.5))
        tfm += [T.ToTensor(), T.Normalize([0.5], [0.5])]
        self.transform = T.Compose(tfm)

    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("L"))


# ==============================================================================
# 4.  [NEW] EQUALIZED LEARNING RATE
# ==============================================================================
class EqLinear(nn.Module):
    """
    Linear layer with Equalized Learning Rate (EqLR).

    WHY EqLR?
      Standard He-initialisation sets weights to scale proportional to
      1/sqrt(fan_in) AT INIT TIME.  But during optimisation, Adam's adaptive
      step size partially cancels this — layers with small fan-in get larger
      effective updates than layers with large fan-in because Adam normalises
      by gradient magnitude, not by weight scale.

      EqLR fixes this by:
        1. Always initialising weights as N(0, 1)  (no pre-scaling at init)
        2. Multiplying by scale = 1/sqrt(fan_in) AT FORWARD TIME

      Because Adam sees raw N(0,1) weights, all layers have the same gradient
      statistics and therefore the same effective learning rate.  The scale
      correction happens at runtime, so the actual output is identical to He-init
      but with much better optimisation dynamics.

    HOW:
      w_eff = w_stored * scale          (scale = 1/sqrt(fan_in))
      y     = x @ w_eff.T + b_stored   (bias is also scaled)
    """
    def __init__(self, in_features, out_features, bias=True,
                 lr_multiplier=1.0, bias_init=0.0):
        super().__init__()
        self.scale        = lr_multiplier / math.sqrt(in_features)
        self.bias_scale   = lr_multiplier
        # THE CRITICAL FIX: divide stored weight by lr_multiplier at init.
        #
        # At forward time:  w_eff = (randn/lr_mult) * (lr_mult/sqrt(fan_in))
        #                         = randn/sqrt(fan_in)   <- correct He-init
        # regardless of lr_multiplier value.
        #
        # Adam's gradient on stored weight is lr_mult * dL/dw_eff, so the
        # effective update on w_eff is lr_mult times smaller -- exactly what
        # we want for the mapping network (lr_mult=0.01 = 100x slower updates).
        #
        # OLD BUG: weight = randn  (not divided). With lr_mult=0.01:
        #   w_eff = randn * 0.01/sqrt(512) ~ N(0, 4e-8) -- 100x too small.
        #   After 8 mapping layers: w ~ 0.01^8 ~ 1e-16 → AdaIN degenerates.
        self.weight       = nn.Parameter(
            torch.randn(out_features, in_features) / lr_multiplier)
        self.bias         = nn.Parameter(
            torch.full([out_features], bias_init / lr_multiplier)) if bias else None

    def forward(self, x):
        w = self.weight * self.scale
        b = self.bias * self.bias_scale if self.bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self):
        return (f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
                f"scale={self.scale:.4f}")


class EqConv2d(nn.Module):
    """
    Conv2d with Equalized Learning Rate.

    Same idea as EqLinear: store weights as N(0,1), multiply by
    scale = 1/sqrt(fan_in) = 1/sqrt(in_channels * kH * kW) at forward time.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 bias=True, lr_multiplier=1.0):
        super().__init__()
        fan_in          = in_ch * kernel_size * kernel_size
        self.scale      = lr_multiplier / math.sqrt(fan_in)
        self.stride     = stride
        self.padding    = padding
        # Same fix as EqLinear: store randn/lr_multiplier so effective
        # weight at forward time = randn/sqrt(fan_in) (He-init) always.
        self.weight     = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size) / lr_multiplier)
        self.bias       = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale,
                        self.bias, self.stride, self.padding)


# ==============================================================================
# 5.  BUILDING BLOCKS  (upgraded to use EqLR convolutions)
# ==============================================================================

class PixelNorm(nn.Module):
    """Normalise each pixel's feature vector to unit length (unchanged)."""
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation (unchanged logic, upgraded to EqLinear).

    WHY EqLinear here:
      The style projections (W -> scale, W -> bias) are the most critical
      linear layers in the generator.  EqLR ensures that both the shallow
      (coarse) and deep (fine) AdaIN projections learn at the same rate,
      preventing the fine-detail AdaIN from dominating or lagging.
    """
    def __init__(self, w_dim, n_channels):
        super().__init__()
        # EqLinear replaces nn.Linear from v1
        self.style_scale = EqLinear(w_dim, n_channels, bias_init=1.0)
        self.style_bias  = EqLinear(w_dim, n_channels, bias_init=0.0)

    def forward(self, x, w):
        B, C, H, W = x.shape
        mean   = x.mean(dim=[2,3], keepdim=True)
        std    = x.std (dim=[2,3], keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        y_s = self.style_scale(w).view(B, C, 1, 1)
        y_b = self.style_bias (w).view(B, C, 1, 1)
        return y_s * x_norm + y_b


class SynthBlock(nn.Module):
    """
    One synthesis resolution stage (upgraded to EqConv2d).

    [NEW] Stochastic noise injection:
      Before each AdaIN we add spatially-uncorrelated Gaussian noise,
      scaled by a learned per-channel scalar B[c].

      WHY:
        AdaIN removes all spatial variation (it normalises each feature
        map to zero mean / unit variance).  The noise injection re-introduces
        fine stochastic detail AFTER normalisation.  This is what produces
        the micro-texture (tiny intensity variations, film grain analogue)
        in X-ray synthesis.  Without it, each generated image would be
        slightly "too smooth" — every sample from the same w would look
        identical at the pixel level.

      HOW:
        noise  ~ N(0, 1)  shape [B, 1, H, W]  (broadcast over channels)
        B_c    = learned scalar per channel
        output = conv(x) + B_c * noise
    """
    def __init__(self, in_ch, out_ch, w_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1    = EqConv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2    = EqConv2d(out_ch, out_ch, 3, padding=1)
        self.adain1   = AdaIN(w_dim, out_ch)
        self.adain2   = AdaIN(w_dim, out_ch)
        self.act      = nn.LeakyReLU(0.2, inplace=True)
        # Learned noise scalers: one per channel, initialised to 0
        # (so noise starts off having no effect and is gradually learned)
        self.noise_scale1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
        # Conv + noise + AdaIN (pass 1)
        x = self.conv1(x)
        noise1 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3],
                             device=x.device)
        x = self.act(x + self.noise_scale1 * noise1)
        x = self.adain1(x, w)
        # Conv + noise + AdaIN (pass 2)
        x = self.conv2(x)
        noise2 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3],
                             device=x.device)
        x = self.act(x + self.noise_scale2 * noise2)
        x = self.adain2(x, w)
        return x


# ==============================================================================
# 6.  GENERATOR
# ==============================================================================
class MappingNetwork(nn.Module):
    """
    z -> w  (unchanged logic, upgraded to EqLinear with lr_multiplier=0.01).

    WHY lr_multiplier=0.01 for mapping network?
      The mapping network is 8 layers deep and sits upstream of everything.
      If it updates too fast, w changes rapidly, and every subsequent block
      has to chase it.  Reducing its effective LR by 100× makes w evolve
      slowly and gives the synthesis network time to adapt.
      This is a standard StyleGAN2 trick.
    """
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = [PixelNorm()]
        in_dim = z_dim
        for _ in range(num_layers):
            layers += [EqLinear(in_dim, w_dim, lr_multiplier=0.01),
                       nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Generator(nn.Module):
    """
    Full StyleGAN generator.

    [NEW] Style Mixing:
      Instead of mapping a single z -> w and injecting w into every block,
      we sometimes map two different z vectors (z1, z2) -> (w1, w2) and
      inject w1 into the first `crossover` blocks and w2 into the rest.

      WHY:
        Forces each resolution block to be truly independent — coarse blocks
        cannot leak fine-block information through a shared w.  This
        regularises the mapping network and improves disentanglement.

      HOW:
        crossover ~ Uniform(1, num_blocks)
        blocks 0 .. crossover-1  receive w1
        blocks crossover .. end  receive w2

    [NEW] Truncation trick (inference only):
      w_trunc = w_mean + psi * (w - w_mean)
      psi < 1 pulls w toward the centre of W-space, trading diversity
      for quality.  psi = 1 = no truncation.
    """
    def __init__(self, z_dim=512, w_dim=512, image_size=256,
                 base_channels=512, max_channels=512, map_layers=8):
        super().__init__()
        self.z_dim     = z_dim
        self.w_dim     = w_dim
        self.mapping   = MappingNetwork(z_dim, w_dim, map_layers)
        self.log2_size = int(math.log2(image_size))
        num_blocks     = self.log2_size - 1   # 7 for 256×256

        def ch(r): return min(base_channels >> max(r - 2, 0), max_channels)

        self.const  = nn.Parameter(torch.randn(1, ch(2), 4, 4))
        self.blocks = nn.ModuleList()
        in_ch = ch(2)
        for i in range(num_blocks):
            out_ch = ch(i + 3)
            self.blocks.append(SynthBlock(in_ch, out_ch, w_dim, upsample=(i>0)))
            in_ch = out_ch

        self.to_rgb = EqConv2d(in_ch, 1, 1)
        self.tanh   = nn.Tanh()

        # Running EMA of mean w (used for truncation at inference)
        self.register_buffer("w_mean", torch.zeros(w_dim))
        self.w_mean_initialized = False

    def get_w(self, z):
        """Map z to w and update running w_mean EMA."""
        w = self.mapping(z)
        # Update running mean with EMA (used at inference for truncation)
        with torch.no_grad():
            self.w_mean.lerp_(w.detach().mean(0), 0.001)
        return w

    def synthesize(self, ws):
        """
        ws: either [B, w_dim] for single style
            or     [B, num_blocks, w_dim] for per-block style (mixing)
        """
        B = ws.shape[0]
        x = self.const.expand(B, -1, -1, -1)
        for i, block in enumerate(self.blocks):
            # Select per-block w
            w_i = ws[:, i, :] if ws.dim() == 3 else ws
            x   = block(x, w_i)
        return self.tanh(self.to_rgb(x))

    def forward(self, z, mix_prob=0.0, truncation_psi=1.0):
        """
        Args:
            z            : [B, z_dim]  noise
            mix_prob     : probability of applying style mixing this forward pass
            truncation_psi: 1.0 = no truncation (training), <1 for inference
        """
        num_blocks = len(self.blocks)
        w = self.get_w(z)                                  # [B, w_dim]

        if truncation_psi < 1.0:
            w = self.w_mean.lerp(w, truncation_psi)        # pull toward mean

        # Style mixing: sample a second z and cross-inject at a random block
        if mix_prob > 0.0 and random.random() < mix_prob:
            z2  = torch.randn_like(z)
            w2  = self.get_w(z2)
            crossover = random.randint(1, num_blocks - 1)
            # Build per-block w tensor [B, num_blocks, w_dim]
            ws  = w.unsqueeze(1).expand(-1, num_blocks, -1).clone()
            ws[:, crossover:, :] = w2.unsqueeze(1).expand(-1, num_blocks - crossover, -1)
            return self.synthesize(ws)

        return self.synthesize(w)


# ==============================================================================
# 7.  DISCRIMINATOR  (upgraded + MinibatchStd)
# ==============================================================================

class DiscBlock(nn.Module):
    """Downsampling discriminator block (upgraded to EqConv2d)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            EqConv2d(in_ch,  out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EqConv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )
    def forward(self, x): return self.net(x)


class MinibatchStdDev(nn.Module):
    """
    [NEW] Minibatch Standard Deviation layer.

    WHY:
      Mode collapse happens when G generates only a small variety of images
      (e.g., always the same lung pattern).  The discriminator, seeing the
      same image repeated in a batch, cannot detect this because it evaluates
      each image *independently*.

      MinibatchStd gives D a signal about *diversity within the batch*.
      If all samples in a batch look the same (mode collapse), the std is
      near zero — an obvious tell for D.  If diversity is healthy, std is
      larger.  D learns to use this signal, which forces G to generate
      diverse outputs or be caught.

    HOW:
      1. Take feature maps x of shape [B, C, H, W]
      2. Compute std over the batch dimension for every (C, H, W) position
      3. Average that std over all channels and spatial positions -> scalar per sample
      4. Tile the scalar into a [B, 1, H, W] feature map
      5. Concatenate to x -> [B, C+1, H, W]

      The group size g controls how many samples are compared together.
      g=4 means we compare within groups of 4 (matches batch_size here).

    Args:
        group_size: number of samples to compare; clip to batch size at runtime
        n_channels: number of output std channels (usually 1)
    """
    def __init__(self, group_size=4, n_channels=1):
        super().__init__()
        self.group_size = group_size
        self.n_channels = n_channels

    def forward(self, x):
        B, C, H, W = x.shape
        G  = min(self.group_size, B)     # can't exceed batch size
        F_ = self.n_channels
        c  = C // F_                     # channels per std feature

        # Step 1: group the batch and compute std across the group
        #   x        : [B, C, H, W]
        #   reshaped : [G, B//G, F_, c, H, W]  (6-D)
        y = x.reshape(G, -1, F_, c, H, W).float()

        # Step 2: variance over the group axis (dim=0), then sqrt -> std
        #   result: [B//G, F_, c, H, W]  (5-D)
        y = y.var(dim=0, unbiased=False).sqrt()

        # Step 3: average over the channel-split and spatial dims
        #   mean(dim=[2,3,4]) -> [B//G, F_]  (2-D, no keepdim)
        #   then view to     -> [B//G, F_, 1, 1]  (4-D, ready for expand)
        y = y.mean(dim=[2, 3, 4])              # [B//G, F_]
        y = y.view(-1, F_, 1, 1)              # [B//G, F_, 1, 1]

        # Step 4: tile back to full batch size and expand over spatial dims
        #   repeat(G, 1, 1, 1) -> [B, F_, 1, 1]
        #   expand             -> [B, F_, H, W]  (4-D, matches x)
        y = y.repeat(G, 1, 1, 1)              # [B, F_, 1, 1]
        y = y.expand(-1, -1, H, W)            # [B, F_, H, W]
        y = y.to(x.dtype)

        return torch.cat([x, y], dim=1)        # [B, C+F_, H, W]


class Discriminator(nn.Module):
    """
    Discriminator with MinibatchStd and EqConv2d throughout.

    The MinibatchStd layer is inserted just before the 4×4 final head,
    which is where it has maximum receptive field and therefore the most
    informative batch diversity signal.
    """
    def __init__(self, image_size=256, base_channels=512, max_channels=512):
        super().__init__()
        log2_size = int(math.log2(image_size))

        def ch(r): return min(base_channels >> max(r - 2, 0), max_channels)

        self.from_rgb = nn.Sequential(
            EqConv2d(1, ch(log2_size), 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        in_ch  = ch(log2_size)
        for i in range(log2_size - 2, 0, -1):
            out_ch = ch(i)
            blocks.append(DiscBlock(in_ch, out_ch))
            in_ch  = out_ch
        self.blocks = nn.Sequential(*blocks)

        # MinibatchStd adds 1 extra channel before the final head
        self.mbstd = MinibatchStdDev(group_size=4)
        extra      = 1   # one extra std channel from MinibatchStd

        self.head = nn.Sequential(
            EqConv2d(in_ch + extra, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            EqLinear(in_ch * 4 * 4, in_ch),
            nn.LeakyReLU(0.2, inplace=True),
            EqLinear(in_ch, 1),
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.mbstd(x)       # inject batch diversity signal
        return self.head(x)


# ==============================================================================
# 8.  [NEW] ADAPTIVE DISCRIMINATOR AUGMENTATION (ADA)
# ==============================================================================
class AdaptiveAugment:
    """
    ADA — Adaptive Discriminator Augmentation (Karras et al. 2020b).

    THE CORE PROBLEM FOR 93 IMAGES:
      With only 93 training images, D can memorise the entire dataset in
      a few hundred epochs.  Once memorised, D(real) = +inf for all 93 images
      and D(fake) = -inf for everything else.  G gets no useful gradient.

      Standard data augmentation applied to the training data would leak
      into G (G learns to generate augmented/distorted images).

    THE ADA SOLUTION:
      Augment images ONLY when they are passed to D, not when they are
      passed to G or when computing metrics.  This way:
        - D sees augmented images and cannot memorise individual real images
        - G is still trained to generate clean, non-augmented images
        - The FID metric is computed on clean generated images vs clean real images

    THE ADAPTIVE PART:
      We don't know the right augmentation probability p in advance.
      ADA measures the overfitting signal r_t = E[sign(D(real))]:
        - r_t close to +1.0: D assigns positive logit to ALL real images
          -> D is overfitting -> increase p (more augmentation)
        - r_t close to 0.0 : D is uncertain about real images
          -> D is appropriately challenged -> decrease p (less augmentation)
        - We target r_t = 0.6  (D slightly preferring real, but not certain)

      p is updated every `ada_interval` D-steps:
        sign = +1 if current_r_t > target else -1
        p += sign * batch_size / (ada_kimg * 1000)

    AUGMENTATION OPERATIONS (ordered from least to most destructive):
      All operations are applied stochastically with probability p.
      We use operations safe for medical images:
        - Horizontal flip    (p/4)  [anatomical symmetry approx holds]
        - Integer translation (p/2) [sub-image-size shifts]
        - Brightness jitter   (p/2)
        - Contrast jitter     (p/2)
        - Gaussian noise      (p/4)  [mimics X-ray quantum noise]
      We do NOT use: rotation, elastic deformation, colour jitter
      (these would alter anatomically meaningful spatial relationships)
    """
    def __init__(self, target_rt=0.6, interval=4, kimg=500, batch_size=4):
        self.target_rt   = target_rt
        self.interval    = interval
        self.kimg        = kimg
        self.batch_size  = batch_size
        self.p           = 0.0        # current augmentation probability
        self._step       = 0
        self._rt_accum   = 0.0
        self._rt_n       = 0

    def augment(self, imgs):
        """
        Apply stochastic augmentations to a batch of images [B, 1, H, W].
        Each image independently decides whether to apply each augment.
        """
        if self.p <= 0:
            return imgs

        B = imgs.shape[0]
        out = imgs.clone()

        # ── Horizontal flip ───────────────────────────────────────────────
        mask = (torch.rand(B, 1, 1, 1, device=imgs.device) < self.p * 0.5)
        out  = torch.where(mask.expand_as(out), out.flip(-1), out)

        # ── Integer translation (up to 1/8 of image size) ────────────────
        if random.random() < self.p:
            H, W  = imgs.shape[2], imgs.shape[3]
            tx    = random.randint(-W//8, W//8)
            ty    = random.randint(-H//8, H//8)
            out   = torch.roll(out, shifts=(ty, tx), dims=(2, 3))

        # ── Brightness jitter ─────────────────────────────────────────────
        if random.random() < self.p:
            delta = (torch.rand(B, 1, 1, 1, device=imgs.device) - 0.5) * 0.4
            out   = (out + delta).clamp(-1, 1)

        # ── Contrast jitter ───────────────────────────────────────────────
        if random.random() < self.p:
            factor = torch.rand(B, 1, 1, 1, device=imgs.device) * 1.5 + 0.5
            mean   = out.mean(dim=[2,3], keepdim=True)
            out    = ((out - mean) * factor + mean).clamp(-1, 1)

        # ── Gaussian noise (quantum noise analogue) ───────────────────────
        if random.random() < self.p * 0.5:
            sigma = random.uniform(0.0, 0.1)
            out   = (out + torch.randn_like(out) * sigma).clamp(-1, 1)

        return out

    def accumulate_rt(self, real_logits):
        """
        Accumulate r_t = mean(sign(D(real))) over a window.
        sign(D(real)) = +1 means D thinks the real image is real (overfitting).
        """
        with torch.no_grad():
            self._rt_accum += real_logits.sign().mean().item()
            self._rt_n     += 1

    def update_p(self):
        """Adjust p up/down based on accumulated r_t vs target."""
        self._step += 1
        if self._step % self.interval != 0:
            return
        if self._rt_n == 0:
            return
        rt = self._rt_accum / self._rt_n
        self._rt_accum = 0.0
        self._rt_n     = 0
        # Adjustment: positive if overfitting (rt > target), negative otherwise
        adjust = self.batch_size / (self.kimg * 1000)
        if rt > self.target_rt:
            self.p = min(self.p + adjust, 1.0)
        else:
            self.p = max(self.p - adjust, 0.0)

    @property
    def augment_p(self):
        return self.p


# ==============================================================================
# 9.  LOSS FUNCTIONS
# ==============================================================================

def d_loss_fn(D, real_aug, fake_aug):
    """Non-saturating discriminator loss on ADA-augmented images."""
    bce       = nn.BCEWithLogitsLoss()
    r_logit   = D(real_aug)
    f_logit   = D(fake_aug)
    loss_real = bce(r_logit, torch.ones_like (r_logit))
    loss_fake = bce(f_logit, torch.zeros_like(f_logit))
    return loss_real + loss_fake, r_logit   # return real logits for r_t tracking


def g_loss_fn(D, fake_aug):
    """Non-saturating generator loss."""
    bce = nn.BCEWithLogitsLoss()
    return bce(D(fake_aug), torch.ones_like(D(fake_aug)))


def r1_penalty(D, real_aug, device):
    """
    R1 gradient penalty  ||grad_x D(x)||^2  evaluated on real images.

    TWO IMPLEMENTATIONS depending on device:

    CUDA/CPU (exact):
      Uses autograd.grad with create_graph=True so the penalty's gradient
      flows back through the gradient computation itself to D's parameters.
      This is mathematically exact.

    MPS (finite-difference approximation):
      Apple MPS does not reliably support create_graph=True in autograd.grad
      as of PyTorch ≤ 2.3 — it silently returns NaN gradients instead of
      raising an error, which corrupts all model weights within a few batches.

      Instead we use a finite-difference approximation:
        ||grad_x D(x)||^2  ≈  E_n[ (D(x + ε·n) - D(x))^2 ] / ε^2
      where n ~ N(0, I).  This is because:
        E[ (n · grad_x D(x))^2 ] = ||grad_x D(x)||^2   (for n ~ N(0,I))
      The approximation uses only first-order autograd and is fully MPS-safe.
      Gradient flows back to D's parameters through D(x + ε·n).

      Limitation: finite-diff R1 is slightly noisier than exact R1 but still
      provides the key stabilising effect (smooth D around real images).
    """
    if device.type == 'mps':
        # ── MPS path: finite-difference R1 ───────────────────────────────────
        eps   = 1e-3
        noise = torch.randn_like(real_aug) * eps
        with torch.no_grad():
            d_real = D(real_aug.detach())
        d_pert = D((real_aug.detach() + noise))
        # (D(x+εn) - D(x)) / ε  ≈  n · grad_x D(x).  Square and mean.
        return ((d_pert - d_real) / eps).pow(2).mean()
    else:
        # ── CUDA/CPU path: exact R1 ───────────────────────────────────────────
        real_aug = real_aug.detach().requires_grad_(True)
        logit    = D(real_aug)
        grad     = torch.autograd.grad(
            outputs=logit.sum(), inputs=real_aug, create_graph=True)[0]
        return grad.pow(2).reshape(real_aug.size(0), -1).sum(1).mean()


# ==============================================================================
# 10.  [NEW] PATH LENGTH REGULARISATION
# ==============================================================================
def path_length_regularise(G, z, pl_mean, pl_decay, device):
    """
    Path Length Regularisation (Karras et al. 2019).

    Two implementations — same reason as r1_penalty:

    CUDA/CPU (exact):
      Computes the Jacobian-vector product  J^T * n  via autograd.grad
      with create_graph=True.  Exact and efficient.

    MPS (finite-difference approximation):
      create_graph=True is broken on MPS (silent NaN).
      We instead approximate the path length via a perturbation in W-space:

        pl_length ≈ ||G(w + ε·δ) - G(w)|| / ε
      where δ ~ N(0, I/sqrt(w_dim)) is a random unit-ish direction in W.

      This estimates how far a unit step in W moves the image — exactly the
      path length concept — using only first-order autograd.
      Gradient flows to G through G(w) in the loss term.

    NaN guard:
      If pl_mean has been corrupted to NaN (e.g. from a previous bad batch),
      reset it to 0.0 before computing to prevent NaN from propagating forever.
    """
    # Guard: reset pl_mean if it has become NaN
    if torch.isnan(pl_mean):
        pl_mean = torch.zeros_like(pl_mean)

    if device.type == 'mps':
        # ── MPS path: finite-difference PLR ──────────────────────────────────
        w   = G.get_w(z)                               # [B, w_dim]  detach-safe
        img1 = G.synthesize(w)                         # [B, 1, H, W]
        _, _, H, W = img1.shape

        eps   = 0.05
        delta = torch.randn_like(w) / math.sqrt(w.shape[1])  # unit-ish direction
        # Second image: synthesize from slightly perturbed w (no grad on w here)
        with torch.no_grad():
            img2 = G.synthesize(w.detach() + eps * delta)

        # Path length per sample: image-space L2 distance / step size
        pl_length = (img1.detach() - img2).reshape(z.shape[0], -1).norm(dim=1) / eps

        pl_mean_new = pl_mean + pl_decay * (pl_length.mean() - pl_mean)

        # Differentiable loss: penalise img1 for deviating from mean path length
        diff = (img1 - img2.detach()).reshape(z.shape[0], -1).norm(dim=1) / eps
        pl_loss = (diff - pl_mean_new.detach()).pow(2).mean()

        return pl_loss, pl_mean_new.detach().clamp(min=0)

    else:
        # ── CUDA/CPU path: exact PLR ──────────────────────────────────────────
        w    = G.get_w(z)
        img  = G.synthesize(w)
        _, _, H, W = img.shape

        noise = torch.randn_like(img) / math.sqrt(H * W)
        grad  = torch.autograd.grad(
            outputs=(img * noise).sum(),
            inputs=w,
            create_graph=True)[0]

        pl_length   = grad.pow(2).sum(dim=1).sqrt()
        pl_mean_new = pl_mean + pl_decay * (pl_length.mean().detach() - pl_mean)
        pl_loss     = (pl_length - pl_mean_new).pow(2).mean()

        return pl_loss, pl_mean_new.detach()


# ==============================================================================
# 11.  [NEW] SSIM PERCEPTUAL LOSS
# ==============================================================================
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) as a differentiable loss.

    WHY FOR MEDICAL IMAGES:
      Pixel-wise MSE treats all pixel errors equally.  A 10-pixel shift of
      a sharp lung border gets the same penalty as a uniform brightness
      change — but clinically the border location matters far more.

      SSIM measures three things that human visual perception cares about:
        1. Luminance  : mean intensity comparison    → μ_x vs μ_y
        2. Contrast   : std intensity comparison     → σ_x vs σ_y
        3. Structure  : normalised cross-correlation → σ_xy / (σ_x σ_y)

      Formula:
        SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) /
                     [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]

      SSIM is in [−1, 1]; 1 = identical.  Loss = 1 − SSIM.

    HOW WE USE IT IN G's TRAINING:
      For each fake image in a batch, we find the *nearest real image*
      (by pixel MSE), then compute SSIM loss between the fake and that
      real image.  This encourages G to produce images that are
      structurally similar to at least one real X-ray pattern.

    NOTE: We use a small patch window (11×11) to compute local SSIM
    rather than global, which preserves sensitivity to local texture.

    Args:
        window_size: convolution kernel size (11 is standard)
        sigma      : Gaussian kernel sigma
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        # Build Gaussian kernel
        k = torch.arange(window_size).float() - window_size // 2
        kernel_1d = torch.exp(-k**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        # Shape: [1, 1, W, W] for conv2d with groups=1
        self.register_buffer("kernel",
            kernel_2d.unsqueeze(0).unsqueeze(0))

    def _mu_sigma(self, x):
        pad = self.window_size // 2
        mu  = F.conv2d(x, self.kernel, padding=pad)
        mu2 = mu * mu
        sig = F.conv2d(x * x, self.kernel, padding=pad) - mu2
        return mu, sig.clamp(min=0)

    def ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        mu_x, sig_x = self._mu_sigma(x)
        mu_y, sig_y = self._mu_sigma(y)
        sig_xy = F.conv2d(x * y, self.kernel,
                          padding=self.window_size//2) - mu_x * mu_y
        num  = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
        den  = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
        return (num / den.clamp(min=1e-8)).mean()

    def forward(self, fake, real_batch):
        """
        For each fake image, find the nearest real image (by MSE) and
        compute 1 - SSIM as the structural loss.

        FIX: mse.argmin() returns a 0-dim device tensor.  Using it directly
        in Python slice notation (real_batch[tensor : tensor+1]) is undefined
        on MPS — the tensor's __index__ dunder may not be implemented.
        Always call .item() to get a plain Python int before slicing.
        """
        loss = 0.0
        for i in range(fake.shape[0]):
            f    = fake[i:i+1]                                    # [1, 1, H, W]
            mse  = ((real_batch - f.detach()) ** 2).mean(dim=[1, 2, 3])
            idx  = int(mse.argmin().item())                       # plain Python int
            best = real_batch[idx : idx + 1]                      # [1, 1, H, W]
            loss = loss + (1.0 - self.ssim(f, best.detach()))
        return loss / fake.shape[0]


# ==============================================================================
# 12.  GENERATOR EMA
# ==============================================================================
class GeneratorEMA:
    """
    Exponential Moving Average of the generator weights.

    WHY:
      During adversarial training, G's weights oscillate.  The EMA acts as
      a low-pass filter on the weight trajectory:
        w_ema = beta * w_ema + (1 - beta) * w_current

      With beta=0.999, the EMA has a time constant of ~1000 steps.
      Generated images from the EMA generator are smoother and typically
      score better on FID than images from the training generator.

    HOW TO USE:
      - Call ema.update(G) after every G optimiser step
      - Use ema.G_ema for sampling / visualisation / FID computation
      - Only G (not D) gets an EMA
    """
    def __init__(self, G, beta=0.999):
        self.beta  = beta
        self.G_ema = copy.deepcopy(G)
        self.G_ema.eval()

    @torch.no_grad()
    def update(self, G):
        for p_ema, p in zip(self.G_ema.parameters(), G.parameters()):
            p_ema.copy_(p_ema.lerp(p.data, 1 - self.beta))
        for b_ema, b in zip(self.G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)


# ==============================================================================
# 13.  METRICS
# ==============================================================================

def save_real_images_for_fid(dataset, out_dir):
    real_dir = os.path.join(out_dir, "fid_real")
    os.makedirs(real_dir, exist_ok=True)
    if len(os.listdir(real_dir)) == len(dataset):
        return real_dir   # already saved
    for i, img in enumerate(dataset):
        arr = ((img.squeeze().numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(arr, "L").save(os.path.join(real_dir, f"real_{i:04d}.png"))
    return real_dir


def compute_fid(G_ema, real_dir, out_dir, n_samples, z_dim, device):
    """FID computed on EMA generator — more stable than training generator."""
    if not HAS_FIDELITY:
        return None

    fake_dir = os.path.join(out_dir, "fid_fake")
    os.makedirs(fake_dir, exist_ok=True)
    G_ema.eval()
    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            b    = min(8, n_samples - generated)
            z    = torch.randn(b, z_dim, device=device)
            imgs = G_ema(z, mix_prob=0.0, truncation_psi=0.7).cpu()
            for j, img in enumerate(imgs):
                arr = ((img.squeeze().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                Image.fromarray(arr,"L").convert("RGB").save(
                    os.path.join(fake_dir, f"fake_{generated+j:04d}.png"))
            generated += b

    real_rgb = os.path.join(out_dir, "fid_real_rgb")
    if not os.path.exists(real_rgb):
        os.makedirs(real_rgb)
        for p in glob.glob(os.path.join(real_dir, "*.png")):
            Image.open(p).convert("RGB").save(
                os.path.join(real_rgb, os.path.basename(p)))

    m = calculate_metrics(input1=real_rgb, input2=fake_dir,
                          cuda=False, fid=True, verbose=False)
    return m["frechet_inception_distance"]


# ==============================================================================
# 14.  UTILITIES
# ==============================================================================

def save_sample_grid(G_ema, z_fixed, epoch, out_dir, device):
    G_ema.eval()
    with torch.no_grad():
        samples = G_ema(z_fixed.to(device),
                        mix_prob=0.0, truncation_psi=0.7).cpu()
    grid    = vutils.make_grid(samples, nrow=4, normalize=True, value_range=(-1,1))
    grid_np = grid.permute(1,2,0).numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off"); plt.title(f"Epoch {epoch} [EMA, ψ=0.7]"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "samples", f"epoch_{epoch:05d}.png"), dpi=100)
    plt.close()


def save_checkpoint(G, D, G_ema, opt_G, opt_D, pl_mean, ada, epoch, out_dir):
    path = os.path.join(out_dir, "checkpoints", f"ckpt_{epoch:05d}.pt")
    torch.save({
        "epoch"  : epoch,
        "G"      : G.state_dict(),
        "D"      : D.state_dict(),
        "G_ema"  : G_ema.G_ema.state_dict(),
        "opt_G"  : opt_G.state_dict(),
        "opt_D"  : opt_D.state_dict(),
        "pl_mean": pl_mean,
        "ada_p"  : ada.p,
    }, path)
    print(f"  [CKPT] {path}  (ADA p={ada.p:.3f})")


# ==============================================================================
# 15.  TRAINING LOOP
# ==============================================================================

def train():
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = get_device()
    os.makedirs(CFG["out_dir"], exist_ok=True)
    os.makedirs(os.path.join(CFG["out_dir"], "samples"),     exist_ok=True)
    os.makedirs(os.path.join(CFG["out_dir"], "checkpoints"), exist_ok=True)

    with open(os.path.join(CFG["out_dir"], "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = XRayDataset(CFG["data_dir"], CFG["image_size"])
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         shuffle=True, num_workers=0,
                         pin_memory=False, drop_last=True)
    print(f"[DATA] {len(loader)} batches/epoch")

    # ── Models ─────────────────────────────────────────────────────────────────
    G = Generator(
        z_dim=CFG["z_dim"], w_dim=CFG["w_dim"],
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"],
        max_channels=CFG["max_channels"],
        map_layers=CFG["map_layers"]).to(device)

    D = Discriminator(
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"],
        max_channels=CFG["max_channels"]).to(device)

    # EMA wrapper
    ema = GeneratorEMA(G, beta=CFG["ema_beta"])

    g_params = sum(p.numel() for p in G.parameters())/1e6
    d_params = sum(p.numel() for p in D.parameters())/1e6
    print(f"[MODEL] G: {g_params:.2f}M  D: {d_params:.2f}M params")

    # ── Optimisers ─────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(),
                             lr=CFG["lr_G"], betas=CFG["betas"])
    opt_D = torch.optim.Adam(D.parameters(),
                             lr=CFG["lr_D"], betas=CFG["betas"])

    # ── Auxiliary modules ──────────────────────────────────────────────────────
    ssim_loss_fn = SSIMLoss().to(device)
    ada          = AdaptiveAugment(
        target_rt  = CFG["ada_target"],
        interval   = CFG["ada_interval"],
        kimg       = CFG["ada_kimg"],
        batch_size = CFG["batch_size"])

    # Running EMA of path length (starts at 0.0; adapts quickly)
    pl_mean = torch.tensor(0.0, device=device)

    z_fixed       = torch.randn(16, CFG["z_dim"])
    real_fid_dir  = save_real_images_for_fid(dataset, CFG["out_dir"])
    log           = {"epoch":[], "loss_G":[], "loss_D":[],
                     "ada_p":[], "pl_mean":[], "fid":[]}
    d_step        = 0   # global discriminator step counter
    g_step        = 0   # global generator step counter

    print("\n[TRAIN] Starting …")
    print("        Watch ADA p: it should rise from 0 towards ~0.6+ within")
    print("        the first 500 epochs if D is overfitting (expected with 93 imgs)")

    for epoch in range(1, CFG["num_epochs"] + 1):
        e_loss_G = e_loss_D = 0.0
        t0 = time.time()

        for real_batch in loader:
            real_batch = real_batch.to(device)    # [B, 1, H, W]  clean
            B = real_batch.size(0)

            # ================================================================
            # DISCRIMINATOR STEP
            # ================================================================
            opt_D.zero_grad()

            # Generate fake images (no gradient to G at this stage)
            z = torch.randn(B, CFG["z_dim"], device=device)
            with torch.no_grad():
                fake = G(z, mix_prob=CFG["style_mix_prob"])

            # Apply ADA augmentation to BOTH real and fake before D
            # (critical: we augment fake too so D cannot distinguish
            #  "augmented real" from "non-augmented fake")
            real_aug = ada.augment(real_batch)
            fake_aug = ada.augment(fake.detach())

            loss_D, real_logits = d_loss_fn(D, real_aug, fake_aug)
            ada.accumulate_rt(real_logits)

            # R1 on augmented real (lazy regularisation every r1_interval steps)
            if d_step % CFG["r1_interval"] == 0:
                r1 = r1_penalty(D, real_aug.detach().clone(), device)
                loss_D = loss_D + (CFG["r1_gamma"]/2) * r1 * CFG["r1_interval"]

            loss_D.backward()
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()
            ada.update_p()       # adjust ADA probability
            d_step += 1

            # ================================================================
            # GENERATOR STEP
            # ================================================================
            opt_G.zero_grad()

            z    = torch.randn(B, CFG["z_dim"], device=device)
            fake = G(z, mix_prob=CFG["style_mix_prob"])

            # Adversarial loss (on ADA-augmented fake)
            fake_aug_G = ada.augment(fake)
            adv_loss   = g_loss_fn(D, fake_aug_G)

            # SSIM structural loss (on clean fake vs clean real)
            s_loss = ssim_loss_fn(fake, real_batch)

            loss_G = adv_loss + CFG["ssim_weight"] * s_loss

            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            # Update EMA after every G step
            ema.update(G)
            g_step += 1

            # Path Length Regularisation (lazy, every pl_interval G-steps)
            if g_step % CFG["pl_interval"] == 0:
                opt_G.zero_grad()
                z_pl     = torch.randn(B, CFG["z_dim"], device=device)
                pl_loss, pl_mean = path_length_regularise(
                    G, z_pl, pl_mean, CFG["pl_decay"], device)
                # Guard: skip the PLR backward if the loss is NaN
                # (can happen at very start of training before G stabilises)
                if not torch.isnan(pl_loss):
                    (CFG["pl_weight"] * pl_loss * CFG["pl_interval"]).backward()
                    nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                    opt_G.step()
                    ema.update(G)

            # ── NaN guard ────────────────────────────────────────────────────
            # If either loss is NaN (e.g. exploding gradients on first steps),
            # skip accumulation for this batch rather than poisoning the epoch avg.
            if torch.isnan(loss_G) or torch.isnan(loss_D):
                print(f"  [WARN] NaN loss detected at d_step={d_step} "
                      f"g_step={g_step} — skipping batch")
                # Reset pl_mean if it's part of the problem
                pl_mean = torch.zeros_like(pl_mean)
                continue

            e_loss_G += loss_G.item()
            e_loss_D += loss_D.item()

        # ── Epoch summary ─────────────────────────────────────────────────────
        n       = max(len(loader), 1)   # safeguard against all-NaN epoch
        avg_G   = e_loss_G / n
        avg_D   = e_loss_D / n
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:5d}/{CFG['num_epochs']}]  "
              f"G={avg_G:.4f}  D={avg_D:.4f}  "
              f"ADA_p={ada.p:.3f}  PLmean={pl_mean.item():.2f}  "
              f"({elapsed:.1f}s)")

        log["epoch"].append(epoch);   log["loss_G"].append(avg_G)
        log["loss_D"].append(avg_D);  log["ada_p"].append(ada.p)
        log["pl_mean"].append(pl_mean.item())

        if epoch % CFG["sample_interval"] == 0 or epoch == 1:
            save_sample_grid(ema.G_ema, z_fixed, epoch, CFG["out_dir"], device)

        if epoch % CFG["ckpt_interval"] == 0:
            save_checkpoint(G, D, ema, opt_G, opt_D,
                            pl_mean, ada, epoch, CFG["out_dir"])

        if epoch % CFG["fid_interval"] == 0 and HAS_FIDELITY:
            fid = compute_fid(ema.G_ema, real_fid_dir, CFG["out_dir"],
                              CFG["n_fid_samples"], CFG["z_dim"], device)
            print(f"  [FID] {fid:.2f}")
            log["fid"].append((epoch, fid))

        if epoch % 100 == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(log["epoch"], log["loss_G"], label="G")
            axes[0].plot(log["epoch"], log["loss_D"], label="D")
            axes[0].set_title("Losses"); axes[0].legend()
            axes[1].plot(log["epoch"], log["ada_p"], color="orange")
            axes[1].set_title("ADA augmentation probability p")
            axes[1].set_ylim(0, 1)
            axes[2].plot(log["epoch"], log["pl_mean"], color="purple")
            axes[2].set_title("Path Length mean (PLR EMA)")
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["out_dir"], "training_curves.png"))
            plt.close()

    print("\n[TRAIN] Done.")
    save_checkpoint(G, D, ema, opt_G, opt_D,
                    pl_mean, ada, CFG["num_epochs"], CFG["out_dir"])


# ==============================================================================
# 16.  INFERENCE  (with proper truncation trick)
# ==============================================================================

def generate(checkpoint_path, n=16, out_path="./generated.png",
             truncation_psi=0.7):
    """
    Generate n synthetic X-rays from the EMA generator.

    Args:
        checkpoint_path: path to a .pt checkpoint file
        n              : number of images
        out_path       : output PNG path
        truncation_psi : quality/diversity tradeoff.
            0.5  = very conservative / high quality / low diversity
            0.7  = recommended (good balance)
            1.0  = no truncation (max diversity but occasional artefacts)
    """
    device = get_device()
    G = Generator(
        z_dim=CFG["z_dim"], w_dim=CFG["w_dim"],
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"],
        max_channels=CFG["max_channels"],
        map_layers=CFG["map_layers"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G_ema"])    # use EMA weights
    G.eval()

    with torch.no_grad():
        z    = torch.randn(n, CFG["z_dim"], device=device)
        imgs = G(z, mix_prob=0.0, truncation_psi=truncation_psi).cpu()

    grid    = vutils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    grid_np = (grid.permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(grid_np.squeeze(), "L").save(out_path)
    print(f"[GEN] {n} images saved -> {out_path}")


# ==============================================================================
# 17.  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    """
    SETUP  (identical to v1):
    ------------------------------------------------------------------
      pip install torch torchvision pillow matplotlib numpy
      pip install torch-fidelity

      Edit CFG['data_dir'] to point at your .tif folder.
      python stylegan_xray_v2.py

    KEY THINGS TO MONITOR (new in v2):
    ------------------------------------------------------------------
    ADA p (augmentation probability):
      Starts at 0.  Should rise to 0.3–0.8 within 500–1000 epochs
      as D starts to overfit to the 93 real images.
      If p = 0 after 1000 epochs: D is not overfitting (unlikely).
      If p = 1.0 and D loss still -> 0: dataset is too small even for ADA.

    PLR mean (path length EMA):
      Starts at 0.  Settles to a stable value ~50–200 range.
      If PLR mean grows without bound: pl_weight too high, reduce it.
      If PLR mean is always 0: pl_interval too large, reduce it.

    MinibatchStd effect:
      Not directly observable in the loss, but its presence prevents
      D from assigning identical scores to repeated fake images.
      Symptom of MinibatchStd working: G loss does not drop sharply
      to near-zero early in training (mode collapse prevented).

    SSIM component:
      Watch the G loss — if it suddenly spikes up, the SSIM term is
      conflicting with the adversarial term.  Reduce ssim_weight to 0.05.

    SUMMARY TABLE: v1 -> v2 improvements
    ------------------------------------------------------------------
    Problem in v1            Fix in v2              Where in code
    ─────────────────────── ──────────────────────  ─────────────────
    Unequal learning rates  EqLR (EqLinear/EqConv)  Section 4
    Mode collapse blind spot MinibatchStd            Section 7
    D memorises 93 images   ADA                     Section 8
    Anisotropic W space     Path Length Reg.        Section 10
    No structural loss      SSIM perceptual loss    Section 11
    W-space entanglement    Style mixing            Section 6/Generator
    Oscillating G weights   Generator EMA           Section 12
    No quality/diversity ctl Truncation trick       Generator.forward()
    ------------------------------------------------------------------
    """
    train()
