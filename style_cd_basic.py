"""
==============================================================================
  VANILLA STYLEGAN FOR MEDICAL X-RAY SYNTHESIS
  Target: Mac Mini M4 (24 GB RAM) | 93 grayscale 8-bit TIFF images
==============================================================================

WHAT THIS FILE DOES (end to end):
  1. Loads your 93 grayscale X-ray TIFFs, normalises them to [-1, 1]
  2. Defines a StyleGAN Generator:
       Mapping Network  : z (random noise) --> w (disentangled latent code)
       Synthesis Network: w injected via AdaIN into a sequence of
                          upsampling Conv blocks
  3. Defines a PatchGAN-style Discriminator
  4. Trains with non-saturating GAN loss + R1 gradient penalty
  5. Logs FID (Frechet Inception Distance) every N epochs
  6. Saves generated grids and model checkpoints periodically

WHAT IS *NOT* HERE (intentional for "vanilla" starting point):
  - Progressive growing of resolution
  - Style mixing regularisation
  - Truncation trick at inference
  - Equalized learning rate
  - Minibatch std in discriminator
  - Path length regularisation
  (All of these will be added in later iterations)

HARDWARE NOTE:
  Uses PyTorch MPS backend (Apple Silicon Metal).
  Falls back to CPU if MPS is unavailable.
==============================================================================
"""

# -----------------------------------------------------------------------------
# 0.  IMPORTS
# -----------------------------------------------------------------------------
import os, math, time, glob, json, random
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
    print("[WARN] torch-fidelity not found. FID will be skipped.")
    print("       Install with: pip install torch-fidelity")

# -----------------------------------------------------------------------------
# 1.  GLOBAL CONFIGURATION
#     Everything lives here so you never have to hunt through the code.
# -----------------------------------------------------------------------------
CFG = {
    # -- Data ------------------------------------------------------------------
    "data_dir"       : "xray_tifs",   # folder containing your .tif files
    "image_size"     : 256,             # resize all images to this square
    # 256 is a good balance: enough texture detail, fits in 24 GB easily.
    # Try 512 later once training is stable.

    # -- Latent space ----------------------------------------------------------
    "z_dim"          : 512,  # dimension of the input noise vector z
    "w_dim"          : 512,  # dimension of the disentangled latent w
    # In StyleGAN, z is a *random* point in a spherical Gaussian;
    # w is a *learned* point in a flattened, more disentangled space.

    # -- Architecture ----------------------------------------------------------
    "map_layers"     : 8,    # depth of the mapping network (z -> w)
    # More layers = more non-linear disentanglement.
    # 8 is the original StyleGAN value; reduce to 4 if training is slow.
    "base_channels"  : 512,  # feature-map width at 4x4 (smallest) resolution
    "max_channels"   : 512,  # cap on feature-map width at any resolution
    # channel count halves as resolution doubles:
    #   4x4   -> 512 ch
    #   8x8   -> 512 ch
    #  16x16  -> 512 ch
    #  32x32  -> 256 ch
    #  64x64  -> 128 ch
    # 128x128 ->  64 ch
    # 256x256 ->  32 ch  (final layer)

    # -- Training --------------------------------------------------------------
    "batch_size"     : 8,    # SMALL because you only have 93 images.
    # With 93 images and batch 4, one epoch = ~23 steps.
    "num_epochs"     : 3000,
    "lr_G"           : 1e-4, # generator learning rate
    "lr_D"           : 4e-4, # discriminator LR (2-4x higher is common)
    # This is the "TTUR" trick (Two Time-scale Update Rule).
    "betas"          : (0.0, 0.99),  # Adam beta1=0 is standard in StyleGAN
    # beta1=0 means NO momentum on the first moment (gradient mean).
    # Prevents the optimizer coasting on stale momentum in the adversarial game.

    # -- Losses / regularisation -----------------------------------------------
    "r1_gamma"       : 10.0, # R1 gradient penalty weight
    # Penalises D for large gradients on real images.
    # Gamma=10 is the default from Mescheder et al. 2018.
    "r1_interval"    : 16,   # apply R1 only every N D-steps (saves compute)

    # -- Logging / saving ------------------------------------------------------
    "sample_interval": 100,  # save image grid every N epochs
    "ckpt_interval"  : 500,  # save model checkpoint every N epochs
    "fid_interval"   : 500,  # compute FID every N epochs (expensive)
    "n_fid_samples"  : 93,   # #synthetic images for FID (match real count)
    "out_dir"        : "basic_outputs",
    "seed"           : 42,
}

# -----------------------------------------------------------------------------
# 2.  DEVICE SETUP  (MPS = Apple Metal, the M4's GPU)
# -----------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal) backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] Using CUDA GPU.")
    else:
        device = torch.device("cpu")
        print("[WARN] No GPU found - running on CPU (will be slow).")
    return device

# -----------------------------------------------------------------------------
# 3.  DATASET
#     Loads 8-bit grayscale TIFF files, returns tensors in [-1, 1].
# -----------------------------------------------------------------------------
class XRayDataset(Dataset):
    """
    Loads grayscale X-ray TIFFs.

    Preprocessing pipeline:
      1. Open with PIL (handles 8-bit TIFF natively)
      2. Convert to 'L' (single-channel 8-bit grayscale) -- just in case
         the TIFF has extra metadata channels
      3. Resize to image_size x image_size  (bilinear interpolation)
      4. ToTensor()  : [0, 255] uint8  ->  [0.0, 1.0] float32, adds C dim
      5. Normalize() : [0, 1]          ->  [-1, 1]
         mu = 0.5, sigma = 0.5  ->  (x - 0.5) / 0.5

    Why [-1, 1]?
      The generator's final activation is Tanh, which outputs [-1, 1].
      Matching data range to network output range avoids a systematic
      bias in the discriminator.

    Data augmentation (light):
      Random horizontal flip only.  X-rays are approximately left-right
      symmetric for the chest, so this is safe and effectively doubles
      your dataset.  We avoid rotation/crop because medical images have
      specific spatial semantics (heart on left, liver on right, etc.).
    """
    def __init__(self, data_dir, image_size, augment=True):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")) +
                            glob.glob(os.path.join(data_dir, "*.tiff")))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No .tif/.tiff files found in '{data_dir}'.\n"
                f"Set CFG['data_dir'] to the folder containing your images."
            )
        print(f"[DATA] Found {len(self.paths)} images in '{data_dir}'.")

        transforms_list = [
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BILINEAR),
        ]
        if augment:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        transforms_list += [
            T.ToTensor(),                        # -> [0, 1]
            T.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1]
        ]
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")  # force grayscale
        return self.transform(img)   # shape: [1, H, W]


# -----------------------------------------------------------------------------
# 4.  BUILDING BLOCKS
# -----------------------------------------------------------------------------

# -- 4a. Pixel Normalisation ---------------------------------------------------
class PixelNorm(nn.Module):
    """
    Normalises each pixel's feature vector to unit length.
    Used in the mapping network to prevent the latent code from drifting
    to very large magnitudes, which can destabilise early training.

    Formula:  x / sqrt(mean(x^2) + eps)
    """
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


# -- 4b. Adaptive Instance Normalisation (AdaIN) -------------------------------
class AdaIN(nn.Module):
    """
    The CORE of StyleGAN: how style (w) is injected into the image.

    Standard Instance Normalisation:
        IN(x) = (x - mu(x)) / sigma(x)
        where mu, sigma are computed *per feature map per sample*.

    AdaIN extends this by replacing the normalised mean and std with
    LEARNED values derived from the style code w:
        AdaIN(x, y) = y_s * IN(x) + y_b
        y_s, y_b = two learned linear projections of w

    WHY THIS WORKS FOR STYLE:
      After instance normalisation, x carries only *spatial structure*.
      y_s and y_b then impose a style (texture statistics) on top.
      Because each block has its own AdaIN, coarse blocks learn
      coarse style (overall brightness, contrast), fine blocks learn
      fine style (micro-texture, noise-like details).

    Args:
        w_dim     : dimension of w
        n_channels: number of feature maps in x
    """
    def __init__(self, w_dim, n_channels):
        super().__init__()
        self.style_scale = nn.Linear(w_dim, n_channels)
        self.style_bias  = nn.Linear(w_dim, n_channels)
        # Initialise scale to 1 so AdaIN starts as vanilla InstanceNorm
        nn.init.ones_(self.style_scale.weight)
        nn.init.zeros_(self.style_bias.weight)

    def forward(self, x, w):
        # x : [B, C, H, W]
        # w : [B, w_dim]
        B, C, H, W = x.shape
        mean  = x.mean(dim=[2, 3], keepdim=True)         # [B, C, 1, 1]
        std   = x.std (dim=[2, 3], keepdim=True) + 1e-8  # [B, C, 1, 1]
        x_norm = (x - mean) / std

        y_s = self.style_scale(w).view(B, C, 1, 1)       # [B, C, 1, 1]
        y_b = self.style_bias (w).view(B, C, 1, 1)       # [B, C, 1, 1]
        return y_s * x_norm + y_b


# -- 4c. Synthesis Block -------------------------------------------------------
class SynthBlock(nn.Module):
    """
    One resolution stage in the generator synthesis network.

    Pipeline per block:
        [optional upsample 2x]
        -> Conv 3x3  -> LeakyReLU
        -> AdaIN(w)                  <- style injection
        -> Conv 3x3  -> LeakyReLU
        -> AdaIN(w)                  <- style injection again

    Injecting style TWICE per block lets the block independently
    control both the early-pass texture and the late-pass texture.

    The noise input (Gaussian noise added before each AdaIN) that
    appears in the full StyleGAN paper is OMITTED here for simplicity.
    It controls stochastic fine detail -- less critical for X-rays
    than for faces.

    Args:
        in_ch   : input feature channels
        out_ch  : output feature channels
        w_dim   : latent w dimension
        upsample: whether to 2x upsample at the start of this block
    """
    def __init__(self, in_ch, out_ch, w_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adain1 = AdaIN(w_dim, out_ch)
        self.adain2 = AdaIN(w_dim, out_ch)
        self.act    = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        if self.upsample:
            # Bilinear upsample: smoother than nearest-neighbour,
            # avoids the checkerboard artefacts of transposed convolutions.
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
        x = self.act(self.conv1(x))
        x = self.adain1(x, w)
        x = self.act(self.conv2(x))
        x = self.adain2(x, w)
        return x


# -----------------------------------------------------------------------------
# 5.  GENERATOR
# -----------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    """
    Maps a random noise vector z ~ N(0,I) to a disentangled latent w.

    Architecture: PixelNorm -> [Linear -> LeakyReLU] x num_layers

    WHY A SEPARATE MAPPING NETWORK?
      In a plain GAN, z is passed directly as style.  The problem is that
      z is drawn from a *spherical Gaussian*, which is a fixed distribution.
      The generator must *warp* this fixed shape to match the true data
      manifold, and this warping entangles attributes (brightness & texture
      might be fused in a single dimension of z).

      The mapping network learns a non-linear transformation that
      *unfolds* the Gaussian into a more disentangled space W where
      ideally each dimension controls one interpretable attribute.
      This is why in StyleGAN2 you can do meaningful style mixing.
    """
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = [PixelNorm()]
        in_dim = z_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # [B, w_dim]


class Generator(nn.Module):
    """
    Full StyleGAN generator.

    Structure:
        z [B, z_dim]
          v  MappingNetwork
        w [B, w_dim]
          v  (broadcast to each SynthBlock)
        4x4 learned constant
          v  SynthBlock (no upsample)    4x4
          v  SynthBlock (2x upsample)    8x8
          v  SynthBlock (2x upsample)   16x16
          v  SynthBlock (2x upsample)   32x32
          v  SynthBlock (2x upsample)   64x64
          v  SynthBlock (2x upsample)  128x128
          v  SynthBlock (2x upsample)  256x256
          v  Conv 1x1  (project to 1 channel grayscale)
          v  Tanh      (clamp to [-1, 1])

    The *learned constant* (4x4 starting tensor):
      In a standard DCGAN you'd pass w directly into a linear layer to
      make the 4x4 seed.  StyleGAN instead uses a single learned constant
      (same for every sample), because the style is already injected via
      AdaIN -- there is no need to encode it in the spatial seed.
      This enforces a cleaner separation between structure (the constant's
      learned pattern) and style (w via AdaIN).
    """
    def __init__(self, z_dim=512, w_dim=512, image_size=256,
                 base_channels=512, max_channels=512, map_layers=8):
        super().__init__()
        self.mapping    = MappingNetwork(z_dim, w_dim, map_layers)
        self.log2_size  = int(math.log2(image_size))
        num_blocks      = self.log2_size - 1  # 7 blocks for 256x256

        def ch(log2_res):
            return min(base_channels >> max(log2_res - 2, 0), max_channels)

        # 4x4 learned constant  [1, base_channels, 4, 4]
        self.const = nn.Parameter(torch.randn(1, ch(2), 4, 4))

        self.blocks = nn.ModuleList()
        in_ch = ch(2)
        for i in range(num_blocks):
            log2_res = i + 2
            out_ch   = ch(log2_res + 1)
            upsample = (i > 0)
            self.blocks.append(SynthBlock(in_ch, out_ch, w_dim, upsample))
            in_ch = out_ch

        self.to_rgb = nn.Conv2d(in_ch, 1, 1)
        self.tanh   = nn.Tanh()

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return self.tanh(self.to_rgb(x))


# -----------------------------------------------------------------------------
# 6.  DISCRIMINATOR
# -----------------------------------------------------------------------------
class DiscBlock(nn.Module):
    """
    One downsampling stage in the discriminator.

    Pipeline:
        Conv 3x3 -> LeakyReLU
        Conv 3x3 -> LeakyReLU
        AvgPool 2x2 (stride 2 downsample)

    We use AvgPool for downsampling rather than strided Conv because
    AvgPool is smoother and reduces high-frequency aliasing artefacts.

    No BatchNorm in discriminator:
      BN in D has been shown to leak information between samples in a
      batch and can cause training instabilities.  We use no normalisation
      in the discriminator (the R1 penalty takes care of stability).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )
    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    Mirrored architecture of the generator (resolution descends
    from image_size -> 4x4).

    At the 4x4 stage:
        Flatten -> Linear -> 1 scalar (logit, *not* sigmoid)

    Why no sigmoid?
      We use non-saturating loss which expects raw logits.
      The loss itself applies the sigmoid internally via BCEWithLogitsLoss,
      which is numerically more stable.
    """
    def __init__(self, image_size=256, base_channels=512, max_channels=512):
        super().__init__()
        log2_size = int(math.log2(image_size))

        def ch(log2_res):
            return min(base_channels >> max(log2_res - 2, 0), max_channels)

        self.from_rgb = nn.Sequential(
            nn.Conv2d(1, ch(log2_size), 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        in_ch = ch(log2_size)
        for i in range(log2_size - 2, 0, -1):
            out_ch = ch(i)
            blocks.append(DiscBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, 1),
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# 7.  LOSS FUNCTIONS
# -----------------------------------------------------------------------------

def d_loss_fn(D, real, fake_detach):
    """
    Non-saturating GAN discriminator loss.

      L_D = E[softplus(-D(real))] + E[softplus(D(fake))]
          = -E[log sigma(D(real))] - E[log(1 - sigma(D(fake)))]

    Equivalent to cross-entropy:
      real images -> label 1   (D should output large positive values)
      fake images -> label 0   (D should output large negative values)

    BCEWithLogitsLoss fuses sigmoid+log in one numerically stable op.
    """
    bce = nn.BCEWithLogitsLoss()
    loss_real = bce(D(real),         torch.ones_like (D(real)))
    loss_fake = bce(D(fake_detach),  torch.zeros_like(D(fake_detach)))
    return loss_real + loss_fake


def g_loss_fn(D, fake):
    """
    Non-saturating GAN generator loss.

      L_G = E[softplus(-D(fake))] = -E[log sigma(D(fake))]

    The generator tries to make D output large positive values for fakes.

    WHY "non-saturating"?
      The original GAN used L_G = -E[log(1 - D(fake))].
      At the start of training D is good and D(fake) ~= 0, so
      log(1 - 0) ~= 0 and the gradient vanishes (saturates).
      The non-saturating form -log(D(fake)) has a large gradient
      when D(fake) is small -- exactly when G needs guidance most.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(D(fake), torch.ones_like(D(fake)))


def r1_penalty(D, real):
    """
    R1 Gradient Penalty (Mescheder et al. 2018).

      R1 = (gamma/2) * E[||grad_x D(x)||^2]  on REAL images only.

    Penalises D for having large gradients w.r.t. its input at real
    data points.  Forces D to be locally Lipschitz around the real
    data manifold, preventing D from memorising individual training
    images via sharp spike functions.

    Without regularisation with only 93 images:
      D can achieve zero loss by perfectly memorising all 93 images,
      giving G zero useful gradient.  R1 prevents this.

    create_graph=True is needed so we can backprop through the
    gradient computation itself (the penalty's own gradient must
    flow back to D's parameters).
    """
    real.requires_grad_(True)
    real_logit = D(real)
    grad = torch.autograd.grad(
        outputs=real_logit.sum(),
        inputs=real,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(real.size(0), -1).sum(1).mean()


# -----------------------------------------------------------------------------
# 8.  METRICS
# -----------------------------------------------------------------------------

def save_real_images_for_fid(dataset, out_dir, device):
    """Save all real images as PNGs for FID comparison."""
    real_dir = os.path.join(out_dir, "fid_real")
    os.makedirs(real_dir, exist_ok=True)
    for i, img in enumerate(dataset):
        img_np = ((img.squeeze().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np, mode="L").save(
            os.path.join(real_dir, f"real_{i:04d}.png"))
    return real_dir


def compute_fid(G, real_dir, out_dir, n_samples, z_dim, device):
    """
    Generates n_samples fake images and computes FID.

    FID (Frechet Inception Distance):
      Embeds both real and fake images with InceptionV3, fits a
      multivariate Gaussian to each embedding set, then measures:

        FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r * Sigma_f))

      Lower FID = more similar distributions = better generator.
        FID < 10  : very good  (for large datasets)
        FID < 50  : decent
        FID > 100 : poor

    With only 93 real images, FID estimates are noisy.  Use it as a
    rough trend indicator rather than an absolute number.

    InceptionV3 expects RGB, so we repeat the grayscale channel 3x.
    This is standard practice for grayscale medical image FID.
    """
    if not HAS_FIDELITY:
        return None

    fake_dir = os.path.join(out_dir, "fid_fake")
    os.makedirs(fake_dir, exist_ok=True)
    G.eval()
    with torch.no_grad():
        generated = 0
        while generated < n_samples:
            batch = min(8, n_samples - generated)
            z    = torch.randn(batch, z_dim, device=device)
            imgs = G(z).cpu()
            for j, img in enumerate(imgs):
                img_np = ((img.squeeze().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np, "L").convert("RGB").save(
                    os.path.join(fake_dir, f"fake_{generated+j:04d}.png"))
            generated += batch
    G.train()

    real_rgb_dir = os.path.join(out_dir, "fid_real_rgb")
    if not os.path.exists(real_rgb_dir):
        os.makedirs(real_rgb_dir)
        for p in glob.glob(os.path.join(real_dir, "*.png")):
            Image.open(p).convert("RGB").save(
                os.path.join(real_rgb_dir, os.path.basename(p)))

    metrics = calculate_metrics(
        input1=real_rgb_dir, input2=fake_dir,
        cuda=False, fid=True, verbose=False)
    return metrics["frechet_inception_distance"]


# -----------------------------------------------------------------------------
# 9.  UTILITIES
# -----------------------------------------------------------------------------

def save_sample_grid(G, z_fixed, epoch, out_dir, device):
    G.eval()
    with torch.no_grad():
        samples = G(z_fixed.to(device)).cpu()
    G.train()
    grid = vutils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off"); plt.title(f"Epoch {epoch}"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "samples", f"epoch_{epoch:05d}.png"), dpi=100)
    plt.close()


def save_checkpoint(G, D, opt_G, opt_D, epoch, out_dir):
    path = os.path.join(out_dir, "checkpoints", f"ckpt_epoch_{epoch:05d}.pt")
    torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()}, path)
    print(f"  [CKPT] Saved -> {path}")


# -----------------------------------------------------------------------------
# 10.  TRAINING LOOP
# -----------------------------------------------------------------------------

def train():
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = get_device()
    os.makedirs(CFG["out_dir"], exist_ok=True)
    os.makedirs(os.path.join(CFG["out_dir"], "samples"),     exist_ok=True)
    os.makedirs(os.path.join(CFG["out_dir"], "checkpoints"), exist_ok=True)

    with open(os.path.join(CFG["out_dir"], "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    # -- Dataset & DataLoader --------------------------------------------------
    dataset = XRayDataset(CFG["data_dir"], CFG["image_size"])
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         shuffle=True, num_workers=0,
                         pin_memory=False, drop_last=True)
    print(f"[DATA] Batches per epoch: {len(loader)}")

    # -- Models ----------------------------------------------------------------
    G = Generator(
        z_dim=CFG["z_dim"], w_dim=CFG["w_dim"],
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"], max_channels=CFG["max_channels"],
        map_layers=CFG["map_layers"]).to(device)

    D = Discriminator(
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"],
        max_channels=CFG["max_channels"]).to(device)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[MODEL] G: {g_params:.2f}M params  |  D: {d_params:.2f}M params")

    # -- Optimisers ------------------------------------------------------------
    # Adam with beta1=0, beta2=0.99 (standard for StyleGAN).
    # beta1=0: no momentum on gradient mean -> safer in adversarial setting.
    opt_G = torch.optim.Adam(G.parameters(), lr=CFG["lr_G"], betas=CFG["betas"])
    opt_D = torch.optim.Adam(D.parameters(), lr=CFG["lr_D"], betas=CFG["betas"])

    z_fixed = torch.randn(16, CFG["z_dim"])  # fixed noise for visualisation
    real_fid_dir = save_real_images_for_fid(dataset, CFG["out_dir"], device)
    log = {"epoch": [], "loss_G": [], "loss_D": [], "fid": []}
    d_step_count = 0

    print("\n[TRAIN] Starting training ...")

    for epoch in range(1, CFG["num_epochs"] + 1):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        t0 = time.time()

        for real_batch in loader:
            real_batch = real_batch.to(device)
            B = real_batch.size(0)

            # ================================================================
            # STEP 1: TRAIN DISCRIMINATOR
            # Goal: D(real) -> 1,  D(fake) -> 0
            # ================================================================
            opt_D.zero_grad()

            z = torch.randn(B, CFG["z_dim"], device=device)
            with torch.no_grad():
                fake = G(z)
            loss_D = d_loss_fn(D, real_batch, fake)

            r1_loss = torch.tensor(0.0, device=device)
            if d_step_count % CFG["r1_interval"] == 0:
                r1_loss = r1_penalty(D, real_batch.detach().clone())
                # Scale penalty by interval (lazy regularisation scaling)
                loss_D = loss_D + (CFG["r1_gamma"] / 2) * r1_loss * CFG["r1_interval"]

            loss_D.backward()
            # Gradient clipping: prevents catastrophic discriminator updates.
            # Not in vanilla StyleGAN but a useful safety net for small datasets.
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()
            d_step_count += 1

            # ================================================================
            # STEP 2: TRAIN GENERATOR
            # Goal: make D(fake) -> 1  (fool D)
            # ================================================================
            opt_G.zero_grad()
            z    = torch.randn(B, CFG["z_dim"], device=device)
            fake = G(z)
            loss_G = g_loss_fn(D, fake)
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

        n_batches = len(loader)
        avg_G = epoch_loss_G / n_batches
        avg_D = epoch_loss_D / n_batches
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:5d}/{CFG['num_epochs']}]  "
              f"loss_G={avg_G:.4f}  loss_D={avg_D:.4f}  ({elapsed:.1f}s)")

        log["epoch"].append(epoch); log["loss_G"].append(avg_G)
        log["loss_D"].append(avg_D)

        if epoch % CFG["sample_interval"] == 0 or epoch == 1:
            save_sample_grid(G, z_fixed, epoch, CFG["out_dir"], device)

        if epoch % CFG["ckpt_interval"] == 0:
            save_checkpoint(G, D, opt_G, opt_D, epoch, CFG["out_dir"])

        if epoch % CFG["fid_interval"] == 0 and HAS_FIDELITY:
            fid = compute_fid(G, real_fid_dir, CFG["out_dir"],
                              CFG["n_fid_samples"], CFG["z_dim"], device)
            print(f"  [FID] epoch {epoch}: {fid:.2f}")
            log["fid"].append((epoch, fid))

        if epoch % 100 == 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(log["epoch"], log["loss_G"], label="G")
            plt.plot(log["epoch"], log["loss_D"], label="D")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.legend(); plt.title("GAN Losses")
            if log["fid"]:
                plt.subplot(1, 2, 2)
                fid_epochs, fid_vals = zip(*log["fid"])
                plt.plot(fid_epochs, fid_vals, color="green")
                plt.xlabel("Epoch"); plt.ylabel("FID")
                plt.title("FID Score (lower = better)")
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["out_dir"], "training_curves.png"))
            plt.close()

    print("\n[TRAIN] Done.")
    save_checkpoint(G, D, opt_G, opt_D, CFG["num_epochs"], CFG["out_dir"])


# -----------------------------------------------------------------------------
# 11.  INFERENCE UTILITY
# -----------------------------------------------------------------------------

def generate(checkpoint_path, n=16, out_path="./generated.png", truncation=1.0):
    """
    Load a trained checkpoint and produce n synthetic X-ray images.

    Args:
        checkpoint_path : path to a .pt checkpoint file
        n               : number of images to generate
        out_path        : where to save the PNG grid
        truncation      : psi in (0, 1].
            Truncation trick (stub for later implementation):
              w = w_mean + psi * (w - w_mean)
              psi < 1 trades diversity for quality (pulls toward the mean).
              psi = 1 means no truncation (full diversity).
    """
    device = get_device()
    G = Generator(
        z_dim=CFG["z_dim"], w_dim=CFG["w_dim"],
        image_size=CFG["image_size"],
        base_channels=CFG["base_channels"], max_channels=CFG["max_channels"],
        map_layers=CFG["map_layers"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    with torch.no_grad():
        z    = torch.randn(n, CFG["z_dim"], device=device) * truncation
        imgs = G(z).cpu()

    grid = vutils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np.squeeze(), mode="L").save(out_path)
    print(f"[GEN] Saved {n} images -> {out_path}")


# -----------------------------------------------------------------------------
# 12.  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    SETUP INSTRUCTIONS (Mac Mini M4):
    ------------------------------------------------------------------
    1. Install dependencies:
         pip install torch torchvision pillow matplotlib numpy
         pip install torch-fidelity          # for FID metric

    2. Put your 93 .tif files in a folder, e.g. ./xray_tifs/

    3. Edit CFG["data_dir"] above to point to that folder.

    4. Run:
         python stylegan_xray.py

    EXPECTED BEHAVIOUR:
    ------------------------------------------------------------------
    - First few hundred epochs: generated images will look like noise.
    - ~500-1000 epochs: coarse structure (lung outlines) emerges.
    - ~2000+ epochs: finer texture (vascular markings) appears.
    - With only 93 images the model WILL overfit eventually.
      This is expected and normal. We will address it with augmentation
      and regularisation in later iterations.

    WHAT TO WATCH:
    ------------------------------------------------------------------
    - loss_D should stay in [0.3, 1.5].
      If it drops to ~0, D has won and G gets no gradient (mode collapse).
    - loss_G should generally decrease over time but will fluctuate.
    - FID should trend downward over thousands of epochs.
    - Sample grids saved every 100 epochs in ./stylegan_output/samples/

    WHAT COMES NEXT (later prompts):
    ------------------------------------------------------------------
    1. Equalized learning rate (per-layer LR scaling by fan-in)
    2. Minibatch standard deviation in discriminator
    3. Path length regularisation for G
    4. Adaptive augmentation (ADA) -- critical for 93-image datasets
    5. Progressive growing or multi-scale discriminator
    6. Perceptual / SSIM loss for medical texture preservation
    7. Style mixing regularisation
    ------------------------------------------------------------------
    """
    train()
