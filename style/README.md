# Coating Microstructure Generative Model
## StyleGAN2-ADA for Scientific Microstructure Augmentation

---

## Architecture Overview

| Component | Choice | Rationale |
|---|---|---|
| Generator | StyleGAN2 (W-space mapping + modulated conv) | Disentangled style control, state-of-the-art for texture synthesis |
| Discriminator | ResNet-style + MinibatchStdDev | Stable training, catches mode collapse |
| Augmentation | ADA (Adaptive Discriminator Augmentation) | **Critical** for 200-image datasets — prevents D overfitting |
| Perceptual loss | VGG19 relu1_1–relu4_1 features | Multi-scale feature matching |
| Texture loss | VGG Gram matrices | Explicitly penalizes texture distribution mismatch |
| Frequency loss | FFT magnitude L1 | Preserves micro-texture patterns invisible to pixel losses |
| Regularization | Lazy R1 (every 16 steps) | Stable D, prevents gradient blow-up |
| Precision | AMP (float16) | Halves VRAM; RTX 6000 ADA can run batch_size=8 at 512px |

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pillow numpy matplotlib scipy scikit-image
pip install clean-fid  # optional, for FID evaluation
```

### 2. Prepare data
```bash
# Your 200 TIF files go in one directory — no train/val split needed
# The dataset will auto-tile each 6:1 image into overlapping 512x512 patches
# 200 images × ~12 patches each × augmentation = ~10,000+ effective samples
ls /path/to/your/images/*.tif | wc -l  # should show 200
```

### 3. Train
```bash
python train.py \
  --data_dir /path/to/your/images \
  --output_dir ./runs/coating_v1 \
  --patch_size 512 \
  --batch_size 8 \
  --total_steps 200000 \
  --lambda_style 50.0 \
  --lambda_perc 0.1 \
  --lambda_fft 0.05
```

**RTX 6000 ADA (48GB VRAM) guidance:**
- `--batch_size 8` is safe at 512px; try 16 if VRAM allows
- Expected throughput: ~3–5 steps/sec → ~11–18 hours for 200k steps
- Monitor `ADA_p` — should stabilize around 0.4–0.8 (healthy D regularization)
- First 10k steps: G/D losses may fluctuate — this is normal
- Good convergence signs: `Style` loss drops steadily, samples look realistic by step 50k

### 4. Generate augmented data
```bash
# Generate 500 random images
python generate.py \
  --ckpt ./runs/coating_v1/latest.pt \
  --mode random \
  --n 500 \
  --truncation 0.7 \
  --output_dir ./generated/random

# Generate a wide strip (matching original 6:1 aspect ratio)
python generate.py \
  --ckpt ./runs/coating_v1/latest.pt \
  --mode wide \
  --n_tiles 6 \
  --output_dir ./generated/strips

# Latent interpolation (to verify smooth latent space)
python generate.py \
  --ckpt ./runs/coating_v1/latest.pt \
  --mode interpolate \
  --n_frames 30 \
  --output_dir ./generated/interp
```

### 5. Evaluate quality
```bash
python evaluate.py \
  --real_dir /path/to/your/images \
  --fake_dir ./generated/random \
  --output_dir ./eval_results
```

---

## Key Design Decisions for Small Datasets

### ADA (Adaptive Discriminator Augmentation)
The single most important choice for 200 images. Without it, the discriminator memorizes
all real images within a few thousand steps, causing G to produce random noise.

ADA dynamically adjusts augmentation probability `p` based on the discriminator's
sign-accuracy on real data (target: D correctly classifies ~60% of reals). This keeps
D in a productive "confused-but-learning" state throughout training.

Expected `p` trajectory: 0 → 0.3–0.7 (stabilizes after 20k steps)

### Multi-scale Texture Supervision
Pixel-level adversarial loss alone is insufficient for microstructures because:
- Features span multiple scales (grain boundaries = macro, surface roughness = micro)
- GAN discriminator may focus on easy global statistics

The triple loss (adversarial + VGG perceptual + Gram matrix + FFT) directly supervises
texture at multiple spatial frequencies.

### Patch-based Training vs Full Resolution
The 6:1 aspect ratio images are processed as overlapping 512×512 patches. This:
- Increases effective dataset size 10–15×
- Allows the model to learn local texture statistics
- Enables generation of arbitrarily-sized outputs by tiling

### Lazy Regularization
R1 gradient penalty applied every 16 steps (not every step). This maintains
regularization benefits at 1/16th the compute cost.

---

## Truncation Trick for Quality vs Diversity
```bash
# High quality, less diverse (truncation=0.5)
python generate.py --ckpt latest.pt --truncation 0.5 --mode random

# Balanced (truncation=0.7, recommended default)
python generate.py --ckpt latest.pt --truncation 0.7 --mode random

# Maximum diversity, slightly lower quality (truncation=1.0)
python generate.py --ckpt latest.pt --truncation 1.0 --mode random
```

---

## Hyperparameter Tuning

| Parameter | Default | If underfitting | If overfitting/mode collapse |
|---|---|---|---|
| `lambda_style` | 50.0 | Increase to 100 | Decrease to 20 |
| `lambda_perc` | 0.1 | Increase to 0.5 | Decrease to 0.05 |
| `lambda_fft` | 0.05 | Increase to 0.1 | Keep or decrease |
| `r1_gamma` | 10.0 | Decrease to 5.0 | Increase to 20 |
| `ada.target_rt` | 0.6 | — | Increase to 0.7 |

---

## File Structure
```
coating_stylegan/
├── train.py        # Main training loop + all model definitions
├── generate.py     # Sampling: random, wide strips, interpolation, truncation sweep
├── evaluate.py     # FID, SSIM, PSD, Gram matrix distance
├── README.md       # This file
└── requirements.txt
```
