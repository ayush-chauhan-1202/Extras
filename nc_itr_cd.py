"""
ct_recon_sirt.py — GPU-accelerated CT Reconstruction (SIRT) using CuPy

Setup:
    Cylinder centre:  (0, 0, 0)
    Source centre:    (0, -830.71, 0)
    Detector centre:  (0, +332.29, 0)
    Rotation:         Around Z axis, 15 angles: 0, 24, 48, ..., 336 degrees
    Detector:         350x350 pixels, 0.4 mm/pixel
    Projections:      32-bit float TIFs, air ~2000, material ~400

Install dependencies:
    pip install cupy-cuda12x tifffile numpy matplotlib tqdm
    (use cupy-cuda11x if on CUDA 11)

Run:
    python ct_recon_sirt.py --proj_dir ./projections --out_dir ./output --iters 50
"""

import os
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    GPU_AVAILABLE = True
    print(f"CuPy found. GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    print("CuPy not found — falling back to CPU (slow). Install: pip install cupy-cuda12x")
    import numpy as cp
    from scipy.ndimage import map_coordinates
    GPU_AVAILABLE = False

# ─── Configuration ─────────────────────────────────────────────────────────────

SAD          = 830.71   # mm, source to isocentre
DDD          = 332.29   # mm, detector to isocentre
SDD          = SAD + DDD

DET_W        = 350      # detector pixels (columns)
DET_H        = 350      # detector pixels (rows)
DET_PIX_MM   = 0.4      # mm per detector pixel

VOL_NX       = 300      # reconstruction volume voxels
VOL_NY       = 300
VOL_NZ       = 300
VOL_PIX_MM   = 0.4      # mm per voxel

AIR_VALUE    = None     # set to float to override auto-estimation from data

RELAXATION   = 0.5      # SIRT step size (0.1–1.0)
RAY_STEP_MM  = 0.2      # ray marching step (smaller = more accurate, slower)

ANGLES_DEG   = list(range(0, 360, 24))   # [0, 24, 48, ..., 336]

# ─── Geometry ──────────────────────────────────────────────────────────────────

def build_geometry(angles_deg):
    """
    Returns list of dicts, one per angle, with source position,
    detector centre, and detector u/v axes.
    """
    geoms = []
    for theta_deg in angles_deg:
        theta = np.radians(theta_deg)
        s, c = np.sin(theta), np.cos(theta)

        src    = np.array([-SAD * s, -SAD * c, 0.0])
        det_c  = np.array([ DDD * s,  DDD * c, 0.0])

        # u: detector column axis (horizontal, perpendicular to beam in XY)
        u_axis = np.array([ c, -s, 0.0])
        # v: detector row axis (vertical = Z)
        v_axis = np.array([ 0.0, 0.0, 1.0])

        geoms.append({
            'angle': theta_deg,
            'src':   src,
            'det_c': det_c,
            'u':     u_axis,
            'v':     v_axis,
        })
    return geoms

# ─── Beer-Lambert conversion ───────────────────────────────────────────────────

def to_attenuation(projs_raw, I0):
    """
    Convert raw detector values to attenuation sinogram.
    Raw: air ~2000 (bright), material ~400 (dark)
    Output: air -> 0, material -> positive (~1.6 for your values)
    """
    projs = np.clip(projs_raw.astype(np.float32), 1.0, None)
    ratio = np.clip(projs / I0, 1e-7, 1.0)
    return -np.log(ratio)

# ─── Load projections ──────────────────────────────────────────────────────────

def load_projections(proj_dir, angles_deg):
    projs = []
    for a in angles_deg:
        path = os.path.join(proj_dir, f"z{a}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing projection: {path}")
        img = tifffile.imread(path).astype(np.float32)
        assert img.shape == (DET_H, DET_W), \
            f"Expected ({DET_H},{DET_W}), got {img.shape} in {path}"
        projs.append(img)
    return np.stack(projs, axis=0)   # [N_angles, DET_H, DET_W]

# ─── Forward projector ─────────────────────────────────────────────────────────

def forward_project(volume_gpu, geom):
    """
    Cone-beam forward projection using ray marching + trilinear interpolation.
    volume_gpu: CuPy array [NZ, NY, NX]
    Returns CuPy array [DET_H, DET_W]
    """
    src   = cp.array(geom['src'],   dtype=cp.float32)
    det_c = cp.array(geom['det_c'], dtype=cp.float32)
    u     = cp.array(geom['u'],     dtype=cp.float32)
    v     = cp.array(geom['v'],     dtype=cp.float32)

    # Build detector pixel world positions: [DET_H, DET_W, 3]
    col_idx = cp.arange(DET_W, dtype=cp.float32) - DET_W / 2.0 + 0.5  # [DET_W]
    row_idx = cp.arange(DET_H, dtype=cp.float32) - DET_H / 2.0 + 0.5  # [DET_H]

    col_mm = col_idx * DET_PIX_MM   # [DET_W]
    row_mm = row_idx * DET_PIX_MM   # [DET_H]

    # Pixel positions in world: det_c + col*u + row*v
    # Shape: [DET_H, DET_W, 3]
    pix_pos = (det_c[None, None, :]
               + col_mm[None, :, None] * u[None, None, :]
               + row_mm[:, None, None] * v[None, None, :])

    # Ray direction: from source to each pixel
    ray_dir = pix_pos - src[None, None, :]              # [DET_H, DET_W, 3]
    ray_len = cp.linalg.norm(ray_dir, axis=-1, keepdims=True)
    ray_dir = ray_dir / ray_len                          # normalise

    # Volume half-sizes in mm
    half_x = VOL_NX * 0.5 * VOL_PIX_MM
    half_y = VOL_NY * 0.5 * VOL_PIX_MM
    half_z = VOL_NZ * 0.5 * VOL_PIX_MM

    # Ray-AABB intersection for each pixel
    src_np = src                                         # [3]
    ox, oy, oz = src_np[0], src_np[1], src_np[2]

    dx = ray_dir[..., 0]   # [DET_H, DET_W]
    dy = ray_dir[..., 1]
    dz = ray_dir[..., 2]

    def slab(orig, d, lo, hi):
        inv = cp.where(cp.abs(d) > 1e-9, 1.0 / d, cp.sign(d) * 1e9)
        t1 = (lo - orig) * inv
        t2 = (hi - orig) * inv
        return cp.minimum(t1, t2), cp.maximum(t1, t2)

    tx0, tx1 = slab(ox, dx, -half_x, half_x)
    ty0, ty1 = slab(oy, dy, -half_y, half_y)
    tz0, tz1 = slab(oz, dz, -half_z, half_z)

    tmin = cp.maximum(cp.maximum(tx0, ty0), tz0)
    tmax = cp.minimum(cp.minimum(tx1, ty1), tz1)
    tmin = cp.maximum(tmin, 0.0)

    # Ray march
    proj_out = cp.zeros((DET_H, DET_W), dtype=cp.float32)
    hit_mask = tmax > tmin

    n_steps_max = int((2.0 * max(half_x, half_y, half_z)) / RAY_STEP_MM) + 1

    for step in range(n_steps_max):
        t = tmin + (step + 0.5) * RAY_STEP_MM
        active = hit_mask & (t < tmax)
        if not cp.any(active):
            break

        # World positions
        wx = ox + t * dx
        wy = oy + t * dy
        wz = oz + t * dz

        # Convert to voxel indices (continuous)
        vx_f = (wx + half_x) / VOL_PIX_MM - 0.5
        vy_f = (wy + half_y) / VOL_PIX_MM - 0.5
        vz_f = (wz + half_z) / VOL_PIX_MM - 0.5

        # Clip to valid range
        in_vol = (active &
                  (vx_f >= 0) & (vx_f <= VOL_NX - 1) &
                  (vy_f >= 0) & (vy_f <= VOL_NY - 1) &
                  (vz_f >= 0) & (vz_f <= VOL_NZ - 1))

        if not cp.any(in_vol):
            continue

        # Trilinear interpolation via map_coordinates
        # volume is [NZ, NY, NX] so coords order is z, y, x
        coords = cp.stack([
            cp.where(in_vol, vz_f, 0.0),
            cp.where(in_vol, vy_f, 0.0),
            cp.where(in_vol, vx_f, 0.0),
        ], axis=0)   # [3, DET_H, DET_W]

        interp = map_coordinates(volume_gpu, coords, order=1, mode='constant', cval=0.0)
        proj_out += cp.where(in_vol, interp * RAY_STEP_MM, 0.0)

    return proj_out

# ─── Back projector ────────────────────────────────────────────────────────────

def back_project(diff_gpu, geom, volume_shape):
    """
    Voxel-driven back projection: for each voxel, find where it projects
    onto the detector and accumulate the diff value.
    diff_gpu: CuPy array [DET_H, DET_W]
    Returns (update [NZ, NY, NX], weight [NZ, NY, NX])
    """
    src   = cp.array(geom['src'],   dtype=cp.float32)
    det_c = cp.array(geom['det_c'], dtype=cp.float32)
    u     = cp.array(geom['u'],     dtype=cp.float32)
    v_ax  = cp.array(geom['v'],     dtype=cp.float32)

    nz, ny, nx = volume_shape

    half_x = nx * 0.5 * VOL_PIX_MM
    half_y = ny * 0.5 * VOL_PIX_MM
    half_z = nz * 0.5 * VOL_PIX_MM

    # Voxel centres in world coords
    xi = cp.arange(nx, dtype=cp.float32)
    yi = cp.arange(ny, dtype=cp.float32)
    zi = cp.arange(nz, dtype=cp.float32)

    wx = (xi + 0.5) * VOL_PIX_MM - half_x   # [NX]
    wy = (yi + 0.5) * VOL_PIX_MM - half_y   # [NY]
    wz = (zi + 0.5) * VOL_PIX_MM - half_z   # [NZ]

    # Build 3D grids: [NZ, NY, NX]
    WX, WY, WZ = cp.meshgrid(wx, wy, wz, indexing='ij')   # [NX, NY, NZ]
    WX = WX.transpose(2, 1, 0)   # [NZ, NY, NX]
    WY = WY.transpose(2, 1, 0)
    WZ = WZ.transpose(2, 1, 0)

    # Vector from source to voxel
    rx = WX - src[0]
    ry = WY - src[1]
    rz = WZ - src[2]

    # Beam direction (source to detector centre, normalised)
    beam = det_c - src
    beam_len = float(cp.linalg.norm(beam))
    beam_n = beam / beam_len

    # t to reach detector plane: n·(p - det_c) = 0 → t = n·(det_c - src) / n·r
    denom = beam_n[0]*rx + beam_n[1]*ry + beam_n[2]*rz
    t_det = beam_len / cp.where(cp.abs(denom) > 1e-9, denom, 1e-9)

    # Hit point on detector plane
    hx = src[0] + t_det * rx - det_c[0]
    hy = src[1] + t_det * ry - det_c[1]
    hz = src[2] + t_det * rz - det_c[2]

    # Detector pixel coords (continuous)
    pu_f = (hx*u[0] + hy*u[1] + hz*u[2]) / DET_PIX_MM + DET_W * 0.5 - 0.5
    pv_f = (hx*v_ax[0] + hy*v_ax[1] + hz*v_ax[2]) / DET_PIX_MM + DET_H * 0.5 - 0.5

    # Mask valid projections (t>0 and within detector)
    valid = ((t_det > 0) &
             (pu_f >= 0) & (pu_f <= DET_W - 1) &
             (pv_f >= 0) & (pv_f <= DET_H - 1))

    # Bilinear interpolation on diff image
    coords_det = cp.stack([
        cp.where(valid, pv_f, 0.0),
        cp.where(valid, pu_f, 0.0),
    ], axis=0)

    interp_diff = map_coordinates(diff_gpu, coords_det, order=1,
                                  mode='constant', cval=0.0)

    update = cp.where(valid, interp_diff * RAY_STEP_MM, 0.0)
    weight = cp.where(valid, RAY_STEP_MM ** 2, 0.0)

    return update, weight

# ─── SIRT main loop ────────────────────────────────────────────────────────────

def sirt_reconstruct(projs_att, geoms, num_iters, out_dir):
    """
    projs_att: numpy array [N_angles, DET_H, DET_W] of attenuation values
    geoms:     list of geometry dicts
    """
    vol_shape = (VOL_NZ, VOL_NY, VOL_NX)
    volume = cp.zeros(vol_shape, dtype=cp.float32)

    n_angles = len(geoms)
    rss_history = []

    for iteration in range(num_iters):
        vol_update  = cp.zeros(vol_shape, dtype=cp.float32)
        vol_weight  = cp.zeros(vol_shape, dtype=cp.float32)
        total_rss   = 0.0

        pbar = tqdm(range(n_angles),
                    desc=f"Iter {iteration+1:3d}/{num_iters}",
                    leave=False)

        for ai in pbar:
            meas_gpu = cp.array(projs_att[ai], dtype=cp.float32)

            # Forward project
            fwd = forward_project(volume, geoms[ai])

            # Residual
            diff = meas_gpu - fwd
            total_rss += float(cp.sum(diff ** 2))

            # Back project residual
            update, weight = back_project(diff, geoms[ai], vol_shape)
            vol_update += update
            vol_weight += weight

        # Apply SIRT update
        safe_weight = cp.where(vol_weight > 1e-8, vol_weight, 1.0)
        volume += RELAXATION * vol_update / safe_weight
        volume  = cp.maximum(volume, 0.0)   # non-negativity constraint

        rss_history.append(total_rss)
        print(f"  Iter {iteration+1:3d}/{num_iters}  RSS = {total_rss:.4e}")

        # Save intermediate every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == num_iters - 1:
            print(f"  → Saving volume after iter {iteration+1}...")
            save_volume(volume, out_dir)

    return cp.asnumpy(volume), rss_history

# ─── Save / load volume ────────────────────────────────────────────────────────

def save_volume(volume_gpu, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vol_np = cp.asnumpy(volume_gpu) if GPU_AVAILABLE else volume_gpu
    nz = vol_np.shape[0]
    for iz in range(nz):
        path = os.path.join(out_dir, f"slice_{iz:04d}.tif")
        tifffile.imwrite(path, vol_np[iz])
    print(f"    Saved {nz} slices to {out_dir}/")

def show_results(vol, rss_history):
    nz, ny, nx = vol.shape
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(vol[nz//2], cmap='gray', vmin=0)
    axes[0, 0].set_title(f'XY plane — centre slice (z={nz//2})')
    axes[0, 0].set_xlabel('X (px)'); axes[0, 0].set_ylabel('Y (px)')

    axes[0, 1].imshow(vol[:, ny//2, :], cmap='gray', vmin=0)
    axes[0, 1].set_title(f'XZ plane (y={ny//2})')
    axes[0, 1].set_xlabel('X (px)'); axes[0, 1].set_ylabel('Z (px)')

    axes[1, 0].imshow(vol[:, :, nx//2], cmap='gray', vmin=0)
    axes[1, 0].set_title(f'YZ plane (x={nx//2})')
    axes[1, 0].set_xlabel('Y (px)'); axes[1, 0].set_ylabel('Z (px)')

    axes[1, 1].plot(range(1, len(rss_history)+1), rss_history, 'b-o', markersize=3)
    axes[1, 1].set_title('SIRT Convergence')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Residual Sum of Squares')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.suptitle('CT Reconstruction — SIRT', fontsize=14)
    plt.tight_layout()
    plt.show()

# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CT SIRT Reconstruction (GPU/CuPy)')
    parser.add_argument('--proj_dir', required=True,
                        help='Directory with z0.tif, z24.tif, ..., z336.tif')
    parser.add_argument('--out_dir',  required=True,
                        help='Output directory for reconstructed slices')
    parser.add_argument('--iters',    type=int, default=50,
                        help='Number of SIRT iterations (default: 50)')
    parser.add_argument('--I0',       type=float, default=None,
                        help='Air value I0 for Beer-Lambert (default: auto from data)')
    parser.add_argument('--show',     action='store_true',
                        help='Show result plots after reconstruction')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load raw projections ──
    print(f"\nLoading {len(ANGLES_DEG)} projections from {args.proj_dir}...")
    raw = load_projections(args.proj_dir, ANGLES_DEG)
    print(f"  Loaded: shape={raw.shape}, dtype={raw.dtype}")
    print(f"  Raw range: min={raw.min():.1f}  max={raw.max():.1f}")

    # ── Estimate or use provided I0 ──
    if args.I0 is not None:
        I0 = args.I0
        print(f"  Using provided I0 = {I0:.1f}")
    else:
        I0 = float(np.percentile(raw, 99.5))
        print(f"  Auto-estimated I0 (99.5th percentile) = {I0:.1f}")

    # ── Beer-Lambert conversion ──
    projs_att = to_attenuation(raw, I0)
    print(f"  Attenuation range: min={projs_att.min():.4f}  max={projs_att.max():.4f}")
    print(f"  (air ≈ 0, material > 0 — expected max ~{-np.log(400/I0):.2f})")

    # ── Build geometry ──
    geoms = build_geometry(ANGLES_DEG)
    print(f"\nGeometry (first 3 angles as check):")
    for g in geoms[:3]:
        print(f"  θ={g['angle']:3d}°  src={g['src'].round(2)}  det={g['det_c'].round(2)}")

    # ── Reconstruct ──
    print(f"\nStarting SIRT reconstruction ({args.iters} iterations)...")
    print(f"  Volume: {VOL_NX}×{VOL_NY}×{VOL_NZ} voxels @ {VOL_PIX_MM} mm")
    print(f"  Ray step: {RAY_STEP_MM} mm,  Relaxation: {RELAXATION}")

    vol, rss = sirt_reconstruct(projs_att, geoms, args.iters, args.out_dir)

    print(f"\nDone! Volume stats:")
    print(f"  min={vol.min():.4f}  max={vol.max():.4f}  mean={vol.mean():.6f}")
    print(f"  Output slices: {args.out_dir}/slice_0000.tif ... slice_{VOL_NZ-1:04d}.tif")

    if args.show:
        show_results(vol, rss)

if __name__ == '__main__':
    main()
