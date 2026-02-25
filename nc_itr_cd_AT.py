"""
ct_recon_sirt.py — GPU-accelerated CT Reconstruction (SIRT) using CuPy
                   Supports arbitrary projection trajectories via Excel input

Setup:
    Cylinder centre:  (0, 0, 0)
    Source centre:    (0, -830.71, 0)  at all-zero angles
    Detector centre:  (0, +332.29, 0)  at all-zero angles
    Rotation order:   Z first, then X, then Y (applied to initial pose)
    Detector:         350x350 pixels, 0.4 mm/pixel
    Projections:      32-bit float TIFs, air ~2000, material ~400

Excel input format (one row per projection):
    | filename                | angle_z | angle_x | angle_y |
    |-------------------------|---------|---------|---------|
    | z10_x20_y30.tif         |  10     |  20     |  30     |
    | z0_x0_y0.tif            |   0     |   0     |   0     |
    | z45_x15_y0.tif          |  45     |  15     |   0     |

Install dependencies:
    pip install cupy-cuda12x tifffile numpy matplotlib tqdm openpyxl pandas

Run:
    python ct_recon_sirt.py --proj_dir ./projections
                            --excel    ./projections/angles.xlsx
                            --out_dir  ./output
                            --iters    50
"""

import os
import argparse
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── GPU imports ───────────────────────────────────────────────────────────────

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

SAD         = 830.71   # mm, source to isocentre
DDD         = 332.29   # mm, detector to isocentre
SDD         = SAD + DDD

DET_W       = 350      # detector pixels (columns)
DET_H       = 350      # detector pixels (rows)
DET_PIX_MM  = 0.4      # mm per detector pixel

VOL_NX      = 300      # reconstruction volume voxels
VOL_NY      = 300
VOL_NZ      = 300
VOL_PIX_MM  = 0.4      # mm per voxel

RELAXATION  = 0.05     # SIRT step size — start conservative, raise if converging slowly
RAY_STEP_MM = 0.2      # ray marching step in mm

# ─── Rotation matrix utilities ─────────────────────────────────────────────────

def rot_x(deg):
    """3x3 rotation matrix around the X axis by `deg` degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=np.float64)

def rot_y(deg):
    """3x3 rotation matrix around the Y axis by `deg` degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c],
    ], dtype=np.float64)

def rot_z(deg):
    """3x3 rotation matrix around the Z axis by `deg` degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1],
    ], dtype=np.float64)

def combined_rotation(angle_z, angle_x, angle_y):
    """
    Combined rotation matrix applying Z rotation first, then X, then Y.
    This matches the intuitive order: primary orbit (Z), then tilt (X),
    then secondary tilt (Y).

    Written as matrix multiplication: R_total = R_y @ R_x @ R_z
    A column vector is rotated as: v' = R_total @ v
    """
    return rot_y(angle_y) @ rot_x(angle_x) @ rot_z(angle_z)

# ─── Geometry builder ──────────────────────────────────────────────────────────

def build_geometry_from_angles(angle_z, angle_x, angle_y, filename=''):
    """
    Build a single projection geometry dict from three Euler angles (degrees).

    Initial pose (all angles = 0):
        source    at (0, -SAD, 0)
        detector  at (0, +DDD, 0)
        u_axis    = (1, 0, 0)   — detector horizontal = world X
        v_axis    = (0, 0, 1)   — detector vertical   = world Z

    The combined rotation matrix is applied to all four vectors to get
    the actual world-space geometry for this projection.
    """
    # Reference positions and axes at zero rotation
    src0 = np.array([0.0, -SAD, 0.0])
    det0 = np.array([0.0,  DDD, 0.0])
    u0   = np.array([1.0,  0.0, 0.0])   # detector horizontal axis
    v0   = np.array([0.0,  0.0, 1.0])   # detector vertical axis

    # Apply combined rotation
    R     = combined_rotation(angle_z, angle_x, angle_y)
    src   = R @ src0
    det_c = R @ det0
    u_ax  = R @ u0
    v_ax  = R @ v0

    # Renormalise to guard against floating point drift
    u_ax = u_ax / np.linalg.norm(u_ax)
    v_ax = v_ax / np.linalg.norm(v_ax)

    return {
        'filename': filename,
        'angle_z':  angle_z,
        'angle_x':  angle_x,
        'angle_y':  angle_y,
        'src':      src.astype(np.float32),
        'det_c':    det_c.astype(np.float32),
        'u':        u_ax.astype(np.float32),
        'v':        v_ax.astype(np.float32),
    }

def build_geometry_from_excel(excel_path):
    """
    Read the Excel file and return a list of geometry dicts,
    one per row, in the order they appear in the Excel.

    Required columns (case-insensitive): filename, angle_z, angle_x, angle_y
    """
    df = pd.read_excel(excel_path)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {'filename', 'angle_z', 'angle_x', 'angle_y'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required: filename, angle_z, angle_x, angle_y"
        )

    geoms = []
    for _, row in df.iterrows():
        g = build_geometry_from_angles(
            angle_z  = float(row['angle_z']),
            angle_x  = float(row['angle_x']),
            angle_y  = float(row['angle_y']),
            filename = str(row['filename']).strip(),
        )
        geoms.append(g)

    print(f"  Loaded {len(geoms)} projection entries from Excel.")
    return geoms

# ─── Beer-Lambert conversion ───────────────────────────────────────────────────

def to_attenuation(projs_raw, I0):
    """
    Convert raw detector values to attenuation via Beer-Lambert law.
    Raw: air ~2000 (bright = unattenuated), material ~400 (dark = attenuated)
    Output: air -> 0.0, material -> positive (~1.6 for your data)
    """
    projs = np.clip(projs_raw.astype(np.float32), 1.0, None)
    ratio = np.clip(projs / I0, 1e-7, 1.0)
    return -np.log(ratio)

# ─── Load projections ──────────────────────────────────────────────────────────

def load_projections(proj_dir, geoms):
    """
    Load each TIF file named in the geometry list.
    Files are loaded in the same order as the geometry list (= Excel row order).
    Returns numpy array [N_projections, DET_H, DET_W].
    """
    projs = []
    for g in geoms:
        path = os.path.join(proj_dir, g['filename'])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Projection file not found: {path}\n"
                f"  (listed in Excel as '{g['filename']}')"
            )
        img = tifffile.imread(path).astype(np.float32)
        if img.shape != (DET_H, DET_W):
            raise ValueError(
                f"Shape mismatch in {g['filename']}: "
                f"expected ({DET_H},{DET_W}), got {img.shape}"
            )
        projs.append(img)

    return np.stack(projs, axis=0)   # [N, DET_H, DET_W]

# ─── Forward projector ─────────────────────────────────────────────────────────

def forward_project(volume_gpu, geom):
    """
    Cone-beam forward projection via ray marching + trilinear interpolation.
    Geometry-agnostic: works for any source/detector position and orientation.

    volume_gpu : CuPy array [NZ, NY, NX]
    geom       : geometry dict with src, det_c, u, v
    Returns    : CuPy array [DET_H, DET_W]
    """
    src   = cp.array(geom['src'],   dtype=cp.float32)
    det_c = cp.array(geom['det_c'], dtype=cp.float32)
    u     = cp.array(geom['u'],     dtype=cp.float32)
    v     = cp.array(geom['v'],     dtype=cp.float32)

    # Pixel offsets from detector centre in mm
    col_idx = cp.arange(DET_W, dtype=cp.float32) - DET_W / 2.0 + 0.5
    row_idx = cp.arange(DET_H, dtype=cp.float32) - DET_H / 2.0 + 0.5
    col_mm  = col_idx * DET_PIX_MM
    row_mm  = row_idx * DET_PIX_MM

    # World position of every detector pixel [DET_H, DET_W, 3]
    pix_pos = (det_c[None, None, :]
               + col_mm[None, :, None] * u[None, None, :]
               + row_mm[:, None, None] * v[None, None, :])

    # Unit ray directions source -> pixel [DET_H, DET_W, 3]
    ray_dir = pix_pos - src[None, None, :]
    ray_len = cp.linalg.norm(ray_dir, axis=-1, keepdims=True)
    ray_dir = ray_dir / ray_len

    half_x = VOL_NX * 0.5 * VOL_PIX_MM
    half_y = VOL_NY * 0.5 * VOL_PIX_MM
    half_z = VOL_NZ * 0.5 * VOL_PIX_MM

    ox = float(src[0]); oy = float(src[1]); oz = float(src[2])
    dx = ray_dir[..., 0]
    dy = ray_dir[..., 1]
    dz = ray_dir[..., 2]

    def slab(orig, d, lo, hi):
        inv = cp.where(cp.abs(d) > 1e-9, 1.0 / d, cp.sign(d) * 1e9)
        t1  = (lo - orig) * inv
        t2  = (hi - orig) * inv
        return cp.minimum(t1, t2), cp.maximum(t1, t2)

    tx0, tx1 = slab(ox, dx, -half_x, half_x)
    ty0, ty1 = slab(oy, dy, -half_y, half_y)
    tz0, tz1 = slab(oz, dz, -half_z, half_z)

    tmin = cp.maximum(cp.maximum(tx0, ty0), tz0)
    tmax = cp.minimum(cp.minimum(tx1, ty1), tz1)
    tmin = cp.maximum(tmin, 0.0)

    proj_out = cp.zeros((DET_H, DET_W), dtype=cp.float32)
    hit_mask = tmax > tmin
    n_steps  = int((2.0 * max(half_x, half_y, half_z)) / RAY_STEP_MM) + 1

    for step in range(n_steps):
        t      = tmin + (step + 0.5) * RAY_STEP_MM
        active = hit_mask & (t < tmax)
        if not cp.any(active):
            break

        wx = ox + t * dx
        wy = oy + t * dy
        wz = oz + t * dz

        vx_f = (wx + half_x) / VOL_PIX_MM - 0.5
        vy_f = (wy + half_y) / VOL_PIX_MM - 0.5
        vz_f = (wz + half_z) / VOL_PIX_MM - 0.5

        in_vol = (active &
                  (vx_f >= 0) & (vx_f <= VOL_NX - 1) &
                  (vy_f >= 0) & (vy_f <= VOL_NY - 1) &
                  (vz_f >= 0) & (vz_f <= VOL_NZ - 1))

        if not cp.any(in_vol):
            continue

        # Volume is [NZ, NY, NX] so coordinate order is z, y, x
        coords = cp.stack([
            cp.where(in_vol, vz_f, 0.0),
            cp.where(in_vol, vy_f, 0.0),
            cp.where(in_vol, vx_f, 0.0),
        ], axis=0)

        interp   = map_coordinates(volume_gpu, coords, order=1, mode='constant', cval=0.0)
        proj_out += cp.where(in_vol, interp * RAY_STEP_MM, 0.0)

    return proj_out

# ─── Back projector ────────────────────────────────────────────────────────────

def back_project(diff_gpu, geom, volume_shape):
    """
    Voxel-driven back projection.
    For each voxel, find where it projects onto the detector and accumulate
    the residual diff value. Geometry-agnostic.

    diff_gpu     : CuPy array [DET_H, DET_W]
    geom         : geometry dict
    volume_shape : (NZ, NY, NX)
    Returns      : (update [NZ, NY, NX], weight [NZ, NY, NX])
    """
    src   = cp.array(geom['src'],   dtype=cp.float32)
    det_c = cp.array(geom['det_c'], dtype=cp.float32)
    u     = cp.array(geom['u'],     dtype=cp.float32)
    v_ax  = cp.array(geom['v'],     dtype=cp.float32)

    nz, ny, nx = volume_shape
    half_x = nx * 0.5 * VOL_PIX_MM
    half_y = ny * 0.5 * VOL_PIX_MM
    half_z = nz * 0.5 * VOL_PIX_MM

    xi = (cp.arange(nx, dtype=cp.float32) + 0.5) * VOL_PIX_MM - half_x
    yi = (cp.arange(ny, dtype=cp.float32) + 0.5) * VOL_PIX_MM - half_y
    zi = (cp.arange(nz, dtype=cp.float32) + 0.5) * VOL_PIX_MM - half_z

    WX, WY, WZ = cp.meshgrid(xi, yi, zi, indexing='ij')  # [NX, NY, NZ]
    WX = WX.transpose(2, 1, 0)   # -> [NZ, NY, NX]
    WY = WY.transpose(2, 1, 0)
    WZ = WZ.transpose(2, 1, 0)

    rx = WX - src[0]
    ry = WY - src[1]
    rz = WZ - src[2]

    beam     = det_c - src
    beam_len = float(cp.linalg.norm(beam))
    beam_n   = beam / beam_len

    denom = beam_n[0]*rx + beam_n[1]*ry + beam_n[2]*rz
    t_det = beam_len / cp.where(cp.abs(denom) > 1e-9, denom, 1e-9)

    hx = src[0] + t_det * rx - det_c[0]
    hy = src[1] + t_det * ry - det_c[1]
    hz = src[2] + t_det * rz - det_c[2]

    pu_f = (hx*u[0]    + hy*u[1]    + hz*u[2])    / DET_PIX_MM + DET_W * 0.5 - 0.5
    pv_f = (hx*v_ax[0] + hy*v_ax[1] + hz*v_ax[2]) / DET_PIX_MM + DET_H * 0.5 - 0.5

    valid = ((t_det > 0) &
             (pu_f >= 0) & (pu_f <= DET_W - 1) &
             (pv_f >= 0) & (pv_f <= DET_H - 1))

    coords_det = cp.stack([
        cp.where(valid, pv_f, 0.0),
        cp.where(valid, pu_f, 0.0),
    ], axis=0)

    interp_diff = map_coordinates(diff_gpu, coords_det, order=1,
                                  mode='constant', cval=0.0)

    update = cp.where(valid, interp_diff * RAY_STEP_MM, 0.0)
    weight = cp.where(valid, RAY_STEP_MM ** 2,          0.0)

    return update, weight

# ─── SIRT main loop ────────────────────────────────────────────────────────────

def sirt_reconstruct(projs_att, geoms, num_iters, out_dir):
    """
    projs_att : numpy array [N_projections, DET_H, DET_W] — attenuation values
    geoms     : list of geometry dicts, same order as projs_att axis 0
    """
    vol_shape   = (VOL_NZ, VOL_NY, VOL_NX)
    volume      = cp.zeros(vol_shape, dtype=cp.float32)
    n_projs     = len(geoms)
    rss_history = []

    for iteration in range(num_iters):
        vol_update = cp.zeros(vol_shape, dtype=cp.float32)
        vol_weight = cp.zeros(vol_shape, dtype=cp.float32)
        total_rss  = 0.0

        pbar = tqdm(range(n_projs),
                    desc=f"Iter {iteration+1:3d}/{num_iters}",
                    leave=False)

        for pi in pbar:
            meas_gpu = cp.array(projs_att[pi], dtype=cp.float32)
            fwd      = forward_project(volume, geoms[pi])
            diff     = meas_gpu - fwd
            total_rss += float(cp.sum(diff ** 2))

            update, weight = back_project(diff, geoms[pi], vol_shape)
            vol_update += update
            vol_weight += weight

        safe_weight = cp.where(vol_weight > 1e-8, vol_weight, 1.0)
        volume     += RELAXATION * vol_update / safe_weight
        volume      = cp.maximum(volume, 0.0)

        rss_history.append(total_rss)
        print(f"  Iter {iteration+1:3d}/{num_iters}  RSS = {total_rss:.4e}")

        if (iteration + 1) % 10 == 0 or iteration == num_iters - 1:
            print(f"  → Saving volume after iter {iteration+1}...")
            save_volume(volume, out_dir)

    return (cp.asnumpy(volume) if GPU_AVAILABLE else volume), rss_history

# ─── Save volume ───────────────────────────────────────────────────────────────

def save_volume(volume_gpu, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vol_np = cp.asnumpy(volume_gpu) if GPU_AVAILABLE else volume_gpu
    for iz in range(vol_np.shape[0]):
        tifffile.imwrite(os.path.join(out_dir, f"slice_{iz:04d}.tif"), vol_np[iz])
    print(f"    Saved {vol_np.shape[0]} slices to {out_dir}/")

# ─── Visualisation ─────────────────────────────────────────────────────────────

def show_results(vol, rss_history, geoms):
    nz, ny, nx = vol.shape
    fig, axes  = plt.subplots(2, 2, figsize=(13, 11))

    axes[0, 0].imshow(vol[nz//2], cmap='gray', vmin=0)
    axes[0, 0].set_title(f'XY slice (z={nz//2})')

    axes[0, 1].imshow(vol[:, ny//2, :], cmap='gray', vmin=0)
    axes[0, 1].set_title(f'XZ slice (y={ny//2})')

    axes[1, 0].imshow(vol[:, :, nx//2], cmap='gray', vmin=0)
    axes[1, 0].set_title(f'YZ slice (x={nx//2})')

    axes[1, 1].plot(range(1, len(rss_history)+1), rss_history, 'b-o', markersize=3)
    axes[1, 1].set_title('SIRT Convergence')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('RSS')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.suptitle('CT Reconstruction — SIRT (Arbitrary Trajectory)', fontsize=13)
    plt.tight_layout()
    plt.show()

    # Geometry summary table
    print("\nProjection geometry summary:")
    print(f"  {'#':<4} {'File':<32} {'Z':>6} {'X':>6} {'Y':>6}  "
          f"{'Source position':>28}  {'Detector centre':>28}")
    print(f"  {'':-<4} {'':-<32} {'':-<6} {'':-<6} {'':-<6}  {'':-<28}  {'':-<28}")
    for i, g in enumerate(geoms):
        s = g['src'];   d = g['det_c']
        print(f"  {i:<4} {g['filename']:<32} {g['angle_z']:>6.1f} "
              f"{g['angle_x']:>6.1f} {g['angle_y']:>6.1f}  "
              f"({s[0]:7.1f},{s[1]:7.1f},{s[2]:7.1f})  "
              f"({d[0]:7.1f},{d[1]:7.1f},{d[2]:7.1f})")

# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CT SIRT Reconstruction — arbitrary trajectory (GPU/CuPy)')
    parser.add_argument('--proj_dir', required=True,
                        help='Directory containing projection TIF files')
    parser.add_argument('--excel',    required=True,
                        help='Excel file: columns filename, angle_z, angle_x, angle_y')
    parser.add_argument('--out_dir',  required=True,
                        help='Output directory for reconstructed volume slices')
    parser.add_argument('--iters',    type=int,   default=50)
    parser.add_argument('--I0',       type=float, default=None,
                        help='Air intensity for Beer-Lambert (default: auto 99.5th pct)')
    parser.add_argument('--show',     action='store_true',
                        help='Show orthogonal slices + convergence plot when done')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Read Excel ──
    print(f"\nReading geometry from: {args.excel}")
    geoms = build_geometry_from_excel(args.excel)

    print(f"\n  {'#':<4} {'File':<35} {'Z':>7} {'X':>7} {'Y':>7}")
    print(f"  {'':-<4} {'':-<35} {'':-<7} {'':-<7} {'':-<7}")
    for i, g in enumerate(geoms):
        print(f"  {i:<4} {g['filename']:<35} {g['angle_z']:>7.1f} "
              f"{g['angle_x']:>7.1f} {g['angle_y']:>7.1f}")

    # ── Load TIFs ──
    print(f"\nLoading {len(geoms)} projections from: {args.proj_dir}")
    raw = load_projections(args.proj_dir, geoms)
    print(f"  Array shape  : {raw.shape}")
    print(f"  Pixel range  : {raw.min():.1f} – {raw.max():.1f}")

    # ── Beer-Lambert ──
    I0 = args.I0 if args.I0 is not None else float(np.percentile(raw, 99.5))
    print(f"  I0 used      : {I0:.1f}  {'(provided)' if args.I0 else '(auto-estimated)'}")
    projs_att = to_attenuation(raw, I0)
    print(f"  Attenuation  : {projs_att.min():.4f} – {projs_att.max():.4f}")

    # ── Reconstruct ──
    print(f"\nStarting SIRT ({args.iters} iterations)...")
    print(f"  Volume     : {VOL_NX}×{VOL_NY}×{VOL_NZ} @ {VOL_PIX_MM} mm/voxel")
    print(f"  Ray step   : {RAY_STEP_MM} mm")
    print(f"  Relaxation : {RELAXATION}")

    vol, rss = sirt_reconstruct(projs_att, geoms, args.iters, args.out_dir)

    print(f"\nDone. Volume range: {vol.min():.4f} – {vol.max():.4f}")
    print(f"Slices: {args.out_dir}/slice_0000.tif … slice_{VOL_NZ-1:04d}.tif")

    if args.show:
        show_results(vol, rss, geoms)

if __name__ == '__main__':
    main()
