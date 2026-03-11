"""
ct_recon_sirt.py
GPU-accelerated CT Reconstruction using SIRT and custom CUDA kernels.

Setup:
    Cylinder centre : (0, 0, 0)
    Source centre   : (0, -830.71, 0)
    Detector centre : (0, +332.29, 0)
    Rotation        : Around Z axis, 15 angles 0,24,...,336 degrees
    Detector        : 350x350 pixels, 0.4 mm/pixel
    Projections     : 32-bit float TIFs, air ~2000, material ~400

Install:
    pip install cupy-cuda12x tifffile numpy matplotlib tqdm

Run:
    python ct_recon_sirt.py --proj_dir ./projections --out_dir ./output --iters 50
"""

import os
import math
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print("GPU :", dev["name"].decode())
    print("VRAM:", dev["totalGlobalMem"] // 1024**3, "GB")
except ImportError:
    raise RuntimeError("CuPy not found. Install: pip install cupy-cuda12x")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAD        = 830.71   # mm  source to isocentre
DDD        = 332.29   # mm  detector to isocentre

DET_W      = 350      # detector columns
DET_H      = 350      # detector rows
DET_PIX    = 0.4      # mm per detector pixel

VOL_NX     = 300      # volume voxels X
VOL_NY     = 300      # volume voxels Y
VOL_NZ     = 300      # volume voxels Z
VOL_PIX    = 0.4      # mm per voxel

RELAXATION = 0.05     # SIRT step size
STEP_MM    = 0.2      # ray march step in mm

ANGLES     = list(range(0, 360, 24))   # 0, 24, 48 ... 336


# ---------------------------------------------------------------------------
# CUDA kernel source
# No C++ lambdas, no auto keyword, plain C style throughout.
# ---------------------------------------------------------------------------

CUDA_SRC = """

/* Trilinear interpolation inside volume [NZ][NY][NX] */
__device__ float trilinear(
    const float* vol,
    int nx, int ny, int nz,
    float vx, float vy, float vz)
{
    int x0 = (int)vx;
    int y0 = (int)vy;
    int z0 = (int)vz;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    if (x0 < 0 || x1 >= nx || y0 < 0 || y1 >= ny || z0 < 0 || z1 >= nz)
        return 0.0f;

    float fx = vx - (float)x0;
    float fy = vy - (float)y0;
    float fz = vz - (float)z0;
    float _fx = 1.0f - fx;
    float _fy = 1.0f - fy;
    float _fz = 1.0f - fz;

    int stride_y = nx;
    int stride_z = nx * ny;

    float v000 = vol[z0*stride_z + y0*stride_y + x0];
    float v100 = vol[z0*stride_z + y0*stride_y + x1];
    float v010 = vol[z0*stride_z + y1*stride_y + x0];
    float v110 = vol[z0*stride_z + y1*stride_y + x1];
    float v001 = vol[z1*stride_z + y0*stride_y + x0];
    float v101 = vol[z1*stride_z + y0*stride_y + x1];
    float v011 = vol[z1*stride_z + y1*stride_y + x0];
    float v111 = vol[z1*stride_z + y1*stride_y + x1];

    return _fz * (_fy * (_fx*v000 + fx*v100) + fy * (_fx*v010 + fx*v110))
          + fz * (_fy * (_fx*v001 + fx*v101) + fy * (_fx*v011 + fx*v111));
}


/* Bilinear interpolation on detector image [DET_H][DET_W] */
__device__ float bilinear(
    const float* img,
    int iw, int ih,
    float pu, float pv)
{
    int u0 = (int)pu;
    int v0 = (int)pv;
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    if (u0 < 0 || u1 >= iw || v0 < 0 || v1 >= ih)
        return 0.0f;

    float fu = pu - (float)u0;
    float fv = pv - (float)v0;

    return (1.0f-fv) * ((1.0f-fu)*img[v0*iw+u0] + fu*img[v0*iw+u1])
          +       fv * ((1.0f-fu)*img[v1*iw+u0] + fu*img[v1*iw+u1]);
}


/* Slab test for ray-AABB intersection.
   Updates tmin and tmax in place. */
__device__ void slab_test(
    float orig, float dir,
    float lo,   float hi,
    float* tmin, float* tmax)
{
    float inv;
    if (fabsf(dir) > 1e-9f)
        inv = 1.0f / dir;
    else
        inv = (dir >= 0.0f) ? 1e9f : -1e9f;

    float t1 = (lo - orig) * inv;
    float t2 = (hi - orig) * inv;

    if (t1 > t2) {
        float tmp = t1; t1 = t2; t2 = tmp;
    }

    if (t1 > *tmin) *tmin = t1;
    if (t2 < *tmax) *tmax = t2;
}


/* Forward projector.
   One CUDA thread per detector pixel.
   The ray march loop runs entirely inside the thread. */
extern "C" __global__ void forward_project_kernel(
    const float* volume,
    float*       proj_out,
    float src_x,   float src_y,   float src_z,
    float det_cx,  float det_cy,  float det_cz,
    float ux,      float uy,      float uz,
    float vx,      float vy,      float vz,
    int   det_w,   int   det_h,   float det_pix,
    int   nx,      int   ny,      int   nz,
    float vox_mm,  float step_mm)
{
    int pu = blockIdx.x * blockDim.x + threadIdx.x;
    int pv = blockIdx.y * blockDim.y + threadIdx.y;
    if (pu >= det_w || pv >= det_h) return;

    /* Detector pixel world position */
    float du = ((float)pu - (float)det_w * 0.5f + 0.5f) * det_pix;
    float dv = ((float)pv - (float)det_h * 0.5f + 0.5f) * det_pix;

    float px = det_cx + du*ux + dv*vx;
    float py = det_cy + du*uy + dv*vy;
    float pz = det_cz + du*uz + dv*vz;

    /* Unit ray direction */
    float rdx = px - src_x;
    float rdy = py - src_y;
    float rdz = pz - src_z;
    float rlen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);
    rdx /= rlen;
    rdy /= rlen;
    rdz /= rlen;

    /* Volume half extents */
    float hx = (float)nx * 0.5f * vox_mm;
    float hy = (float)ny * 0.5f * vox_mm;
    float hz = (float)nz * 0.5f * vox_mm;

    /* Ray-AABB intersection */
    float tmin = 0.0f;
    float tmax = 1e9f;
    slab_test(src_x, rdx, -hx, hx, &tmin, &tmax);
    slab_test(src_y, rdy, -hy, hy, &tmin, &tmax);
    slab_test(src_z, rdz, -hz, hz, &tmin, &tmax);

    float acc = 0.0f;

    if (tmax > tmin) {
        int n_steps = (int)((tmax - tmin) / step_mm) + 1;
        int s;
        for (s = 0; s < n_steps; s++) {
            float t = tmin + ((float)s + 0.5f) * step_mm;
            if (t >= tmax) break;

            float wx = src_x + t * rdx;
            float wy = src_y + t * rdy;
            float wz = src_z + t * rdz;

            /* World mm to continuous voxel index */
            float vxf = (wx + hx) / vox_mm - 0.5f;
            float vyf = (wy + hy) / vox_mm - 0.5f;
            float vzf = (wz + hz) / vox_mm - 0.5f;

            acc += trilinear(volume, nx, ny, nz, vxf, vyf, vzf) * step_mm;
        }
    }

    proj_out[pv * det_w + pu] = acc;
}


/* Back projector.
   One CUDA thread per voxel.
   Finds where the voxel projects onto the detector
   and accumulates the residual value. */
extern "C" __global__ void back_project_kernel(
    float*       vol_update,
    float*       vol_weight,
    const float* diff,
    float src_x,   float src_y,   float src_z,
    float det_cx,  float det_cy,  float det_cz,
    float ux,      float uy,      float uz,
    float vx,      float vy,      float vz,
    int   det_w,   int   det_h,   float det_pix,
    int   nx,      int   ny,      int   nz,
    float vox_mm,  float step_mm)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    float hx = (float)nx * 0.5f * vox_mm;
    float hy = (float)ny * 0.5f * vox_mm;
    float hz = (float)nz * 0.5f * vox_mm;

    /* Voxel centre in world coords */
    float wx = ((float)ix + 0.5f) * vox_mm - hx;
    float wy = ((float)iy + 0.5f) * vox_mm - hy;
    float wz = ((float)iz + 0.5f) * vox_mm - hz;

    /* Vector from source to voxel */
    float rx = wx - src_x;
    float ry = wy - src_y;
    float rz = wz - src_z;

    /* Beam direction (source to detector centre), normalised */
    float bx = det_cx - src_x;
    float by = det_cy - src_y;
    float bz = det_cz - src_z;
    float blen = sqrtf(bx*bx + by*by + bz*bz);
    float bnx = bx / blen;
    float bny = by / blen;
    float bnz = bz / blen;

    /* t to reach detector plane */
    float denom = bnx*rx + bny*ry + bnz*rz;
    if (fabsf(denom) < 1e-9f) return;
    float t = blen / denom;
    if (t <= 0.0f) return;

    /* Hit point relative to detector centre */
    float hpx = src_x + t*rx - det_cx;
    float hpy = src_y + t*ry - det_cy;
    float hpz = src_z + t*rz - det_cz;

    /* Continuous detector pixel coords */
    float pu = (hpx*ux + hpy*uy + hpz*uz) / det_pix + (float)det_w*0.5f - 0.5f;
    float pv = (hpx*vx + hpy*vy + hpz*vz) / det_pix + (float)det_h*0.5f - 0.5f;

    if (pu < 0.0f || pu > (float)(det_w-1) ||
        pv < 0.0f || pv > (float)(det_h-1)) return;

    float d = bilinear(diff, det_w, det_h, pu, pv);

    int vidx = iz*ny*nx + iy*nx + ix;
    atomicAdd(&vol_update[vidx], d * step_mm);
    atomicAdd(&vol_weight[vidx], step_mm * step_mm);
}


/* SIRT update + non-negativity clamp, fused in one kernel.
   One thread per voxel (flat index). */
extern "C" __global__ void sirt_update_kernel(
    float*       volume,
    const float* vol_update,
    const float* vol_weight,
    float        relaxation,
    int          n_voxels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_voxels) return;

    float w = vol_weight[i];
    if (w > 1e-8f)
        volume[i] += relaxation * vol_update[i] / w;

    if (volume[i] < 0.0f)
        volume[i] = 0.0f;
}

"""


# ---------------------------------------------------------------------------
# Compile CUDA kernels once at startup
# ---------------------------------------------------------------------------

print("Compiling CUDA kernels...", end=" ", flush=True)
_mod           = cp.RawModule(code=CUDA_SRC, options=("-O3",))
_fwd_kernel    = _mod.get_function("forward_project_kernel")
_bwd_kernel    = _mod.get_function("back_project_kernel")
_update_kernel = _mod.get_function("sirt_update_kernel")
print("done.")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def build_geometry(angles_deg):
    geoms = []
    for theta_deg in angles_deg:
        theta = np.radians(theta_deg)
        s = float(np.sin(theta))
        c = float(np.cos(theta))
        geoms.append({
            "angle": theta_deg,
            "src":   np.array([-SAD*s, -SAD*c,  0.0], dtype=np.float32),
            "det_c": np.array([ DDD*s,  DDD*c,  0.0], dtype=np.float32),
            "u":     np.array([ c,     -s,       0.0], dtype=np.float32),
            "v":     np.array([ 0.0,    0.0,     1.0], dtype=np.float32),
        })
    return geoms


# ---------------------------------------------------------------------------
# Beer-Lambert conversion
# ---------------------------------------------------------------------------

def to_attenuation(raw, I0):
    clipped = np.clip(raw.astype(np.float32), 1.0, None)
    ratio   = np.clip(clipped / I0, 1e-7, 1.0)
    return -np.log(ratio)


# ---------------------------------------------------------------------------
# Load projections
# ---------------------------------------------------------------------------

def load_projections(proj_dir, angles_deg):
    projs = []
    for a in angles_deg:
        path = os.path.join(proj_dir, "z{}.tif".format(a))
        if not os.path.exists(path):
            raise FileNotFoundError("Missing projection: {}".format(path))
        img = tifffile.imread(path).astype(np.float32)
        if img.shape != (DET_H, DET_W):
            raise ValueError("Shape mismatch in {}: expected ({},{}), got {}".format(
                path, DET_H, DET_W, img.shape))
        projs.append(img)
    return np.stack(projs, axis=0)


# ---------------------------------------------------------------------------
# Helpers: pack geometry into kernel args
# ---------------------------------------------------------------------------

def geom_args(g):
    s = g["src"]
    d = g["det_c"]
    u = g["u"]
    v = g["v"]
    return (
        np.float32(s[0]), np.float32(s[1]), np.float32(s[2]),
        np.float32(d[0]), np.float32(d[1]), np.float32(d[2]),
        np.float32(u[0]), np.float32(u[1]), np.float32(u[2]),
        np.float32(v[0]), np.float32(v[1]), np.float32(v[2]),
    )


# ---------------------------------------------------------------------------
# Kernel launchers
# ---------------------------------------------------------------------------

def launch_forward(volume, geom, proj_buf):
    """
    Forward project volume into proj_buf.
    volume   : cp.ndarray [NZ, NY, NX]
    proj_buf : cp.ndarray [DET_H, DET_W]  pre-zeroed
    """
    block = (16, 16, 1)
    grid  = (math.ceil(DET_W / 16), math.ceil(DET_H / 16), 1)

    args = (
        volume, proj_buf,
    ) + geom_args(geom) + (
        np.int32(DET_W),  np.int32(DET_H),  np.float32(DET_PIX),
        np.int32(VOL_NX), np.int32(VOL_NY), np.int32(VOL_NZ),
        np.float32(VOL_PIX), np.float32(STEP_MM),
    )
    _fwd_kernel(grid, block, args)


def launch_back(diff, geom, vol_update, vol_weight):
    """
    Back project diff into vol_update and vol_weight.
    diff       : cp.ndarray [DET_H, DET_W]
    vol_update : cp.ndarray [NZ, NY, NX]  accumulates
    vol_weight : cp.ndarray [NZ, NY, NX]  accumulates
    """
    bx, by, bz = 8, 8, 8
    grid = (
        math.ceil(VOL_NX / bx),
        math.ceil(VOL_NY / by),
        math.ceil(VOL_NZ / bz),
    )

    args = (
        vol_update, vol_weight, diff,
    ) + geom_args(geom) + (
        np.int32(DET_W),  np.int32(DET_H),  np.float32(DET_PIX),
        np.int32(VOL_NX), np.int32(VOL_NY), np.int32(VOL_NZ),
        np.float32(VOL_PIX), np.float32(STEP_MM),
    )
    _bwd_kernel(grid, (bx, by, bz), args)


def launch_sirt_update(volume, vol_update, vol_weight):
    """Fused update + clamp kernel."""
    n     = volume.size
    block = (256,)
    grid  = (math.ceil(n / 256),)
    args  = (
        volume, vol_update, vol_weight,
        np.float32(RELAXATION), np.int32(n),
    )
    _update_kernel(grid, block, args)


# ---------------------------------------------------------------------------
# SIRT reconstruction loop
# ---------------------------------------------------------------------------

def sirt_reconstruct(projs_att, geoms, num_iters, out_dir):
    """
    projs_att : numpy array [N, DET_H, DET_W]
    geoms     : list of geometry dicts, same order as projs_att
    """
    vol_shape = (VOL_NZ, VOL_NY, VOL_NX)
    n_projs   = len(geoms)

    # Upload all projections to GPU once before the loop
    print("  Uploading projections to GPU...")
    projs_gpu = [cp.asarray(projs_att[i]) for i in range(n_projs)]

    # Persistent GPU buffers
    volume     = cp.zeros(vol_shape, dtype=cp.float32)
    vol_update = cp.zeros(vol_shape, dtype=cp.float32)
    vol_weight = cp.zeros(vol_shape, dtype=cp.float32)
    proj_buf   = cp.zeros((DET_H, DET_W), dtype=cp.float32)
    diff_buf   = cp.zeros((DET_H, DET_W), dtype=cp.float32)

    rss_history = []

    for iteration in range(num_iters):

        vol_update.fill(0.0)
        vol_weight.fill(0.0)
        total_rss = 0.0

        for pi in tqdm(range(n_projs),
                       desc="Iter {:3d}/{}".format(iteration + 1, num_iters),
                       leave=False):

            meas = projs_gpu[pi]

            # Forward project
            proj_buf.fill(0.0)
            launch_forward(volume, geoms[pi], proj_buf)

            # Residual
            cp.subtract(meas, proj_buf, out=diff_buf)
            total_rss += float(cp.dot(diff_buf.ravel(), diff_buf.ravel()))

            # Back project residual
            launch_back(diff_buf, geoms[pi], vol_update, vol_weight)

        # SIRT update + non-negativity
        launch_sirt_update(volume, vol_update, vol_weight)

        rss_history.append(total_rss)
        print("  Iter {:3d}/{}  RSS = {:.4e}".format(
            iteration + 1, num_iters, total_rss))

        if (iteration + 1) % 10 == 0 or iteration == num_iters - 1:
            print("  Saving volume...")
            save_volume(volume, out_dir)

    return cp.asnumpy(volume), rss_history


# ---------------------------------------------------------------------------
# Save volume
# ---------------------------------------------------------------------------

def save_volume(volume_gpu, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vol = cp.asnumpy(volume_gpu)
    for iz in range(vol.shape[0]):
        tifffile.imwrite(
            os.path.join(out_dir, "slice_{:04d}.tif".format(iz)),
            vol[iz])
    print("    Saved {} slices to {}".format(vol.shape[0], out_dir))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def show_results(vol, rss_history):
    nz, ny, nx = vol.shape
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(vol[nz // 2],      cmap="gray", vmin=0)
    axes[0, 0].set_title("XY  z={}".format(nz // 2))

    axes[0, 1].imshow(vol[:, ny // 2, :], cmap="gray", vmin=0)
    axes[0, 1].set_title("XZ  y={}".format(ny // 2))

    axes[1, 0].imshow(vol[:, :, nx // 2], cmap="gray", vmin=0)
    axes[1, 0].set_title("YZ  x={}".format(nx // 2))

    axes[1, 1].plot(range(1, len(rss_history) + 1), rss_history, "b-o", markersize=3)
    axes[1, 1].set_title("SIRT Convergence")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("RSS")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)

    plt.suptitle("CT Reconstruction - SIRT", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CT SIRT Reconstruction")
    parser.add_argument("--proj_dir", required=True,
                        help="Folder with z0.tif, z24.tif ... z336.tif")
    parser.add_argument("--out_dir",  required=True,
                        help="Output folder for reconstructed slices")
    parser.add_argument("--iters",    type=int,   default=50,
                        help="Number of SIRT iterations (default 50)")
    parser.add_argument("--I0",       type=float, default=None,
                        help="Air pixel value for Beer-Lambert (default: auto)")
    parser.add_argument("--show",     action="store_true",
                        help="Show orthogonal slice plots when done")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nLoading projections from", args.proj_dir)
    raw = load_projections(args.proj_dir, ANGLES)
    print("  Shape :", raw.shape)
    print("  Range :", raw.min(), "-", raw.max())

    if args.I0 is not None:
        I0 = args.I0
        print("  I0    :", I0, "(provided)")
    else:
        I0 = float(np.percentile(raw, 99.5))
        print("  I0    :", I0, "(auto from 99.5th percentile)")

    projs_att = to_attenuation(raw, I0)
    print("  Attenuation range:", projs_att.min(), "-", projs_att.max())

    geoms = build_geometry(ANGLES)
    print("\nFirst 3 geometries:")
    for g in geoms[:3]:
        print("  angle={:3d}  src={}  det={}".format(
            g["angle"],
            g["src"].round(1),
            g["det_c"].round(1)))

    print("\nStarting SIRT")
    print("  Volume     :", VOL_NX, "x", VOL_NY, "x", VOL_NZ, "voxels @", VOL_PIX, "mm")
    print("  Ray step   :", STEP_MM, "mm")
    print("  Relaxation :", RELAXATION)
    print("  Iterations :", args.iters)

    vol, rss = sirt_reconstruct(projs_att, geoms, args.iters, args.out_dir)

    print("\nDone.")
    print("  Volume range:", vol.min(), "-", vol.max())

    if args.show:
        show_results(vol, rss)


if __name__ == "__main__":
    main()
