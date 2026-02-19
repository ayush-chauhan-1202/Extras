# ============================================================
# Cone-beam SIRT Reconstruction
# Projections: rotation about Z axis
# Pixel size: 0.4 mm
# Detector: 350 x 350
# Projections: z0.tif, z24.tif, ..., z336.tif (32-bit float)
# ============================================================

import numpy as np
import imageio.v3 as iio
import glob
import os

# ============================================================
# -------------------- PARAMETERS ----------------------------
# ============================================================

# Geometry (mm)
DSO = 830.71   # source to origin
ODD = 332.29   # origin to detector
DSD = DSO + ODD

pixel_size = 0.4  # mm

Nu = 350
Nv = 350

# Volume
voxel_size = 1.0  # mm
Nx = Ny = Nz = 80

# SIRT
lambda_relax = 0.03
n_iter = 20
num_samples = 300

# ============================================================
# ---------------- INTENSITY → ATTENUATION -------------------
# ============================================================

def intensity_to_attenuation(proj):
    """
    Convert 32-bit float detector intensity to line integral.

    Air = high intensity
    Material = lower intensity

    I0 estimated robustly from brightest 0.1% pixels.
    """

    proj = proj.astype(np.float32)

    # Estimate air intensity (robust to variation)
    I0 = np.percentile(proj, 99.9)

    eps = 1e-6
    proj_norm = proj / (I0 + eps)

    # Prevent log(0) or negative
    proj_norm = np.clip(proj_norm, eps, 1.0)

    return -np.log(proj_norm)


# ============================================================
# -------------------- LOAD PROJECTIONS ----------------------
# ============================================================

def load_projections(folder):
    files = sorted(glob.glob(os.path.join(folder, "z*.tif")))

    projections = []
    angles_deg = []

    for f in files:
        name = os.path.basename(f)

        # Extract angle from filename zXX.tif
        angle = float(name[1:].replace(".tif", ""))
        angles_deg.append(angle)

        img = iio.imread(f).astype(np.float32)

        if img.shape != (Nu, Nv):
            raise ValueError(f"Projection {name} has wrong size {img.shape}")

        projections.append(img)

    projections = np.stack(projections, axis=0)
    angles_deg = np.array(angles_deg)

    return projections, angles_deg


# Load raw intensities
projections_raw, angles_deg = load_projections("data")

# Convert to attenuation
projections = np.zeros_like(projections_raw)

for i in range(len(projections_raw)):
    projections[i] = intensity_to_attenuation(projections_raw[i])

print("Projection attenuation range:",
      projections.min(), projections.max())

# ============================================================
# -------------------- GEOMETRY ------------------------------
# ============================================================

# Base (0°) geometry: rotation about Z axis
S0 = np.array([0.0, -DSO, 0.0])
D0 = np.array([0.0,  ODD, 0.0])

u0 = np.array([pixel_size, 0.0, 0.0])  # detector x-axis
v0 = np.array([0.0, 0.0, pixel_size])  # detector z-axis


def Rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def compute_geometry(angle_deg):
    theta = np.deg2rad(angle_deg)
    R = Rz(theta)

    S = R @ S0
    D = R @ D0
    u = R @ u0
    v = R @ v0

    return S, D, u, v


# ============================================================
# -------------------- VOLUME GRID ---------------------------
# ============================================================

x = (np.arange(Nx) - Nx/2) * voxel_size
y = (np.arange(Ny) - Ny/2) * voxel_size
z = (np.arange(Nz) - Nz/2) * voxel_size

volume = np.zeros((Nx, Ny, Nz), dtype=np.float32)


# ============================================================
# -------------------- FORWARD PROJECTOR ---------------------
# ============================================================

def ray_integral(volume, S, Pdet, num_samples):
    ray = Pdet - S
    L = np.linalg.norm(ray)
    ray_dir = ray / L

    dt = L / num_samples
    val = 0.0

    for k in range(num_samples):
        p = S + k * dt * ray_dir

        ix = int(p[0] / voxel_size + Nx/2)
        iy = int(p[1] / voxel_size + Ny/2)
        iz = int(p[2] / voxel_size + Nz/2)

        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            val += volume[ix, iy, iz]

    return val * dt


# ============================================================
# -------------------- BACKPROJECTOR -------------------------
# ============================================================

def backproject_ray(correction, S, Pdet, residual, num_samples):
    ray = Pdet - S
    L = np.linalg.norm(ray)
    ray_dir = ray / L

    dt = L / num_samples
    weight = dt

    for k in range(num_samples):
        p = S + k * dt * ray_dir

        ix = int(p[0] / voxel_size + Nx/2)
        iy = int(p[1] / voxel_size + Ny/2)
        iz = int(p[2] / voxel_size + Nz/2)

        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            correction[ix, iy, iz] += residual * weight


# ============================================================
# -------------------- SIRT LOOP -----------------------------
# ============================================================

center_u = (Nu - 1) / 2.0
center_v = (Nv - 1) / 2.0

for it in range(n_iter):

    correction = np.zeros_like(volume)

    for p_idx in range(len(projections)):

        angle = angles_deg[p_idx]
        S, D, u, v = compute_geometry(angle)

        proj_meas = projections[p_idx]
        proj_sim = np.zeros_like(proj_meas)

        # ---- Forward projection ----
        for i in range(Nu):
            for j in range(Nv):

                Pdet = (
                    D
                    + (i - center_u) * u
                    + (j - center_v) * v
                )

                proj_sim[i, j] = ray_integral(
                    volume, S, Pdet, num_samples
                )

        residual = proj_meas - proj_sim

        # ---- Backprojection ----
        for i in range(Nu):
            for j in range(Nv):

                Pdet = (
                    D
                    + (i - center_u) * u
                    + (j - center_v) * v
                )

                backproject_ray(
                    correction,
                    S,
                    Pdet,
                    residual[i, j],
                    num_samples
                )

    volume += lambda_relax * correction

    # Enforce non-negativity (physical μ ≥ 0)
    volume = np.clip(volume, 0, None)

    print(f"Iteration {it+1}/{n_iter} complete")
