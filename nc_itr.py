#Extract projections and angles

import numpy as np
import imageio.v3 as iio
import glob
import os

def intensity_to_attenuation(proj):
    # Estimate I0 from brightest pixels
    I0 = np.percentile(proj, 99.9)

    eps = 1e-6
    proj_norm = proj / (I0 + eps)

    proj_norm = np.clip(proj_norm, eps, 1.0)

    return -np.log(proj_norm)

def load_projections(folder):
    files = sorted(glob.glob(os.path.join(folder, "z*.tif")))

    projections = []
    angles_deg = []

    for f in files:
        name = os.path.basename(f)
        angle = float(name[1:].replace(".tif", ""))  # extract 0,24,...336
        angles_deg.append(angle)

        img = iio.imread(f).astype(np.float32)
        projections.append(img)

    projections = np.stack(projections, axis=0)
    angles_deg = np.array(angles_deg)

    return projections, angles_deg

projections_raw, angles_deg = load_projections("data")

projections = np.zeros_like(projections_raw)

for i in range(len(projections_raw)):
    projections[i] = intensity_to_attenuation(projections_raw[i])

print(projections.min(), projections.max())



#Geom Definition
DSO = 830.71 mm
ODD = 332.29 mm
DSD = 1163 mm
pixel_size = 0.4 mm
detector = 350 × 350

S0 = (0, -830.71, 0)
D0 = (0,  332.29, 0)
u0 = (1, 0, 0) * 0.4
v0 = (0, 0, 1) * 0.4

#Rotation about Z axis
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

    S0 = np.array([0.0, -830.71, 0.0])
    D0 = np.array([0.0,  332.29, 0.0])

    u0 = np.array([0.4, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.4])

    R = Rz(theta)

    S = R @ S0
    D = R @ D0
    u = R @ u0
    v = R @ v0

    return S, D, u, v


#Volume Grid
voxel_size = 1.0
Nx = Ny = Nz = 80

x = (np.arange(Nx) - Nx/2) * voxel_size
y = (np.arange(Ny) - Ny/2) * voxel_size
z = (np.arange(Nz) - Nz/2) * voxel_size

volume = np.zeros((Nx, Ny, Nz), dtype=np.float32)



#Forward Projector
def ray_integral(volume, S, Pdet, num_samples=300):
    ray = Pdet - S
    L = np.linalg.norm(ray)
    ray_dir = ray / L

    dt = L / num_samples
    val = 0.0

    for k in range(num_samples):
        p = S + k * dt * ray_dir

        ix = int(p[0]/voxel_size + Nx/2)
        iy = int(p[1]/voxel_size + Ny/2)
        iz = int(p[2]/voxel_size + Nz/2)

        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            val += volume[ix, iy, iz]

    return val * dt


#Adjoint Backprojector
def backproject_ray(correction, S, Pdet, residual, num_samples=300):
    ray = Pdet - S
    L = np.linalg.norm(ray)
    ray_dir = ray / L

    dt = L / num_samples
    weight = dt

    for k in range(num_samples):
        p = S + k * dt * ray_dir

        ix = int(p[0]/voxel_size + Nx/2)
        iy = int(p[1]/voxel_size + Ny/2)
        iz = int(p[2]/voxel_size + Nz/2)

        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            correction[ix, iy, iz] += residual * weight

#SIRT loop
lambda_relax = 0.03
n_iter = 20

for it in range(n_iter):

    correction = np.zeros_like(volume)

    for p_idx in range(len(projections)):

        angle = angles_deg[p_idx]
        S, D, u, v = compute_geometry(angle)

        proj_meas = projections[p_idx]
        proj_sim = np.zeros_like(proj_meas)

        for i in range(350):
            for j in range(350):

                Pdet = (
                    D
                    + (i - 175) * u
                    + (j - 175) * v
                )

                proj_sim[i, j] = ray_integral(volume, S, Pdet)

        residual = proj_meas - proj_sim

        for i in range(350):
            for j in range(350):

                Pdet = (
                    D
                    + (i - 175) * u
                    + (j - 175) * v
                )

                backproject_ray(
                    correction,
                    S,
                    Pdet,
                    residual[i, j]
                )

    volume += lambda_relax * correction
    volume = np.clip(volume, 0, None)

    print("Iteration", it, "complete")
























