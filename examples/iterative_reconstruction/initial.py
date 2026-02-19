"""
Iterative BAO Reconstruction Program (Particle-based with JAX interface)
"""

import os
import sys
import numpy as np
from jax import numpy as jnp
import jax

# Add path for FastLoop_multi and Quijote_lib (compiled Cython modules)
# Hardcoded because the program may be copied to different locations during evolution
_MLR_ROOT = '/home/tianyi/real_tianyi/code/MadEvolve-Cosmo/examples/iterative_reconstruction/MLR'
sys.path.insert(0, os.path.join(_MLR_ROOT, 'CosCalNL_multi/FastLoop_multi'))
sys.path.insert(0, _MLR_ROOT)

# EVOLVE-BLOCK-START

# TUNABLE: R_init = 10.0, bounds=(5.0, 20.0), method=grid
# TUNABLE: R_min = 1.0, bounds=(0.5, 3.0), method=grid
# TUNABLE: decay_factor = 0.5, bounds=(0.3, 0.8), method=grid

def reconstruct(data: jnp.ndarray, R_init: float = 10.0, R_min: float = 1.0,
                decay_factor: float = 0.5,
                box_size: float = 1000.0, n_iterations: int = 8) -> jnp.ndarray:
    """
    Performs iterative 3D BAO reconstruction using particle-based method.

    This follows the RecCal_multi approach:
    1. Load particle positions from snapshot
    2. Iteratively: CIC paint -> smooth -> compute Zeldovich displacement -> shift particles
    3. Compute displacement field from particle movements
    4. Estimate linear density from divergence of displacement

    Args:
        data: Input 3D density field (1+delta) - used to determine grid size
        R_init: Initial Gaussian smoothing scale in Mpc/h (default: 10.0)
        R_min: Minimum smoothing scale in Mpc/h (default: 1.0)
        decay_factor: Factor by which R is multiplied each iteration (default: 0.5)
        box_size: Box size in Mpc/h (default: 1000.0)
        n_iterations: Number of iterations (default: 8)

    Returns:
        Reconstructed linear density field (delta) as a JAX array
    """
    # Get simulation index from global context (set by run_experiment)
    global _current_sim_idx
    sim_idx = _current_sim_idx

    nmesh = data.shape[0]

    # Run particle-based reconstruction
    result = apply_particle_reconstruction(
        sim_idx, nmesh, R_init, R_min, decay_factor, box_size, n_iterations
    )

    return jnp.asarray(result)


def apply_particle_reconstruction(sim_idx, nmesh, R_init, R_min, decay_factor,
                                   box_size, n_iterations):
    """
    Applies iterative reconstruction using particle method (RecCal_multi approach).
    """
    import FastLoop_multi
    from Quijote_lib import readgadget

    # Load particle positions
    snapPATH = f'/scratch/zangsh/Quijote_Simulations/Snapshots/fiducial/{sim_idx}/snapdir_004/snap_004'
    ptype = [1]
    Position = (readgadget.read_block(snapPATH, "POS ", ptype) / 1e3).astype(np.float32)
    Size = Position.shape[0]
    Position0 = np.array(Position)

    # Setup k-space grid
    fn = np.fft.fftfreq(nmesh, 1. / nmesh).astype(np.float32)
    rfn = np.fft.rfftfreq(nmesh, 1. / nmesh).astype(np.float32)
    fnx = (fn[:, None, None] + np.zeros(nmesh, dtype=np.float32)[None, :, None]
           + np.zeros(int(nmesh/2 + 1), dtype=np.float32)[None, None, :])
    fny = (np.zeros(nmesh, dtype=np.float32)[:, None, None] + fn[None, :, None]
           + np.zeros(int(nmesh/2 + 1), dtype=np.float32)[None, None, :])
    fnz = (np.zeros(nmesh, dtype=np.float32)[:, None, None]
           + np.zeros(nmesh, dtype=np.float32)[None, :, None] + rfn[None, None, :])
    k_ind = ((fn[:, None, None]**2. + fn[None, :, None]**2. + rfn[None, None, :]**2.)**(0.5)).astype(np.float32)
    Kf = 2 * np.pi / box_size

    # Iterative reconstruction loop
    for i in range(n_iterations):
        # CIC paint particles to mesh
        delta = FastLoop_multi.CICPaint_multi(Position, nmesh, int(box_size), Size) - 1

        # Compute smoothing scale: R = max(decay^i * R_init, R_min)
        R = max(decay_factor**i * R_init, R_min)

        # Apply Gaussian smoothing in Fourier space
        deltaK = np.fft.rfftn(delta)
        window = np.exp(-0.5 * (k_ind * Kf)**2 * R**2).astype(np.float32)
        deltaK *= window

        # Compute Zeldovich displacement: Psi = -i * k / k^2 * delta_k / Kf
        k_ind[0, 0, 0] = 1  # Avoid division by zero
        temp = -1j * deltaK / k_ind**2 / Kf
        k_ind[0, 0, 0] = 0

        Dis = np.empty([nmesh, nmesh, nmesh, 3], dtype=np.float32)
        Dis[:, :, :, 0] = np.fft.irfftn(temp * fnx).real.astype(np.float32)
        Dis[:, :, :, 1] = np.fft.irfftn(temp * fny).real.astype(np.float32)
        Dis[:, :, :, 2] = np.fft.irfftn(temp * fnz).real.astype(np.float32)

        # Shift particles using CIC interpolation
        Position = FastLoop_multi.Shift_multi(
            Position.astype(np.float32), Dis.astype(np.float32),
            nmesh, int(box_size), Size
        )

    # Compute final displacement field
    Density = FastLoop_multi.CICPaint_multi(Position, nmesh, int(box_size), Size)
    Dis = FastLoop_multi.DisInter(
        Position0.astype(np.float32), Position.astype(np.float32),
        Density, nmesh, int(box_size), Size
    )

    sx, sy, sz = Dis[:, :, :, 0], Dis[:, :, :, 1], Dis[:, :, :, 2]

    # Compute divergence of displacement to get reconstructed density
    dxk = np.fft.rfftn(sx)
    dyk = np.fft.rfftn(sy)
    dzk = np.fft.rfftn(sz)
    delta0k = (1j * Kf) * (fnx * dxk + fny * dyk + fnz * dzk)
    deltaX0 = np.fft.irfftn(delta0k).astype(np.float32)

    return deltaX0

# EVOLVE-BLOCK-END


# Data path configuration
DATA_BASE_PATH = '/scratch/zangsh/Quijote_Simulations/Density/fiducial'
NMESH = 256

# Global variable to track current simulation index
_current_sim_idx = 0


def read_files(sim_idx: int = 0):
    """Load simulation data."""
    global _current_sim_idx
    _current_sim_idx = sim_idx

    dm_density_filepath = f'{DATA_BASE_PATH}/{sim_idx}/Density_DM_{NMESH}_De.bin'
    ic_filepath = f'{DATA_BASE_PATH}/{sim_idx}/Density_IC_scaled_{NMESH}_De.bin'

    if not os.path.exists(dm_density_filepath):
        raise FileNotFoundError(f"Dark matter density file not found: {dm_density_filepath}")
    if not os.path.exists(ic_filepath):
        raise FileNotFoundError(f"Initial condition file not found: {ic_filepath}")

    dm_density_field = jnp.asarray(np.fromfile(dm_density_filepath, dtype=np.float32).reshape([NMESH, NMESH, NMESH]))
    density_field_gt = jnp.asarray(np.fromfile(ic_filepath, dtype=np.float32).reshape([NMESH, NMESH, NMESH]))

    return dm_density_field, density_field_gt


def run_experiment(sim_idx: int = 0, **kwargs):
    """
    Main experiment function called by the evaluator.
    """
    import time

    global _current_sim_idx
    _current_sim_idx = sim_idx

    dm_density_field, density_field_gt = read_files(sim_idx)

    start_time = time.time()
    reconstructed_field = reconstruct(dm_density_field)
    computation_time = time.time() - start_time

    return reconstructed_field, density_field_gt, computation_time


def compute_rk_metrics(recon_field, gt_field, kmin=0.01, kmax=0.5,
                       nbins=20, box_size=1000.0):
    """
    Compute r(k) curve and average r(k) within specified k range.
    """
    delta_r = jnp.fft.fftshift(jnp.fft.fftn(recon_field))
    delta_g = jnp.fft.fftshift(jnp.fft.fftn(gt_field))

    Nz, Ny, Nx = recon_field.shape

    kx = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nx)) * Nx / box_size
    ky = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Ny)) * Ny / box_size
    kz = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nz)) * Nz / box_size

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).flatten()

    P_rg = (delta_r * jnp.conj(delta_g)).flatten().real
    P_rr = (delta_r * jnp.conj(delta_r)).flatten().real
    P_gg = (delta_g * jnp.conj(delta_g)).flatten().real

    k_bins = jnp.linspace(kmin, kmax, nbins+1)
    r_k = jnp.zeros(nbins)
    k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

    for i in range(nbins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if jnp.any(mask):
            P_rg_bin = P_rg[mask].sum()
            P_rr_bin = P_rr[mask].sum()
            P_gg_bin = P_gg[mask].sum()
            denom = jnp.sqrt(P_rr_bin * P_gg_bin)
            if denom > 0:
                r_k = r_k.at[i].set(P_rg_bin / denom)
            else:
                r_k = r_k.at[i].set(0)
        else:
            r_k = r_k.at[i].set(0)

    avg_r = float(jnp.mean(r_k))

    return r_k, k_centers, avg_r


def main():
    """
    Test the iterative reconstruction algorithm.
    """
    print("Loading data...")
    data, gt_field = read_files(0)
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{float(data.min()):.4f}, {float(data.max()):.4f}]")
    print(f"Data mean: {float(data.mean()):.4f}")

    print("\nRunning particle-based reconstruction with 8 iterations...")
    import time
    start = time.time()
    recon_field = reconstruct(data)
    elapsed = time.time() - start
    print(f"Reconstruction time: {elapsed:.2f}s")
    print(f"Recon shape: {recon_field.shape}")
    print(f"Recon range: [{float(recon_field.min()):.4f}, {float(recon_field.max()):.4f}]")

    print("\nComputing metrics...")
    r_k, k_centers, avg_r = compute_rk_metrics(recon_field, gt_field)
    print(f"Average r(k) in [0.01, 0.5]: {avg_r:.4f}")

    print("\nr(k) per bin:")
    for k, r in zip(k_centers, r_k):
        print(f"  k={float(k):.4f}: r={float(r):.4f}")


if __name__ == "__main__":
    main()
