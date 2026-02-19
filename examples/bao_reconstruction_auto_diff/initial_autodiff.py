"""
Initial BAO Reconstruction Program (JAX Autodiff Version)

This module implements the standard 3D BAO (Baryon Acoustic Oscillation)
reconstruction algorithm based on the Zeldovich approximation.

IMPORTANT: This code is designed to be fully differentiable using JAX.
All operations in the EVOLVE-BLOCK must use jax.numpy (jnp) to enable
gradient-based parameter optimization via autodiff.
"""
import os
import numpy as np
from jax import numpy as jnp
import jax

# EVOLVE-BLOCK-START

# TUNABLE: smoothing_scale_R_s = 15.0, bounds=(1.0, 30.0), method=autodiff
# TUNABLE: bias = 1.8, bounds=(1.0, 3.0), method=autodiff

def reconstruct(data: jnp.ndarray, smoothing_scale_R_s: float = 15.0,
                bias: float = 1.8, box_size: float = 1000.0) -> jnp.ndarray:
    """
    Performs 3D BAO reconstruction for dark matter density fields.

    Args:
        data: Input 3D density field (dark matter overdensity) as a NumPy array.
              The input is expected to be a cubic array.
        smoothing_scale_R_s: Gaussian smoothing scale in Mpc/h.
                             The k-space density field delta_k is multiplied by
                             exp(-k^2 * R_s^2 / 2) where k includes the 2π factor.
                             This suppresses small-scale noise in the displacement field.
                             If R_s <= 0, no smoothing is applied. Default is 10.0.
        bias: Bias factor for dark matter. Default is 1.8.
        box_size: Box size in Mpc/h. Default is 1000.0.

    Returns:
        Reconstructed 3D density field as a NumPy array.
              The output array will have the same dtype as the input array.
    """
    if not isinstance(data, jnp.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    if not (data.shape[0] == data.shape[1] == data.shape[2]):
        raise ValueError("Input data must be a cubic 3D array.")

    N = data.shape[0]

    if N == 0:
        return jnp.array([], dtype=data.dtype).reshape(0, 0, 0)

    reconstructed_field = apply_reconstruction(data, smoothing_scale_R_s, bias, box_size)

    return reconstructed_field


def apply_reconstruction(density_field, smoothing_scale, bias, BoxSize):
    """Applies standard reconstruction to a density field.

    Standard reconstruction in cosmology is a technique used to sharpen the
    baryon acoustic oscillation (BAO) signal by removing non-linear effects
    of structure formation. It works by estimating the displacement field
    from the observed density field and then moving galaxies back to their
    initial positions.

    Args:
        density_field: 3D numpy array representing the density field
        smoothing_scale: smoothing scale in Mpc/h for the Gaussian filter
        bias: Bias factor for dark matter
        BoxSize: Box size in Mpc/h

    Returns:
        reconstructed_field: 3D numpy array of the reconstructed density field
    """
    # Get dimensions of the density field
    nx, ny, nz = density_field.shape

    # Convert density field to overdensity field
    overdensity = density_field - 1.0

    # FFT of the overdensity field
    delta_k = jnp.fft.fftn(overdensity)

    # Create k-space grid
    kx = jnp.fft.fftfreq(nx, 1/nx) * 2 * jnp.pi / BoxSize
    ky = jnp.fft.fftfreq(ny, 1/ny) * 2 * jnp.pi / BoxSize
    kz = jnp.fft.fftfreq(nz, 1/nz) * 2 * jnp.pi / BoxSize

    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2

    # Apply Gaussian smoothing
    smoothing_factor = jnp.exp(-0.5 * k_sq * smoothing_scale**2)

    # Avoid division by zero
    k_sq = k_sq.at[0, 0, 0].set(1.0)

    # Calculate displacement field in k-space
    # Using the Zeldovich approximation: Ψ(k) = -ik/k² δ(k)
    displacement_x_k = -1j * kx_grid / k_sq * delta_k * smoothing_factor / bias
    displacement_y_k = -1j * ky_grid / k_sq * delta_k * smoothing_factor / bias
    displacement_z_k = -1j * kz_grid / k_sq * delta_k * smoothing_factor / bias

    # Transform back to real space
    displacement_x = jnp.real(jnp.fft.ifftn(displacement_x_k))
    displacement_y = jnp.real(jnp.fft.ifftn(displacement_y_k))
    displacement_z = jnp.real(jnp.fft.ifftn(displacement_z_k))

    # Apply displacement to get reconstructed field
    reconstructed_field = apply_displacement(density_field.astype(jnp.float32),
                                          displacement_x.astype(jnp.float32),
                                          displacement_y.astype(jnp.float32),
                                          displacement_z.astype(jnp.float32),
                                          BoxSize,
                                          bias)

    return reconstructed_field


def apply_displacement(density_field, disp_x, disp_y, disp_z, BoxSize, bias):
    """Applies displacement field to the density field.

    Args:
        density_field: Original density field
        disp_x, disp_y, disp_z: Components of the displacement field
        BoxSize: Box size in Mpc/h
        bias: Bias factor for dark matter

    Returns:
        Reconstructed density field
    """
    nx, ny, nz = density_field.shape

    # Create grid particles
    H = BoxSize/nx
    X = jnp.arange(nx)*H
    Y = jnp.arange(ny)*H
    Z = jnp.arange(nz)*H
    Particle_grid = (jnp.array(jnp.meshgrid(X, Y, Z, indexing='ij')).reshape([3, -1]).transpose()).astype(jnp.float32)

    # Shift particles
    Position_shifted = Shift(Particle_grid, disp_x, disp_y, disp_z, nx, BoxSize, nx * ny * nz)

    # Paint shifted particles onto grid using CIC implementation
    delta_d = CICPaint_Single(Position_shifted, density_field.reshape(-1), nx, BoxSize, nx * ny * nz) - 1
    delta_s = CICPaint_Single(Position_shifted, jnp.ones(nx * ny * nz, dtype=jnp.float32), nx, BoxSize, nx * ny * nz) - 1

    # Apply shift
    reconstructed_field = (delta_d - delta_s) / bias

    return reconstructed_field


def Shift(particle_positions, disp_x, disp_y, disp_z, nmesh, box_size, num_particles=None):
    """
    Optimized vectorized implementation of particle position shifting with displacement fields.

    Args:
        particle_positions: Array of particle positions, shape (num_particles, 3)
        disp_x, disp_y, disp_z: 3D displacement field arrays, shape (nmesh, nmesh, nmesh)
        nmesh: Grid size
        box_size: Box size in physical units
        num_particles: Number of particles

    Returns:
        shifted_positions: Array of shifted particle positions with periodic boundary conditions
    """
    # Grid spacing
    cell_size = box_size / nmesh

    # Get particle positions
    pos_x = particle_positions[:, 0]
    pos_y = particle_positions[:, 1]
    pos_z = particle_positions[:, 2]

    # Find grid cell indices for all particles at once (vectorized)
    ix = (pos_x / cell_size).astype(int) % nmesh
    iy = (pos_y / cell_size).astype(int) % nmesh
    iz = (pos_z / cell_size).astype(int) % nmesh

    # Get displacements at particle positions using advanced indexing
    disp_at_x = disp_x[ix, iy, iz]
    disp_at_y = disp_y[ix, iy, iz]
    disp_at_z = disp_z[ix, iy, iz]

    # Apply displacement to all particles at once
    new_x = pos_x + disp_at_x
    new_y = pos_y + disp_at_y
    new_z = pos_z + disp_at_z

    # Apply periodic boundary conditions (vectorized)
    new_x = jnp.mod(new_x, box_size)
    new_y = jnp.mod(new_y, box_size)
    new_z = jnp.mod(new_z, box_size)

    # Stack results
    shifted_positions = jnp.column_stack([new_x, new_y, new_z])

    return shifted_positions.astype(jnp.float32)


def CICPaint_Single(particle_positions, particle_values, nmesh, box_size, num_particles=None):
    """
    Optimized vectorized implementation of Cloud-In-Cell (CIC) painting.
    Assigns particle values to a grid using CIC interpolation.

    Args:
        particle_positions: Array of particle positions, shape (num_particles, 3)
        particle_values: Array of particle values, shape (num_particles,)
        nmesh: Grid size
        box_size: Box size in physical units
        num_particles: Number of particles

    Returns:
        grid: 3D grid with painted values, shape (nmesh, nmesh, nmesh)
    """
    # Initialize grid
    grid = jnp.zeros((nmesh, nmesh, nmesh), dtype=jnp.float32)

    # Grid spacing
    cell_size = box_size / nmesh

    # Get all particle positions at once
    pos_x = particle_positions[:, 0]
    pos_y = particle_positions[:, 1]
    pos_z = particle_positions[:, 2]

    # Convert to grid coordinates
    x_grid = pos_x / cell_size
    y_grid = pos_y / cell_size
    z_grid = pos_z / cell_size

    # Find lower corner indices for all particles
    ix0 = jnp.floor(x_grid).astype(int)
    iy0 = jnp.floor(y_grid).astype(int)
    iz0 = jnp.floor(z_grid).astype(int)

    # Calculate fractional distances from lower corner for all particles
    dx = x_grid - ix0
    dy = y_grid - iy0
    dz = z_grid - iz0

    # Process each of the 8 corners
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Calculate indices for this corner
                ix = (ix0 + di) % nmesh
                iy = (iy0 + dj) % nmesh
                iz = (iz0 + dk) % nmesh

                # Calculate CIC weights for all particles at once
                wx = dx if di == 1 else (1 - dx)
                wy = dy if dj == 1 else (1 - dy)
                wz = dz if dk == 1 else (1 - dz)
                weights = wx * wy * wz

                # Add contributions to grid using np.add.at
                # This handles multiple particles contributing to the same grid point
                grid = grid.at[ix, iy, iz].add(particle_values * weights)

    return grid

# EVOLVE-BLOCK-END

# Data path configuration (single source of truth)
DATA_BASE_PATH = '/scratch/zangsh/Quijote_Simulations/Density/fiducial'
NMESH = 256

def read_files(sim_idx: int = 0):
    """Load simulation data for autodiff optimization."""
    dm_density_filepath = f'{DATA_BASE_PATH}/{sim_idx}/Density_DM_{NMESH}_De.bin'
    ic_filepath = f'{DATA_BASE_PATH}/{sim_idx}/Density_IC_scaled_{NMESH}_De.bin'

    if not os.path.exists(dm_density_filepath):
        raise FileNotFoundError(f"Dark matter density file not found: {dm_density_filepath}")
    if not os.path.exists(ic_filepath):
        raise FileNotFoundError(f"Initial condition file not found: {ic_filepath}")

    dm_density_field = jnp.asarray(np.fromfile(dm_density_filepath, dtype=jnp.float32).reshape([NMESH, NMESH, NMESH]))
    density_field_gt = jnp.asarray(np.fromfile(ic_filepath, dtype=jnp.float32).reshape([NMESH, NMESH, NMESH]))

    return dm_density_field, density_field_gt

def run_experiment(sim_idx: int = 0, **kwargs):
    """
    Main experiment function called by the evaluator.

    Args:
        sim_idx: Index of the simulation to process
        **kwargs: Additional keyword arguments

    Returns:
        Tuple containing (reconstructed_field, ground_truth_field, computation_time)
    """
    import time

    # Load data using shared function
    dm_density_field, density_field_gt = read_files(sim_idx)

    # Run reconstruction with default parameters
    # Note: Parameters will be optimized by the autodiff optimizer
    start_time = time.time()
    reconstructed_field = reconstruct(dm_density_field)
    computation_time = time.time() - start_time

    return reconstructed_field, density_field_gt, computation_time

def compute_rk_metrics(recon_field, gt_field, kmin=0.05, kmax=0.5,
                      nbins=50, box_size=1000.0):
    """
    Compute r(k) curve and average r(k) within BAO scales (kmin–kmax).

    Parameters
    ----------
    recon_field : ndarray
        Reconstructed density field (3D)
    gt_field : ndarray
        Ground truth density field (3D)
    kmin : float
        Minimum k for BAO range in h/Mpc
    kmax : float
        Maximum k for BAO range in h/Mpc
    nbins : int
        Number of k bins
    box_size : float
        Box size in Mpc/h (default: 1000.0 for Quijote simulations)

    Returns
    -------
    r_k : ndarray
        Cross-correlation coefficient r(k) for each k bin
    k_bin_centers : ndarray
        Centers of k bins in h/Mpc
    avg_r_bao : float
        Average r(k) within BAO scales [kmin, kmax]
    """
    # 3-D FFT (shift zero-freq to center)
    delta_r = jnp.fft.fftshift(jnp.fft.fftn(recon_field))
    delta_g = jnp.fft.fftshift(jnp.fft.fftn(gt_field))

    Nz, Ny, Nx = recon_field.shape

    # Calculate k in physical units (h/Mpc)
    kx = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nx)) * Nx / box_size
    ky = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Ny)) * Ny / box_size
    kz = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nz)) * Nz / box_size

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).flatten()

    # Compute power spectra
    P_rg = (delta_r * jnp.conj(delta_g)).flatten().real  # Cross power spectrum
    P_rr = (delta_r * jnp.conj(delta_r)).flatten().real  # Reconstructed auto power
    P_gg = (delta_g * jnp.conj(delta_g)).flatten().real  # Ground truth auto power

    # Create k bins
    k_bins = jnp.linspace(k_mag.min(), k_mag.max(), nbins+1)
    r_k = jnp.zeros(nbins)
    k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

    # Calculate r(k) for each bin
    for i in range(nbins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if jnp.any(mask):
            P_rg_bin = P_rg[mask].mean()
            P_rr_bin = P_rr[mask].mean()
            P_gg_bin = P_gg[mask].mean()
            denom = jnp.sqrt(P_rr_bin * P_gg_bin)
            if denom > 0:
                r_k = r_k.at[i].set(P_rg_bin / denom)
            else:
                r_k = r_k.at[i].set(0)
        else:
            r_k = r_k.at[i].set(0)

    # Average r(k) over BAO range
    bao_mask = (k_centers >= kmin) & (k_centers <= kmax)
    avg_r_bao = r_k[bao_mask].mean() if jnp.any(bao_mask) else 0.0

    return r_k, k_centers, avg_r_bao

def main():
    """
    Test autodiff capability of the reconstruction algorithm.
    This demonstrates that gradients can flow through the entire pipeline.
    """
    data, gt_field = read_files(0)
    avg_r = run_exp(data, gt_field)
    print(f'avg_r = {float(avg_r):.6f}')


def run_exp(data, gt_field):
    """
    Run reconstruction and compute metric (for smoke-test).
    """
    reconstructed_field = reconstruct(data)
    r_k, k_centers, avg_r_bao = compute_rk_metrics(reconstructed_field, gt_field)
    return avg_r_bao


if __name__ == "__main__":
    main()