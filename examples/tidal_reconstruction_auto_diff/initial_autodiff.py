"""
Initial Tidal Reconstruction Program (JAX Autodiff Version)

This module implements the tidal reconstruction algorithm for cosmological
density field reconstruction using the tidal tensor method.

IMPORTANT: This code is designed to be fully differentiable using JAX.
All operations in the EVOLVE-BLOCK must use jax.numpy (jnp) to enable
gradient-based parameter optimization via autodiff.
"""
import os
import numpy as np
from jax import numpy as jnp
import jax


# EVOLVE-BLOCK-START

# TUNABLE: filter_scale = 1.38155, bounds=(0.5, 5.0), method=autodiff

def run_reconstruction(data: jnp.ndarray, filter_scale: float = 1.38155) -> jnp.ndarray:
    """
    Perform tidal reconstruction on a degraded density field.

    This is the initial implementation based on the tidal reconstruction algorithm.
    The algorithm computes displacement fields from the density field, calculates
    the tidal tensor components, and reconstructs the density field.

    All operations are fully differentiable using JAX.

    Args:
        data: Input 3D density field (1+delta) as a JAX array
        filter_scale: Gaussian filter scale in Mpc/h (default: 1.625)

    Returns:
        Reconstructed 3D density field (1+delta) as a JAX array
    """
    if not isinstance(data, jnp.ndarray):
        raise TypeError("Input data must be a JAX array.")
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    if not (data.shape[0] == data.shape[1] == data.shape[2]):
        raise ValueError("Input data must be a cubic 3D array.")

    nmesh = data.shape[0]
    boxsize = 1000.0  # Mpc/h
    kf = 2 * jnp.pi / boxsize

    # Initialize frequency grid using JAX
    fn = jnp.fft.fftfreq(nmesh, 1.0 / nmesh)

    # 3D wavenumber magnitude
    k_ind = jnp.sqrt(
        fn[:, None, None]**2 +
        fn[None, :, None]**2 +
        fn[None, None, :]**2
    )

    # Convert to overdensity delta = rho/rho_bar - 1
    delta = data - 1.0

    # Fourier transform
    delta_k = jnp.fft.fftn(delta)

    # Apply Gaussian filter: exp(-k^2 * R^2 / 2)
    window = jnp.exp(-0.5 * (k_ind * kf)**2 * filter_scale**2)
    delta_k_filtered = delta_k * window

    # Compute displacement field
    # Calculate ik * delta_k
    temp = kf * 1j * delta_k_filtered

    # Gradients in three directions
    psi_k1 = fn[:, None, None] * temp
    psi_k2 = fn[None, :, None] * temp
    psi_k3 = fn[None, None, :] * temp

    # Set DC component to zero (using JAX's at method for immutability)
    psi_k1 = psi_k1.at[0, 0, 0].set(0)
    psi_k2 = psi_k2.at[0, 0, 0].set(0)
    psi_k3 = psi_k3.at[0, 0, 0].set(0)

    # Inverse Fourier transform to real space
    psi1 = jnp.real(jnp.fft.ifftn(psi_k1))
    psi2 = jnp.real(jnp.fft.ifftn(psi_k2))
    psi3 = jnp.real(jnp.fft.ifftn(psi_k3))

    # Calculate components of tidal tensor
    s11 = (psi1 * psi1 - psi2 * psi2) * 0.5
    s12 = psi1 * psi2
    s13 = psi1 * psi3
    s23 = psi2 * psi3
    s33 = (2 * psi3 * psi3 - psi1 * psi1 - psi2 * psi2) / 6

    # Fourier transform back to k-space
    s11_k = jnp.fft.fftn(s11)
    s12_k = jnp.fft.fftn(s12)
    s13_k = jnp.fft.fftn(s13)
    s23_k = jnp.fft.fftn(s23)
    s33_k = jnp.fft.fftn(s33)

    # Calculate reconstructed density field
    # Avoid division by zero by adding small epsilon
    k_sq = k_ind**2
    temp_inv = jnp.where(k_sq > 0, 1.0 / (2 * k_sq), 0.0)

    # Combine all components
    delta_rec_k = (
        (fn[:, None, None]**2 - fn[None, :, None]**2) * temp_inv * s11_k +
        (2 * fn[:, None, None] * fn[None, :, None]) * temp_inv * s12_k +
        (2 * fn[:, None, None] * fn[None, None, :]) * temp_inv * s13_k +
        (2 * fn[None, :, None] * fn[None, None, :]) * temp_inv * s23_k +
        (2 * fn[None, None, :]**2 - fn[:, None, None]**2 - fn[None, :, None]**2) * temp_inv * s33_k
    )

    # Set DC component to zero
    delta_rec_k = delta_rec_k.at[0, 0, 0].set(0)

    # Inverse Fourier transform to real space
    delta_rec = jnp.real(jnp.fft.ifftn(delta_rec_k))

    # Convert back to density 1+delta
    density_rec = delta_rec + 1.0

    return density_rec

# EVOLVE-BLOCK-END


# Data path configuration (single source of truth)
DATA_BASE_PATH = '/scratch/zangsh/Quijote_Simulations/Density/fiducial'
NMESH = 512

def read_files(sim_idx: int = 0):
    """Load simulation data for autodiff optimization."""
    degraded_density_filename = f"Density_DM_z1.0_lost_noise2_{NMESH}_De.bin"
    degraded_density_filepath = f'{DATA_BASE_PATH}/{sim_idx}/{degraded_density_filename}'

    gt_density_filename = f"Density_DM_z1.0_{NMESH}_De.bin"
    gt_density_filepath = f'{DATA_BASE_PATH}/{sim_idx}/{gt_density_filename}'

    if not os.path.exists(degraded_density_filepath):
        raise FileNotFoundError(f"Degraded density file not found: {degraded_density_filepath}")
    if not os.path.exists(gt_density_filepath):
        raise FileNotFoundError(f"Ground truth file not found: {gt_density_filepath}")

    # Load degraded density field (input for reconstruction)
    degraded_density_field = jnp.asarray(np.fromfile(degraded_density_filepath, dtype=np.float32).reshape([NMESH, NMESH, NMESH]))

    # Load ground truth density field (for comparison)
    gt_density_field = jnp.asarray(np.fromfile(gt_density_filepath, dtype=np.float32).reshape([NMESH, NMESH, NMESH]))

    return degraded_density_field, gt_density_field


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
    degraded_density_field, gt_density_field = read_files(sim_idx)

    # Run reconstruction with default parameters
    # Note: Parameters will be optimized by the autodiff optimizer
    start_time = time.time()
    reconstructed_field = run_reconstruction(degraded_density_field)
    computation_time = time.time() - start_time

    return reconstructed_field, gt_density_field, computation_time


def compute_r2d_metrics_differentiable(recon_field, gt_field, box_size=1000.0):
    """
    Compute 2D r(k_perp, k_para) map and average metric over specified region.
    Fully differentiable version using JAX.

    The metric is avg(r_2D[i,j]) for 1<=i<=5, 1<=j<=5

    Parameters
    ----------
    recon_field : jnp.ndarray
        Reconstructed density field (3D), expected to be (1+delta) format
    gt_field : jnp.ndarray
        Ground truth density field (3D), expected to be (1+delta) format
    box_size : float
        Box size in Mpc/h (default: 1000.0)

    Returns
    -------
    metric : float
        Average of r_2D[1:6, 1:6] (indices 1-5 inclusive in both dimensions)
    """
    nmesh = recon_field.shape[0]
    kf = 2 * jnp.pi / box_size

    # Convert to overdensity delta = rho/rho_bar - 1
    delta_recon = recon_field - 1.0
    delta_gt = gt_field - 1.0

    # Compute Fourier transforms
    norm_factor = 1.0 / nmesh**3
    delta_recon_k = jnp.fft.fftn(delta_recon) * norm_factor
    delta_gt_k = jnp.fft.fftn(delta_gt) * norm_factor

    # Compute power spectra in k-space
    volume_factor = box_size**3
    PK11 = jnp.real(delta_recon_k * jnp.conj(delta_recon_k) * volume_factor)
    PK22 = jnp.real(delta_gt_k * jnp.conj(delta_gt_k) * volume_factor)
    PK12 = jnp.real(delta_recon_k * jnp.conj(delta_gt_k) * volume_factor)

    # Get frequency grid
    fn = jnp.fft.fftfreq(nmesh, 1.0 / nmesh)

    # Compute k_perp and k_para
    # k_para is along the z-axis (line-of-sight)
    # k_perp is the transverse component sqrt(kx^2 + ky^2)
    kx = fn[:, None, None]
    ky = fn[None, :, None]
    kz = fn[None, None, :]

    k_perp = jnp.sqrt(kx**2 + ky**2)
    k_para = jnp.abs(kz)

    # Bin edges for 2D binning
    # Use the same binning as the original evaluator
    n_bins = 32  # Number of bins in each dimension
    k_max = nmesh // 2  # Maximum k in grid units

    k_perp_edges = jnp.linspace(0, k_max, n_bins + 1)
    k_para_edges = jnp.linspace(0, k_max, n_bins + 1)

    # Compute 2D binned power spectra using soft binning (differentiable)
    # For each bin, compute sum of power weighted by how much each mode belongs to that bin
    r_2D = jnp.zeros((n_bins, n_bins))

    # Create bin center arrays
    k_perp_centers = 0.5 * (k_perp_edges[:-1] + k_perp_edges[1:])
    k_para_centers = 0.5 * (k_para_edges[:-1] + k_para_edges[1:])

    # Compute r_2D for each bin
    for i in range(n_bins):
        for j in range(n_bins):
            # Define bin mask
            perp_mask = (k_perp >= k_perp_edges[i]) & (k_perp < k_perp_edges[i + 1])
            para_mask = (k_para >= k_para_edges[j]) & (k_para < k_para_edges[j + 1])
            bin_mask = perp_mask & para_mask

            # Sum power spectra in this bin
            P11_sum = jnp.sum(jnp.where(bin_mask, PK11, 0.0))
            P22_sum = jnp.sum(jnp.where(bin_mask, PK22, 0.0))
            P12_sum = jnp.sum(jnp.where(bin_mask, PK12, 0.0))

            # Compute r(k) for this bin
            denom = jnp.sqrt(P11_sum * P22_sum)
            r_bin = jnp.where(denom > 1e-10, P12_sum / denom, 0.0)
            r_2D = r_2D.at[i, j].set(r_bin)

    # Calculate metric: average of r_2D[1:6, 1:6] (indices 1-5 inclusive)
    metric = jnp.mean(r_2D[1:6, 1:6])

    return metric


def run_exp(data, gt_field):
    """
    Run reconstruction and compute metric (for smoke-test).
    """
    reconstructed_field = run_reconstruction(data)
    metric = compute_r2d_metrics_differentiable(reconstructed_field, gt_field)
    return metric


def main():
    """
    Quick smoke-test of the reconstruction.
    """
    data, gt_field = read_files(0)
    metric = run_exp(data, gt_field)
    print(f'r_2D metric: {float(metric):.6f}')


if __name__ == "__main__":
    main()
