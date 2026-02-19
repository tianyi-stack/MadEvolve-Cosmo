"""
Initial Tidal Reconstruction Program

This module implements the tidal reconstruction algorithm for cosmological
density field reconstruction using the tidal tensor method.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq


# EVOLVE-BLOCK-START

# TUNABLE: filter_scale = 1.25, bounds=(0.5, 5.0), method=grid

def run_reconstruction(data: np.ndarray, filter_scale: float = 1.25) -> np.ndarray:
    """
    Perform tidal reconstruction on a degraded density field.

    This is the initial implementation based on the tidal reconstruction algorithm.
    The algorithm computes displacement fields from the density field, calculates
    the tidal tensor components, and reconstructs the density field.

    Args:
        data: Input 3D density field (1+δ) as a NumPy array
        filter_scale: Gaussian filter scale in Mpc/h (default: 1.25)

    Returns:
        Reconstructed 3D density field (1+δ) as a NumPy array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    if not (data.shape[0] == data.shape[1] == data.shape[2]):
        raise ValueError("Input data must be a cubic 3D array.")

    nmesh = data.shape[0]
    boxsize = 1000.0  # Mpc/h
    kf = 2 * np.pi / boxsize

    # Initialize frequency grid
    fn = fftfreq(nmesh, 1.0 / nmesh).astype(np.float64)

    # 3D wavenumber magnitude
    k_ind = np.sqrt(
        fn[:, None, None]**2 +
        fn[None, :, None]**2 +
        fn[None, None, :]**2
    ).astype(np.float32)

    # Convert to overdensity δ = ρ/ρ̄ - 1
    delta = data - 1.0

    # Fourier transform
    delta_k = fftn(delta.astype(np.float32))

    # Apply Gaussian filter: exp(-k^2 * R^2 / 2)
    window = np.exp(-0.5 * (k_ind * kf)**2 * filter_scale**2).astype(np.float32)
    delta_k_filtered = delta_k * window

    # Compute displacement field
    # Calculate ik * δ_k
    temp = (kf * 1j * delta_k_filtered).astype(np.complex64)

    # Gradients in three directions
    psi_k1 = (fn[:, None, None] * temp).astype(np.complex64)
    psi_k2 = (fn[None, :, None] * temp).astype(np.complex64)
    psi_k3 = (fn[None, None, :] * temp).astype(np.complex64)

    # Set DC component to zero
    psi_k1[0, 0, 0] = 0
    psi_k2[0, 0, 0] = 0
    psi_k3[0, 0, 0] = 0

    # Inverse Fourier transform to real space
    psi1 = ifftn(psi_k1).real.astype(np.float32)
    psi2 = ifftn(psi_k2).real.astype(np.float32)
    psi3 = ifftn(psi_k3).real.astype(np.float32)

    # Calculate components of tidal tensor
    s11 = (psi1 * psi1 - psi2 * psi2) * 0.5
    s12 = psi1 * psi2
    s13 = psi1 * psi3
    s23 = psi2 * psi3
    s33 = (2 * psi3 * psi3 - psi1 * psi1 - psi2 * psi2) / 6

    # Fourier transform back to k-space
    s11_k = fftn(s11)
    s12_k = fftn(s12)
    s13_k = fftn(s13)
    s23_k = fftn(s23)
    s33_k = fftn(s33)

    # Calculate reconstructed density field
    temp_inv = 1.0 / (2 * k_ind**2)
    temp_inv[0, 0, 0] = 0  # Avoid division by zero

    # Combine all components
    delta_rec_k = (
        (fn[:, None, None]**2 - fn[None, :, None]**2) * temp_inv * s11_k +
        (2 * fn[:, None, None] * fn[None, :, None]) * temp_inv * s12_k +
        (2 * fn[:, None, None] * fn[None, None, :]) * temp_inv * s13_k +
        (2 * fn[None, :, None] * fn[None, None, :]) * temp_inv * s23_k +
        (2 * fn[None, None, :]**2 - fn[:, None, None]**2 - fn[None, :, None]**2) * temp_inv * s33_k
    )

    delta_rec_k[0, 0, 0] = 0

    # Inverse Fourier transform to real space
    delta_rec = ifftn(delta_rec_k).real

    # Convert back to density 1+δ
    density_rec = delta_rec + 1.0

    return density_rec.astype(np.float32)
# EVOLVE-BLOCK-END


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
    import os

    # Load simulation data (512^3 resolution)
    nmesh = 512
    para = 'fiducial'

    degraded_density_filename = f"Density_DM_z1.0_lost_noise2_{nmesh}_De.bin"
    degraded_density_filepath = f'/scratch/zangsh/Quijote_Simulations/Density/{para}/{sim_idx}/{degraded_density_filename}'

    gt_density_filename = f"Density_DM_z1.0_{nmesh}_De.bin"
    gt_density_filepath = f'/scratch/zangsh/Quijote_Simulations/Density/{para}/{sim_idx}/{gt_density_filename}'

    # Check if files exist
    if not os.path.exists(degraded_density_filepath):
        raise FileNotFoundError(f"Degraded density file not found: {degraded_density_filepath}")
    if not os.path.exists(gt_density_filepath):
        raise FileNotFoundError(f"Ground truth file not found: {gt_density_filepath}")

    # Load degraded density field (input for reconstruction)
    degraded_density_field = np.fromfile(degraded_density_filepath, dtype=np.float32).reshape([nmesh, nmesh, nmesh])

    # Load ground truth density field (for comparison)
    gt_density_field = np.fromfile(gt_density_filepath, dtype=np.float32).reshape([nmesh, nmesh, nmesh])

    # Run reconstruction
    start_time = time.time()
    reconstructed_field = run_reconstruction(degraded_density_field)
    computation_time = time.time() - start_time

    return reconstructed_field, gt_density_field, computation_time
