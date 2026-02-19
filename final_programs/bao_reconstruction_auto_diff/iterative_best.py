"""
Iterative BAO Reconstruction Program (Particle-based with JAX interface)

This module implements the iterative BAO reconstruction algorithm based on
the method from https://arxiv.org/abs/1704.06634 (RecCal_multi approach)

The algorithm uses particle-based reconstruction with CIC painting and shifting.
This provides accurate reconstruction comparable to the original RecCal_multi.

For autodiff parameter optimization, gradients are computed using finite differences
since the Cython particle operations are not differentiable.
"""

import os
import sys
import numpy as np
from jax import numpy as jnp
import jax

# Add path for FastLoop_multi - use hardcoded absolute path
# This is necessary because the program may be copied to different locations during evolution
_PROJECT_ROOT = '/home/tianyi/real_tianyi/code/MadEvolve-Cosmo/examples/iterative_reconstruction'
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'MLR/CosCalNL_multi/FastLoop_multi'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'MLR'))

# EVOLVE-BLOCK-START

# TUNABLE: k_gate        = 0.2,   bounds=(0.2, 0.35), method=autodiff
# TUNABLE: gate_width    = 0.005,       bounds=(0.005, 0.1), method=autodiff
# TUNABLE: R_env         = 2,    bounds=(2.0, 10.0),  method=autodiff
# TUNABLE: gamma_diff    = -3,       bounds=(-3.0, 3.0),  method=autodiff
# TUNABLE: gamma_advect  = 0.266979,       bounds=(-1.0, 1.0),  method=autodiff
# TUNABLE: beta_2lpt     = 0.531379,        bounds=(0.0, 3.0),   method=autodiff
# TUNABLE: gamma_env     = 0.0937301,      bounds=(-2.0, 2.0),  method=autodiff
# TUNABLE: cs2           = 0.497698,      bounds=(-1.0, 1.0),  method=autodiff
# TUNABLE: deconv_amp    = 0.0841926,        bounds=(0.0, 2.0),   method=autodiff
# TUNABLE: nudge_amp     = 0.0233238,      bounds=(0.0, 0.5),   method=autodiff

def reconstruct(
    data: jnp.ndarray,
    k_gate: float = 0.2,
    gate_width: float = 0.005,
    R_env: float = 2,
    gamma_diff: float = -3,
    gamma_advect: float = 0.266979,
    beta_2lpt: float = 0.531379,
    gamma_env: float = 0.0937301,
    cs2: float = 0.497698,
    deconv_amp: float = 0.0841926,
    nudge_amp: float = 0.0233238,
    box_size: float = 1000.0,
    n_iterations: int = 8,
    R_init: float = 10.0,
    R_min: float = 1.0,
    decay_factor: float = 0.5
) -> jnp.ndarray:
    """
    Shear-aware + EFT crossover hybrid reconstruction:

    - Baseline: particle-based iterative ZA (non-diff, robust large scales).
    - Refinement (JAX, differentiable):
        * Large-scale environment + shear invariant from smoothed potential
        * Guided anisotropic diffusion + residual advection
        * 2LPT source modulated by environment+shear
        * Isotropic EFT counterterm (applied to residual only)
        * CIC deconvolution injection
        * Stable complex phase-nudging toward arcsinh-compressed nonlinear data
    - Strict low-k protection: zero corrections for k < 0.21 h/Mpc.
    """
    global _current_sim_idx
    sim_idx = _current_sim_idx
    nmesh = data.shape[0]

    # 1) Baseline Particle Reconstruction (black-box, independent of tunables here)
    delta_base_np = apply_particle_reconstruction(
        sim_idx, nmesh, R_init, R_min, decay_factor,
        box_size, n_iterations
    )
    delta_base = jnp.asarray(delta_base_np, dtype=jnp.float32)
    delta_base = jax.lax.stop_gradient(delta_base)

    # 2) Differentiable refinement
    return _shear_eft_refine(
        delta_base, data, box_size,
        k_gate, gate_width, R_env,
        gamma_diff, gamma_advect,
        beta_2lpt, gamma_env, cs2,
        deconv_amp, nudge_amp
    )

def _build_kgrid_rfft(nmesh: int, box_size: float):
    dx = box_size / nmesh
    kxv = 2.0 * jnp.pi * jnp.fft.fftfreq(nmesh, d=dx)
    kyv = 2.0 * jnp.pi * jnp.fft.fftfreq(nmesh, d=dx)
    kzv = 2.0 * jnp.pi * jnp.fft.rfftfreq(nmesh, d=dx)
    kx = kxv[:, None, None]
    ky = kyv[None, :, None]
    kz = kzv[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    k = jnp.sqrt(k2)
    inv_k2 = jnp.where(k2 > 0.0, 1.0 / k2, 0.0)
    return dx, kx, ky, kz, k2, k, inv_k2

def _shear_eft_refine(
    delta: jnp.ndarray,
    data: jnp.ndarray,
    box_size: float,
    k_gate: float,
    gate_width: float,
    R_env: float,
    gamma_diff: float,
    gamma_advect: float,
    beta_2lpt: float,
    gamma_env: float,
    cs2: float,
    deconv_amp: float,
    nudge_amp: float,
) -> jnp.ndarray:
    nmesh = delta.shape[0]
    dx, kx, ky, kz, k2, k, inv_k2 = _build_kgrid_rfft(nmesh, box_size)
    sshape = (nmesh, nmesh, nmesh)

    rfftn = jnp.fft.rfftn
    irfftn = lambda xk: jnp.fft.irfftn(xk, s=sshape).real

    # Base spectrum
    dk = rfftn(delta)

    # Low-k hard protection threshold (absolute, to prevent any [0.01,0.2] degradation)
    k_protect = jnp.array(0.21, dtype=k.dtype)

    # Smooth gate + hard clip below k_protect
    gate = jax.nn.sigmoid((k - k_gate) / gate_width)
    gate = jnp.where(k < k_protect, 0.0, gate)

    # --- Environment backbone ---
    env_window = jnp.exp(-0.5 * k2 * (R_env**2))
    delta_env_k = dk * env_window
    delta_env = irfftn(delta_env_k)

    # Large-scale potential for tides/flow
    phi_env_k = -delta_env_k * inv_k2

    # Large-scale tidal tensor (Hessian of phi_env)
    txx_L = irfftn(-kx * kx * phi_env_k)
    tyy_L = irfftn(-ky * ky * phi_env_k)
    tzz_L = irfftn(-kz * kz * phi_env_k)
    txy_L = irfftn(-kx * ky * phi_env_k)
    txz_L = irfftn(-kx * kz * phi_env_k)
    tyz_L = irfftn(-ky * kz * phi_env_k)

    # Shear invariant from large-scale tides
    s2_env = (txx_L - tyy_L)**2 + (txx_L - tzz_L)**2 + (tyy_L - tzz_L)**2 + 6.0 * (txy_L**2 + txz_L**2 + tyz_L**2)
    # Soft normalization to [0,1) to avoid huge dynamic range
    s2n = s2_env / (1.0 + s2_env)

    # Modulation weight: env + shear, but shear coupling derived from gamma_env (no extra tunable)
    # shear_c in [-0.5, 0.5]
    shear_c = 0.5 * jnp.tanh(gamma_env)
    w_mod = 1.0 + gamma_env * jnp.tanh(delta_env) + shear_c * jnp.tanh(3.0 * s2n)
    # keep positive-ish to avoid sign-flip instabilities
    w_mod = jnp.clip(w_mod, 0.25, 3.0)

    # --- Term A: Guided anisotropic diffusion (modulated) ---
    dx_delta = irfftn(1j * kx * dk)
    dy_delta = irfftn(1j * ky * dk)
    dz_delta = irfftn(1j * kz * dk)

    Jx = w_mod * (txx_L * dx_delta + txy_L * dy_delta + txz_L * dz_delta)
    Jy = w_mod * (txy_L * dx_delta + tyy_L * dy_delta + tyz_L * dz_delta)
    Jz = w_mod * (txz_L * dx_delta + tyz_L * dy_delta + tzz_L * dz_delta)

    Jx_k = rfftn(Jx)
    Jy_k = rfftn(Jy)
    Jz_k = rfftn(Jz)
    div_J_k = (1j * kx) * Jx_k + (1j * ky) * Jy_k + (1j * kz) * Jz_k
    term_diff_k = gamma_diff * div_J_k

    # --- Term B: Residual advection along large-scale flow (modulated) ---
    delta_resid_k = dk - delta_env_k
    dx_resid = irfftn(1j * kx * delta_resid_k)
    dy_resid = irfftn(1j * ky * delta_resid_k)
    dz_resid = irfftn(1j * kz * delta_resid_k)

    vx_L = irfftn(-1j * kx * phi_env_k)
    vy_L = irfftn(-1j * ky * phi_env_k)
    vz_L = irfftn(-1j * kz * phi_env_k)

    advect_resid_real = -(vx_L * dx_resid + vy_L * dy_resid + vz_L * dz_resid)
    term_advect_k = gamma_advect * rfftn(w_mod * advect_resid_real)

    # --- Term C: Local 2LPT (high-k) modulated by env+shear weight ---
    phi_k = -dk * inv_k2
    txx = irfftn(-kx * kx * phi_k)
    tyy = irfftn(-ky * ky * phi_k)
    tzz = irfftn(-kz * kz * phi_k)
    txy = irfftn(-kx * ky * phi_k)
    txz = irfftn(-kx * kz * phi_k)
    tyz = irfftn(-ky * kz * phi_k)

    src_2lpt = -(3.0 / 7.0) * (txx * tyy + txx * tzz + tyy * tzz - txy**2 - txz**2 - tyz**2)
    term_2lpt_k = beta_2lpt * rfftn(w_mod * src_2lpt)

    # --- Term D: Isotropic EFT counterterm (applied to residual only) ---
    # This targets small-scale mode-coupling while keeping large-scale backbone intact.
    term_eft_k = -cs2 * k2 * delta_resid_k

    # --- Term E: CIC deconvolution injection (spectral) ---
    arg = 0.5 * dx / jnp.pi
    Wcic = (jnp.sinc(kx * arg) * jnp.sinc(ky * arg) * jnp.sinc(kz * arg))**2
    deconv_fac = (jnp.clip(Wcic, 1e-6, 1.0) ** (-1.5)) - 1.0
    term_deconv_k = deconv_amp * dk * deconv_fac

    # --- Term F: Stable complex phase nudging toward arcsinh-compressed nonlinear density ---
    # Use a unit-complex ratio to estimate signed phase offset without angle(dk).
    delta_ref = jnp.arcsinh(0.7 * (data - 1.0))
    dk_ref = rfftn(delta_ref)

    eps = jnp.array(1e-6, dtype=jnp.float32)
    u_rec = dk / (jnp.abs(dk) + eps)
    u_ref = dk_ref / (jnp.abs(dk_ref) + eps)
    ratio = u_ref * jnp.conj(u_rec)  # ~ exp(i * dphi)
    dphi = jnp.arctan2(jnp.imag(ratio), jnp.real(ratio))  # in [-pi, pi]

    # k-dependent emphasis (fixed shape; no new tunables): mostly high-k
    k0_nudge = jnp.array(0.30, dtype=k.dtype)
    nudge_kweight = k2 / (k2 + k0_nudge**2)

    # Rotate reconstructed dk slightly toward reference phases
    dk_nudged = dk * jnp.exp(1j * (nudge_amp * nudge_kweight * dphi))
    diff_nudge_real = irfftn(dk_nudged) - irfftn(dk)
    term_nudge_k = rfftn(w_mod * diff_nudge_real)

    # --- Assembly with strict low-k safety ---
    corr_k = term_diff_k + term_advect_k + term_2lpt_k + term_eft_k + term_deconv_k + term_nudge_k
    dk_total = dk + gate * corr_k
    dk_total = jnp.where(k < k_protect, dk, dk_total)

    out = irfftn(dk_total)
    out = (out - jnp.mean(out)).astype(jnp.float32)
    return out

def apply_particle_reconstruction(
    sim_idx, nmesh, R_init, R_min, decay_factor,
    box_size, n_iterations
):
    import FastLoop_multi
    from Quijote_lib import readgadget

    snapPATH = f'/scratch/zangsh/Quijote_Simulations/Snapshots/fiducial/{sim_idx}/snapdir_004/snap_004'
    Position = (readgadget.read_block(snapPATH, "POS ", [1]) / 1e3).astype(np.float32)
    Size = Position.shape[0]
    Position0 = Position.copy()

    fn = np.fft.fftfreq(nmesh, 1.0/nmesh).astype(np.float32)
    rfn = np.fft.rfftfreq(nmesh, 1.0/nmesh).astype(np.float32)
    fnx, fny, fnz = fn[:,None,None], fn[None,:,None], rfn[None,None,:]
    k_ind = np.sqrt(fnx**2 + fny**2 + fnz**2).astype(np.float32)
    Kf = 2.0 * np.pi / box_size

    for i in range(n_iterations):
        delta = FastLoop_multi.CICPaint_multi(Position, nmesh, int(box_size), Size) - 1.0
        R = max((decay_factor**i) * R_init, R_min)
        dk = np.fft.rfftn(delta)
        dk *= np.exp(-0.5 * (k_ind * Kf)**2 * R**2).astype(np.float32)
        ksafe = np.where(k_ind > 0, k_ind, 1.0)
        psi = -1j * dk / (ksafe**2) / Kf
        Dis = np.empty((nmesh, nmesh, nmesh, 3), dtype=np.float32)
        Dis[..., 0] = np.fft.irfftn(psi * fnx).real
        Dis[..., 1] = np.fft.irfftn(psi * fny).real
        Dis[..., 2] = np.fft.irfftn(psi * fnz).real
        Position = FastLoop_multi.Shift_multi(Position, Dis, nmesh, int(box_size), Size)

    Density = FastLoop_multi.CICPaint_multi(Position, nmesh, int(box_size), Size)
    Dis_tot = FastLoop_multi.DisInter(Position0, Position, Density, nmesh, int(box_size), Size).astype(np.float32)
    dxk, dyk, dzk = np.fft.rfftn(Dis_tot[...,0]), np.fft.rfftn(Dis_tot[...,1]), np.fft.rfftn(Dis_tot[...,2])
    delta0k = (1j*Kf)*(fnx*dxk + fny*dyk + fnz*dzk)
    return np.fft.irfftn(delta0k).real.astype(np.float32)

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
    reconstructed_field = reconstruct(
        dm_density_field,
        R_init=10,
        R_min=1,
        decay_factor=0.5,
        n_iterations=8
    )
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
    recon_field = reconstruct(data, R_init=10, R_min=1, decay_factor=0.5, n_iterations=8)
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