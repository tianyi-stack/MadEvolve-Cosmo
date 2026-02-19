"""
Initial Tidal Reconstruction Program

This module implements the tidal reconstruction algorithm for cosmological
density field reconstruction using the tidal tensor method.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq


# EVOLVE-BLOCK-START

# --- Parameters Inherited from Parent ---
# Wavelet Decomposition
# TUNABLE: wavelet_scale_perp_1 = 2.81024, bounds=(0.5, 4.0), method=grid
# TUNABLE: wavelet_scale_par_1 = 0.0219371, bounds=(0.01, 4.0), method=grid
# TUNABLE: wavelet_scale_perp_2 = 1.37686, bounds=(0.2, 3.0), method=grid
# TUNABLE: wavelet_scale_par_2 = 1.4916, bounds=(0.01, 3.0), method=grid
# TUNABLE: k_min_par = 0.186274, bounds=(0.0, 0.4), method=grid
# TUNABLE: wavelet_mix_snr_pow = 2.33961, bounds=(0.5, 3.0), method=grid
# Potential Calculation
# TUNABLE: spectral_index = 1.53583, bounds=(-1.0, 2.5), method=grid
# TUNABLE: f_ani = 1.45076, bounds=(-0.9, 5.0), method=grid
# TUNABLE: reg_value = 0.0591252, bounds=(0.001, 0.1), method=grid
# TUNABLE: pot_power_perp = 1.64769, bounds=(0.5, 2.5), method=grid
# TUNABLE: pot_power_par = 0.788897, bounds=(0.5, 2.5), method=grid
# Projection & Band-specific weights
# TUNABLE: w_par_1 = 0.523645, bounds=(0.0, 3.0), method=grid
# TUNABLE: w_cross_1 = 2.9352, bounds=(0.0, 3.0), method=grid
# TUNABLE: w_par_2 = 2.74627, bounds=(0.0, 3.0), method=grid
# TUNABLE: w_cross_2 = 1.4405, bounds=(0.0, 3.0), method=grid
# TUNABLE: kpar_weight_proj = 4.0546, bounds=(0.2, 5.0), method=grid
# Final Fusion & Gating
# TUNABLE: quad_weight = 0.312289, bounds=(0.0, 2.0), method=grid
# TUNABLE: lin_weight = 0.254311, bounds=(0.0, 2.0), method=grid
# TUNABLE: lin_smooth_perp = 1.17681, bounds=(0.0, 5.0), method=grid
# TUNABLE: lin_smooth_par = 4.95358, bounds=(0.0, 5.0), method=grid
# TUNABLE: kz_gate = 0.155033, bounds=(0.05, 1.2), method=grid
# TUNABLE: gate_pow = 2.89852, bounds=(0.5, 6.0), method=grid
# TUNABLE: kperp_gate_coupling = 0.221573, bounds=(0.0, 1.0), method=grid
# TUNABLE: snr_amp = 0.273968, bounds=(0.01, 0.5), method=grid
# TUNABLE: gate_hybrid_mix = 0.16981, bounds=(0.0, 1.0), method=grid

# --- New/Modified Parameters (novel architecture) ---
# Band-specific Saturation
# TUNABLE: sat_amp_1 = 4.98676, bounds=(0.5, 10.0), method=grid
# TUNABLE: sat_amp_2 = 7.76562, bounds=(0.5, 10.0), method=grid
# Band-specific Trace Weight (kept for inheritance)
# TUNABLE: w_trace_1 = 0.816982, bounds=(-1.0, 1.0), method=grid
# TUNABLE: w_trace_2 = 0.668268, bounds=(-1.0, 1.0), method=grid
# Fully Anisotropic Output Tapering
# TUNABLE: out_kpar_cut = 8.25062, bounds=(0.5, 10.0), method=grid
# TUNABLE: out_par_pow = 5.80199, bounds=(0.5, 6.0), method=grid
# TUNABLE: out_kperp_cut = 4.74921, bounds=(0.5, 10.0), method=grid
# TUNABLE: out_perp_pow = 1.95208, bounds=(0.5, 6.0), method=grid
# Decoupled Power-Law Noise Model
# TUNABLE: noise_k_perp_fac = 1.39414, bounds=(0.0, 2.0), method=grid
# TUNABLE: noise_pow_par = 1.50069, bounds=(0.5, 2.5), method=grid
# TUNABLE: noise_pow_perp = 1.6218, bounds=(0.5, 2.5), method=grid

# 1) Scale-Dependent Anisotropy for Potential
# TUNABLE: f_ani_k_scale = 2.92902, bounds=(0.0, 5.0), method=grid
# TUNABLE: f_ani_k_pow = 1.10938, bounds=(0.5, 2.5), method=grid

# 2) Band-Specific Divergence Damping (perp vs parallel/cross)
# Band 1
# TUNABLE: div_damp_k_perp_1 = 2.87363, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_k_par_1 = 4.37408, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_pow_1 = 2.34834, bounds=(0.5, 6.0), method=grid
# TUNABLE: div_damp_k_perp_par_1 = 1.9083, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_k_par_par_1 = 1.52338, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_pow_par_1 = 3.83011, bounds=(0.5, 6.0), method=grid
# Band 2
# TUNABLE: div_damp_k_perp_2 = 1.10719, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_k_par_2 = 0.505564, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_pow_2 = 5.22597, bounds=(0.5, 6.0), method=grid
# TUNABLE: div_damp_k_perp_par_2 = 3.29744, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_k_par_par_2 = 1.24861, bounds=(0.05, 5.0), method=grid
# TUNABLE: div_damp_pow_par_2 = 5.76338, bounds=(0.5, 6.0), method=grid

# 3) Anisotropic Trace Weights (per band)
# TUNABLE: w_trace_perp_1 = -0.683257, bounds=(-1.0, 1.0), method=grid
# TUNABLE: w_trace_par_1 = -0.635406, bounds=(-1.0, 1.0), method=grid
# TUNABLE: w_trace_perp_2 = 0.353299, bounds=(-1.0, 1.0), method=grid
# TUNABLE: w_trace_par_2 = -0.865571, bounds=(-1.0, 1.0), method=grid

# 4) Full Anisotropic Wedge for Inter-Band Fusion
# TUNABLE: band_mix_kz_gate = 0.936203, bounds=(0.05, 1.2), method=grid
# TUNABLE: band_mix_kperp_coupling = 0.816507, bounds=(0.0, 1.0), method=grid
# TUNABLE: band_mix_gate_pow = 3.5234, bounds=(0.5, 6.0), method=grid
# TUNABLE: band_mix_gate_slope = 0.511776, bounds=(-0.5, 1.0), method=grid
# (inherit) # TUNABLE: band_mix_gate_par = 0.0978777, bounds=(0.05, 0.6), method=grid
# (inherit) # TUNABLE: band_mix_hybrid_ratio = 0.177992, bounds=(0.0, 1.0), method=grid

# 5) Gated Additive Signal Injection (Trispectrum Boost)
# TUNABLE: inject_amp = 1.44264, bounds=(0.0, 2.0), method=grid
# TUNABLE: inject_pow = 1.57218, bounds=(0.5, 3.5), method=grid
# TUNABLE: inject_kperp_cut = 0.0734451, bounds=(0.05, 2.0), method=grid
# TUNABLE: inject_kperp_pow = 3.63214, bounds=(0.5, 6.0), method=grid


def run_reconstruction(
    data: np.ndarray,
    # Wavelet bands
    wavelet_scale_perp_1: float = 2.0, wavelet_scale_par_1: float = 0.8,
    wavelet_scale_perp_2: float = 0.7, wavelet_scale_par_2: float = 0.1,
    k_min_par: float = 0.07, wavelet_mix_snr_pow: float = 1.0,
    # Potential model
    spectral_index: float = 0.7, f_ani: float = -0.25,
    reg_value: float = 0.1, pot_power_perp: float = 1.0, pot_power_par: float = 1.0,
    # Saturation and trace weights
    sat_amp_1: float = 3.4, sat_amp_2: float = 3.4,
    w_trace_1: float = 0.85, w_trace_2: float = 0.85,
    # Divergence projection weights
    w_par_1: float = 1.0, w_cross_1: float = 1.0,
    w_par_2: float = 1.0, w_cross_2: float = 1.0,
    # Projection and output taper
    kpar_weight_proj: float = 1.0,
    out_kpar_cut: float = 10.0, out_par_pow: float = 2.0,
    out_kperp_cut: float = 10.0, out_perp_pow: float = 2.0,
    # Final fusion, linear smooth, and wedge
    quad_weight: float = 1.0, lin_weight: float = 1.0,
    lin_smooth_perp: float = 0.0, lin_smooth_par: float = 0.0,
    kz_gate: float = 0.6, gate_pow: float = 2.0, kperp_gate_coupling: float = 0.1,
    # Noise model for SNR gating
    noise_k_perp_fac: float = 0.5, noise_pow_par: float = 1.0, noise_pow_perp: float = 1.0,
    snr_amp: float = 0.1, gate_hybrid_mix: float = 0.5,
    # Band mixing hybrid weight (inherited)
    band_mix_gate_par: float = 0.3, band_mix_hybrid_ratio: float = 0.5,
    # New: scale-dependent anisotropy
    f_ani_k_scale: float = 0.0, f_ani_k_pow: float = 1.0,
    # New: band-specific divergence damping
    div_damp_k_perp_1: float = 1.25, div_damp_k_par_1: float = 0.65, div_damp_pow_1: float = 2.0,
    div_damp_k_perp_par_1: float = 0.75, div_damp_k_par_par_1: float = 0.45, div_damp_pow_par_1: float = 2.0,
    div_damp_k_perp_2: float = 0.95, div_damp_k_par_2: float = 0.55, div_damp_pow_2: float = 2.2,
    div_damp_k_perp_par_2: float = 0.65, div_damp_k_par_par_2: float = 0.40, div_damp_pow_par_2: float = 2.2,
    # New: anisotropic trace weights
    w_trace_perp_1: float = 0.0, w_trace_par_1: float = 0.0,
    w_trace_perp_2: float = 0.0, w_trace_par_2: float = 0.0,
    # New: anisotropic wedge mixing for bands
    band_mix_kz_gate: float = 0.32, band_mix_kperp_coupling: float = 0.45,
    band_mix_gate_pow: float = 1.65, band_mix_gate_slope: float = 0.12,
    # New: gated additive signal injection (trispectrum boost)
    inject_amp: float = 0.12, inject_pow: float = 1.15,
    inject_kperp_cut: float = 0.35, inject_kperp_pow: float = 2.0
) -> np.ndarray:
    """
    Trispectrum-boosted hierarchical tidal reconstruction with:
    - Scale-dependent anisotropy in the potential
    - Band-specific divergence damping
    - Anisotropic trace weighting per band
    - Full wedge-based inter-band fusion
    - Gated additive signal injection in the wedge transition region
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")

    nmesh = data.shape[0]
    boxsize = 1000.0
    kf = 2.0 * np.pi / boxsize
    eps = np.float32(1e-20)

    # --- K-SPACE SETUP ---
    fn = fftfreq(nmesh, 1.0 / nmesh).astype(np.float32)
    k_x = fn[:, None, None] * kf
    k_y = fn[None, :, None] * kf
    k_z = fn[None, None, :] * kf

    k_perp_sq = (k_x**2 + k_y**2).astype(np.float32)
    k_par_sq = (k_z**2).astype(np.float32)
    k_sq = (k_perp_sq + k_par_sq).astype(np.float32)

    k_par = np.abs(k_z).astype(np.float32)
    k_perp = np.sqrt(k_perp_sq + eps, dtype=np.float32)

    # --- INPUT FIELD ---
    delta = (data - 1.0).astype(np.float32)
    delta_k = fftn(delta).astype(np.complex64)

    # --- STAGE 1: ANISOTROPIC WAVELET DECOMPOSITION ---
    w_highpass_par = (1.0 - np.exp(-0.5 * (k_par / (k_min_par * kf + eps))**2)).astype(np.float32)
    w_wavelet_1 = np.exp(-0.5 * (k_perp_sq * wavelet_scale_perp_1**2 + k_par_sq * wavelet_scale_par_1**2)).astype(np.float32)
    w_wavelet_2 = np.exp(-0.5 * (k_perp_sq * wavelet_scale_perp_2**2 + k_par_sq * wavelet_scale_par_2**2)).astype(np.float32)

    delta_k_band1 = (delta_k * w_wavelet_1 * w_highpass_par).astype(np.complex64)
    delta_k_band2 = (delta_k * w_wavelet_2 * w_highpass_par).astype(np.complex64)

    # --- STAGE 2: SPECTRAL POTENTIAL WITH SCALE-DEPENDENT ANISOTROPY ---
    eps_reg = (reg_value * kf)**2
    # f_ani_eff = f_ani * (1 + f_ani_k_scale * k_perp^f_ani_k_pow)
    f_ani_eff = (f_ani * (1.0 + f_ani_k_scale * np.power(k_perp + eps, f_ani_k_pow))).astype(np.float32)

    term_perp_pot = np.power(k_perp_sq + eps, pot_power_perp, dtype=np.float32)
    term_par_pot = (1.0 + f_ani_eff) * np.power(k_par_sq + eps, pot_power_par, dtype=np.float32)
    denom_pot_base = (np.abs(term_perp_pot + term_par_pot) + eps_reg).astype(np.float32)

    weight_spectral = np.power(denom_pot_base, -0.5 * spectral_index).astype(np.float32)
    weight_spectral[0, 0, 0] = 0.0

    # Projection denominator
    k_sq_eff = (k_perp_sq + kpar_weight_proj * k_par_sq).astype(np.float32)
    k_sq_eff_safe = k_sq_eff.copy()
    k_sq_eff_safe[0, 0, 0] = 1.0

    # Output tapering (anisotropic)
    denom_out = (1.0 +
                 np.power(k_par / (out_kpar_cut + eps), out_par_pow) +
                 np.power(k_perp / (out_kperp_cut + eps), out_perp_pow)).astype(np.float32)
    w_out = (1.0 / denom_out).astype(np.float32)

    # --- Helper: per-band processing closure ---
    def process_band(delta_k_band: np.ndarray,
                     sat_amp: float, w_par: float, w_cross: float,
                     w_trace_iso: float, w_trace_perp: float, w_trace_par: float,
                     # damping params
                     dkp_perp: float, dkp_par: float, dpow: float,
                     dkp_perp_par: float, dkp_par_par: float, dpow_par: float) -> np.ndarray:
        # Potential gradient to get vector field
        common_factor = (1j * delta_k_band * weight_spectral).astype(np.complex64)
        vec_x = ifftn(k_x * common_factor).real.astype(np.float32)
        vec_y = ifftn(k_y * common_factor).real.astype(np.float32)
        vec_z = ifftn(k_z * common_factor).real.astype(np.float32)

        # Band-specific robust saturation
        lim_x = sat_amp * (np.std(vec_x, dtype=np.float64).astype(np.float32) + 1e-10)
        lim_y = sat_amp * (np.std(vec_y, dtype=np.float64).astype(np.float32) + 1e-10)
        lim_z = sat_amp * (np.std(vec_z, dtype=np.float64).astype(np.float32) + 1e-10)
        vec_x = (lim_x * np.tanh(vec_x / (lim_x + 1e-12))).astype(np.float32)
        vec_y = (lim_y * np.tanh(vec_y / (lim_y + 1e-12))).astype(np.float32)
        vec_z = (lim_z * np.tanh(vec_z / (lim_z + 1e-12))).astype(np.float32)

        # Quadratic tensor (Fourier)
        t_xx_k = fftn((vec_x * vec_x).astype(np.float32)).astype(np.complex64)
        t_yy_k = fftn((vec_y * vec_y).astype(np.float32)).astype(np.complex64)
        t_zz_k = fftn((vec_z * vec_z).astype(np.float32)).astype(np.complex64)
        t_xy_k = fftn((vec_x * vec_y).astype(np.float32)).astype(np.complex64)
        t_xz_k = fftn((vec_x * vec_z).astype(np.float32)).astype(np.complex64)
        t_yz_k = fftn((vec_y * vec_z).astype(np.float32)).astype(np.complex64)

        # Anisotropic divergence components
        div_perp = (k_x**2 * t_xx_k + k_y**2 * t_yy_k + 2.0 * k_x * k_y * t_xy_k).astype(np.complex64)
        div_par = (k_z**2 * t_zz_k).astype(np.complex64)
        div_cross = (2.0 * k_x * k_z * t_xz_k + 2.0 * k_y * k_z * t_yz_k).astype(np.complex64)

        # Band-specific divergence damping (perp vs parallel/cross)
        w_damp_perp = (1.0 / (1.0 +
                              np.power(k_perp / (dkp_perp + eps), dpow) +
                              np.power(k_par / (dkp_par + eps), dpow))).astype(np.float32)
        w_damp_par = (1.0 / (1.0 +
                             np.power(k_perp / (dkp_perp_par + eps), dpow_par) +
                             np.power(k_par / (dkp_par_par + eps), dpow_par))).astype(np.float32)

        div_perp *= w_damp_perp
        div_par *= w_damp_par
        div_cross *= w_damp_par

        term_aniso = (div_perp + w_par * div_par + w_cross * div_cross) / k_sq_eff_safe

        # Anisotropic trace term
        trace_iso = (t_xx_k + t_yy_k + t_zz_k).astype(np.complex64)
        trace_aniso = (w_trace_perp * (t_xx_k + t_yy_k) + w_trace_par * t_zz_k).astype(np.complex64)
        quad_k = (term_aniso + w_trace_iso * trace_iso + trace_aniso).astype(np.complex64)

        quad_k[0, 0, 0] = 0.0
        return (quad_k * w_out).astype(np.complex64)

    # Process both bands
    rec_k_quad_band1 = process_band(delta_k_band1, sat_amp_1, w_par_1, w_cross_1,
                                    w_trace_1, w_trace_perp_1, w_trace_par_1,
                                    div_damp_k_perp_1, div_damp_k_par_1, div_damp_pow_1,
                                    div_damp_k_perp_par_1, div_damp_k_par_par_1, div_damp_pow_par_1)

    rec_k_quad_band2 = process_band(delta_k_band2, sat_amp_2, w_par_2, w_cross_2,
                                    w_trace_2, w_trace_perp_2, w_trace_par_2,
                                    div_damp_k_perp_2, div_damp_k_par_2, div_damp_pow_2,
                                    div_damp_k_perp_par_2, div_damp_k_par_par_2, div_damp_pow_par_2)

    # --- STAGE 3: HIERARCHICAL BAND FUSION WITH FULL ANISOTROPIC WEDGE ---
    # Anisotropic noise proxy for SNR
    anisotropic_noise_power_spec = (np.power(k_par_sq + eps, noise_pow_par) +
                                    noise_k_perp_fac * np.power(k_perp_sq + eps, noise_pow_perp)).astype(np.float32)

    snr1 = np.abs(rec_k_quad_band1).astype(np.float32) / (anisotropic_noise_power_spec + eps)
    snr2 = np.abs(rec_k_quad_band2).astype(np.float32) / (anisotropic_noise_power_spec + eps)
    total_snr = (snr1 + snr2 + eps).astype(np.float32)

    w1 = np.power(snr1 / total_snr, wavelet_mix_snr_pow, dtype=np.float32)
    w2 = np.power(snr2 / total_snr, wavelet_mix_snr_pow, dtype=np.float32)
    mix_ratio_snr = (w2 / (w1 + w2 + eps)).astype(np.float32)  # weight of band2

    # Full anisotropic wedge gate for band fusion (favor small-scale band2 inside wedge)
    band_mix_thresh = (band_mix_kz_gate * (1.0 + band_mix_kperp_coupling * k_perp_sq) +
                       band_mix_gate_slope * k_perp).astype(np.float32)
    band_mix_thresh = np.maximum(band_mix_thresh, 1e-5).astype(np.float32)
    mix_ratio_geom = (1.0 / (1.0 + np.power(k_par / band_mix_thresh, band_mix_gate_pow))).astype(np.float32)

    # Hybrid mixing
    band_mix_ratio = (band_mix_hybrid_ratio * mix_ratio_geom +
                      (1.0 - band_mix_hybrid_ratio) * mix_ratio_snr).astype(np.float32)

    mixed_rec_k_quad = (rec_k_quad_band1 * (1.0 - band_mix_ratio) +
                        rec_k_quad_band2 * band_mix_ratio).astype(np.complex64)

    # --- STAGE 4: FINAL FUSION (QUADRATIC + LINEAR) WITH GATED INJECTION ---
    # Linear fallback (smoothed)
    w_lin_smooth = np.exp(-0.5 * (k_perp_sq * lin_smooth_perp**2 + k_par_sq * lin_smooth_par**2), dtype=np.float32)
    delta_k_linear = (delta_k * w_lin_smooth).astype(np.complex64)
    delta_k_linear[0, 0, 0] = 0.0

    # Geometric wedge gate
    effective_kz_gate = (kz_gate * (1.0 + kperp_gate_coupling * k_perp_sq)).astype(np.float32)
    gate_thresh_geom = np.maximum(effective_kz_gate, 1e-6).astype(np.float32)
    gate_wedge = (1.0 / (1.0 + np.power(k_par / gate_thresh_geom, gate_pow))).astype(np.float32)

    # SNR gate
    snr_raw_final = np.abs(mixed_rec_k_quad).astype(np.float32) / (anisotropic_noise_power_spec + eps)
    gate_snr = (1.0 - np.exp(-np.power(snr_raw_final / (snr_amp + eps), 2.0))).astype(np.float32)
    gate_snr = np.clip(gate_snr, 0.0, 1.0).astype(np.float32)

    # Hybrid final gating
    combined_gate = (gate_hybrid_mix * gate_wedge + (1.0 - gate_hybrid_mix) * gate_snr).astype(np.float32)

    rec_k = (quad_weight * combined_gate * mixed_rec_k_quad +
             lin_weight * (1.0 - combined_gate) * delta_k_linear).astype(np.complex64)

    # Targeted trispectrum-like signal injection (only where wedge active and SNR low)
    if inject_amp > 1e-8:
        high_perp_gate = (1.0 - np.exp(-np.power(k_perp / (inject_kperp_cut + eps), inject_kperp_pow))).astype(np.float32)
        # Core scaling emphasizes low k_par relative to k_perp^2 while remaining finite at k=0
        inj_core = np.power((k_par + (1e-6 * kf)) / (k_perp_sq + (1e-6 * kf**2)), inject_pow, dtype=np.float32)
        inj_gate = (gate_wedge * (1.0 - gate_snr) * high_perp_gate).astype(np.float32)
        delta_k_correction = (inject_amp * inj_core * inj_gate).astype(np.float32) * delta_k.astype(np.complex64) * w_out.astype(np.float32)
        rec_k = (rec_k + delta_k_correction.astype(np.complex64)).astype(np.complex64)

    rec_k[0, 0, 0] = 0.0

    delta_rec = ifftn(rec_k).real.astype(np.float32)

    return (delta_rec + 1.0).astype(np.float32)

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