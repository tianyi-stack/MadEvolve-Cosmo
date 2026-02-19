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

# TUNABLE: bias                   = 2.31525,    bounds=(0.8, 3.0),    method=autodiff
# TUNABLE: init_R_perp            = 7.35429,    bounds=(1.0, 15.0),   method=autodiff
# TUNABLE: init_R_para            = 5.57176,    bounds=(1.0, 15.0),   method=autodiff
# TUNABLE: gamma_adv              = 0.961606,        bounds=(0.0, 3.0),    method=autodiff
# TUNABLE: gamma_I2               = -3.55083,   bounds=(-5.0, 1.0),   method=autodiff
# TUNABLE: mix_coeff              = 2.42024,     bounds=(0.0, 4.0),    method=autodiff
# TUNABLE: hk_strength            = 5.34525,    bounds=(-6.0, 6.0),   method=autodiff
# TUNABLE: c0_threshold           = 0.56021,        bounds=(0.05, 0.7),   method=autodiff
# TUNABLE: c0_slope               = -1.43745,        bounds=(-2.0, 2.0),   method=autodiff
# TUNABLE: dewarp_ridge_pow       = -2.11529,       bounds=(-6.0, -0.5),  method=autodiff

def _fftn_batched(x):
    return jnp.fft.fftn(x, axes=(-3, -2, -1))

def _ifftn_batched(x):
    return jnp.fft.ifftn(x, axes=(-3, -2, -1))

def _compute_tri_invariants(s_x, s_y, s_z, kx_grid, ky_grid, kz_grid):
    sx_k = jnp.fft.fftn(s_x)
    sy_k = jnp.fft.fftn(s_y)
    sz_k = jnp.fft.fftn(s_z)

    grad_sx = jnp.real(_ifftn_batched(jnp.stack([1j * kx_grid * sx_k,
                                                1j * ky_grid * sx_k,
                                                1j * kz_grid * sx_k], axis=0)))
    grad_sy = jnp.real(_ifftn_batched(jnp.stack([1j * kx_grid * sy_k,
                                                1j * ky_grid * sy_k,
                                                1j * kz_grid * sy_k], axis=0)))
    grad_sz = jnp.real(_ifftn_batched(jnp.stack([1j * kx_grid * sz_k,
                                                1j * ky_grid * sz_k,
                                                1j * kz_grid * sz_k], axis=0)))

    T_xx, T_xy, T_xz = grad_sx[0], grad_sx[1], grad_sx[2]
    T_yx, T_yy, T_yz = grad_sy[0], grad_sy[1], grad_sy[2]
    T_zx, T_zy, T_zz = grad_sz[0], grad_sz[1], grad_sz[2]

    I1 = T_xx + T_yy + T_zz
    Tr_T_sq = (T_xx**2 + T_xy*T_yx + T_xz*T_zx +
               T_yx*T_xy + T_yy**2 + T_yz*T_zy +
               T_zx*T_xz + T_zy*T_yz + T_zz**2)
    I2 = 0.5 * (I1**2 - Tr_T_sq)
    I3 = (T_xx * (T_yy * T_zz - T_yz * T_zy) -
          T_xy * (T_yx * T_zz - T_yz * T_zx) +
          T_xz * (T_yx * T_zy - T_yy * T_zx))
    return I1, I2, I3

def _compute_tidal_invariant(d1, kx, ky, kz, k_sq_safe):
    d1_k = jnp.fft.fftn(d1)
    H_stack_k = jnp.stack([
        -(kx**2 / k_sq_safe) * d1_k,
        -(ky**2 / k_sq_safe) * d1_k,
        -(kz**2 / k_sq_safe) * d1_k,
        -(kx*ky / k_sq_safe) * d1_k,
        -(kx*kz / k_sq_safe) * d1_k,
        -(ky*kz / k_sq_safe) * d1_k
    ], axis=0)
    H = jnp.real(_ifftn_batched(H_stack_k))
    Hxx, Hyy, Hzz = H[0], H[1], H[2]
    Hxy, Hxz, Hyz = H[3], H[4], H[5]
    sum_sq = (Hxx**2 + Hyy**2 + Hzz**2 + 2*(Hxy**2 + Hxz**2 + Hyz**2))
    s2 = sum_sq - (1.0/3.0) * (d1**2)
    return s2

def reconstruct(
    data: jnp.ndarray,
    smoothing_scale_R_s: float = 6.96604,
    bias: float = 2.31525,
    box_size: float = 1000.0,
    init_R_perp: float = 7.35429,
    init_R_para: float = 5.57176,
    gamma_adv: float = 0.961606,
    gamma_I2: float = -3.55083,
    mix_coeff: float = 2.42024,
    hk_strength: float = 5.34525,
    c0_threshold: float = 0.56021,
    c0_slope: float = -1.43745,
    dewarp_ridge_pow: float = -2.11529,
) -> jnp.ndarray:
    n = data.shape[0]
    overd = data - 1.0

    # Fixed parameters
    gamma_div = 2.0
    gamma_I3 = 5.0
    stab_scale = 1.12
    guard_k = 0.01
    c_s2 = 0.12

    # Grid setup
    k1d = 2 * jnp.pi * jnp.fft.fftfreq(n, d=box_size / n)
    kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing='ij')
    k_perp_sq, k_par_sq = kx**2 + ky**2, kz**2
    k_sq = k_perp_sq + k_par_sq
    k_sq_safe = jnp.where(k_sq == 0.0, 1.0, k_sq)
    k_mag = jnp.sqrt(k_sq)
    H = box_size / n
    k_nyq = jnp.pi / H

    def kernel(Rp, Rl):
        arg = jnp.sqrt(k_perp_sq * Rp**2 + 1e-24) + jnp.sqrt(k_par_sq * Rl**2 + 1e-24)
        return jnp.exp(-0.5 * arg)

    def solve_disp(delta_k, Rp, Rl, eff_bias):
        fac = (-1j) * delta_k * kernel(Rp, Rl) / (eff_bias * k_sq_safe)
        s_vec = jnp.real(_ifftn_batched(jnp.stack([kx * fac, ky * fac, kz * fac], axis=0)))
        return s_vec[0], s_vec[1], s_vec[2]

    # --- Pass 1: Standard Zeldovich ---
    overd_k = jnp.fft.fftn(overd)
    s0x, s0y, s0z = solve_disp(overd_k, init_R_perp, init_R_para, bias)
    rec0 = _apply_differentiable_displacement(data.astype(jnp.float32), s0x, s0y, s0z, box_size, bias)

    # --- Physics Augmentation ---
    delta_res = rec0 - 1.0
    I1, I2, I3 = _compute_tri_invariants(s0x, s0y, s0z, kx, ky, kz)
    I2c = I2 - jnp.mean(I2)
    I3c = I3 - jnp.mean(I3)

    dr_k = jnp.fft.fftn(jnp.arcsinh(delta_res))
    grad_res = jnp.real(_ifftn_batched(jnp.stack([1j * kx * dr_k, 1j * ky * dr_k, 1j * kz * dr_k], axis=0)))
    adv = s0x * grad_res[0] + s0y * grad_res[1] + s0z * grad_res[2]

    # Tunable source composition (including advection)
    src1 = (delta_res + gamma_adv * adv + gamma_div * I1 +
            gamma_I2 * I2c + gamma_I3 * I3c)

    beta = jnp.maximum(stab_scale, 1e-6)
    src1_stab = jnp.arcsinh(beta * src1) / beta
    s1x, s1y, s1z = solve_disp(jnp.fft.fftn(src1_stab), init_R_perp * 0.05, init_R_para * 0.05, bias)

    # Secondary Channel: Tidal
    src2_stab = jnp.arcsinh(beta * 2.0 * I2c) / beta
    fac2 = 1j * jnp.fft.fftn(src2_stab) / k_sq_safe
    s2_vec = jnp.real(_ifftn_batched(jnp.stack([kx * fac2, ky * fac2, kz * fac2], axis=0)))
    s2x, s2y, s2z = s2_vec[0], s2_vec[1], s2_vec[2]

    # Gram-Schmidt Orthogonalization (s2 vs s1)
    w_hp = 1.0 / (1.0 + jnp.exp(-(k_mag - guard_k) / 0.12))
    w_hp = jnp.where(k_sq == 0.0, 0.0, w_hp)

    def apply_hp(v): return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(v) * w_hp))
    s1f = [apply_hp(v) for v in (s1x, s1y, s1z)]
    # Use higher power for s2 to isolate smaller scales
    def apply_hp2(v): return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(v) * w_hp**2))
    s2f = [apply_hp2(v) for v in (s2x, s2y, s2z)]

    dot_11 = jnp.mean(s1f[0]**2 + s1f[1]**2 + s1f[2]**2) + 1e-12
    dot_21 = jnp.mean(s2f[0]*s1f[0] + s2f[1]*s1f[1] + s2f[2]*s1f[2])
    proj_21 = dot_21 / dot_11

    s2o = [s2f[i] - proj_21 * s1f[i] for i in range(3)]

    mix_val = mix_coeff * (1.0 - 0.1 * jnp.tanh(I2c))
    fx = s0x + mix_val * (s1f[0] + c_s2 * s2o[0])
    fy = s0y + mix_val * (s1f[1] + c_s2 * s2o[1])
    fz = s0z + mix_val * (s1f[2] + c_s2 * s2o[2])

    # E-mode Projection & Deconvolution
    Fk = _fftn_batched(jnp.stack([fx, fy, fz], axis=0))
    div_F = kx * Fk[0] + ky * Fk[1] + kz * Fk[2]
    E_k = jnp.stack([kx * div_F / k_sq_safe, ky * div_F / k_sq_safe, kz * div_F / k_sq_safe], axis=0)
    E_k = jnp.where(k_sq == 0.0, 0.0, E_k)

    sinc_k = (jnp.sinc(kx * H / (2*jnp.pi)) * jnp.sinc(ky * H / (2*jnp.pi)) * jnp.sinc(kz * H / (2*jnp.pi)))**2
    taper = jnp.exp(-jnp.power(k_mag / (0.685 * k_nyq + 1e-9), 8.0))
    clean = jnp.where(k_sq == 0.0, 1.0, taper / jnp.where(sinc_k == 0.0, 1.0, sinc_k))

    disp_f = jnp.real(_ifftn_batched(E_k * clean))
    rec_final = _apply_differentiable_displacement(data.astype(jnp.float32), disp_f[0], disp_f[1], disp_f[2], box_size, bias)

    # --- High-k Refinement ---
    delta_base_k = jnp.fft.fftn(rec_final - 1.0) * clean
    delta_base_k = jnp.where(k_sq == 0.0, 0.0, delta_base_k)

    # Windowing
    w_hk_soft = (jax.nn.sigmoid((k_mag - 0.22) / 0.02)) ** 6
    w_hi = jnp.exp(-jnp.power(k_mag / (0.85 * k_nyq + 1e-9), 6.0))
    w_hk = jnp.where(k_mag <= 0.2, 0.0, w_hk_soft * w_hi)
    w_hk = jnp.where(k_sq == 0.0, 0.0, w_hk)
    
    aux_filter = taper / jnp.sqrt(jnp.where(sinc_k==0, 1.0, sinc_k))

    # --- Stage A: Tri-Basis De-Warping with Tunable Ridge ---
    d1 = jnp.arcsinh(beta * (rec_final - 1.0)) / beta
    d1 = d1 - jnp.mean(d1)
    
    # B1: Density^2
    b1_k = jnp.fft.fftn(d1**2 - jnp.mean(d1**2)) * aux_filter
    
    # B2: Grad^2
    d1_k_full = jnp.fft.fftn(d1)
    g_vec = jnp.real(_ifftn_batched(jnp.stack([1j*kx*d1_k_full, 1j*ky*d1_k_full, 1j*kz*d1_k_full], axis=0)))
    g2 = jnp.sum(g_vec**2, axis=0)
    b2_k = jnp.fft.fftn(g2 - jnp.mean(g2)) * aux_filter
    
    # B3: Tidal Scalar
    s2_field = _compute_tidal_invariant(d1, kx, ky, kz, k_sq_safe)
    b3_k = jnp.fft.fftn(s2_field - jnp.mean(s2_field)) * aux_filter

    band_centers = jnp.array([0.23, 0.30, 0.38, 0.48, 0.62], dtype=k_mag.dtype)
    band_sigma = jnp.array(0.075, dtype=k_mag.dtype)

    def _dewarp_scan(carry, kc):
        num_acc, den_acc = carry
        wb = w_hk * jnp.exp(-0.5 * jnp.square((k_mag - kc) / (band_sigma + 1e-6)))
        wb = jnp.where(k_sq == 0.0, 0.0, wb)

        # 3x3 System Construction
        bases = [b1_k, b2_k, b3_k]
        M = jnp.zeros((3, 3))
        T = jnp.zeros(3)
        
        # Unrolled construction
        for i in range(3):
            T = T.at[i].set(jnp.sum(wb * jnp.real(delta_base_k * jnp.conj(bases[i]))))
            for j in range(3):
                M = M.at[i, j].set(jnp.sum(wb * jnp.real(bases[i] * jnp.conj(bases[j]))))

        # Tunable Regularization
        ridge = (10.0**dewarp_ridge_pow) * (jnp.trace(M) + 1e-12)
        M = M + ridge * jnp.eye(3)
        
        coeffs = jnp.linalg.solve(M, T)
        delta_fit = coeffs[0]*b1_k + coeffs[1]*b2_k + coeffs[2]*b3_k
        return (num_acc + wb * delta_fit, den_acc + wb), None

    init_dw = (jnp.zeros_like(delta_base_k), jnp.zeros_like(k_mag))
    (num_dw, den_dw), _ = jax.lax.scan(_dewarp_scan, init_dw, band_centers)

    delta2_k = num_dw / (den_dw + 1e-12)
    delta2_k = jnp.where(k_sq == 0.0, 0.0, delta2_k)

    # Global Gating
    num_coh = jnp.sum(w_hk * jnp.real(delta_base_k * jnp.conj(delta2_k)))
    den_coh = jnp.sqrt(jnp.sum(w_hk * jnp.abs(delta_base_k)**2) * jnp.sum(w_hk * jnp.abs(delta2_k)**2) + 1e-12)
    coh_2 = num_coh / (den_coh + 1e-12)
    
    ratio = jnp.sqrt(jnp.sum(w_hk * jnp.abs(delta2_k)**2)) / (jnp.sqrt(jnp.sum(w_hk * jnp.abs(delta_base_k)**2)) + 1e-12)
    gate = jax.nn.sigmoid(10.0 * (coh_2 - 0.10)) * jax.nn.sigmoid(6.0 * (1.0 - ratio))

    delta_ref_k = delta_base_k - (w_hk * gate) * delta2_k
    delta_ref_k = jnp.where(k_sq == 0.0, 0.0, delta_ref_k)

    # --- Stage B: Fusion with K-Dependent Gating ---
    # Prepare tracers
    Dk = _fftn_batched(disp_f)
    delta_div_k = (-bias) * (1j * (kx * Dk[0] + ky * Dk[1] + kz * Dk[2])) * aux_filter
    
    logrho = jnp.log(jax.nn.softplus(rec_final) + 1e-6)
    delta_log_k = jnp.fft.fftn(logrho - jnp.mean(logrho)) * aux_filter
    
    dens_nl = jnp.arcsinh(beta * (rec_final - 1.0)) / beta
    delta_nl_k = jnp.fft.fftn(dens_nl - jnp.mean(dens_nl)) * aux_filter
    
    I3_src = jnp.arcsinh(beta * I3c) / beta
    delta_i3_k = jnp.fft.fftn(I3_src - jnp.mean(I3_src)) * aux_filter

    s0k = _fftn_batched(jnp.stack([s0x, s0y, s0z], axis=0))
    delta_div0_k = (-bias) * (1j * (kx * s0k[0] + ky * s0k[1] + kz * s0k[2])) * aux_filter
    
    # Scale dependent bias tracer
    bias_eff = jnp.maximum(bias * (1.0 - 0.15 * k_sq), 0.2)
    W_alt = jnp.exp(-0.5 * (jnp.power(k_perp_sq * init_R_perp**2 + 1e-24, 0.675) + 
                            jnp.power(k_par_sq * init_R_para**2 + 1e-24, 0.675)))
    delta_sdbias_k = (bias / bias_eff) * overd_k * W_alt * aux_filter
    
    tracers = [delta_div_k, delta_log_k, delta_nl_k, delta_i3_k, delta_div0_k, delta_sdbias_k]
    
    a_sig = jnp.array(10.0, dtype=k_mag.dtype)

    def _fusion_scan(carry, kc):
        num_acc, den_acc, conf_acc = carry
        wb = w_hk * jnp.exp(-0.5 * jnp.square((k_mag - kc) / (band_sigma + 1e-6)))
        wb = jnp.where(k_sq == 0.0, 0.0, wb)
        
        # Adaptive Threshold
        c0_eff = c0_threshold + c0_slope * (kc - 0.35)

        hybrid = jnp.zeros_like(delta_ref_k)
        w_sum = jnp.zeros_like(kc)

        for t_k in tracers:
            num = jnp.sum(wb * delta_ref_k * jnp.conj(t_k))
            den = jnp.sum(wb * jnp.abs(t_k)**2) + 1e-12
            coeff = num / den
            pred = coeff * t_k
            
            # Real coherence for strict correlation
            num_coh = jnp.sum(wb * jnp.real(delta_ref_k * jnp.conj(pred)))
            den_coh = jnp.sqrt(jnp.sum(wb * jnp.abs(delta_ref_k)**2) * jnp.sum(wb * jnp.abs(pred)**2) + 1e-12)
            coh = num_coh / (den_coh + 1e-12)
            
            w = jax.nn.sigmoid(a_sig * (coh - c0_eff))
            hybrid = hybrid + w * pred
            w_sum = w_sum + w
            
        hybrid = hybrid / (w_sum + 1e-6)
        
        # Wiener Blend
        Pxx = jnp.sum(wb * jnp.abs(delta_ref_k)**2)
        Pyy = jnp.sum(wb * jnp.abs(hybrid)**2)
        Pxy = jnp.sum(wb * jnp.real(delta_ref_k * jnp.conj(hybrid)))
        
        denom = (Pxx + Pyy - 2.0*Pxy) + 1e-12
        wx = (Pyy - Pxy) / denom
        wy = (Pxx - Pxy) / denom
        
        coh_band = Pxy / (jnp.sqrt(Pxx*Pyy + 1e-12) + 1e-12)
        gate_band = jax.nn.sigmoid(8.0 * (coh_band - 0.20))
        
        # Apply gate
        wx = gate_band * wx + (1.0 - gate_band) * 1.0
        wy = gate_band * wy * jax.nn.sigmoid(4.0 * (coh_band - 0.20))
        
        band_est = wx * delta_ref_k + wy * hybrid
        
        return (num_acc + wb * band_est, den_acc + wb, conf_acc + wb * jnp.clip(coh_band, 0.0, 1.0)), None

    init_fus = (jnp.zeros_like(delta_ref_k), jnp.zeros_like(k_mag), jnp.zeros_like(k_mag))
    (num_fus, den_fus, conf_fus), _ = jax.lax.scan(_fusion_scan, init_fus, band_centers)
    
    est_k = num_fus / (den_fus + 1e-12)
    conf_k = conf_fus / (den_fus + 1e-12)

    eta = jax.nn.sigmoid(hk_strength)
    damp = jax.nn.softplus(hk_strength)
    
    trust = w_hk * conf_k**2
    delta_fused_k = delta_ref_k + (eta * trust) * (est_k - delta_ref_k)
    delta_fused_k = delta_fused_k * jnp.exp(-damp * w_hk * (1.0 - conf_k)**2)
    delta_fused_k = jnp.where(k_sq == 0.0, 0.0, delta_fused_k)

    # --- Hard Low-k Lock ---
    final_k = jnp.where(k_mag <= 0.2, delta_base_k, delta_fused_k)
    final_k = jnp.where(k_sq == 0.0, 0.0, final_k)

    return (jnp.real(jnp.fft.ifftn(final_k)) + 1.0).astype(data.dtype)

def _get_interpolated_displacement(pos, dx, dy, dz, nmesh, box):
    cell = box/nmesh
    x, y, z = pos[:,0]/cell, pos[:,1]/cell, pos[:,2]/cell
    ix, iy, iz = jnp.floor(x).astype(int)%nmesh, jnp.floor(y).astype(int)%nmesh, jnp.floor(z).astype(int)%nmesh
    fx, fy, fz = x - jnp.floor(x), y - jnp.floor(y), z - jnp.floor(z)
    ox, oy, oz = jnp.zeros_like(fx), jnp.zeros_like(fy), jnp.zeros_like(fz)
    for di in range(2):
        wx = jnp.where(di, fx, 1-fx)
        for dj in range(2):
            wy = jnp.where(dj, fy, 1-fy)
            for dk in range(2):
                wz = jnp.where(dk, fz, 1-fz)
                w = wx*wy*wz
                ix0, iy0, iz0 = (ix+di)%nmesh, (iy+dj)%nmesh, (iz+dk)%nmesh
                ox += dx[ix0, iy0, iz0] * w
                oy += dy[ix0, iy0, iz0] * w
                oz += dz[ix0, iy0, iz0] * w
    return ox, oy, oz

def _differentiable_shift(pos, dx, dy, dz, nmesh, box):
    sx, sy, sz = _get_interpolated_displacement(pos, dx, dy, dz, nmesh, box)
    return jnp.mod(pos + jnp.stack([sx, sy, sz], axis=-1), box).astype(jnp.float32)

def _differentiable_paint(pos, vals, nmesh, box):
    grid = jnp.zeros((nmesh, nmesh, nmesh), dtype=jnp.float32)
    cell = box/nmesh
    x, y, z = pos[:,0]/cell, pos[:,1]/cell, pos[:,2]/cell
    ix0, iy0, iz0 = jnp.floor(x).astype(int), jnp.floor(y).astype(int), jnp.floor(z).astype(int)
    dx, dy, dz = x-ix0, y-iy0, z-iz0
    for di in range(2):
        wx = jnp.where(di, dx, 1-dx)
        for dj in range(2):
            wy = jnp.where(dj, dy, 1-dy)
            for dk in range(2):
                wz = jnp.where(dk, dz, 1-dz)
                w = wx*wy*wz
                grid = grid.at[(ix0+di)%nmesh, (iy0+dj)%nmesh, (iz0+dk)%nmesh].add(vals*w)
    return grid

def _apply_differentiable_displacement(dens, dx, dy, dz, box, bias):
    n = dens.shape[0]
    H = box/n
    coords = jnp.stack(jnp.meshgrid(jnp.arange(n)*H, jnp.arange(n)*H, jnp.arange(n)*H, indexing='ij'),
                      axis=-1).reshape(-1, 3).astype(jnp.float32)
    pos2 = _differentiable_shift(coords, dx, dy, dz, n, box)
    overd = dens.reshape(-1) - 1.0
    grid_d = _differentiable_paint(pos2, overd, n, box)
    grid_u = _differentiable_paint(pos2, jnp.ones_like(overd), n, box) - 1.0
    return (grid_d - grid_u)/bias + 1.0

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
    reconstructed_field = reconstruct(dm_density_field, smoothing_scale_R_s=6.96604, bias=2.31525)
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
    R0 = jax.numpy.array(10.0, dtype=jax.numpy.float32)
    bias0 = jax.numpy.array(1.938253, dtype=jax.numpy.float32)
    data, gt_field = read_files(0)

    # Test gradient w.r.t. smoothing scale
    dr_dR = jax.grad(run_exp, 0)(R0, bias0, data, gt_field)
    print(f'dr_avg / dR (at R={float(R0):.2f}) = {float(dr_dR):.6f}')

    # Test gradient w.r.t. bias
    dr_dbias = jax.grad(run_exp, 1)(R0, bias0, data, gt_field)
    print(f'dr_avg / dbias (at bias={float(bias0):.4f}) = {float(dr_dbias):.6f}')


def run_exp(smoothing_scale_R_s, bias, data, gt_field):
    """
    Run reconstruction and compute metric (for autodiff).
    """
    reconstructed_field = reconstruct(data, smoothing_scale_R_s, bias)
    r_k, k_centers, avg_r_bao = compute_rk_metrics(reconstructed_field, gt_field)
    return avg_r_bao


if __name__ == "__main__":
    main()