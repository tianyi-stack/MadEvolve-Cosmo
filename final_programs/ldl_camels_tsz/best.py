"""
Evolve the model architecture for predicting stellar mass from dark matter particles.

This example uses CAMELS CV_0 simulation with 16x downsampling and Nmesh=64.
The model architecture is based on Lagrangian Deep Learning (LDL).
"""

import numpy as np
from mpi4py import MPI
from vmad.lib import fastpm, unary, linalg, mpi
from vmad import autooperator, operator
from vmad.core.symbol import Literal
from vmad.core.stdlib import eval as vmad_eval
from vmad.core.stdlib.operators import binary
from scipy.optimize import minimize
from pmesh.pm import ParticleMesh
from nbodykit.lab import FieldMesh
import sys
import os
import time
import gc
from collections import OrderedDict

# Add LDL path to import data loading utilities
# Use absolute path to ensure it works when file is copied to different locations
LDL_PATH = '/home/tianyi/real_tianyi/code/LDL'
sys.path.insert(0, LDL_PATH)
from readTNG import load_TNG_map, load_TNG_data, get_cosmology


# ==================== Fixed Operators ====================
# These operators are NOT evolved, they provide basic building blocks

@operator
class ReLU:
    """Standard ReLU activation"""
    aout = {'y': 'RealField'}
    ain = {'x': 'RealField'}

    def apl(node, x):
        y = x.copy()
        y[y <= 0] = 0.
        return dict(y=y)

    def vjp(node, _y, x):
        _x = _y.copy()
        mask = x <= 0
        _x[mask] = 0
        return dict(_x=_x)


@operator
class masking:
    """Mask field for train/validation/test split"""
    aout = {'y': 'RealField'}
    ain = {'x': 'RealField'}

    def apl(node, x, mask):
        y = x.copy()[mask]
        return dict(y=y)

    def vjp(node, _y, x, mask):
        _x = np.zeros_like(x)
        _x[mask] = _y
        return dict(_x=_x)


@operator
class complex_mul(binary):
    """Complex multiplication with proper gradient"""
    def apl(node, x1, x2):
        return dict(y=x1 * x2)

    def vjp(node, _y, x1, x2):
        return dict(_x1=_y * np.conj(x2),
                    _x2=_y * np.conj(x1))


@operator
class compensate2factor:
    """Correct gradients in Fourier space for self-conjugate modes"""
    aout = {'y': 'ComplexField'}
    ain = {'x': 'ComplexField'}

    def apl(node, x):
        y = x.copy()
        return dict(y=y)

    def vjp(node, _y):
        _x = _y.pm.create(type='complex')
        for i, a, b in zip(_x.slabs.i, _x.slabs, _y.slabs):
            mask = np.ones(a.shape, '?')
            mask &= ((_x.Nmesh[-1] - i[-1]) % _x.Nmesh[-1] == i[-1])
            a[~mask] = b[~mask] + np.conj(b[~mask])
            a[mask] = b[mask]
        return dict(_x=_x)


# ==================== EVOLVABLE MODEL ARCHITECTURE ====================
# The following block contains the core model architecture that will be evolved

# EVOLVE-BLOCK-START

@autooperator('param->X1')
def Displacement(param, X, pm, Nstep):
    """
    Lagrangian displacement operator.
    Moves particles according to learned displacement fields.

    OPTIMIZED: Uses the simplified filter design (removed 'n' tilt parameter)
    from the high-performance inspiration model. This ensures numerical stability
    and guarantees parameter slot availability for the baryon model.
    """
    EPS = 1e-8
    # Normalization factor for density
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    # Pre-compute k-norm for filter construction
    K = Literal(pm.create(type='complex', value=1).apply(
        lambda k, v: k.normp(2, zeromode=EPS) ** 0.5
    ))

    for i in range(Nstep):
        # 1. Paint current particles to mesh
        layout = fastpm.decompose(X, pm)
        xl = fastpm.exchange(X, layout)
        delta = fac * fastpm.paint(xl, 1.0, None, pm)

        # 2. Extract Parameters (skipping index 4, 9... for hijacking)
        alpha = linalg.take(param, 5*i+0, axis=0)
        gamma = linalg.take(param, 5*i+1, axis=0)
        kh    = linalg.take(param, 5*i+2, axis=0)
        kl    = linalg.take(param, 5*i+3, axis=0)
        # param 5*i+4 is intentionally unused here

        # 3. Nonlinear Source Term
        # Log-stabilized density transformation: log((1+d)^gamma + 1)
        gamma = mpi.allbcast(gamma, comm=pm.comm)
        gamma = linalg.broadcast_to(gamma, vmad_eval(delta, lambda x: x.shape))
        source = unary.log((delta + EPS) ** gamma + 1.0)
        source_k = fastpm.r2c(source)

        # 4. Construct Filter (Compensated Bandpass - No Tilt)
        kh = mpi.allbcast(kh, comm=pm.comm)
        kh = linalg.broadcast_to(kh, vmad_eval(K, lambda x: x.shape))
        kl = mpi.allbcast(kl, comm=pm.comm)
        kl = linalg.broadcast_to(kl, vmad_eval(K, lambda x: x.shape))

        # Robust filter form: -(1 - exp(-k^2/kh^2)) * exp(-k^2/kl^2)
        Filter = - unary.exp(-K**2 / (kl**2 + EPS)) * (1.0 - unary.exp(-K**2 / (kh**2 + EPS)))
        Filter = compensate2factor(Filter)

        # 5. Apply Transfer and Compute Displacement
        pot_k = complex_mul(source_k, Filter)

        r1 = []
        for d in range(pm.ndim):
            # Displacement is gradient of potential
            disp_k = fastpm.apply_transfer(pot_k, fastpm.fourier_space_neg_gradient(d, pm, order=1))
            disp_r = fastpm.c2r(disp_k)
            disp_l = fastpm.readout(disp_r, xl, None)
            disp_p = fastpm.gather(disp_l, layout)
            r1.append(disp_p)

        S = linalg.stack(r1, axis=-1)

        # 6. Update Positions
        alpha = mpi.allbcast(alpha, comm=pm.comm)
        alpha = linalg.broadcast_to(alpha, vmad_eval(S, lambda x: x.shape))
        X = X + S * alpha

    return X


@autooperator('param->F')
def LDL(param, X, pm, Nstep, baryon=True):
    """
    Hybrid Thermo-Virial LDL Model.

    Combines the robust 'Screened Potential' virial temperature from the
    Current model with the 'Compression Gated' shock model from the Inspiration
    model.

    Model: P_e = b_calib * n_e * T_eff
    n_e = (1 + delta)^mu
    T_eff = T_virial * T_shock_gated
    """
    EPS = 1e-8
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    # 1. Evolve Dark Matter Structure
    X = Displacement(param, X, pm, Nstep)

    # 2. Compute Density Field
    layout = fastpm.decompose(X, pm)
    Xl = fastpm.exchange(X, layout)
    delta = fac * fastpm.paint(Xl, 1., None, pm)

    if baryon:
        # --- Parameter Extraction ---
        # Standard baryon parameters
        mu      = linalg.take(param, 5*Nstep+0, axis=0)
        b_calib = linalg.take(param, 5*Nstep+1, axis=0)
        b_shock = linalg.take(param, 5*Nstep+2, axis=0)

        # Hijacked parameters from unused 'n' slots in Displacement
        # Slot 1 (Step 0): Virial Screening (k_s)
        p_ks = linalg.take(param, 4, axis=0)
        # Slot 2 (Last Step): Shear smoothing scale (R_s_shear)
        last_n_idx = 5*(Nstep-1)+4
        p_rs_shear = linalg.take(param, last_n_idx, axis=0)
        # Slot 3 (Step 1 or fallback to last): Shock smoothing (R_s_shock) & shear weight (w_shear)
        combo_idx = 9 if Nstep > 1 else last_n_idx
        p_combo_shock = linalg.take(param, combo_idx, axis=0)

        # Broadcast
        bcast_params = [mu, b_calib, b_shock, p_ks, p_rs_shear, p_combo_shock]
        mu, b_calib, b_shock, p_ks, p_rs_shear, p_combo_shock = [
            mpi.allbcast(p, comm=pm.comm) for p in bcast_params
        ]

        # --- Physical mappings ---
        k_s = 0.5 + 4.5 / (1.0 + unary.exp(-p_ks)) # Virial screening scale
        R_s_shear = (0.02 + 0.10 / (1.0 + unary.exp(-p_rs_shear))) * pm.BoxSize[0] # Shear smoothing
        R_s_shock = (0.02 + 0.10 / (1.0 + unary.exp(-p_combo_shock))) * pm.BoxSize[0] # Shock smoothing
        w_shear = 1.0 / (1.0 + unary.exp(p_combo_shock)) # Shear weight (anti-correlated with R_s_shock)

        # Broadcast to field shapes
        mu_f      = linalg.broadcast_to(mu, vmad_eval(delta, lambda x: x.shape))
        b_calib_f = linalg.broadcast_to(b_calib, vmad_eval(delta, lambda x: x.shape))
        b_shock_f = linalg.broadcast_to(b_shock, vmad_eval(delta, lambda x: x.shape))
        w_shear_f = linalg.broadcast_to(w_shear, vmad_eval(delta, lambda x: x.shape))

        # --- Fourier Setup ---
        delta_k = fastpm.r2c(delta)
        K = Literal(pm.create(type='complex', value=1).apply(lambda k, v: k.normp(2, zeromode=EPS)))
        k_s_f = linalg.broadcast_to(k_s, vmad_eval(K, lambda x: x.shape))

        # --- 1. Electron Density Proxy ---
        n_e = (delta + 1.0 + EPS) ** mu_f

        # --- 2. Virial Temperature (Screened Potential) ---
        InvScreen = 1.0 / (K**2 + k_s_f**2 + EPS)
        phi_k = complex_mul(delta_k, InvScreen)
        T_virial_raw = fastpm.c2r(phi_k)
        T_virial = unary.log(1.0 + unary.exp(T_virial_raw)) # Softplus for positivity

        # --- 3. Gated Shock Temperature (Multi-Scale) ---
        # Create two independent smoothing kernels
        R_s_shear_f = linalg.broadcast_to(R_s_shear, vmad_eval(K, lambda x: x.shape))
        R_s_shock_f = linalg.broadcast_to(R_s_shock, vmad_eval(K, lambda x: x.shape))
        Smooth_shear = unary.exp(-(K * R_s_shear_f)**2)
        Smooth_shock = unary.exp(-(K * R_s_shock_f)**2)

        # Create two independently smoothed potential fields
        phi_sm_shear_k = complex_mul(phi_k, Smooth_shear)
        phi_sm_shock_k = complex_mul(phi_k, Smooth_shock)

        # A. Isotropic Gradient Term (from shock-smoothed potential)
        grad_sq = delta * 0.0
        for d in range(pm.ndim):
            def grad_kernel(k,v,d=d): return 1j*k[d]
            Gk = Literal(pm.create(type='complex').apply(grad_kernel))
            g_k = complex_mul(phi_sm_shock_k, Gk)
            g_r = fastpm.c2r(g_k)
            grad_sq = grad_sq + g_r**2

        # B. Anisotropic Tidal Shear Term (from shear-smoothed potential)
        def get_hessian_comp(d1, d2):
            def h_kernel(k, v, d1=d1, d2=d2): return -k[d1] * k[d2]
            Hk = Literal(pm.create(type='complex').apply(h_kernel))
            return fastpm.c2r(complex_mul(phi_sm_shear_k, Hk))

        Hxx, Hyy, Hzz = get_hessian_comp(0,0), get_hessian_comp(1,1), get_hessian_comp(2,2)
        Hxy, Hxz, Hyz = get_hessian_comp(0,1), get_hessian_comp(0,2), get_hessian_comp(1,2)
        Havg = (Hxx + Hyy + Hzz) / 3.0
        s2 = (Hxx-Havg)**2 + (Hyy-Havg)**2 + (Hzz-Havg)**2 + 2.0*(Hxy**2+Hxz**2+Hyz**2)

        # C. Compression Gate (from Inspiration)
        # Emphasize shock heating in high-density (collapsing) regions
        # trH = laplacian(Phi) ~ delta.
        # We want a weight that scales with density/collapse.
        gate = (1.0 + ReLU(delta)) ** 0.5

        # Combined Kinetic Term
        # T_kin ~ (1 + Gate * ((1-w)Grad^2 + w*s^2))
        kin_mix = (1.0 - w_shear_f) * grad_sq + w_shear_f * s2
        T_shock = (1.0 + gate * kin_mix + EPS) ** b_shock_f

        # --- Final Combination ---
        F = b_calib_f * n_e * T_virial * T_shock
        F = ReLU(F)

    else:
        F = delta

    return F


def smoothing(n):
    """
    Compensated Smoothing Kernel for Residue.
    Balances weighting between large scales (k^-n) and small scales (1).
    The denominator term (1 + 0.1*k^n) prevents excessive domination of large scales.
    """
    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)**0.5
        mask = kk == 0
        kk[mask] = 1.0
        # Compensated weight: (k^-n + 1) / (1 + alpha * k^n)
        b = v * ((kk**(-n) + 1.0) / (1.0 + 0.1 * (kk**n)))
        b[mask] = v[mask]
        return b
    return kernel


@autooperator('param->residue')
def smoothed_residue(param, X, pm, Nstep, target, n, baryon=True):
    F = LDL(param, X, pm, Nstep, baryon=baryon)
    residue = F - target
    Filter = pm.create(type='complex', value=1).apply(smoothing(n=n))
    residuek = fastpm.r2c(residue) * Filter
    residue = fastpm.c2r(residuek)
    return residue


@autooperator('residue->loss')
def lossfunc(residue, mask, comm=MPI.COMM_WORLD, L1=True):
    residue = unary.absolute(residue)
    loss = masking(residue, mask)
    Npixel = np.sum(mask)
    loss = linalg.sum(loss) if L1 else linalg.sum(loss**2)
    loss = mpi.allreduce(loss, comm=comm)
    Npixel = mpi.allreduce(Npixel, comm=comm)
    return loss / Npixel

# EVOLVE-BLOCK-END


# ==================== Training Infrastructure ====================
# These are helper functions that support training

def loss_and_gradient(param, residue_model, loss_train_model, loss_validate_model=None):
    """Calculate loss and gradient using vmad automatic differentiation"""
    # Forward pass: compute residue
    residue, residue_tape = residue_model.compute('residue', init=dict(param=param), return_tape=True)
    vjpvout = residue_tape.get_vjp_vout()
    vjp = residue_tape.get_vjp()

    # Training loss and gradient
    loss_train, _residue = loss_train_model.compute_with_vjp(
        init=dict(residue=residue), v=dict(_loss=1.0))
    loss_train = loss_train[0]
    _residue = _residue[0]

    # Backward pass: gradient w.r.t. parameters
    gradient = vjp.compute(vjpvout, init=OrderedDict(dict(_residue=_residue)))
    gradient = gradient[0]

    # Validation loss (no gradient needed)
    if loss_validate_model:
        loss_validate = loss_validate_model.compute('loss', init=dict(residue=residue))
    else:
        loss_validate = None

    gc.collect()
    return loss_train, gradient, loss_validate


class MemoizeJac:
    """Memoize function evaluations and track best parameters"""
    def __init__(self, fun, save=None, verbose=True, comm=MPI.COMM_WORLD):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None
        self.best_value = np.inf
        self.best_x = None
        self.save = save
        self.verbose = verbose
        self.comm = comm

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            t = time.time()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]

            # Track best validation loss
            if fg[2] is not None:
                if self.best_value > fg[2]:
                    self.best_value = fg[2]
                    self.best_x = np.asarray(x).copy()
                    if self.save:
                        np.savetxt(self.save, self.best_x)
            elif self.best_value > fg[0]:
                self.best_value = fg[0]
                self.best_x = np.asarray(x).copy()
                if self.save:
                    np.savetxt(self.save, self.best_x)

            if self.verbose and self.comm.rank == 0:
                if fg[2] is not None:
                    print(f'Train Loss: {self._value:.6f}, Val Loss: {fg[2]:.6f}, '
                          f'Best Val: {self.best_value:.6f}, Time: {time.time()-t:.2f}s')
                else:
                    print(f'Train Loss: {self._value:.6f}, Best: {self.best_value:.6f}, '
                          f'Time: {time.time()-t:.2f}s')

    def __call__(self, x, *args):
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac


def create_spatial_masks(pm, Nmesh, region='train'):
    """
    Create spatial masks for train/validation/test split.

    Strategy: Use block-based spatial division for better memory locality and performance.
    This approach provides better CPU cache utilization compared to interleaved patterns.

    Split ratios (similar to LDL paper):
    - Train: ~70% of volume (main cube region)
    - Validation: ~15% of volume (back strip)
    - Test: ~15% of volume (remaining regions)

    Args:
        pm: ParticleMesh object
        Nmesh: Mesh resolution
        region: 'train', 'validate', or 'test'

    Returns:
        mask: Boolean array indicating which cells belong to this region

    Performance Note:
        Block-based splitting provides ~2x speedup compared to interleaved patterns
        due to improved memory access patterns and CPU cache efficiency.
    """
    index = pm.mesh_coordinates()
    i, j, k = index[:, 0], index[:, 1], index[:, 2]

    # Define split boundaries for approximately 70-15-15 split
    # Similar to LDL repository approach but with slightly different ratios
    train_boundary = int(0.85 * Nmesh)  # Train gets most of the volume
    validate_boundary = int(0.70 * Nmesh)  # Validation is a smaller region

    if region == 'train':
        # Train: Main bulk of the simulation (~70%)
        # Includes: front region + back-top region
        mask = (
            # Front large cube (main training region)
            ((i < train_boundary) & (j < train_boundary) & (k < train_boundary)) |
            # Additional strips to reach ~70%
            ((i >= train_boundary) & (j >= validate_boundary))
        )
    elif region == 'validate':
        # Validate: Back strip (~15%)
        mask = (
            (i >= train_boundary) & (j < validate_boundary) & (k < train_boundary)
        )
    elif region == 'test':
        # Test: Remaining top region (~15%)
        mask = (
            (k >= train_boundary) &
            ~((i >= train_boundary) & (j >= validate_boundary))
        )
    else:
        raise ValueError(f"Unknown region: {region}")

    return mask.reshape(pm.create(type='real').shape)


def run_inference(sim_id, param, run_idx=0, Nstep=2, n=1.0, verbose=False):
    """
    Run LDL inference (no training) on CAMELS data with given parameters.

    This function performs forward pass only to evaluate a trained model
    on a new dataset without optimization.

    Args:
        sim_id: Simulation ID ('CV_0', 'CV_1', 'CV_2')
        param: Model parameters (from previous training)
        run_idx: Run index for reproducibility
        Nstep: Number of displacement layers
        n: Smoothing kernel exponent
        verbose: Whether to print progress

    Returns:
        dict with inference results including loss and predicted fields
    """
    comm = MPI.COMM_WORLD

    # Configuration
    TARGET = 'nT'  # tSZ signal (electron pressure = ne * T)
    NMESH = 64
    DOWNSAMPLE = 16
    SNAPNUM = 90

    # Paths
    TNG_PATH = os.path.join(LDL_PATH, 'data', 'IllustrisTNG', 'CV', sim_id)
    TNG_DARK_PATH = os.path.join(LDL_PATH, 'data', 'IllustrisTNG_DM', 'CV', sim_id)

    if comm.rank == 0 and verbose:
        print(f"\n{'='*60}")
        print(f"Running LDL Inference - Sim: {sim_id}, Run: {run_idx}")
        print(f"Target: {TARGET}, Nmesh: {NMESH}, Nstep: {Nstep}, n: {n}")
        print(f"{'='*60}\n")

    try:
        # Read cosmology parameters
        cosmology = get_cosmology(TNG_DARK_PATH, SNAPNUM)
        BoxSize = cosmology['BoxSize']
        h = cosmology['h']

        # Initialize ParticleMesh
        pm = ParticleMesh(Nmesh=[NMESH]*3, BoxSize=BoxSize, resampler='cic')

        # Load dark matter particles
        X = []
        for mdi in range(3):
            X.append(load_TNG_data(
                TNG_basepath=TNG_DARK_PATH,
                snapNum=SNAPNUM,
                partType='dm',
                field='Coordinates',
                mdi=mdi,
                BoxSize=BoxSize,
                downsample=DOWNSAMPLE
            ))
        X = np.array(X).T

        # Load target map
        targetmap = load_TNG_map(
            TNG_basepath=TNG_PATH,
            snapNum=SNAPNUM,
            field=TARGET,
            pm=pm,
            cosmology=cosmology
        )

        # Create mask for full box
        mask_full = np.ones(pm.create(type='real').shape, dtype=bool)

        # Build models for inference (no training)
        residue_model = smoothed_residue.build(
            X=X, pm=pm, Nstep=Nstep, target=targetmap, n=n, baryon=True
        )
        loss_model = lossfunc.build(mask=mask_full, comm=comm, L1=True)
        ldl_model = LDL.build(X=X, pm=pm, Nstep=Nstep, baryon=True)

        # Forward pass only (no optimization)
        start_time = time.time()

        # Compute residue and loss
        residue = residue_model.compute('residue', init=dict(param=param))
        final_loss = loss_model.compute('loss', init=dict(residue=residue))

        # Generate predicted field for power spectrum analysis
        predicted_field = ldl_model.compute('F', init=dict(param=param))

        inference_time = time.time() - start_time

        # Convert pmesh RealField to numpy array for serialization
        # Use preview() to gather the full field from all MPI ranks to rank 0
        # np.array() only gets the local slice, which is incorrect for power spectrum
        predicted_field_data = predicted_field.preview()
        target_field_data = targetmap.preview()

        if comm.rank == 0:
            result = {
                'loss': float(final_loss),
                'param': param.tolist() if isinstance(param, np.ndarray) else param,
                'success': True,
                'inference_time': inference_time,
                'sim_id': sim_id,
                'run_idx': run_idx,
                'Nstep': Nstep,
                'n': n,
                'error': None,

                # Add field data for power spectrum computation
                'predicted_field_data': predicted_field_data,
                'target_field_data': target_field_data,
                'BoxSize': BoxSize,
                'Nmesh': NMESH,
            }
        else:
            result = None

        # Broadcast result to all ranks
        result = comm.bcast(result, root=0)

        return result

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        if comm.rank == 0:
            result = {
                'loss': float('inf'),
                'error': error_msg,
                'success': False,
                'sim_id': sim_id,
                'run_idx': run_idx
            }
        else:
            result = None

        # Broadcast error to all ranks
        result = comm.bcast(result, root=0)
        return result


def run_experiment(sim_id='CV_0', run_idx=0, Nstep=2, n=1.0, max_iter=500, verbose=False):
    """
    Run LDL training experiment on CAMELS data.

    CORRECT USAGE IN EVALUATION PIPELINE:
    - This function should be called on the TRAINING set (CV_0) during evolution
    - The returned parameters should then be used with run_inference() on multiple validation sets
    - Final testing should use run_inference() on test sets (CV_5 ~ CV_10)

    Data split strategy (multi-validation approach):
    - CV_0: Training simulation (use this function to optimize parameters)
    - CV_1, CV_2, CV_3, CV_4: Validation simulations (use run_inference() with trained parameters)
    - CV_5 ~ CV_10: Test simulations (use run_inference() for final evaluation)

    This multi-validation approach:
    - Prevents overfitting to a single validation set's characteristics
    - Selects architectures with better generalization across different cosmic variance
    - Provides more robust fitness estimates

    Args:
        sim_id: Simulation ID ('CV_0', 'CV_1', ..., 'CV_10')
                - Should be 'CV_0' for training during evolution
        run_idx: Run index for reproducibility
        Nstep: Number of displacement layers
        n: Smoothing kernel exponent
        max_iter: Maximum optimization iterations
        verbose: Whether to print training progress

    Returns:
        dict with training results including losses and parameters
    """
    comm = MPI.COMM_WORLD

    # Configuration
    TARGET = 'nT'  # tSZ signal (electron pressure = ne * T)
    NMESH = 64
    DOWNSAMPLE = 16
    SNAPNUM = 90

    # Paths - now using sim_id parameter
    TNG_PATH = os.path.join(LDL_PATH, 'data', 'IllustrisTNG', 'CV', sim_id)
    TNG_DARK_PATH = os.path.join(LDL_PATH, 'data', 'IllustrisTNG_DM', 'CV', sim_id)

    if comm.rank == 0 and verbose:
        print(f"\n{'='*60}")
        print(f"Running LDL CAMELS - Sim: {sim_id}, Run: {run_idx}")
        print(f"Target: {TARGET}, Nmesh: {NMESH}, Nstep: {Nstep}, n: {n}")
        print(f"{'='*60}\n")

    try:
        # Read cosmology parameters
        cosmology = get_cosmology(TNG_DARK_PATH, SNAPNUM)
        BoxSize = cosmology['BoxSize']
        h = cosmology['h']

        # Initialize ParticleMesh
        pm = ParticleMesh(Nmesh=[NMESH]*3, BoxSize=BoxSize, resampler='cic')

        # Load dark matter particles
        X = []
        for mdi in range(3):
            X.append(load_TNG_data(
                TNG_basepath=TNG_DARK_PATH,
                snapNum=SNAPNUM,
                partType='dm',
                field='Coordinates',
                mdi=mdi,
                BoxSize=BoxSize,
                downsample=DOWNSAMPLE
            ))
        X = np.array(X).T

        # Load target map (stellar mass)
        targetmap = load_TNG_map(
            TNG_basepath=TNG_PATH,
            snapNum=SNAPNUM,
            field=TARGET,
            pm=pm,
            cosmology=cosmology
        )

        # Create mask for full box (no spatial masking with cross-simulation split)
        # All cells are used for training/validation
        mask_full = np.ones(pm.create(type='real').shape, dtype=bool)

        # Build models
        residue_model = smoothed_residue.build(
            X=X, pm=pm, Nstep=Nstep, target=targetmap, n=n, baryon=True
        )
        # Use full box for loss computation
        loss_model = lossfunc.build(mask=mask_full, comm=comm, L1=True)

        # Initialize parameters
        x0 = [0.001, 0.5, 1., 8., 0.] * Nstep
        x0 += [1., targetmap.csum() / comm.allreduce(len(X), op=MPI.SUM), 0.]
        x0 = np.array(x0)

        # Parameter bounds
        bounds = [(None, None), (0.05, 2), (0.03, 2*np.pi*NMESH/205.),
                  (0.03, 2*np.pi*NMESH/205.), (-4.5, 4.5)] * Nstep
        bounds += [(0.1, None), (0., None), (None, None)]

        # Setup memoized loss function
        loss = MemoizeJac(loss_and_gradient, save=None, verbose=verbose, comm=comm)

        # Optimize
        start_time = time.time()
        res = minimize(
            loss,
            x0=x0,
            method='L-BFGS-B',
            args=(residue_model, loss_model, None),  # No separate validation during training
            jac=loss.derivative,
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        training_time = time.time() - start_time

        # Get best parameters
        best_param = loss.best_x if loss.best_x is not None else res.x

        # Compute final loss on full box
        residue_model_eval = smoothed_residue.build(
            X=X, pm=pm, Nstep=Nstep, target=targetmap, n=n, baryon=True
        )
        loss_eval = lossfunc.build(mask=mask_full, comm=comm, L1=True)
        residue = residue_model_eval.compute('residue', init=dict(param=best_param))
        final_loss = loss_eval.compute('loss', init=dict(residue=residue))

        # Generate predicted field for power spectrum analysis
        ldl_model = LDL.build(X=X, pm=pm, Nstep=Nstep, baryon=True)
        predicted_field = ldl_model.compute('F', init=dict(param=best_param))

        # Convert pmesh RealField to numpy array for serialization
        # Use preview() to gather the full field from all MPI ranks to rank 0
        # np.array() only gets the local slice, which is incorrect for power spectrum
        predicted_field_data = predicted_field.preview()
        target_field_data = targetmap.preview()

        if comm.rank == 0:
            result = {
                'loss': float(final_loss),
                'best_param': best_param.tolist(),
                'success': res.success,
                'nit': res.nit,
                'training_time': training_time,
                'sim_id': sim_id,
                'run_idx': run_idx,
                'Nstep': Nstep,
                'n': n,
                'error': None,

                # Add field data for power spectrum computation
                # Store as numpy arrays to enable pickling
                'predicted_field_data': predicted_field_data,
                'target_field_data': target_field_data,
                'BoxSize': BoxSize,
                'Nmesh': NMESH,
            }
        else:
            result = None

        # Broadcast result to all ranks
        result = comm.bcast(result, root=0)

        return result

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        if comm.rank == 0:
            result = {
                'loss': float('inf'),
                'error': error_msg,
                'success': False,
                'sim_id': sim_id,
                'run_idx': run_idx
            }
        else:
            result = None

        # Broadcast error to all ranks
        result = comm.bcast(result, root=0)
        return result


if __name__ == '__main__':
    # Test the model locally
    # Default to CV_0 for training
    result = run_experiment(sim_id='CV_0', run_idx=0, Nstep=2, n=1.0, max_iter=100, verbose=True)
    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "="*60)
        print("Training Result:")
        print("="*60)
        for key, value in result.items():
            if key != 'best_param':  # Skip printing full parameter array
                print(f"{key}: {value}")
        print("="*60)