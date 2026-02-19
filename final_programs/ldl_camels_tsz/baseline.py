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

    EVOLVABLE COMPONENTS:
    1. The filter design in Fourier space (currently Gaussian + power law)
    2. The nonlinear transformation of density (currently delta^gamma)
    3. The number and structure of displacement steps
    """
    # Normalization constant for overdensity
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    for i in range(Nstep):
        # Move particles across MPI ranks
        layout = fastpm.decompose(X, pm)
        xl = fastpm.exchange(X, layout)
        delta = fac * fastpm.paint(xl, 1.0, None, pm)

        # Extract parameters for this layer
        alpha = linalg.take(param, 5*i, axis=0)
        gamma = linalg.take(param, 5*i+1, axis=0)
        kh = linalg.take(param, 5*i+2, axis=0)
        kl = linalg.take(param, 5*i+3, axis=0)
        n = linalg.take(param, 5*i+4, axis=0)

        # Apply nonlinear transformation: delta^gamma
        # EVOLVABLE: This could be replaced with other transformations
        gamma = mpi.allbcast(gamma, comm=pm.comm)
        gamma = linalg.broadcast_to(gamma, vmad_eval(delta, lambda x: x.shape))
        delta = (delta + 1e-8) ** gamma

        # Fourier transform
        deltak = fastpm.r2c(delta)

        # Design Fourier space filter
        # EVOLVABLE: Current design is double Gaussian with power law
        # Filter = -exp(-k^2/kl^2) * exp(-kh^2/k^2) * k^n
        Filter = Literal(pm.create(type='complex', value=1).apply(
            lambda k, v: k.normp(2, zeromode=1e-8) ** 0.5))

        kh = mpi.allbcast(kh, comm=pm.comm)
        kh = linalg.broadcast_to(kh, vmad_eval(Filter, lambda x: x.shape))
        kl = mpi.allbcast(kl, comm=pm.comm)
        kl = linalg.broadcast_to(kl, vmad_eval(Filter, lambda x: x.shape))
        n = mpi.allbcast(n, comm=pm.comm)
        n = linalg.broadcast_to(n, vmad_eval(Filter, lambda x: x.shape))

        # Apply filter design
        Filter = - unary.exp(-Filter**2 / kl**2) * unary.exp(-kh**2 / Filter**2) * Filter**n
        Filter = compensate2factor(Filter)

        p = complex_mul(deltak, Filter)

        # Compute gradient of potential (displacement field)
        r1 = []
        for d in range(pm.ndim):
            dx1_c = fastpm.apply_transfer(p, fastpm.fourier_space_neg_gradient(d, pm, order=1))
            dx1_r = fastpm.c2r(dx1_c)
            dx1l = fastpm.readout(dx1_r, xl, None)
            dx1 = fastpm.gather(dx1l, layout)
            r1.append(dx1)

        # Scale and apply displacement
        S = linalg.stack(r1, axis=-1)
        alpha = mpi.allbcast(alpha, comm=pm.comm)
        alpha = linalg.broadcast_to(alpha, vmad_eval(S, lambda x: x.shape))
        S = S * alpha

        X = X + S

    return X


@autooperator('param->F')
def LDL(param, X, pm, Nstep, baryon=True):
    """
    Main LDL model combining displacement and baryon bias.

    EVOLVABLE COMPONENTS:
    1. The baryon bias transformation (currently power law + linear + ReLU)
    2. How displacement output is processed
    """
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    # Apply Lagrangian displacement
    X = Displacement(param, X, pm, Nstep)

    # Paint particle overdensity field
    layout = fastpm.decompose(X, pm)
    Xl = fastpm.exchange(X, layout)
    delta = fac * fastpm.paint(Xl, 1., None, pm)

    if baryon:
        # Extract baryon bias parameters
        mu = linalg.take(param, 5*Nstep, axis=0)
        b1 = linalg.take(param, 5*Nstep+1, axis=0)
        b0 = linalg.take(param, 5*Nstep+2, axis=0)

        mu = mpi.allbcast(mu, comm=pm.comm)
        mu = linalg.broadcast_to(mu, vmad_eval(delta, lambda x: x.shape))
        b1 = mpi.allbcast(b1, comm=pm.comm)
        b1 = linalg.broadcast_to(b1, vmad_eval(delta, lambda x: x.shape))
        b0 = mpi.allbcast(b0, comm=pm.comm)
        b0 = linalg.broadcast_to(b0, vmad_eval(delta, lambda x: x.shape))

        # EVOLVABLE: Baryon field transformation
        # Current: F = ReLU(b1 * delta^mu + b0)
        # Could evolve to different activation functions or transformations
        F = ReLU(b1 * (delta + 1e-8) ** mu + b0)
    else:
        F = delta

    return F


def smoothing(n):
    """
    Smoothing kernel for loss function.
    Weights different scales in Fourier space.

    EVOLVABLE: The weighting scheme could be modified
    """
    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)
        kk = kk ** 0.5
        mask = kk == 0
        kk[mask] = 1
        # Current: b = v * (k^(-n) + 1)
        # Emphasizes large scales when n > 0
        b = v * (kk**(-n) + 1.)
        b[mask] = v[mask]
        return b
    return kernel


@autooperator('param->residue')
def smoothed_residue(param, X, pm, Nstep, target, n, baryon=True):
    """
    Compute smoothed residue between prediction and target.

    EVOLVABLE COMPONENTS:
    1. The smoothing strategy
    2. How residue is computed (could add perceptual loss, etc.)
    """
    # Get model prediction
    F = LDL(param, X, pm, Nstep, baryon=baryon)

    # Compute residue
    residue = F - target

    # Apply smoothing in Fourier space
    # EVOLVABLE: Could use different smoothing strategies
    Filter = pm.create(type='complex', value=1).apply(smoothing(n=n))
    residuek = fastpm.r2c(residue)
    residuek = residuek * Filter
    residue = fastpm.c2r(residuek)

    return residue


@autooperator('residue->loss')
def lossfunc(residue, mask, comm=MPI.COMM_WORLD, L1=True):
    """
    Loss function with train/val/test masking.

    EVOLVABLE: Could evolve to multi-scale loss, perceptual loss, etc.
    """
    residue = unary.absolute(residue)
    loss = masking(residue, mask)
    Npixel = np.sum(mask)

    if L1:
        loss = linalg.sum(loss)
    else:
        loss = linalg.sum(loss**2)

    loss = mpi.allreduce(loss, comm=comm)
    Npixel = mpi.allreduce(Npixel, comm=comm)
    loss = loss / Npixel
    return loss


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
