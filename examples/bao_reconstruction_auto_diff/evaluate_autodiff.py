"""
BAO Reconstruction Evaluator with Autodiff Parameter Optimization

This module evaluates BAO reconstruction algorithms by measuring the
cross-correlation r(k) between reconstructed and ground truth density fields.

Key Feature: Uses JAX autodiff for parameter optimization instead of grid search.
The evolved programs must be fully differentiable (using JAX operations) so that
gradients can flow through the entire reconstruction pipeline.
"""

import json
import os
import sys
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import importlib.util


@dataclass
class TunableParam:
    """Represents a single tunable parameter."""
    name: str
    default_value: float
    bounds: Tuple[float, float]
    method: str = "grid"
    dtype: str = "float"
    frozen_value: Optional[float] = None


class ParamParser:
    """Parses tunable parameter declarations from source code."""
    TUNABLE_PATTERN = re.compile(
        r'#\s*TUNABLE:\s*(\w+)\s*=\s*([^,]+),\s*bounds\s*=\s*\(([^)]+)\)'
        r'(?:,\s*method\s*=\s*(\w+))?'
        r'(?:,\s*dtype\s*=\s*(\w+))?',
        re.IGNORECASE
    )

    @classmethod
    def parse_tunable_declarations(cls, code: str) -> List[TunableParam]:
        params = []
        for match in cls.TUNABLE_PATTERN.finditer(code):
            name = match.group(1)
            default_str = match.group(2).strip()
            bounds_str = match.group(3)
            method = match.group(4) or "grid"
            dtype = match.group(5) or "float"
            try:
                default_value = float(default_str)
            except ValueError:
                continue
            try:
                bounds_parts = [x.strip() for x in bounds_str.split(",")]
                bounds = (float(bounds_parts[0]), float(bounds_parts[1]))
            except (ValueError, IndexError):
                continue
            params.append(TunableParam(
                name=name, default_value=default_value,
                bounds=bounds, method=method.lower(), dtype=dtype.lower(),
            ))
        return params


def log_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def log_section(title: str, char: str = "-", width: int = 80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def log_kv(key: str, value: Any, indent: int = 2):
    """Print a key-value pair with indentation."""
    print(f"{' ' * indent}{key}: {value}")


def timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class AutodiffParamConfig:
    """Configuration for autodiff-based parameter optimization."""
    learning_rate: float = 0.1
    max_iterations: int = 50
    convergence_threshold: float = 1e-5
    verbose: bool = True
    use_adam: bool = True
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    log_every: int = 1  # Log every N iterations


def compute_avg_r_bao_differentiable(recon_field: jnp.ndarray, gt_field: jnp.ndarray,
                                      kmin: float = 0.01, kmax: float = 0.5,
                                      box_size: float = 1000.0, n_bins: int = 20) -> jnp.ndarray:
    """
    Compute average r(k) in BAO range in a fully differentiable way.
    This is the objective function for autodiff optimization.

    Key improvement: First bin r(k) to 1D, then average over bins.
    This gives equal weight to each k bin, preventing large-k modes from
    dominating the metric (since shell volume ~ k^2).

    This implementation matches compute_rk_metrics_numpy and compare_algorithms.py exactly:
    - Uses fftshift for both FFT output and k-grid
    - k range [0.01, 0.5], 20 bins
    - Uses sum for power spectra in each bin

    Args:
        recon_field: Reconstructed density field (3D)
        gt_field: Ground truth density field (3D)
        kmin: Minimum k for BAO range (h/Mpc), default 0.01
        kmax: Maximum k for BAO range (h/Mpc), default 0.3
        box_size: Box size in Mpc/h
        n_bins: Number of k bins for 1D binning

    Returns a scalar JAX array for gradient computation.
    """
    # 3-D FFT with fftshift (matches numpy versions)
    delta_r = jnp.fft.fftshift(jnp.fft.fftn(recon_field))
    delta_g = jnp.fft.fftshift(jnp.fft.fftn(gt_field))

    Nz, Ny, Nx = recon_field.shape

    # Calculate k in physical units (h/Mpc) with fftshift
    kx = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nx)) * Nx / box_size
    ky = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Ny)) * Ny / box_size
    kz = 2*jnp.pi * jnp.fft.fftshift(jnp.fft.fftfreq(Nz)) * Nz / box_size

    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = jnp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # Compute power spectra (element-wise)
    P_rg = (delta_r * jnp.conj(delta_g)).real
    P_rr = (delta_r * jnp.conj(delta_r)).real
    P_gg = (delta_g * jnp.conj(delta_g)).real

    # Create k bin edges within BAO range
    k_edges = jnp.linspace(kmin, kmax, n_bins + 1)

    # Compute r(k) for each bin, then average
    # This gives equal weight to each k bin (not each mode)
    r_k_bins = []
    for i in range(n_bins):
        k_lo = k_edges[i]
        k_hi = k_edges[i + 1]
        bin_mask = (k_mag >= k_lo) & (k_mag < k_hi)

        # Sum power spectra in this bin
        P_rg_sum = jnp.sum(jnp.where(bin_mask, P_rg, 0.0))
        P_rr_sum = jnp.sum(jnp.where(bin_mask, P_rr, 0.0))
        P_gg_sum = jnp.sum(jnp.where(bin_mask, P_gg, 0.0))

        # Compute r(k) for this bin
        denom = jnp.sqrt(P_rr_sum * P_gg_sum)
        r_k_bin = jnp.where(denom > 1e-10, P_rg_sum / denom, 0.0)
        r_k_bins.append(r_k_bin)

    # Average r(k) across all bins (equal weight per bin)
    r_k_array = jnp.stack(r_k_bins)
    avg_r_bao = jnp.mean(r_k_array)

    return avg_r_bao


class AutodiffParamOptimizer:
    """
    Optimizes parameters using JAX autodiff (gradient descent).

    This optimizer requires the evolved program to be fully differentiable,
    meaning all operations must be JAX-compatible.
    """

    def __init__(self, config: Optional[AutodiffParamConfig] = None):
        self.config = config or AutodiffParamConfig()
        self.optimization_history = []

    def optimize(
        self,
        params_to_optimize: List[TunableParam],
        reconstruct_fn: Callable,
        input_data: jnp.ndarray,
        gt_field: jnp.ndarray,
        fixed_params: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], int, float]:
        """
        Optimize parameters using gradient descent with autodiff.

        Uses jax.value_and_grad for efficient computation (single forward+backward pass).
        """
        if not params_to_optimize:
            return {}, 0, float('-inf')

        fixed_params = fixed_params or {}
        self.optimization_history = []

        # Print optimization header
        log_header("AUTODIFF PARAMETER OPTIMIZATION")
        print(f"  Timestamp: {timestamp()}")
        print(f"  Number of parameters to optimize: {len(params_to_optimize)}")
        print()

        print("  Parameters:")
        for p in params_to_optimize:
            print(f"    - {p.name}:")
            print(f"        default = {p.default_value}")
            print(f"        bounds  = [{p.bounds[0]}, {p.bounds[1]}]")
            print(f"        method  = {p.method}")

        if fixed_params:
            print()
            print("  Fixed (inherited) parameters:")
            for name, value in fixed_params.items():
                print(f"    - {name} = {value}")

        print()
        print("  Optimizer settings:")
        print(f"    - Algorithm: {'Adam' if self.config.use_adam else 'SGD'}")
        print(f"    - Learning rate: {self.config.learning_rate}")
        print(f"    - Max iterations: {self.config.max_iterations}")
        print(f"    - Convergence threshold: {self.config.convergence_threshold}")
        if self.config.use_adam:
            print(f"    - Adam beta1: {self.config.adam_b1}")
            print(f"    - Adam beta2: {self.config.adam_b2}")

        # Initialize parameters
        param_names = [p.name for p in params_to_optimize]
        param_values = jnp.array([p.default_value for p in params_to_optimize])
        param_bounds = [(p.bounds[0], p.bounds[1]) for p in params_to_optimize]

        # Define the loss function (negative r_bao for minimization)
        def loss_fn(params_array):
            param_dict = dict(zip(param_names, params_array))
            param_dict.update(fixed_params)
            recon_field = reconstruct_fn(input_data, **param_dict)
            avg_r_bao = compute_avg_r_bao_differentiable(recon_field, gt_field)
            return -avg_r_bao  # Negative for minimization

        # Use value_and_grad for efficient computation
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        # Initialize Adam optimizer state
        if self.config.use_adam:
            m = jnp.zeros_like(param_values)
            v = jnp.zeros_like(param_values)
            t = 0

        best_params = param_values.copy()
        best_loss = float('inf')
        best_iteration = 0

        log_section("Optimization Progress")
        print()
        print("  Iter | " + " | ".join(f"{name:>12}" for name in param_names) +
              " | r_bao      | grad_norm  | status")
        print("  " + "-" * (7 + 15 * len(param_names) + 40))

        total_forward_passes = 0
        optimization_start_time = time.time()

        for iteration in range(self.config.max_iterations):
            iter_start_time = time.time()

            try:
                # Compute loss and gradient in one forward+backward pass
                current_loss, grads = value_and_grad_fn(param_values)
                total_forward_passes += 1  # value_and_grad counts as ~1.5-2 forward passes

                current_score = -float(current_loss)
                grad_norm = float(jnp.linalg.norm(grads))

                # Check for NaN
                if jnp.any(jnp.isnan(grads)) or np.isnan(current_score):
                    status = "NaN DETECTED"
                    self._log_iteration(iteration, param_names, param_values, current_score,
                                       grad_norm, status, is_best=False)
                    print(f"\n  [!] NaN detected at iteration {iteration + 1}, stopping optimization")
                    break

                # Check if this is the best
                is_best = current_loss < best_loss
                if is_best:
                    best_loss = current_loss
                    best_params = param_values.copy()
                    best_iteration = iteration + 1
                    status = "*** BEST ***"
                else:
                    status = ""

                # Log progress
                if iteration % self.config.log_every == 0 or is_best or iteration == 0:
                    self._log_iteration(iteration, param_names, param_values, current_score,
                                       grad_norm, status, is_best)

                # Record history
                self.optimization_history.append({
                    'iteration': iteration + 1,
                    'params': {n: float(v) for n, v in zip(param_names, param_values)},
                    'score': current_score,
                    'grad_norm': grad_norm,
                    'is_best': is_best,
                })

                # Check convergence
                if grad_norm < self.config.convergence_threshold:
                    print(f"\n  [✓] Converged at iteration {iteration + 1} (grad_norm={grad_norm:.2e} < {self.config.convergence_threshold})")
                    break

                # Update parameters with Adam or SGD
                if self.config.use_adam:
                    t += 1
                    m = self.config.adam_b1 * m + (1 - self.config.adam_b1) * grads
                    v = self.config.adam_b2 * v + (1 - self.config.adam_b2) * (grads ** 2)
                    m_hat = m / (1 - self.config.adam_b1 ** t)
                    v_hat = v / (1 - self.config.adam_b2 ** t)
                    param_values = param_values - self.config.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.config.adam_eps)
                else:
                    param_values = param_values - self.config.learning_rate * grads

                # Clip to bounds
                for i, (low, high) in enumerate(param_bounds):
                    param_values = param_values.at[i].set(jnp.clip(param_values[i], low, high))

            except Exception as e:
                print(f"\n  [!] Error at iteration {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                break

        optimization_time = time.time() - optimization_start_time
        final_iteration = iteration + 1

        # Build result
        best_values = dict(zip(param_names, [float(v) for v in best_params]))
        best_score = -best_loss

        # Print summary
        log_section("Optimization Summary")
        print()
        print(f"  Completed iterations: {final_iteration}/{self.config.max_iterations}")
        print(f"  Total forward passes: ~{total_forward_passes} (value_and_grad)")
        print(f"  Total time: {optimization_time:.2f}s")
        print(f"  Time per iteration: {optimization_time/max(final_iteration, 1)*1000:.1f}ms")
        print()
        print(f"  Best iteration: {best_iteration}")
        print(f"  Best score (avg r_bao): {best_score:.6f}")
        print()
        print("  Optimized parameter values:")
        for name, value in best_values.items():
            # Find the original param to show bounds
            orig_param = next((p for p in params_to_optimize if p.name == name), None)
            if orig_param:
                print(f"    {name}: {value:.6g}  (was {orig_param.default_value}, bounds [{orig_param.bounds[0]}, {orig_param.bounds[1]}])")
            else:
                print(f"    {name}: {value:.6g}")

        return best_values, final_iteration, best_score

    def _log_iteration(self, iteration: int, param_names: List[str], param_values: jnp.ndarray,
                       score: float, grad_norm: float, status: str, is_best: bool):
        """Log a single iteration."""
        param_strs = [f"{float(v):>12.4g}" for v in param_values]
        marker = "→" if is_best else " "
        print(f"  {marker}{iteration+1:4d} | " + " | ".join(param_strs) +
              f" | {score:>10.6f} | {grad_norm:>10.4g} | {status}")


def validate_reconstruction(
    run_output: Tuple[np.ndarray, np.ndarray, float]
) -> Tuple[bool, str]:
    """Validates reconstruction results."""
    reconstructed_field, gt_field, _ = run_output

    # Convert JAX arrays to numpy if needed
    if hasattr(reconstructed_field, 'block_until_ready'):
        reconstructed_field = np.asarray(reconstructed_field)
    if hasattr(gt_field, 'block_until_ready'):
        gt_field = np.asarray(gt_field)

    if not isinstance(reconstructed_field, np.ndarray):
        return False, f"Reconstructed field must be ndarray, got {type(reconstructed_field)}"
    if not isinstance(gt_field, np.ndarray):
        return False, f"Ground truth field must be ndarray, got {type(gt_field)}"
    if reconstructed_field.ndim != 3:
        return False, f"Reconstructed field must be 3D, got {reconstructed_field.ndim}D"
    if gt_field.ndim != 3:
        return False, f"Ground truth field must be 3D, got {gt_field.ndim}D"
    if reconstructed_field.shape != gt_field.shape:
        return False, f"Shape mismatch: reconstructed {reconstructed_field.shape} vs GT {gt_field.shape}"
    if not np.all(np.isfinite(reconstructed_field)):
        return False, "Reconstructed field contains non-finite values (NaN or Inf)"

    return True, "Reconstruction validation passed"


def get_bao_kwargs(run_index: int, sim_start: int = 0) -> Dict[str, Any]:
    """Provides keyword arguments for BAO reconstruction runs."""
    return {"sim_idx": run_index + sim_start}


def compute_rk_metrics_numpy(recon_field, gt_field, kmin=0.01, kmax=0.5,
                             n_bins=20, box_size=1000.0):
    """Compute r(k) using numpy (for final metrics computation).

    This matches compute_avg_r_bao_differentiable exactly:
    - Same k range [0.01, 0.5]
    - Same number of bins (20)
    - Bins directly in [kmin, kmax], not full k range
    - Equal weight to each k bin
    """
    import numpy.fft as fft

    if hasattr(recon_field, 'block_until_ready'):
        recon_field = np.asarray(recon_field)
    if hasattr(gt_field, 'block_until_ready'):
        gt_field = np.asarray(gt_field)

    delta_r = fft.fftshift(fft.fftn(recon_field))
    delta_g = fft.fftshift(fft.fftn(gt_field))

    Nz, Ny, Nx = recon_field.shape

    kx = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nx)) * Nx / box_size
    ky = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Ny)) * Ny / box_size
    kz = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nz)) * Nz / box_size

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).flatten()

    P_rg = (delta_r * np.conj(delta_g)).flatten().real
    P_rr = (delta_r * np.conj(delta_r)).flatten().real
    P_gg = (delta_g * np.conj(delta_g)).flatten().real

    # Bin directly in BAO range [kmin, kmax], same as compute_avg_r_bao_differentiable
    k_edges = np.linspace(kmin, kmax, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    r_k = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        if np.any(mask):
            P_rg_sum = P_rg[mask].sum()
            P_rr_sum = P_rr[mask].sum()
            P_gg_sum = P_gg[mask].sum()
            denom = np.sqrt(P_rr_sum * P_gg_sum)
            r_k[i] = P_rg_sum / denom if denom > 0 else 0.0
        else:
            r_k[i] = 0.0

    # Average over all bins (equal weight per bin)
    avg_r_bao = float(np.mean(r_k))

    return r_k, k_centers, avg_r_bao


# Large-scale r(k) protection: penalize degradation from baseline
# Baseline r(k) values from gen0.py (initial algorithm) for k in [0.01, 0.2]
# These are the 8 bins (bins 0-7) in the large-scale range
LARGE_SCALE_K_MIN = 0.01
LARGE_SCALE_K_MAX = 0.2
BASELINE_R_K_LARGE_SCALE = [
    0.999425,  # k=0.0223
    0.998325,  # k=0.0467
    0.997222,  # k=0.0713
    0.996355,  # k=0.0958
    0.994659,  # k=0.1202
    0.988216,  # k=0.1448
    0.971646,  # k=0.1693
    0.945917,  # k=0.1938
]
# Penalty coefficient for maximum degradation from baseline
# If max_degradation = 0.05, penalty = 10 * 0.05 = 0.5
LAMBDA_MAX_DEGRADATION = 10.0


def compute_rk_in_range_numpy(recon_field, gt_field, kmin, kmax, n_bins=20, box_size=1000.0,
                              return_per_bin=False):
    """Compute average r(k) in a specific k range.

    Args:
        recon_field: Reconstructed density field
        gt_field: Ground truth density field
        kmin: Minimum k value
        kmax: Maximum k value
        n_bins: Number of k bins
        box_size: Box size in Mpc/h
        return_per_bin: If True, also return per-bin r(k) values

    Returns:
        If return_per_bin=False: float (average r(k))
        If return_per_bin=True: tuple (average r(k), per-bin r(k) array)
    """
    import numpy.fft as fft

    if hasattr(recon_field, 'block_until_ready'):
        recon_field = np.asarray(recon_field)
    if hasattr(gt_field, 'block_until_ready'):
        gt_field = np.asarray(gt_field)

    Nz, Ny, Nx = recon_field.shape

    delta_r = fft.fftshift(fft.fftn(recon_field))
    delta_g = fft.fftshift(fft.fftn(gt_field))

    kx = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nx)) * Nx / box_size
    ky = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Ny)) * Ny / box_size
    kz = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(Nz)) * Nz / box_size

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).flatten()

    P_rg = (delta_r * np.conj(delta_g)).flatten().real
    P_rr = (delta_r * np.conj(delta_r)).flatten().real
    P_gg = (delta_g * np.conj(delta_g)).flatten().real

    k_edges = np.linspace(kmin, kmax, n_bins + 1)
    r_k = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        if np.any(mask):
            P_rg_sum = P_rg[mask].sum()
            P_rr_sum = P_rr[mask].sum()
            P_gg_sum = P_gg[mask].sum()
            denom = np.sqrt(P_rr_sum * P_gg_sum)
            r_k[i] = P_rg_sum / denom if denom > 0 else 0.0

    avg_r_k = float(np.mean(r_k))

    if return_per_bin:
        return avg_r_k, r_k
    return avg_r_k


def aggregate_bao_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]],
    results_dir: str
) -> Dict[str, Any]:
    """Aggregates metrics across multiple BAO reconstruction runs.

    Combined score calculation:
    1. Base score = mean r(k) over full range [0.01, 0.5]
    2. If mean r(k) in large-scale range [0.01, 0.2] < C0 (initial algorithm threshold),
       apply a heavy penalty to discourage sacrificing large-scale performance.
    """
    log_section("Aggregating Metrics")

    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    all_avg_r_bao = []
    all_avg_r_large_scale = []  # r(k) in [0.01, 0.2] range
    all_r_k_large_scale = []  # per-bin r(k) in large-scale range
    all_correlation = []
    all_comp_time = []
    all_r_k = []
    all_k_centers = []

    print(f"\n  Processing {len(results)} simulation results...")

    for idx, (reconstructed_field, gt_field, computation_time) in enumerate(results):
        if hasattr(reconstructed_field, 'block_until_ready'):
            reconstructed_field = np.asarray(reconstructed_field)
        if hasattr(gt_field, 'block_until_ready'):
            gt_field = np.asarray(gt_field)

        # Full range r(k) [0.01, 0.5]
        r_k, k_centers, avg_r_bao = compute_rk_metrics_numpy(reconstructed_field, gt_field)
        all_avg_r_bao.append(avg_r_bao)
        all_r_k.append(r_k)
        all_k_centers.append(k_centers)

        # Large-scale r(k) [0.01, 0.2] - get both average and per-bin values
        # Use n_bins=8 to match the baseline (8 bins in [0.01, 0.2])
        avg_r_large_scale, r_k_large_scale = compute_rk_in_range_numpy(
            reconstructed_field, gt_field,
            kmin=LARGE_SCALE_K_MIN, kmax=LARGE_SCALE_K_MAX,
            n_bins=8, return_per_bin=True
        )
        all_avg_r_large_scale.append(avg_r_large_scale)
        all_r_k_large_scale.append(r_k_large_scale)

        correlation = np.corrcoef(
            reconstructed_field.flatten(),
            gt_field.flatten()
        )[0, 1]
        all_correlation.append(correlation)
        all_comp_time.append(computation_time)

        print(f"    Sim {idx}: r_bao={avg_r_bao:.4f}, r_large_scale={avg_r_large_scale:.4f}, corr={correlation:.4f}, time={computation_time:.2f}s")

    mean_avg_r_bao = float(np.mean(all_avg_r_bao))
    mean_avg_r_large_scale = float(np.mean(all_avg_r_large_scale))
    mean_correlation = float(np.mean(all_correlation))
    mean_comp_time = float(np.mean(all_comp_time))
    std_avg_r_bao = float(np.std(all_avg_r_bao))

    # Compute per-bin degradation from baseline (averaged across simulations)
    mean_r_k_large_scale = np.mean(all_r_k_large_scale, axis=0)  # shape: (8,)

    # Compute degradation: how much worse than baseline at each k bin
    # degradation[i] = max(0, baseline[i] - evolved[i])
    degradations = np.maximum(0, np.array(BASELINE_R_K_LARGE_SCALE) - mean_r_k_large_scale)
    max_degradation = float(np.max(degradations))
    max_degradation_bin = int(np.argmax(degradations))

    # k bin centers for reporting
    k_edges_large_scale = np.linspace(LARGE_SCALE_K_MIN, LARGE_SCALE_K_MAX, 9)
    k_centers_large_scale = 0.5 * (k_edges_large_scale[:-1] + k_edges_large_scale[1:])

    # Compute combined score with maximum degradation penalty
    # penalty = LAMBDA_MAX_DEGRADATION * max_degradation
    penalty = LAMBDA_MAX_DEGRADATION * max_degradation
    combined_score = mean_avg_r_bao - penalty

    print(f"\n  Large-scale r(k) per bin (vs baseline):")
    for i in range(8):
        deg_str = f"  (deg: {degradations[i]:.4f})" if degradations[i] > 0 else ""
        worst_marker = " <-- MAX DEGRADATION" if i == max_degradation_bin and max_degradation > 0 else ""
        print(f"    k={k_centers_large_scale[i]:.4f}: evolved={mean_r_k_large_scale[i]:.4f}, baseline={BASELINE_R_K_LARGE_SCALE[i]:.4f}{deg_str}{worst_marker}")

    if max_degradation > 0:
        print(f"\n  *** LARGE-SCALE DEGRADATION PENALTY ***")
        print(f"      Max degradation: {max_degradation:.4f} at k={k_centers_large_scale[max_degradation_bin]:.4f}")
        print(f"      Penalty: λ={LAMBDA_MAX_DEGRADATION} × {max_degradation:.4f} = {penalty:.4f}")
        print(f"      Combined score: {mean_avg_r_bao:.4f} - {penalty:.4f} = {combined_score:.4f}")
    else:
        print(f"\n  No degradation from baseline - no penalty applied.")
        print(f"  Combined score: {combined_score:.4f}")

    public_metrics = {
        "mean_avg_r_bao": mean_avg_r_bao,
        "std_avg_r_bao": std_avg_r_bao,
        "mean_avg_r_large_scale": mean_avg_r_large_scale,
        "mean_correlation": mean_correlation,
        "num_simulations": len(results),
        "max_degradation": max_degradation,
        "max_degradation_k": float(k_centers_large_scale[max_degradation_bin]),
        "penalty_applied": penalty,
    }

    private_metrics = {
        "mean_computation_time": mean_comp_time,
        "all_avg_r_bao": all_avg_r_bao,
        "all_avg_r_large_scale": all_avg_r_large_scale,
        "all_r_k_large_scale": [r.tolist() for r in all_r_k_large_scale],
        "all_correlation": all_correlation,
        "baseline_r_k_large_scale": BASELINE_R_K_LARGE_SCALE,
        "degradations": degradations.tolist(),
        "lambda_max_degradation": LAMBDA_MAX_DEGRADATION,
    }

    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            r_k=np.array(all_r_k),
            k_centers=np.array(all_k_centers),
            avg_r_bao=np.array(all_avg_r_bao),
            correlation=np.array(all_correlation),
        )
        print(f"\n  Saved detailed r(k) data to {extra_file}")
    except Exception as e:
        print(f"\n  Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    # Build per-bin comparison string
    per_bin_lines = []
    for i in range(8):
        deg_str = f" (degradation: {degradations[i]:.4f})" if degradations[i] > 0.001 else ""
        per_bin_lines.append(f"  k={k_centers_large_scale[i]:.4f}: {mean_r_k_large_scale[i]:.4f} vs baseline {BASELINE_R_K_LARGE_SCALE[i]:.4f}{deg_str}")
    per_bin_str = "\n".join(per_bin_lines)

    penalty_info = ""
    if max_degradation > 0.001:
        penalty_info = f"""
*** LARGE-SCALE DEGRADATION PENALTY ***
- Maximum degradation from baseline: {max_degradation:.4f} at k={k_centers_large_scale[max_degradation_bin]:.4f}
- Penalty = λ × max_degradation = {LAMBDA_MAX_DEGRADATION} × {max_degradation:.4f} = {penalty:.4f}
- Final combined_score: {mean_avg_r_bao:.4f} - {penalty:.4f} = {combined_score:.4f}

Your algorithm performed WORSE than baseline at some large-scale k bins.
Each k bin in [0.01, 0.2] must match or exceed baseline performance.
The penalty is proportional to the MAXIMUM degradation (λ={LAMBDA_MAX_DEGRADATION}).
"""

    feedback = f"""
BAO Reconstruction Performance (with Autodiff Optimization):
- Average r(k) in full range [0.01, 0.5]: {mean_avg_r_bao:.4f} ± {std_avg_r_bao:.4f}
- Average r(k) in large-scale range [0.01, 0.2]: {mean_avg_r_large_scale:.4f}
- Combined score: {combined_score:.4f}
- Mean correlation: {mean_correlation:.4f}
- Evaluated on {len(results)} simulations
- Mean computation time: {mean_comp_time:.2f}s

Large-scale r(k) per bin (must not degrade from baseline):
{per_bin_str}
{penalty_info}
Higher r(k) values indicate better recovery of the initial density field.
Large-scale (small k) performance is especially important for BAO measurements.
DO NOT sacrifice any large-scale k bin - penalty = {LAMBDA_MAX_DEGRADATION} × max_degradation.

IMPORTANT: Your algorithm must be fully JAX-differentiable to benefit from
autodiff parameter optimization. Use jax.numpy instead of numpy for all
operations in the EVOLVE-BLOCK.
"""
    metrics["text_feedback"] = feedback.strip()

    return metrics


def run_autodiff_optimization(
    program_path: str,
    sim_idx: int = 0,
    autodiff_config: Optional[AutodiffParamConfig] = None,
    is_initial_program: bool = False,
    parent_optimized_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], int, float, AutodiffParamOptimizer]:
    """
    Run autodiff-based parameter optimization on a single simulation.

    Returns:
        Tuple of (optimized_params, num_iterations, best_score, optimizer)
    """
    config = autodiff_config or AutodiffParamConfig()

    log_section("Loading Program")
    print(f"\n  Program path: {program_path}")
    print(f"  Simulation index: {sim_idx}")

    # Load the program module
    spec = importlib.util.spec_from_file_location("evolved_module", program_path)
    module = importlib.util.module_from_spec(spec)

    load_start = time.time()
    spec.loader.exec_module(module)
    print(f"  Module loaded in {time.time() - load_start:.2f}s")

    # Read code and parse TUNABLE declarations
    with open(program_path, 'r') as f:
        code = f.read()

    tunable_params = ParamParser.parse_tunable_declarations(code)

    print(f"\n  Found {len(tunable_params)} TUNABLE parameter(s)")
    for p in tunable_params:
        print(f"    - {p.name} = {p.default_value}, bounds={p.bounds}, method={p.method}")

    if not tunable_params:
        print("\n  [!] No TUNABLE parameters found, skipping optimization")
        return {}, 0, 0.0, None

    if is_initial_program:
        print("\n  [!] Initial program: using default parameter values (no optimization)")
        return {p.name: p.default_value for p in tunable_params}, 0, 0.0, None

    # Get functions
    if not hasattr(module, 'reconstruct'):
        raise ValueError("Program must have a 'reconstruct' function")
    if not hasattr(module, 'read_files'):
        raise ValueError("Program must have a 'read_files' function")

    reconstruct_fn = module.reconstruct

    # Load data
    log_section("Loading Simulation Data")
    data_load_start = time.time()
    input_data, gt_field = module.read_files(sim_idx)
    input_data = jnp.asarray(input_data)
    gt_field = jnp.asarray(gt_field)
    print(f"\n  Data shape: {input_data.shape}")
    print(f"  Data dtype: {input_data.dtype}")
    print(f"  Load time: {time.time() - data_load_start:.2f}s")

    # Build fixed params from inheritance
    fixed_params = parent_optimized_params or {}
    params_to_optimize = [p for p in tunable_params if p.name not in fixed_params]

    if not params_to_optimize:
        print("\n  [!] All parameters inherited from parent, no optimization needed")
        return fixed_params, 0, 0.0, None

    # Run optimization
    optimizer = AutodiffParamOptimizer(config)

    def reconstruct_wrapper(data, **params):
        return reconstruct_fn(data, **params)

    optimized_values, num_iters, best_score = optimizer.optimize(
        params_to_optimize=params_to_optimize,
        reconstruct_fn=reconstruct_wrapper,
        input_data=input_data,
        gt_field=gt_field,
        fixed_params=fixed_params,
    )

    # Merge with fixed params
    all_params = fixed_params.copy()
    all_params.update(optimized_values)

    return all_params, num_iters, best_score, optimizer


def inject_optimized_params(program_path: str, param_values: Dict[str, float]):
    """Inject optimized parameter values into the program file."""
    import re

    log_section("Injecting Optimized Parameters")
    print(f"\n  Target file: {program_path}")

    with open(program_path, 'r') as f:
        code = f.read()

    modified_code = code

    for param_name, value in param_values.items():
        # Update TUNABLE comment default value
        tunable_pattern = rf'(#\s*TUNABLE:\s*{param_name}\s*=\s*)([-]?[0-9.e+-]+)'
        if isinstance(value, int):
            tunable_replacement = rf'\g<1>{value}'
        else:
            tunable_replacement = rf'\g<1>{value:.6g}'
        modified_code = re.sub(tunable_pattern, tunable_replacement, modified_code)

        # Update function parameter default value
        param_pattern = rf'(\b{param_name}\s*(?::\s*\w+)?\s*=\s*)([-]?[0-9.e+-]+)'
        if isinstance(value, int):
            param_replacement = rf'\g<1>{value}'
        else:
            param_replacement = rf'\g<1>{value:.6g}'
        modified_code = re.sub(param_pattern, param_replacement, modified_code)

        print(f"    {param_name} → {value:.6g}")

    with open(program_path, 'w') as f:
        f.write(modified_code)

    backup_path = program_path + ".optimized.py"
    with open(backup_path, 'w') as f:
        f.write(modified_code)
    print(f"\n  Backup saved to: {backup_path}")


def main(
    program_path: str,
    results_dir: str,
    sim_start: int = 0,
    sim_end: int = 3,
    autodiff_enabled: bool = True,
    autodiff_lr: float = 0.1,
    autodiff_max_iter: int = 50,
    is_initial_program: Optional[bool] = None,
    parent_program_path: Optional[str] = None,
):
    """
    Runs the BAO reconstruction evaluation with autodiff parameter optimization.
    """
    total_start_time = time.time()

    # Auto-detect if this is the initial program
    if is_initial_program is None:
        is_initial_program = "/gen_0/" in results_dir or results_dir.endswith("/gen_0")

    log_header("BAO RECONSTRUCTION EVALUATION WITH AUTODIFF")
    print(f"  Timestamp: {timestamp()}")
    print(f"  Program: {program_path}")
    print(f"  Results directory: {results_dir}")
    print(f"  Simulations: {sim_start} to {sim_end - 1} ({sim_end - sim_start} total)")
    print(f"  Autodiff optimization: {'ENABLED' if autodiff_enabled else 'DISABLED'}")
    if autodiff_enabled:
        print(f"    - Learning rate: {autodiff_lr}")
        print(f"    - Max iterations: {autodiff_max_iter}")
    print(f"  Is initial program: {is_initial_program}")
    if parent_program_path:
        print(f"  Parent program: {parent_program_path}")

    if is_initial_program:
        log_section("INITIAL PROGRAM MODE")
        print("\n  Skipping parameter optimization - using default values from code")

    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = sim_end - sim_start
    optimized_params = {}
    autodiff_cost = 0
    optimizer = None

    # Run autodiff optimization
    if autodiff_enabled and not is_initial_program:
        autodiff_config = AutodiffParamConfig(
            learning_rate=autodiff_lr,
            max_iterations=autodiff_max_iter,
            verbose=True,
            log_every=1,
        )

        try:
            optimized_params, autodiff_cost, opt_score, optimizer = run_autodiff_optimization(
                program_path=program_path,
                sim_idx=sim_start,
                autodiff_config=autodiff_config,
                is_initial_program=is_initial_program,
            )

            if optimized_params:
                inject_optimized_params(program_path, optimized_params)

        except Exception as e:
            log_section("AUTODIFF OPTIMIZATION FAILED")
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

    # Run final evaluation
    log_section("Final Evaluation on All Simulations")
    print(f"\n  Running {num_experiment_runs} simulations with optimized parameters...")

    correct = True
    error_msg = None
    metrics = {}

    eval_start_time = time.time()
    try:
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        experiment_fn = getattr(module, "run_experiment")

        all_results = []
        for i in range(num_experiment_runs):
            kwargs = get_bao_kwargs(i, sim_start)
            run_result = experiment_fn(**kwargs)

            is_valid, val_msg = validate_reconstruction(run_result)
            if not is_valid:
                correct = False
                if error_msg is None:
                    error_msg = f"Validation failed: {val_msg}"

            all_results.append(run_result)
            print(f"  Run {i + 1}/{num_experiment_runs} completed")

        metrics = aggregate_bao_metrics(all_results, results_dir)

    except Exception as e:
        correct = False
        error_msg = str(e)
        metrics = {"combined_score": 0.0}
        import traceback
        traceback.print_exc()

    eval_time = time.time() - eval_start_time

    # Add autodiff info to metrics
    metrics['optimized_params'] = optimized_params
    metrics['autodiff_cost'] = autodiff_cost
    metrics['autodiff_enabled'] = autodiff_enabled
    if optimizer and optimizer.optimization_history:
        metrics['optimization_history'] = optimizer.optimization_history

    # Write result.json (MadEvolve format)
    result = {
        "success": correct,
        "combined_score": metrics.get("combined_score", 0.0),
        "public_metrics": metrics.get("public", {}),
        "private_metrics": metrics.get("private", {}),
        "text_feedback": metrics.get("text_feedback", ""),
        "error": error_msg,
    }

    result_path = os.path.join(results_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Result saved to {result_path}")

    # Print final summary
    total_time = time.time() - total_start_time

    log_header("EVALUATION COMPLETE")
    print(f"  Timestamp: {timestamp()}")
    print(f"  Status: {'SUCCESS' if correct else 'FAILED'}")
    if not correct:
        print(f"  Error: {error_msg}")
    print()
    print("  Timing:")
    print(f"    - Autodiff optimization: {autodiff_cost} iterations")
    print(f"    - Final evaluation: {eval_time:.2f}s")
    print(f"    - Total time: {total_time:.2f}s")
    print()

    if optimized_params:
        print("  Optimized Parameters:")
        for param_name, param_value in optimized_params.items():
            print(f"    - {param_name}: {param_value:.6g}")
        print()

    print("  Final Metrics:")
    print(f"    - Combined score: {metrics.get('combined_score', 'N/A')}")
    if 'public' in metrics:
        for k, v in metrics['public'].items():
            if isinstance(v, float):
                print(f"    - {k}: {v:.6f}")
            else:
                print(f"    - {k}: {v}")

    return metrics, correct, error_msg


if __name__ == "__main__":
    # MadEvolve dispatcher interface: python evaluate_autodiff.py <candidate.py>
    # Also supports legacy argparse interface for standalone use.
    if len(sys.argv) == 2 and not sys.argv[1].startswith("--"):
        # MadEvolve mode: positional arg is candidate path, cwd is work_dir
        program_path = sys.argv[1]
        results_dir = os.getcwd()
        main(
            program_path=program_path,
            results_dir=results_dir,
            sim_start=0,
            sim_end=1,
            autodiff_enabled=True,
            autodiff_lr=0.1,
            autodiff_max_iter=20,
        )
    else:
        # Legacy argparse mode
        parser = argparse.ArgumentParser(
            description="BAO reconstruction evaluator with autodiff parameter optimization"
        )
        parser.add_argument("--program_path", type=str, default="initial_autodiff.py")
        parser.add_argument("--results_dir", type=str, default="test_results")
        parser.add_argument("--sim_start", type=int, default=0)
        parser.add_argument("--sim_end", type=int, default=3)
        parser.add_argument("--autodiff_enabled", action="store_true", default=True)
        parser.add_argument("--no_autodiff", action="store_true")
        parser.add_argument("--autodiff_lr", type=float, default=0.1)
        parser.add_argument("--autodiff_max_iter", type=int, default=50)
        parser.add_argument("--is_initial_program", action="store_true")
        parser.add_argument("--parent_program_path", type=str, default=None)

        args = parser.parse_args()

        autodiff_enabled = args.autodiff_enabled and not args.no_autodiff

        main(
            args.program_path,
            args.results_dir,
            args.sim_start,
            args.sim_end,
            autodiff_enabled=autodiff_enabled,
            autodiff_lr=args.autodiff_lr,
            autodiff_max_iter=args.autodiff_max_iter,
            is_initial_program=args.is_initial_program,
            parent_program_path=args.parent_program_path,
        )
