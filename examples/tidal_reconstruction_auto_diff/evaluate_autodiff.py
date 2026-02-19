"""
Tidal Reconstruction Evaluator with Autodiff Parameter Optimization

This module evaluates tidal reconstruction algorithms by measuring the
2D cross-correlation r(k_perp, k_para) between reconstructed and ground truth density fields.

Key Feature: Uses JAX autodiff for parameter optimization instead of grid search.
The evolved programs must be fully differentiable (using JAX operations) so that
gradients can flow through the entire reconstruction pipeline.
"""

import os
import sys
import json
import re
import argparse
import time
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
TUNABLE_PATTERN = re.compile(
    r'#\s*TUNABLE:\s*(\w+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s*,\s*bounds\s*=\s*\(\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*'
    r',\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\)'
    r'(?:\s*,\s*method\s*=\s*(\w+))?'
)


@dataclass
class TunableParam:
    name: str
    default_value: float
    bounds: Tuple[float, float]
    method: str = "autodiff"


class ParamParser:
    @staticmethod
    def parse_tunable_declarations(code: str) -> List[TunableParam]:
        params = []
        for m in TUNABLE_PATTERN.finditer(code):
            params.append(TunableParam(
                name=m.group(1),
                default_value=float(m.group(2)),
                bounds=(float(m.group(3)), float(m.group(4))),
                method=m.group(5) or "autodiff",
            ))
        return params


def _save_madevolve_result(results_dir, metrics, correct, error=None):
    """Write result.json in MadEvolve format."""
    os.makedirs(results_dir, exist_ok=True)
    result = {
        "success": correct,
        "combined_score": metrics.get("combined_score", 0.0) if metrics else 0.0,
        "public_metrics": metrics.get("public", {}) if metrics else {},
        "private_metrics": metrics.get("private", {}) if metrics else {},
        "text_feedback": metrics.get("text_feedback", "") if metrics else "",
        "error": error,
    }
    result_path = os.path.join(results_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Result saved to {result_path}")


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


def compute_r2d_metric_differentiable(recon_field: jnp.ndarray, gt_field: jnp.ndarray,
                                       box_size: float = 1000.0) -> jnp.ndarray:
    """
    Compute average r_2D[1:6, 1:6] in a fully differentiable way.
    This is the objective function for autodiff optimization.

    The metric captures cross-correlation in 2D (k_perp, k_para) space,
    focusing on scales 1<=i<=5, 1<=j<=5.

    Args:
        recon_field: Reconstructed density field (3D), (1+delta) format
        gt_field: Ground truth density field (3D), (1+delta) format
        box_size: Box size in Mpc/h

    Returns a scalar JAX array for gradient computation.
    """
    nmesh = recon_field.shape[0]

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
    n_bins = 32  # Number of bins in each dimension
    k_max = nmesh // 2  # Maximum k in grid units

    k_perp_edges = jnp.linspace(0, k_max, n_bins + 1)
    k_para_edges = jnp.linspace(0, k_max, n_bins + 1)

    # Compute r_2D for bins 1-5 (we only need these for the metric)
    r_2D_region = jnp.zeros((5, 5))

    for i in range(1, 6):  # indices 1-5
        for j in range(1, 6):  # indices 1-5
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
            r_2D_region = r_2D_region.at[i-1, j-1].set(r_bin)

    # Calculate metric: average of r_2D[1:6, 1:6]
    metric = jnp.mean(r_2D_region)

    return metric


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

        # Define the loss function (negative r_2d for minimization)
        def loss_fn(params_array):
            param_dict = dict(zip(param_names, params_array))
            param_dict.update(fixed_params)
            recon_field = reconstruct_fn(input_data, **param_dict)
            metric = compute_r2d_metric_differentiable(recon_field, gt_field)
            return -metric  # Negative for minimization

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
              " | r_2D       | grad_norm  | status")
        print("  " + "-" * (7 + 15 * len(param_names) + 40))

        total_forward_passes = 0
        optimization_start_time = time.time()

        for iteration in range(self.config.max_iterations):
            iter_start_time = time.time()

            try:
                # Compute loss and gradient in one forward+backward pass
                current_loss, grads = value_and_grad_fn(param_values)
                total_forward_passes += 1

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
                    print(f"\n  [OK] Converged at iteration {iteration + 1} (grad_norm={grad_norm:.2e} < {self.config.convergence_threshold})")
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
        print(f"  Best score (avg r_2D): {best_score:.6f}")
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
        marker = ">" if is_best else " "
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


def get_tidal_kwargs(run_index: int, sim_start: int = 0) -> Dict[str, Any]:
    """Provides keyword arguments for tidal reconstruction runs."""
    return {"sim_idx": run_index + sim_start}


def compute_r2d_metrics_numpy(recon_field, gt_field, box_size=1000.0):
    """Compute r_2D using numpy (for final metrics computation).

    Uses CosCal.FastLoop.Dimension.BiDimension for 2D decomposition,
    matching the original evaluator.
    """
    from CosCal.FastLoop import Dimension
    from scipy import fft

    if hasattr(recon_field, 'block_until_ready'):
        recon_field = np.asarray(recon_field)
    if hasattr(gt_field, 'block_until_ready'):
        gt_field = np.asarray(gt_field)

    nmesh = recon_field.shape[0]
    kf = 2 * np.pi / box_size

    # Convert to overdensity delta = rho/rho_bar - 1
    delta_recon = (recon_field - 1.0).astype(np.float32)
    delta_gt = (gt_field - 1.0).astype(np.float32)

    # Compute Fourier transforms
    norm_factor = 1.0 / nmesh**3
    delta_recon_k = fft.fftn(delta_recon) * norm_factor
    delta_gt_k = fft.fftn(delta_gt) * norm_factor

    # Compute power spectra in k-space
    delta_recon_k_conj = np.conjugate(delta_recon_k)
    delta_gt_k_conj = np.conjugate(delta_gt_k)

    volume_factor = box_size**3
    PK11 = (delta_recon_k * delta_recon_k_conj * volume_factor).real.astype(np.float32)
    PK22 = (delta_gt_k * delta_gt_k_conj * volume_factor).real.astype(np.float32)
    PK12 = (delta_recon_k * delta_gt_k_conj * volume_factor).real.astype(np.float32)

    # Get frequency grid for BiDimension
    fn = fft.fftfreq(nmesh, 1.0 / nmesh)

    # Use BiDimension to decompose into 2D (k_perp, k_para)
    modes11, k_perp11, k_para11, Pk11_2d = Dimension.BiDimension(PK11, fn, nmesh)
    modes22, k_perp22, k_para22, Pk22_2d = Dimension.BiDimension(PK22, fn, nmesh)
    modes12, k_perp12, k_para12, Pk12_2d = Dimension.BiDimension(PK12, fn, nmesh)

    # Calculate r_2D = P12 / sqrt(P11 * P22)
    denominator = np.sqrt(Pk11_2d * Pk22_2d)
    r_2D = np.zeros_like(Pk12_2d)
    mask = denominator > 0
    r_2D[mask] = Pk12_2d[mask] / denominator[mask]

    # Calculate metric: average of r_2D[1:6, 1:6] (indices 1-5 inclusive)
    if r_2D.shape[0] > 5 and r_2D.shape[1] > 5:
        metric = float(np.mean(r_2D[1:6, 1:6]))
    else:
        metric = float(np.mean(r_2D))

    return r_2D, metric


def aggregate_tidal_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]],
    results_dir: str
) -> Dict[str, Any]:
    """Aggregates metrics across multiple tidal reconstruction runs."""
    log_section("Aggregating Metrics")

    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    all_r2d_metrics = []
    all_correlation = []
    all_comp_time = []
    all_r_2D_maps = []

    print(f"\n  Processing {len(results)} simulation results...")

    for idx, (reconstructed_field, gt_field, computation_time) in enumerate(results):
        if hasattr(reconstructed_field, 'block_until_ready'):
            reconstructed_field = np.asarray(reconstructed_field)
        if hasattr(gt_field, 'block_until_ready'):
            gt_field = np.asarray(gt_field)

        # Compute r_2D metrics using CosCal
        r_2D, metric_value = compute_r2d_metrics_numpy(reconstructed_field, gt_field)
        all_r2d_metrics.append(metric_value)
        all_r_2D_maps.append(r_2D)

        # Compute correlation as secondary metric (using sampling for speed)
        sample_step = 10
        recon_sample = reconstructed_field[::sample_step, ::sample_step, ::sample_step].flatten()
        gt_sample = gt_field[::sample_step, ::sample_step, ::sample_step].flatten()
        correlation = np.corrcoef(recon_sample, gt_sample)[0, 1]
        all_correlation.append(correlation)
        all_comp_time.append(computation_time)

        print(f"    Sim {idx}: r_2D={metric_value:.4f}, corr={correlation:.4f}, time={computation_time:.2f}s")

    mean_r2d = float(np.mean(all_r2d_metrics))
    mean_correlation = float(np.mean(all_correlation))
    mean_comp_time = float(np.mean(all_comp_time))
    std_r2d = float(np.std(all_r2d_metrics))

    # Combined score for optimization
    combined_score = mean_r2d

    public_metrics = {
        "mean_r2d_metric": mean_r2d,
        "std_r2d_metric": std_r2d,
        "mean_correlation": mean_correlation,
        "num_simulations": len(results),
    }

    private_metrics = {
        "mean_computation_time": mean_comp_time,
        "all_r2d_metrics": all_r2d_metrics,
        "all_correlation": all_correlation,
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
            r_2D_maps=np.array(all_r_2D_maps),
            r2d_metrics=np.array(all_r2d_metrics),
            correlation=np.array(all_correlation),
        )
        print(f"\n  Saved detailed r_2D data to {extra_file}")
    except Exception as e:
        print(f"\n  Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    feedback = f"""
Tidal Reconstruction Performance (with Autodiff Optimization):
- Average r_2D[1:6, 1:6] metric: {mean_r2d:.4f} +/- {std_r2d:.4f}
- Mean correlation: {mean_correlation:.4f}
- Evaluated on {len(results)} simulations
- Mean computation time: {mean_comp_time:.2f}s

Higher r_2D values indicate better recovery of the density field
at different scales and directions (k_perp, k_para), which improves
the reconstruction quality for tidal field analysis.

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
    if not hasattr(module, 'run_reconstruction'):
        raise ValueError("Program must have a 'run_reconstruction' function")
    if not hasattr(module, 'read_files'):
        raise ValueError("Program must have a 'read_files' function")

    reconstruct_fn = module.run_reconstruction

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

        print(f"    {param_name} -> {value:.6g}")

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
    sim_end: int = 2,
    autodiff_enabled: bool = True,
    autodiff_lr: float = 0.1,
    autodiff_max_iter: int = 50,
    is_initial_program: Optional[bool] = None,
    parent_program_path: Optional[str] = None,
):
    """
    Runs the tidal reconstruction evaluation with autodiff parameter optimization.
    """
    total_start_time = time.time()

    # Auto-detect if this is the initial program
    if is_initial_program is None:
        is_initial_program = "/gen_0/" in results_dir or results_dir.endswith("/gen_0")

    log_header("TIDAL RECONSTRUCTION EVALUATION WITH AUTODIFF")
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

    eval_start_time = time.time()
    correct = True
    error_msg = None
    metrics = {}

    try:
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        experiment_fn = getattr(module, "run_experiment")

        # Run experiments
        results = []
        for run_idx in range(num_experiment_runs):
            kwargs = get_tidal_kwargs(run_idx, sim_start)
            try:
                result = experiment_fn(**kwargs)
                is_valid, val_msg = validate_reconstruction(result)
                if is_valid:
                    results.append(result)
                else:
                    print(f"  Run {run_idx} validation failed: {val_msg}")
            except Exception as e:
                print(f"  Run {run_idx} error: {e}")

        if not results:
            correct = False
            error_msg = "All runs failed validation"
            metrics = {"combined_score": 0.0}
        else:
            metrics = aggregate_tidal_metrics(results, results_dir)

    except Exception as e:
        import traceback
        correct = False
        error_msg = f"Evaluation error: {e}\n{traceback.format_exc()}"
        metrics = {"combined_score": 0.0}

    eval_time = time.time() - eval_start_time

    # Add autodiff info to metrics
    metrics['optimized_params'] = optimized_params
    metrics['autodiff_cost'] = autodiff_cost
    metrics['autodiff_enabled'] = autodiff_enabled
    if optimizer and optimizer.optimization_history:
        metrics['optimization_history'] = optimizer.optimization_history

    # Save result.json for MadEvolve
    _save_madevolve_result(results_dir, metrics, correct, error=error_msg)

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
    if len(sys.argv) == 2 and not sys.argv[1].startswith("--"):
        # MadEvolve dispatcher mode: python evaluate_autodiff.py candidate.py
        program_path = sys.argv[1]
        results_dir = os.getcwd()
        main(program_path=program_path, results_dir=results_dir,
             sim_start=0, sim_end=2,
             autodiff_enabled=True, autodiff_lr=0.1, autodiff_max_iter=50)
    else:
        # Legacy argparse mode
        parser = argparse.ArgumentParser(
            description="Tidal reconstruction evaluator with autodiff parameter optimization"
        )
        parser.add_argument("--program_path", type=str, default="initial_autodiff.py")
        parser.add_argument("--results_dir", type=str, default="test_results")
        parser.add_argument("--sim_start", type=int, default=0)
        parser.add_argument("--sim_end", type=int, default=2)
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
