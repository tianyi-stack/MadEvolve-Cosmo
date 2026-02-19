"""
Evaluation script for LDL CAMELS example in MadEvolve.

This script evaluates model architectures using CORRECT two-stage evaluation:

STAGE 1 - TRAINING (CV_0):
  - Trains the model on CV_0 simulation (training set)
  - Optimizes model parameters using L-BFGS-B
  - Extracts best_param from training

STAGE 2 - VALIDATION (CV_1, CV_2, CV_3, CV_4):
  - Runs inference (no training) on MULTIPLE validation simulations
  - Uses best_param from Stage 1
  - Computes power spectrum metrics (r(k), transfer function) for each
  - Averages r(k) across all validation sets for robust fitness

FITNESS COMPUTATION:
  - Based on AVERAGED performance across multiple validation sets
  - Primary metric: mean of mean_r_k_all_scales from CV_1, CV_2, CV_3, CV_4
  - Fitness = averaged_mean_r_k - failed_runs*1000

DATA SPLIT STRATEGY:
  - CV_0: Training simulation (for parameter optimization)
  - CV_1, CV_2, CV_3, CV_4: Validation simulations (for fitness evaluation)
  - CV_5 ~ CV_10: Test simulations (for final testing, see test.py)

This multi-validation approach:
  - Prevents overfitting to a single validation set's characteristics
  - Selects architectures with better generalization across different cosmic variance
  - Provides more robust fitness estimates
"""

import json
import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add project root path (for MPI wrapper scripts that need access to dependencies)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _save_madevolve_result(results_dir, metrics, correct, error=None):
    """Write result.json in MadEvolve format."""
    os.makedirs(results_dir, exist_ok=True)
    result = {
        "success": correct,
        "combined_score": metrics.get("combined_score", 0.0) if metrics else 0.0,
        "public_metrics": {k: v for k, v in metrics.items()
                          if k not in ("combined_score", "text_feedback") and isinstance(v, (int, float, str, bool))} if metrics else {},
        "private_metrics": {},
        "text_feedback": metrics.get("text_feedback", "") if metrics else "",
        "error": error,
    }
    result_path = os.path.join(results_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Result saved to {result_path}")


def get_training_kwargs(run_idx: int) -> dict:
    """
    Generate kwargs for training run on CV_0.

    Args:
        run_idx: Index of the current run (0)

    Returns:
        Dictionary of kwargs to pass to run_experiment for training
    """
    # Training configuration
    # tSZ (thermal SZ effect) uses 3 layers based on paper (Dai & Seljak 2020, Table 2)
    return {
        'sim_id': 'CV_0',     # Train on CV_0 (training set)
        'run_idx': run_idx,
        'Nstep': 3,           # Number of displacement layers (tSZ: 3 layers)
        'n': 1.0,             # Smoothing kernel exponent
        'max_iter': 250,      # Maximum optimization iterations
        'verbose': False,
    }


def get_validation_kwargs(run_idx: int, sim_id: str = 'CV_1') -> dict:
    """
    Generate kwargs for validation run on a specific simulation.

    Args:
        run_idx: Index of the current run (0)
        sim_id: Simulation ID for validation (e.g., 'CV_1', 'CV_2', 'CV_3', 'CV_4')

    Returns:
        Dictionary of kwargs to pass to run_inference for validation
    """
    # Validation configuration
    # Uses same architecture hyperparameters as training
    return {
        'sim_id': sim_id,     # Validate on specified simulation
        'run_idx': run_idx,
        'Nstep': 3,           # Number of displacement layers (tSZ: 3 layers)
        'n': 1.0,             # Smoothing kernel exponent
        'verbose': False,
    }


# Define validation simulations
VALIDATION_SIMS = ['CV_1', 'CV_2', 'CV_3', 'CV_4']


def aggregate_metrics(results: list) -> dict:
    """
    Aggregate results from multiple runs into a single fitness score.

    FITNESS STRATEGY:
    - Primary: Mean r(k) over ALL scales on full volume (higher is better)
    - Failure penalty: Large penalty for failed runs

    Args:
        results: List of result dicts from multiple runs

    Returns:
        Dictionary containing:
        - combined_score: Main fitness score = mean_r_k_all_scales - failures*1000
                         (HIGHER is better for MadEvolve)
        - Power spectrum metrics and validation losses for monitoring
    """
    # Collect losses and field data
    losses = []
    training_times = []
    predicted_fields = []
    target_fields = []
    failed_runs = 0

    for r in results:
        if r.get('error') or not r.get('success', False):
            failed_runs += 1
        else:
            losses.append(r['loss'])
            if 'training_time' in r:
                training_times.append(r['training_time'])
            # Collect field data for power spectrum (now used for FITNESS)
            if 'predicted_field_data' in r and 'target_field_data' in r:
                predicted_fields.append(r['predicted_field_data'])
                target_fields.append(r['target_field_data'])

    # Handle complete failure case
    if len(predicted_fields) == 0:
        return {
            'combined_score': float(-1e10),
            'mean_validation_loss': 1e6 if len(losses) == 0 else float(np.mean(losses)),
            'failed_runs': failed_runs,
            'success_rate': 0.0,
            'mean_r_k_all_scales': 0.0,
        }

    # === PRIMARY FITNESS: Mean r(k) over ALL scales ===
    # Compute r(k) on full volume (not masked) over all k modes
    r_k_all_scales_list = []
    cross_correlations_large = []
    transfer_accuracies_large = []

    for i, (pred_field_data, targ_field_data) in enumerate(zip(predicted_fields, target_fields)):
        try:
            # pred_field_data and targ_field_data are numpy arrays from pmesh fields
            # Convert to ArrayMesh for nbodykit
            from nbodykit.lab import ArrayMesh, FFTPower
            from mpi4py import MPI

            # Get BoxSize and Nmesh from the result
            BoxSize = results[i].get('BoxSize', 205.0)  # Default from TNG300
            Nmesh = results[i].get('Nmesh', 64)

            # Verify and reshape if needed
            expected_shape = (Nmesh, Nmesh, Nmesh)
            if pred_field_data.shape != expected_shape:
                if np.prod(pred_field_data.shape) == Nmesh**3:
                    pred_field_data = pred_field_data.reshape(expected_shape)
                    targ_field_data = targ_field_data.reshape(expected_shape)

            # Create ArrayMesh objects from numpy arrays (without comm to avoid MPI issues)
            mesh_pred = ArrayMesh(pred_field_data, BoxSize=BoxSize)
            mesh_targ = ArrayMesh(targ_field_data, BoxSize=BoxSize)

            # Compute auto power spectra
            P_pred = FFTPower(mesh_pred, mode='1d', kmin=0.01, kmax=10.0)
            P_targ = FFTPower(mesh_targ, mode='1d', kmin=0.01, kmax=10.0)

            # Compute cross power spectrum
            P_cross = FFTPower(mesh_pred, mode='1d', second=mesh_targ, kmin=0.01, kmax=10.0)

            k = P_pred.power['k']
            P_pred_k = P_pred.power['power'].real
            P_targ_k = P_targ.power['power'].real
            P_cross_k = P_cross.power['power'].real

            # Avoid division by zero
            valid_mask = (P_pred_k > 0) & (P_targ_k > 0)

            # Cross-correlation coefficient r(k)
            r_k = np.zeros_like(k)
            r_k[valid_mask] = P_cross_k[valid_mask] / np.sqrt(P_pred_k[valid_mask] * P_targ_k[valid_mask])

            # === PRIMARY FITNESS: Mean r(k) over ALL valid scales ===
            if np.sum(valid_mask) > 0:
                r_k_all_scales = float(np.mean(r_k[valid_mask]))
                r_k_all_scales_list.append(r_k_all_scales)
            else:
                r_k_all_scales_list.append(0.0)

            # Transfer function T(k)
            T_k = np.zeros_like(k)
            T_k[valid_mask] = np.sqrt(P_pred_k[valid_mask] / P_targ_k[valid_mask])

            # Also compute large scale metrics for monitoring
            large_scale_mask = (k < 0.5) & valid_mask

            if np.sum(large_scale_mask) > 0:
                # Average metrics at large scales
                r_large = np.mean(r_k[large_scale_mask])
                T_accuracy_large = 1.0 - np.mean(np.abs(T_k[large_scale_mask] - 1.0))

                cross_correlations_large.append(r_large)
                transfer_accuracies_large.append(T_accuracy_large)
            else:
                # No valid large scale modes - use fallback
                cross_correlations_large.append(0.0)
                transfer_accuracies_large.append(0.0)

        except Exception as e:
            # If power spectrum computation fails, append zeros
            print(f"Warning: Power spectrum computation failed for run {i}: {e}")
            r_k_all_scales_list.append(0.0)
            cross_correlations_large.append(0.0)
            transfer_accuracies_large.append(0.0)

    # Compute mean r(k) over all scales (PRIMARY FITNESS)
    if len(r_k_all_scales_list) > 0:
        mean_r_k_all_scales = float(np.mean(r_k_all_scales_list))
        std_r_k_all_scales = float(np.std(r_k_all_scales_list))
    else:
        mean_r_k_all_scales = 0.0
        std_r_k_all_scales = 0.0

    # Compute large scale metrics (for monitoring only)
    if len(cross_correlations_large) > 0:
        mean_r_large = float(np.mean(cross_correlations_large))
        std_r_large = float(np.std(cross_correlations_large))
        mean_T_accuracy = float(np.mean(transfer_accuracies_large))
        std_T_accuracy = float(np.std(transfer_accuracies_large))
    else:
        mean_r_large = 0.0
        std_r_large = 0.0
        mean_T_accuracy = 0.0
        std_T_accuracy = 0.0

    # === COMBINED SCORE: Mean r(k) over all scales - failure penalty ===
    combined_score = (
        mean_r_k_all_scales +         # Higher r(k) = better correlation (already positive, 0-1 range)
        -failed_runs * 1000.0         # Harsh penalty for failures
    )

    # Compute loss statistics (for monitoring)
    if len(losses) > 0:
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses)) if len(losses) > 1 else 0.0
    else:
        mean_loss = 1e6
        std_loss = 0.0

    # === RETURN METRICS ===
    metrics = {
        # PRIMARY FITNESS (what evolution optimizes)
        'combined_score': float(combined_score),

        # PRIMARY FITNESS COMPONENT: r(k) averaged over ALL scales
        'mean_r_k_all_scales': mean_r_k_all_scales,
        'std_r_k_all_scales': std_r_k_all_scales,

        # MONITORING: Loss values (not used in fitness, for monitoring only)
        'mean_validation_loss': mean_loss,  # Note: Now just "loss" in results, but keeping name for compatibility
        'std_validation_loss': std_loss,
        'mean_training_time': float(np.mean(training_times)) if training_times else 0.0,
        'std_training_time': float(np.std(training_times)) if len(training_times) > 1 else 0.0,

        # Operational metrics
        'failed_runs': int(failed_runs),
        'success_rate': float(len(losses) / len(results)) if len(results) > 0 else 0.0,

        # MONITORING: Power spectrum metrics at large scales
        'cross_correlation_large_scale': mean_r_large,
        'std_cross_correlation': std_r_large,
        'transfer_function_accuracy': mean_T_accuracy,
        'std_transfer_function': std_T_accuracy,

        # Score breakdown
        'r_k_component': mean_r_k_all_scales,
        'failure_penalty': float(-failed_runs * 1000.0),
        'note': 'Fitness = mean_r_k_all_scales - failures*1000. r(k) computed on full volume over all k scales.',
    }

    return metrics


def validate_result(result: dict) -> tuple[bool, str | None]:
    """
    Validate a single run result for correctness.

    This function checks if the training run completed successfully
    and produced reasonable results.

    Args:
        result: Result dictionary from a single run

    Returns:
        (is_valid, error_message):
        - is_valid: True if result is valid, False otherwise
        - error_message: Description of error if invalid, None otherwise
    """
    # Check for explicit errors
    if result.get('error'):
        return False, f"Training error: {result['error']}"

    # Check if optimization succeeded
    # Note: For quick tests with low max_iter, we allow non-convergence
    # as long as the loss values are reasonable
    if not result.get('success', False):
        # Check if we at least got valid loss values
        loss = result.get('loss', float('inf'))
        if np.isnan(loss) or np.isinf(loss) or loss > 1e6:
            return False, "Optimization did not converge and produced invalid results"
        # Otherwise, allow it (useful for quick tests with reduced iterations)

    # Check for numerical issues
    loss = result.get('loss', float('inf'))

    if np.isnan(loss) or np.isinf(loss):
        return False, f"Invalid loss: {loss}"

    # Check if loss exploded (indicates numerical instability)
    if loss > 1e6:
        return False, f"Loss exploded: {loss:.2e}"

    # Check if loss is suspiciously low (might indicate bug)
    if loss < 1e-10:
        return False, f"Loss suspiciously low: {loss:.2e}"

    # All checks passed
    return True, None


def main(program_path: str, results_dir: str, use_mpi: bool = True, num_mpi_ranks: int = 4):
    """
    Main evaluation entry point called by MadEvolve.

    CORRECT TWO-STAGE EVALUATION:
    1. Stage 1 - Training: Train on CV_0 (training set) to get optimal parameters
    2. Stage 2 - Validation: Evaluate trained model on CV_1 (validation set)
    3. Compute fitness score based on VALIDATION performance

    This ensures proper train/validation split and prevents data leakage.

    Args:
        program_path: Path to the Python file to evaluate (evolved initial.py)
        results_dir: Directory to save evaluation results
        use_mpi: Whether to use MPI for parallel execution (default: True)
        num_mpi_ranks: Number of MPI processes to use (default: 4)

    Returns:
        Tuple of (metrics, correct, error):
        - metrics: Dictionary of aggregated metrics from validation set
        - correct: Boolean indicating if evaluation succeeded
        - error: Error message if evaluation failed, None otherwise
    """
    if use_mpi:
        print(f"\n{'='*70}")
        print(f"Two-Stage Evaluation")
        print(f"{'='*70}")
        print(f"Program: {program_path}")
        print(f"Results: {results_dir}")
        print(f"{'='*70}\n")

        # === STAGE 1: TRAINING ON CV_0 ===
        print("=" * 70)
        print("STAGE 1: Training on CV_0 (Training Set)")
        print("=" * 70)

        train_results_dir = os.path.join(results_dir, 'train')
        train_result, train_correct, train_error = _run_with_mpi(
            program_path=program_path,
            results_dir=train_results_dir,
            num_mpi_ranks=num_mpi_ranks,
        )

        if not train_correct:
            error_msg = f"Training failed: {train_error}"
            _save_madevolve_result(results_dir, {}, False, error=error_msg)
            return {}, False, error_msg

        # Extract trained parameters
        trained_param = train_result['best_param']
        print(f"\nTraining completed. Loss on CV_0: {train_result['loss']:.6f}")

        # === STAGE 2: VALIDATION ON MULTIPLE SIMULATIONS ===
        print("\n" + "=" * 70)
        print(f"STAGE 2: Validation on {len(VALIDATION_SIMS)} simulations: {', '.join(VALIDATION_SIMS)}")
        print("=" * 70)

        val_results = []
        val_losses = []

        for val_sim in VALIDATION_SIMS:
            print(f"\n  Running validation on {val_sim}...")
            val_results_dir = os.path.join(results_dir, f'val_{val_sim}')
            val_result, val_correct, val_error = _run_inference_with_mpi(
                program_path=program_path,
                results_dir=val_results_dir,
                param=trained_param,
                sim_id=val_sim,
                num_mpi_ranks=num_mpi_ranks,
            )

            if not val_correct:
                print(f"  WARNING: Validation on {val_sim} failed: {val_error}")
                # Continue with other validation sets instead of failing completely
                continue

            # Validate validation result
            is_valid, validation_err = validate_result(val_result)
            if not is_valid:
                print(f"  WARNING: Validation check failed for {val_sim}: {validation_err}")
                continue

            val_results.append(val_result)
            val_losses.append(val_result['loss'])
            print(f"  {val_sim} completed. Loss: {val_result['loss']:.6f}")

        # Check if we have at least one successful validation
        if len(val_results) == 0:
            error_msg = "All validation sets failed"
            _save_madevolve_result(results_dir, {}, False, error=error_msg)
            return {}, False, error_msg

        print(f"\nValidation completed on {len(val_results)}/{len(VALIDATION_SIMS)} simulations")
        print(f"Mean validation loss: {np.mean(val_losses):.6f}")

        # === COMPUTE FITNESS FROM VALIDATION SETS ===
        print("\n" + "=" * 70)
        print("Computing Fitness from Multiple Validation Sets")
        print("=" * 70)

        # Aggregate metrics based on VALIDATION performance across all validation sets
        metrics = aggregate_metrics(val_results)

        # Add training info for monitoring
        metrics['train_loss'] = float(train_result['loss'])
        metrics['mean_val_loss'] = float(np.mean(val_losses))
        metrics['std_val_loss'] = float(np.std(val_losses)) if len(val_losses) > 1 else 0.0
        metrics['num_validation_sims'] = len(val_results)
        metrics['validation_sims'] = [r.get('sim_id', 'unknown') for r in val_results]

        # Store individual validation losses
        for i, (sim_id, loss) in enumerate(zip(VALIDATION_SIMS[:len(val_losses)], val_losses)):
            metrics[f'val_loss_{sim_id}'] = float(loss)

        print(f"\nFitness Score (averaged over {len(val_results)} validation sets): {metrics['combined_score']:.6f}")
        print(f"  - Mean r(k) all scales: {metrics['mean_r_k_all_scales']:.6f}")
        print(f"  - Std r(k) across val sets: {metrics.get('std_r_k_all_scales', 0.0):.6f}")
        print(f"  - Training loss (CV_0): {metrics['train_loss']:.6f}")
        print(f"  - Mean validation loss: {metrics['mean_val_loss']:.6f} +/- {metrics['std_val_loss']:.6f}")
        print("=" * 70 + "\n")

        # Save results to JSON (required by MadEvolve)
        _save_madevolve_result(results_dir, metrics, True)

        return metrics, True, None

    else:
        # Fall back to serial execution (not recommended, very slow)
        error_msg = "Serial execution not implemented for two-stage evaluation. Please use MPI."
        _save_madevolve_result(results_dir, {}, False, error=error_msg)
        return {}, False, error_msg


def _run_inference_with_mpi(program_path: str, results_dir: str, param: list,
                             sim_id: str = 'CV_1', num_mpi_ranks: int = 4):
    """
    Run inference using MPI for parallel performance (no training).

    This function creates a temporary wrapper script that:
    1. Loads the program module
    2. Calls run_inference with trained parameters
    3. Saves results from rank 0 only

    Args:
        program_path: Path to the Python file to evaluate
        results_dir: Directory to save inference results
        param: Trained model parameters
        sim_id: Simulation ID for validation (e.g., 'CV_1', 'CV_2', 'CV_3', 'CV_4')
        num_mpi_ranks: Number of MPI processes to use

    Returns:
        Tuple of (result_dict, correct, error)
    """
    import subprocess
    import tempfile
    import json

    os.makedirs(results_dir, exist_ok=True)

    # Get validation kwargs for specified simulation
    kwargs = get_validation_kwargs(0, sim_id=sim_id)

    # Save parameters to file for MPI script to load
    param_file = os.path.join(results_dir, "trained_params.pkl")
    with open(param_file, "wb") as f:
        pickle.dump(param, f)

    # Create MPI wrapper script for inference
    wrapper_script = f"""
import sys
import os
import json
import pickle
import numpy as np

# Add MadEvolve to path
project_root = '{str(PROJECT_ROOT)}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load the program module
import importlib.util
spec = importlib.util.spec_from_file_location("program", "{program_path}")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {program_path}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Load trained parameters
with open("{param_file}", "rb") as f:
    param = pickle.load(f)

# Convert list to numpy array if needed
if isinstance(param, list):
    param = np.array(param)

# Run inference with MPI (no training)
result = module.run_inference(
    sim_id='{kwargs['sim_id']}',
    param=param,
    run_idx={kwargs['run_idx']},
    Nstep={kwargs['Nstep']},
    n={kwargs['n']},
    verbose={kwargs['verbose']}
)

# Save result from rank 0 only
from mpi4py import MPI
if MPI.COMM_WORLD.rank == 0:
    os.makedirs("{results_dir}", exist_ok=True)

    # Save as pickle (handles numpy arrays better than JSON)
    with open("{results_dir}/result_run_0.pkl", "wb") as f:
        pickle.dump(result, f)

    print(f"\\nMPI inference completed successfully on {{MPI.COMM_WORLD.size}} ranks")
"""

    # Write wrapper script
    wrapper_path = os.path.join(results_dir, "mpi_inference_wrapper.py")
    with open(wrapper_path, "w") as f:
        f.write(wrapper_script)

    try:
        # Run with mpirun
        cmd = ["mpirun", "-n", str(num_mpi_ranks), "python", wrapper_path]

        if num_mpi_ranks > 1:
            print(f"    Running inference on {sim_id} with {num_mpi_ranks} MPI ranks...")
        else:
            print(f"    Running inference on {sim_id} in serial mode...")

        # Capture output for logging
        log_file = os.path.join(results_dir, "mpi_inference_execution.log")
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                timeout=1800,  # 30 min timeout (inference is faster than training)
            )

        if result.returncode != 0:
            with open(log_file, "r") as f:
                error_log = f.read()
            error_msg = f"MPI inference failed with return code {result.returncode}. Log: {error_log[-500:]}"
            return {}, False, error_msg

        # Load result
        result_path = os.path.join(results_dir, "result_run_0.pkl")
        if not os.path.exists(result_path):
            error_msg = f"Inference result file not found: {result_path}"
            return {}, False, error_msg

        with open(result_path, "rb") as f:
            run_result = pickle.load(f)

        return run_result, True, None

    except subprocess.TimeoutExpired:
        return {}, False, "MPI inference timed out after 30 minutes"
    except Exception as e:
        import traceback
        error_msg = f"MPI inference error: {str(e)}\n{traceback.format_exc()}"
        return {}, False, error_msg


def _run_with_mpi(program_path: str, results_dir: str, num_mpi_ranks: int = 4):
    """
    Run training using MPI for parallel performance.

    This function creates a temporary wrapper script that:
    1. Loads the program module
    2. Calls run_experiment with proper kwargs
    3. Saves results from rank 0 only

    Args:
        program_path: Path to the Python file to evaluate
        results_dir: Directory to save training results
        num_mpi_ranks: Number of MPI processes to use

    Returns:
        Tuple of (result_dict, correct, error)
    """
    import subprocess
    import tempfile
    import json

    os.makedirs(results_dir, exist_ok=True)

    # Get training kwargs (CV_0)
    kwargs = get_training_kwargs(0)

    # Create MPI wrapper script
    wrapper_script = f"""
import sys
import os
import json
import pickle

# Add MadEvolve to path
project_root = '{str(PROJECT_ROOT)}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load the program module
import importlib.util
spec = importlib.util.spec_from_file_location("program", "{program_path}")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {{program_path}}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Run experiment with MPI
result = module.run_experiment(
    sim_id='{kwargs['sim_id']}',
    run_idx={kwargs['run_idx']},
    Nstep={kwargs['Nstep']},
    n={kwargs['n']},
    max_iter={kwargs['max_iter']},
    verbose={kwargs['verbose']}
)

# Save result from rank 0 only
from mpi4py import MPI
if MPI.COMM_WORLD.rank == 0:
    os.makedirs("{results_dir}", exist_ok=True)

    # Save as pickle (handles numpy arrays better than JSON)
    with open("{results_dir}/result_run_0.pkl", "wb") as f:
        pickle.dump(result, f)

    print(f"\\nMPI evaluation completed successfully on {{MPI.COMM_WORLD.size}} ranks")
"""

    # Write wrapper script
    wrapper_path = os.path.join(results_dir, "mpi_wrapper.py")
    with open(wrapper_path, "w") as f:
        f.write(wrapper_script)

    try:
        # Run with mpirun
        cmd = ["mpirun", "-n", str(num_mpi_ranks), "python", wrapper_path]

        if num_mpi_ranks > 1:
            print(f"Running training on CV_0 (training set) with {num_mpi_ranks} MPI ranks...")
        else:
            print(f"Running training on CV_0 (training set) in serial mode...")

        # Capture output for logging
        log_file = os.path.join(results_dir, "mpi_execution.log")
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                timeout=3600,  # 1 hour timeout
            )

        if result.returncode != 0:
            with open(log_file, "r") as f:
                error_log = f.read()
            error_msg = f"MPI execution failed with return code {result.returncode}. Log: {error_log[-500:]}"
            _save_madevolve_result(results_dir, {}, False, error=error_msg)
            return {}, False, error_msg

        # Load result
        result_path = os.path.join(results_dir, "result_run_0.pkl")
        if not os.path.exists(result_path):
            error_msg = f"Result file not found: {result_path}"
            _save_madevolve_result(results_dir, {}, False, error=error_msg)
            return {}, False, error_msg

        with open(result_path, "rb") as f:
            run_result = pickle.load(f)

        # Validate result
        is_valid, validation_err = validate_result(run_result)

        if not is_valid:
            error_msg = f"Validation failed: {validation_err}"
            return {}, False, error_msg

        # Return training result (not final metrics yet)
        return run_result, True, None

    except subprocess.TimeoutExpired:
        return {}, False, "MPI training timed out after 1 hour"
    except Exception as e:
        import traceback
        error_msg = f"MPI training error: {str(e)}\n{traceback.format_exc()}"
        return {}, False, error_msg


if __name__ == '__main__':
    """
    Two modes:
    1. MadEvolve dispatcher: python evaluate.py candidate.py
    2. Standalone testing:   python evaluate.py
    """
    if len(sys.argv) == 2 and not sys.argv[1].startswith("--"):
        # MadEvolve dispatcher mode: python evaluate.py candidate.py
        program_path = sys.argv[1]
        results_dir = os.getcwd()
        main(program_path=program_path, results_dir=results_dir)
    else:
        # Standalone testing mode
        import tempfile

        initial_path = Path(__file__).parent / 'initial.py'

        if not initial_path.exists():
            print(f"Error: Could not find initial.py at {initial_path}")
            sys.exit(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n{'='*70}")
            print(f"Testing LDL CAMELS Evaluation")
            print(f"{'='*70}")
            print(f"Program: {initial_path}")
            print(f"Results dir: {tmpdir}")
            print(f"{'='*70}\n")

            metrics, correct, error = main(str(initial_path), tmpdir)

            print(f"\n{'='*70}")
            print(f"Evaluation Results")
            print(f"{'='*70}")
            print(f"Success: {correct}")
            if error:
                print(f"Error: {error}")
            if metrics:
                print(f"\nMetrics:")
                for key, value in sorted(metrics.items()):
                    print(f"  {key:30s}: {value}")
            print(f"{'='*70}\n")
