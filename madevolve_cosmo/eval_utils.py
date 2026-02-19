"""Evaluation utilities for MadEvolve-Cosmo.

This module provides helper functions for running evaluations of evolved programs
in cosmology applications.
"""

from __future__ import annotations

import importlib.util
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def load_program_module(program_path: str) -> Any:
    """
    Dynamically load a Python module from file path.

    Args:
        program_path: Path to the Python file to load

    Returns:
        Loaded module object

    Raises:
        ImportError: If module cannot be loaded
    """
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None
) -> None:
    """
    Save evaluation results to JSON files.

    Creates two files:
    - metrics.json: Contains all computed metrics
    - correct.json: Contains success status and error message

    Args:
        results_dir: Directory to save results
        metrics: Dictionary of computed metrics
        correct: Whether evaluation succeeded
        error: Error message if evaluation failed
    """
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save correctness status
    correct_path = os.path.join(results_dir, "correct.json")
    correct_data = {"correct": correct}
    if error:
        correct_data["error"] = error
    with open(correct_path, "w") as f:
        json.dump(correct_data, f, indent=2)


def run_evaluation(
    program_path: str,
    results_dir: str,
    experiment_fn_name: str = "run_experiment",
    num_runs: int = 1,
    get_experiment_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
    validate_fn: Optional[Callable[[Any], Tuple[bool, str]]] = None,
    aggregate_metrics_fn: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Run evaluation of an evolved program.

    This function:
    1. Loads the program from the specified path
    2. Runs the experiment function multiple times with different kwargs
    3. Validates each result using the validation function
    4. Aggregates results using the aggregation function
    5. Saves results to JSON files

    Args:
        program_path: Path to the Python file to evaluate
        results_dir: Directory to save results
        experiment_fn_name: Name of the function to call in the program
        num_runs: Number of times to run the experiment
        get_experiment_kwargs: Function that returns kwargs for each run index
        validate_fn: Function to validate each run result, returns (is_valid, error_msg)
        aggregate_metrics_fn: Function to aggregate results into metrics

    Returns:
        Tuple of (metrics_dict, is_correct, error_message)
    """
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Load the program module
        module = load_program_module(program_path)

        # Get the experiment function
        if not hasattr(module, experiment_fn_name):
            error_msg = f"Program does not have '{experiment_fn_name}' function"
            save_json_results(results_dir, {}, False, error_msg)
            return {}, False, error_msg

        experiment_fn = getattr(module, experiment_fn_name)

        # Run experiments
        results = []
        for run_idx in range(num_runs):
            try:
                # Get kwargs for this run
                kwargs = {}
                if get_experiment_kwargs is not None:
                    kwargs = get_experiment_kwargs(run_idx)

                # Run experiment
                result = experiment_fn(**kwargs)

                # Validate result
                if validate_fn is not None:
                    is_valid, validation_error = validate_fn(result)
                    if not is_valid:
                        print(f"Run {run_idx} validation failed: {validation_error}")
                        continue

                results.append(result)

            except Exception as e:
                print(f"Run {run_idx} failed with error: {e}")
                traceback.print_exc()
                continue

        # Check if we have any valid results
        if not results:
            error_msg = "All experiment runs failed"
            save_json_results(results_dir, {}, False, error_msg)
            return {}, False, error_msg

        # Aggregate metrics
        if aggregate_metrics_fn is not None:
            metrics = aggregate_metrics_fn(results)
        else:
            # Default aggregation: just count successful runs
            metrics = {
                "combined_score": len(results) / num_runs,
                "successful_runs": len(results),
                "total_runs": num_runs,
            }

        # Save results
        save_json_results(results_dir, metrics, True, None)
        return metrics, True, None

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
        save_json_results(results_dir, {}, False, error_msg)
        return {}, False, error_msg


# Alias for backward compatibility
run_alpha_evolve_eval = run_evaluation
