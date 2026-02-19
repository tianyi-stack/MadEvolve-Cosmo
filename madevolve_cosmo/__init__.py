"""MadEvolve-Cosmo: Cosmology applications using MadEvolve framework.

This package provides cosmology-specific extensions and adapters for the
MadEvolve LLM-driven evolution framework. It includes adapters for:
- BAO (Baryon Acoustic Oscillation) reconstruction
- Tidal field reconstruction
- LDL (Learned Displacement for Lagrangian) models

The core evolution functionality is provided by the madevolve package.
"""

__version__ = "0.1.0"

# Re-export core MadEvolve components
from madevolve import (
    EvolutionOrchestrator,
    EvolutionConfig,
    PopulationConfig,
    ModelConfig,
    ExecutorConfig,
    StorageConfig,
)

# Cosmology adapters (local implementations)
from madevolve_cosmo.analyzer import (
    BAOAdapter,
    LDLAdapter,
    TidalReconstructionAdapter,
    # Metrics classes
    BAOMetrics,
    LDLMetrics,
    TidalMetrics,
    # Convenience function
    register_all_adapters,
)

# Evaluation utilities
from madevolve_cosmo.eval_utils import (
    run_evaluation,
    run_alpha_evolve_eval,  # Alias for backward compatibility
    save_json_results,
    load_program_module,
)

__all__ = [
    # Core evolution
    "EvolutionOrchestrator",
    "EvolutionConfig",
    "PopulationConfig",
    "ModelConfig",
    "ExecutorConfig",
    "StorageConfig",
    # Cosmology adapters
    "BAOAdapter",
    "LDLAdapter",
    "TidalReconstructionAdapter",
    # Metrics classes
    "BAOMetrics",
    "LDLMetrics",
    "TidalMetrics",
    # Adapter registration
    "register_all_adapters",
    # Evaluation utilities
    "run_evaluation",
    "run_alpha_evolve_eval",
    "save_json_results",
    "load_program_module",
]
