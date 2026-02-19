"""
Scenario adapters for the MadEvolve-Cosmo report generator.

Each adapter provides domain-specific:
- Metrics parsing and comparison
- LLM prompts for analysis
- Report templates

Available Adapters:
- BAOAdapter: BAO (Baryon Acoustic Oscillation) reconstruction
- LDLAdapter: Lagrangian Deep Learning for tSZ prediction
- TidalReconstructionAdapter: Tidal field reconstruction (21cm intensity mapping)
"""

from .bao import BAOAdapter, BAOMetrics, BAOMetricsAdapter
from .ldl import LDLAdapter, LDLMetrics, LDLMetricsAdapter
from .tidal import TidalReconstructionAdapter, TidalMetrics, TidalMetricsAdapter

__all__ = [
    # BAO
    "BAOAdapter",
    "BAOMetrics",
    "BAOMetricsAdapter",
    # LDL
    "LDLAdapter",
    "LDLMetrics",
    "LDLMetricsAdapter",
    # Tidal
    "TidalReconstructionAdapter",
    "TidalMetrics",
    "TidalMetricsAdapter",
]
