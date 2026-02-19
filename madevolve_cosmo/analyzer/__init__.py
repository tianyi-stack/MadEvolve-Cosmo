"""
MadEvolve-Cosmo Analyzer Module

Provides cosmology-specific adapters for the MadEvolve report generator.

This module extends the MadEvolve analyzer framework with adapters for:
- BAO (Baryon Acoustic Oscillation) reconstruction
- LDL (Lagrangian Deep Learning) for tSZ prediction
- Tidal field reconstruction (21cm intensity mapping)

Usage:
    from madevolve_cosmo.analyzer.adapters import BAOAdapter, LDLAdapter, TidalReconstructionAdapter
    from madevolve.analyzer import DataExtractor, ReportGenerator, register_adapter

    # Register the adapters
    register_adapter('bao', BAOAdapter)
    register_adapter('ldl', LDLAdapter)
    register_adapter('tidal', TidalReconstructionAdapter)

    # Use an adapter
    adapter = BAOAdapter()
    extractor = DataExtractor(adapter)
    data = extractor.extract_evolution_data("/path/to/results")

    generator = ReportGenerator(adapter)
    report = generator.generate_full_report(data)
"""

from .adapters import (
    BAOAdapter,
    BAOMetrics,
    BAOMetricsAdapter,
    LDLAdapter,
    LDLMetrics,
    LDLMetricsAdapter,
    TidalReconstructionAdapter,
    TidalMetrics,
    TidalMetricsAdapter,
)

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


def register_all_adapters():
    """
    Convenience function to register all cosmology adapters with MadEvolve.

    Usage:
        from madevolve_cosmo.analyzer import register_all_adapters
        register_all_adapters()

        # Now you can use:
        from madevolve.analyzer import get_adapter
        adapter = get_adapter('bao')
    """
    from madevolve.analyzer import register_adapter

    register_adapter('bao', BAOAdapter)
    register_adapter('ldl', LDLAdapter)
    register_adapter('tidal', TidalReconstructionAdapter)
