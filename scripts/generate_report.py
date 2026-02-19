#!/usr/bin/env python3
"""
Report Generator Script for MadEvolve-Cosmo Evolution Experiments.

This script generates comprehensive analysis reports from evolution experiment results.
Supports BAO reconstruction, LDL tSZ prediction, and Tidal field reconstruction scenarios.

Usage:
    python scripts/generate_report.py /path/to/results --scenario bao
    python scripts/generate_report.py /path/to/results --scenario ldl --model gpt-4o
    python scripts/generate_report.py /path/to/results --scenario tidal --quick
    python scripts/generate_report.py /path/to/results --list-scenarios

Examples:
    # Generate full LLM-powered report for BAO experiment
    python scripts/generate_report.py results/bao_exp_001 -s bao

    # Generate quick report (no LLM) for LDL experiment
    python scripts/generate_report.py results/ldl_exp_001 -s ldl -q

    # Generate report and export to PDF
    python scripts/generate_report.py results/tidal_exp_001 -s tidal --pdf

    # Auto-detect scenario and use specific LLM model
    python scripts/generate_report.py results/exp_001 --model claude-3-5-sonnet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_adapters():
    """Register all cosmology adapters with MadEvolve."""
    from madevolve.analyzer import register_adapter
    from madevolve_cosmo.analyzer import (
        BAOAdapter,
        LDLAdapter,
        TidalReconstructionAdapter,
    )

    register_adapter('bao', BAOAdapter)
    register_adapter('ldl', LDLAdapter)
    register_adapter('tidal', TidalReconstructionAdapter)


def auto_detect_scenario(results_dir: str) -> str:
    """
    Attempt to auto-detect the scenario from results directory.

    Checks experiment_config.yaml and metrics.json for hints.
    """
    from pathlib import Path
    import json
    import yaml

    results_path = Path(results_dir)

    # Check experiment_config.yaml
    config_path = results_path / "experiment_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            task_desc = str(config.get('task_description', '')).lower()

            if 'bao' in task_desc or 'baryon' in task_desc or 'reconstruction' in task_desc:
                return 'bao'
            elif 'ldl' in task_desc or 'tsz' in task_desc or 'sunyaev' in task_desc or 'camels' in task_desc:
                return 'ldl'
            elif 'tidal' in task_desc or '21cm' in task_desc:
                return 'tidal'
        except Exception:
            pass

    # Check metrics.json in gen_0 or best directory
    for subdir in ['best', 'gen_0']:
        metrics_path = results_path / subdir / 'results' / 'metrics.json'
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                public = metrics.get('public', metrics)

                # BAO-specific fields
                if 'mean_avg_r_bao' in public or 'mean_avg_r_large_scale' in public:
                    return 'bao'
                # LDL-specific fields
                if 'mean_r_k_all_scales' in public or 'train_loss' in public:
                    return 'ldl'
                # Tidal-specific fields
                if 'mean_r2d_metric' in public or 'tunable_params' in metrics:
                    return 'tidal'
            except Exception:
                pass

    # Default to bao if can't detect
    print("Warning: Could not auto-detect scenario, defaulting to 'bao'")
    return 'bao'


def validate_results_dir(results_dir: str) -> bool:
    """Validate that the results directory has required files."""
    from pathlib import Path

    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return False

    # Check for required files/directories
    required = ['evolution_db.sqlite', 'gen_0']
    missing = []
    for item in required:
        if not (results_path / item).exists():
            missing.append(item)

    if missing:
        print(f"Error: Missing required items in results directory: {missing}")
        return False

    # Check for best directory or at least gen_0
    if not (results_path / 'best').exists():
        print("Warning: No 'best' directory found, will use highest generation")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis reports for MadEvolve-Cosmo evolution experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Scenarios:
  bao     - BAO (Baryon Acoustic Oscillation) Reconstruction
  ldl     - Lagrangian Deep Learning for tSZ Prediction
  tidal   - Tidal Field Reconstruction (21cm Intensity Mapping)

Examples:
  %(prog)s results/bao_exp -s bao
  %(prog)s results/ldl_exp -s ldl --quick
  %(prog)s results/tidal_exp -s tidal --pdf --model gpt-4o
        """
    )

    parser.add_argument(
        'results_dir',
        nargs='?',
        help='Path to evolution results directory'
    )
    parser.add_argument(
        '-s', '--scenario',
        choices=['bao', 'ldl', 'tidal'],
        help='Scenario type (auto-detected if not specified)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: results_dir/evolution_report.md)'
    )
    parser.add_argument(
        '-m', '--model',
        default='gemini/gemini-2.5-pro-preview-05-06',
        help='LLM model to use for analysis (default: gemini/gemini-2.5-pro-preview-05-06)'
    )
    parser.add_argument(
        '-q', '--quick',
        action='store_true',
        help='Generate quick report without LLM analysis'
    )
    parser.add_argument(
        '-g', '--generation',
        type=int,
        help='Use specific generation as best (instead of best/ directory)'
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Export report to PDF'
    )
    parser.add_argument(
        '--no-code',
        action='store_true',
        help='Do not include algorithm code in appendix'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print evolution summary, do not generate full report'
    )
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenarios and exit'
    )

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        print("\nAvailable Scenarios:")
        print("-" * 50)
        print("  bao   - BAO (Baryon Acoustic Oscillation) Reconstruction")
        print("          Metrics: r(k) BAO range, r(k) large scale, degradation")
        print()
        print("  ldl   - Lagrangian Deep Learning (tSZ Prediction)")
        print("          Metrics: r(k) all scales, cross-correlation, validation loss")
        print()
        print("  tidal - Tidal Field Reconstruction (21cm Intensity Mapping)")
        print("          Metrics: r_2D metric, correlation, parameter optimization")
        print("-" * 50)
        return 0

    # Check results_dir is provided
    if not args.results_dir:
        parser.error("results_dir is required (use --list-scenarios to see available scenarios)")

    # Validate results directory
    if not validate_results_dir(args.results_dir):
        return 1

    # Setup adapters
    try:
        setup_adapters()
    except ImportError as e:
        print(f"Error: Failed to import required modules: {e}")
        print("Make sure madevolve is installed: pip install madevolve")
        return 1

    # Import after setup
    from madevolve.analyzer import (
        get_adapter,
        DataExtractor,
        ReportGenerator,
        LITELLM_AVAILABLE,
        export_markdown_to_pdf,
    )

    # Auto-detect or use specified scenario
    scenario = args.scenario or auto_detect_scenario(args.results_dir)
    print(f"Using scenario: {scenario}")

    # Get adapter
    try:
        adapter = get_adapter(scenario)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Extract data
    print(f"Extracting data from: {args.results_dir}")
    extractor = DataExtractor(adapter)
    try:
        data = extractor.extract_evolution_data(
            args.results_dir,
            best_generation=args.generation
        )
    except Exception as e:
        print(f"Error extracting data: {e}")
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print(f"Evolution Summary - {adapter.metrics_adapter.scenario_description}")
    print("=" * 60)
    print(f"Results Directory: {data.results_dir}")
    print(f"Duration: {data.duration_hours:.2f} hours")
    print(f"Total Generations: {data.history.total_generations}")
    print(f"Programs Evaluated: {data.history.total_programs}")
    print(f"Successful Programs: {data.history.successful_programs}")

    if data.baseline.metrics and data.best.metrics:
        baseline_score = data.baseline.metrics.combined_score
        best_score = data.best.metrics.combined_score
        improvement = best_score - baseline_score
        if baseline_score != 0:
            improvement_pct = 100 * improvement / abs(baseline_score)
        else:
            improvement_pct = 0

        print(f"\nBaseline Score: {baseline_score:.4f}")
        print(f"Best Score: {best_score:.4f}")
        print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"Best Generation: {data.best.generation}")

    print("=" * 60)

    if args.summary_only:
        return 0

    # Determine output path
    output_path = args.output
    if not output_path:
        output_path = str(Path(args.results_dir) / "evolution_report.md")

    # Generate report
    print(f"\nGenerating report...")

    use_quick = args.quick or not LITELLM_AVAILABLE
    if use_quick and not args.quick:
        print("Note: LLM not available, generating quick report")

    generator = ReportGenerator(adapter, model=args.model)

    try:
        if use_quick:
            report = generator.generate_quick_report(data, output_path=output_path)
            print(f"Quick report generated: {output_path}")
        else:
            print(f"Using LLM model: {args.model}")
            report = generator.generate_full_report(
                data,
                output_path=output_path,
                include_code=not args.no_code
            )
            print(f"Full report generated: {output_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export to PDF if requested
    if args.pdf:
        pdf_path = output_path.replace('.md', '.pdf')
        print(f"Exporting to PDF: {pdf_path}")
        try:
            result = export_markdown_to_pdf(output_path, pdf_path)
            if result:
                print(f"PDF exported: {result}")
            else:
                print("Warning: PDF export failed (is playwright installed?)")
        except Exception as e:
            print(f"Warning: PDF export failed: {e}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
