"""
Tidal Reconstruction Adapter for MadEvolve-Cosmo.

This adapter provides domain-specific handling for:
- Tidal field reconstruction metrics (21cm intensity mapping)
- Cosmology algorithm analysis prompts
- Scientific report templates

The tidal reconstruction problem involves recovering large-scale density
fluctuations from 21cm intensity maps where foreground subtraction has
removed line-of-sight modes. The algorithm uses the tidal modulation of
small-scale clustering to infer the missing information.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from madevolve.analyzer import (
    AlgorithmInfo,
    BaseMetrics,
    EvolutionData,
    MetricsAdapter,
    PromptAdapter,
    ReportTemplateAdapter,
    ScenarioAdapter,
)


@dataclass
class TunableParameter:
    """Container for a tunable parameter."""
    name: str
    default_value: float
    bounds: tuple
    method: str
    frozen_value: Optional[float] = None


# Parameter descriptions for tidal reconstruction algorithms
PARAMETER_DESCRIPTIONS: Dict[str, str] = {
    # Input filtering parameters
    "smooth_scale": "Gaussian smoothing scale (Mpc/h) applied to the input density field to suppress high-frequency noise while preserving large-scale tidal features.",
    "filter_scale": "Gaussian filter scale (Mpc/h) for the input density field, controlling the separation between tracer fluctuations and background tidal field.",
    "k_min_par": "High-pass cutoff for line-of-sight (k_parallel) modes. Removes foreground-contaminated low-k_parallel modes from the input, mimicking the 21cm foreground wedge.",

    # Spectral/potential parameters
    "spectral_index": "Power-law index controlling how different k-scales are weighted when computing the vector field. Interpolates between displacement field (~0) and tidal field (~2).",
    "f_ani": "Anisotropy factor that modifies the effective potential by weighting k_z differently from k_perp. Accounts for redshift-space distortions and anisotropic noise geometry.",

    # Non-linearity handling
    "sat_amp": "Saturation amplitude (in units of field std dev) for the tanh clipping. Suppresses high-density peaks (halos) that introduce non-Gaussian shot noise.",

    # Reconstruction kernel weights
    "w_trace": "Weight for the isotropic trace component (T_xx + T_yy + T_zz) in the reconstruction. Controls contribution of monopole information.",
    "w_par": "Weight for longitudinal tensor derivatives (nabla_z^2 T_zz). Adjusts how much the reconstruction trusts line-of-sight correlations vs transverse.",
    "w_cross": "Weight for cross tensor derivatives (nabla_perp nabla_z T_perp_z). Controls the mixing between transverse and longitudinal information.",
    "w_perp": "Weight for transverse tensor derivatives (nabla_perp^2 T_perp). Usually normalized to 1.0 as the reference weight.",

    # Regularization
    "reg_value": "Regularization parameter to prevent division by zero in potential inversion. Stabilizes the reconstruction at very low k.",
    "reg": "Regularization parameter for numerical stability in spectral operations.",

    # Legacy/alternative names
    "alpha": "Spectral index or power-law exponent for scale-dependent weighting.",
    "beta": "Secondary spectral parameter, often used for anisotropic corrections.",
    "gamma": "Tertiary parameter for fine-tuning reconstruction kernel.",
}


@dataclass
class TidalMetrics(BaseMetrics):
    """Tidal reconstruction-specific metrics."""
    # Primary metrics
    mean_r2d_metric: float = 0.0
    std_r2d_metric: float = 0.0
    mean_correlation: float = 0.0
    num_simulations: int = 0
    mean_computation_time: float = 0.0

    # Detailed metrics
    all_r2d_metrics: List[float] = field(default_factory=list)
    all_correlation: List[float] = field(default_factory=list)

    # Parameter optimization info
    num_tunable_params: int = 0
    num_optimized_params: int = 0
    param_opt_cost: int = 0
    tunable_params: List[TunableParameter] = field(default_factory=list)
    optimized_params: Dict[str, float] = field(default_factory=dict)


def parse_tunable_parameters_from_metrics(metrics_data: Dict) -> List[TunableParameter]:
    """
    Parse tunable parameters from metrics.json data.

    The metrics.json contains a 'tunable_params' list with parameter info.
    """
    parameters = []
    tunable_list = metrics_data.get('tunable_params', [])

    for param_info in tunable_list:
        if isinstance(param_info, dict):
            bounds = param_info.get('bounds', [0, 1])
            if isinstance(bounds, list) and len(bounds) >= 2:
                bounds = (bounds[0], bounds[1])
            else:
                bounds = (0, 1)

            param = TunableParameter(
                name=param_info.get('name', 'unknown'),
                default_value=param_info.get('default_value', 0.0),
                bounds=bounds,
                method=param_info.get('method', 'grid'),
                frozen_value=param_info.get('frozen_value'),
            )
            parameters.append(param)

    return parameters


class TidalMetricsAdapter(MetricsAdapter):
    """Adapter for parsing and displaying tidal reconstruction metrics."""

    @property
    def scenario_name(self) -> str:
        return "tidal"

    @property
    def scenario_description(self) -> str:
        return "Tidal Field Reconstruction (21cm Intensity Mapping)"

    def parse_metrics(self, metrics_data: Dict[str, Any]) -> TidalMetrics:
        """Parse tidal reconstruction metrics from metrics.json."""
        public = metrics_data.get('public', {})
        private = metrics_data.get('private', {})

        # Parse tunable parameters
        tunable_params = parse_tunable_parameters_from_metrics(metrics_data)
        optimized_params = metrics_data.get('optimized_params', {})

        return TidalMetrics(
            combined_score=metrics_data.get('combined_score', 0.0),
            text_feedback=metrics_data.get('text_feedback', ''),
            execution_time_mean=metrics_data.get('execution_time_mean', 0.0),
            execution_time_std=metrics_data.get('execution_time_std', 0.0),
            raw_public=public,
            raw_private=private,
            # Tidal-specific metrics
            mean_r2d_metric=public.get('mean_r2d_metric', 0.0),
            std_r2d_metric=public.get('std_r2d_metric', 0.0),
            mean_correlation=public.get('mean_correlation', 0.0),
            num_simulations=public.get('num_simulations', 0),
            mean_computation_time=private.get('mean_computation_time', 0.0),
            all_r2d_metrics=private.get('all_r2d_metrics', []),
            all_correlation=private.get('all_correlation', []),
            # Parameter optimization info
            num_tunable_params=len(tunable_params),
            num_optimized_params=len(optimized_params),
            param_opt_cost=metrics_data.get('param_opt_cost', 0),
            tunable_params=tunable_params,
            optimized_params=optimized_params,
        )

    def get_metrics_comparison_table(
        self,
        baseline: BaseMetrics,
        best: BaseMetrics
    ) -> str:
        """Generate tidal reconstruction metrics comparison table."""
        if not isinstance(baseline, TidalMetrics):
            baseline = TidalMetrics(combined_score=baseline.combined_score)
        if not isinstance(best, TidalMetrics):
            best = TidalMetrics(combined_score=best.combined_score)

        # Calculate improvement percentage
        if baseline.mean_r2d_metric != 0:
            r2d_improvement_pct = 100 * (best.mean_r2d_metric - baseline.mean_r2d_metric) / baseline.mean_r2d_metric
        else:
            r2d_improvement_pct = 0

        lines = [
            "| Metric | Baseline | Best | Improvement |",
            "|--------|----------|------|-------------|",
            f"| **r_2D Metric (avg)** | {baseline.mean_r2d_metric:.4f} | {best.mean_r2d_metric:.4f} | {best.mean_r2d_metric - baseline.mean_r2d_metric:+.4f} ({r2d_improvement_pct:+.1f}%) |",
            f"| r_2D Std Dev | {baseline.std_r2d_metric:.4f} | {best.std_r2d_metric:.4f} | {best.std_r2d_metric - baseline.std_r2d_metric:+.4f} |",
            f"| Mean Correlation | {baseline.mean_correlation:.4f} | {best.mean_correlation:.4f} | {best.mean_correlation - baseline.mean_correlation:+.4f} |",
            f"| Computation Time (s) | {baseline.mean_computation_time:.2f} | {best.mean_computation_time:.2f} | {best.mean_computation_time - baseline.mean_computation_time:+.2f} |",
            f"| Num Simulations | {baseline.num_simulations} | {best.num_simulations} | - |",
        ]

        # Add parameter optimization info if available
        if best.num_tunable_params > 0:
            lines.append("")
            lines.append("### Parameter Optimization")
            lines.append("| Metric | Baseline | Best |")
            lines.append("|--------|----------|------|")
            lines.append(f"| Tunable Parameters | {baseline.num_tunable_params} | {best.num_tunable_params} |")
            lines.append(f"| Optimized Parameters | {baseline.num_optimized_params} | {best.num_optimized_params} |")
            lines.append(f"| Optimization Cost | {baseline.param_opt_cost} | {best.param_opt_cost} |")

        return "\n".join(lines)

    def get_metrics_summary(self, metrics: BaseMetrics) -> str:
        """Generate a brief text summary of tidal metrics."""
        if not isinstance(metrics, TidalMetrics):
            return f"Score: {metrics.combined_score:.4f}"

        return (
            f"r_2D: {metrics.mean_r2d_metric:.4f} +/- {metrics.std_r2d_metric:.4f}, "
            f"Corr: {metrics.mean_correlation:.4f}, "
            f"Time: {metrics.mean_computation_time:.1f}s"
        )

    def get_key_metrics_for_llm(self, metrics: BaseMetrics) -> str:
        """Format key tidal reconstruction metrics for LLM context."""
        if not isinstance(metrics, TidalMetrics):
            return f"Combined Score: {metrics.combined_score:.4f}"

        param_info = ""
        if metrics.num_tunable_params > 0:
            param_info = f"""
Parameter Optimization:
- Tunable Parameters: {metrics.num_tunable_params}
- Optimized Parameters: {metrics.num_optimized_params}
- Optimization Cost: {metrics.param_opt_cost} evaluations"""

        return f"""Performance Metrics:
- r_2D Metric (avg[1:6,1:6]): {metrics.mean_r2d_metric:.4f} +/- {metrics.std_r2d_metric:.4f}
- Mean Correlation: {metrics.mean_correlation:.4f}
- Computation Time: {metrics.mean_computation_time:.2f}s
- Evaluated on {metrics.num_simulations} simulations{param_info}"""


class TidalPromptAdapter(PromptAdapter):
    """Prompts for analyzing tidal reconstruction algorithms."""

    def get_algorithm_analysis_prompt(
        self,
        algorithm: AlgorithmInfo,
        is_baseline: bool,
    ) -> str:
        """Generate prompt for tidal reconstruction algorithm analysis."""
        algo_type = "baseline" if is_baseline else "evolved best"

        metrics_info = ""
        if algorithm.metrics and isinstance(algorithm.metrics, TidalMetrics):
            m = algorithm.metrics
            metrics_info = f"""
Performance Metrics:
- r_2D Metric (avg[1:6,1:6]): {m.mean_r2d_metric:.4f} +/- {m.std_r2d_metric:.4f}
- Mean Correlation: {m.mean_correlation:.4f}
- Computation Time: {m.mean_computation_time:.2f}s
- Tunable Parameters: {m.num_tunable_params}
"""

        return f"""Analyze the following {algo_type} tidal field reconstruction algorithm and provide a scientific interpretation.

This is the {algo_type} algorithm from an evolutionary optimization run for 21cm tidal reconstruction in cosmology.

**Physical Context:**
In 21cm intensity mapping, bright astrophysical foregrounds wipe out long-wavelength line-of-sight modes (small k_parallel).
The algorithm must recover large-scale density fluctuations using the tidal modulation of small-scale clustering, which leaves an anisotropic footprint that encodes the large-scale field.

The primary metric is r_2D[1:6, 1:6] - the average Fourier-space correlation coefficient between reconstruction and ground truth in the k_perp-k_parallel plane.

{metrics_info}

ALGORITHM CODE:
```python
{algorithm.code}
```

Provide a concise analysis covering:
1. **Core Approach**: What reconstruction technique does this algorithm use? (e.g., Wiener filtering, iterative methods, machine learning, etc.)
2. **Key Features**: What are the main computational steps and their physical meaning?
3. **Parameter Handling**: How does the algorithm use tunable parameters? Are they physically motivated?
4. **Novel Elements**: Any innovative aspects or departures from standard methods?
5. **Strengths and Limitations**: Based on the code structure, what are potential strengths and weaknesses?

Keep the analysis focused and scientific. Use 3-5 paragraphs total."""

    def get_improvement_analysis_prompt(
        self,
        baseline: AlgorithmInfo,
        best: AlgorithmInfo,
        metrics_comparison: str,
    ) -> str:
        """Generate prompt for comparing baseline vs best tidal algorithms."""
        return f"""Analyze the improvements made by the evolved tidal reconstruction algorithm compared to the baseline.

**Physical Context:**
This algorithm reconstructs large-scale density fluctuations from 21cm intensity maps where foreground subtraction has removed line-of-sight modes. The tidal modulation of small-scale clustering encodes the missing large-scale information.

The key metric is r_2D[1:6,1:6] - higher values (closer to 1.0) indicate better recovery of the density field in Fourier space.

{metrics_comparison}

BASELINE ALGORITHM:
```python
{baseline.code}
```

EVOLVED BEST ALGORITHM:
```python
{best.code}
```

Provide a detailed analysis covering:
1. **Key Modifications**: What are the main changes from baseline to evolved algorithm?
2. **Physical Justification**: Do these changes make physical sense for tidal reconstruction? Why or why not?
3. **Novel Techniques**: Any innovative approaches discovered during evolution?
4. **Parameter Evolution**: How did the parameter space change? Are new parameters physically meaningful?
5. **Trade-offs**: What trade-offs were made (e.g., speed vs accuracy, complexity vs performance)?
6. **Scientific Validity**: Are these improvements likely to generalize, or might they be overfit to specific test cases?

Be critical and scientific in your assessment. If some changes seem questionable, point that out."""

    def get_executive_summary_prompt(
        self,
        data: EvolutionData,
        metrics_comparison: str,
    ) -> str:
        """Generate prompt for executive summary."""
        history = data.history

        if history.best_scores:
            initial_score = history.best_scores[0]
            final_score = max(history.best_scores)
            if initial_score != 0:
                improvement_pct = 100 * (final_score - initial_score) / abs(initial_score)
            else:
                improvement_pct = 0
        else:
            initial_score = final_score = improvement_pct = 0

        # Get parameter info if available
        param_info = ""
        if data.best.metrics and isinstance(data.best.metrics, TidalMetrics):
            m = data.best.metrics
            if m.num_tunable_params > 0:
                param_info = f"""
Parameter Optimization:
- Final algorithm has {m.num_tunable_params} tunable parameters
- {m.num_optimized_params} parameters were optimized
- Total optimization cost: {m.param_opt_cost} evaluations"""

        return f"""Write a brief executive summary for this tidal reconstruction algorithm evolution experiment.

**Physical Context:**
This experiment evolved algorithms for 21cm tidal field reconstruction - recovering large-scale density fluctuations from intensity maps where foreground subtraction has removed line-of-sight modes.

Evolution Statistics:
- Duration: {data.duration_hours:.2f} hours
- Total Generations: {history.total_generations}
- Programs Evaluated: {history.total_programs}
- Successful Programs: {history.successful_programs}
- Initial r_2D Score: {initial_score:.4f}
- Final r_2D Score: {final_score:.4f}
- Improvement: {improvement_pct:.1f}%
{param_info}

{metrics_comparison}

Write a 2-3 paragraph executive summary that:
1. Summarizes the key performance improvements in reconstruction quality (r_2D metric)
2. Highlights the most significant algorithmic innovations discovered
3. Discusses implications for 21cm cosmology and potential scientific applications

Focus on scientific insights and practical implications."""


class TidalTemplateAdapter(ReportTemplateAdapter):
    """Report template for tidal reconstruction reports."""

    def get_report_title(self) -> str:
        return "Tidal Field Reconstruction Evolution Report"

    def get_section_headers(self) -> Dict[str, str]:
        return {
            "metrics": "Reconstruction Quality Metrics",
            "baseline": "Baseline Algorithm Analysis",
            "best": "Best Evolved Algorithm Analysis",
            "improvement": "Evolution Improvements",
            "summary": "Executive Summary",
            "parameters": "Optimized Parameters",
            "config": "Experiment Configuration",
        }

    def format_final_report(
        self,
        data: EvolutionData,
        metrics_table: str,
        baseline_analysis: str,
        best_analysis: str,
        improvement_analysis: str,
        executive_summary: str,
    ) -> str:
        """Assemble the final tidal reconstruction report."""
        history = data.history

        if history.best_scores:
            initial_score = history.best_scores[0]
            final_score = max(history.best_scores)
            if initial_score != 0:
                improvement_pct = 100 * (final_score - initial_score) / abs(initial_score)
            else:
                improvement_pct = 0
        else:
            initial_score = final_score = improvement_pct = 0

        headers = self.get_section_headers()

        # Generate parameter table if best algorithm has tunable parameters
        param_section = ""
        if data.best.metrics and isinstance(data.best.metrics, TidalMetrics):
            metrics = data.best.metrics
            if metrics.tunable_params:
                param_lines = [
                    f"\n## {headers['parameters']}\n",
                    f"The best algorithm uses {metrics.num_tunable_params} tunable parameters optimized via grid search:\n",
                ]

                # Show first 20 parameters in a table, with summary if more
                params_to_show = metrics.tunable_params[:20]
                param_lines.extend([
                    "| Parameter | Default | Optimized | Bounds | Method |",
                    "|-----------|---------|-----------|--------|--------|",
                ])

                for p in params_to_show:
                    frozen_str = f"{p.frozen_value:.4f}" if p.frozen_value is not None else "N/A"
                    param_lines.append(
                        f"| `{p.name}` | {p.default_value:.4f} | {frozen_str} | ({p.bounds[0]:.2f}, {p.bounds[1]:.2f}) | {p.method} |"
                    )

                if len(metrics.tunable_params) > 20:
                    param_lines.append(f"\n*... and {len(metrics.tunable_params) - 20} more parameters*\n")

                # Add parameter descriptions section
                param_lines.append("\n### Parameter Descriptions\n")
                for p in params_to_show:
                    desc = PARAMETER_DESCRIPTIONS.get(p.name, "No description available.")
                    frozen_str = f"{p.frozen_value:.4f}" if p.frozen_value is not None else "N/A"
                    change_direction = ""
                    if p.frozen_value is not None:
                        diff = p.frozen_value - p.default_value
                        if abs(diff) > 1e-6:
                            change_direction = "up" if diff > 0 else "down"
                            pct_change = 100 * diff / abs(p.default_value) if p.default_value != 0 else 0
                            change_direction = f" ({change_direction} {abs(pct_change):.1f}%)"
                    param_lines.append(f"**`{p.name}`** ({p.default_value:.4f} -> {frozen_str}{change_direction})")
                    param_lines.append(f": {desc}\n")

                param_section = "\n".join(param_lines) + "\n"

        # Evolution progress data
        evolution_data_md = ""
        if history.generations and history.best_scores:
            evolution_data_md = """
### Evolution Progress

```
Generation | Best r_2D Score
-----------|----------------
"""
            sample_points = min(15, len(history.generations))
            step = max(1, len(history.generations) // sample_points)
            for i in range(0, len(history.generations), step):
                evolution_data_md += f"{history.generations[i]:10d} | {history.best_scores[i]:.6f}\n"
            evolution_data_md += "```"

        return f"""# {self.get_report_title()}

---

## Table of Contents

1. [{headers['summary']}](#executive-summary)
2. [Evolution Overview](#evolution-overview)
3. [{headers['metrics']}](#reconstruction-quality-metrics)
4. [{headers['baseline']}](#baseline-algorithm-analysis)
5. [{headers['best']}](#best-evolved-algorithm-analysis)
6. [{headers['improvement']}](#evolution-improvements)
7. [{headers['parameters']}](#optimized-parameters)
8. [{headers['config']}](#experiment-configuration)
9. [Appendix: Algorithm Code](#appendix-algorithm-code)

---

## {headers['summary']}

{executive_summary}

---

## Evolution Overview

| Metric | Value |
|--------|-------|
| Duration | {data.duration_hours:.2f} hours |
| Total Generations | {history.total_generations} |
| Programs Evaluated | {history.total_programs} |
| Successful Programs | {history.successful_programs} ({100*history.successful_programs/max(1,history.total_programs):.1f}%) |
| Best Found at Generation | {data.best.generation} |
| Initial r_2D Score | {initial_score:.4f} |
| Final Best r_2D Score | {final_score:.4f} |
| Relative Improvement | {improvement_pct:.1f}% |

{evolution_data_md}

---

## {headers['metrics']}

{metrics_table}
{param_section}
---

## {headers['baseline']}

{baseline_analysis}

---

## {headers['best']}

{best_analysis}

---

## {headers['improvement']}

{improvement_analysis}

---

## {headers['config']}

| Setting | Value |
|---------|-------|
| Number of Islands | {data.config.num_islands} |
| Migration Interval | {data.config.migration_interval} generations |
| Max Generations | {data.config.num_generations} |
| LLM Models Used | {', '.join(data.config.llm_models) if data.config.llm_models else 'N/A'} |

### Task Description

The algorithm evolves to solve 21cm tidal field reconstruction:
- Recover large-scale density fluctuations from 21cm intensity maps
- Foreground subtraction removes line-of-sight modes (small k_parallel)
- Use tidal modulation of small-scale clustering to infer missing information
- Primary metric: r_2D[1:6,1:6] correlation coefficient in Fourier space

"""


class TidalReconstructionAdapter(ScenarioAdapter):
    """
    Complete adapter for tidal reconstruction evolution.

    Usage:
        from madevolve_cosmo.analyzer.adapters import TidalReconstructionAdapter
        from madevolve.analyzer import DataExtractor, ReportGenerator, register_adapter

        register_adapter('tidal', TidalReconstructionAdapter)
        adapter = TidalReconstructionAdapter()
        extractor = DataExtractor(adapter)
        data = extractor.extract_evolution_data("/path/to/results")

        generator = ReportGenerator(adapter)
        report = generator.generate_full_report(data)
    """

    def __init__(self):
        self._metrics_adapter = TidalMetricsAdapter()
        self._prompt_adapter = TidalPromptAdapter()
        self._template_adapter = TidalTemplateAdapter()

    @property
    def metrics_adapter(self) -> MetricsAdapter:
        return self._metrics_adapter

    @property
    def prompt_adapter(self) -> PromptAdapter:
        return self._prompt_adapter

    @property
    def template_adapter(self) -> ReportTemplateAdapter:
        return self._template_adapter

    def extract_code_block(self, code: str) -> str:
        """
        Extract the EVOLVE-BLOCK for tidal reconstruction algorithms.
        """
        if "# EVOLVE-BLOCK-START" in code and "# EVOLVE-BLOCK-END" in code:
            start_idx = code.find("# EVOLVE-BLOCK-START")
            end_idx = code.find("# EVOLVE-BLOCK-END") + len("# EVOLVE-BLOCK-END")
            return code[start_idx:end_idx]
        return code
