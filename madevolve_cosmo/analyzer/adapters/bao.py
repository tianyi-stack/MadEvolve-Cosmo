"""
BAO Reconstruction Adapter for MadEvolve-Cosmo.

This adapter provides domain-specific handling for:
- BAO (Baryon Acoustic Oscillation) reconstruction metrics
- Cosmology algorithm analysis prompts
- Scientific report templates

Supports both standard BAO reconstruction and JAX autodiff variants.
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
    value: float
    bounds: tuple
    method: str
    description: str = ""


# Parameter descriptions for BAO reconstruction algorithms
PARAMETER_DESCRIPTIONS = {
    # Galaxy bias and basic smoothing
    "bias": "Galaxy bias parameter relating observed galaxy density to underlying matter density. Higher values indicate stronger bias.",
    "init_R_perp": "Initial smoothing scale (Mpc/h) perpendicular to the line of sight. Controls transverse smoothing of the density field.",
    "init_R_para": "Initial smoothing scale (Mpc/h) parallel to the line of sight. Controls radial smoothing, accounting for redshift-space distortions.",
    "smoothing_scale_R_s": "Overall Gaussian smoothing scale (Mpc/h) for the density field reconstruction.",

    # Advection and invariant coefficients
    "gamma_adv": "Advection correction coefficient. Controls the strength of the advective term s*nabla*delta in the reconstruction.",
    "gamma_div": "Divergence coefficient for the first invariant I1 = nabla*s of the displacement field.",
    "gamma_I2": "Second invariant coefficient. Controls contribution of I2 (related to shear) to the source term.",
    "gamma_I3": "Third invariant coefficient. Controls contribution of I3 (determinant of deformation tensor) to the source term.",

    # Mixing and regularization
    "mix_coeff": "Mixing coefficient for blending different displacement field channels (primary Zeldovich + corrections).",
    "hk_strength": "High-k regularization strength. Controls suppression of small-scale (high wavenumber) modes to prevent noise amplification.",
    "stab_scale": "Stabilization scale for arcsinh transformation, preventing numerical instabilities in highly nonlinear regions.",

    # Correction parameters
    "c0_threshold": "Threshold parameter for adaptive correction. Determines where correction kicks in based on local density.",
    "c0_slope": "Slope parameter controlling the rate of correction transition around the threshold.",
    "c_s2": "Coefficient for the tidal shear invariant s2 contribution to the reconstruction.",

    # Dewarp and iteration parameters
    "dewarp_ridge_pow": "Power-law exponent for dewarp ridge regularization. Controls scale-dependent damping in iterative steps.",
    "dewarp_strength": "Overall strength of the dewarping correction applied to remove residual distortions.",
    "n_iter": "Number of iterations for iterative reconstruction schemes.",

    # RSD correction parameters
    "f_growth": "Linear growth rate f ~ Omega_m^0.55 for redshift-space distortion correction.",
    "beta_rsd": "RSD parameter beta = f/b relating peculiar velocities to density field.",
}


@dataclass
class BAOMetrics(BaseMetrics):
    """BAO reconstruction-specific metrics."""
    mean_avg_r_bao: float = 0.0
    std_avg_r_bao: float = 0.0
    mean_avg_r_large_scale: float = 0.0
    mean_correlation: float = 0.0
    num_simulations: int = 0
    max_degradation: float = 0.0
    max_degradation_k: float = 0.0
    penalty_applied: float = 0.0
    mean_computation_time: float = 0.0
    r_k_large_scale: List[float] = field(default_factory=list)
    baseline_r_k_large_scale: List[float] = field(default_factory=list)
    degradations: List[float] = field(default_factory=list)

    # Tunable parameters (parsed from code)
    tunable_parameters: List[TunableParameter] = field(default_factory=list)


def parse_tunable_parameters(code: str) -> List[TunableParameter]:
    """
    Parse TUNABLE parameter declarations from code.

    Format: # TUNABLE: param_name = default_value, bounds=(min, max), method=autodiff
    """
    parameters = []
    pattern = r'#\s*TUNABLE:\s*(\w+)\s*=\s*([\d.e+-]+)\s*,\s*bounds\s*=\s*\(([\d.e+-]+)\s*,\s*([\d.e+-]+)\)\s*,\s*method\s*=\s*(\w+)'

    for match in re.finditer(pattern, code):
        param_name = match.group(1)
        param = TunableParameter(
            name=param_name,
            value=float(match.group(2)),
            bounds=(float(match.group(3)), float(match.group(4))),
            method=match.group(5),
            description=PARAMETER_DESCRIPTIONS.get(param_name, "No description available.")
        )
        parameters.append(param)

    return parameters


class BAOMetricsAdapter(MetricsAdapter):
    """Adapter for parsing and displaying BAO reconstruction metrics."""

    # Standard k values for BAO analysis
    K_VALUES = [0.0219, 0.0456, 0.0694, 0.0931, 0.1169, 0.1406, 0.1644, 0.1881]

    @property
    def scenario_name(self) -> str:
        return "bao"

    @property
    def scenario_description(self) -> str:
        return "BAO (Baryon Acoustic Oscillation) Reconstruction"

    def parse_metrics(self, metrics_data: Dict[str, Any]) -> BAOMetrics:
        """Parse BAO metrics from metrics.json."""
        public = metrics_data.get('public', {})
        private = metrics_data.get('private', {})

        r_k_large_scale = private.get('all_r_k_large_scale', [[]])[0]
        if not r_k_large_scale:
            r_k_large_scale = []

        return BAOMetrics(
            combined_score=metrics_data.get('combined_score', 0.0),
            text_feedback=metrics_data.get('text_feedback', ''),
            execution_time_mean=metrics_data.get('execution_time_mean', 0.0),
            execution_time_std=metrics_data.get('execution_time_std', 0.0),
            raw_public=public,
            raw_private=private,
            # BAO-specific
            mean_avg_r_bao=public.get('mean_avg_r_bao', 0.0),
            std_avg_r_bao=public.get('std_avg_r_bao', 0.0),
            mean_avg_r_large_scale=public.get('mean_avg_r_large_scale', 0.0),
            mean_correlation=public.get('mean_correlation', 0.0),
            num_simulations=public.get('num_simulations', 0),
            max_degradation=public.get('max_degradation', 0.0),
            max_degradation_k=public.get('max_degradation_k', 0.0),
            penalty_applied=public.get('penalty_applied', 0.0),
            mean_computation_time=private.get('mean_computation_time', 0.0),
            r_k_large_scale=r_k_large_scale,
            baseline_r_k_large_scale=private.get('baseline_r_k_large_scale', []),
            degradations=private.get('degradations', []),
        )

    def get_metrics_comparison_table(
        self,
        baseline: BaseMetrics,
        best: BaseMetrics
    ) -> str:
        """Generate BAO metrics comparison table."""
        if not isinstance(baseline, BAOMetrics):
            baseline = BAOMetrics(combined_score=baseline.combined_score)
        if not isinstance(best, BAOMetrics):
            best = BAOMetrics(combined_score=best.combined_score)

        # Calculate improvement percentage
        if baseline.mean_avg_r_bao != 0:
            improvement_pct = 100 * (best.mean_avg_r_bao - baseline.mean_avg_r_bao) / baseline.mean_avg_r_bao
        else:
            improvement_pct = 0

        lines = [
            "| Metric | Baseline | Best | Improvement |",
            "|--------|----------|------|-------------|",
            f"| **Combined Score** | {baseline.combined_score:.4f} | {best.combined_score:.4f} | {best.combined_score - baseline.combined_score:+.4f} |",
            f"| Mean r(k) BAO Range [0.01, 0.5] | {baseline.mean_avg_r_bao:.4f} | {best.mean_avg_r_bao:.4f} | {best.mean_avg_r_bao - baseline.mean_avg_r_bao:+.4f} ({improvement_pct:+.1f}%) |",
            f"| Mean r(k) Large Scale [0.01, 0.2] | {baseline.mean_avg_r_large_scale:.4f} | {best.mean_avg_r_large_scale:.4f} | {best.mean_avg_r_large_scale - baseline.mean_avg_r_large_scale:+.4f} |",
            f"| Mean Correlation | {baseline.mean_correlation:.4f} | {best.mean_correlation:.4f} | {best.mean_correlation - baseline.mean_correlation:+.4f} |",
            f"| Max Degradation | {baseline.max_degradation:.4f} | {best.max_degradation:.4f} | {best.max_degradation - baseline.max_degradation:+.4f} |",
            f"| Penalty Applied | {baseline.penalty_applied:.4f} | {best.penalty_applied:.4f} | {best.penalty_applied - baseline.penalty_applied:+.4f} |",
            f"| Computation Time (s) | {baseline.mean_computation_time:.2f} | {best.mean_computation_time:.2f} | {best.mean_computation_time - baseline.mean_computation_time:+.2f} |",
        ]

        # Add per-k r(k) values if available
        if baseline.r_k_large_scale and best.r_k_large_scale:
            lines.append("")
            lines.append("### Per-k Large-Scale r(k) Values")
            lines.append("| k (h/Mpc) | Baseline r(k) | Best r(k) | Reference | Improvement |")
            lines.append("|-----------|---------------|-----------|-----------|-------------|")
            for i, k in enumerate(self.K_VALUES[:len(baseline.r_k_large_scale)]):
                base_val = baseline.r_k_large_scale[i] if i < len(baseline.r_k_large_scale) else 0
                best_val = best.r_k_large_scale[i] if i < len(best.r_k_large_scale) else 0
                ref_val = baseline.baseline_r_k_large_scale[i] if i < len(baseline.baseline_r_k_large_scale) else 0
                improvement = best_val - base_val
                lines.append(f"| {k:.4f} | {base_val:.4f} | {best_val:.4f} | {ref_val:.4f} | {improvement:+.4f} |")

        return "\n".join(lines)

    def get_metrics_summary(self, metrics: BaseMetrics) -> str:
        """Generate a brief text summary of BAO metrics."""
        if not isinstance(metrics, BAOMetrics):
            return f"Score: {metrics.combined_score:.4f}"

        return (
            f"Score: {metrics.combined_score:.4f}, "
            f"r(k) BAO: {metrics.mean_avg_r_bao:.4f}, "
            f"r(k) Large: {metrics.mean_avg_r_large_scale:.4f}"
        )

    def get_key_metrics_for_llm(self, metrics: BaseMetrics) -> str:
        """Format key BAO metrics for LLM context."""
        if not isinstance(metrics, BAOMetrics):
            return f"Combined Score: {metrics.combined_score:.4f}"

        return f"""Performance Metrics:
- Combined Score: {metrics.combined_score:.4f}
- Mean r(k) BAO Range [0.01, 0.5]: {metrics.mean_avg_r_bao:.4f}
- Mean r(k) Large Scale [0.01, 0.2]: {metrics.mean_avg_r_large_scale:.4f}
- Mean Correlation: {metrics.mean_correlation:.4f}
- Max Degradation: {metrics.max_degradation:.4f}
- Penalty Applied: {metrics.penalty_applied:.4f}
- Computation Time: {metrics.mean_computation_time:.2f}s"""


class BAOPromptAdapter(PromptAdapter):
    """Prompts for analyzing BAO reconstruction algorithms."""

    def get_algorithm_analysis_prompt(
        self,
        algorithm: AlgorithmInfo,
        is_baseline: bool,
    ) -> str:
        """Generate prompt for BAO algorithm analysis."""
        algo_type = "baseline" if is_baseline else "evolved best"

        metrics_info = ""
        if algorithm.metrics and isinstance(algorithm.metrics, BAOMetrics):
            m = algorithm.metrics
            metrics_info = f"""
Performance Metrics:
- Combined Score: {m.combined_score:.4f}
- Mean r(k) BAO Range [0.01, 0.5]: {m.mean_avg_r_bao:.4f}
- Mean r(k) Large Scale [0.01, 0.2]: {m.mean_avg_r_large_scale:.4f}
- Mean Correlation: {m.mean_correlation:.4f}
- Max Degradation: {m.max_degradation:.4f}
- Computation Time: {m.mean_computation_time:.2f}s
"""

        return f"""Analyze the following {algo_type} BAO reconstruction algorithm and provide a scientific interpretation.

This is the {algo_type} algorithm from an evolutionary optimization run for 3D BAO (Baryon Acoustic Oscillation) reconstruction in cosmology.

**Physical Context:**
BAO reconstruction aims to recover the initial density field from the evolved (redshift-space) density field by reversing the effects of bulk flows and nonlinear structure formation. The key metric r(k) measures the cross-correlation between reconstructed and true initial density fields in Fourier space.

The algorithm must:
1. Maximize r(k) in the BAO range (k ~ 0.01-0.5 h/Mpc)
2. Preserve large-scale structure (k < 0.2 h/Mpc) without degradation
3. Be computationally efficient

{metrics_info}

ALGORITHM CODE:
```python
{algorithm.code}
```

Provide a concise analysis covering:
1. **Core Approach**: What reconstruction technique does this algorithm use? (e.g., Zeldovich approximation, iterative methods, etc.)
2. **Key Features**: What are the main computational steps and their physical meaning?
3. **Novel Elements**: Any innovative aspects or departures from standard methods?
4. **Strengths and Limitations**: Based on the code structure, what are potential strengths and weaknesses?

Keep the analysis focused and scientific. Use 3-5 paragraphs total."""

    def get_improvement_analysis_prompt(
        self,
        baseline: AlgorithmInfo,
        best: AlgorithmInfo,
        metrics_comparison: str,
    ) -> str:
        """Generate prompt for comparing baseline vs best BAO algorithms."""
        return f"""Analyze the improvements made by the evolved BAO reconstruction algorithm compared to the baseline.

**Physical Context:**
BAO reconstruction recovers the initial density field from evolved observations. The key metric r(k) measures reconstruction quality - higher values (closer to 1.0) indicate better recovery of the primordial density fluctuations.

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
2. **Physical Justification**: Do these changes make physical sense for BAO reconstruction? Why or why not?
3. **Novel Techniques**: Any innovative approaches discovered during evolution?
4. **Trade-offs**: What trade-offs were made (e.g., speed vs accuracy, complexity vs performance)?
5. **Scientific Validity**: Are these improvements likely to generalize, or might they be overfit to specific test cases?

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

        return f"""Write a brief executive summary for this BAO reconstruction algorithm evolution experiment.

**Physical Context:**
This experiment evolved algorithms for BAO (Baryon Acoustic Oscillation) reconstruction - a key technique in precision cosmology for measuring the expansion history of the universe.

Evolution Statistics:
- Duration: {data.duration_hours:.2f} hours
- Total Generations: {history.total_generations}
- Programs Evaluated: {history.total_programs}
- Successful Programs: {history.successful_programs}
- Initial Score: {initial_score:.4f}
- Final Score: {final_score:.4f}
- Improvement: {improvement_pct:.1f}%

{metrics_comparison}

Write a 2-3 paragraph executive summary that:
1. Summarizes the key performance improvements in reconstruction quality
2. Highlights the most significant algorithmic innovations
3. Discusses implications for cosmological analysis

Focus on scientific insights and practical implications."""


class BAOTemplateAdapter(ReportTemplateAdapter):
    """Report template for BAO reconstruction reports."""

    def get_report_title(self) -> str:
        return "BAO Reconstruction Evolution Report"

    def get_section_headers(self) -> Dict[str, str]:
        return {
            "metrics": "Reconstruction Quality Metrics",
            "baseline": "Baseline Algorithm Analysis",
            "best": "Best Evolved Algorithm Analysis",
            "improvement": "Evolution Improvements",
            "summary": "Executive Summary",
            "parameters": "Tunable Parameters",
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
        """Assemble the final BAO reconstruction report."""
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
        if data.best.metrics and isinstance(data.best.metrics, BAOMetrics):
            params = parse_tunable_parameters(data.best.code)
            if params:
                param_lines = [
                    f"\n## {headers['parameters']}\n",
                    "The best algorithm uses the following tunable parameters, optimized via automatic differentiation:\n",
                    "| Parameter | Value | Bounds | Method |",
                    "|-----------|-------|--------|--------|",
                ]
                for p in params:
                    param_lines.append(
                        f"| `{p.name}` | {p.value:.6f} | ({p.bounds[0]:.2f}, {p.bounds[1]:.2f}) | {p.method} |"
                    )
                param_lines.append("")
                param_lines.append("### Parameter Descriptions\n")
                for p in params:
                    param_lines.append(f"- **`{p.name}`**: {p.description}")
                param_section = "\n".join(param_lines) + "\n"

        # Evolution progress data
        evolution_data_md = ""
        if history.generations and history.best_scores:
            evolution_data_md = """
### Evolution Progress

```
Generation | Best Score
-----------|------------
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
7. [{headers['config']}](#experiment-configuration)
8. [Appendix: Algorithm Code](#appendix-algorithm-code)

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
| Initial Score | {initial_score:.4f} |
| Final Best Score | {final_score:.4f} |
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

The algorithm evolves to solve BAO (Baryon Acoustic Oscillation) reconstruction:
- Recover the initial density field from evolved observations
- Maximize cross-correlation r(k) in BAO range (k ~ 0.01-0.5 h/Mpc)
- Preserve large-scale structure without degradation
- Key constraint: r(k) at large scales must not degrade from baseline

"""


class BAOAdapter(ScenarioAdapter):
    """
    Complete adapter for BAO reconstruction evolution.

    Usage:
        from madevolve_cosmo.analyzer.adapters import BAOAdapter
        from madevolve.analyzer import DataExtractor, ReportGenerator, register_adapter

        register_adapter('bao', BAOAdapter)
        adapter = BAOAdapter()
        extractor = DataExtractor(adapter)
        data = extractor.extract_evolution_data("/path/to/results")

        generator = ReportGenerator(adapter)
        report = generator.generate_full_report(data)
    """

    def __init__(self):
        self._metrics_adapter = BAOMetricsAdapter()
        self._prompt_adapter = BAOPromptAdapter()
        self._template_adapter = BAOTemplateAdapter()

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
        Extract the EVOLVE-BLOCK for BAO algorithms.
        """
        if "# EVOLVE-BLOCK-START" in code and "# EVOLVE-BLOCK-END" in code:
            start_idx = code.find("# EVOLVE-BLOCK-START")
            end_idx = code.find("# EVOLVE-BLOCK-END") + len("# EVOLVE-BLOCK-END")
            return code[start_idx:end_idx]
        return code
