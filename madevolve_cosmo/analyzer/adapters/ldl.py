"""
LDL (Lagrangian Deep Learning) Adapter for MadEvolve-Cosmo.

This adapter provides domain-specific handling for:
- LDL model evolution for cosmological simulations
- tSZ (thermal Sunyaev-Zeldovich) effect prediction
- CAMELS simulation data analysis

LDL uses learned displacement fields applied to dark matter particles
to predict baryonic properties like electron pressure (tSZ signal).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

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
class LDLMetrics(BaseMetrics):
    """LDL-specific metrics for tSZ prediction."""
    # Primary metric
    mean_r_k_all_scales: float = 0.0
    std_r_k_all_scales: float = 0.0

    # Cross-correlation metrics
    cross_correlation_large_scale: float = 0.0
    std_cross_correlation: float = 0.0
    transfer_function_accuracy: float = 0.0
    std_transfer_function: float = 0.0

    # Training/validation metrics
    train_loss: float = 0.0
    mean_val_loss: float = 0.0
    std_val_loss: float = 0.0
    num_validation_sims: int = 0

    # Per-simulation validation losses
    validation_sims: List[str] = field(default_factory=list)
    val_losses_per_sim: Dict[str, float] = field(default_factory=dict)

    # Reliability metrics
    success_rate: float = 1.0
    failed_runs: int = 0
    failure_penalty: float = 0.0
    mean_training_time: float = 0.0


class LDLMetricsAdapter(MetricsAdapter):
    """Adapter for parsing and displaying LDL metrics."""

    @property
    def scenario_name(self) -> str:
        return "ldl"

    @property
    def scenario_description(self) -> str:
        return "Lagrangian Deep Learning (tSZ Prediction)"

    def parse_metrics(self, metrics_data: Dict[str, Any]) -> LDLMetrics:
        """Parse LDL metrics from metrics.json."""
        # Extract per-simulation validation losses
        val_losses = {}
        validation_sims = metrics_data.get('validation_sims', [])
        for sim in validation_sims:
            key = f'val_loss_{sim}'
            if key in metrics_data:
                val_losses[sim] = metrics_data[key]

        return LDLMetrics(
            combined_score=metrics_data.get('combined_score', 0.0),
            text_feedback=metrics_data.get('text_feedback', ''),
            execution_time_mean=metrics_data.get('execution_time_mean', 0.0),
            execution_time_std=metrics_data.get('execution_time_std', 0.0),
            raw_public=metrics_data,
            raw_private={},
            # Primary metric
            mean_r_k_all_scales=metrics_data.get('mean_r_k_all_scales', 0.0),
            std_r_k_all_scales=metrics_data.get('std_r_k_all_scales', 0.0),
            # Cross-correlation
            cross_correlation_large_scale=metrics_data.get('cross_correlation_large_scale', 0.0),
            std_cross_correlation=metrics_data.get('std_cross_correlation', 0.0),
            transfer_function_accuracy=metrics_data.get('transfer_function_accuracy', 0.0),
            std_transfer_function=metrics_data.get('std_transfer_function', 0.0),
            # Training/validation
            train_loss=metrics_data.get('train_loss', 0.0),
            mean_val_loss=metrics_data.get('mean_val_loss', metrics_data.get('mean_validation_loss', 0.0)),
            std_val_loss=metrics_data.get('std_val_loss', metrics_data.get('std_validation_loss', 0.0)),
            num_validation_sims=metrics_data.get('num_validation_sims', 0),
            validation_sims=validation_sims,
            val_losses_per_sim=val_losses,
            # Reliability
            success_rate=metrics_data.get('success_rate', 1.0),
            failed_runs=metrics_data.get('failed_runs', 0),
            failure_penalty=metrics_data.get('failure_penalty', 0.0),
            mean_training_time=metrics_data.get('mean_training_time', 0.0),
        )

    def get_metrics_comparison_table(
        self,
        baseline: BaseMetrics,
        best: BaseMetrics
    ) -> str:
        """Generate LDL metrics comparison table."""
        if not isinstance(baseline, LDLMetrics):
            baseline = LDLMetrics(combined_score=baseline.combined_score)
        if not isinstance(best, LDLMetrics):
            best = LDLMetrics(combined_score=best.combined_score)

        # Calculate improvement percentage
        if baseline.mean_r_k_all_scales != 0:
            improvement_pct = 100 * (best.mean_r_k_all_scales - baseline.mean_r_k_all_scales) / baseline.mean_r_k_all_scales
        else:
            improvement_pct = 0

        lines = [
            "| Metric | Baseline | Best | Improvement |",
            "|--------|----------|------|-------------|",
            f"| **Mean r(k) All Scales** | {baseline.mean_r_k_all_scales:.4f} | {best.mean_r_k_all_scales:.4f} | {best.mean_r_k_all_scales - baseline.mean_r_k_all_scales:+.4f} ({improvement_pct:+.1f}%) |",
            f"| Std r(k) | {baseline.std_r_k_all_scales:.4f} | {best.std_r_k_all_scales:.4f} | {best.std_r_k_all_scales - baseline.std_r_k_all_scales:+.4f} |",
            f"| Large-Scale Cross-Corr | {baseline.cross_correlation_large_scale:.4f} | {best.cross_correlation_large_scale:.4f} | {best.cross_correlation_large_scale - baseline.cross_correlation_large_scale:+.4f} |",
            f"| Transfer Function Acc | {baseline.transfer_function_accuracy:.4f} | {best.transfer_function_accuracy:.4f} | {best.transfer_function_accuracy - baseline.transfer_function_accuracy:+.4f} |",
        ]

        # Add training/validation metrics
        lines.append("")
        lines.append("### Training & Validation")
        lines.append("| Metric | Baseline | Best | Change |")
        lines.append("|--------|----------|------|--------|")
        lines.append(f"| Training Loss | {baseline.train_loss:.4f} | {best.train_loss:.4f} | {best.train_loss - baseline.train_loss:+.4f} |")
        lines.append(f"| Mean Validation Loss | {baseline.mean_val_loss:.4f} | {best.mean_val_loss:.4f} | {best.mean_val_loss - baseline.mean_val_loss:+.4f} |")
        lines.append(f"| Success Rate | {baseline.success_rate:.2%} | {best.success_rate:.2%} | - |")

        # Add per-simulation validation losses if available
        if best.val_losses_per_sim:
            lines.append("")
            lines.append("### Per-Simulation Validation Loss")
            lines.append("| Simulation | Baseline | Best | Change |")
            lines.append("|------------|----------|------|--------|")
            for sim in best.validation_sims:
                base_val = baseline.val_losses_per_sim.get(sim, 0)
                best_val = best.val_losses_per_sim.get(sim, 0)
                lines.append(f"| {sim} | {base_val:.4f} | {best_val:.4f} | {best_val - base_val:+.4f} |")

        return "\n".join(lines)

    def get_metrics_summary(self, metrics: BaseMetrics) -> str:
        """Generate a brief text summary of LDL metrics."""
        if not isinstance(metrics, LDLMetrics):
            return f"Score: {metrics.combined_score:.4f}"

        return (
            f"r(k): {metrics.mean_r_k_all_scales:.4f} +/- {metrics.std_r_k_all_scales:.4f}, "
            f"Val Loss: {metrics.mean_val_loss:.4f}"
        )

    def get_key_metrics_for_llm(self, metrics: BaseMetrics) -> str:
        """Format key LDL metrics for LLM context."""
        if not isinstance(metrics, LDLMetrics):
            return f"Combined Score: {metrics.combined_score:.4f}"

        return f"""Performance Metrics:
- Mean r(k) All Scales: {metrics.mean_r_k_all_scales:.4f} +/- {metrics.std_r_k_all_scales:.4f}
- Large-Scale Cross-Correlation: {metrics.cross_correlation_large_scale:.4f}
- Transfer Function Accuracy: {metrics.transfer_function_accuracy:.4f}
- Training Loss: {metrics.train_loss:.4f}
- Mean Validation Loss: {metrics.mean_val_loss:.4f} +/- {metrics.std_val_loss:.4f}
- Success Rate: {metrics.success_rate:.2%}"""


class LDLPromptAdapter(PromptAdapter):
    """Prompts for analyzing LDL model architectures."""

    def get_algorithm_analysis_prompt(
        self,
        algorithm: AlgorithmInfo,
        is_baseline: bool,
    ) -> str:
        """Generate prompt for LDL model analysis."""
        algo_type = "baseline" if is_baseline else "evolved best"

        metrics_info = ""
        if algorithm.metrics and isinstance(algorithm.metrics, LDLMetrics):
            m = algorithm.metrics
            metrics_info = f"""
Performance Metrics:
- Mean r(k) All Scales: {m.mean_r_k_all_scales:.4f} +/- {m.std_r_k_all_scales:.4f}
- Large-Scale Cross-Correlation: {m.cross_correlation_large_scale:.4f}
- Transfer Function Accuracy: {m.transfer_function_accuracy:.4f}
- Training Loss: {m.train_loss:.4f}
- Mean Validation Loss: {m.mean_val_loss:.4f}
"""

        return f"""Analyze the following {algo_type} Lagrangian Deep Learning (LDL) model and provide a scientific interpretation.

This is the {algo_type} model from an evolutionary optimization run for predicting the thermal Sunyaev-Zeldovich (tSZ) effect from dark matter particle positions using CAMELS simulation data.

**Physical Context:**
- LDL applies learned displacement fields to dark matter particles to predict baryonic properties
- The tSZ effect (electron pressure n_e * T) traces hot gas in galaxy clusters
- The model must capture gas physics: cooling, heating, shocks, AGN/stellar feedback
- Target: Maximize cross-correlation r(k) between predicted and true electron pressure fields

{metrics_info}

MODEL CODE:
```python
{algorithm.code}
```

Provide a concise analysis covering:
1. **Architecture**: What is the model structure? (displacement layers, operators, etc.)
2. **Key Features**: What physical or mathematical operations are used?
3. **Innovation**: Any novel aspects compared to standard LDL approaches?
4. **Strengths/Limitations**: Based on the code, what are potential strengths and weaknesses?

Keep the analysis focused and scientific. Use 3-5 paragraphs total."""

    def get_improvement_analysis_prompt(
        self,
        baseline: AlgorithmInfo,
        best: AlgorithmInfo,
        metrics_comparison: str,
    ) -> str:
        """Generate prompt for comparing baseline vs best LDL models."""
        return f"""Analyze the improvements made by the evolved LDL model compared to the baseline.

**Physical Context:**
Lagrangian Deep Learning predicts baryonic properties (tSZ signal) from dark matter positions. The model learns displacement fields that map DM particles to gas distributions, capturing complex baryonic physics.

{metrics_comparison}

BASELINE MODEL:
```python
{baseline.code}
```

EVOLVED BEST MODEL:
```python
{best.code}
```

Provide a detailed analysis covering:
1. **Architectural Changes**: What structural modifications were made?
2. **Physical Justification**: Do changes make physical sense for tSZ prediction?
3. **Novel Techniques**: Any innovative approaches discovered during evolution?
4. **Trade-offs**: What trade-offs were made (complexity vs performance, training stability)?
5. **Generalization**: Are improvements likely to generalize to other simulations?

Be critical and scientific in your assessment."""

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

        return f"""Write a brief executive summary for this LDL model evolution experiment.

**Physical Context:**
This experiment evolved Lagrangian Deep Learning models for predicting the thermal Sunyaev-Zeldovich (tSZ) effect from dark matter simulations (CAMELS data). The tSZ signal traces hot gas in galaxy clusters and is critical for CMB experiments.

Evolution Statistics:
- Duration: {data.duration_hours:.2f} hours
- Total Generations: {history.total_generations}
- Programs Evaluated: {history.total_programs}
- Successful Programs: {history.successful_programs}
- Initial r(k): {initial_score:.4f}
- Final r(k): {final_score:.4f}
- Improvement: {improvement_pct:.1f}%

{metrics_comparison}

Write a 2-3 paragraph executive summary that:
1. Summarizes the key performance improvements in tSZ prediction quality
2. Highlights the most significant architectural innovations discovered
3. Discusses implications for cosmological analysis and CMB science

Focus on scientific insights and practical implications."""


class LDLTemplateAdapter(ReportTemplateAdapter):
    """Report template for LDL model reports."""

    def get_report_title(self) -> str:
        return "LDL tSZ Prediction Evolution Report"

    def get_section_headers(self) -> Dict[str, str]:
        return {
            "metrics": "Model Performance Metrics",
            "baseline": "Baseline Model Analysis",
            "best": "Best Evolved Model Analysis",
            "improvement": "Evolution Improvements",
            "summary": "Executive Summary",
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
        """Assemble the final LDL report."""
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

        # Evolution progress data
        evolution_data_md = ""
        if history.generations and history.best_scores:
            evolution_data_md = """
### Evolution Progress

```
Generation | Best r(k) Score
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
3. [{headers['metrics']}](#model-performance-metrics)
4. [{headers['baseline']}](#baseline-model-analysis)
5. [{headers['best']}](#best-evolved-model-analysis)
6. [{headers['improvement']}](#evolution-improvements)
7. [{headers['config']}](#experiment-configuration)
8. [Appendix: Model Code](#appendix-model-code)

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
| Initial r(k) Score | {initial_score:.4f} |
| Final Best r(k) Score | {final_score:.4f} |
| Relative Improvement | {improvement_pct:.1f}% |

{evolution_data_md}

---

## {headers['metrics']}

{metrics_table}

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

The model evolves to predict the thermal Sunyaev-Zeldovich (tSZ) effect:
- Input: Dark matter particle positions from CAMELS simulations
- Output: Electron pressure field (n_e * T)
- Primary metric: Cross-correlation r(k) averaged over all scales
- Physics: Gas cooling/heating, shock heating, AGN/stellar feedback

"""


class LDLAdapter(ScenarioAdapter):
    """
    Complete adapter for LDL model evolution.

    Usage:
        from madevolve_cosmo.analyzer.adapters import LDLAdapter
        from madevolve.analyzer import DataExtractor, ReportGenerator, register_adapter

        register_adapter('ldl', LDLAdapter)
        adapter = LDLAdapter()
        extractor = DataExtractor(adapter)
        data = extractor.extract_evolution_data("/path/to/results")

        generator = ReportGenerator(adapter)
        report = generator.generate_full_report(data)
    """

    def __init__(self):
        self._metrics_adapter = LDLMetricsAdapter()
        self._prompt_adapter = LDLPromptAdapter()
        self._template_adapter = LDLTemplateAdapter()

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
        Extract the EVOLVE-BLOCK for LDL models.
        """
        if "# EVOLVE-BLOCK-START" in code and "# EVOLVE-BLOCK-END" in code:
            start_idx = code.find("# EVOLVE-BLOCK-START")
            end_idx = code.find("# EVOLVE-BLOCK-END") + len("# EVOLVE-BLOCK-END")
            return code[start_idx:end_idx]
        return code
