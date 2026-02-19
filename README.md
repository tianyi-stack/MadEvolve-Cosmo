# MadEvolve-Cosmo

Cosmology experiment suite for the [MadEvolve](https://github.com/madevolve/MadEvolve) LLM-driven code evolution framework.

## What is this?

MadEvolve-Cosmo contains four cosmological algorithm discovery tasks. Each task defines:

- An **initial program** (the baseline algorithm to evolve from)
- An **evaluator** (scores candidate algorithms on simulation data)
- A **config** (tells MadEvolve how to run the evolution)

MadEvolve's LLM agents iteratively mutate the initial program, the evaluator scores each mutation, and the population evolves toward better algorithms.

## Tasks

| Task | Directory | Metric | Method |
|------|-----------|--------|--------|
| **BAO Reconstruction** | `bao_reconstruction_auto_diff` | r(k) in [0.01, 0.5] h/Mpc | JAX autodiff |
| **Iterative BAO Reconstruction** | `iterative_reconstruction` | r(k) in [0.01, 0.5] h/Mpc | Particle-based + JAX |
| **Tidal Reconstruction** | `tidal_reconstruction_auto_diff` | avg r_2D[1:6, 1:6] | JAX autodiff |
| **LDL CAMELS tSZ** | `ldl_camels` | r(k) all scales | vmad + MPI |

### BAO Reconstruction (autodiff)

Recover the initial density field from a nonlinear dark matter density field using Fourier-space filtering. Parameters are optimized via JAX gradient descent. Grid: 256^3, BoxSize=1000 Mpc/h.

### Iterative BAO Reconstruction

Particle-based iterative reconstruction (RecCal_multi). Repeatedly displaces particles backward using Zeldovich approximation with decreasing smoothing scales. Uses Cython-compiled CIC painting. Grid: 256^3.

### Tidal Reconstruction (autodiff)

Recover large-scale line-of-sight density modes lost to 21cm foreground contamination by exploiting anisotropic tidal modulation of small-scale clustering. JAX-differentiable. Grid: 512^3.

### LDL CAMELS tSZ

Lagrangian Deep Learning model predicting the thermal Sunyaev-Zeldovich signal (electron pressure n_e x T) from dark matter particle positions. Uses vmad (differentiable computation graph) + MPI parallelism. Two-stage evaluation: train on CV_0, validate on CV_1-CV_4.

## Quick Start

### Prerequisites

- Python >= 3.9
- [MadEvolve](https://github.com/madevolve/MadEvolve) installed (`pip install -e /path/to/MadEvolve`)
- API keys for at least one LLM provider (OpenAI, Google Gemini, etc.)

### Setup

```bash
git clone <repo-url>
cd MadEvolve-Cosmo

# Create .env with your API keys (see .env.example or copy from below)
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GEMINI_API_KEY=AIza...
EOF
```

### Run an evolution

```bash
source .env
python -m madevolve run -c configs/<config>.yaml -o ./results/<experiment_name>
```

Available configs:

```
configs/bao_reconstruction_autodiff_limit_params.yaml
configs/iterative_reconstruction_autodiff.yaml
configs/tidal_reconstruction_autodiff.yaml
configs/ldl_camels_tsz.yaml
```

Example:

```bash
source .env
python -m madevolve run \
  -c configs/bao_reconstruction_autodiff_limit_params.yaml \
  -o ./results/bao_autodiff
```

## Project Structure

```
MadEvolve-Cosmo/
├── configs/                       # MadEvolve evolution configs
│   ├── bao_reconstruction_autodiff_limit_params.yaml
│   ├── iterative_reconstruction_autodiff.yaml
│   ├── tidal_reconstruction_autodiff.yaml
│   └── ldl_camels_tsz.yaml
├── examples/                      # Task definitions
│   ├── bao_reconstruction_auto_diff/
│   │   ├── initial_autodiff.py    # Baseline: Gaussian filter + Fourier reconstruction
│   │   └── evaluate_autodiff.py   # Evaluator: r(k) with large-scale penalty
│   ├── iterative_reconstruction/
│   │   ├── initial.py             # Baseline: RecCal_multi (8 iterations)
│   │   ├── evaluate.py            # Evaluator: r(k) with large-scale penalty
│   │   └── MLR/                   # Cython libs for particle-based reconstruction
│   ├── tidal_reconstruction_auto_diff/
│   │   ├── initial_autodiff.py    # Baseline: Tidal tensor method
│   │   ├── evaluate_autodiff.py   # Evaluator: r_2D(k_perp, k_para)
│   │   └── CosCal/                # CIC/NGP painting & power spectrum libs
│   └── ldl_camels/
│       ├── initial.py             # Baseline: LDL with Displacement + baryon bias
│       └── evaluate.py            # Evaluator: MPI two-stage train/validate
├── final_programs/                # Best evolved programs
│   ├── bao_reconstruction_auto_diff/
│   ├── ldl_camels_tsz/
│   └── tidal_reconstruction/
├── madevolve_cosmo/               # Python package (analysis adapters)
│   ├── analyzer/                  # Report generation adapters
│   ├── eval_utils.py              # Shared evaluation utilities
│   └── logging_utils.py           # Logging helpers
├── scripts/
│   └── generate_report.py         # Post-evolution report generation
```

## How Evaluation Works

Each evaluator follows the MadEvolve dispatcher contract:

```
python evaluate.py candidate.py
```

The evaluator:
1. Loads `candidate.py` as a module
2. Calls `run_experiment()` on simulation data
3. Validates results (finite values, correct shapes)
4. Computes domain-specific metrics
5. Writes `result.json` with:

```json
{
  "success": true,
  "combined_score": 0.85,
  "public_metrics": {"mean_r_k": 0.85, "...": "..."},
  "private_metrics": {},
  "text_feedback": "r(k) improved at k > 0.2...",
  "error": null
}
```

### Autodiff Parameter Optimization

BAO and tidal tasks use JAX-based autodiff. Programs declare tunable parameters:

```python
# TUNABLE: filter_scale = 1.5, bounds=(0.5, 5.0), method=autodiff

def run_reconstruction(data, filter_scale=1.5):
    ...  # all operations must use jax.numpy
```

The evaluator parses these declarations, runs Adam optimization using `jax.value_and_grad`, and injects optimized values back into the program before final evaluation.


## Generating Reports

After an evolution run completes:

```bash
python scripts/generate_report.py results/<experiment_dir> -s bao
python scripts/generate_report.py results/<experiment_dir> -s tidal
python scripts/generate_report.py results/<experiment_dir> -s ldl
```

## License

MIT License
