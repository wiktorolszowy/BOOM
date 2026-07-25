# Reproducing results from the BOOM paper

All commands should be run from the **repository root**.

## 1. Install uv

[uv](https://github.com/astral-sh/uv) is used to manage the Python environment and dependencies.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Set up the environment

Install all project dependencies and set up pre-commit hooks:

```bash
uv sync
uv add --dev pre-commit && uv run pre-commit install
```

## 3. Run the reproduction script

This script reproduces Figure 2 results across all 10 endpoints (Density, HoF, and the 8 QM9 properties) for four models — Random Forest, Chemprop (MPNN), Elastic Net, and XGBoost. For every endpoint it evaluates **two** out-of-distribution definitions: the property-value (KDE) OOD from the paper, and a chemical-**structure** OOD based on UMAP clusters of Morgan fingerprints. The script handles data download, OOD split generation, feature caching, and model training/evaluation automatically.

Runs are **seeded**. The seed is chosen with `--seed` (default `42`) and is encoded in the results **filename** — `reproduce/results_incremental_seed<seed>.json` — never inside the file. The data splits (both KDE and UMAP) are held fixed across seeds; only model training varies, so only the stochastic learners (Chemprop, Random Forest) change between seeds — Elastic Net and default XGBoost are deterministic and give identical numbers across seeds.

For a quick sanity check (reduced data and fewer estimators; writes to `results_incremental_smoke_seed<seed>.json`):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --smoke-test
```

For the full run with the default seed 42 (this will take a while):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 42
```

For the multi-seed analysis, run the other two seeds as well:

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 43
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 44
```

Each produces its own `reproduce/results_incremental_seed<seed>.json`.

## 4. Generate heatmaps

Once a per-seed results file has been populated, generate the summary heatmaps (RMSE, binned R², ρ², and the structure-OOD R²/RMSE) for that seed:

```bash
uv run python reproduce/make_heatmaps.py --seed 42
```

Output figures are saved in `reproduce/figures/`.
