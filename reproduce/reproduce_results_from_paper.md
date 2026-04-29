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

This script reproduces Figure 2 results for Random Forest and Chemprop (MPNN) across all 10 endpoints (Density, HoF, and the 8 QM9 properties), and additionally runs an Elastic Net baseline not present in the original paper. The script handles data download, OOD split generation, feature caching, and model training/evaluation automatically. Results are written to `reproduce/results_incremental.json`.

For a quick sanity check (reduced data and fewer estimators):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_elastic_net.py --smoke-test
```

For the full run (this will take a while):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_elastic_net.py
```

## 4. Generate heatmaps

Once `reproduce/results_incremental.json` has been populated, generate the summary heatmaps:

```bash
uv run python reproduce/make_heatmaps.py
```

Output figures are saved in `reproduce/figures/`.
