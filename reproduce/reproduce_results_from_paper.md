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

This script reproduces Figure 2 results across all 10 endpoints (Density, HoF, and the 8 QM9 properties) for four models — Random Forest, Chemprop (MPNN), Elastic Net, and XGBoost. For every endpoint it evaluates **two** out-of-distribution definitions: the property-value (KDE) OOD from the paper, and a chemical-**structure** OOD based on UMAP + HDBSCAN over Morgan fingerprints (holding out the most detached clusters). The script handles data download, OOD split generation, feature caching, and model training/evaluation automatically.

Runs are **seeded**. The seed is chosen with `--seed` (default `42`) and is encoded in the results **filename** — `reproduce/results_incremental_seed<seed>.json` — never inside the file. The data splits (both KDE and UMAP) are held fixed across seeds; only model training varies, so only the stochastic learners (Chemprop, Random Forest) change between seeds — Elastic Net and default XGBoost are deterministic and give identical numbers across seeds.

For a quick sanity check (reduced data and fewer estimators; writes to `results_incremental_smoke_seed<seed>.json`):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --smoke-test
```

To generate only the structure-based UMAP/HDBSCAN splits and plots (no model training):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --umap-splits-only
```

For the full run with the default seed 42 (this will take a while):

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 42
```

Run in background with `nohup` (safe if your terminal/session disconnects):

```bash
nohup uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 42 > run_seed42.out 2>&1 &
```

For the multi-seed analysis, run the other two seeds as well:

```bash
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 43
uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed 44
```

Each produces its own `reproduce/results_incremental_seed<seed>.json`.

To run all three seeds sequentially (42 → 43 → 44) in one foreground command:

```bash
for s in 42 43 44; do
	echo "Running seed $s"
	uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed "$s" > "run_seed${s}.out" 2>&1 || break
	echo "Seed $s finished"
done
```

Background (`nohup`) sequential run:

```bash
nohup bash -lc '
for s in 42 43 44; do
	echo "Running seed $s"
	uv run python reproduce/reproduce_parts_of_fig_2_and_add_models.py --seed "$s" > "run_seed${s}.out" 2>&1 || break
	echo "Seed $s finished"
done
' > run_all_seeds.out 2>&1 &
```

## 4. Generate heatmaps

Once a per-seed results file has been populated, generate the summary heatmaps (RMSE, binned R², ρ², and the structure-OOD R²/RMSE) for that seed:

```bash
uv run python reproduce/make_heatmaps.py --seed 42
```

Background (`nohup`) heatmap generation:

```bash
nohup uv run python reproduce/make_heatmaps.py --seed 42 > heatmaps_seed42.out 2>&1 &
```

Output figures are saved in `reproduce/figures/`.

## 5. (Optional) Add GotenNet — a 3D equivariant model

GotenNet ([sarpaykent/GotenNet](https://github.com/sarpaykent/GotenNet), ICLR 2025) is a 3D-equivariant graph network. It is added as a **fifth** model *on top of* an existing per-seed results file, without touching the other four models. Because it needs the PyG CUDA extension stack (`torch_scatter` / `torch_sparse` / `torch_cluster`) built against a specific torch build, it lives in its **own** virtual environment and communicates with the rest of the pipeline only through the split CSVs and the results JSON.

The target hardware for this repo is a single **NVIDIA A10G**. Following the BOOM protocol, GotenNet is trained for **50 epochs** per endpoint. (The paper used an A100; on the A10G the same 50-epoch schedule runs but takes longer.)

First create the isolated environment (one time):

```bash
bash reproduce/experiments/gotennet/setup_env.sh
```

Then run the model. It reads `reproduce/results_incremental_seed<seed>.json`, trains GotenNet on the **same** KDE and UMAP-structure splits, and merges only `results["GotenNet"]` back in:

```bash
source reproduce/experiments/gotennet/.venv_goten/bin/activate
# quick sanity check on one endpoint (tiny net, 3 epochs, writes a *_smoke file):
python reproduce/experiments/gotennet/run_gotennet.py --smoke-test --endpoints hof
# full 50-epoch run for seed 42, all 10 endpoints:
python reproduce/experiments/gotennet/run_gotennet.py --seed 42
```

3D conformers are generated once per dataset group with RDKit (ETKDG + MMFF) and cached to `reproduce/experiments/data/goten_3d_{10k,QM9}.pkl`. Run the other seeds (43, 44) the same way. Because the run can be long, `nohup` is recommended:

```bash
nohup python reproduce/experiments/gotennet/run_gotennet.py --seed 42 > goten_seed42.out 2>&1 &
```

After GotenNet is merged, re-run the heatmaps for that seed (they auto-discover the new model):

```bash
uv run python reproduce/make_heatmaps.py --seed 42
```
