# Chemprop hyperparameter-optimization (HPO) study

Reproduces the BOOM appendix ablation that asks **how much hyperparameter tuning
can change out-of-distribution (OOD) performance**, on the 10 BOOM property
(KDE) OOD splits.

For each of the 10 endpoints (`hof`, `density`, `homo`, `lumo`, `gap`, `zpve`,
`r2`, `alpha`, `mu`, `cv`) we train:

- **1 default** Chemprop model (Chemprop's default hyperparameters), and
- **50 randomly-sampled** Chemprop models (the HP search),

and record ID/OOD metrics for every run.

## BOOM protocol (verbatim)

> To understand to what extent hyperparameter optimization can affect OOD
> performance, we first train the Chemprop model (v1.4.0) with all default
> hyperparameters on all 10 BOOM datasets. Then, we train Chemprop on each of the
> 10 BOOM datasets with 50 independent, random choices of model hyperparameters
> (i.e. with 50 random seeds). The tuned model hyperparameters are the message
> passing depth, sampled from between 2-6 layers, the fraction of dropout in the
> neural network sampled between 0-0.40 with an increment of 0.05, the number of
> feed-forward layers sampled from between 1-3 layers, and the size of the hidden
> layers, sampled between 300-2400 with an increment of 100.

## Search space

| hyperparameter        | values sampled                          | Chemprop v2 knob(s)             |
| --------------------- | --------------------------------------- | ------------------------------- |
| message-passing depth | `{2, 3, 4, 5, 6}`                       | `BondMessagePassing.depth`      |
| dropout               | `{0.00, 0.05, …, 0.40}` (9 values)      | dropout (message passing + FFN) |
| feed-forward layers   | `{1, 2, 3}`                             | `RegressionFFN.n_layers`        |
| hidden size           | `{300, 400, …, 2400}` (22 values)       | **both** `d_h` and FFN hidden   |

The **default** baseline is `depth=3, dropout=0.0, ffn_n_layers=2, hidden=300`
(plus `max_lr=1e-3`, `batch_norm=True`, `batch_size=1024`) — identical to
`CHEMPROP_PARAMS` used everywhere else in the reproduction.

## Reproduction notes / differences from BOOM

- **Chemprop version.** BOOM used **v1.4.0**; this repo uses **Chemprop v2**. The
  training recipe mirrors `_train_chemprop` in
  `reproduce/reproduce_parts_of_fig_2_and_add_models.py` for consistency.
- **Single `hidden_size` → two knobs.** BOOM v1.4.0 had one `hidden_size`;
  Chemprop v2 splits it into `d_h` (message passing) and `ffn_hidden_dim`. To stay
  faithful, the sampled hidden size sets **both**.
- **Epochs.** BOOM does not state the epoch count. We use `--max-epochs 100`
  (with early stopping, `patience=10`) by default, matching `max_epochs=100`
  used for the single-run numbers in `results_incremental_seed42.json`, so this
  study's DEFAULT run lines up closely with that file's Chemprop entry (same
  seed, same val split, same hyperparameters, same epoch budget).

## Determinism

A rerun produces the **same 50 configurations** and (very close to) the same
metrics. Three deterministic seed roles:

- `--config-seed` (default `12345`) — fixes the 50 sampled HP configurations.
- `--val-seed` (default `42`) — fixes the 90/10 train/val split, the **same** for
  the default run and all 50 configs (fair comparison). Defaults to 42 so the
  DEFAULT run aligns with the existing seed-42 Chemprop pipeline results. Also
  used as the DEFAULT run's training seed.
- `--train-seed-base` (default `1000`) — each config's training seed is
  `base + config_index`, so the 50 runs differ yet are reproducible.

Full torch determinism flags are set
(`seed_everything(workers=True)`, `use_deterministic_algorithms(warn_only=True)`,
cuDNN deterministic). **GPU kernels are not guaranteed bit-for-bit
deterministic**, so "very similar" is the realistic guarantee, not "identical".

Note: this script deliberately does **not** call
`torch.set_float32_matmul_precision("high")` (TF32) or change dataloader batch
sizes/worker persistence away from the original pipeline's settings. An earlier
version of this study enabled TF32 for speed, but that measurably changed the
DEFAULT run's OOD metrics relative to `results_incremental_seed42.json` (TF32
uses reduced-mantissa matmuls). It was reverted to prioritize fidelity to the
existing pipeline results over training speed.

## Model selection (why we store `val_loss`)

Every run stores its best validation loss and all OOD metrics, so the summary can

Every run stores its best validation loss and all OOD metrics, so the summary can
report the tuned model two ways:

- **honest** — the config with the **lowest validation loss**; report its OOD.
  This is the fair estimate (selection does not touch the test set).
- **optimistic** — the config with the **best OOD test metric**. This is what
  BOOM did; it is optimistic because it selects on the test set.

## Usage

Run from the repo root, in the main environment.

Smoke test first (tiny model, 2 epochs, 3 configs, subset data — validates wiring
and determinism in a few minutes):

```bash
uv run python reproduce/experiments/chemprop_hpo/run_chemprop_hpo.py --smoke-test
```

Full study (all 10 endpoints × (1 + 50) = 510 trainings; long-running):

```bash
uv run python reproduce/experiments/chemprop_hpo/run_chemprop_hpo.py
```

Because this is compute-heavy, run it detached and let it resume if interrupted:

```bash
nohup uv run python reproduce/experiments/chemprop_hpo/run_chemprop_hpo.py \
    > reproduce/experiments/chemprop_hpo/hpo_run.log 2>&1 &
```

### Useful flags

| flag                 | default | meaning                                                   |
| -------------------- | ------- | --------------------------------------------------------- |
| `--smoke-test`       | off     | tiny/fast validation run, writes `*_smoke.json`           |
| `--endpoints`        | `all`   | comma-separated keys, e.g. `hof,density`                  |
| `--n-configs`        | `50`    | number of random HP configs per endpoint                  |
| `--max-epochs`       | `100`   | max epochs (early stopping applies)                       |
| `--patience`         | `10`    | early-stopping patience                                   |
| `--batch-size`       | `1024`  | halved automatically on CUDA OOM                          |
| `--num-workers`      | `4`     | DataLoader workers                                        |
| `--config-seed`      | `12345` | seed for sampling HP configs                              |
| `--val-seed`         | `42`    | 90/10 val-split seed + DEFAULT run train seed             |
| `--train-seed-base`  | `1000`  | per-config train seed = base + index                     |
| `--progress`         | off     | show the Lightning progress bar                           |

## Resumability

Results are saved incrementally (atomic write) after **every** run to:

- `chemprop_hpo_results.json` (full run), or
- `chemprop_hpo_results_smoke.json` (smoke test).

Re-running the command **skips** the DEFAULT run and any configs already present
for each endpoint, so an interrupted study can simply be restarted with the same
command.

## Output format

```jsonc
{
  "meta": {
    "chemprop_version": "...",
    "n_configs": 50, "max_epochs": 100, "patience": 10, "batch_size": 1024,
    "config_seed": 12345, "val_split_seed": 42, "train_seed_base": 1000,
    "fixed": { "max_lr": 0.001, "batch_norm": true, "batch_size": 1024 },
    "default_hp": { "depth": 3, "dropout": 0.0, "ffn_n_layers": 2, "hidden": 300 },
    "hp_grid": { "depth": [...], "dropout": [...], "ffn_n_layers": [...], "hidden": [...] },
    "sampled_configs": [ { "depth": .., "dropout": .., "ffn_n_layers": .., "hidden": .. }, ... ]
  },
  "endpoints": {
    "hof": {
      "label": "HoF",
      "default": { "hp": {...}, "train_seed": 42, "val_loss": .., "stopped_epoch": ..,
                   "id_r2": .., "id_r2_corr": .., "id_rmse": ..,
                   "ood_r2_binned": .., "ood_r2_corr_binned": .., "ood_rmse": ..,
                   "train_median": .., "n_params": .., "batch_used": .. },
      "configs": [ { "config_index": 0, "hp": {...}, "train_seed": 1000, ... }, ... ]
    },
    ...
  }
}
```

A failed run (e.g. CUDA OOM that could not be recovered even at the minimum batch
size) is recorded as `{ "failed": "oom", "hp": {...}, "batch_used": .. }` and the
study continues.

At the end, a summary table prints per-endpoint **DEFAULT vs honest vs
optimistic** OOD binned R².
