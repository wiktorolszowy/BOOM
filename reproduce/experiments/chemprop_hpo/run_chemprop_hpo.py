#!/usr/bin/env python
"""
Chemprop hyperparameter-optimization (HPO) study on the BOOM property (KDE) OOD splits.
============================================================================================

Purpose
-------
Reproduce the ablation described in the BOOM appendix:

    "To understand to what extent hyperparameter optimization can affect OOD
    performance, we first train the Chemprop model (v1.4.0) with all default
    hyperparameters on all 10 BOOM datasets. Then, we train Chemprop on each of
    the 10 BOOM datasets with 50 independent, random choices of model
    hyperparameters (i.e. with 50 random seeds). The tuned model hyperparameters
    are the message passing depth, sampled from between 2-6 layers, the fraction
    of dropout in the neural network sampled between 0-0.40 with an increment of
    0.05, the number of feed-forward layers sampled from between 1-3 layers, and
    the size of the hidden layers, sampled between 300-2400 with an increment of
    100."

For every one of the 10 endpoints we train:
  * one DEFAULT model (Chemprop's default hyperparameters), and
  * N_CONFIGS (default 50) models with randomly sampled hyperparameters,
and we record OOD (and ID) metrics for each, so the effect of hyperparameter
tuning on OOD generalization can be quantified.

Important reproduction notes
----------------------------
* BOOM used Chemprop **v1.4.0**; this repo's pipeline (and therefore this
  script) uses Chemprop **v2**. The training recipe here mirrors
  ``_train_chemprop`` in ``reproduce/reproduce_parts_of_fig_2_and_add_models.py``
  so results are consistent with the rest of the reproduction.
* BOOM v1.4.0 exposed a single ``hidden_size`` knob. Chemprop v2 splits this
  into ``d_h`` (message-passing hidden dim) and ``ffn_hidden_dim``. To stay
  faithful to BOOM, the sampled "hidden size" sets **both** of these.
* BOOM does not state the number of training epochs. We use ``--max-epochs 100``
  by default (with early stopping, ``patience=10``), matching ``max_epochs=100``
  used for the single-run pipeline numbers in ``results_incremental_seed42.json``
  so this study's DEFAULT config lines up closely with that file's Chemprop entry.

Determinism
-----------
The study is designed to be reproducible: a rerun should give the same 50
configurations and (very close to) the same metrics.
  * ``--config-seed``    fixes the 50 sampled hyperparameter configurations.
  * ``--val-seed``       fixes the 90/10 train/val split, and is the SAME for the
                         default run and every one of the 50 configs (fair
                         comparison). It defaults to 42 so the DEFAULT run lines
                         up with the existing seed-42 Chemprop pipeline results.
  * per-config train seed = ``--train-seed-base`` + config index (distinct per
                         config so the 50 runs are genuinely different, yet
                         deterministic). The DEFAULT run uses ``--val-seed``.
Full torch determinism flags are set, but GPU kernels are not guaranteed to be
bit-for-bit deterministic (``use_deterministic_algorithms(warn_only=True)``), so
"identical" is the target and "very similar" is the realistic guarantee.

Model selection
---------------
For every run we store both the best validation loss (``val_loss``) and all OOD
metrics. This lets the summary report the tuned model chosen two ways:
  * HONEST     — the config with the lowest validation loss, report its OOD.
  * OPTIMISTIC — the config with the best OOD test metric (this is what BOOM did;
                 it is optimistic because it selects on the test set).

Usage
-----
Smoke test (fast wiring/determinism check — tiny model, few epochs/configs):
    uv run python reproduce/experiments/chemprop_hpo/run_chemprop_hpo.py --smoke-test

Full study (all 10 endpoints, 50 configs each; resumable — safe to re-run):
    uv run python reproduce/experiments/chemprop_hpo/run_chemprop_hpo.py

The runner saves results incrementally and resumes automatically: re-running
skips endpoints/configs already present in the output JSON.
"""

import argparse
import datetime
import json
import os
import sys
import time

import lightning as pl
import numpy as np
import torch
from chemprop import data as chemprop_data
from chemprop import models as chemprop_models
from chemprop import nn as chemprop_nn
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score, root_mean_squared_error

try:
    import chemprop as _chemprop_pkg

    CHEMPROP_VERSION = getattr(_chemprop_pkg, "__version__", "unknown")
except Exception:  # pragma: no cover
    CHEMPROP_VERSION = "unknown"

RDLogger.logger().setLevel(RDLogger.ERROR)

# NOTE: we deliberately do NOT call torch.set_float32_matmul_precision("high")
# here. TF32 matmuls are faster but numerically diverge from the original
# pipeline (reproduce_parts_of_fig_2_and_add_models.py), which never sets this
# and therefore always runs at full FP32 precision. Keeping the default here
# maximizes fidelity to that script's results.

# ---------------------------------------------------------------------------
# Paths.  This file lives at reproduce/experiments/chemprop_hpo/.
#   REPO_ROOT = .../BOOM
#   DATA_DIR  = reproduce/experiments/data  (SMILESDataset resolves split CSVs
#               relative to os.getcwd(), so we chdir there).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(REPO_ROOT, "reproduce", "experiments", "data")
sys.path.insert(0, REPO_ROOT)
os.chdir(DATA_DIR)

from boom.datasets.SMILESDataset import SMILESDataset  # noqa: E402

# ===== Endpoints ============================================================
# (property_key, human_label). property_key must match SMILESDataset property.
ENDPOINTS = [
    ("hof", "HoF"),
    ("density", "Density"),
    ("homo", "HOMO"),
    ("lumo", "LUMO"),
    ("gap", "GAP"),
    ("zpve", "ZPVE"),
    ("r2", "R²"),
    ("alpha", "α"),
    ("mu", "μ"),
    ("cv", "Cᵥ"),
]

# ===== Hyperparameter search space (BOOM appendix) ==========================
# message-passing depth: 2-6 layers
HP_DEPTHS = [2, 3, 4, 5, 6]
# dropout fraction: 0-0.40, increment 0.05
HP_DROPOUTS = [round(0.05 * k, 2) for k in range(9)]  # 0.0 .. 0.40
# feed-forward layers: 1-3
HP_FFN_LAYERS = [1, 2, 3]
# hidden size: 300-2400, increment 100  (maps to BOTH d_h and ffn_hidden_dim)
HP_HIDDENS = list(range(300, 2401, 100))  # 300 .. 2400

# Chemprop "default" hyperparameters — identical to CHEMPROP_PARAMS in
# reproduce_parts_of_fig_2_and_add_models.py (the config used everywhere else).
DEFAULT_HP = {"depth": 3, "dropout": 0.0, "ffn_n_layers": 2, "hidden": 300}

# Fixed (non-tuned) hyperparameters, shared by the default and all 50 configs.
FIXED = {
    "max_lr": 1e-3,
    "batch_norm": True,
    "batch_size": 1024,
}

# ===== Deterministic seeds (overridable via CLI) ============================
CONFIG_SEED = 12345  # fixes the 50 sampled configs
VAL_SPLIT_SEED = 42  # fixes the 90/10 val split (same for default + all configs)
TRAIN_SEED_BASE = 1000  # per-config train seed = TRAIN_SEED_BASE + config_index

# ===== Smoke-test caps ======================================================
SMOKE_TRAIN_MAX = 512
SMOKE_EVAL_MAX = 256


# ===== Timing helpers =======================================================
def _ts():
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def _elapsed(t0):
    secs = time.time() - t0
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ===== Metrics (mirrors reproduce_parts_of_fig_2_and_add_models.py) =========
def binned_r2(true, pred, train_median):
    """Average coefficient-of-determination R² over the lower/upper tails of the
    OOD set, split at the training median (BOOM's binned R²)."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    vals = []
    for mask in (lower, ~lower):
        if int(mask.sum()) >= 2:
            vals.append(r2_score(true[mask], pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def corr_r2(true, pred):
    """Square of the Pearson correlation coefficient (ρ²), always in [0, 1]."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    if len(true) < 2:
        return float("nan")
    rho = np.corrcoef(true, pred)[0, 1]
    return float(rho**2)


def binned_corr_r2(true, pred, train_median):
    """Binned ρ²: same tail split as ``binned_r2`` but using ρ² per bin."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    vals = []
    for mask in (lower, ~lower):
        if int(mask.sum()) >= 2:
            vals.append(corr_r2(true[mask], pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


# ===== Deterministic config sampling ========================================
def sample_configs(n, seed):
    """Sample ``n`` hyperparameter configurations deterministically.

    Uses ``RandomState`` + integer indexing (``randint``) rather than
    ``choice`` for stability across numpy versions.  Independent draws, so
    duplicates are possible (matching BOOM's "50 independent random choices").
    """
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n):
        configs.append(
            {
                "depth": int(HP_DEPTHS[rng.randint(len(HP_DEPTHS))]),
                "dropout": float(HP_DROPOUTS[rng.randint(len(HP_DROPOUTS))]),
                "ffn_n_layers": int(HP_FFN_LAYERS[rng.randint(len(HP_FFN_LAYERS))]),
                "hidden": int(HP_HIDDENS[rng.randint(len(HP_HIDDENS))]),
            }
        )
    return configs


# ===== Datapoint construction ===============================================
def _make_datapoints(smiles_dataset):
    """SMILES -> (list[MoleculeDatapoint], np.ndarray of targets). Skips
    molecules RDKit cannot parse."""
    dps, ys = [], []
    for smi, target in smiles_dataset:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            dps.append(chemprop_data.MoleculeDatapoint(mol, y=np.array([target])))
            ys.append(float(target))
    return dps, np.array(ys, dtype=np.float64)


def _accelerator():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


# ===== Single training run ==================================================
def train_one(
    train_dps,
    id_dps,
    id_true,
    ood_dps,
    ood_true,
    train_median,
    hp,
    train_seed,
    val_seed,
    max_epochs,
    patience,
    batch_size,
    num_workers,
    smoke,
    show_progress,
):
    """Train one Chemprop MPNN with hyperparameters ``hp`` and return a dict of
    ID/OOD metrics (original target scale), best val_loss and stopped epoch.

    ``hp`` keys: depth, dropout, ffn_n_layers, hidden. The sampled ``hidden``
    sets both the message-passing ``d_h`` and the FFN hidden dim (BOOM's single
    ``hidden_size``). On CUDA OOM the run is retried at half batch size
    (deterministic on the same GPU); the batch size actually used is recorded.
    """
    pl.seed_everything(train_seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hidden = hp["hidden"]

    # Fixed 90/10 train/val split (same across default + all configs).
    rng = np.random.RandomState(val_seed)
    idx = rng.permutation(len(train_dps))
    val_n = max(1, len(train_dps) // 10)
    val_dps = [train_dps[i] for i in idx[:val_n]]
    tr_dps = [train_dps[i] for i in idx[val_n:]]

    train_dataset = chemprop_data.MoleculeDataset(tr_dps)
    val_dataset = chemprop_data.MoleculeDataset(val_dps)
    id_dataset = chemprop_data.MoleculeDataset(id_dps)
    ood_dataset = chemprop_data.MoleculeDataset(ood_dps)

    # Chemprop's target scaler is the sole normalisation; UnscaleTransform
    # reverses it so predictions come back in the original target scale.
    scaler = train_dataset.normalize_targets()
    val_dataset.normalize_targets(scaler)

    accelerator = _accelerator()

    def _attempt(bs):
        train_loader = chemprop_data.build_dataloader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
        )
        val_loader = chemprop_data.build_dataloader(val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
        id_loader = chemprop_data.build_dataloader(id_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
        ood_loader = chemprop_data.build_dataloader(ood_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

        mp = chemprop_nn.BondMessagePassing(d_h=hidden, depth=hp["depth"], dropout=hp["dropout"])
        agg = chemprop_nn.MeanAggregation()
        output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = chemprop_nn.RegressionFFN(
            n_tasks=1,
            input_dim=hidden,
            hidden_dim=hidden,
            n_layers=hp["ffn_n_layers"],
            dropout=hp["dropout"],
            output_transform=output_transform,
        )
        mpnn = chemprop_models.MPNN(mp, agg, ffn, batch_norm=FIXED["batch_norm"], max_lr=FIXED["max_lr"])
        n_params = sum(p.numel() for p in mpnn.parameters())

        early_stopping = EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=False)
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=show_progress,
            enable_model_summary=False,
            accelerator=accelerator,
            devices=1,
            max_epochs=max_epochs,
            callbacks=[early_stopping],
            limit_train_batches=20 if smoke else 1.0,
            limit_val_batches=5 if smoke else 1.0,
            num_sanity_val_steps=0,
        )
        trainer.fit(mpnn, train_loader, val_loader)
        stopped_epoch = trainer.current_epoch
        best = early_stopping.best_score
        val_loss = float(best) if best is not None else float("nan")

        with torch.inference_mode():
            id_preds = trainer.predict(mpnn, id_loader)
            ood_preds = trainer.predict(mpnn, ood_loader)
        id_pred = torch.cat(id_preds, dim=0).detach().cpu().numpy().flatten()
        ood_pred = torch.cat(ood_preds, dim=0).detach().cpu().numpy().flatten()
        return id_pred, ood_pred, val_loss, stopped_epoch, n_params

    # OOM-tolerant execution: halve batch size until it fits (>= 64).
    bs = batch_size
    id_pred = ood_pred = None
    val_loss = stopped_epoch = n_params = None
    batch_used = bs
    while True:
        try:
            id_pred, ood_pred, val_loss, stopped_epoch, n_params = _attempt(bs)
            batch_used = bs
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 64:
                return {"failed": "oom", "hp": hp, "batch_used": bs}
            bs = bs // 2
            print(f"      [OOM] retrying with batch_size={bs}")

    return {
        "hp": hp,
        "n_params": int(n_params),
        "batch_used": int(batch_used),
        "val_loss": val_loss,
        "stopped_epoch": int(stopped_epoch),
        "train_median": float(train_median),
        "id_r2": float(r2_score(id_true, id_pred)),
        "id_r2_corr": corr_r2(id_true, id_pred),
        "id_rmse": float(root_mean_squared_error(id_true, id_pred)),
        "ood_r2_binned": binned_r2(ood_true, ood_pred, train_median),
        "ood_r2_corr_binned": binned_corr_r2(ood_true, ood_pred, train_median),
        "ood_rmse": float(root_mean_squared_error(ood_true, ood_pred)),
    }


# ===== Results persistence ==================================================
def results_path(smoke):
    tag = "_smoke" if smoke else ""
    return os.path.join(SCRIPT_DIR, f"chemprop_hpo_results{tag}.json")


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_results(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)  # atomic; a crash mid-write cannot corrupt the file


# ===== Study driver =========================================================
def run_study(args):
    smoke = args.smoke_test
    max_epochs = args.max_epochs if not smoke else 2
    patience = args.patience if not smoke else 1
    batch_size = args.batch_size if not smoke else 256
    n_configs = args.n_configs if not smoke else 3

    endpoints = ENDPOINTS
    if args.endpoints and args.endpoints.lower() != "all":
        wanted = {e.strip().lower() for e in args.endpoints.split(",")}
        endpoints = [(k, lbl) for (k, lbl) in ENDPOINTS if k in wanted]
        if not endpoints:
            raise SystemExit(f"No endpoints matched {args.endpoints!r}. Options: {[k for k, _ in ENDPOINTS]}")

    configs = sample_configs(n_configs, args.config_seed)

    path = results_path(smoke)
    results = load_results(path)
    if results is None:
        results = {"meta": {}, "endpoints": {}}
    # (Re)write meta each launch — it is purely descriptive.
    results["meta"] = {
        "chemprop_version": CHEMPROP_VERSION,
        "smoke_test": smoke,
        "n_configs": n_configs,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "config_seed": args.config_seed,
        "val_split_seed": args.val_seed,
        "train_seed_base": args.train_seed_base,
        "fixed": FIXED,
        "default_hp": DEFAULT_HP,
        "hp_grid": {
            "depth": HP_DEPTHS,
            "dropout": HP_DROPOUTS,
            "ffn_n_layers": HP_FFN_LAYERS,
            "hidden": HP_HIDDENS,
        },
        "sampled_configs": configs,
        "note": (
            "Chemprop v2 reproduction of BOOM v1.4.0 HPO ablation. 'hidden' sets "
            "both d_h and ffn_hidden_dim. Default run uses train_seed=val_seed."
        ),
    }
    save_results(path, results)

    t_all = time.time()
    print(f"{_ts()} Chemprop HPO study | chemprop {CHEMPROP_VERSION} | accelerator={_accelerator()}")
    print(
        f"  smoke={smoke}  endpoints={[k for k, _ in endpoints]}  n_configs={n_configs}  "
        f"max_epochs={max_epochs}  patience={patience}  batch_size={batch_size}"
    )
    print(f"  results -> {path}")

    for prop, label in endpoints:
        print(f"\n{'='*78}\n{_ts()} Endpoint: {label} ({prop})\n{'='*78}")
        ep = results["endpoints"].setdefault(prop, {"label": label, "default": None, "configs": []})

        # --- Load splits once per endpoint -------------------------------
        train_ds = SMILESDataset(prop, "train")
        id_ds = SMILESDataset(prop, "id")
        ood_ds = SMILESDataset(prop, "ood")
        train_dps, train_y = _make_datapoints(train_ds)
        id_dps, id_true = _make_datapoints(id_ds)
        ood_dps, ood_true = _make_datapoints(ood_ds)

        if smoke:
            train_dps = train_dps[:SMOKE_TRAIN_MAX]
            train_y = train_y[:SMOKE_TRAIN_MAX]
            id_dps, id_true = id_dps[:SMOKE_EVAL_MAX], id_true[:SMOKE_EVAL_MAX]
            ood_dps, ood_true = ood_dps[:SMOKE_EVAL_MAX], ood_true[:SMOKE_EVAL_MAX]

        train_median = float(np.median(train_y))
        print(
            f"  datapoints: train={len(train_dps)} id={len(id_dps)} ood={len(ood_dps)} "
            f"train_median={train_median:.4g}"
        )

        common = dict(
            train_dps=train_dps,
            id_dps=id_dps,
            id_true=id_true,
            ood_dps=ood_dps,
            ood_true=ood_true,
            train_median=train_median,
            val_seed=args.val_seed,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            num_workers=args.num_workers,
            smoke=smoke,
            show_progress=args.progress,
        )

        # --- DEFAULT run (resumable) -------------------------------------
        if ep.get("default") is None:
            t0 = time.time()
            print(f"  {_ts()} DEFAULT {DEFAULT_HP}")
            rec = train_one(hp=DEFAULT_HP, train_seed=args.val_seed, **common)
            rec["train_seed"] = args.val_seed
            ep["default"] = rec
            save_results(path, results)
            _print_rec("DEFAULT", rec, t0)
        else:
            print("  DEFAULT already done -> skipping")

        # --- 50 sampled configs (resumable) ------------------------------
        done = len(ep["configs"])
        for i in range(done, n_configs):
            hp = configs[i]
            train_seed = args.train_seed_base + i
            t0 = time.time()
            print(f"  {_ts()} config {i+1}/{n_configs} {hp} (train_seed={train_seed})")
            rec = train_one(hp=hp, train_seed=train_seed, **common)
            rec["train_seed"] = train_seed
            rec["config_index"] = i
            ep["configs"].append(rec)
            save_results(path, results)
            _print_rec(f"config {i+1}", rec, t0)

    _print_summary(results, endpoints)
    print(f"\n{_ts()} Done. Total {_elapsed(t_all)}. Results: {path}")


def _print_rec(tag, rec, t0):
    if rec.get("failed"):
        print(f"    {tag}: FAILED ({rec['failed']})  [{_elapsed(t0)}]")
        return
    print(
        f"    {tag}: OOD binnedR²={rec['ood_r2_binned']:.4f} OOD RMSE={rec['ood_rmse']:.4g} "
        f"ID R²={rec['id_r2']:.4f} val_loss={rec['val_loss']:.4g} "
        f"epoch={rec['stopped_epoch']} params={rec['n_params']:,}  [{_elapsed(t0)}]"
    )


def _print_summary(results, endpoints):
    """Per-endpoint: DEFAULT vs HONEST (best val_loss) vs OPTIMISTIC (best OOD)."""
    print(f"\n{'='*78}\nSUMMARY — OOD binned R² (higher is better)\n{'='*78}")
    header = f"{'endpoint':<10} {'default':>10} {'honest':>10} {'optimistic':>12}"
    print(header)
    print("-" * len(header))
    for prop, label in endpoints:
        ep = results["endpoints"].get(prop)
        if not ep:
            continue
        default = ep.get("default") or {}
        d_ood = default.get("ood_r2_binned", float("nan"))
        good = [c for c in ep.get("configs", []) if not c.get("failed")]
        honest = optimistic = float("nan")
        if good:
            # HONEST: pick config with lowest val_loss, report its OOD.
            valid_val = [c for c in good if c.get("val_loss") == c.get("val_loss")]  # not NaN
            if valid_val:
                honest = min(valid_val, key=lambda c: c["val_loss"])["ood_r2_binned"]
            # OPTIMISTIC: best OOD binned R² directly (BOOM-style test selection).
            valid_ood = [c for c in good if c.get("ood_r2_binned") == c.get("ood_r2_binned")]
            if valid_ood:
                optimistic = max(valid_ood, key=lambda c: c["ood_r2_binned"])["ood_r2_binned"]
        print(f"{label:<10} {d_ood:>10.4f} {honest:>10.4f} {optimistic:>12.4f}")
    print("\nhonest = config with best validation loss (fair model selection)")
    print("optimistic = config with best OOD test metric (BOOM-style; selects on test)")


def parse_args():
    p = argparse.ArgumentParser(description="Chemprop HPO study on BOOM property (KDE) OOD splits.")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast wiring/determinism check: tiny model, 2 epochs, 3 configs, subset data.",
    )
    p.add_argument("--endpoints", default="all", help="Comma-separated endpoint keys (e.g. 'hof,density') or 'all'.")
    p.add_argument("--n-configs", type=int, default=50, help="Number of random HP configs per endpoint.")
    p.add_argument("--max-epochs", type=int, default=100, help="Max training epochs (early stopping applies).")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs).")
    p.add_argument("--batch-size", type=int, default=1024, help="Batch size (halved automatically on CUDA OOM).")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--config-seed", type=int, default=CONFIG_SEED, help="Seed for sampling the HP configs.")
    p.add_argument(
        "--val-seed",
        type=int,
        default=VAL_SPLIT_SEED,
        help="Seed for the 90/10 val split (same for all runs); also the DEFAULT run's train seed.",
    )
    p.add_argument(
        "--train-seed-base", type=int, default=TRAIN_SEED_BASE, help="Per-config train seed = base + config_index."
    )
    p.add_argument("--progress", action="store_true", help="Show the Lightning progress bar.")
    return p.parse_args()


if __name__ == "__main__":
    run_study(parse_args())
