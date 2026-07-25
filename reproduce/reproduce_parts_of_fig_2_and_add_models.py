#!/usr/bin/env python
"""
Reproduce Figure 2 results from BOOM for all 10 endpoints
(Density, HoF, alpha, cv, gap, homo, lumo, mu, r2, zpve).
Models: Random Forest and Chemprop (MPNN) from the paper, plus
Elastic Net (degree-2 interaction features) as an additional baseline.

Self-contained: handles data download, CSV preparation, OOD split
generation (with progress output), feature caching, and model
training/evaluation.

Run from:  repo root
Usage:     uv run python reproduce_parts_of_fig_2_and_add_models.py
"""

import argparse
import datetime
import json
import math
import os
import platform
import random
import subprocess
import sys
import tarfile
import time
import warnings
from multiprocessing import Pool

import lightning as pl
import numpy as np
import torch
from chemprop import data as chemprop_data
from chemprop import models as chemprop_models
from chemprop import nn as chemprop_nn
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MolFromSmiles, MolToInchi
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import gaussian_kde
from tqdm import tqdm
from xgboost import XGBRegressor

RDLogger.logger().setLevel(RDLogger.ERROR)  # suppress InChI warnings

# ---------------------------------------------------------------------------
# SCRIPT_DIR = reproduce/ (where this file lives).
# REPO_ROOT  = parent of SCRIPT_DIR (where the boom package lives).
# DATA_DIR   = reproduce/experiments/data/ (self-contained downloads,
#              splits, and caches).
# os.chdir(DATA_DIR) is required because SMILESDataset resolves split
# files relative to os.getcwd().
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "experiments", "data")
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, REPO_ROOT)
os.chdir(DATA_DIR)

from boom.data.prepare_splits_10k import generate_splits_10k  # noqa: E402
from boom.data.prepare_splits_umap import (  # noqa: E402
    generate_umap_structure_split,
    load_umap_structure_split,
)
from boom.datasets.SMILESDataset import SMILESDataset  # noqa: E402


# ===== Tee: duplicate stdout to terminal + log file ========================
class Tee:
    """Write to multiple streams simultaneously (terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ===== Timing / memory helpers ==============================================


def _ts():
    """Current wall-clock time as [HH:MM:SS]."""
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def _elapsed(t0):
    """Human-readable elapsed time since *t0* (from time.time())."""
    secs = time.time() - t0
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _pred_range_str(true, pred):
    """One-line summary of prediction range vs. true range for sanity checking."""
    return f"true [{true.min():.4g}, {true.max():.4g}]  " f"pred [{pred.min():.4g}, {pred.max():.4g}]"


def _flag_metrics(id_r2, ood_r2_binned):
    """Return a warning string if metrics look extreme or invalid."""
    flags = []
    for name, val in [("ID R²", id_r2), ("OOD R²", ood_r2_binned)]:
        if math.isnan(val) or math.isinf(val):
            flags.append(f"{name}=NaN/Inf")
        elif val < -100:
            flags.append(f"{name}={val:.1f}")
    return f"  [!] Extreme metrics: {', '.join(flags)}" if flags else ""


def _mem_mb():
    """Peak RSS memory in MB (Linux kB→MB; macOS B→MB). Returns '-' on error."""
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = kb / 1e3 if sys.platform != "darwin" else kb / 1e6
        return f"{mb:.0f} MB"
    except Exception:
        return "-"


# ===== Configuration ========================================================
DEEPCHEM_S3_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_TAR = os.path.join(DATA_DIR, "gdb9.tar.gz")
QM9_SDF = os.path.join(DATA_DIR, "gdb9.sdf")
QM9_CSV = os.path.join(DATA_DIR, "gdb9.sdf.csv")

# 8 QM9 properties — names must be lowercase so the splits CSV columns
# match what _load_qm9_data expects (it looks for "qm9_" + target.lower()).
QM9_PROPS = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"]
QM9_SPLITS_FILE = os.path.join(DATA_DIR, "qm9_data_with_ood_splits_with_inchi.csv")
TENK_SPLITS_FILE = os.path.join(DATA_DIR, "10k_data_with_ood_splits.csv")

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


def _default_n_cpus():
    """Detect usable CPU count."""
    return max(1, os.cpu_count() or 1)


N_CPUS = _default_n_cpus()
CHEMPROP_SMOKE_TEST = False
SMOKE_TRAIN_MAX = 512
SMOKE_EVAL_MAX = 256

# ---- Seeds --------------------------------------------------------------
# DEFAULT_SEED reproduces the original single-run numbers.  SEEDS is used by
# the final multi-seed analysis (mean +/- std over these three runs).  Seeds
# vary *model training* only; the data splits (KDE + UMAP) stay fixed.
DEFAULT_SEED = 42
SEEDS = [42, 43, 44]
# Active model-training seed for this run (overridden by --seed in __main__).
# It is written into the results *filename* (results_incremental_seed<seed>.json),
# never into the file contents.
RUN_SEED = DEFAULT_SEED

# ---- UMAP structure-based OOD (added alongside the KDE property OOD) -----
UMAP_N_CLUSTERS = 20
UMAP_OOD_FRAC = 0.10
UMAP_SEED = 42  # split is frozen; independent of the model training seed
# Cap the UMAP universe in smoke-test mode so the embedding is fast.
UMAP_SMOKE_MAX_N = 1500


# ===== Helpers for parallelisation and fast featurisation ===================


def _smiles_to_inchi(smi):
    """Convert one SMILES -> InChI (picklable, for multiprocessing.Pool)."""
    mol = MolFromSmiles(smi)
    if mol:
        inchi = MolToInchi(mol)
        return inchi.replace(",", "$") if inchi else ""
    return ""


def _fast_postprocess(features, feature_names_orig):
    """Log-Ipc, vectorised NaN-column removal, correlated-feature removal."""
    feature_names = feature_names_orig.copy()
    ipc_idx = np.where(feature_names == "Ipc")[0]
    if len(ipc_idx) > 0:
        feature_names[ipc_idx[0]] = "log_Ipc"
        features[:, ipc_idx] = np.log(features[:, ipc_idx] + 1)
    # Vectorised NaN-column removal (replaces O(n*m) Python loop)
    good_cols = ~np.isnan(features).any(axis=0)
    features = features[:, good_cols]
    feature_names = feature_names[good_cols]
    # Remove hardcoded correlated features
    corr = {"Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "MaxAbsEStateIndex", "ExactMolWt", "NumHeteroatoms"}
    keep = np.array([fn not in corr for fn in feature_names])
    return features[:, keep], feature_names[keep]


def _compute_descriptors_for_smiles(smi):
    """Compute all RDKit 2D descriptors for one SMILES (picklable worker)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors as _Desc

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (smi, None)
    try:
        vals = [fn(mol) for _, fn in _Desc.descList]
        return (smi, vals)
    except Exception:
        return (smi, None)


def _cache_path(group_name):
    """Return the .npz path for a cached feature group."""
    return os.path.join(DATA_DIR, f"feature_cache_{group_name}.npz")


def _save_feature_cache(path, smiles_list, features_2d, feature_names):
    """Persist feature cache to disk as a compressed .npz."""
    np.savez_compressed(
        path,
        smiles=np.array(smiles_list, dtype=object),
        features=features_2d,
        feature_names=feature_names,
    )
    print(f"  Saved feature cache to {path}")


def _load_feature_cache(path):
    """Load feature cache from disk. Returns (cache_dict, feature_names)."""
    data = np.load(path, allow_pickle=True)
    smiles = data["smiles"]
    features = data["features"]
    feature_names = data["feature_names"]
    cache = {smi: features[i] for i, smi in enumerate(smiles)}
    print(f"  Loaded feature cache from {path} ({len(cache)} molecules)")
    return cache, feature_names


def _featurize_group(smiles_list, group_name=None):
    """RDKit-descriptor featurisation with disk caching.

    Postprocessing (log-Ipc, NaN-column removal, correlated-feature removal)
    is applied once on ALL unique molecules so that every split shares the
    exact same feature columns.  The cache stores *clean* features.

    If a cache file exists for *group_name* **and** it covers all requested
    SMILES, load from disk.  Otherwise compute in parallel and save.
    """
    n_raw_descs = len(Descriptors.descList)

    # Try loading from cache first
    if group_name is not None:
        cp = _cache_path(group_name)
        if os.path.exists(cp):
            cached, fn = _load_feature_cache(cp)
            # Detect stale cache saved before postprocessing fix
            if len(fn) == n_raw_descs:
                print("  Stale (raw) cache detected — recomputing ...")
            elif all(s in cached for s in smiles_list):
                return cached, fn
            else:
                print("  Cache incomplete — recomputing ...")

    feature_names = np.array([f[0] for f in Descriptors.descList])
    raw_cache = {}
    with Pool(N_CPUS) as pool:
        results = list(
            tqdm(
                pool.imap(_compute_descriptors_for_smiles, smiles_list, chunksize=256),
                total=len(smiles_list),
                desc="  Descriptors (parallel)",
                unit="mol",
            )
        )
    for smi, vals in results:
        if vals is not None:
            raw_cache[smi] = np.asarray(vals, dtype=np.float64)

    # ---------- postprocess on ALL molecules at once ----------
    ordered_smi = [s for s in smiles_list if s in raw_cache]
    features_2d = np.vstack([raw_cache[s] for s in ordered_smi])
    features_2d, clean_names = _fast_postprocess(features_2d, feature_names)
    cache = {smi: features_2d[i] for i, smi in enumerate(ordered_smi)}

    # Persist to disk for next run
    if group_name is not None and cache:
        _save_feature_cache(_cache_path(group_name), ordered_smi, features_2d, clean_names)
    return cache, clean_names


def _build_split_features(dataset, clean_cache, train_mean, train_std, split_name=""):
    """Assemble feature matrix + normalised labels from cached *clean* features.

    No per-split postprocessing — that was already done globally in
    ``_featurize_group`` so every split shares the same feature columns.
    Logs a warning when molecules are dropped due to cache misses.
    """
    rows, labels = [], []
    n_missing = 0
    for smi, target in dataset:
        if smi in clean_cache:
            rows.append(clean_cache[smi])
            labels.append(target)
        else:
            n_missing += 1
    if n_missing:
        pct = 100 * n_missing / max(len(dataset), 1)
        print(f"    [!] {split_name}: {n_missing}/{len(dataset)} molecules dropped (cache miss, {pct:.1f}%)")
    if not rows:
        return np.empty((0, 0)), np.empty(0)
    features = np.vstack(rows)
    labels = np.array(labels, dtype=np.float64)
    labels = (labels - train_mean) / train_std
    return features, labels


def _slice_dataset(dataset, limit):
    """Return the first ``limit`` rows from a dataset-like iterable."""
    if limit is None or len(dataset) <= limit:
        return dataset
    return [dataset[i] for i in range(limit)]


# ===== Step 1: Ensure QM9 per-property CSVs exist ==========================
def ensure_qm9_property_csvs():
    """Download QM9 from deepchem S3 and create per-property CSVs."""
    t0_step = time.time()
    needed = [p for p in QM9_PROPS if not os.path.exists(os.path.join(DATA_DIR, f"qm9_{p}.csv"))]
    if not needed:
        print(f"[Step 1] {_ts()} All QM9 per-property CSVs already exist – skipping.")
        return

    print(f"[Step 1] {_ts()} Need CSVs for: {needed}")

    # 1a. Download tar if absent
    if not os.path.exists(QM9_TAR):
        print(f"  Downloading {DEEPCHEM_S3_URL} ...")
        subprocess.check_call(
            ["curl", "-L", "-o", QM9_TAR, DEEPCHEM_S3_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"  Downloaded ({os.path.getsize(QM9_TAR)/1e6:.1f} MB)")

    # 1b. Extract SDF + CSV from tar
    if not os.path.exists(QM9_SDF) or not os.path.exists(QM9_CSV):
        print("  Extracting gdb9.tar.gz ...")
        with tarfile.open(QM9_TAR, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("  Extracted gdb9.sdf and gdb9.sdf.csv")

    # 1c. Build SMILES from SDF (the CSV has no SMILES column)
    print("  Parsing SMILES from SDF (this takes ~1 min for 132k molecules) ...")
    supplier = Chem.SDMolSupplier(QM9_SDF, removeHs=True)
    smiles_list = []
    for mol in tqdm(supplier, desc="  SDF→SMILES", unit="mol"):
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))
        else:
            smiles_list.append(None)
    print(f"  Parsed {sum(s is not None for s in smiles_list)} valid molecules")

    # 1d. Read property CSV
    with open(QM9_CSV) as f:
        csv_lines = f.read().splitlines()
    csv_header = csv_lines[0].split(",")
    csv_rows = [line.split(",") for line in csv_lines[1:]]

    # 1e. Write per-property CSVs
    for prop in needed:
        col_idx = csv_header.index(prop)
        out_path = os.path.join(DATA_DIR, f"qm9_{prop}.csv")
        n_written = 0
        with open(out_path, "w") as f:
            f.write(f"smiles,{prop}\n")
            for smi, row in zip(smiles_list, csv_rows):
                if smi is not None:
                    f.write(f"{smi},{row[col_idx]}\n")
                    n_written += 1
        print(f"  Wrote {out_path} ({n_written} molecules)")
    print(f"[Step 1] Done ({_elapsed(t0_step)})")  # noqa: F821 (t0_step defined above)


# ===== Step 2: Ensure 10k split CSV exists ==================================
def ensure_10k_splits():
    t0_step = time.time()
    if os.path.exists(TENK_SPLITS_FILE):
        print(f"[Step 2] {_ts()} 10k splits CSV already exists – skipping.")
        return
    print(f"[Step 2] {_ts()} Generating 10k OOD splits ...")
    generate_splits_10k(DATA_DIR, "10k_data_with_ood_splits.csv")
    print(f"  Wrote {TENK_SPLITS_FILE} ({_elapsed(t0_step)})")


# ===== Step 3: Generate QM9 OOD splits with KDE ============================
def generate_qm9_splits():
    """
    Re-implementation of prepare_splits_qm9 with tqdm progress bars on
    the heavy loops (InChI computation and KDE scoring).
    """
    t0_step3 = time.time()
    if os.path.exists(QM9_SPLITS_FILE):
        print(f"[Step 3] {_ts()} QM9 splits CSV already exists – skipping.")
        return

    print(f"[Step 3] {_ts()} Generating QM9 OOD splits (KDE-based) ...")
    num_ood_samples = 10000

    property_files = [os.path.join(DATA_DIR, f"qm9_{p}.csv") for p in QM9_PROPS]
    dataframe = {}  # smiles → {prop_val, prop_score, prop_ood, ...}

    # ---- Parallel InChI pre-computation (uses all CPUs) ----
    print("  Pre-computing InChI for all molecules in parallel ...")
    with open(property_files[0]) as f:
        _first_lines = f.read().splitlines()[1:]
    _all_smi = [line.split(",")[0] for line in _first_lines]
    with Pool(N_CPUS) as pool:
        _all_inchi = list(
            tqdm(
                pool.imap(_smiles_to_inchi, _all_smi, chunksize=500),
                total=len(_all_smi),
                desc="  InChI (parallel)",
                unit="mol",
            )
        )
    _inchi_lookup = dict(zip(_all_smi, _all_inchi))
    del _first_lines, _all_smi, _all_inchi

    for pf in property_files:
        # Reset seed per property to match original prepare_splits_qm9.py
        random.seed(42)
        prop_name = os.path.basename(pf).replace(".csv", "")  # e.g. "qm9_alpha"
        print(f"\n  --- Processing {prop_name} ---")

        with open(pf) as f:
            lines = f.read().splitlines()[1:]  # skip header

        # 3a. Collect property values (InChI from pre-computed lookup)
        print(f"  Reading {len(lines)} rows ...")
        for line in lines:
            smi, val = line.split(",")
            if smi not in dataframe:
                dataframe[smi] = {
                    prop_name: float(val),
                    "inchi": _inchi_lookup.get(smi, _smiles_to_inchi(smi)),
                }
            else:
                dataframe[smi][prop_name] = float(val)

        # Ordered lists for this property
        smiles_list = list(dataframe.keys())
        values = np.array([dataframe[s][prop_name] for s in smiles_list], dtype=np.float64).reshape(-1, 1)

        # 3b. KDE fit + score (scipy gaussian_kde is vectorised and
        #     orders of magnitude faster than sklearn for 1-D data)
        print(f"  Fitting KDE (n={len(values)}) ...")
        t0 = time.time()
        vals_1d = values.flatten()
        kde = gaussian_kde(vals_1d, bw_method="scott")
        scores = kde.evaluate(vals_1d)
        print(f"  KDE fit+score done in {time.time()-t0:.1f}s")

        # 3c. Select OOD (lowest-density tail)
        ood_indices = set(np.argpartition(scores, num_ood_samples)[:num_ood_samples])

        for i, smi in enumerate(smiles_list):
            dataframe[smi][prop_name + "_score"] = float(np.exp(scores[i]))
            if i in ood_indices:
                dataframe[smi][prop_name + "_ood"] = 1
                dataframe[smi][prop_name + "_train"] = 0
                dataframe[smi][prop_name + "_iid"] = 0
            else:
                dataframe[smi][prop_name + "_ood"] = 0
                if random.random() < 0.05:
                    dataframe[smi][prop_name + "_train"] = 0
                    dataframe[smi][prop_name + "_iid"] = 1
                else:
                    dataframe[smi][prop_name + "_train"] = 1
                    dataframe[smi][prop_name + "_iid"] = 0

        print(
            f"  {prop_name}: {len(ood_indices)} OOD, "
            f"{sum(1 for s in smiles_list if dataframe[s][prop_name+'_iid']==1)} ID-test, "
            f"{sum(1 for s in smiles_list if dataframe[s][prop_name+'_train']==1)} train"
        )

    # 3e. Write combined splits CSV
    print("\n  Writing combined splits CSV ...")
    prop_names = [os.path.basename(pf).replace(".csv", "") for pf in property_files]
    with open(QM9_SPLITS_FILE, "w") as f:
        header_parts = ["smiles"]
        for pn in prop_names:
            header_parts.extend([pn, pn + "_score", pn + "_ood", pn + "_train", pn + "_iid"])
        f.write(",".join(header_parts) + "\n")

        for smi in dataframe:
            parts = [smi]
            for pn in prop_names:
                parts.extend(
                    [
                        f"{dataframe[smi][pn]}",
                        f"{dataframe[smi][pn+'_score']}",
                        f"{dataframe[smi][pn+'_ood']}",
                        f"{dataframe[smi][pn+'_train']}",
                        f"{dataframe[smi][pn+'_iid']}",
                    ]
                )
            parts.append(dataframe[smi]["inchi"])
            f.write(",".join(parts) + "\n")

    print(f"  Wrote {QM9_SPLITS_FILE} ({len(dataframe)} molecules, {_elapsed(t0_step3)} total)")


# ===== Step 4: Train models and report metrics ==============================


def binned_r2(true, pred, train_median):
    """
    Binned R² as described in the BOOM paper: split OOD into lower tail
    (true < train_median) and upper tail (true >= train_median), compute R²
    within each bin, and return the average.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    upper = ~lower
    r2_vals = []
    for mask, name in [(lower, "lower"), (upper, "upper")]:
        n = int(mask.sum())
        if n >= 2:
            r2_val = r2_score(true[mask], pred[mask])
            r2_vals.append(r2_val)
            print(f"    {name} tail: n={n}, R²={r2_val:.4f}")
        else:
            print(f"    Warning: {name} tail has <2 samples, skipping")
    return float(np.mean(r2_vals)) if r2_vals else float("nan")


def corr_r2(true, pred):
    """Square of Pearson correlation coefficient (ρ²).

    This is sometimes reported as R² in the literature, but differs from the
    coefficient of determination when predictions are biased.  ρ² is always
    in [0, 1]; it ignores systematic offset or scale errors.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    if len(true) < 2:
        return float("nan")
    rho = np.corrcoef(true, pred)[0, 1]
    return float(rho**2)


def binned_corr_r2(true, pred, train_median):
    """Binned ρ²: same tail-split as binned_r2 but using corr_r2 per bin."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    upper = ~lower
    vals = []
    for mask, name in [(lower, "lower"), (upper, "upper")]:
        n = int(mask.sum())
        if n >= 2:
            val = corr_r2(true[mask], pred[mask])
            vals.append(val)
            print(f"    {name} tail ρ²: n={n}, ρ²={val:.4f}")
        else:
            print(f"    Warning: {name} tail has <2 samples, skipping ρ²")
    return float(np.mean(vals)) if vals else float("nan")


# ===== Model definitions ====================================================
# Chemprop hyperparameters (MPNN, trained via PyTorch Lightning)
CHEMPROP_PARAMS = {
    "d_h": 300,  # hidden dim for message passing
    "depth": 3,  # message-passing depth
    "ffn_hidden_dim": 300,  # FFN hidden dim
    "ffn_n_layers": 2,  # FFN layers
    "dropout": 0.0,
    "max_lr": 1e-3,
    "batch_norm": True,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 10,  # early-stopping patience
}

# Descriptor-based models.
# Factories take a *seed* so the whole pipeline can be re-run across several
# seeds (the final multi-seed analysis) without touching call sites.
DESCRIPTOR_MODELS = {
    "RF": lambda seed=DEFAULT_SEED: RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=N_CPUS,
        random_state=seed,
        verbose=0,
    ),
    "ElasticNet": lambda seed=DEFAULT_SEED: ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=30,
        cv=3,
        max_iter=10000,
        selection="random",
        n_jobs=N_CPUS,
        random_state=seed,
    ),
    "XGBoost": lambda seed=DEFAULT_SEED: XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=N_CPUS,
        random_state=seed,
    ),
}

# Ordered list of all model names (Chemprop first, then descriptor-based)
ALL_MODEL_NAMES = ["Chemprop", *DESCRIPTOR_MODELS]


# ===== Chemprop (MPNN) training helper =======================================


def _train_chemprop(train_ds, id_ds, ood_ds, seed=DEFAULT_SEED):
    """Train a chemprop MPNN on SMILES and return (id_pred, ood_pred) in
    original scale.  Uses PyTorch Lightning with CPU, early stopping,
    and the RegressionFFN output transform for automatic unscaling."""
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t_cp = time.time()
    p = CHEMPROP_PARAMS.copy()

    def _make_datapoints(smiles_dataset):
        dps = []
        ys = []
        for smi, target in smiles_dataset:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                dps.append(chemprop_data.MoleculeDatapoint(mol, y=np.array([target])))
                ys.append(float(target))
        return dps, np.array(ys, dtype=np.float64)

    print(f"    {_ts()} Building datapoints (SMILES → molecule objects) ...")
    train_dps, _ = _make_datapoints(train_ds)
    id_dps, id_true = _make_datapoints(id_ds)
    ood_dps, ood_true = _make_datapoints(ood_ds)
    print(f"    Datapoints: train={len(train_dps)}, id={len(id_dps)}, ood={len(ood_dps)}  ({_elapsed(t_cp)})")

    if CHEMPROP_SMOKE_TEST:
        # Tiny model + tiny subsets to validate pipeline wiring quickly.
        p.update(
            {
                "d_h": 64,
                "depth": 2,
                "ffn_hidden_dim": 64,
                "ffn_n_layers": 1,
                "batch_size": 1024,
                "max_epochs": 2,
                "patience": 1,
            }
        )
        max_train = SMOKE_TRAIN_MAX
        max_eval = SMOKE_EVAL_MAX
        train_dps = train_dps[:max_train]
        id_dps = id_dps[:max_eval]
        ood_dps = ood_dps[:max_eval]
        id_true = id_true[:max_eval]
        ood_true = ood_true[:max_eval]
        print("    [Chemprop smoke-test] " f"train={len(train_dps)}, id={len(id_dps)}, ood={len(ood_dps)}")

    if len(train_dps) < 2:
        raise RuntimeError("Chemprop needs at least 2 valid training molecules.")

    # 90/10 train / val split for early stopping
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_dps))
    val_n = max(1, len(train_dps) // 10)
    val_dps = [train_dps[i] for i in idx[:val_n]]
    tr_dps = [train_dps[i] for i in idx[val_n:]]

    train_dataset = chemprop_data.MoleculeDataset(tr_dps)
    val_dataset = chemprop_data.MoleculeDataset(val_dps)
    id_dataset = chemprop_data.MoleculeDataset(id_dps)
    ood_dataset = chemprop_data.MoleculeDataset(ood_dps)

    # Normalise targets — chemprop's scaler is the sole normalisation.
    # output_transform will reverse this, so predictions are in original scale.
    scaler = train_dataset.normalize_targets()
    val_dataset.normalize_targets(scaler)

    _nw = 4  # benchmark: num_workers=4 gives 2.7x faster batch loading vs 0 (42ms vs 114ms per batch)
    train_loader = chemprop_data.build_dataloader(
        train_dataset,
        batch_size=p["batch_size"],
        shuffle=True,
        num_workers=_nw,
    )
    val_loader = chemprop_data.build_dataloader(
        val_dataset,
        batch_size=p["batch_size"],
        shuffle=False,
        num_workers=_nw,
    )
    id_loader = chemprop_data.build_dataloader(
        id_dataset,
        batch_size=p["batch_size"],
        shuffle=False,
        num_workers=_nw,
    )
    ood_loader = chemprop_data.build_dataloader(
        ood_dataset,
        batch_size=p["batch_size"],
        shuffle=False,
        num_workers=_nw,
    )

    # Build MPNN for regression
    mp = chemprop_nn.BondMessagePassing(
        d_h=p["d_h"],
        depth=p["depth"],
        dropout=p["dropout"],
    )
    agg = chemprop_nn.MeanAggregation()
    output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = chemprop_nn.RegressionFFN(
        n_tasks=1,
        input_dim=p["d_h"],
        hidden_dim=p["ffn_hidden_dim"],
        n_layers=p["ffn_n_layers"],
        dropout=p["dropout"],
        output_transform=output_transform,
    )
    mpnn = chemprop_models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=p["batch_norm"],
        max_lr=p["max_lr"],
    )
    n_params = sum(par.numel() for par in mpnn.parameters())
    print(f"    MPNN: {n_params:,} parameters")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=p["patience"],
        mode="min",
        verbose=False,
    )
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    print(f"    Lightning accelerator: {accelerator}")
    t_fit = time.time()
    print(f"    {_ts()} Starting trainer.fit ...")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=1,
        max_epochs=p["max_epochs"],
        callbacks=[early_stopping],
        limit_train_batches=20 if CHEMPROP_SMOKE_TEST else 1.0,
        limit_val_batches=5 if CHEMPROP_SMOKE_TEST else 1.0,
        num_sanity_val_steps=0 if CHEMPROP_SMOKE_TEST else 2,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    stopped_epoch = trainer.current_epoch + 1
    print(f"    Stopped at epoch {stopped_epoch}/{p['max_epochs']}  (training: {_elapsed(t_fit)})")

    # Predict — output_transform automatically unscales to original target scale
    t_pred = time.time()
    print(f"    {_ts()} Running predictions ...")
    with torch.inference_mode():
        id_preds = trainer.predict(mpnn, id_loader)
        ood_preds = trainer.predict(mpnn, ood_loader)

    id_pred = torch.cat(id_preds, dim=0).detach().cpu().numpy().flatten()
    ood_pred = torch.cat(ood_preds, dim=0).detach().cpu().numpy().flatten()
    print(f"    Predictions done  (inference: {_elapsed(t_pred)}, total chemprop: {_elapsed(t_cp)})")

    # output_transform unscales predictions to original target scale.
    return id_true, id_pred, ood_true, ood_pred


# ===== Incremental results persistence =======================================
# The active seed is encoded in the FILENAME only (not inside the JSON), e.g.
# results_incremental_seed42.json / results_incremental_smoke_seed42.json.


def _results_json_path():
    tag = "_smoke" if CHEMPROP_SMOKE_TEST else ""
    return os.path.join(SCRIPT_DIR, f"results_incremental{tag}_seed{RUN_SEED}.json")


def _save_results_json(results):
    """Persist current results dict to JSON (called after each endpoint)."""
    path = _results_json_path()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved intermediate results to {path}]")


def _load_results_json():
    """Load previously saved results, if any."""
    path = _results_json_path()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Count how many endpoints are covered
        n = sum(len(v) for v in data.values())
        print(f"  Loaded {n} model×endpoint results from {path}")
        return data
    return None


# ===== Structure-based (UMAP) OOD =========================================
# This is a SECOND, independent OOD analysis based on chemical structure
# clusters.  It never touches the property-value (KDE) OOD code path above;
# results are stored under distinct keys ("struct_ood_*") alongside them.


def _ensure_struct_split(group_name, unique_smiles):
    """Return {smiles: struct_ood (0/1)} for a group, generating & caching once."""
    tag = "_smoke" if CHEMPROP_SMOKE_TEST else ""
    csv_path = os.path.join(DATA_DIR, f"umap_splits_{group_name}{tag}.csv")
    if not os.path.exists(csv_path):
        figures_dir = os.path.join(SCRIPT_DIR, "figures")
        generate_umap_structure_split(
            unique_smiles,
            csv_path,
            figures_dir,
            group_name,
            n_clusters=UMAP_N_CLUSTERS,
            ood_frac=UMAP_OOD_FRAC,
            seed=UMAP_SEED,
            max_n=UMAP_SMOKE_MAX_N if CHEMPROP_SMOKE_TEST else None,
        )
    else:
        print(f"  [umap:{group_name}] using cached split {os.path.basename(csv_path)}")
    return load_umap_structure_split(csv_path)


def _descriptor_struct_predict(model_name, seed, tr_X, tr_y, ev_X, mean, std):
    """Fit one descriptor model on the structure-train split and predict ev_X.

    Mirrors the feature handling of the property-OOD path (RF: raw features;
    ElasticNet: scale -> SelectKBest -> degree-2 interactions) but fits every
    transform on the structure-train split so nothing leaks from the KDE path.
    """
    model = DESCRIPTOR_MODELS[model_name](seed)
    if model_name == "ElasticNet":
        scaler = StandardScaler().fit(tr_X)
        tr_s = scaler.transform(tr_X)
        ev_s = scaler.transform(ev_X)
        k = min(20 if CHEMPROP_SMOKE_TEST else 80, tr_s.shape[1])
        selector = SelectKBest(f_regression, k=k).fit(tr_s, tr_y)
        tr_sel = selector.transform(tr_s)
        ev_sel = selector.transform(ev_s)
        if CHEMPROP_SMOKE_TEST:
            tr_p, ev_p = tr_sel, ev_sel
        else:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            tr_p = poly.fit_transform(tr_sel)
            ev_p = poly.transform(ev_sel)
        model.fit(tr_p, tr_y)
        return model.predict(ev_p) * std + mean
    model.fit(tr_X, tr_y)
    return model.predict(ev_X) * std + mean


def _run_structure_ood(prop, label, struct_membership, value_map, clean_cache, results, seed=DEFAULT_SEED):
    """Train each model on the structure-train split and evaluate on the
    structure-OOD split, recording plain (unbinned) R^2 and RMSE under
    'struct_ood_r2' / 'struct_ood_rmse' for every model."""
    t0 = time.time()
    print(f"  {_ts()} Structure-OOD (UMAP) pass ...")

    train_ds = [(s, value_map[s]) for s, is_ood in struct_membership.items() if is_ood == 0 and s in value_map]
    ood_ds = [(s, value_map[s]) for s, is_ood in struct_membership.items() if is_ood == 1 and s in value_map]

    if CHEMPROP_SMOKE_TEST:
        train_ds = _slice_dataset(train_ds, SMOKE_TRAIN_MAX)
        ood_ds = _slice_dataset(ood_ds, SMOKE_EVAL_MAX)

    if len(train_ds) < 2 or len(ood_ds) < 2:
        print(f"    [!] structure split too small for {prop!r} (train={len(train_ds)}, ood={len(ood_ds)}); skipping")
        return

    tr_targets = np.array([t for _, t in train_ds], dtype=np.float64)
    mean = float(tr_targets.mean())
    std = float(tr_targets.std())
    if std < 1e-10:
        print(f"    [!] struct train_std ~ 0 for {prop!r}; skipping structure-OOD")
        return

    tr_X, tr_y = _build_split_features(train_ds, clean_cache, mean, std, "struct_train")
    ood_X, ood_y = _build_split_features(ood_ds, clean_cache, mean, std, "struct_ood")
    ood_true = ood_y * std + mean
    print(f"    Sizes: struct_train={len(tr_y)}, struct_ood={len(ood_y)}")

    def _store(model_name, ood_pred):
        r2 = r2_score(ood_true, ood_pred)
        rmse = root_mean_squared_error(ood_true, ood_pred)
        results[model_name].setdefault(prop, {})
        results[model_name][prop]["struct_ood_r2"] = r2
        results[model_name][prop]["struct_ood_rmse"] = rmse
        print(f"    {model_name:10s} struct-OOD  R²={r2:.4f}  RMSE={rmse:.4f}")

    # Chemprop (SMILES). _train_chemprop returns (id, id_pred, ood, ood_pred);
    # we pass the OOD set as the "id" argument too and use only the OOD output.
    _, _, cp_ood_true, cp_ood_pred = _train_chemprop(train_ds, ood_ds, ood_ds, seed=seed)
    r2 = r2_score(cp_ood_true, cp_ood_pred)
    rmse = root_mean_squared_error(cp_ood_true, cp_ood_pred)
    results["Chemprop"].setdefault(prop, {})
    results["Chemprop"][prop]["struct_ood_r2"] = r2
    results["Chemprop"][prop]["struct_ood_rmse"] = rmse
    print(f"    {'Chemprop':10s} struct-OOD  R²={r2:.4f}  RMSE={rmse:.4f}")

    # Descriptor models.
    for model_name in DESCRIPTOR_MODELS:
        ood_pred = _descriptor_struct_predict(model_name, seed, tr_X, tr_y, ood_X, mean, std)
        _store(model_name, ood_pred)

    print(f"    (structure-OOD pass: {_elapsed(t0)})")


def run_all_models(start_from=None):
    t_total = time.time()
    print(f"\n[Step 4] {_ts()} Training models for each endpoint ...")
    print(f"  Models: {', '.join(ALL_MODEL_NAMES)}\n")

    # Group endpoints by underlying dataset so each molecule is featurised once
    tenk_eps = [(p, lb) for p, lb in ENDPOINTS if p in ("density", "hof")]
    qm9_eps = [(p, lb) for p, lb in ENDPOINTS if p not in ("density", "hof")]

    # results[model_name] = {prop: {"id": ..., "ood": ...}}
    # Load previous partial results if available
    prev = _load_results_json()
    results = prev if prev else {name: {} for name in ALL_MODEL_NAMES}
    # Ensure every model key exists (handles adding new models to an old JSON)
    for name in ALL_MODEL_NAMES:
        results.setdefault(name, {})

    # --start-from: skip endpoints until we reach the requested one
    skip = start_from is not None

    for group_name, eps in [("10k", tenk_eps), ("QM9", qm9_eps)]:
        # ---- collect all unique SMILES in this group ----
        all_datasets = {}
        unique_smiles_set = set()
        for prop, _label in eps:
            for split in ("train", "id", "ood"):
                ds = SMILESDataset(prop, split)
                all_datasets[(prop, split)] = ds
                unique_smiles_set.update(s for s, _ in ds)
        unique_smiles = sorted(unique_smiles_set)

        # ---- featurise ALL unique molecules once (with disk cache) ----
        print(f"[{group_name}] {_ts()} Featurising {len(unique_smiles)} unique molecules ...")
        t0 = time.time()
        clean_cache, _feat_names = _featurize_group(unique_smiles, group_name=group_name)
        print(
            f"  Done in {_elapsed(t0)} "
            f"({len(clean_cache)}/{len(unique_smiles)} valid, "
            f"n_features={len(_feat_names)}, peak mem: {_mem_mb()})\n"
        )

        # ---- structure-based (UMAP) OOD split for this group (property-independent) ----
        struct_membership = _ensure_struct_split(group_name, unique_smiles)
        # ---- train models per endpoint using cached features ----
        for prop, label in eps:
            if skip:
                if prop == start_from:
                    skip = False
                    print(f"--- {label} --- (resuming from here)")
                else:
                    print(f"--- {label} --- SKIPPED (--start-from {start_from})")
                    continue
            else:
                print(f"\n--- {label} ---  {_ts()}")
            t0_ep = time.time()

            train_ds = all_datasets[(prop, "train")]
            id_ds = all_datasets[(prop, "id")]
            ood_ds = all_datasets[(prop, "ood")]

            if CHEMPROP_SMOKE_TEST:
                train_ds = _slice_dataset(train_ds, SMOKE_TRAIN_MAX)
                id_ds = _slice_dataset(id_ds, SMOKE_EVAL_MAX)
                ood_ds = _slice_dataset(ood_ds, SMOKE_EVAL_MAX)

            # Pre-compute train mean/std once
            train_targets = np.array([t for _, t in train_ds], dtype=np.float64)
            train_mean = float(train_targets.mean())
            train_std = float(train_targets.std())

            # Log target statistics; guard against zero std (would cause NaN in normalisation)
            print(
                f"  Target stats: mean={train_mean:.4g}, std={train_std:.4g}, "
                f"min={train_targets.min():.4g}, max={train_targets.max():.4g}"
            )
            if train_std < 1e-10:
                print(f"  [!] train_std ~ 0 for {prop!r} — normalisation will produce NaN/Inf, skipping endpoint")
                continue

            t0_feat = time.time()
            print(f"  Building features from cache " f"(train={len(train_ds)}, id={len(id_ds)}, ood={len(ood_ds)}) ...")
            train_X, train_y = _build_split_features(train_ds, clean_cache, train_mean, train_std, "train")
            id_X, id_y = _build_split_features(id_ds, clean_cache, train_mean, train_std, "id")
            ood_X, ood_y = _build_split_features(ood_ds, clean_cache, train_mean, train_std, "ood")
            print(f"  Effective sizes after cache lookup: train={len(train_y)}, id={len(id_y)}, ood={len(ood_y)}")

            assert train_X.shape[1] == id_X.shape[1] == ood_X.shape[1], (
                f"Feature dim mismatch: train={train_X.shape[1]}, " f"id={id_X.shape[1]}, ood={ood_X.shape[1]}"
            )

            # Compute train median in original scale for binning
            train_med = float(np.median(train_y * train_std + train_mean))

            # Denormalise true labels once (shared across models)
            id_true = id_y * train_std + train_mean
            ood_true = ood_y * train_std + train_mean

            # Linear models benefit from feature scaling; RF is scale-invariant.
            # Fit scaler once on train, apply to all splits.
            scaler = StandardScaler().fit(train_X)
            train_X_scaled = scaler.transform(train_X)
            id_X_scaled = scaler.transform(id_X)
            ood_X_scaled = scaler.transform(ood_X)

            # ElasticNet normally uses degree-2 interaction features.
            # In smoke mode, skip the expansion entirely to keep memory tiny.
            n_feats = train_X_scaled.shape[1]
            _POLY_K = 20 if CHEMPROP_SMOKE_TEST else 80
            k_actual = min(_POLY_K, n_feats)
            selector = SelectKBest(f_regression, k=k_actual)
            selector.fit(train_X_scaled, train_y)
            train_X_selected = selector.transform(train_X_scaled)
            id_X_selected = selector.transform(id_X_scaled)
            ood_X_selected = selector.transform(ood_X_scaled)
            if CHEMPROP_SMOKE_TEST:
                train_X_poly = train_X_selected
                id_X_poly = id_X_selected
                ood_X_poly = ood_X_selected
            else:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                train_X_poly = poly.fit_transform(train_X_selected)
                id_X_poly = poly.transform(id_X_selected)
                ood_X_poly = poly.transform(ood_X_selected)
            poly_mem_gb = train_X_poly.nbytes / 1e9
            print(
                f"  Poly features (top-{k_actual}): "
                f"{n_feats} → {k_actual} → "
                f"{train_X_poly.shape[1]} "
                f"({poly_mem_gb:.1f} GB train matrix, "
                f"feature-build: {_elapsed(t0_feat)}, peak mem: {_mem_mb()})"
            )

            # ---- Chemprop (MPNN): operates on SMILES directly ----
            t0 = time.time()
            print(f"  {_ts()} Training Chemprop (MPNN) ...")
            cp_id_true, cp_id_pred, cp_ood_true, cp_ood_pred = _train_chemprop(
                train_ds,
                id_ds,
                ood_ds,
                seed=RUN_SEED,
            )
            cp_id_r2 = r2_score(cp_id_true, cp_id_pred)
            cp_ood_r2_binned = binned_r2(cp_ood_true, cp_ood_pred, train_med)
            cp_id_r2_corr = corr_r2(cp_id_true, cp_id_pred)
            cp_ood_r2_corr_binned = binned_corr_r2(cp_ood_true, cp_ood_pred, train_med)
            cp_id_rmse = root_mean_squared_error(cp_id_true, cp_id_pred)
            cp_ood_rmse = root_mean_squared_error(cp_ood_true, cp_ood_pred)
            results["Chemprop"][prop] = {
                "id_r2": cp_id_r2,
                "ood_r2_binned": cp_ood_r2_binned,
                "id_r2_corr": cp_id_r2_corr,
                "ood_r2_corr_binned": cp_ood_r2_corr_binned,
                "id_rmse": cp_id_rmse,
                "ood_rmse": cp_ood_rmse,
            }
            print(f"    Pred range:   ID  {_pred_range_str(cp_id_true, cp_id_pred)}")
            print(f"    Pred range:   OOD {_pred_range_str(cp_ood_true, cp_ood_pred)}")
            print(f"    Train median  = {train_med:.4f}")
            print(f"    ID  R²        = {cp_id_r2:.4f}")
            print(f"    OOD R² binned = {cp_ood_r2_binned:.4f}")
            print(f"    ID  ρ²        = {cp_id_r2_corr:.4f}")
            print(f"    OOD ρ² binned = {cp_ood_r2_corr_binned:.4f}")
            print(f"    ID  RMSE      = {cp_id_rmse:.4f}")
            print(f"    OOD RMSE      = {cp_ood_rmse:.4f}")
            _flag = _flag_metrics(cp_id_r2, cp_ood_r2_binned)
            if _flag:
                print(_flag)
            print(f"    ({_elapsed(t0)}, peak mem: {_mem_mb()})")

            # ---- Descriptor-based models ----
            for model_name, model_factory in DESCRIPTOR_MODELS.items():
                t0 = time.time()
                print(f"  {_ts()} Training {model_name} ...")
                model = model_factory(RUN_SEED)

                if model_name == "ElasticNet":
                    # Scaled + degree-2 interaction features.
                    # Limit CV parallelism to avoid OOM: each worker copies
                    # the poly matrix (~3× for coordinate-descent internals).
                    _mem_per_worker = poly_mem_gb * 3
                    try:
                        import psutil as _psutil

                        _avail_gb = _psutil.virtual_memory().available / 1e9
                    except Exception:
                        _avail_gb = 24.0  # conservative fallback
                    _safe_jobs = max(1, int(_avail_gb * 0.75 / max(_mem_per_worker, 0.1)))
                    model.n_jobs = min(N_CPUS, _safe_jobs)
                    if model.n_jobs < N_CPUS:
                        print(f"    (limiting to {model.n_jobs} CV workers " f"to fit in RAM)")
                    with warnings.catch_warnings(record=True) as _caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(train_X_poly, train_y)
                    _cw_count = sum(1 for w in _caught if issubclass(w.category, ConvergenceWarning))
                    if _cw_count:
                        print(f"    [!] ElasticNet: {_cw_count} ConvergenceWarning(s) — consider increasing max_iter")
                    id_pred = model.predict(id_X_poly) * train_std + train_mean
                    ood_pred = model.predict(ood_X_poly) * train_std + train_mean
                else:
                    # RF: raw features, no scaling needed
                    model.fit(train_X, train_y)
                    id_pred = model.predict(id_X) * train_std + train_mean
                    ood_pred = model.predict(ood_X) * train_std + train_mean

                id_r2 = r2_score(id_true, id_pred)
                ood_r2_binned = binned_r2(ood_true, ood_pred, train_med)
                id_r2_corr = corr_r2(id_true, id_pred)
                ood_r2_corr_binned = binned_corr_r2(ood_true, ood_pred, train_med)
                id_rmse = root_mean_squared_error(id_true, id_pred)
                ood_rmse = root_mean_squared_error(ood_true, ood_pred)

                results[model_name][prop] = {
                    "id_r2": id_r2,
                    "ood_r2_binned": ood_r2_binned,
                    "id_r2_corr": id_r2_corr,
                    "ood_r2_corr_binned": ood_r2_corr_binned,
                    "id_rmse": id_rmse,
                    "ood_rmse": ood_rmse,
                }
                if hasattr(model, "alpha_"):
                    print(f"    Best alpha={model.alpha_:.4g}, " f"l1_ratio={model.l1_ratio_:.2f}")
                print(f"    Pred range:   ID  {_pred_range_str(id_true, id_pred)}")
                print(f"    Pred range:   OOD {_pred_range_str(ood_true, ood_pred)}")
                print(f"    Train median  = {train_med:.4f}")
                print(f"    ID  R²        = {id_r2:.4f}")
                print(f"    OOD R² binned = {ood_r2_binned:.4f}")
                print(f"    ID  ρ²        = {id_r2_corr:.4f}")
                print(f"    OOD ρ² binned = {ood_r2_corr_binned:.4f}")
                print(f"    ID  RMSE      = {id_rmse:.4f}")
                print(f"    OOD RMSE      = {ood_rmse:.4f}")
                _flag = _flag_metrics(id_r2, ood_r2_binned)
                if _flag:
                    print(_flag)
                print(f"    ({_elapsed(t0)}, peak mem: {_mem_mb()})")

            # ---- Structure-based (UMAP) OOD pass (adds struct_ood_* metrics) ----
            struct_value_map = {}
            for _split in ("train", "id", "ood"):
                for _s, _v in all_datasets[(prop, _split)]:
                    struct_value_map[_s] = _v
            _run_structure_ood(prop, label, struct_membership, struct_value_map, clean_cache, results, seed=RUN_SEED)

            print(f"  Endpoint total: {_elapsed(t0_ep)}  {_ts()}\n")
            _save_results_json(results)  # persist after each endpoint

    # Check that all endpoints are covered before final summary
    all_props = [p for p, _ in ENDPOINTS]
    missing = [(m, p) for m in ALL_MODEL_NAMES for p in all_props if p not in results.get(m, {})]
    if missing:
        print(f"\n  WARNING: {len(missing)} model×endpoint results still missing.")
        print("  Run again with --start-from to fill in gaps, ")
        print("  then run reproduce/make_heatmaps.py once all endpoints are done.")
        print("  Missing:", [(m, p) for m, p in missing[:10]], "..." if len(missing) > 10 else "")
        return

    print(f"\nDone. Total training time: {_elapsed(t_total)}")
    print("Run reproduce/make_heatmaps.py to generate heatmaps.")


# ===== Main =================================================================
def _parse_args():
    valid_endpoints = [p for p, _ in ENDPOINTS]
    parser = argparse.ArgumentParser(
        description="Reproduce BOOM Figure 2 with 4 models.",
    )
    parser.add_argument(
        "--start-from",
        choices=valid_endpoints,
        default=None,
        help=(
            "Skip endpoints before this one and resume. "
            "Existing results are loaded from results_incremental.json. "
            f"Choices: {', '.join(valid_endpoints)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Model-training seed. Written into the results FILENAME only "
            f"(results_incremental_seed<seed>.json). Default: {DEFAULT_SEED}. "
            "The UMAP + KDE splits are held fixed across seeds."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        "--chemprop-smoke-test",
        dest="smoke_test",
        action="store_true",
        help=(
            "Run a very fast sanity-check mode for all models "
            "(small subsets; Chemprop uses a smaller network; "
            "ElasticNet skips interaction expansion)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    CHEMPROP_SMOKE_TEST = args.smoke_test
    RUN_SEED = args.seed

    # Set up logging to both terminal and a results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"run_all_r2_{timestamp}.log")
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    t_script_start = time.time()
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Log file: {log_path}")
    print(f"N_CPUS={N_CPUS}  smoke-test={CHEMPROP_SMOKE_TEST}  seed={RUN_SEED}")
    print(f"Results file: {_results_json_path()}")

    # System / package info
    print(f"Python: {sys.version.split()[0]}  Platform: {platform.platform()}")
    try:
        import torch as _t
        import lightning as _l
        import chemprop as _c
        import sklearn as _sk

        print(
            f"Versions — torch={_t.__version__}  lightning={_l.__version__}"
            f"  chemprop={_c.__version__}  sklearn={_sk.__version__}"
        )
    except Exception as _e:
        print(f"  (version check skipped: {_e})")
    try:
        import psutil

        _ram = psutil.virtual_memory()
        print(f"RAM: {_ram.total/1e9:.1f} GB total, {_ram.available/1e9:.1f} GB available")
    except ImportError:
        pass
    print()

    try:
        ensure_qm9_property_csvs()  # Step 1
        ensure_10k_splits()  # Step 2
        generate_qm9_splits()  # Step 3
        run_all_models(start_from=args.start_from)  # Step 4
    finally:
        total_time = _elapsed(t_script_start)
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"\nTotal wall-clock time: {total_time}")
        print(f"Results saved to: {log_path}")
