#!/usr/bin/env python
"""
Reproduce RF R² from BOOM Figure 2 for all 10 endpoints
(Density, HoF, alpha, cv, gap, homo, lumo, mu, r2, zpve).
Also adds Chemprop (MPNN), Elastic Net (with degree-2 interaction
features), and XGBoost-linear with early stopping.

Self-contained: handles data download, CSV preparation, OOD split
generation (with progress output), feature caching, and model
training/evaluation.

Run from:  repo root
Usage:     python reproduce_parts_of_fig_2_and_add_more_models.py
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import tarfile
import time
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
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

RDLogger.logger().setLevel(RDLogger.ERROR)  # suppress InChI warnings

# ---------------------------------------------------------------------------
# SCRIPT_DIR = repo root (where this file now lives).
# DATA_DIR   = experiments/data/ (shared downloads, splits, caches).
# os.chdir(DATA_DIR) is required because SMILESDataset resolves split
# files relative to os.getcwd().
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "experiments", "data")
os.makedirs(DATA_DIR, exist_ok=True)
sys.path.insert(0, SCRIPT_DIR)
os.chdir(DATA_DIR)

from boom.data.prepare_splits_10k import generate_splits_10k  # noqa: E402
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


# ===== Configuration ========================================================
DEEPCHEM_S3_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_TAR = os.path.join(DATA_DIR, "gdb9.tar.gz")
QM9_SDF = os.path.join(DATA_DIR, "gdb9.sdf")
QM9_CSV = os.path.join(DATA_DIR, "gdb9.sdf.csv")

# 8 QM9 properties — names must be lowercase so the splits CSV columns
# match what _load_qm9_data expects (it looks for "qm9_" + target.lower()).
QM9_PROPS = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"]
# Mapping from lowercase prop name to the column name in gdb9.sdf.csv
QM9_CSV_COL = {
    "mu": "mu",
    "alpha": "alpha",
    "homo": "homo",
    "lumo": "lumo",
    "gap": "gap",
    "r2": "r2",
    "zpve": "zpve",
    "cv": "cv",
}
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

N_CPUS = 16  # available hardware: 16 CPUs, 32 GB RAM


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


def _build_split_features(dataset, clean_cache, train_mean, train_std):
    """Assemble feature matrix + normalised labels from cached *clean* features.

    No per-split postprocessing — that was already done globally in
    ``_featurize_group`` so every split shares the same feature columns.
    """
    rows, labels = [], []
    for smi, target in dataset:
        if smi in clean_cache:
            rows.append(clean_cache[smi])
            labels.append(target)
    if not rows:
        return np.empty((0, 0)), np.empty(0)
    features = np.vstack(rows)
    labels = np.array(labels, dtype=np.float64)
    labels = (labels - train_mean) / train_std
    return features, labels


# ===== Step 1: Ensure QM9 per-property CSVs exist ==========================
def ensure_qm9_property_csvs():
    """Download QM9 from deepchem S3 and create per-property CSVs."""
    needed = [p for p in QM9_PROPS if not os.path.exists(os.path.join(DATA_DIR, f"qm9_{p}.csv"))]
    if not needed:
        print("[Step 1] All QM9 per-property CSVs already exist – skipping.")
        return

    print(f"[Step 1] Need CSVs for: {needed}")

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
    from rdkit import Chem

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
        col_name = QM9_CSV_COL[prop]
        col_idx = csv_header.index(col_name)
        out_path = os.path.join(DATA_DIR, f"qm9_{prop}.csv")
        n_written = 0
        with open(out_path, "w") as f:
            f.write(f"smiles,{prop}\n")
            for smi, row in zip(smiles_list, csv_rows):
                if smi is not None:
                    f.write(f"{smi},{row[col_idx]}\n")
                    n_written += 1
        print(f"  Wrote {out_path} ({n_written} molecules)")


# ===== Step 2: Ensure 10k split CSV exists ==================================
def ensure_10k_splits():
    if os.path.exists(TENK_SPLITS_FILE):
        print("[Step 2] 10k splits CSV already exists – skipping.")
        return
    print("[Step 2] Generating 10k OOD splits ...")
    generate_splits_10k(DATA_DIR, "10k_data_with_ood_splits.csv")
    print(f"  Wrote {TENK_SPLITS_FILE}")


# ===== Step 3: Generate QM9 OOD splits with KDE ============================
def generate_qm9_splits():
    """
    Re-implementation of prepare_splits_qm9 with tqdm progress bars on
    the heavy loops (InChI computation and KDE scoring).
    """
    if os.path.exists(QM9_SPLITS_FILE):
        print("[Step 3] QM9 splits CSV already exists – skipping.")
        return

    print("[Step 3] Generating QM9 OOD splits (KDE-based) ...")
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

        # 3b. KDE fit
        print(f"  Fitting KDE (n={len(values)}) ...")
        t0 = time.time()
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(values)
        print(f"  KDE fit done in {time.time()-t0:.1f}s")

        # 3c. KDE score
        print(f"  Scoring {len(values)} samples ...")
        t0 = time.time()
        log_scores = kde.score_samples(values)
        scores = np.exp(log_scores)
        print(f"  Scoring done in {time.time()-t0:.1f}s")

        # 3d. Select OOD (lowest-density tail)
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

    print(f"  Wrote {QM9_SPLITS_FILE} ({len(dataframe)} molecules)")


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
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,  # early-stopping patience
}

# Descriptor-based models (sklearn / xgboost)
DESCRIPTOR_MODELS = {
    "RF": lambda: RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=N_CPUS,
        verbose=0,
    ),
    "ElasticNet": lambda: ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=30,
        cv=3,
        max_iter=2000,
        selection="random",
        n_jobs=N_CPUS,
        random_state=42,
    ),
    "XGB-linear": lambda: XGBRegressor(
        booster="gblinear",
        n_estimators=5000,
        learning_rate=0.01,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        n_jobs=N_CPUS,
        random_state=42,
        verbosity=0,
    ),
}

# Ordered list of all model names (Chemprop first, then descriptor-based)
ALL_MODEL_NAMES = ["Chemprop", *DESCRIPTOR_MODELS]


# ===== Chemprop (MPNN) training helper =======================================


def _train_chemprop(train_ds, id_ds, ood_ds):
    """Train a chemprop MPNN on SMILES and return (id_pred, ood_pred) in
    original scale.  Uses PyTorch Lightning with CPU, early stopping,
    and the RegressionFFN output transform for automatic unscaling."""
    p = CHEMPROP_PARAMS

    def _make_datapoints(smiles_dataset):
        dps = []
        for smi, target in smiles_dataset:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                dps.append(chemprop_data.MoleculeDatapoint(mol, y=np.array([target])))
        return dps

    train_dps = _make_datapoints(train_ds)
    id_dps = _make_datapoints(id_ds)
    ood_dps = _make_datapoints(ood_ds)

    # 90/10 train / val split for early stopping
    rng = np.random.RandomState(42)
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

    _nw = min(N_CPUS - 1, 15)  # dataloader workers
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
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="cpu",
        max_epochs=p["max_epochs"],
        callbacks=[early_stopping],
    )
    trainer.fit(mpnn, train_loader, val_loader)
    stopped_epoch = trainer.current_epoch + 1
    print(f"    Stopped at epoch {stopped_epoch}/{p['max_epochs']}")

    # Predict — output_transform automatically unscales to original target scale
    with torch.inference_mode():
        id_preds = trainer.predict(mpnn, id_loader)
        ood_preds = trainer.predict(mpnn, ood_loader)

    id_pred = torch.cat(id_preds, dim=0).numpy().flatten()
    ood_pred = torch.cat(ood_preds, dim=0).numpy().flatten()

    # output_transform unscales predictions to original target scale.
    return id_pred, ood_pred


# ===== Incremental results persistence =======================================
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results_incremental.json")


def _save_results_json(results):
    """Persist current results dict to JSON (called after each endpoint)."""
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved intermediate results to {RESULTS_JSON}]")


def _load_results_json():
    """Load previously saved results, if any."""
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        # Count how many endpoints are covered
        n = sum(len(v) for v in data.values())
        print(f"  Loaded {n} model×endpoint results from {RESULTS_JSON}")
        return data
    return None


def run_all_models(start_from=None):
    print("\n[Step 4] Training models for each endpoint ...")
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
        print(f"[{group_name}] Featurising {len(unique_smiles)} unique molecules ...")
        t0 = time.time()
        clean_cache, _feat_names = _featurize_group(unique_smiles, group_name=group_name)
        print(f"  Done in {time.time()-t0:.1f}s " f"({len(clean_cache)}/{len(unique_smiles)} valid)\n")

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
                print(f"--- {label} ---")
            t0_ep = time.time()

            train_ds = all_datasets[(prop, "train")]
            id_ds = all_datasets[(prop, "id")]
            ood_ds = all_datasets[(prop, "ood")]

            # Pre-compute train mean/std once
            train_targets = np.array([t for _, t in train_ds], dtype=np.float64)
            train_mean = float(train_targets.mean())
            train_std = float(train_targets.std())

            print(f"  Building features from cache " f"(train={len(train_ds)}, id={len(id_ds)}, ood={len(ood_ds)}) ...")
            train_X, train_y = _build_split_features(train_ds, clean_cache, train_mean, train_std)
            id_X, id_y = _build_split_features(id_ds, clean_cache, train_mean, train_std)
            ood_X, ood_y = _build_split_features(ood_ds, clean_cache, train_mean, train_std)

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

            # Degree-2 interaction features for ElasticNet.
            # Always cap with SelectKBest to keep coordinate descent fast.
            n_feats = train_X_scaled.shape[1]
            _POLY_K = 80
            k_actual = min(_POLY_K, n_feats)
            selector = SelectKBest(f_regression, k=k_actual)
            selector.fit(train_X_scaled, train_y)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            train_X_poly = poly.fit_transform(selector.transform(train_X_scaled))
            id_X_poly = poly.transform(selector.transform(id_X_scaled))
            ood_X_poly = poly.transform(selector.transform(ood_X_scaled))
            poly_mem_gb = train_X_poly.nbytes / 1e9
            print(
                f"  Poly features (top-{k_actual}): "
                f"{n_feats} → {k_actual} → "
                f"{train_X_poly.shape[1]} "
                f"({poly_mem_gb:.1f} GB)"
            )

            # XGB-linear: 90/10 validation split for early stopping
            _rng = np.random.RandomState(42)
            _shuf = _rng.permutation(len(train_y))
            _val_n = max(1, len(train_y) // 10)
            _xgb_val_idx = _shuf[:_val_n]
            _xgb_tr_idx = _shuf[_val_n:]

            # ---- Chemprop (MPNN): operates on SMILES directly ----
            t0 = time.time()
            print("  Training Chemprop (MPNN) ...")
            cp_id_pred, cp_ood_pred = _train_chemprop(
                train_ds,
                id_ds,
                ood_ds,
            )
            # True labels for chemprop: original scale, built from the raw
            # SMILES datasets (chemprop may have slightly different valid
            # molecules than the descriptor pipeline, but MolFromSmiles
            # failures are extremely rare in QM9/10k).
            cp_id_true = np.array(
                [t for s, t in id_ds if Chem.MolFromSmiles(s) is not None],
                dtype=np.float64,
            )
            cp_ood_true = np.array(
                [t for s, t in ood_ds if Chem.MolFromSmiles(s) is not None],
                dtype=np.float64,
            )
            cp_id_r2 = r2_score(cp_id_true, cp_id_pred)
            cp_ood_r2 = r2_score(cp_ood_true, cp_ood_pred)
            cp_ood_r2_binned = binned_r2(cp_ood_true, cp_ood_pred, train_med)
            cp_id_rmse = root_mean_squared_error(cp_id_true, cp_id_pred)
            cp_ood_rmse = root_mean_squared_error(cp_ood_true, cp_ood_pred)
            results["Chemprop"][prop] = {
                "id_r2": cp_id_r2,
                "ood_r2": cp_ood_r2,
                "ood_r2_binned": cp_ood_r2_binned,
                "id_rmse": cp_id_rmse,
                "ood_rmse": cp_ood_rmse,
            }
            elapsed = time.time() - t0
            print(f"    Train median  = {train_med:.4f}")
            print(f"    ID  R²        = {cp_id_r2:.4f}")
            print(f"    OOD R² (plain)= {cp_ood_r2:.4f}")
            print(f"    OOD R² binned = {cp_ood_r2_binned:.4f}")
            print(f"    ID  RMSE      = {cp_id_rmse:.4f}")
            print(f"    OOD RMSE      = {cp_ood_rmse:.4f}")
            print(f"    ({elapsed:.0f}s)")

            # ---- Descriptor-based models ----
            for model_name, model_factory in DESCRIPTOR_MODELS.items():
                t0 = time.time()
                print(f"  Training {model_name} ...")
                model = model_factory()

                if model_name == "ElasticNet":
                    # Scaled + degree-2 interaction features.
                    # Limit CV parallelism to avoid OOM: each worker copies
                    # the poly matrix (~3× for coordinate-descent internals).
                    _mem_per_worker = poly_mem_gb * 3
                    _safe_jobs = max(1, int(24.0 / max(_mem_per_worker, 0.1)))
                    model.n_jobs = min(N_CPUS, _safe_jobs)
                    if model.n_jobs < N_CPUS:
                        print(f"    (limiting to {model.n_jobs} CV workers " f"to fit in RAM)")
                    model.fit(train_X_poly, train_y)
                    id_pred = model.predict(id_X_poly) * train_std + train_mean
                    ood_pred = model.predict(ood_X_poly) * train_std + train_mean
                elif model_name == "XGB-linear":
                    # Scaled features + early stopping on 10% held-out
                    model.fit(
                        train_X_scaled[_xgb_tr_idx],
                        train_y[_xgb_tr_idx],
                        eval_set=[(train_X_scaled[_xgb_val_idx], train_y[_xgb_val_idx])],
                        verbose=False,
                    )
                    id_pred = model.predict(id_X_scaled) * train_std + train_mean
                    ood_pred = model.predict(ood_X_scaled) * train_std + train_mean
                else:
                    # RF: raw features, no scaling needed
                    model.fit(train_X, train_y)
                    id_pred = model.predict(id_X) * train_std + train_mean
                    ood_pred = model.predict(ood_X) * train_std + train_mean

                id_r2 = r2_score(id_true, id_pred)
                ood_r2_plain = r2_score(ood_true, ood_pred)
                ood_r2_binned = binned_r2(ood_true, ood_pred, train_med)
                id_rmse = root_mean_squared_error(id_true, id_pred)
                ood_rmse = root_mean_squared_error(ood_true, ood_pred)

                results[model_name][prop] = {
                    "id_r2": id_r2,
                    "ood_r2": ood_r2_plain,
                    "ood_r2_binned": ood_r2_binned,
                    "id_rmse": id_rmse,
                    "ood_rmse": ood_rmse,
                }
                elapsed = time.time() - t0
                if hasattr(model, "alpha_"):
                    print(f"    Best alpha={model.alpha_:.4g}, " f"l1_ratio={model.l1_ratio_:.2f}")
                if hasattr(model, "best_iteration"):
                    print(
                        f"    Early stopped at round "
                        f"{model.best_iteration} / "
                        f"{model.get_params()['n_estimators']}"
                    )
                print(f"    Train median  = {train_med:.4f}")
                print(f"    ID  R²        = {id_r2:.4f}")
                print(f"    OOD R² (plain)= {ood_r2_plain:.4f}")
                print(f"    OOD R² binned = {ood_r2_binned:.4f}")
                print(f"    ID  RMSE      = {id_rmse:.4f}")
                print(f"    OOD RMSE      = {ood_rmse:.4f}")
                print(f"    ({elapsed:.0f}s)")

            print(f"  Endpoint total: {time.time()-t0_ep:.0f}s\n")
            _save_results_json(results)  # persist after each endpoint

    # Check that all endpoints are covered before summary/heatmaps
    all_props = [p for p, _ in ENDPOINTS]
    missing = [(m, p) for m in ALL_MODEL_NAMES for p in all_props if p not in results.get(m, {})]
    if missing:
        print(f"\n  WARNING: {len(missing)} model×endpoint results still missing.")
        print("  Run again with --start-from to fill in gaps, ")
        print("  or use --heatmaps-only once all endpoints are done.")
        print("  Missing:", [(m, p) for m, p in missing[:10]], "..." if len(missing) > 10 else "")
        return

    # Final summary (preserve original ENDPOINTS order)
    for model_name in ALL_MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"  {model_name}")
        print("=" * 80)
        print(
            f"{'Endpoint':10s} {'ID R²':>8s}  {'OOD R²':>8s}  " f"{'OOD R²bin':>10s}  {'ID RMSE':>9s}  {'OOD RMSE':>9s}"
        )
        print("-" * 80)
        for prop, label in ENDPOINTS:
            r = results[model_name][prop]
            print(
                f"{label:10s} {r['id_r2']:8.4f}  {r['ood_r2']:8.4f}  "
                f"{r['ood_r2_binned']:10.4f}  "
                f"{r['id_rmse']:9.4f}  {r['ood_rmse']:9.4f}"
            )
        print("=" * 80)

    # ---- Heatmaps ----
    plot_heatmaps(results)


# ===== Heatmaps ==============================================================


def plot_heatmaps(results):
    """Save R², RMSE, and Binned-R² heatmaps (each as ID + OOD stacked)."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names = list(ALL_MODEL_NAMES)
    prop_labels = [label for _, label in ENDPOINTS]
    prop_keys = [prop for prop, _ in ENDPOINTS]

    n_models = len(model_names)
    n_props = len(prop_keys)
    fig_w = max(12, n_props * 1.4)
    fig_h = n_models * 1.5 * 2 + 2  # space for two subplots + titles

    def _axis_bottom(ax, ylabel, title, xlabel="Property"):
        """Move x-tick labels to bottom and set labels/title."""
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="x", rotation=0)

    # ================================================================
    # 1) R² heatmaps  (ID on top, OOD below)
    #    Color scale fixed [0, 1]; negative values get the "0" color
    #    but annotations still show the true number.
    # ================================================================
    id_r2 = np.array([[results[m][p]["id_r2"] for p in prop_keys] for m in model_names])
    ood_r2 = np.array([[results[m][p]["ood_r2"] for p in prop_keys] for m in model_names])

    fig_r2, (ax_id_r2, ax_ood_r2) = plt.subplots(2, 1, figsize=(fig_w, fig_h))

    r2_common = dict(
        annot=True,
        fmt=".3f",
        xticklabels=prop_labels,
        yticklabels=model_names,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 11},
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "R²", "shrink": 0.8},
    )

    sns.heatmap(id_r2, ax=ax_id_r2, **r2_common)
    _axis_bottom(ax_id_r2, "Model", "ID Splits  (R²)")

    sns.heatmap(ood_r2, ax=ax_ood_r2, **r2_common)
    _axis_bottom(ax_ood_r2, "Model", "OOD Splits  (R²)")

    fig_r2.tight_layout()
    r2_path = os.path.join(SCRIPT_DIR, "heatmap_r2.png")
    fig_r2.savefig(r2_path, dpi=150, bbox_inches="tight")
    plt.close(fig_r2)
    print(f"\nR² heatmap saved to {r2_path}")

    # ================================================================
    # 2) RMSE heatmaps  (ID on top, OOD below)
    #    Different properties live on wildly different scales, so a
    #    single color range is meaningless.  Instead, normalise each
    #    property-column to [0, 1]  (0 = best model, 1 = worst) and
    #    annotate with the real RMSE values.
    # ================================================================
    id_rmse = np.array([[results[m][p]["id_rmse"] for p in prop_keys] for m in model_names])
    ood_rmse = np.array([[results[m][p]["ood_rmse"] for p in prop_keys] for m in model_names])

    def _col_normalise(arr):
        """Min-max normalise each column independently to [0, 1]."""
        col_min = arr.min(axis=0, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
        return (arr - col_min) / denom

    # Build annotation arrays (strings) with the real RMSE values
    def _fmt_annot(arr):
        return np.array([[f"{v:.3f}" for v in row] for row in arr])

    fig_rmse, (ax_id_rmse, ax_ood_rmse) = plt.subplots(2, 1, figsize=(fig_w, fig_h))

    rmse_common = dict(
        xticklabels=prop_labels,
        yticklabels=model_names,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 11},
        cmap="YlGnBu_r",  # blue = low/good, yellow = high/bad
        vmin=0,
        vmax=1,
        fmt="",  # annotations are pre-formatted strings
        cbar_kws={"label": "Relative RMSE\n(per property, 0 = best)", "shrink": 0.8},
    )

    sns.heatmap(_col_normalise(id_rmse), ax=ax_id_rmse, annot=_fmt_annot(id_rmse), **rmse_common)
    _axis_bottom(ax_id_rmse, "Model", "ID Splits  (RMSE)")

    sns.heatmap(_col_normalise(ood_rmse), ax=ax_ood_rmse, annot=_fmt_annot(ood_rmse), **rmse_common)
    _axis_bottom(ax_ood_rmse, "Model", "OOD Splits  (RMSE)")

    fig_rmse.tight_layout()
    rmse_path = os.path.join(SCRIPT_DIR, "heatmap_rmse.png")
    fig_rmse.savefig(rmse_path, dpi=150, bbox_inches="tight")
    plt.close(fig_rmse)
    print(f"RMSE heatmap saved to {rmse_path}")

    # ================================================================
    # 3) R² heatmaps with BINNED OOD R²  (ID on top, OOD-binned below)
    #    Same [0, 1] color clamping as plain R².
    # ================================================================
    ood_r2_binned = np.array([[results[m][p]["ood_r2_binned"] for p in prop_keys] for m in model_names])

    fig_bin, (ax_id_bin, ax_ood_bin) = plt.subplots(2, 1, figsize=(fig_w, fig_h))

    r2b_common = dict(
        annot=True,
        fmt=".3f",
        xticklabels=prop_labels,
        yticklabels=model_names,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 11},
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "R²", "shrink": 0.8},
    )

    sns.heatmap(id_r2, ax=ax_id_bin, **r2b_common)
    _axis_bottom(ax_id_bin, "Model", "ID Splits  (R²)")

    sns.heatmap(ood_r2_binned, ax=ax_ood_bin, **r2b_common)
    _axis_bottom(ax_ood_bin, "Model", "OOD Splits  (Binned R²)")

    fig_bin.tight_layout()
    bin_path = os.path.join(SCRIPT_DIR, "heatmap_r2_binned.png")
    fig_bin.savefig(bin_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bin)
    print(f"Binned-R² heatmap saved to {bin_path}")


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
        "--heatmaps-only",
        action="store_true",
        help=("Skip all training. Load results_incremental.json and " "regenerate summary tables + heatmap PNGs."),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Set up logging to both terminal and a results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(SCRIPT_DIR, f"run_all_r2_{timestamp}.log")
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Working directory: {os.getcwd()}")
    print(f"Log file: {log_path}\n")

    try:
        if args.heatmaps_only:
            # Just load saved results and regenerate plots
            prev = _load_results_json()
            if prev is None:
                print(f"ERROR: {RESULTS_JSON} not found. Run training first.")
                sys.exit(1)
            # Print summary tables
            for model_name in ALL_MODEL_NAMES:
                print("\n" + "=" * 80)
                print(f"  {model_name}")
                print("=" * 80)
                print(
                    f"{'Endpoint':10s} {'ID R²':>8s}  {'OOD R²':>8s}  "
                    f"{'OOD R²bin':>10s}  {'ID RMSE':>9s}  {'OOD RMSE':>9s}"
                )
                print("-" * 80)
                for prop, label in ENDPOINTS:
                    r = prev.get(model_name, {}).get(prop, {})
                    if r:
                        print(
                            f"{label:10s} {r['id_r2']:8.4f}  {r['ood_r2']:8.4f}  "
                            f"{r['ood_r2_binned']:10.4f}  "
                            f"{r['id_rmse']:9.4f}  {r['ood_rmse']:9.4f}"
                        )
                    else:
                        print(f"{label:10s}  -- missing --")
                print("=" * 80)
            plot_heatmaps(prev)
        else:
            ensure_qm9_property_csvs()  # Step 1
            ensure_10k_splits()  # Step 2
            generate_qm9_splits()  # Step 3
            run_all_models(start_from=args.start_from)  # Step 4
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"\nResults saved to: {log_path}")
