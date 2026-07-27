#!/usr/bin/env python
"""Standalone GotenNet runner that ADDS a 3D-equivariant model to the BOOM
reproduction results without touching any of the existing models.

It loads the shared ``results_incremental_seed<seed>.json`` produced by
``reproduce/reproduce_parts_of_fig_2_and_add_models.py`` (the 4 descriptor /
MPNN models), trains GotenNet on the SAME splits, and writes ONLY the
``results["GotenNet"]`` sub-tree back -- every other model is preserved
byte-for-byte.

Why a separate script / environment?
------------------------------------
GotenNet (sarpaykent/GotenNet, ICLR 2025) needs the PyG C++/CUDA extension
stack (torch_scatter / torch_sparse / torch_cluster) built against a specific
torch build.  That does not coexist with the main project's torch, so this
runner lives in its own virtual-env (``.venv_goten``, torch 2.5.1+cu124) and
talks to the rest of the pipeline only through the split CSVs and the results
JSON -- no ``boom`` import, no shared Python process.

Splits (identical definitions to the main pipeline)
---------------------------------------------------
* KDE property-value OOD  -> ID / OOD from the ``*_data_with_ood_splits*.csv``.
* Structure (UMAP/HDBSCAN) OOD -> ``umap_splits_{10k,QM9}.csv`` (struct_ood).

3D geometry
-----------
Conformers are generated once per dataset group with RDKit (ETKDG embed +
MMFF optimisation) and cached to a pickle keyed by the exact SMILES string used
in the split CSVs.  QM9's original DFT geometries are NOT used -- this is a
documented approximation; the OOD *split definitions* (which is what the
benchmark compares) are identical to every other model.

Metrics (identical keys / definitions to the main pipeline)
-----------------------------------------------------------
id_r2, ood_r2_binned, id_r2_corr, ood_r2_corr_binned, id_rmse, ood_rmse,
struct_ood_r2, struct_ood_rmse.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, root_mean_squared_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from gotennet import GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff

RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# Paths.  SCRIPT_DIR = reproduce/experiments/gotennet/.
# REPRODUCE_DIR = reproduce/ (where the results JSON lives).
# DATA_DIR = reproduce/experiments/data/ (split CSVs + conformer caches).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
REPRODUCE_DIR = os.path.dirname(EXPERIMENTS_DIR)
DATA_DIR = os.path.join(EXPERIMENTS_DIR, "data")

TENK_SPLITS = os.path.join(DATA_DIR, "10k_data_with_ood_splits.csv")
QM9_SPLITS = os.path.join(DATA_DIR, "qm9_data_with_ood_splits_with_inchi.csv")

# (prop, label) -- same order/labels as the main pipeline's ENDPOINTS.
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
TENK_PROPS = {"hof", "density"}
GROUP_OF = {p: ("10k" if p in TENK_PROPS else "QM9") for p, _ in ENDPOINTS}

MODEL_NAME = "GotenNet"
CONFORMER_SEED = 42  # fixed so the conformer cache is reproducible


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


# ===== Metrics (identical definitions to the main pipeline) =================
def binned_r2(true, pred, train_median):
    """Average of R² over the lower/upper tails split at the train median."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    vals = []
    for mask in (lower, ~lower):
        if int(mask.sum()) >= 2:
            vals.append(r2_score(true[mask], pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def corr_r2(true, pred):
    """Square of the Pearson correlation coefficient (ρ²)."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    if len(true) < 2:
        return float("nan")
    rho = np.corrcoef(true, pred)[0, 1]
    return float(rho**2)


def binned_corr_r2(true, pred, train_median):
    """Average of ρ² over the lower/upper tails split at the train median."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = true < train_median
    vals = []
    for mask in (lower, ~lower):
        if int(mask.sum()) >= 2:
            vals.append(corr_r2(true[mask], pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


# ===== Split parsing (reads the exact CSVs the other models use) ============
def _load_10k_splits():
    """Return {prop: {smiles: (value, split)}} for hof and density.

    Columns: smiles,density,hof,density_score,hof_score,density_ood,
    density_train,density_iid,hof_ood,hof_train,hof_iid.
    """
    out = {"density": {}, "hof": {}}
    with open(TENK_SPLITS) as f:
        f.readline()  # header
        for line in f:
            v = line.rstrip("\n").split(",")
            if len(v) < 11:
                continue
            smiles = v[0]
            density, hof = float(v[1]), float(v[2])
            d_ood, d_train = int(v[5]), int(v[6])
            h_ood, h_train = int(v[8]), int(v[9])
            out["density"][smiles] = (density, "train" if d_train else "ood" if d_ood else "id")
            out["hof"][smiles] = (hof, "train" if h_train else "ood" if h_ood else "id")
    return out


def _load_qm9_split(target):
    """Return {smiles: (value, split)} for one QM9 target."""
    out = {}
    with open(QM9_SPLITS) as f:
        header = f.readline().rstrip("\n").split(",")
        vcol = header.index(f"qm9_{target}")
        ocol, tcol = vcol + 2, vcol + 3
        for line in f:
            v = line.rstrip("\n").split(",")
            if len(v) <= tcol:
                continue
            smiles = v[0]
            val = float(v[vcol])
            ood, train = int(v[ocol]), int(v[tcol])
            out[smiles] = (val, "train" if train else "ood" if ood else "id")
    return out


def _load_prop_split(prop):
    if prop in TENK_PROPS:
        return _load_10k_splits()[prop]
    return _load_qm9_split(prop)


def _load_struct_membership(group):
    """Return {smiles: struct_ood (0/1)} from umap_splits_<group>.csv."""
    csv_path = os.path.join(DATA_DIR, f"umap_splits_{group}.csv")
    membership = {}
    with open(csv_path) as f:
        header = f.readline().rstrip("\n").split(",")
        si, oi = header.index("smiles"), header.index("struct_ood")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= oi:
                continue
            membership[parts[si]] = int(parts[oi])
    return membership


# ===== 3D conformer generation + cache ======================================
def _embed_smiles(smiles):
    """Return (atomic_numbers[int64], positions[float32,(N,3)]) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = CONFORMER_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        # retry with random coords as a fallback
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    pos = conf.GetPositions().astype(np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return z, pos


def _build_conformer_cache(group, smiles_list, smoke):
    """Generate/reuse a {smiles: (z, pos)} pickle cache for a dataset group."""
    tag = "_smoke" if smoke else ""
    cache_path = os.path.join(DATA_DIR, f"goten_3d_{group}{tag}.pkl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    missing = [s for s in smiles_list if s not in cache]
    if missing:
        print(f"  [3d:{group}] embedding {len(missing)} new conformers " f"({len(cache)} cached) ...")
        t0 = time.time()
        for i, smi in enumerate(missing, 1):
            res = _embed_smiles(smi)
            cache[smi] = res  # may be None (recorded so we don't retry)
            if i % 2000 == 0:
                print(f"    {i}/{len(missing)}  ({_elapsed(t0)})")
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  [3d:{group}] done ({_elapsed(t0)}); cache -> {os.path.basename(cache_path)}")
    else:
        print(f"  [3d:{group}] all {len(smiles_list)} conformers cached")
    return cache


# ===== Torch model: GotenNet representation + regression head ===============
class GotenNetRegressor(nn.Module):
    def __init__(self, n_atom_basis, n_interactions, lmax, n_rbf, num_heads, cutoff):
        super().__init__()
        self.representation = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            lmax=lmax,
            n_rbf=n_rbf,
            num_heads=num_heads,
            cutoff_fn=CosineCutoff(cutoff),
        )
        self.head = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 1),
        )

    def forward(self, batch):
        h, _ = self.representation(batch)  # h: [num_nodes, n_atom_basis]
        pooled = scatter(h, batch.batch, dim=0, reduce="mean")  # [num_graphs, n_atom_basis]
        return self.head(pooled).squeeze(-1)  # [num_graphs]


# ===== Dataset assembly =====================================================
def _make_data_list(pairs, conformers, mean, std):
    """pairs: list of (smiles, value). Returns PyG Data list (skips failures)."""
    data_list = []
    for smi, val in pairs:
        zp = conformers.get(smi)
        if zp is None:
            continue
        z, pos = zp
        data_list.append(
            Data(
                z=torch.from_numpy(z),
                pos=torch.from_numpy(pos),
                y=torch.tensor([(val - mean) / std], dtype=torch.float32),
            )
        )
    return data_list


def _predict(model, data_list, device, batch_size, mean, std):
    """Return (true_orig, pred_orig) numpy arrays."""
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.detach().cpu())
            trues.append(batch.y.detach().cpu())
    pred = torch.cat(preds).numpy() * std + mean
    true = torch.cat(trues).numpy() * std + mean
    return true, pred


def _train(model, train_list, device, args, seed):
    """Train for args.epochs; keep the best-val state; return the model."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_list))
    val_n = max(1, len(train_list) // 10)
    val_list = [train_list[i] for i in idx[:val_n]]
    tr_list = [train_list[i] for i in idx[val_n:]]

    tr_loader = DataLoader(
        tr_list,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_list,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            run += loss.item() * batch.num_graphs
        scheduler.step()

        model.eval()
        vloss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                vloss += criterion(pred, batch.y.view(-1)).item() * batch.num_graphs
        vloss /= max(1, len(val_list))
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"    epoch {epoch:3d}/{args.epochs}  "
                f"train_mse={run / max(1, len(tr_list)):.4f}  val_mse={vloss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _new_model(args, device):
    model = GotenNetRegressor(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        lmax=args.lmax,
        n_rbf=args.n_rbf,
        num_heads=args.num_heads,
        cutoff=args.cutoff,
    ).to(device)
    return model


# ===== Results persistence ==================================================
def _results_path(args):
    tag = "_smoke" if args.smoke_test else ""
    if args.results:
        return args.results
    return os.path.join(REPRODUCE_DIR, f"results_incremental{tag}_seed{args.seed}.json")


def _load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(
        f"  [!] {os.path.basename(path)} not found; starting a fresh results dict " f"(only GotenNet will be present)."
    )
    return {}


def _save_results(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved results -> {path}]")


# ===== Per-endpoint driver ==================================================
def _cap(pairs, n, smoke):
    return pairs[:n] if smoke else pairs


def run_endpoint(prop, label, args, device, results):
    group = GROUP_OF[prop]
    print(f"\n--- {label} ({group}) ---  {_ts()}")
    t0 = time.time()

    split_map = _load_prop_split(prop)  # {smiles: (value, split)}
    struct_map = _load_struct_membership(group)  # {smiles: struct_ood}
    value_map = {s: v for s, (v, _) in split_map.items()}

    # ---- KDE property-value OOD pass -------------------------------------
    train_pairs = [(s, v) for s, (v, sp) in split_map.items() if sp == "train"]
    id_pairs = [(s, v) for s, (v, sp) in split_map.items() if sp == "id"]
    ood_pairs = [(s, v) for s, (v, sp) in split_map.items() if sp == "ood"]

    # ---- Structure (UMAP) OOD pass pairs ---------------------------------
    struct_train_pairs = [(s, value_map[s]) for s, o in struct_map.items() if o == 0 and s in value_map]
    struct_ood_pairs = [(s, value_map[s]) for s, o in struct_map.items() if o == 1 and s in value_map]

    if args.smoke_test:
        train_pairs = _cap(train_pairs, args.smoke_train, True)
        id_pairs = _cap(id_pairs, args.smoke_eval, True)
        ood_pairs = _cap(ood_pairs, args.smoke_eval, True)
        struct_train_pairs = _cap(struct_train_pairs, args.smoke_train, True)
        struct_ood_pairs = _cap(struct_ood_pairs, args.smoke_eval, True)

    # Conformers only for the SMILES actually used (keeps smoke mode tiny).
    needed = {s for s, _ in train_pairs + id_pairs + ood_pairs + struct_train_pairs + struct_ood_pairs}
    conformers = _build_conformer_cache(group, sorted(needed), args.smoke_test)

    tr_vals = np.array([v for _, v in train_pairs], dtype=np.float64)
    mean, std = float(tr_vals.mean()), float(tr_vals.std())
    if std < 1e-10:
        print(f"    [!] train std ~ 0 for {prop!r}; skipping endpoint")
        return
    train_med = float(np.median(tr_vals))

    tr_list = _make_data_list(train_pairs, conformers, mean, std)
    id_list = _make_data_list(id_pairs, conformers, mean, std)
    ood_list = _make_data_list(ood_pairs, conformers, mean, std)
    print(f"    KDE sizes: train={len(tr_list)}, id={len(id_list)}, ood={len(ood_list)}")

    if len(tr_list) < 2 or len(id_list) < 2 or len(ood_list) < 2:
        print(f"    [!] KDE split too small for {prop!r}; skipping endpoint")
        return

    torch.manual_seed(args.seed)
    model = _new_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    GotenNet params: {n_params:,}  ({_ts()} training {args.epochs} epochs) ...")
    model = _train(model, tr_list, device, args, args.seed)

    id_true, id_pred = _predict(model, id_list, device, args.batch_size, mean, std)
    ood_true, ood_pred = _predict(model, ood_list, device, args.batch_size, mean, std)

    entry = {
        "id_r2": r2_score(id_true, id_pred),
        "ood_r2_binned": binned_r2(ood_true, ood_pred, train_med),
        "id_r2_corr": corr_r2(id_true, id_pred),
        "ood_r2_corr_binned": binned_corr_r2(ood_true, ood_pred, train_med),
        "id_rmse": root_mean_squared_error(id_true, id_pred),
        "ood_rmse": root_mean_squared_error(ood_true, ood_pred),
    }
    print(f"    ID  R²={entry['id_r2']:.4f}  RMSE={entry['id_rmse']:.4f}")
    print(f"    OOD R²binned={entry['ood_r2_binned']:.4f}  RMSE={entry['ood_rmse']:.4f}")

    # ---- Structure (UMAP) OOD pass ---------------------------------------
    if len(struct_train_pairs) >= 2 and len(struct_ood_pairs) >= 2:
        s_vals = np.array([v for _, v in struct_train_pairs], dtype=np.float64)
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std())
        if s_std < 1e-10:
            print(f"    [!] struct train std ~ 0 for {prop!r}; skipping structure-OOD")
        else:
            s_tr_list = _make_data_list(struct_train_pairs, conformers, s_mean, s_std)
            s_ood_list = _make_data_list(struct_ood_pairs, conformers, s_mean, s_std)
            print(f"    Struct sizes: train={len(s_tr_list)}, ood={len(s_ood_list)}")
            if len(s_tr_list) >= 2 and len(s_ood_list) >= 2:
                torch.manual_seed(args.seed)
                s_model = _new_model(args, device)
                s_model = _train(s_model, s_tr_list, device, args, args.seed)
                s_true, s_pred = _predict(s_model, s_ood_list, device, args.batch_size, s_mean, s_std)
                entry["struct_ood_r2"] = r2_score(s_true, s_pred)
                entry["struct_ood_rmse"] = root_mean_squared_error(s_true, s_pred)
                print(f"    struct-OOD R²={entry['struct_ood_r2']:.4f}  " f"RMSE={entry['struct_ood_rmse']:.4f}")
    else:
        print(f"    [!] structure split too small for {prop!r}; skipping structure-OOD")

    results.setdefault(MODEL_NAME, {})[prop] = entry
    print(f"  Endpoint total: {_elapsed(t0)}  {_ts()}")


# ===== Main =================================================================
def _parse_args():
    valid = [p for p, _ in ENDPOINTS]
    ap = argparse.ArgumentParser(description="Add GotenNet (3D) to the BOOM results JSON.")
    ap.add_argument("--seed", type=int, default=42, help="Model-training seed (also selects the results file).")
    ap.add_argument("--endpoints", default="all", help=f"Comma-separated subset of {{{','.join(valid)}}} or 'all'.")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs (BOOM protocol: 50).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    # Model hyper-parameters (paper "small"-ish; reduce for speed on smaller GPUs).
    ap.add_argument("--n-atom-basis", type=int, default=128)
    ap.add_argument("--n-interactions", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--n-rbf", type=int, default=32)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--results", default=None, help="Override results JSON path.")
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast sanity check: tiny subsets, few epochs, small net, *_smoke JSON.",
    )
    ap.add_argument("--smoke-train", type=int, default=256)
    ap.add_argument("--smoke-eval", type=int, default=128)
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 3)
        args.n_atom_basis = min(args.n_atom_basis, 64)
        args.n_interactions = min(args.n_interactions, 2)
        args.lmax = min(args.lmax, 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.endpoints == "all":
        selected = ENDPOINTS
    else:
        want = {e.strip() for e in args.endpoints.split(",")}
        selected = [(p, lb) for p, lb in ENDPOINTS if p in want]

    print(f"Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Device: {device} " f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Seed={args.seed}  epochs={args.epochs}  batch={args.batch_size}  smoke={args.smoke_test}")
    print(f"Endpoints: {', '.join(p for p, _ in selected)}")

    results_path = _results_path(args)
    results = _load_results(results_path)

    t_all = time.time()
    for prop, label in selected:
        run_endpoint(prop, label, args, device, results)
        _save_results(results_path, results)  # persist after each endpoint

    print(f"\nDone. Total: {_elapsed(t_all)}. GotenNet results merged into " f"{os.path.basename(results_path)}.")


if __name__ == "__main__":
    sys.exit(main())
