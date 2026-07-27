#!/usr/bin/env python
"""Standalone MoLFormer runner that ADDS the pretrained chemical language
model to the BOOM reproduction results without touching any of the existing
models.

It loads the shared ``results_incremental_seed<seed>.json`` produced by
``reproduce/reproduce_parts_of_fig_2_and_add_models.py`` (the 4 descriptor /
MPNN models), fine-tunes MoLFormer on the SAME splits, and writes ONLY the
``results["MolFormer"]`` sub-tree back -- every other model is preserved
byte-for-byte.

Why a separate script / environment?
------------------------------------
MoLFormer (IBM, Ross et al. 2022; the model used in the BOOM paper's Table 2)
uses the IDIAP pytorch-fast-transformers linear-attention transformer with
rotary embeddings. The build/runtime stack does not coexist cleanly with the
main project's torch, and the BOOM protocol additionally substitutes
``torch_optimizer.Lamb`` for the original ``apex.optimizers.FusedLAMB``
(documented in experiments/molformer/readme.md), so this runner lives in its
own virtual-env (``.venv_molformer``) and talks to the rest of the pipeline
only through the split CSVs and the results JSON -- no ``boom`` import,
no shared Python process.

Vendored upstream code
----------------------
The MoLFormer LightningModule, tokenizer, rotary linear-attention blocks and
vocab file live under
``experiments/molformer/src/molformer-main/notebooks/pretrained_molformer/``
and are used as-is (no source modifications). This runner adds that directory
to ``sys.path`` and briefly ``os.chdir``'s into it while instantiating the
model (the vendored ``Encoder`` loads a vocab .pth via a relative path).

Splits (identical definitions to the main pipeline)
---------------------------------------------------
* KDE property-value OOD  -> ID / OOD from the ``*_data_with_ood_splits*.csv``.
* Structure (UMAP/HDBSCAN) OOD -> ``umap_splits_{10k,QM9}.csv`` (struct_ood).

Fine-tuning
-----------
The pretrained checkpoint is loaded per endpoint (2 fits per endpoint: one for
KDE-OOD, one for structure-OOD), the linear MLM head is discarded, and a
2-layer regression head is placed on top of masked-mean-pooled encoder
embeddings. Optimiser is ``torch_optimizer.Lamb`` with a cosine schedule
(BOOM protocol, matches the paper's Section 4.1 / Appendix 8.4 -- "same
fine-tune schedule for both scratch and pretrained MoLFormer").

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
import random
import sys
import time
from argparse import Namespace

# The IDIAP fast_transformers random-feature map used by MoLFormer's linear
# attention calls torch.qr on the queries.device. MPS has no aten::linalg_qr
# kernel; enable the CPU fallback (must be set BEFORE `import torch`). Harmless
# on CPU / CUDA -- the flag is only consulted by the MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths.
# SCRIPT_DIR       = reproduce/experiments/molformer/.
# REPRODUCE_DIR    = reproduce/ (where the shared results JSON lives).
# DATA_DIR         = reproduce/experiments/data/ (split CSVs).
# CKPT_DIR         = reproduce/experiments/data/molformer_ckpts/ (user-supplied).
# MOLFORMER_DIR    = experiments/molformer/src/molformer-main/notebooks/
#                    pretrained_molformer/ (vendored upstream code + vocab).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
REPRODUCE_DIR = os.path.dirname(EXPERIMENTS_DIR)
REPO_ROOT = os.path.dirname(REPRODUCE_DIR)
DATA_DIR = os.path.join(EXPERIMENTS_DIR, "data")
CKPT_DIR = os.path.join(DATA_DIR, "molformer_ckpts")
MOLFORMER_DIR = os.path.join(
    REPO_ROOT,
    "experiments",
    "molformer",
    "src",
    "molformer-main",
    "notebooks",
    "pretrained_molformer",
)

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

MODEL_NAME = "MolFormer"  # matches paper Table 2 row exactly

# Pretrained MoLFormer architecture (matches the checkpoint shipped at
# https://ibm.box.com/v/MoLFormer-data). Do NOT change unless swapping ckpt.
_MOLFORMER_HPARAMS = dict(
    n_embd=768,
    n_layer=12,
    n_head=12,
    d_dropout=0.1,
    num_feats=32,  # GeneralizedRandomFeatures dimensionality
    max_len=202,  # MoLFormer default; will be overridden by CLI --max-len
)
DEFAULT_CKPT_NAME = "N-Step-Checkpoint_3_30000.ckpt"  # BOOM "Pretrained" variant


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
    """Return {prop: {smiles: (value, split)}} for hof and density."""
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


# ===== MoLFormer model / tokenizer loading ==================================
def _molformer_config(max_len):
    """Build a Namespace with the fields the vendored LightningModule reads.

    The pretrained checkpoint expects exactly these architecture values; the
    training-only fields (lr_start, lr_multiplier, restart_path, seed, debug)
    are required by ``LightningModule.__init__`` but never used by our
    fine-tuning loop -- we replace the LM head and use our own optimiser.
    """
    cfg = dict(_MOLFORMER_HPARAMS)
    cfg["max_len"] = max_len
    cfg["restart_path"] = "unused"  # non-empty -> skips seed_everything call
    cfg["seed"] = 12345
    cfg["debug"] = False
    cfg["lr_start"] = 3e-5
    cfg["lr_multiplier"] = 1
    return Namespace(**cfg)


def _load_molformer(ckpt_path, max_len):
    """Load the vendored LightningModule + tokenizer from a pretrained ckpt.

    Runs briefly under ``os.chdir(MOLFORMER_DIR)`` so the vendored
    ``pubchem_encoder.Encoder`` can find its relative vocab .pth file, and so
    ``MolTranBertTokenizer('bert_vocab.txt')`` resolves the tokenizer vocab.
    """
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"MoLFormer checkpoint not found: {ckpt_path}\n"
            f"Download 'Pretrained MoLFormer.zip' from https://ibm.box.com/v/MoLFormer-data,\n"
            f"extract it, and copy checkpoints/{DEFAULT_CKPT_NAME} into:\n"
            f"  {CKPT_DIR}"
        )

    if MOLFORMER_DIR not in sys.path:
        sys.path.insert(0, MOLFORMER_DIR)

    orig_cwd = os.getcwd()
    os.chdir(MOLFORMER_DIR)
    try:
        from tokenizer.tokenizer import MolTranBertTokenizer  # noqa: E402
        from train_pubchem_light import LightningModule  # noqa: E402

        tokenizer = MolTranBertTokenizer("bert_vocab.txt")
        config = _molformer_config(max_len)
        # strict=False: our fine-tuned head is added later; also tolerates any
        # tiny key mismatches between vendored LightningModule and the ckpt.
        lm = LightningModule.load_from_checkpoint(
            ckpt_path,
            config=config,
            vocab=tokenizer.vocab,
            strict=False,
            map_location="cpu",
        )
    finally:
        os.chdir(orig_cwd)
    return lm, tokenizer


# ===== Regression wrapper ===================================================
class MolFormerRegressor(nn.Module):
    """Pretrained MoLFormer encoder + masked-mean-pool + 2-layer MLP head.

    The forward mirrors the ``embed()`` helper in the vendored notebook
    ``finetuned_embeddings_RF_regression.ipynb`` but is end-to-end
    differentiable (no torch.no_grad wrapper) so gradients flow through the
    pretrained transformer during fine-tuning.
    """

    def __init__(self, lm, hidden=768, dropout=0.1):
        super().__init__()
        self.lm = lm  # vendored LightningModule; we only use .tok_emb + .blocks
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        # Local import: the fast_transformers package is only guaranteed
        # available inside .venv_molformer (see setup_env.sh).
        from fast_transformers.masking import LengthMask as LM

        x = self.lm.tok_emb(input_ids)
        x = self.lm.blocks(x, length_mask=LM(attention_mask.sum(-1)))
        mask_e = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


# ===== Dataset / dataloader =================================================
class _SmilesRegressionDataset(Dataset):
    def __init__(self, pairs, mean, std):
        # pairs: list of (smiles, value)
        self.smiles = [s for s, _ in pairs]
        self.y = np.array([(v - mean) / std for _, v in pairs], dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.y[i]


def _make_collate(tokenizer, max_len):
    def collate(batch):
        smiles_list = [b[0] for b in batch]
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        enc = tokenizer.batch_encode_plus(
            smiles_list,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        return input_ids, mask, targets

    return collate


# ===== Train / predict loops ================================================
def _predict(model, loader, device, mean, std):
    model.eval()
    preds, trues = [], []
    # torch.no_grad (not inference_mode) for the same feature_map.omega
    # reason documented in _train().
    with torch.no_grad():
        for input_ids, mask, y in loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            out = model(input_ids, mask).detach().cpu()
            preds.append(out)
            trues.append(y)
    pred = torch.cat(preds).numpy() * std + mean
    true = torch.cat(trues).numpy() * std + mean
    return true, pred


def _train(model, train_pairs, tokenizer, device, args, seed, mean, std):
    """Train for args.epochs; keep best-val state; return the model."""
    import torch_optimizer as toptim

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_pairs))
    val_n = max(1, len(train_pairs) // 10)
    val_pairs = [train_pairs[i] for i in idx[:val_n]]
    tr_pairs = [train_pairs[i] for i in idx[val_n:]]

    collate = _make_collate(tokenizer, args.max_len)
    tr_loader = DataLoader(
        _SmilesRegressionDataset(tr_pairs, mean, std),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        _SmilesRegressionDataset(val_pairs, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    # BOOM protocol: torch_optimizer.Lamb (substitute for apex FusedLAMB).
    optimizer = toptim.Lamb(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        n_seen = 0
        for input_ids, mask, y in tr_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(input_ids, mask)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run += loss.item() * y.size(0)
            n_seen += y.size(0)
        scheduler.step()

        model.eval()
        vloss = 0.0
        n_val = 0
        # NOTE: use torch.no_grad(), not torch.inference_mode(). The IDIAP
        # fast_transformers linear-attention feature map regenerates its
        # ``omega`` tensor on every forward via an in-place write; running
        # that under inference_mode marks omega as an inference tensor, and
        # later ``load_state_dict(best_state)`` then fails with "Inplace
        # update to inference tensor outside InferenceMode is not allowed".
        with torch.no_grad():
            for input_ids, mask, y in val_loader:
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                y = y.to(device)
                pred = model(input_ids, mask)
                vloss += criterion(pred, y).item() * y.size(0)
                n_val += y.size(0)
        vloss /= max(1, n_val)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"    epoch {epoch:3d}/{args.epochs}  " f"train_mse={run / max(1, n_seen):.4f}  val_mse={vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _new_regressor(ckpt_path, args, device):
    lm, tokenizer = _load_molformer(ckpt_path, args.max_len)
    model = MolFormerRegressor(lm, hidden=_MOLFORMER_HPARAMS["n_embd"], dropout=0.1)
    model.to(device)
    return model, tokenizer


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
        f"  [!] {os.path.basename(path)} not found; starting a fresh results dict " f"(only MolFormer will be present)."
    )
    return {}


def _save_results(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved results -> {path}]")


# ===== Per-endpoint driver ==================================================
def _cap(pairs, n, smoke):
    return pairs[:n] if smoke else pairs


def _make_loader(pairs, tokenizer, args, device, mean, std, shuffle=False):
    return DataLoader(
        _SmilesRegressionDataset(pairs, mean, std),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_make_collate(tokenizer, args.max_len),
    )


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

    # ---- Structure (UMAP) OOD pass ---------------------------------------
    struct_train_pairs = [(s, value_map[s]) for s, o in struct_map.items() if o == 0 and s in value_map]
    struct_ood_pairs = [(s, value_map[s]) for s, o in struct_map.items() if o == 1 and s in value_map]

    if args.smoke_test:
        train_pairs = _cap(train_pairs, args.smoke_train, True)
        id_pairs = _cap(id_pairs, args.smoke_eval, True)
        ood_pairs = _cap(ood_pairs, args.smoke_eval, True)
        struct_train_pairs = _cap(struct_train_pairs, args.smoke_train, True)
        struct_ood_pairs = _cap(struct_ood_pairs, args.smoke_eval, True)

    tr_vals = np.array([v for _, v in train_pairs], dtype=np.float64)
    if len(tr_vals) < 2:
        print(f"    [!] KDE train too small for {prop!r}; skipping endpoint")
        return
    mean, std = float(tr_vals.mean()), float(tr_vals.std())
    if std < 1e-10:
        print(f"    [!] train std ~ 0 for {prop!r}; skipping endpoint")
        return
    train_med = float(np.median(tr_vals))

    print(f"    KDE sizes: train={len(train_pairs)}, id={len(id_pairs)}, ood={len(ood_pairs)}")
    if len(id_pairs) < 2 or len(ood_pairs) < 2:
        print(f"    [!] KDE split too small for {prop!r}; skipping endpoint")
        return

    torch.manual_seed(args.seed)
    model, tokenizer = _new_regressor(args.ckpt, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"    MolFormer params: {n_params:,} total, {n_train:,} trainable  "
        f"({_ts()} training {args.epochs} epochs) ..."
    )
    model = _train(model, train_pairs, tokenizer, device, args, args.seed, mean, std)

    id_loader = _make_loader(id_pairs, tokenizer, args, device, mean, std)
    ood_loader = _make_loader(ood_pairs, tokenizer, args, device, mean, std)
    id_true, id_pred = _predict(model, id_loader, device, mean, std)
    ood_true, ood_pred = _predict(model, ood_loader, device, mean, std)

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

    # free the KDE-model before spinning up the structure-OOD one to keep
    # peak memory under control (limited memory / small GPUs).
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # ---- Structure (UMAP) OOD pass ---------------------------------------
    if len(struct_train_pairs) >= 2 and len(struct_ood_pairs) >= 2:
        s_vals = np.array([v for _, v in struct_train_pairs], dtype=np.float64)
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std())
        if s_std < 1e-10:
            print(f"    [!] struct train std ~ 0 for {prop!r}; skipping structure-OOD")
        else:
            print(f"    Struct sizes: train={len(struct_train_pairs)}, ood={len(struct_ood_pairs)}")
            torch.manual_seed(args.seed)
            s_model, s_tokenizer = _new_regressor(args.ckpt, args, device)
            s_model = _train(s_model, struct_train_pairs, s_tokenizer, device, args, args.seed, s_mean, s_std)
            s_loader = _make_loader(struct_ood_pairs, s_tokenizer, args, device, s_mean, s_std)
            s_true, s_pred = _predict(s_model, s_loader, device, s_mean, s_std)
            entry["struct_ood_r2"] = r2_score(s_true, s_pred)
            entry["struct_ood_rmse"] = root_mean_squared_error(s_true, s_pred)
            print(f"    struct-OOD R²={entry['struct_ood_r2']:.4f}  " f"RMSE={entry['struct_ood_rmse']:.4f}")
            del s_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
    else:
        print(f"    [!] structure split too small for {prop!r}; skipping structure-OOD")

    results.setdefault(MODEL_NAME, {})[prop] = entry
    print(f"  Endpoint total: {_elapsed(t0)}  {_ts()}")


# ===== Device selection =====================================================
def _pick_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===== Main =================================================================
def _parse_args():
    valid = [p for p, _ in ENDPOINTS]
    default_ckpt = os.path.join(CKPT_DIR, DEFAULT_CKPT_NAME)
    ap = argparse.ArgumentParser(description="Add MoLFormer to the BOOM results JSON.")
    ap.add_argument("--seed", type=int, default=42, help="Model-training seed (also selects the results file).")
    ap.add_argument("--endpoints", default="all", help=f"Comma-separated subset of {{{','.join(valid)}}} or 'all'.")
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs. Default matches the BOOM ChemBERTa "
        "runners (num_epochs=5); the paper's Appendix 8.4 "
        "states the two transformers share the fine-tune "
        "schedule.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Fine-tune batch size (smaller default than the paper " "to fit limited memory; bump on larger GPUs).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Peak learning rate. Default 1e-5 matches BOOM's "
        "ChemBERTa AdamW lr; BOOM's MolFormer readme swaps "
        "the optimizer for torch_optimizer.Lamb.",
    )
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=202, help="Max SMILES token length (pretraining default: 202).")
    ap.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers. 0 avoids multiprocessing fork issues."
    )
    ap.add_argument("--ckpt", default=default_ckpt, help="Path to the pretrained MoLFormer .ckpt file.")
    ap.add_argument("--results", default=None, help="Override results JSON path.")
    ap.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. 'cpu', 'cuda', 'mps'). " "Default: cuda if available else cpu.",
    )
    ap.add_argument("--allow-mps", action="store_true", help="Use the Apple MPS backend if available.")
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast sanity check: tiny subsets, few epochs, *_smoke JSON.",
    )
    ap.add_argument("--smoke-train", type=int, default=256)
    ap.add_argument("--smoke-eval", type=int, default=128)
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args)
    if args.endpoints == "all":
        selected = ENDPOINTS
    else:
        want = {e.strip() for e in args.endpoints.split(",")}
        selected = [(p, lb) for p, lb in ENDPOINTS if p in want]

    print(f"Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        dev_name = "MPS"
    else:
        dev_name = "CPU"
    print(f"Device: {device} ({dev_name})")
    print(f"Ckpt: {args.ckpt}")
    print(
        f"Seed={args.seed}  epochs={args.epochs}  batch={args.batch_size}  "
        f"max_len={args.max_len}  smoke={args.smoke_test}"
    )
    print(f"Endpoints: {', '.join(p for p, _ in selected)}")

    results_path = _results_path(args)
    results = _load_results(results_path)

    t_all = time.time()
    for prop, label in selected:
        run_endpoint(prop, label, args, device, results)
        _save_results(results_path, results)  # persist after each endpoint

    print(f"\nDone. Total: {_elapsed(t_all)}. MolFormer results merged into " f"{os.path.basename(results_path)}.")


if __name__ == "__main__":
    sys.exit(main())
