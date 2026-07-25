"""Structure-based out-of-distribution (OOD) splits via UMAP + clustering.

This is an *additional*, chemical-structure-based OOD definition that lives
alongside the existing property-value (KDE) OOD splits — it does not replace or
modify them.  The procedure is:

1. Compute Morgan fingerprints (2048-bit, radius 2) with RDKit.
2. Embed them into 2-D with UMAP (default hyper-parameters).
3. Cluster the 2-D embedding with K-means (>= 20 clusters).
4. Select whole clusters totalling ~10% of the molecules as the OOD test set;
   the remaining clusters form the training set.  A cluster is never split, so
   every molecule of a cluster is entirely in train OR entirely in test.

The split is *property-independent*: one clustering per dataset group, reused
for every endpoint in that group.

Determinism: fingerprints are deterministic; UMAP and K-means are given a fixed
``seed`` (UMAP runs single-threaded when seeded, which is slower but
reproducible).  The result is written to a CSV so every downstream run reads the
same frozen split.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

# CSV columns written by :func:`generate_umap_structure_split`.
CSV_COLUMNS = ["smiles", "umap_x", "umap_y", "cluster", "struct_ood", "struct_train"]


def morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Return (features, valid_smiles) for the parseable molecules.

    ``features`` is an (n_valid, n_bits) uint8 array; ``valid_smiles`` is the
    matching list of SMILES (invalid SMILES are dropped, order preserved).
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    feats = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats.append(gen.GetFingerprintAsNumPy(mol))
        valid.append(smi)
    if not feats:
        return np.empty((0, n_bits), dtype=np.uint8), []
    return np.vstack(feats).astype(np.uint8), valid


def _select_ood_clusters(cluster_sizes, n_total, ood_frac):
    """Pick whole clusters whose combined size is closest to ``ood_frac``.

    Deterministic: clusters are considered smallest-first (tie-broken by id).
    Returns a sorted list of chosen cluster ids.
    """
    target = ood_frac * n_total
    order = sorted(cluster_sizes, key=lambda c: (cluster_sizes[c], c))

    chosen = []
    cur = 0
    for c in order:
        if cur + cluster_sizes[c] <= target:
            chosen.append(c)
            cur += cluster_sizes[c]

    # Optionally add one more cluster if it gets us closer to the target.
    best_c, best_improve = None, 0.0
    for c in order:
        if c in chosen:
            continue
        improvement = abs(cur - target) - abs(cur + cluster_sizes[c] - target)
        if improvement > best_improve:
            best_improve, best_c = improvement, c
    if best_c is not None:
        chosen.append(best_c)

    # Guarantee a non-empty, non-total OOD set.
    if not chosen:
        chosen = [order[0]]
    if len(chosen) >= len(cluster_sizes):
        chosen = [order[0]]
    return sorted(chosen)


def _save_plots(embedding, labels, ood_mask, figures_dir, group_name):
    """UMAP scatter coloured (a) by cluster and (b) by train/OOD assignment."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=3, linewidths=0)
    ax.set_title(f"UMAP structure clusters — {group_name}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    p1 = os.path.join(figures_dir, f"umap_{group_name}_clusters.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embedding[~ood_mask, 0], embedding[~ood_mask, 1], c="lightgrey", s=3, linewidths=0, label="train")
    ax.scatter(embedding[ood_mask, 0], embedding[ood_mask, 1], c="crimson", s=3, linewidths=0, label="OOD test")
    ax.set_title(f"UMAP structure OOD split — {group_name}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    p2 = os.path.join(figures_dir, f"umap_{group_name}_split.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def generate_umap_structure_split(
    smiles_list,
    out_csv,
    figures_dir,
    group_name,
    n_clusters=20,
    ood_frac=0.10,
    seed=42,
    max_n=None,
    verbose=True,
):
    """Generate (or return path to) a structure-based OOD split CSV.

    Parameters
    ----------
    smiles_list : list[str]
        Universe of molecules for this dataset group.
    out_csv : str
        Where to write the split CSV (and the summary JSON alongside it).
    figures_dir : str
        Directory for the UMAP summary plots.
    n_clusters : int
        Number of K-means clusters (>= 20 per the study design).
    ood_frac : float
        Target fraction of molecules assigned to the OOD test set.
    seed : int
        Random seed for UMAP + K-means (kept fixed across model seeds so the
        split itself is constant).
    max_n : int | None
        If set and the universe is larger, deterministically subsample to this
        many molecules (used for fast smoke tests).

    Returns
    -------
    dict
        Summary dictionary (also written next to the CSV as ``*_summary.json``).
    """
    import umap  # imported lazily so importing this module stays cheap

    def _log(msg):
        if verbose:
            print(msg)

    smiles_list = sorted(set(smiles_list))
    if max_n is not None and len(smiles_list) > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(smiles_list), size=max_n, replace=False))
        smiles_list = [smiles_list[i] for i in idx]
        _log(f"  [umap:{group_name}] subsampled to {max_n} molecules for speed")

    _log(f"  [umap:{group_name}] computing Morgan fingerprints for {len(smiles_list)} molecules ...")
    feats, valid = morgan_fingerprints(smiles_list)
    if len(valid) < n_clusters:
        raise RuntimeError(f"Too few valid molecules ({len(valid)}) for {n_clusters} clusters.")

    _log(f"  [umap:{group_name}] running UMAP (seed={seed}, single-threaded) ...")
    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(feats)

    _log(f"  [umap:{group_name}] K-means into {n_clusters} clusters ...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)

    cluster_sizes = {int(c): int((labels == c).sum()) for c in range(n_clusters)}
    ood_clusters = _select_ood_clusters(cluster_sizes, len(valid), ood_frac)
    ood_mask = np.isin(labels, ood_clusters)

    # ---- write split CSV ----
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for smi, (x, y), c, is_ood in zip(valid, embedding, labels, ood_mask):
            f.write(f"{smi},{x:.6f},{y:.6f},{int(c)},{int(is_ood)},{int(not is_ood)}\n")

    n_ood = int(ood_mask.sum())
    summary = {
        "group": group_name,
        "seed": seed,
        "n_total": len(valid),
        "n_clusters": n_clusters,
        "ood_frac_target": ood_frac,
        "ood_frac_actual": n_ood / len(valid),
        "n_ood": n_ood,
        "n_train": len(valid) - n_ood,
        "ood_clusters": ood_clusters,
        "cluster_sizes": cluster_sizes,
    }
    with open(os.path.splitext(out_csv)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    p1, p2 = _save_plots(embedding, labels, ood_mask, figures_dir, group_name)

    _log(
        f"  [umap:{group_name}] OOD clusters={ood_clusters} "
        f"n_ood={n_ood} ({100 * summary['ood_frac_actual']:.1f}%) "
        f"n_train={summary['n_train']}"
    )
    _log(f"  [umap:{group_name}] wrote {out_csv}, plots: {os.path.basename(p1)}, {os.path.basename(p2)}")
    return summary


def load_umap_structure_split(csv_path):
    """Load a split CSV, returning {smiles: struct_ood (0/1)}."""
    membership = {}
    with open(csv_path) as f:
        header = f.readline().rstrip("\n").split(",")
        smi_i = header.index("smiles")
        ood_i = header.index("struct_ood")
        for line in f:
            parts = line.rstrip("\n").split(",")
            membership[parts[smi_i]] = int(parts[ood_i])
    return membership
