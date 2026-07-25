"""Structure-based out-of-distribution (OOD) splits via UMAP + clustering.

This is an *additional*, chemical-structure-based OOD definition that lives
alongside the existing property-value (KDE) OOD splits — it does not replace or
modify them.  The procedure is:

1. Compute Morgan fingerprints (2048-bit, radius 2) with RDKit.
2. Embed them into 2-D with UMAP (default hyper-parameters).
3. Cluster the 2-D embedding with HDBSCAN, which finds groups separated by
   low-density *gaps* (dense points bridged by empty space stay together, and
   sparse bridge points are labelled as noise rather than forced into a cluster).
4. Rank clusters by how *detached* they are from the rest of the map -- measured
   as the width of the empty gap separating a cluster from all other molecules
   (the shortest distance bridging the cluster to any non-cluster point).  Hold
   out the most detached whole clusters until ~10% of the molecules are OOD; the
   remaining clusters plus all noise points form the training set.  A cluster is
   never split, so every molecule of a cluster is entirely in train OR entirely
   in test.

The split is *property-independent*: one clustering per dataset group, reused
for every endpoint in that group.

Determinism: fingerprints are deterministic; UMAP is given a fixed ``seed``
(UMAP runs single-threaded when seeded, which is slower but reproducible);
HDBSCAN and the gap ranking are deterministic given the embedding.  The result
is written to a CSV so every downstream run reads the same frozen split.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import HDBSCAN  # noqa: E402

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


def _cluster_separations(embedding, labels):
    """Return {cluster_id: gap} measuring how detached each cluster is.

    ``gap`` is the width of the empty space separating a cluster from the rest
    of the map: the shortest distance from any point of the cluster to any
    point *not* in that cluster (other clusters and noise both count as "the
    rest").  A larger gap => a more detached, well-separated island.  Noise
    points (label -1) are never scored as clusters.

    Computed exactly: for each cluster a KD-tree is built over all *other*
    points, and the minimum nearest-neighbour distance from the cluster's own
    points is the bridging gap.  (The complement is always non-empty because a
    split is only produced when at least two clusters exist.)
    """
    seps = {}
    for c in (int(x) for x in np.unique(labels) if x != -1):
        in_mask = labels == c
        ctree = cKDTree(embedding[~in_mask])
        d, _ = ctree.query(embedding[in_mask], k=1)
        seps[c] = float(d.min())
    return seps


def _select_detached_clusters(embedding, labels, ood_frac):
    """Pick the most *detached* whole clusters totalling ~``ood_frac``.

    Clusters are ranked by their bridging gap (largest first); whole clusters
    are added most-detached-first while staying within the target size, then
    one more is added if it lands closer to the target.  Noise points are never
    selected.  Deterministic (gap desc, ties broken by cluster id).

    Returns ``(chosen_ids, separations, sizes)``.
    """
    n_total = len(labels)
    target = ood_frac * n_total
    seps = _cluster_separations(embedding, labels)
    sizes = {int(c): int((labels == c).sum()) for c in seps}
    order = sorted(seps, key=lambda c: (-seps[c], c))  # most detached first

    chosen, cur = [], 0
    for c in order:
        if cur + sizes[c] <= target:
            chosen.append(c)
            cur += sizes[c]

    # Add one more detached cluster if it gets the total closer to the target.
    for c in order:
        if c in chosen:
            continue
        if abs(cur + sizes[c] - target) < abs(cur - target):
            chosen.append(c)
            cur += sizes[c]
            break

    # Guarantee a non-empty, non-total OOD set.
    if not chosen:
        chosen = [order[0]]
    if len(chosen) >= len(seps):
        chosen = [order[0]]
    return sorted(chosen), seps, sizes


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
    min_cluster_size=50,
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
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.  Smaller values detect smaller
        detached islands; larger values keep only bigger coherent groups.
    ood_frac : float
        Target fraction of molecules assigned to the OOD test set.
    seed : int
        Random seed for UMAP (kept fixed across model seeds so the split
        itself is constant).  HDBSCAN and the gap ranking are deterministic.
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
    if len(valid) < 2 * min_cluster_size:
        raise RuntimeError(
            f"Too few valid molecules ({len(valid)}) for HDBSCAN " f"with min_cluster_size={min_cluster_size}."
        )

    _log(f"  [umap:{group_name}] running UMAP (seed={seed}, single-threaded) ...")
    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(feats)

    _log(f"  [umap:{group_name}] HDBSCAN (min_cluster_size={min_cluster_size}) ...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)

    n_clusters_found = int(len({int(c) for c in labels if c != -1}))
    n_noise = int((labels == -1).sum())
    if n_clusters_found < 2:
        raise RuntimeError(
            f"HDBSCAN found {n_clusters_found} cluster(s) for {group_name}; " "try lowering min_cluster_size."
        )

    ood_clusters, separations, cluster_sizes = _select_detached_clusters(embedding, labels, ood_frac)
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
        "method": "hdbscan_gap",
        "min_cluster_size": min_cluster_size,
        "n_total": len(valid),
        "n_clusters_found": n_clusters_found,
        "n_noise": n_noise,
        "ood_frac_target": ood_frac,
        "ood_frac_actual": n_ood / len(valid),
        "n_ood": n_ood,
        "n_train": len(valid) - n_ood,
        "ood_clusters": ood_clusters,
        "cluster_sizes": cluster_sizes,
        "cluster_separations": {
            int(c): (round(separations[c], 6) if np.isfinite(separations[c]) else None) for c in separations
        },
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
