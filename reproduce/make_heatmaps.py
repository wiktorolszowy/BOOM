#!/usr/bin/env python
"""Generate heatmaps from results_incremental.json (partial or full)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results_incremental.json")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(RESULTS_JSON) as f:
    results = json.load(f)

# Discover which endpoints are actually present (intersection across all models)
ALL_MODEL_NAMES = list(results.keys())
available_props = set.intersection(*(set(results[m].keys()) for m in ALL_MODEL_NAMES))

# Keep original order, filter to available
FULL_ENDPOINTS = [
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
ENDPOINTS = [(p, lb) for p, lb in FULL_ENDPOINTS if p in available_props]
print(f"Models: {ALL_MODEL_NAMES}")
print(f"Endpoints ({len(ENDPOINTS)}): {[lb for _, lb in ENDPOINTS]}")

plt.rcParams.update({"font.size": 12})

model_names = ALL_MODEL_NAMES
prop_labels = [label for _, label in ENDPOINTS]
prop_keys = [prop for prop, _ in ENDPOINTS]

n_models = len(model_names)
n_props = len(prop_keys)
cell_h = 0.62  # inches per row
fig_w = max(12.0, n_props * 1.25)
fig_h = (n_models * cell_h + 1.5) * 2 + 1.1  # two subplots + shared padding


def _axis_bottom(ax, ylabel, title, xlabel="Property"):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.yaxis.label.set_size(11)


# 1) RMSE heatmaps
id_rmse = np.array([[results[m][p]["id_rmse"] for p in prop_keys] for m in model_names])
ood_rmse = np.array([[results[m][p]["ood_rmse"] for p in prop_keys] for m in model_names])


def _col_normalise(arr):
    col_min = arr.min(axis=0, keepdims=True)
    col_max = arr.max(axis=0, keepdims=True)
    denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    return (arr - col_min) / denom


def _fmt_annot(arr):
    return np.array([[f"{v:.2f}" for v in row] for row in arr])


fig_rmse, (ax_id_rmse, ax_ood_rmse) = plt.subplots(2, 1, figsize=(fig_w, fig_h), gridspec_kw={"hspace": 0.32})
rmse_common = dict(
    xticklabels=prop_labels,
    yticklabels=model_names,
    linewidths=0.2,
    linecolor="white",
    annot_kws={"fontsize": 9},
    cmap="YlGnBu_r",
    vmin=0,
    vmax=1,
    fmt="",
    cbar_kws={"label": "Relative RMSE\n(per property, 0 = best)", "shrink": 0.8},
)
sns.heatmap(_col_normalise(id_rmse), ax=ax_id_rmse, annot=_fmt_annot(id_rmse), **rmse_common)
_axis_bottom(ax_id_rmse, "Model", "ID Splits  (RMSE)")
sns.heatmap(_col_normalise(ood_rmse), ax=ax_ood_rmse, annot=_fmt_annot(ood_rmse), **rmse_common)
_axis_bottom(ax_ood_rmse, "Model", "OOD Splits  (RMSE)")
rmse_path = os.path.join(FIGURES_DIR, "heatmap_rmse.png")
fig_rmse.savefig(rmse_path, dpi=150, bbox_inches="tight")
plt.close(fig_rmse)
print(f"RMSE heatmap saved to {rmse_path}")

# 2) Binned R² heatmaps
id_r2 = np.array([[results[m][p]["id_r2"] for p in prop_keys] for m in model_names])
ood_r2_binned = np.array([[results[m][p]["ood_r2_binned"] for p in prop_keys] for m in model_names])

fig_bin, (ax_id_bin, ax_ood_bin) = plt.subplots(2, 1, figsize=(fig_w, fig_h), gridspec_kw={"hspace": 0.32})
r2b_common = dict(
    annot=True,
    fmt=".2f",
    xticklabels=prop_labels,
    yticklabels=model_names,
    linewidths=0.2,
    linecolor="white",
    annot_kws={"fontsize": 9},
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "R²", "shrink": 0.8},
)
sns.heatmap(id_r2, ax=ax_id_bin, **r2b_common)
_axis_bottom(ax_id_bin, "Model", "ID Splits  (R²)")
sns.heatmap(ood_r2_binned, ax=ax_ood_bin, **r2b_common)
_axis_bottom(ax_ood_bin, "Model", "OOD Splits  (Binned R²)")
bin_path = os.path.join(FIGURES_DIR, "heatmap_r2.png")
fig_bin.savefig(bin_path, dpi=150, bbox_inches="tight")
plt.close(fig_bin)
print(f"Binned-R² heatmap saved to {bin_path}")

# 3) ρ² heatmaps (correlation coefficient squared, as sometimes misused as R²)
# Only generated if id_r2_corr / ood_r2_corr keys are present in results.
has_corr = all(
    "id_r2_corr" in results[m][p] and "ood_r2_corr_binned" in results[m][p]
    for m in model_names
    for p in prop_keys
    if p in results[m]
)
if has_corr:
    id_r2_corr = np.array([[results[m][p]["id_r2_corr"] for p in prop_keys] for m in model_names])
    ood_r2_corr_binned = np.array([[results[m][p]["ood_r2_corr_binned"] for p in prop_keys] for m in model_names])

    fig_corr, (ax_id_corr, ax_ood_corr) = plt.subplots(2, 1, figsize=(fig_w, fig_h), gridspec_kw={"hspace": 0.32})
    r2c_common = dict(
        annot=True,
        fmt=".2f",
        xticklabels=prop_labels,
        yticklabels=model_names,
        linewidths=0.2,
        linecolor="white",
        annot_kws={"fontsize": 9},
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "ρ²", "shrink": 0.8},
    )
    sns.heatmap(id_r2_corr, ax=ax_id_corr, **r2c_common)
    _axis_bottom(ax_id_corr, "Model", "ID Splits  (ρ²  =  corr. coeff. squared)")
    sns.heatmap(ood_r2_corr_binned, ax=ax_ood_corr, **r2c_common)
    _axis_bottom(ax_ood_corr, "Model", "OOD Splits  (Binned ρ²)")
    corr_path = os.path.join(FIGURES_DIR, "heatmap_r2_corr.png")
    fig_corr.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close(fig_corr)
    print(f"ρ² heatmap saved to {corr_path}")
else:
    print(
        "ρ² heatmap skipped — re-run reproduce_parts_of_fig_2_and_add_elastic_net.py to populate id_r2_corr/ood_r2_corr keys."
    )
