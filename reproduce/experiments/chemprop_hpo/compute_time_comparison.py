#!/usr/bin/env python
"""
Compute-time comparison: Chemprop vs. ElasticNet, default vs. tuned.
=====================================================================

Produces a table (for a paper-review rebuttal) comparing how much wall-clock
compute each model spends on (a) a single default-hyperparameter run and
(b) hyperparameter tuning, per BOOM endpoint.

This script does NOT run any training itself — it only parses timestamps and
elapsed-time strings that are already present in two existing log files:

  * ``reproduce/experiments/chemprop_hpo/hpo_run.log``
        The Chemprop HPO study (this repo's reproduction of the BOOM appendix
        ablation): one DEFAULT run + up to 50 randomly-sampled-hyperparameter
        runs per endpoint. Each line ends with a bracketed elapsed time, e.g.
        ``... params=409,201  [4m07s]``, covering that config's full
        train+predict wall time.

  * ``reproduce/logs/run_all_r2_20260725_223343.log``
        The canonical seed=42 run of the main reproduction pipeline
        (``reproduce/reproduce_parts_of_fig_2_and_add_models.py``), which
        produced ``results_incremental_seed42.json``. For each endpoint it
        trains one default-hyperparameter Chemprop model AND one
        ``ElasticNetCV`` model. ElasticNetCV's hyperparameter "tuning" is a
        single ``.fit()`` call that internally cross-validates a grid of
        4 l1_ratios x 30 alphas x 3 folds (see ``DESCRIPTOR_MODELS`` in that
        script) — i.e. tuning is baked into one call, not 50 separate runs.
        We recover its wall time as the gap between the "Training ElasticNet"
        and "Training XGBoost" log lines (the two calls immediately
        surrounding it), since nothing else happens on that thread in between.

Usage
-----
    uv run python reproduce/experiments/chemprop_hpo/compute_time_comparison.py
"""

import argparse
import os
import re
import statistics
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

DEFAULT_HPO_LOG = os.path.join(SCRIPT_DIR, "hpo_run.log")
DEFAULT_BASELINE_LOG = os.path.join(REPO_ROOT, "reproduce", "logs", "run_all_r2_20260725_223343.log")

# hpo_run.log endpoint key -> human label, in the order they appear.
ENDPOINT_LABELS = OrderedDict(
    [
        ("hof", "HoF"),
        ("density", "Density"),
        ("homo", "HOMO"),
        ("lumo", "LUMO"),
        ("gap", "GAP"),
        ("zpve", "ZPVE"),
        ("r2", "R\u00b2"),
        ("alpha", "\u03b1"),
        ("mu", "\u03bc"),
        ("cv", "C\u1d65"),
    ]
)

TIME_RE = re.compile(r"\[(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s\]")
TS_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\]")


def _parse_bracket_time(s):
    """Parse a trailing '[1h23m45.6s]' / '[4m07s]' / '[12.3s]' into seconds."""
    m = TIME_RE.search(s)
    if not m:
        return None
    h, mi, se = m.groups()
    total = float(se)
    if mi:
        total += int(mi) * 60
    if h:
        total += int(h) * 3600
    return total


def _ts_to_seconds(hh, mm, ss):
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _fmt(secs):
    if secs is None:
        return "n/a"
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(round(secs)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_params(n):
    """Round a parameter count to a coarse, human-friendly approximation
    (e.g. 318,901 -> '~300K', 29,196,001 -> '~29M')."""
    if n is None:
        return "n/a"
    if n >= 1_000_000:
        return f"~{round(n / 1_000_000)}M"
    return f"~{round(n / 100_000) * 100}K"


# ===== Parse the Chemprop HPO log ==========================================
def parse_hpo_log(path):
    """Return {endpoint_key: {"default": secs_or_None, "configs": [(secs, n_params), ...]}}."""
    result = {k: {"default": None, "configs": []} for k in ENDPOINT_LABELS}
    current = None
    endpoint_header_re = re.compile(r"Endpoint:\s+\S+\s+\((\w+)\)")
    default_re = re.compile(r"DEFAULT:.*params=([\d,]+)")
    config_re = re.compile(r"config\s+\d+:.*params=([\d,]+)")

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            hdr = endpoint_header_re.search(line)
            if hdr:
                key = hdr.group(1).lower()
                if key in result:
                    current = key
                continue
            if current is None:
                continue
            if "DEFAULT:" in line:
                m = default_re.search(line)
                secs = _parse_bracket_time(line)
                if m and secs is not None:
                    result[current]["default"] = secs
                continue
            m = config_re.search(line)
            if m:
                secs = _parse_bracket_time(line)
                n_params = int(m.group(1).replace(",", ""))
                if secs is not None:
                    result[current]["configs"].append((secs, n_params))
    return result


# ===== Parse the canonical seed=42 baseline log =============================
def parse_baseline_log(path):
    """Return {endpoint_key: {"chemprop_default": secs, "elasticnet_tuning": secs}}."""
    result = {k: {"chemprop_default": None, "elasticnet_tuning": None} for k in ENDPOINT_LABELS}
    label_to_key = {lbl: k for k, lbl in ENDPOINT_LABELS.items()}
    section_re = re.compile(r"^--- (.+?) ---")
    fit_time_re = re.compile(r"Stopped at epoch \d+/\d+\s+\(training:\s*([^)]+)\)")

    current = None
    pending_en_start = None  # (h, m, s) when "Training ElasticNet" was seen

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            sec = section_re.search(line)
            if sec:
                lbl = sec.group(1).strip()
                current = label_to_key.get(lbl)
                pending_en_start = None
                continue
            if current is None:
                continue

            if "Training Chemprop" in line:
                # next "Stopped at epoch" line (a few lines later) has the fit time
                pass
            m = fit_time_re.search(line)
            if m and result[current]["chemprop_default"] is None:
                secs = _parse_bracket_time("[" + m.group(1).strip() + "]")
                result[current]["chemprop_default"] = secs

            if "Training ElasticNet" in line:
                ts = TS_RE.search(line)
                if ts:
                    pending_en_start = _ts_to_seconds(*ts.groups())
            elif "Training XGBoost" in line and pending_en_start is not None:
                ts = TS_RE.search(line)
                if ts:
                    end = _ts_to_seconds(*ts.groups())
                    delta = end - pending_en_start
                    if delta < 0:  # midnight rollover
                        delta += 24 * 3600
                    result[current]["elasticnet_tuning"] = float(delta)
                pending_en_start = None
    return result


def build_table(hpo_data, baseline_data):
    rows = []
    for key, label in ENDPOINT_LABELS.items():
        hpo = hpo_data[key]
        base = baseline_data[key]
        configs = hpo["configs"]
        n = len(configs)
        times = [t for t, _ in configs]
        total = sum(times) if times else None
        mean = statistics.mean(times) if times else None
        median = statistics.median(times) if times else None
        tmin = min(times) if times else None
        tmax = max(times) if times else None
        # smallest / largest network among completed configs, by param count
        min_np = min(configs, key=lambda c: c[1]) if configs else None
        max_np = max(configs, key=lambda c: c[1]) if configs else None

        rows.append(
            {
                "endpoint": label,
                "n_configs": n,
                "cp_default_s": base["chemprop_default"] or hpo["default"],
                "cp_tune_total_s": total,
                "cp_tune_mean_s": mean,
                "cp_tune_median_s": median,
                "cp_tune_min_s": tmin,
                "cp_tune_max_s": tmax,
                "cp_min_params": min_np[1] if min_np else None,
                "cp_min_params_s": min_np[0] if min_np else None,
                "cp_max_params": max_np[1] if max_np else None,
                "cp_max_params_s": max_np[0] if max_np else None,
                "en_tune_s": base["elasticnet_tuning"],
            }
        )
    return rows


def print_markdown(rows):
    print("\n## Chemprop vs. ElasticNet — compute time per endpoint\n")
    print(
        "| Endpoint | CP default | CP tuning (n) | CP tuning mean/run | "
        "CP tuning min\u2013max/run | CP tuning total | "
        "EN tuning (1 call, full internal grid search) | Speed ratio (CP total / EN) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        n_tag = f"{r['n_configs']}/50" if r["n_configs"] < 50 else "50/50"
        ratio = (
            f"{r['cp_tune_total_s'] / r['en_tune_s']:.0f}\u00d7" if r["cp_tune_total_s"] and r["en_tune_s"] else "n/a"
        )
        if r["cp_tune_min_s"] is None:
            min_max = "n/a"
        else:
            min_max = f"{_fmt(r['cp_tune_min_s'])}\u2013{_fmt(r['cp_tune_max_s'])}"
        print(
            f"| {r['endpoint']} | {_fmt(r['cp_default_s'])} | {n_tag} | "
            f"{_fmt(r['cp_tune_mean_s'])} | {min_max} | "
            f"{_fmt(r['cp_tune_total_s'])} | {_fmt(r['en_tune_s'])} | {ratio} |"
        )

    print("\n## Network-size effect on Chemprop tuning time (smallest vs. largest sampled model)\n")
    print("| Endpoint | Smallest net (params / time) | Largest net (params / time) | Slowdown |")
    print("|---|---|---|---|")
    for r in rows:
        if r["cp_min_params"] is None:
            continue
        slowdown = f"{r['cp_max_params_s'] / r['cp_min_params_s']:.1f}\u00d7" if r["cp_min_params_s"] else "n/a"
        print(
            f"| {r['endpoint']} | {_fmt_params(r['cp_min_params'])} / {_fmt(r['cp_min_params_s'])} | "
            f"{_fmt_params(r['cp_max_params'])} / {_fmt(r['cp_max_params_s'])} | {slowdown} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hpo-log", default=DEFAULT_HPO_LOG)
    ap.add_argument("--baseline-log", default=DEFAULT_BASELINE_LOG)
    args = ap.parse_args()

    hpo_data = parse_hpo_log(args.hpo_log)
    baseline_data = parse_baseline_log(args.baseline_log)
    rows = build_table(hpo_data, baseline_data)
    print_markdown(rows)


if __name__ == "__main__":
    main()
