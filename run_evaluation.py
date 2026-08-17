"""
Runs SIFT vs ORB matching evaluation across the whole HPatches sequences dataset,
aggregates precision-style stats, and produces comparison plots + a results CSV.

Usage:
    python run_evaluation.py [--data-dir DATA_DIR] [--out-dir OUT_DIR] [--limit N]

Outputs:
    results/raw_results.csv           -- one row per (sequence, pair, method)
    figures/accuracy_by_baseline.png  -- mean matching accuracy vs. image pair index
    figures/accuracy_viewpoint_vs_illumination.png
    figures/precision_recall.png      -- precision vs threshold, proxy for PR curve
    figures/num_matches.png           -- raw match counts, SIFT vs ORB
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matching_pipeline import process_sequence

THRESHOLDS = (1, 2, 3, 5, 10)


def find_sequences(data_dir):
    """HPatches sequence folders start with 'v_' (viewpoint) or 'i_' (illumination)."""
    seqs = []
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full) and (name.startswith("v_") or name.startswith("i_")):
            seqs.append(full)
    return seqs


def run_all(data_dir, out_dir, limit=None):
    results_dir = os.path.join(out_dir, "results")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    sequences = find_sequences(data_dir)
    if limit:
        sequences = sequences[:limit]

    if not sequences:
        raise RuntimeError(
            f"No sequences found in {data_dir}. Did you run download_data.py first?"
        )

    print(f"Found {len(sequences)} sequences. Running SIFT and ORB on each...")

    rows = []
    for si, seq_dir in enumerate(sequences):
        seq_name = os.path.basename(seq_dir)
        seq_type = "viewpoint" if seq_name.startswith("v_") else "illumination"

        for method in ("sift", "orb"):
            t0 = time.time()
            try:
                pair_results = process_sequence(seq_dir, method, THRESHOLDS)
            except Exception as e:
                print(f"  [skip] {seq_name} ({method}): {e}")
                continue
            elapsed = time.time() - t0

            for r in pair_results:
                row = {
                    "sequence": seq_name,
                    "seq_type": seq_type,
                    "method": method,
                    "pair": f"1_{r['pair'][1]}",
                    "baseline_idx": r["pair"][1],  # 2..6, higher = harder
                    "num_kp1": r["num_kp1"],
                    "num_kp2": r["num_kp2"],
                    "num_matches": r["num_matches"],
                    "time_sec": elapsed / 5,  # amortize across the 5 pairs in sequence
                }
                for t in THRESHOLDS:
                    correct = r["correct_at_threshold"][t]
                    total = r["num_matches"]
                    row[f"correct_at_{t}px"] = correct
                    row[f"precision_at_{t}px"] = correct / total if total > 0 else 0.0
                rows.append(row)

        if (si + 1) % 10 == 0 or si == len(sequences) - 1:
            print(f"  processed {si + 1}/{len(sequences)} sequences")

    # write raw CSV
    csv_path = os.path.join(results_dir, "raw_results.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote raw results to {csv_path}")

    make_plots(rows, figures_dir)
    print(f"Wrote figures to {figures_dir}")
    return rows


def make_plots(rows, figures_dir, primary_threshold=3):
    methods = ["sift", "orb"]
    colors = {"sift": "tab:blue", "orb": "tab:orange"}

    # --- Plot 1: mean precision (accuracy) vs baseline index (2..6), per method ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        by_baseline = {}
        for r in rows:
            if r["method"] != method:
                continue
            by_baseline.setdefault(r["baseline_idx"], []).append(
                r[f"precision_at_{primary_threshold}px"]
            )
        xs = sorted(by_baseline.keys())
        ys = [np.mean(by_baseline[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=method.upper(), color=colors[method])
    ax.set_xlabel("Image pair (1 -> N), increasing baseline severity")
    ax.set_ylabel(f"Mean matching precision (@{primary_threshold}px)")
    ax.set_title("Matching accuracy vs. baseline severity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "accuracy_by_baseline.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: viewpoint vs illumination, per method ---
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    seq_types = ["viewpoint", "illumination"]
    x = np.arange(len(seq_types))
    for i, method in enumerate(methods):
        means = []
        for st in seq_types:
            vals = [
                r[f"precision_at_{primary_threshold}px"]
                for r in rows
                if r["method"] == method and r["seq_type"] == st
            ]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x + (i - 0.5) * width, means, width, label=method.upper(), color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels(["Viewpoint change", "Illumination change"])
    ax.set_ylabel(f"Mean matching precision (@{primary_threshold}px)")
    ax.set_title("Matching accuracy: viewpoint vs. illumination sequences")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "accuracy_viewpoint_vs_illumination.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: precision vs pixel threshold (proxy PR curve) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        ys = []
        for t in THRESHOLDS:
            vals = [r[f"precision_at_{t}px"] for r in rows if r["method"] == method]
            ys.append(np.mean(vals) if vals else 0.0)
        ax.plot(THRESHOLDS, ys, marker="o", label=method.upper(), color=colors[method])
    ax.set_xlabel("Correctness threshold (pixels)")
    ax.set_ylabel("Mean matching precision")
    ax.set_title("Precision vs. correctness threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "precision_recall.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: raw number of matches + runtime, SIFT vs ORB ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for method in methods:
        vals = [r["num_matches"] for r in rows if r["method"] == method]
        axes[0].boxplot(vals, positions=[methods.index(method)], widths=0.5)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([m.upper() for m in methods])
    axes[0].set_ylabel("Number of matches per image pair")
    axes[0].set_title("Raw match counts")
    axes[0].grid(alpha=0.3, axis="y")

    time_means = []
    for method in methods:
        vals = [r["time_sec"] for r in rows if r["method"] == method]
        time_means.append(np.mean(vals) if vals else 0.0)
    axes[1].bar([m.upper() for m in methods], time_means, color=[colors[m] for m in methods])
    axes[1].set_ylabel("Mean time per pair (s)")
    axes[1].set_title("Runtime (detect + describe + match)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "num_matches.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "hpatches-sequences-release"),
        help="Path to extracted HPatches sequences",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(__file__),
        help="Where to write results/ and figures/ subfolders",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N sequences (useful for a quick test run)",
    )
    args = parser.parse_args()
    run_all(args.data_dir, args.out_dir, args.limit)
