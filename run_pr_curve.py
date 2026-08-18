"""
Traces a TRUE precision-recall curve for SIFT and ORB by sweeping the ratio-test
threshold (the parameter that actually trades precision for recall), rather than
the pixel correctness threshold used in run_evaluation.py.

At each ratio value:
- A looser ratio (closer to 1.0) accepts more candidate matches -> more true
  positives found (higher recall), but also more false positives (lower precision).
- A stricter ratio (closer to 0.5) accepts fewer matches -> higher precision,
  but misses some correct correspondences (lower recall).

Recall's denominator (total possible correspondences) is computed once per pair
via matching_pipeline.count_ground_truth_correspondences, independent of ratio.

Usage:
    python run_pr_curve.py [--data-dir DATA_DIR] [--out-dir OUT_DIR] [--limit N]

Outputs:
    results/pr_curve_results.csv     -- one row per (sequence, pair, method, ratio)
    figures/precision_recall_curve.png
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matching_pipeline import process_sequence_pr_sweep
from run_evaluation import find_sequences

RATIOS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
PIXEL_THRESHOLD = 3  # fixed correctness threshold for this sweep


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

    print(f"Found {len(sequences)} sequences. Sweeping ratio threshold for SIFT and ORB...")

    rows = []
    for si, seq_dir in enumerate(sequences):
        seq_name = os.path.basename(seq_dir)
        seq_type = "viewpoint" if seq_name.startswith("v_") else "illumination"

        for method in ("sift", "orb"):
            try:
                sweep_results = process_sequence_pr_sweep(
                    seq_dir, method, RATIOS, pixel_threshold=PIXEL_THRESHOLD
                )
            except Exception as e:
                print(f"  [skip] {seq_name} ({method}): {e}")
                continue

            for r in sweep_results:
                rows.append({
                    "sequence": seq_name,
                    "seq_type": seq_type,
                    "method": method,
                    "pair": f"1_{r['pair'][1]}",
                    "baseline_idx": r["pair"][1],
                    "ratio": r["ratio"],
                    "num_matches": r["num_matches"],
                    "num_correct": r["num_correct"],
                    "num_ground_truth": r["num_ground_truth"],
                })

        if (si + 1) % 10 == 0 or si == len(sequences) - 1:
            print(f"  processed {si + 1}/{len(sequences)} sequences")

    csv_path = os.path.join(results_dir, "pr_curve_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote raw sweep results to {csv_path}")

    make_pr_curve(rows, figures_dir)
    print(f"Wrote PR curve to {figures_dir}")
    return rows


def make_pr_curve(rows, figures_dir):
    methods = ["sift", "orb"]
    colors = {"sift": "tab:blue", "orb": "tab:orange"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for method in methods:
        precisions, recalls = [], []
        for ratio in RATIOS:
            subset = [r for r in rows if r["method"] == method and r["ratio"] == ratio]
            total_matches = sum(r["num_matches"] for r in subset)
            total_correct = sum(r["num_correct"] for r in subset)
            total_gt = sum(r["num_ground_truth"] for r in subset)

            precision = total_correct / total_matches if total_matches > 0 else 0.0
            recall = total_correct / total_gt if total_gt > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)

        # sort by recall so the curve is drawn left-to-right
        order = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[order]
        precisions_sorted = np.array(precisions)[order]

        ax.plot(recalls_sorted, precisions_sorted, marker="o", label=method.upper(),
                 color=colors[method])

    ax.set_xlabel("Recall (correct matches / total possible correspondences)")
    ax.set_ylabel("Precision (correct matches / total matches made)")
    ax.set_title(f"Precision-Recall curve (pooled across all sequences, @{PIXEL_THRESHOLD}px)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "hpatches-sequences-release"),
    )
    parser.add_argument("--out-dir", default=os.path.dirname(__file__))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N sequences (this script is slower than "
             "run_evaluation.py since it reruns matching at 11 ratio values per pair)",
    )
    args = parser.parse_args()
    run_all(args.data_dir, args.out_dir, args.limit)
