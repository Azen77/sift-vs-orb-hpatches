"""
Draws SIFT vs ORB matches side-by-side for a chosen sequence and image pair --
useful for a report/demo (visually shows correct vs incorrect matches).

Usage:
    python visualize_matches.py --seq v_synthetic --pair 5 --data-dir data/hpatches-sequences-release
"""

import argparse
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching_pipeline import (
    load_sequence, get_detector, detect_and_describe, match_descriptors,
    evaluate_matches,
)


def draw_matches_for_method(images, homographies, method, pair_n, threshold=3):
    detector = get_detector(method)
    kps1, descs1 = detect_and_describe(detector, images[1])
    kpsN, descsN = detect_and_describe(detector, images[pair_n])

    matches = match_descriptors(descs1, descsN, method)
    H = homographies[pair_n]

    # classify each match as correct/incorrect for coloring
    correct_flags = []
    for m in matches:
        from matching_pipeline import warp_point
        import numpy as np
        pt1 = kps1[m.queryIdx].pt
        pt2 = kpsN[m.trainIdx].pt
        warped = warp_point(pt1, H)
        dist = np.hypot(warped[0] - pt2[0], warped[1] - pt2[1])
        correct_flags.append(dist <= threshold)

    # draw correct in green, incorrect in red -- do two passes
    img_correct = [m for m, c in zip(matches, correct_flags) if c]
    img_incorrect = [m for m, c in zip(matches, correct_flags) if not c]

    vis = cv2.drawMatches(
        images[1], kps1, images[pair_n], kpsN, img_correct, None,
        matchColor=(0, 200, 0), singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = cv2.drawMatches(
        images[1], kps1, images[pair_n], kpsN, img_incorrect, vis,
        matchColor=(0, 0, 220), singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
        | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    n_correct = len(img_correct)
    n_total = len(matches)
    return vis, n_correct, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="Sequence folder name, e.g. v_synthetic")
    parser.add_argument("--pair", type=int, default=5, help="Target image index (2-6)")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "hpatches-sequences-release"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "figures", "match_visualization.png"),
    )
    args = parser.parse_args()

    seq_dir = os.path.join(args.data_dir, args.seq)
    images, homographies = load_sequence(seq_dir)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for ax, method in zip(axes, ("sift", "orb")):
        vis, n_correct, n_total = draw_matches_for_method(images, homographies, method, args.pair)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        ax.imshow(vis_rgb)
        ax.set_title(
            f"{method.upper()}: {n_correct}/{n_total} correct matches "
            f"(green=correct, red=incorrect) | pair 1->{args.pair}"
        )
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()
