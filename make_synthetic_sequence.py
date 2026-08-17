"""
Creates a fake HPatches-style sequence folder (v_test/) from a single real-ish image
by applying known homographies. Useful to sanity-check matching_pipeline.py and
run_evaluation.py without downloading the full dataset.
"""

import os
import numpy as np
import cv2


def make_textured_image(size=480):
    """Synthetic image with lots of distinctive corners/blobs (good for SIFT/ORB)."""
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 30, dtype=np.uint8)

    # checkerboard-ish base for structure
    for y in range(0, size, 40):
        for x in range(0, size, 40):
            if (x // 40 + y // 40) % 2 == 0:
                img[y:y + 40, x:x + 40] = (60, 60, 60)

    # random circles/rectangles for distinctive keypoints
    for _ in range(150):
        x, y = rng.integers(0, size, 2)
        r = rng.integers(4, 14)
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r), color, -1)

    return img


def make_sequence(out_dir, size=480):
    os.makedirs(out_dir, exist_ok=True)
    base = make_textured_image(size)
    cv2.imwrite(os.path.join(out_dir, "1.ppm"), base)

    rng = np.random.default_rng(1)
    for n in range(2, 7):
        # increasing viewpoint distortion for higher n
        strength = (n - 1) * 8  # pixels of corner jitter
        src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        jitter = rng.uniform(-strength, strength, size=(4, 2)).astype(np.float32)
        dst = src + jitter

        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(base, H, (size, size), borderValue=(30, 30, 30))

        # mild illumination change too
        warped = cv2.convertScaleAbs(warped, alpha=1.0 - 0.03 * n, beta=5 * n)

        cv2.imwrite(os.path.join(out_dir, f"{n}.ppm"), warped)
        np.savetxt(os.path.join(out_dir, f"H_1_{n}"), H, fmt="%.10f")

    print(f"Synthetic sequence written to {out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    make_sequence(os.path.join(here, "data", "hpatches-sequences-release", "v_synthetic"))
