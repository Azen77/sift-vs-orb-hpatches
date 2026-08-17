"""
Core pipeline: for a given HPatches sequence, detect+describe keypoints with a chosen
method (SIFT or ORB), match image 1 against image N, and check correctness of each
match using the ground-truth homography.
"""

import os
import numpy as np
import cv2


def load_homography(path):
    """HPatches stores H_1_2 ... H_1_6 as plain-text 3x3 matrices."""
    return np.loadtxt(path)


def load_sequence(seq_dir):
    """
    Returns:
        images: dict {1: img1, 2: img2, ..., 6: img6}   (BGR, as read by cv2)
        homographies: dict {2: H_1_2, 3: H_1_3, ..., 6: H_1_6}  (maps img1 -> imgN)
    """
    images = {}
    for i in range(1, 7):
        # HPatches ships .ppm images
        for ext in (".ppm", ".png", ".jpg"):
            p = os.path.join(seq_dir, f"{i}{ext}")
            if os.path.exists(p):
                images[i] = cv2.imread(p)
                break
        if i not in images:
            raise FileNotFoundError(f"Could not find image {i} in {seq_dir}")

    homographies = {}
    for i in range(2, 7):
        p = os.path.join(seq_dir, f"H_1_{i}")
        if os.path.exists(p):
            homographies[i] = load_homography(p)
    return images, homographies


def get_detector(method):
    """method: 'sift' or 'orb'"""
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create()
    elif method == "orb":
        return cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_and_describe(detector, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs


def match_descriptors(descs1, descs2, method, ratio=0.75):
    """
    Returns list of cv2.DMatch that survive Lowe's ratio test.
    SIFT descriptors are float -> use L2 norm (NORM_L2).
    ORB descriptors are binary -> use Hamming distance (NORM_HAMMING).
    """
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return []

    norm = cv2.NORM_L2 if method.lower() == "sift" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(descs1, descs2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def warp_point(pt, H):
    """Apply homography H to a single (x, y) point."""
    x, y = pt
    vec = np.array([x, y, 1.0])
    warped = H @ vec
    warped /= warped[2]
    return warped[0], warped[1]


def evaluate_matches(kps1, kps2, matches, H, threshold):
    """
    For each match, warp the keypoint from image 1 into image 2's frame using the
    ground-truth homography H, and check if it lands within `threshold` pixels of
    the keypoint it was matched to in image 2.

    Returns: (num_correct, num_total)
    """
    if len(matches) == 0:
        return 0, 0

    num_correct = 0
    for m in matches:
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        warped = warp_point(pt1, H)
        dist = np.hypot(warped[0] - pt2[0], warped[1] - pt2[1])
        if dist <= threshold:
            num_correct += 1

    return num_correct, len(matches)


def process_sequence(seq_dir, method, thresholds=(1, 2, 3, 5, 10)):
    """
    Runs detection+matching+evaluation for all pairs (1,2)...(1,6) in one sequence.

    Returns a list of dicts, one per pair:
        {
          'pair': (1, N),
          'num_kp1': ..., 'num_kp2': ...,
          'num_matches': ...,
          'correct_at_threshold': {1: c1, 2: c2, ...},
        }
    """
    images, homographies = load_sequence(seq_dir)
    detector = get_detector(method)

    kps_cache = {}
    descs_cache = {}
    for i, img in images.items():
        kps, descs = detect_and_describe(detector, img)
        kps_cache[i] = kps
        descs_cache[i] = descs

    results = []
    for n in range(2, 7):
        if n not in homographies:
            continue
        matches = match_descriptors(descs_cache[1], descs_cache[n], method)
        H = homographies[n]

        correct_at_threshold = {}
        for t in thresholds:
            c, total = evaluate_matches(kps_cache[1], kps_cache[n], matches, H, t)
            correct_at_threshold[t] = c

        results.append({
            "pair": (1, n),
            "num_kp1": len(kps_cache[1]),
            "num_kp2": len(kps_cache[n]),
            "num_matches": len(matches),
            "correct_at_threshold": correct_at_threshold,
        })

    return results
