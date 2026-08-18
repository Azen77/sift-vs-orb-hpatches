# Wide-Baseline Feature Matching: SIFT vs. ORB on HPatches

Compares classical feature detectors/descriptors (SIFT, ORB) for image matching
robustness under viewpoint, scale, and illumination changes, using the HPatches
benchmark (patch sequences with known ground-truth homographies).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get the data

```bash
python download_data.py
```

This downloads and extracts the official HPatches "full sequences" release (~1.3GB)
into `data/hpatches-sequences-release/`. Each of the 116 sequence folders contains:

- `1.ppm` ... `6.ppm` — images (1 = reference)
- `H_1_2` ... `H_1_6` — ground-truth homographies mapping image 1 into image N

Sequences starting with `v_` have viewpoint changes; `i_` have illumination changes.

If the download link is ever down, grab the "Full image sequences" zip manually from
https://github.com/hpatches/hpatches-dataset and place it at
`data/hpatches-sequences-release.zip`, then re-run the script — it'll pick up the
local file and just extract it.

**No internet / want to test the pipeline first?** Run:

```bash
python make_synthetic_sequence.py
```

This generates a couple of synthetic sequences with known homographies so you can
verify everything works before downloading the real 1.3GB dataset.

## 2. Run the evaluation

```bash
python run_evaluation.py
```

Runs SIFT and ORB detection + matching (at a fixed ratio-test threshold of 0.75)
over every sequence and every pair (1→2 through 1→6), checks each match's
correctness against the ground-truth homography, and writes:

- `results/raw_results.csv` — one row per (sequence, pair, method) with match
  counts, **precision and recall** at several pixel thresholds, and timing.
  Recall's denominator (`num_ground_truth_at_Npx`) is the number of keypoints in
  image 1 that have a spatially plausible corresponding keypoint in image N,
  independent of whether the descriptor matcher actually found it — i.e. the
  total number of correspondences a perfect matcher could have found.
- `figures/accuracy_by_baseline.png` — mean precision vs. baseline severity (pair index)
- `figures/accuracy_viewpoint_vs_illumination.png` — SIFT vs ORB, grouped by
  viewpoint-change vs illumination-change sequences
- `figures/precision_recall_vs_threshold.png` — precision AND recall vs. pixel
  correctness threshold (note: this is not a full PR curve — it's one operating
  point swept across pixel tolerance, not across the precision/recall tradeoff)
- `figures/num_matches.png` — raw match counts and runtime, SIFT vs ORB

Useful flags:

```bash
python run_evaluation.py --limit 10   # quick test on first 10 sequences
```

## 3. Trace a true precision-recall curve

```bash
python run_pr_curve.py
```

`run_evaluation.py` only varies the *pixel* correctness threshold at a fixed
matcher setting — that's not an actual precision-recall tradeoff. This script
instead sweeps the matcher's **ratio-test threshold** (0.5 → 0.99), which is
the parameter that genuinely trades precision for recall: a looser ratio accepts
more candidate matches (higher recall, more false positives → lower precision),
a stricter one accepts fewer (lower recall, higher precision). Ground-truth
correspondence counts are computed once per pair since they don't depend on the
ratio. Writes:

- `results/pr_curve_results.csv` — one row per (sequence, pair, method, ratio)
- `figures/precision_recall_curve.png` — the actual PR curve, SIFT vs ORB

This is slower than `run_evaluation.py` since it reruns matching at 11 ratio
values per image pair (detection/description is still only done once, cached).
Use `--limit N` for a quick test.

## 4. Visualize matches for one sequence/pair

```bash
python visualize_matches.py --seq v_boat --pair 5
```

Draws SIFT and ORB matches side by side (green = correct per ground-truth homography,
red = incorrect) — good for a report or demo.

## How correctness is judged

For each match between a keypoint in image 1 and a keypoint in image N, the image-1
keypoint is warped into image N's coordinate frame using the ground-truth homography.
If the warped point lands within a pixel threshold (default sweep: 1, 2, 3, 5, 10px)
of the matched keypoint, the match counts as correct.

**Precision** = correct matches / total matches made — of the matches you produced,
how many were right.

**Recall** = correct matches / total possible correspondences — of all the
correspondences that *could* have been found (any keypoint in image 1 with a
spatially plausible corresponding keypoint in image N, whether or not the matcher
found it), how many did you actually find. This denominator is computed by
`count_ground_truth_correspondences()` in `matching_pipeline.py`, independent of
descriptor matching.

`run_evaluation.py` reports precision and recall at a fixed matcher setting, swept
across pixel tolerance. `run_pr_curve.py` sweeps the matcher's ratio-test threshold
instead, to trace the actual precision/recall tradeoff curve.

## Files

| File | Purpose |
|---|---|
| `matching_pipeline.py` | Core: load sequence, detect/describe, match, evaluate one sequence |
| `run_evaluation.py` | Runs the pipeline across all sequences, aggregates precision/recall, plots |
| `run_pr_curve.py` | Sweeps ratio-test threshold to trace a true precision-recall curve |
| `download_data.py` | Fetches + extracts the HPatches sequences dataset |
| `make_synthetic_sequence.py` | Generates fake sequences for pipeline testing |
| `visualize_matches.py` | Draws correct/incorrect matches side by side |

## Extending

- Add another detector (e.g. `cv2.AKAZE_create()`) — just add it to `get_detector()`
  in `matching_pipeline.py` and to the `methods` list in `run_evaluation.py`.
- The ratio-test threshold (0.75) in `match_descriptors()` is worth tuning/reporting
  on separately — lower it for higher precision at the cost of fewer matches.
