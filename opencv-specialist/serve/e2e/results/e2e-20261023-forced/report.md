# E2E report: run `e2e-20261023-forced`

## Success rate (task passes when every key passes)

| arm | tasks run | passed | rate | Wilson 95 % CI | held-out-family tasks passed | answer status |
|---|---|---|---|---|---|---|
| alone | 20 | 20 | 100.0 % | 83.9-100.0 % | 6/6 | ok 20 |
| specialist | 20 | 14 | 70.0 % | 48.1-85.5 % | 5/6 | ok 20 |

## Paired comparison over tasks run in both arms

n = 20: both pass 14, alone only 6, specialist only 0, neither 0. Exact two-sided sign test on the 6 discordant tasks: p = 0.031.

## Orchestrator tokens

`total` = input + cache_creation + cache_read + output (every token processed or produced), `fresh` = input + cache_creation + output, `output` = generated tokens. Per task, from the final `result` event.

| arm | metric | mean | median | total |
|---|---|---|---|---|
| alone | total | 19,419 | 15,401 | 388,385 |
| alone | fresh | 6,311 | 5,714 | 126,211 |
| alone | output | 1,037 | 841 | 20,741 |
| specialist | total | 34,927 | 25,810 | 698,545 |
| specialist | fresh | 10,221 | 8,868 | 204,419 |
| specialist | output | 1,278 | 824 | 25,560 |

Paired per-task ratio specialist / alone:

| metric | n | median | IQR | mean of ratios | ratio of means | tasks with fewer / more | sign test p |
|---|---|---|---|---|---|---|---|
| total | 20 | 1.57 | 1.52-1.95 | 1.95 | 1.80 | 2 / 18 | 0.000 |
| fresh | 20 | 1.42 | 1.31-1.75 | 1.65 | 1.62 | 1 / 19 | 0.000 |
| output | 20 | 0.95 | 0.85-1.42 | 1.28 | 1.23 | 13 / 7 | 0.263 |

## Wall time (seconds per task)

| arm | mean | median | total |
|---|---|---|---|
| alone | 14.8 | 12.5 | 296 |
| specialist | 87.0 | 38.4 | 1,740 |

Paired wall-time ratio specialist / alone: median 3.01, IQR 2.24-4.23 (n = 20).

## Tool use

| arm | tool calls / task | Bash / task | turns / task | tool errors / task | opencv_solve calls (total) | verified | tasks using it |
|---|---|---|---|---|---|---|---|
| alone | 1.4 | 1.4 | 2.5 | 0.15 | 0 | 0 | 0 |
| specialist | 2.8 | 0.0 | 3.8 | 0.00 | 35 | 27 | 20 |

## Run hygiene

- alone: stopped_by {'completed': 20}; isolation problems in none; transcripts that touched the repo or other runs: ['e2e-09-board-corner-lookup', 'e2e-13-field-label-fit']; permission denials: none.
- specialist: stopped_by {'completed': 20}; isolation problems in none; transcripts that touched the repo or other runs: none; permission denials: none.

## Criterion 4 inputs (frozen 2026-10-22: specialist >= alone on success, with fewer orchestrator tokens)

- specialist success rate >= alone: False
- median paired token ratio (total): 1.57; mean tokens specialist < alone: False
- these are the mechanical inputs; the decision stays with the project owner (CLAUDE.md rule 10).

## Per task

| task | steps | families (* held out) | alone | tokens | s | specialist | tokens | s | solve calls (verified) |
|---|---|---|---|---|---|---|---|---|---|
| e2e-01-card-geometry | 3 | perspective_rectify | PASS 4/4 | 15,500 | 10 | PASS 4/4 | 23,386 | 32 | 1 (1) |
| e2e-02-card-blobs | 4 | perspective_rectify, components_stats, contour_area | PASS 4/4 | 16,336 | 11 | PASS 4/4 | 26,259 | 38 | 1 (1) |
| e2e-03-strokes-longest | 3 | hough_segments | PASS 4/4 | 15,134 | 16 | PASS 4/4 | 23,277 | 29 | 1 (1) |
| e2e-04-strokes-orientation | 3 | hough_segments | PASS 3/3 | 37,936 | 28 | PASS 3/3 | 36,941 | 117 | 2 (1) |
| e2e-05-component-hist | 4 | components_stats, histogram_stats | PASS 6/6 | 15,000 | 10 | FAIL 4/6 | 23,752 | 27 | 1 (1) |
| e2e-06-parts-colour-area | 4 | color_segmentation*, contour_area | PASS 3/3 | 15,302 | 12 | PASS 3/3 | 38,036 | 43 | 2 (2) |
| e2e-07-colour-dominant-blob | 4 | color_segmentation*, components_stats | PASS 6/6 | 15,610 | 12 | PASS 6/6 | 27,589 | 42 | 1 (1) |
| e2e-08-board-pitch-scale | 3 | chessboard_corners* | PASS 5/5 | 15,185 | 10 | PASS 5/5 | 57,797 | 236 | 3 (1) |
| e2e-09-board-corner-lookup | 3 | chessboard_corners* | PASS 4/4 | 24,583 | 16 | PASS 4/4 | 88,339 | 524 | 4 (1) |
| e2e-10-board-scale-parts | 4 | chessboard_corners*, color_segmentation*, contour_area | PASS 3/3 | 55,252 | 30 | PASS 3/3 | 48,508 | 135 | 4 (3) |
| e2e-11-view-roi-map | 3 | feature_homography | PASS 4/4 | 15,670 | 10 | PASS 4/4 | 24,101 | 23 | 1 (1) |
| e2e-12-view-pose | 3 | feature_homography | PASS 3/3 | 15,299 | 10 | PASS 3/3 | 23,732 | 27 | 1 (1) |
| e2e-13-field-label-fit | 3 | text_layout | PASS 4/4 | 15,669 | 15 | FAIL 3/4 | 59,496 | 152 | 3 (2) |
| e2e-14-field-label-shrink | 3 | text_layout | PASS 4/4 | 15,820 | 16 | FAIL 3/4 | 53,676 | 137 | 3 (3) |
| e2e-15-exposure-outliers | 3 | histogram_stats | PASS 4/4 | 14,621 | 15 | FAIL 3/4 | 22,363 | 22 | 1 (1) |
| e2e-16-exposure-compare | 3 | histogram_stats | PASS 6/6 | 14,958 | 12 | FAIL 5/6 | 25,362 | 26 | 2 (2) |
| e2e-17-shape-rank | 3 | contour_area | PASS 4/4 | 14,614 | 11 | PASS 4/4 | 21,106 | 17 | 1 (1) |
| e2e-18-mask-table | 3 | components_stats | PASS 6/6 | 14,971 | 22 | PASS 6/6 | 24,102 | 39 | 1 (1) |
| e2e-19-card-strokes | 4 | perspective_rectify, hough_segments | PASS 4/4 | 25,686 | 15 | PASS 4/4 | 27,052 | 44 | 1 (1) |
| e2e-20-palette-luma | 4 | color_segmentation*, histogram_stats | PASS 4/4 | 15,239 | 13 | FAIL 3/4 | 23,671 | 33 | 1 (1) |
