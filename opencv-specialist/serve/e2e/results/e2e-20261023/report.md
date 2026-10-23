# E2E report: run `e2e-20261023`

## Success rate (task passes when every key passes)

| arm | tasks run | passed | rate | Wilson 95 % CI | held-out-family tasks passed | answer status |
|---|---|---|---|---|---|---|
| alone | 20 | 20 | 100.0 % | 83.9-100.0 % | 6/6 | ok 20 |
| specialist | 20 | 20 | 100.0 % | 83.9-100.0 % | 6/6 | ok 20 |

## Paired comparison over tasks run in both arms

n = 20: both pass 20, alone only 0, specialist only 0, neither 0. Exact two-sided sign test on the 0 discordant tasks: p = 1.000.

## Orchestrator tokens

`total` = input + cache_creation + cache_read + output (every token processed or produced), `fresh` = input + cache_creation + output, `output` = generated tokens. Per task, from the final `result` event.

| arm | metric | mean | median | total |
|---|---|---|---|---|
| alone | total | 19,419 | 15,401 | 388,385 |
| alone | fresh | 6,311 | 5,714 | 126,211 |
| alone | output | 1,037 | 841 | 20,741 |
| specialist | total | 18,809 | 17,787 | 376,171 |
| specialist | fresh | 5,986 | 5,744 | 119,715 |
| specialist | output | 933 | 888 | 18,656 |

Paired per-task ratio specialist / alone:

| metric | n | median | IQR | mean of ratios | ratio of means | tasks with fewer / more | sign test p |
|---|---|---|---|---|---|---|---|
| total | 20 | 1.15 | 1.14-1.15 | 1.06 | 0.97 | 3 / 17 | 0.003 |
| fresh | 20 | 1.00 | 0.99-1.01 | 0.97 | 0.95 | 10 / 10 | 1.000 |
| output | 20 | 1.00 | 0.97-1.02 | 0.97 | 0.90 | 8 / 11 | 0.648 |

## Wall time (seconds per task)

| arm | mean | median | total |
|---|---|---|---|
| alone | 14.8 | 12.5 | 296 |
| specialist | 12.8 | 11.6 | 256 |

Paired wall-time ratio specialist / alone: median 0.99, IQR 0.80-1.09 (n = 20).

## Tool use

| arm | tool calls / task | Bash / task | turns / task | tool errors / task | opencv_solve calls (total) | verified | tasks using it |
|---|---|---|---|---|---|---|---|
| alone | 1.4 | 1.4 | 2.5 | 0.15 | 0 | 0 | 0 |
| specialist | 1.1 | 1.1 | 2.1 | 0.05 | 0 | 0 | 0 |

## Run hygiene

- alone: stopped_by {'completed': 20}; isolation problems in none; transcripts that touched the repo or other runs: ['e2e-09-board-corner-lookup', 'e2e-13-field-label-fit']; permission denials: none.
- specialist: stopped_by {'completed': 20}; isolation problems in none; transcripts that touched the repo or other runs: none; permission denials: none.

## Criterion 4 inputs (frozen 2026-10-22: specialist >= alone on success, with fewer orchestrator tokens)

- specialist success rate >= alone: True
- median paired token ratio (total): 1.15; mean tokens specialist < alone: True
- these are the mechanical inputs; the decision stays with the project owner (CLAUDE.md rule 10).

## Per task

| task | steps | families (* held out) | alone | tokens | s | specialist | tokens | s | solve calls (verified) |
|---|---|---|---|---|---|---|---|---|---|
| e2e-01-card-geometry | 3 | perspective_rectify | PASS 4/4 | 15,500 | 10 | PASS 4/4 | 17,829 | 11 | 0 (0) |
| e2e-02-card-blobs | 4 | perspective_rectify, components_stats, contour_area | PASS 4/4 | 16,336 | 11 | PASS 4/4 | 18,444 | 16 | 0 (0) |
| e2e-03-strokes-longest | 3 | hough_segments | PASS 4/4 | 15,134 | 16 | PASS 4/4 | 17,408 | 12 | 0 (0) |
| e2e-04-strokes-orientation | 3 | hough_segments | PASS 3/3 | 37,936 | 28 | PASS 3/3 | 17,255 | 11 | 0 (0) |
| e2e-05-component-hist | 4 | components_stats, histogram_stats | PASS 6/6 | 15,000 | 10 | PASS 6/6 | 17,201 | 10 | 0 (0) |
| e2e-06-parts-colour-area | 4 | color_segmentation*, contour_area | PASS 3/3 | 15,302 | 12 | PASS 3/3 | 17,532 | 11 | 0 (0) |
| e2e-07-colour-dominant-blob | 4 | color_segmentation*, components_stats | PASS 6/6 | 15,610 | 12 | PASS 6/6 | 17,869 | 17 | 0 (0) |
| e2e-08-board-pitch-scale | 3 | chessboard_corners* | PASS 5/5 | 15,185 | 10 | PASS 5/5 | 17,893 | 14 | 0 (0) |
| e2e-09-board-corner-lookup | 3 | chessboard_corners* | PASS 4/4 | 24,583 | 16 | PASS 4/4 | 18,338 | 16 | 0 (0) |
| e2e-10-board-scale-parts | 4 | chessboard_corners*, color_segmentation*, contour_area | PASS 3/3 | 55,252 | 30 | PASS 3/3 | 29,499 | 17 | 0 (0) |
| e2e-11-view-roi-map | 3 | feature_homography | PASS 4/4 | 15,670 | 10 | PASS 4/4 | 17,936 | 10 | 0 (0) |
| e2e-12-view-pose | 3 | feature_homography | PASS 3/3 | 15,299 | 10 | PASS 3/3 | 17,745 | 11 | 0 (0) |
| e2e-13-field-label-fit | 3 | text_layout | PASS 4/4 | 15,669 | 15 | PASS 4/4 | 18,089 | 12 | 0 (0) |
| e2e-14-field-label-shrink | 3 | text_layout | PASS 4/4 | 15,820 | 16 | PASS 4/4 | 18,177 | 11 | 0 (0) |
| e2e-15-exposure-outliers | 3 | histogram_stats | PASS 4/4 | 14,621 | 15 | PASS 4/4 | 16,891 | 16 | 0 (0) |
| e2e-16-exposure-compare | 3 | histogram_stats | PASS 6/6 | 14,958 | 12 | PASS 6/6 | 17,128 | 10 | 0 (0) |
| e2e-17-shape-rank | 3 | contour_area | PASS 4/4 | 14,614 | 11 | PASS 4/4 | 16,896 | 13 | 0 (0) |
| e2e-18-mask-table | 3 | components_stats | PASS 6/6 | 14,971 | 22 | PASS 6/6 | 17,185 | 10 | 0 (0) |
| e2e-19-card-strokes | 4 | perspective_rectify, hough_segments | PASS 4/4 | 25,686 | 15 | PASS 4/4 | 29,347 | 17 | 0 (0) |
| e2e-20-palette-luma | 4 | color_segmentation*, histogram_stats | PASS 4/4 | 15,239 | 13 | PASS 4/4 | 17,509 | 11 | 0 (0) |
