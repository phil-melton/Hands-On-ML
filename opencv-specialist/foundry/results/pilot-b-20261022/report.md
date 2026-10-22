# Teacher pilot yield: run `pilot-b-20261022`

Teacher: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` (license apache-2.0; quantization awq; GPU L4). 200 of 200 planned specs graded, 800 samples; the specs share 28 distinct prompts.

## Yield per family

| family | subset | specs | samples | Tier A | Tier B | fail | kept | A / spec | distinct | solved % | no code | cut off |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | timeless | 29 | 116 | 7 | 104 | 5 | 7 | 0.24 | 7 | 24.1 | 0 | 0 |
| hough_segments | 5.0 | 29 | 116 | 0 | 0 | 116 | 0 | 0.00 | 0 | 0.0 | 0 | 0 |
| perspective_rectify | timeless | 29 | 116 | 2 | 8 | 106 | 2 | 0.07 | 2 | 6.9 | 0 | 2 |
| feature_homography | 5.0 | 29 | 116 | 82 | 12 | 22 | 52 | 1.79 | 47 | 93.1 | 0 | 0 |
| histogram_stats | 5.0 | 28 | 112 | 39 | 4 | 69 | 30 | 1.07 | 28 | 71.4 | 0 | 0 |
| components_stats | 5.0 | 28 | 112 | 9 | 0 | 103 | 9 | 0.32 | 9 | 28.6 | 0 | 0 |
| text_layout | 5.0 | 28 | 112 | 1 | 87 | 24 | 1 | 0.04 | 1 | 3.6 | 0 | 0 |
| **all** | | 200 | 800 | 140 | 215 | 445 | 101 | 0.51 | 94 | 32.5 | 0 | 2 |

A / spec = Tier-A examples kept (at most 2 per spec, distinct AST-normalized code) per spec. 140 Tier-A samples gave 101 kept; 1 were AST duplicates within a spec, 38 over the cap of 2. distinct = kept examples distinct by (prompt, normalized code) across all specs (94): specs of one family and phrasing share a prompt.

## Pass rate by gate (% of samples; each gate on its own)

| family | exec-only | lint | manufactured | metamorphic | all four (Tier A) |
|---|---:|---:|---:|---:|---:|
| contour_area | 95.7 | 100.0 | 6.0 | 95.7 | 6.0 |
| hough_segments | 62.1 | 96.6 | 0.0 | 0.0 | 0.0 |
| perspective_rectify | 91.4 | 99.1 | 3.4 | 18.1 | 1.7 |
| feature_homography | 87.1 | 100.0 | 73.3 | 81.0 | 70.7 |
| histogram_stats | 96.4 | 99.1 | 34.8 | 38.4 | 34.8 |
| components_stats | 77.7 | 95.5 | 8.0 | 67.9 | 8.0 |
| text_layout | 85.7 | 100.0 | 0.9 | 78.6 | 0.9 |
| **all** | 85.1 | 98.6 | 18.1 | 54.1 | 17.5 |

## Failure classes of the non-Tier-A samples

| family | api | convention | algorithm | lint | format | timeout | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 0 | 0 | 104 | 0 | 0 | 0 | 5 |
| hough_segments | 0 | 61 | 47 | 0 | 4 | 0 | 4 |
| perspective_rectify | 0 | 23 | 74 | 0 | 1 | 0 | 16 |
| feature_homography | 0 | 0 | 19 | 0 | 0 | 0 | 15 |
| histogram_stats | 2 | 1 | 69 | 0 | 1 | 0 | 0 |
| components_stats | 0 | 82 | 4 | 0 | 5 | 0 | 12 |
| text_layout | 0 | 0 | 95 | 0 | 0 | 1 | 15 |

## Cost

GPU seconds = load + generation from timing.json (+ 0 s overhead given): 878 s at 1.7 CU/h = 0.415 CU = $0.04 at $0.1/CU. Decode alone: 0.252 CU; 719 output tokens/s. Session wall clock (new to stop): 964 s = 0.455 CU; predicted 46.0 min, 1.3 CU. 
Seconds are split over families by output tokens.

| family | output tokens | tokens / sample | GPU s | CU | $ | kept | CU / Tier-A | $ / Tier-A | CU / distinct | CU / Tier-A incl. repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 23257 | 200 | 53 | 0.025 | 0.00 | 7 | 0.004 | 0.0004 | 0.004 | 0.007 |
| hough_segments | 89593 | 772 | 205 | 0.097 | 0.01 | 0 | none | none | none | none |
| perspective_rectify | 96659 | 833 | 221 | 0.105 | 0.01 | 2 | 0.052 | 0.0052 | 0.052 | 0.076 |
| feature_homography | 50871 | 439 | 116 | 0.055 | 0.01 | 52 | 0.001 | 0.0001 | 0.001 | 0.001 |
| histogram_stats | 26574 | 237 | 61 | 0.029 | 0.00 | 30 | 0.001 | 0.0001 | 0.001 | 0.001 |
| components_stats | 54153 | 484 | 124 | 0.059 | 0.01 | 9 | 0.007 | 0.0007 | 0.007 | 0.013 |
| text_layout | 42379 | 378 | 97 | 0.046 | 0.00 | 1 | 0.046 | 0.0046 | 0.046 | 0.102 |
| **all** | 383486 | 479 | 878 | 0.415 | 0.04 | 101 | 0.004 | 0.0004 | 0.004 | 0.006 |

## Repair (one round)

| family | repair prompts | samples | repaired to Tier A | success % | repair pairs (Tier A) | repair CU |
|---|---:|---:|---:|---:|---:|---:|
| contour_area | 48 | 48 | 0 | 0.0 | 0 | 0.023 |
| hough_segments | 58 | 58 | 0 | 0.0 | 0 | 0.107 |
| perspective_rectify | 58 | 58 | 1 | 1.7 | 1 | 0.125 |
| feature_homography | 8 | 8 | 2 | 25.0 | 2 | 0.010 |
| histogram_stats | 34 | 34 | 28 | 82.4 | 28 | 0.024 |
| components_stats | 54 | 54 | 0 | 0.0 | 0 | 0.063 |
| text_layout | 56 | 56 | 0 | 0.0 | 0 | 0.056 |
| **all** | 316 | 316 | 31 | 9.8 | 31 | 0.408 |

| failure class | prompts | repaired to Tier A | success % |
|---|---:|---:|---:|
| algorithm | 192 | 31 | 16.1 |
| convention | 86 | 0 | 0.0 |
| format | 4 | 0 | 0.0 |
| other | 33 | 0 | 0.0 |
| timeout | 1 | 0 | 0.0 |
