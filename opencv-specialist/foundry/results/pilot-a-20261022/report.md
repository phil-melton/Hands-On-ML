# Teacher pilot yield: run `pilot-a-20261022`

Teacher: `RedHatAI/Qwen3.8-27B-INT4` (license apache-2.0; quantization none; GPU A100). 200 of 200 planned specs graded, 800 samples; the specs share 28 distinct prompts.

## Yield per family

| family | subset | specs | samples | Tier A | Tier B | fail | kept | A / spec | distinct | solved % | no code | cut off |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | timeless | 29 | 116 | 110 | 5 | 1 | 58 | 2.00 | 51 | 100.0 | 0 | 0 |
| hough_segments | 5.0 | 29 | 116 | 29 | 27 | 60 | 27 | 0.93 | 27 | 58.6 | 0 | 0 |
| perspective_rectify | timeless | 29 | 116 | 111 | 1 | 4 | 58 | 2.00 | 58 | 100.0 | 0 | 0 |
| feature_homography | 5.0 | 29 | 116 | 81 | 0 | 35 | 57 | 1.97 | 57 | 100.0 | 0 | 0 |
| histogram_stats | 5.0 | 28 | 112 | 102 | 1 | 9 | 56 | 2.00 | 56 | 100.0 | 0 | 0 |
| components_stats | 5.0 | 28 | 112 | 87 | 0 | 25 | 56 | 2.00 | 56 | 100.0 | 0 | 0 |
| text_layout | 5.0 | 28 | 112 | 16 | 62 | 34 | 15 | 0.54 | 15 | 39.3 | 0 | 11 |
| **all** | | 200 | 800 | 536 | 96 | 168 | 327 | 1.64 | 320 | 85.5 | 0 | 11 |

A / spec = Tier-A examples kept (at most 2 per spec, distinct AST-normalized code) per spec. 536 Tier-A samples gave 327 kept; 5 were AST duplicates within a spec, 204 over the cap of 2. distinct = kept examples distinct by (prompt, normalized code) across all specs (320): specs of one family and phrasing share a prompt.

## Pass rate by gate (% of samples; each gate on its own)

| family | exec-only | lint | manufactured | metamorphic | all four (Tier A) |
|---|---:|---:|---:|---:|---:|
| contour_area | 100.0 | 100.0 | 95.7 | 99.1 | 94.8 |
| hough_segments | 94.8 | 98.3 | 42.2 | 50.0 | 25.0 |
| perspective_rectify | 99.1 | 100.0 | 97.4 | 96.6 | 95.7 |
| feature_homography | 70.7 | 99.1 | 70.7 | 70.7 | 69.8 |
| histogram_stats | 95.5 | 100.0 | 91.1 | 92.0 | 91.1 |
| components_stats | 88.4 | 100.0 | 77.7 | 78.6 | 77.7 |
| text_layout | 90.2 | 98.2 | 14.3 | 69.6 | 14.3 |
| **all** | 91.2 | 99.4 | 70.0 | 79.5 | 67.0 |

## Failure classes of the non-Tier-A samples

| family | api | convention | algorithm | lint | format | timeout | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| hough_segments | 1 | 9 | 74 | 2 | 0 | 0 | 1 |
| perspective_rectify | 0 | 2 | 1 | 0 | 0 | 0 | 2 |
| feature_homography | 7 | 1 | 0 | 1 | 0 | 0 | 26 |
| histogram_stats | 4 | 3 | 3 | 0 | 0 | 0 | 0 |
| components_stats | 7 | 7 | 11 | 0 | 0 | 0 | 0 |
| text_layout | 0 | 8 | 85 | 0 | 2 | 0 | 1 |

## Cost

GPU seconds = load + generation from timing.json (+ 0 s overhead given): 1266 s at 5.4 CU/h = 1.899 CU = $0.19 at $0.1/CU. Decode alone: 0.991 CU; 867 output tokens/s. Session wall clock (new to stop): 1320 s = 1.981 CU; predicted 35.0 min, 3.2 CU. 
Seconds are split over families by output tokens.

| family | output tokens | tokens / sample | GPU s | CU | $ | kept | CU / Tier-A | $ / Tier-A | CU / distinct | CU / Tier-A incl. repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 33143 | 286 | 73 | 0.110 | 0.01 | 58 | 0.002 | 0.0002 | 0.002 | 0.002 |
| hough_segments | 90414 | 779 | 200 | 0.300 | 0.03 | 27 | 0.011 | 0.0011 | 0.011 | 0.024 |
| perspective_rectify | 86986 | 750 | 192 | 0.289 | 0.03 | 58 | 0.005 | 0.0005 | 0.005 | 0.005 |
| feature_homography | 53008 | 457 | 117 | 0.176 | 0.02 | 57 | 0.003 | 0.0003 | 0.003 | 0.003 |
| histogram_stats | 74942 | 669 | 166 | 0.249 | 0.02 | 56 | 0.004 | 0.0004 | 0.004 | 0.004 |
| components_stats | 94258 | 842 | 209 | 0.313 | 0.03 | 56 | 0.006 | 0.0006 | 0.006 | 0.006 |
| text_layout | 139624 | 1247 | 309 | 0.463 | 0.05 | 15 | 0.031 | 0.0031 | 0.031 | 0.064 |
| **all** | 572375 | 715 | 1266 | 1.899 | 0.19 | 327 | 0.006 | 0.0006 | 0.006 | 0.009 |

## Repair (one round)

| family | repair prompts | samples | repaired to Tier A | success % | repair pairs (Tier A) | repair CU |
|---|---:|---:|---:|---:|---:|---:|
| contour_area | 0 | 0 | 0 | - | 0 | 0.000 |
| hough_segments | 38 | 38 | 3 | 7.9 | 3 | 0.427 |
| perspective_rectify | 0 | 0 | 0 | - | 0 | 0.000 |
| feature_homography | 2 | 2 | 1 | 50.0 | 1 | 0.017 |
| histogram_stats | 0 | 0 | 0 | - | 0 | 0.000 |
| components_stats | 0 | 0 | 0 | - | 0 | 0.000 |
| text_layout | 48 | 48 | 2 | 4.2 | 2 | 0.619 |
| **all** | 88 | 88 | 6 | 6.8 | 6 | 1.063 |

| failure class | prompts | repaired to Tier A | success % |
|---|---:|---:|---:|
| algorithm | 76 | 4 | 5.3 |
| api | 1 | 0 | 0.0 |
| convention | 8 | 1 | 12.5 |
| format | 1 | 0 | 0.0 |
| other | 2 | 1 | 50.0 |
