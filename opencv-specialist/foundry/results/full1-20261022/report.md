# Teacher pilot yield: run `full1-20261022`

Teacher: `RedHatAI/Qwen3.8-27B-INT4` (license apache-2.0; quantization none; GPU A100). 1372 of 1372 planned specs graded, 7188 samples; the specs share 398 distinct prompts.

## Yield per family

| family | subset | specs | samples | Tier A | Tier B | fail | kept | A / spec | distinct | solved % | no code | cut off |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | timeless | 105 | 420 | 400 | 6 | 14 | 210 | 2.00 | 201 | 100.0 | 0 | 0 |
| hough_segments | 5.0 | 162 | 1296 | 286 | 285 | 725 | 229 | 1.41 | 229 | 85.2 | 0 | 3 |
| perspective_rectify | timeless | 105 | 420 | 408 | 1 | 11 | 210 | 2.00 | 210 | 100.0 | 0 | 0 |
| feature_homography | 5.0 | 107 | 428 | 305 | 2 | 121 | 207 | 1.93 | 207 | 100.0 | 0 | 0 |
| histogram_stats | 5.0 | 105 | 420 | 364 | 1 | 55 | 204 | 1.94 | 204 | 97.1 | 0 | 0 |
| components_stats | 5.0 | 105 | 420 | 274 | 2 | 144 | 187 | 1.78 | 187 | 97.1 | 0 | 0 |
| text_layout | 5.0 | 263 | 2104 | 376 | 936 | 792 | 296 | 1.13 | 296 | 70.7 | 0 | 142 |
| drills | drill | 420 | 1680 | 1224 | 0 | 456 | 682 | 1.62 | 604 | 88.3 | 0 | 16 |
| **all** | | 1372 | 7188 | 3637 | 1233 | 2318 | 2225 | 1.62 | 2138 | 88.6 | 0 | 161 |

A / spec = Tier-A examples kept (at most 2 per spec, distinct AST-normalized code) per spec. 3637 Tier-A samples gave 2225 kept; 142 were AST duplicates within a spec, 1270 over the cap of 2. distinct = kept examples distinct by (prompt, normalized code) across all specs (2138): specs of one family and phrasing share a prompt.

## Pass rate by gate (% of samples; each gate on its own)

| family | exec-only | lint | manufactured | metamorphic | all four (Tier A) |
|---|---:|---:|---:|---:|---:|
| contour_area | 99.3 | 100.0 | 97.9 | 96.7 | 95.2 |
| hough_segments | 92.0 | 98.6 | 37.2 | 45.5 | 22.1 |
| perspective_rectify | 98.8 | 100.0 | 97.6 | 97.4 | 97.1 |
| feature_homography | 72.9 | 99.5 | 71.3 | 71.7 | 71.3 |
| histogram_stats | 93.3 | 100.0 | 86.7 | 86.9 | 86.7 |
| components_stats | 83.6 | 99.8 | 65.5 | 71.9 | 65.2 |
| text_layout | 90.1 | 98.6 | 17.9 | 62.5 | 17.9 |
| drills | 0.0 | 0.0 | 0.0 | 0.0 | 72.9 |
| **all** | 69.2 | 75.9 | 36.5 | 51.4 | 50.6 |

## Failure classes of the non-Tier-A samples

| family | api | convention | algorithm | lint | format | timeout | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 1 | 0 | 17 | 0 | 0 | 0 | 2 |
| hough_segments | 11 | 173 | 801 | 13 | 5 | 1 | 6 |
| perspective_rectify | 0 | 9 | 1 | 0 | 0 | 0 | 2 |
| feature_homography | 31 | 11 | 7 | 0 | 5 | 0 | 69 |
| histogram_stats | 19 | 17 | 20 | 0 | 0 | 0 | 0 |
| components_stats | 44 | 57 | 43 | 1 | 0 | 0 | 1 |
| text_layout | 1 | 158 | 1505 | 0 | 29 | 0 | 35 |
| drills | 47 | 242 | 121 | 1 | 3 | 1 | 41 |

## Cost

GPU seconds = load + generation from timing.json (+ 0 s overhead given): 6959 s at 5.4 CU/h = 10.438 CU = $1.04 at $0.1/CU. Decode alone: 9.534 CU; 881 output tokens/s. Session wall clock (new to stop): 7023 s = 10.534 CU; predicted 143.0 min, 12.9 CU. 
Seconds are split over families by output tokens.

| family | output tokens | tokens / sample | GPU s | CU | $ | kept | CU / Tier-A | $ / Tier-A | CU / distinct | CU / Tier-A incl. repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contour_area | 119320 | 284 | 148 | 0.222 | 0.02 | 210 | 0.001 | 0.0001 | 0.001 | - |
| hough_segments | 1060919 | 819 | 1318 | 1.978 | 0.20 | 229 | 0.009 | 0.0009 | 0.009 | - |
| perspective_rectify | 312937 | 745 | 389 | 0.583 | 0.06 | 210 | 0.003 | 0.0003 | 0.003 | - |
| feature_homography | 213165 | 498 | 265 | 0.397 | 0.04 | 207 | 0.002 | 0.0002 | 0.002 | - |
| histogram_stats | 285915 | 681 | 355 | 0.533 | 0.05 | 204 | 0.003 | 0.0003 | 0.003 | - |
| components_stats | 346957 | 826 | 431 | 0.647 | 0.06 | 187 | 0.003 | 0.0003 | 0.003 | - |
| text_layout | 2493309 | 1185 | 3098 | 4.647 | 0.46 | 296 | 0.016 | 0.0016 | 0.016 | - |
| drills | 767369 | 457 | 954 | 1.430 | 0.14 | 682 | 0.002 | 0.0002 | 0.002 | - |
| **all** | 5599891 | 779 | 6959 | 10.438 | 1.04 | 2225 | 0.005 | 0.0005 | 0.005 | - |

## Repair (one round)

No repair stage yet.
