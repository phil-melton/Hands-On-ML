# Bake-off report: bakeoff-20261020

238 tasks, 4 rungs (r1-7b, r2-4b, r3-9b, r4-7b-notes); 2492 replies, 2492 graded; 0 grader errors. Ollama 0.40.4. Grader CPU time 1778 s (48 grades reused from an identical (task, code) pair).

Decoding: greedy `{'num_ctx': 8192, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}`; sampling `{'num_ctx': 8192, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.8, 'top_p': 0.95}`; not set: the model's Modelfile / Ollama defaults apply.

## 1. Pass rates (percent, ± standard error)

pass@1 is one greedy reply at T=0 (binomial SE). pass@5 is the unbiased estimator from the sampled replies at T=0.8 (bootstrap SE over tasks). `n` is tasks in the group; `prompts` is how many distinct prompts they use.

**pass@1, T=0**

| group | n | prompts | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|---|---|
| all | 238 | 77 | 15.5 ± 2.3 | 8.0 ± 1.8 | 32.8 ± 3.0 | 21.4 ± 2.7 |
| families | 198 | 37 | 11.1 ± 2.2 | 8.1 ± 1.9 | 29.3 ± 3.2 | 16.7 ± 2.6 |
| gotchas | 40 | 40 | 37.5 ± 7.7 | 7.5 ± 4.2 | 50.0 ± 7.9 | 45.0 ± 7.9 |
| 5.0 | 149 | 42 | 6.7 ± 2.0 | 3.4 ± 1.5 | 18.8 ± 3.2 | 13.4 ± 2.8 |
| timeless | 89 | 35 | 30.3 ± 4.9 | 15.7 ± 3.9 | 56.2 ± 5.3 | 34.8 ± 5.1 |
| families/5.0 | 132 | 25 | 3.8 ± 1.7 | 3.8 ± 1.7 | 15.2 ± 3.1 | 11.4 ± 2.8 |
| families/timeless | 66 | 12 | 25.8 ± 5.4 | 16.7 ± 4.6 | 57.6 ± 6.1 | 27.3 ± 5.5 |
| gotchas/5.0 | 17 | 17 | 29.4 ± 11.1 | 0.0 ± 0.0 | 47.1 ± 12.1 | 29.4 ± 11.1 |
| gotchas/timeless | 17 | 17 | 41.2 ± 11.9 | 11.8 ± 7.8 | 58.8 ± 11.9 | 52.9 ± 12.1 |
| controls | 6 | 6 | 50.0 ± 20.4 | 16.7 ± 15.2 | 33.3 ± 19.2 | 66.7 ± 19.2 |
| trained families | 154 | 29 | 3.2 ± 1.4 | 3.2 ± 1.4 | 27.3 ± 3.6 | 10.4 ± 2.5 |
| held-out families | 44 | 8 | 38.6 ± 7.3 | 25.0 ± 6.5 | 36.4 ± 7.3 | 38.6 ± 7.3 |

**pass@5, T=0.8**

| group | n | prompts | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|---|---|
| all | 238 | 77 | 40.3 ± 5.6 | 35.1 ± 5.5 | 54.5 ± 5.8 | 45.5 ± 5.7 |
| families | 198 | 37 | 24.3 ± 7.0 | 29.7 ± 7.4 | 48.6 ± 8.0 | 24.3 ± 7.0 |
| gotchas | 40 | 40 | 55.0 ± 7.7 | 40.0 ± 7.7 | 60.0 ± 7.9 | 65.0 ± 7.4 |
| 5.0 | 149 | 42 | 28.6 ± 6.9 | 21.4 ± 6.2 | 33.3 ± 7.3 | 38.1 ± 7.4 |
| timeless | 89 | 35 | 54.3 ± 8.5 | 51.4 ± 8.5 | 80.0 ± 6.8 | 54.3 ± 8.5 |
| families/5.0 | 132 | 25 | 20.0 ± 8.0 | 12.0 ± 6.4 | 24.0 ± 8.6 | 20.0 ± 7.9 |
| families/timeless | 66 | 12 | 33.3 ± 13.5 | 66.7 ± 13.7 | 100.0 ± 0.0 | 33.3 ± 13.5 |
| gotchas/5.0 | 17 | 17 | 41.2 ± 11.9 | 35.3 ± 11.5 | 47.1 ± 12.1 | 64.7 ± 11.4 |
| gotchas/timeless | 17 | 17 | 64.7 ± 11.8 | 47.1 ± 11.8 | 76.5 ± 10.3 | 64.7 ± 11.4 |
| controls | 6 | 6 | 66.7 ± 19.4 | 33.3 ± 19.3 | 50.0 ± 20.1 | 66.7 ± 19.5 |
| trained families | 154 | 29 | 17.2 ± 7.1 | 27.6 ± 8.2 | 48.3 ± 9.0 | 17.2 ± 6.9 |
| held-out families | 44 | 8 | 50.0 ± 17.5 | 37.5 ± 17.0 | 50.0 ± 17.5 | 50.0 ± 17.5 |

A `-` in the pass@5 table means no task in the group has 5 graded sampled replies yet. Groups: `all` every task; `families` the 9 cvbench families; `gotchas` the 40 gotcha checks (incl. controls); `5.0` subset 5.0 (families and gotchas); `timeless` subset timeless (families and gotchas); `families/5.0` families, subset 5.0; `families/timeless` families, subset timeless; `gotchas/5.0` gotchas (role gotcha), subset 5.0; `gotchas/timeless` gotchas (role gotcha), subset timeless; `controls` gotcha controls (behaviour that did not change); `trained families` families that may be training data; `held-out families` families never trained on.

**Clustering check (group `all`)**: the SE if tasks sharing a prompt are not treated as independent.

| rung | pass@1 SE binomial | pass@1 SE by prompt | pass@5 SE tasks | pass@5 SE by prompt |
|---|---|---|---|---|
| r1-7b | 2.3 | 4.5 | 5.6 | 5.5 |
| r2-4b | 1.8 | 3.8 | 5.5 | 5.4 |
| r3-9b | 3.0 | 6.5 | 5.8 | 5.5 |
| r4-7b-notes | 2.7 | 5.2 | 5.7 | 5.7 |

## 2. Per family (pass@1 / pass@5, percent)

| family | subset | n | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|---|---|
| contour_area | timeless | 22 | 0.0 / 0.0 | 0.0 / 50.0 | 50.0 / 100.0 | 0.0 / 0.0 |
| hough_segments | 5.0 | 22 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| chessboard_corners † | 5.0 | 22 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| perspective_rectify | timeless | 22 | 0.0 / 0.0 | 0.0 / 75.0 | 50.0 / 100.0 | 4.5 / 0.0 |
| feature_homography | 5.0 | 22 | 18.2 / 60.0 | 0.0 / 0.0 | 18.2 / 0.0 | 40.9 / 60.0 |
| histogram_stats | 5.0 | 22 | 4.5 / 25.0 | 22.7 / 50.0 | 50.0 / 100.0 | 27.3 / 50.0 |
| color_segmentation † | timeless | 22 | 77.3 / 100.0 | 50.0 / 75.0 | 72.7 / 100.0 | 77.3 / 100.0 |
| components_stats | 5.0 | 22 | 0.0 / 0.0 | 0.0 / 25.0 | 22.7 / 50.0 | 0.0 / 0.0 |
| text_layout | 5.0 | 22 | 0.0 / 25.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| gotcha:shapes | mixed | 14 | 35.7 / 35.7 | 7.1 / 28.6 | 28.6 / 35.7 | 35.7 / 64.3 |
| gotcha:dtypes | mixed | 11 | 18.2 / 63.6 | 18.2 / 72.7 | 54.5 / 72.7 | 54.5 / 72.7 |
| gotcha:drawing | timeless | 6 | 83.3 / 83.3 | 0.0 / 33.3 | 66.7 / 83.3 | 66.7 / 66.7 |
| gotcha:calling | timeless | 5 | 40.0 / 60.0 | 0.0 / 20.0 | 80.0 / 80.0 | 40.0 / 60.0 |
| gotcha:drift | 5.0 | 4 | 25.0 / 50.0 | 0.0 / 25.0 | 50.0 / 50.0 | 25.0 / 50.0 |

† held-out family (never training data).

## 3. Error taxonomy of failing replies

Classes (first matching signal wins; the rest are kept per reply as `also`): `format`, `api`, `lint`, `timeout`, `convention`, `algorithm`, `other`. `api` = hallucinated or removed cv2 name or call signature (prior -> fine-tune); `convention` = shape, dtype, ordering, signature or contract mismatch (prior -> fine-tune); `algorithm` = met the contract, wrong numbers (capacity -> bigger model); `lint`, `format`, `timeout`, `other` as in `bakeoff/taxonomy.py`.

**T=0 failures, all tasks**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 201 | 0 | 2 | 1 | 0 | 98 | 78 | 22 |
| r2-4b | 219 | 35 | 57 | 1 | 0 | 47 | 17 | 62 |
| r3-9b | 160 | 15 | 14 | 1 | 0 | 95 | 33 | 2 |
| r4-7b-notes | 187 | 0 | 18 | 1 | 0 | 51 | 90 | 27 |

**T=0 failures, subset 5.0**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 139 | 0 | 1 | 0 | 0 | 84 | 51 | 3 |
| r2-4b | 144 | 28 | 35 | 0 | 0 | 38 | 10 | 33 |
| r3-9b | 121 | 15 | 12 | 0 | 0 | 81 | 11 | 2 |
| r4-7b-notes | 129 | 0 | 17 | 0 | 0 | 30 | 59 | 23 |

**T=0 failures, subset timeless**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 62 | 0 | 1 | 1 | 0 | 14 | 27 | 19 |
| r2-4b | 75 | 7 | 22 | 1 | 0 | 9 | 7 | 29 |
| r3-9b | 39 | 0 | 2 | 1 | 0 | 14 | 22 | 0 |
| r4-7b-notes | 58 | 0 | 1 | 1 | 0 | 21 | 31 | 4 |

**T=0 failures, families only (the clean `convention` vs `algorithm` split)**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 176 | 0 | 0 | 0 | 0 | 86 | 77 | 13 |
| r2-4b | 182 | 33 | 54 | 0 | 0 | 37 | 15 | 43 |
| r3-9b | 140 | 15 | 11 | 0 | 0 | 80 | 33 | 1 |
| r4-7b-notes | 165 | 0 | 16 | 0 | 0 | 40 | 87 | 22 |

**T=0 failures, gotchas only (`algorithm` here may be a dtype or behaviour convention)**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 25 | 0 | 2 | 1 | 0 | 12 | 1 | 9 |
| r2-4b | 37 | 2 | 3 | 1 | 0 | 10 | 2 | 19 |
| r3-9b | 20 | 0 | 3 | 1 | 0 | 15 | 0 | 1 |
| r4-7b-notes | 22 | 0 | 2 | 1 | 0 | 11 | 3 | 5 |

**T=0.8 sampled failures**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 302 | 1 | 17 | 5 | 0 | 125 | 59 | 95 |
| r2-4b | 343 | 29 | 59 | 9 | 1 | 101 | 30 | 114 |
| r3-9b | 257 | 47 | 28 | 11 | 0 | 129 | 33 | 9 |
| r4-7b-notes | 302 | 1 | 23 | 5 | 0 | 112 | 72 | 89 |

Rules fired (all failing replies, `class:rule`):

| class:rule | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|
| algorithm:verifier-value | 128 | 37 | 55 | 147 |
| other:exception:NameError | 90 | 148 | 1 | 87 |
| convention:cv2-assertion | 77 | 39 | 66 | 71 |
| api:cv2-attribute | 17 | 82 | 32 | 14 |
| convention:verifier-contract | 66 | 12 | 6 | 35 |
| format:lint:syntax | 1 | 52 | 59 | 1 |
| convention:TypeError | 27 | 19 | 36 | 21 |
| convention:bad-output | 0 | 30 | 53 | 0 |
| convention:ValueError | 4 | 22 | 36 | 8 |
| convention:AttributeError | 24 | 9 | 12 | 21 |
| other:exception:ValueError | 23 | 8 | 8 | 27 |
| convention:IndexError | 16 | 16 | 8 | 6 |
| algorithm:gotcha-value | 9 | 10 | 11 | 15 |
| lint:lint:broad-except | 6 | 9 | 12 | 6 |
| api:import | 1 | 5 | 0 | 25 |
| api:cv2-call-signature | 1 | 16 | 8 | 2 |
| convention:stage:contract | 9 | 1 | 6 | 1 |
| other:exception:TypeError | 0 | 16 | 1 | 0 |
| format:syntax-at-runtime | 0 | 12 | 1 | 0 |
| api:call-signature | 0 | 11 | 1 | 0 |
| other:killed | 1 | 2 | 1 | 0 |
| other:exception:UnboundLocalError | 0 | 2 | 0 | 1 |
| api:removed-api:removed_in_5 | 0 | 2 | 1 | 0 |
| other:exception:ZeroDivisionError | 2 | 0 | 0 | 0 |
| format:lint:contract | 0 | 0 | 2 | 0 |
| other:exception:Exception | 1 | 0 | 0 | 0 |
| lint:lint:io | 0 | 1 | 0 | 0 |
| timeout:timeout | 0 | 1 | 0 | 0 |
| convention:stage:malformed | 0 | 0 | 1 | 0 |
| other:exception:OverflowError | 0 | 0 | 0 | 1 |

## 4. Paired comparison on pass@1 (exact McNemar)

A - B is the paired difference in percentage points with its SE; `A only` / `B only` are the discordant tasks; `p` is the exact two-sided McNemar p-value. `p by prompt` is an exact sign test over prompts (one vote per distinct prompt), the conservative companion when many tasks share a prompt.

**group `all`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 238 | 25 | 7 | +7.6 | 2.3 | 4.1 | 0.0021 | 0.00258 |
| r1-7b | r3-9b | 238 | 14 | 55 | -17.2 | 3.3 | 7.0 | 6.92e-07 | 0.0227 |
| r1-7b | r4-7b-notes | 238 | 4 | 18 | -5.9 | 1.9 | 3.2 | 0.00434 | 0.18 |
| r2-4b | r3-9b | 238 | 6 | 65 | -24.8 | 3.2 | 6.6 | 1.33e-13 | 1.62e-06 |
| r2-4b | r4-7b-notes | 238 | 1 | 33 | -13.4 | 2.3 | 4.0 | 4.07e-09 | 1.1e-05 |
| r3-9b | r4-7b-notes | 238 | 52 | 25 | +11.3 | 3.6 | 7.7 | 0.0028 | 0.307 |

**group `5.0`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 149 | 10 | 5 | +3.4 | 2.6 | 4.7 | 0.302 | 0.0703 |
| r1-7b | r3-9b | 149 | 5 | 23 | -12.1 | 3.4 | 6.9 | 0.000912 | 0.109 |
| r1-7b | r4-7b-notes | 149 | 2 | 12 | -6.7 | 2.5 | 4.8 | 0.0129 | 0.688 |
| r2-4b | r3-9b | 149 | 5 | 28 | -15.4 | 3.7 | 7.6 | 6.62e-05 | 0.00342 |
| r2-4b | r4-7b-notes | 149 | 0 | 15 | -10.1 | 2.5 | 4.5 | 6.1e-05 | 0.00781 |
| r3-9b | r4-7b-notes | 149 | 22 | 14 | +5.4 | 4.0 | 8.6 | 0.243 | 0.344 |

**group `timeless`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 89 | 15 | 2 | +14.6 | 4.4 | 7.5 | 0.00235 | 0.0386 |
| r1-7b | r3-9b | 89 | 9 | 32 | -25.8 | 6.7 | 14.4 | 0.000431 | 0.18 |
| r1-7b | r4-7b-notes | 89 | 2 | 6 | -4.5 | 3.2 | 3.2 | 0.289 | 0.289 |
| r2-4b | r3-9b | 89 | 1 | 37 | -40.4 | 5.5 | 11.1 | 2.84e-10 | 0.000519 |
| r2-4b | r4-7b-notes | 89 | 1 | 18 | -19.1 | 4.5 | 7.5 | 7.63e-05 | 0.00183 |
| r3-9b | r4-7b-notes | 89 | 30 | 11 | +21.3 | 6.9 | 14.6 | 0.00432 | 0.791 |

**group `families`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 198 | 11 | 5 | +3.0 | 2.0 | 4.5 | 0.21 | 0.625 |
| r1-7b | r3-9b | 198 | 10 | 46 | -18.2 | 3.6 | 8.2 | 1.25e-06 | 0.0654 |
| r1-7b | r4-7b-notes | 198 | 0 | 11 | -5.6 | 1.6 | 3.6 | 0.000977 | 0.25 |
| r2-4b | r3-9b | 198 | 5 | 47 | -21.2 | 3.3 | 7.7 | 1.28e-09 | 0.0215 |
| r2-4b | r4-7b-notes | 198 | 0 | 17 | -8.6 | 2.0 | 4.3 | 1.53e-05 | 0.0625 |
| r3-9b | r4-7b-notes | 198 | 46 | 21 | +12.6 | 4.0 | 9.2 | 0.00307 | 0.424 |

**group `gotchas`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 40 | 14 | 2 | +30.0 | 8.9 | 8.9 | 0.00418 | 0.00418 |
| r1-7b | r3-9b | 40 | 4 | 9 | -12.5 | 8.9 | 8.9 | 0.267 | 0.267 |
| r1-7b | r4-7b-notes | 40 | 4 | 7 | -7.5 | 8.3 | 8.3 | 0.549 | 0.549 |
| r2-4b | r3-9b | 40 | 1 | 18 | -42.5 | 8.7 | 8.7 | 7.63e-05 | 7.63e-05 |
| r2-4b | r4-7b-notes | 40 | 1 | 16 | -37.5 | 8.5 | 8.5 | 0.000275 | 0.000275 |
| r3-9b | r4-7b-notes | 40 | 6 | 4 | +5.0 | 8.0 | 8.0 | 0.754 | 0.754 |

**group `held-out families`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 44 | 6 | 0 | +13.6 | 5.2 | 13.5 | 0.0312 | 1 |
| r1-7b | r3-9b | 44 | 6 | 5 | +2.3 | 7.6 | 18.9 | 1 | 1 |
| r1-7b | r4-7b-notes | 44 | 0 | 0 | +0.0 | 0.0 | 0.0 | 1 | 1 |
| r2-4b | r3-9b | 44 | 0 | 5 | -11.4 | 4.8 | 11.5 | 0.0625 | 1 |
| r2-4b | r4-7b-notes | 44 | 0 | 6 | -13.6 | 5.2 | 13.5 | 0.0312 | 1 |
| r3-9b | r4-7b-notes | 44 | 5 | 6 | -2.3 | 7.6 | 18.9 | 1 | 1 |

**A - B matrix, group `all`** (row A, column B; points, `*` = McNemar p < 0.05)

| A \ B | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|
| r1-7b |  | +7.6* | -17.2* | -5.9* |
| r2-4b | -7.6* |  | -24.8* | -13.4* |
| r3-9b | +17.2* | +24.8* |  | +11.3* |
| r4-7b-notes | +5.9* | +13.4* | -11.3* |  |

**A - B matrix, group `5.0`** (row A, column B; points, `*` = McNemar p < 0.05)

| A \ B | r1-7b | r2-4b | r3-9b | r4-7b-notes |
|---|---|---|---|---|
| r1-7b |  | +3.4 | -12.1* | -6.7* |
| r2-4b | -3.4 |  | -15.4* | -10.1* |
| r3-9b | +12.1* | +15.4* |  | +5.4 |
| r4-7b-notes | +6.7* | +10.1* | -5.4 |  |

## 5. Throughput and cost

Decode tok/s is `eval_count / eval_duration` per reply, warm-up excluded (it is in `warmups.jsonl`). A reply is truncated when it stopped on `num_predict`. Prompt tokens can undercount when Ollama reuses a cached prompt prefix; the warm-up figure (first request after load) is the full prompt.

| rung | model | replies | tok/s median | p10 | p90 | tokens/answer mean | prompt tokens median | truncated | generation wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | base-qwen2.5-coder-7b | 623 | 63.2 | 62.7 | 63.9 | 238 | 242 | 1 | 30.0 |
| r2-4b | base-qwen3.5-4b | 623 | 79.3 | 78.6 | 80.0 | 965 | 250 | 120 | 96.1 |
| r3-9b | base-qwen3.5-9b | 623 | 52.9 | 52.5 | 53.2 | 1041 | 250 | 186 | 154.5 |
| r4-7b-notes | base-qwen2.5-coder-7b | 623 | 61.9 | 61.5 | 62.8 | 227 | 1697 | 0 | 29.6 |

Total generation wall time: 5.17 h (plus 6 s of discarded warm-ups).

| rung | extraction path | think block | thinking chars | warm-up s | load s | warm-up prompt tokens | on GPU (bytes) | greedy repeats identical |
|---|---|---|---|---|---|---|---|---|
| r1-7b | {'fence': 620, 'bare': 3} | {'none': 623} | 0 | 1.258 | 23.759 | 229 | 5148633988 of 5148633988 | 0 of 0 |
| r2-4b | {'fence': 510, 'fence-unclosed': 113} | {'none': 623} | 0 | 1.116 | 20.856 | 237 | 3190974054 of 3190974054 | 0 of 0 |
| r3-9b | {'fence': 451, 'fence-unclosed': 158, 'bare': 14} | {'none': 623} | 0 | 1.542 | 38.781 | 237 | 5602059223 of 5602059223 | 0 of 0 |
| r4-7b-notes | {'fence': 619, 'bare': 4} | {'none': 623} | 0 | 1.844 | 2.674 | 1684 | 5148633988 of 5148633988 | 0 of 0 |

Measured size of the notes block in `r4-7b-notes`: 1455 prompt tokens (warm-up prompt minus the plain 7B's); the budget is 1500.
