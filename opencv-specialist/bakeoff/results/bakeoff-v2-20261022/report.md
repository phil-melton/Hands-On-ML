# Bake-off report: bakeoff-v2-20261022

238 tasks, 5 rungs (r1-7b, r2-4b, r3-9b, r4-7b-notes, r5-9b-notes); 1190 replies, 1190 graded; 0 grader errors. Ollama 0.40.4. Grader CPU time 967 s (7 grades reused from an identical (task, code) pair).

Decoding: greedy `{'num_ctx': 8192, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}`; sampling `{'num_ctx': 8192, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.8, 'top_p': 0.95}`; not set: the model's Modelfile / Ollama defaults apply.

## 1. Pass rates (percent, ± standard error)

pass@1 is one greedy reply at T=0 (binomial SE). pass@5 is the unbiased estimator from the sampled replies at T=0.8 (bootstrap SE over tasks). `n` is tasks in the group; `prompts` is how many distinct prompts they use.

**pass@1, T=0**

| group | n | prompts | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|---|---|
| all | 238 | 112 | 16.8 ± 2.4 | 18.1 ± 2.5 | 32.8 ± 3.0 | 21.4 ± 2.7 | 38.7 ± 3.2 |
| families | 198 | 72 | 10.6 ± 2.2 | 14.6 ± 2.5 | 30.3 ± 3.3 | 15.7 ± 2.6 | 33.8 ± 3.4 |
| gotchas | 40 | 40 | 47.5 ± 7.9 | 35.0 ± 7.5 | 45.0 ± 7.9 | 50.0 ± 7.9 | 62.5 ± 7.7 |
| 5.0 | 149 | 65 | 10.1 ± 2.5 | 10.1 ± 2.5 | 13.4 ± 2.8 | 13.4 ± 2.8 | 20.1 ± 3.3 |
| timeless | 89 | 47 | 28.1 ± 4.8 | 31.5 ± 4.9 | 65.2 ± 5.1 | 34.8 ± 5.1 | 69.7 ± 4.9 |
| families/5.0 | 132 | 48 | 6.1 ± 2.1 | 7.6 ± 2.3 | 9.8 ± 2.6 | 10.6 ± 2.7 | 16.7 ± 3.2 |
| families/timeless | 66 | 24 | 19.7 ± 4.9 | 28.8 ± 5.6 | 71.2 ± 5.6 | 25.8 ± 5.4 | 68.2 ± 5.7 |
| gotchas/5.0 | 17 | 17 | 41.2 ± 11.9 | 29.4 ± 11.1 | 41.2 ± 11.9 | 35.3 ± 11.6 | 47.1 ± 12.1 |
| gotchas/timeless | 17 | 17 | 52.9 ± 12.1 | 41.2 ± 11.9 | 47.1 ± 12.1 | 64.7 ± 11.6 | 70.6 ± 11.1 |
| controls | 6 | 6 | 50.0 ± 20.4 | 33.3 ± 19.2 | 50.0 ± 20.4 | 50.0 ± 20.4 | 83.3 ± 15.2 |
| trained families | 154 | 56 | 5.2 ± 1.8 | 6.5 ± 2.0 | 26.6 ± 3.6 | 9.1 ± 2.3 | 31.2 ± 3.7 |
| held-out families | 44 | 16 | 29.5 ± 6.9 | 43.2 ± 7.5 | 43.2 ± 7.5 | 38.6 ± 7.3 | 43.2 ± 7.5 |

**pass@5, T=0.8**

| group | n | prompts | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|---|---|
| all | 238 | 112 | - | - | - | - | - |
| families | 198 | 72 | - | - | - | - | - |
| gotchas | 40 | 40 | - | - | - | - | - |
| 5.0 | 149 | 65 | - | - | - | - | - |
| timeless | 89 | 47 | - | - | - | - | - |
| families/5.0 | 132 | 48 | - | - | - | - | - |
| families/timeless | 66 | 24 | - | - | - | - | - |
| gotchas/5.0 | 17 | 17 | - | - | - | - | - |
| gotchas/timeless | 17 | 17 | - | - | - | - | - |
| controls | 6 | 6 | - | - | - | - | - |
| trained families | 154 | 56 | - | - | - | - | - |
| held-out families | 44 | 16 | - | - | - | - | - |

A `-` in the pass@5 table means no task in the group has 5 graded sampled replies yet. Groups: `all` every task; `families` the 9 cvbench families; `gotchas` the 40 gotcha checks (incl. controls); `5.0` subset 5.0 (families and gotchas); `timeless` subset timeless (families and gotchas); `families/5.0` families, subset 5.0; `families/timeless` families, subset timeless; `gotchas/5.0` gotchas (role gotcha), subset 5.0; `gotchas/timeless` gotchas (role gotcha), subset timeless; `controls` gotcha controls (behaviour that did not change); `trained families` families that may be training data; `held-out families` families never trained on.

**Clustering check (group `all`)**: the SE if tasks sharing a prompt are not treated as independent.

| rung | pass@1 SE binomial | pass@1 SE by prompt | pass@5 SE tasks | pass@5 SE by prompt |
|---|---|---|---|---|
| r1-7b | 2.4 | 3.5 | - | - |
| r2-4b | 2.5 | 3.7 | - | - |
| r3-9b | 3.0 | 4.7 | - | - |
| r4-7b-notes | 2.7 | 4.0 | - | - |
| r5-9b-notes | 3.2 | 4.9 | - | - |

## 2. Per family (pass@1 / pass@5, percent)

| family | subset | n | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|---|---|
| contour_area | timeless | 22 | 0.0 / - | 0.0 / - | 63.6 / - | 0.0 / - | 81.8 / - |
| hough_segments | 5.0 | 22 | 0.0 / - | 0.0 / - | 0.0 / - | 0.0 / - | 4.5 / - |
| chessboard_corners † | 5.0 | 22 | 0.0 / - | 0.0 / - | 0.0 / - | 0.0 / - | 0.0 / - |
| perspective_rectify | timeless | 22 | 0.0 / - | 0.0 / - | 63.6 / - | 0.0 / - | 36.4 / - |
| feature_homography | 5.0 | 22 | 36.4 / - | 22.7 / - | 36.4 / - | 63.6 / - | 36.4 / - |
| histogram_stats | 5.0 | 22 | 0.0 / - | 22.7 / - | 13.6 / - | 0.0 / - | 13.6 / - |
| color_segmentation † | timeless | 22 | 59.1 / - | 86.4 / - | 86.4 / - | 77.3 / - | 86.4 / - |
| components_stats | 5.0 | 22 | 0.0 / - | 0.0 / - | 9.1 / - | 0.0 / - | 45.5 / - |
| text_layout | 5.0 | 22 | 0.0 / - | 0.0 / - | 0.0 / - | 0.0 / - | 0.0 / - |
| gotcha:shapes | mixed | 14 | 28.6 / - | 21.4 / - | 42.9 / - | 35.7 / - | 50.0 / - |
| gotcha:dtypes | mixed | 11 | 45.5 / - | 36.4 / - | 45.5 / - | 45.5 / - | 72.7 / - |
| gotcha:drawing | timeless | 6 | 66.7 / - | 50.0 / - | 33.3 / - | 66.7 / - | 83.3 / - |
| gotcha:calling | timeless | 5 | 80.0 / - | 60.0 / - | 60.0 / - | 80.0 / - | 60.0 / - |
| gotcha:drift | 5.0 | 4 | 50.0 / - | 25.0 / - | 50.0 / - | 50.0 / - | 50.0 / - |

† held-out family (never training data).

## 3. Error taxonomy of failing replies

Classes (first matching signal wins; the rest are kept per reply as `also`): `format`, `api`, `lint`, `timeout`, `convention`, `algorithm`, `other`. `api` = hallucinated or removed cv2 name or call signature (prior -> fine-tune); `convention` = shape, dtype, ordering, signature or contract mismatch (prior -> fine-tune); `algorithm` = met the contract, wrong numbers (capacity -> bigger model); `lint`, `format`, `timeout`, `other` as in `bakeoff/taxonomy.py`.

**T=0 failures, all tasks**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 198 | 0 | 2 | 1 | 0 | 101 | 79 | 15 |
| r2-4b | 195 | 9 | 74 | 1 | 0 | 72 | 24 | 15 |
| r3-9b | 160 | 17 | 6 | 1 | 0 | 87 | 44 | 5 |
| r4-7b-notes | 187 | 0 | 22 | 1 | 0 | 82 | 77 | 5 |
| r5-9b-notes | 146 | 15 | 10 | 3 | 0 | 69 | 38 | 11 |

**T=0 failures, subset 5.0**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 134 | 0 | 1 | 0 | 0 | 75 | 47 | 11 |
| r2-4b | 134 | 8 | 36 | 0 | 0 | 61 | 17 | 12 |
| r3-9b | 129 | 17 | 5 | 0 | 0 | 74 | 31 | 2 |
| r4-7b-notes | 129 | 0 | 21 | 0 | 0 | 51 | 52 | 5 |
| r5-9b-notes | 119 | 9 | 7 | 2 | 0 | 57 | 33 | 11 |

**T=0 failures, subset timeless**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 64 | 0 | 1 | 1 | 0 | 26 | 32 | 4 |
| r2-4b | 61 | 1 | 38 | 1 | 0 | 11 | 7 | 3 |
| r3-9b | 31 | 0 | 1 | 1 | 0 | 13 | 13 | 3 |
| r4-7b-notes | 58 | 0 | 1 | 1 | 0 | 31 | 25 | 0 |
| r5-9b-notes | 27 | 6 | 3 | 1 | 0 | 12 | 5 | 0 |

**T=0 failures, families only (the clean `convention` vs `algorithm` split)**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 177 | 0 | 0 | 0 | 0 | 87 | 75 | 15 |
| r2-4b | 169 | 8 | 68 | 0 | 0 | 56 | 23 | 14 |
| r3-9b | 138 | 17 | 4 | 0 | 0 | 69 | 43 | 5 |
| r4-7b-notes | 167 | 0 | 19 | 0 | 0 | 68 | 75 | 5 |
| r5-9b-notes | 131 | 14 | 8 | 2 | 0 | 58 | 38 | 11 |

**T=0 failures, gotchas only (`algorithm` here may be a dtype or behaviour convention)**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 21 | 0 | 2 | 1 | 0 | 14 | 4 | 0 |
| r2-4b | 26 | 1 | 6 | 1 | 0 | 16 | 1 | 1 |
| r3-9b | 22 | 0 | 2 | 1 | 0 | 18 | 1 | 0 |
| r4-7b-notes | 20 | 0 | 3 | 1 | 0 | 14 | 2 | 0 |
| r5-9b-notes | 15 | 1 | 2 | 1 | 0 | 11 | 0 | 0 |

**T=0.8 sampled failures**

| rung | failures | format | api | lint | timeout | convention | algorithm | other |
|---|---|---|---|---|---|---|---|---|
| r1-7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r2-4b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r3-9b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r4-7b-notes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r5-9b-notes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Rules fired (all failing replies, `class:rule`):

| class:rule | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|
| algorithm:verifier-value | 75 | 23 | 43 | 75 | 38 |
| convention:cv2-assertion | 40 | 22 | 16 | 34 | 11 |
| convention:verifier-contract | 44 | 3 | 1 | 25 | 5 |
| api:cv2-attribute | 2 | 66 | 2 | 2 | 4 |
| convention:bad-output | 0 | 12 | 25 | 0 | 18 |
| convention:TypeError | 4 | 18 | 21 | 4 | 5 |
| convention:ValueError | 3 | 8 | 10 | 1 | 25 |
| other:exception:ValueError | 15 | 0 | 5 | 5 | 11 |
| convention:AttributeError | 6 | 0 | 9 | 17 | 4 |
| format:lint:syntax | 0 | 3 | 17 | 0 | 15 |
| api:import | 0 | 1 | 0 | 20 | 0 |
| convention:IndexError | 0 | 8 | 4 | 0 | 1 |
| api:call-signature | 0 | 3 | 4 | 0 | 5 |
| algorithm:gotcha-value | 4 | 1 | 1 | 2 | 0 |
| other:exception:TypeError | 0 | 8 | 0 | 0 | 0 |
| convention:stage:contract | 4 | 1 | 1 | 1 | 0 |
| lint:lint:broad-except | 1 | 1 | 1 | 1 | 3 |
| other:exception:NameError | 0 | 7 | 0 | 0 | 0 |
| format:syntax-at-runtime | 0 | 6 | 0 | 0 | 0 |
| api:cv2-call-signature | 0 | 4 | 0 | 0 | 1 |

## 4. Paired comparison on pass@1 (exact McNemar)

A - B is the paired difference in percentage points with its SE; `A only` / `B only` are the discordant tasks; `p` is the exact two-sided McNemar p-value. `p by prompt` is an exact sign test over prompts (one vote per distinct prompt), the conservative companion when many tasks share a prompt.

**group `all`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 238 | 11 | 14 | -1.3 | 2.1 | 3.0 | 0.69 | 0.804 |
| r1-7b | r3-9b | 238 | 16 | 54 | -16.0 | 3.4 | 5.3 | 5.85e-06 | 0.0201 |
| r1-7b | r4-7b-notes | 238 | 5 | 16 | -4.6 | 1.9 | 2.8 | 0.0266 | 0.388 |
| r1-7b | r5-9b-notes | 238 | 12 | 64 | -21.8 | 3.4 | 5.4 | 1e-09 | 0.000117 |
| r2-4b | r3-9b | 238 | 17 | 52 | -14.7 | 3.4 | 5.2 | 2.93e-05 | 0.00904 |
| r2-4b | r4-7b-notes | 238 | 14 | 22 | -3.4 | 2.5 | 3.7 | 0.243 | 0.286 |
| r2-4b | r5-9b-notes | 238 | 13 | 62 | -20.6 | 3.4 | 5.3 | 8.4e-09 | 4.13e-05 |
| r3-9b | r4-7b-notes | 238 | 44 | 17 | +11.3 | 3.2 | 4.9 | 0.00073 | 0.0987 |
| r3-9b | r5-9b-notes | 238 | 19 | 33 | -5.9 | 3.0 | 4.5 | 0.0704 | 0.0614 |
| r4-7b-notes | r5-9b-notes | 238 | 13 | 54 | -17.2 | 3.3 | 5.1 | 4.47e-07 | 0.00132 |

**group `5.0`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 149 | 7 | 7 | +0.0 | 2.5 | 3.6 | 1 | 1 |
| r1-7b | r3-9b | 149 | 10 | 15 | -3.4 | 3.4 | 5.3 | 0.424 | 0.774 |
| r1-7b | r4-7b-notes | 149 | 2 | 7 | -3.4 | 2.0 | 3.0 | 0.18 | 1 |
| r1-7b | r5-9b-notes | 149 | 7 | 22 | -10.1 | 3.5 | 5.7 | 0.00813 | 0.0923 |
| r2-4b | r3-9b | 149 | 12 | 17 | -3.4 | 3.6 | 5.5 | 0.458 | 0.607 |
| r2-4b | r4-7b-notes | 149 | 8 | 13 | -3.4 | 3.1 | 4.6 | 0.383 | 0.774 |
| r2-4b | r5-9b-notes | 149 | 9 | 24 | -10.1 | 3.8 | 6.0 | 0.0135 | 0.0768 |
| r3-9b | r4-7b-notes | 149 | 9 | 9 | +0.0 | 2.9 | 4.4 | 1 | 1 |
| r3-9b | r5-9b-notes | 149 | 4 | 14 | -6.7 | 2.8 | 4.0 | 0.0309 | 0.227 |
| r4-7b-notes | r5-9b-notes | 149 | 8 | 18 | -6.7 | 3.4 | 5.2 | 0.0755 | 0.18 |

**group `timeless`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 89 | 4 | 7 | -3.4 | 3.7 | 5.4 | 0.549 | 1 |
| r1-7b | r3-9b | 89 | 6 | 39 | -37.1 | 6.5 | 9.8 | 5.42e-07 | 0.0118 |
| r1-7b | r4-7b-notes | 89 | 3 | 9 | -6.7 | 3.8 | 5.7 | 0.146 | 0.453 |
| r1-7b | r5-9b-notes | 89 | 5 | 42 | -41.6 | 6.4 | 9.8 | 2.46e-08 | 0.000855 |
| r2-4b | r3-9b | 89 | 5 | 35 | -33.7 | 6.2 | 9.4 | 1.38e-06 | 0.00443 |
| r2-4b | r4-7b-notes | 89 | 6 | 9 | -3.4 | 4.4 | 6.1 | 0.607 | 0.344 |
| r2-4b | r5-9b-notes | 89 | 4 | 38 | -38.2 | 6.1 | 9.5 | 5.65e-08 | 0.000221 |
| r3-9b | r4-7b-notes | 89 | 35 | 8 | +30.3 | 6.7 | 9.8 | 4.19e-05 | 0.0784 |
| r3-9b | r5-9b-notes | 89 | 15 | 19 | -4.5 | 6.6 | 10.2 | 0.608 | 0.238 |
| r4-7b-notes | r5-9b-notes | 89 | 5 | 36 | -34.8 | 6.2 | 9.7 | 7.84e-07 | 0.00443 |

**group `families`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 198 | 3 | 11 | -4.0 | 1.9 | 3.2 | 0.0574 | 0.375 |
| r1-7b | r3-9b | 198 | 11 | 50 | -19.7 | 3.7 | 6.1 | 4.59e-07 | 0.0026 |
| r1-7b | r4-7b-notes | 198 | 2 | 12 | -5.1 | 1.9 | 3.1 | 0.0129 | 0.375 |
| r1-7b | r5-9b-notes | 198 | 9 | 55 | -23.2 | 3.7 | 6.3 | 3.54e-09 | 0.000488 |
| r2-4b | r3-9b | 198 | 13 | 44 | -15.7 | 3.7 | 6.0 | 4.71e-05 | 0.0169 |
| r2-4b | r4-7b-notes | 198 | 10 | 12 | -1.0 | 2.4 | 4.0 | 0.832 | 1 |
| r2-4b | r5-9b-notes | 198 | 11 | 49 | -19.2 | 3.7 | 6.2 | 7.56e-07 | 0.00434 |
| r3-9b | r4-7b-notes | 198 | 40 | 11 | +14.6 | 3.5 | 5.7 | 5.7e-05 | 0.0118 |
| r3-9b | r5-9b-notes | 198 | 16 | 23 | -3.5 | 3.2 | 5.1 | 0.337 | 0.454 |
| r4-7b-notes | r5-9b-notes | 198 | 9 | 45 | -18.2 | 3.5 | 5.9 | 7.29e-07 | 0.00258 |

**group `gotchas`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 40 | 8 | 3 | +12.5 | 8.2 | 8.2 | 0.227 | 0.227 |
| r1-7b | r3-9b | 40 | 5 | 4 | +2.5 | 7.6 | 7.6 | 1 | 1 |
| r1-7b | r4-7b-notes | 40 | 3 | 4 | -2.5 | 6.7 | 6.7 | 1 | 1 |
| r1-7b | r5-9b-notes | 40 | 3 | 9 | -15.0 | 8.4 | 8.4 | 0.146 | 0.146 |
| r2-4b | r3-9b | 40 | 4 | 8 | -10.0 | 8.6 | 8.6 | 0.388 | 0.388 |
| r2-4b | r4-7b-notes | 40 | 4 | 10 | -15.0 | 9.2 | 9.2 | 0.18 | 0.18 |
| r2-4b | r5-9b-notes | 40 | 2 | 13 | -27.5 | 8.8 | 8.8 | 0.00739 | 0.00739 |
| r3-9b | r4-7b-notes | 40 | 4 | 6 | -5.0 | 8.0 | 8.0 | 0.754 | 0.754 |
| r3-9b | r5-9b-notes | 40 | 3 | 10 | -17.5 | 8.7 | 8.7 | 0.0923 | 0.0923 |
| r4-7b-notes | r5-9b-notes | 40 | 4 | 9 | -12.5 | 8.9 | 8.9 | 0.267 | 0.267 |

**group `held-out families`**

| A | B | n | A only | B only | A - B (pts) | SE | SE by prompt | p | p by prompt |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | r2-4b | 44 | 0 | 6 | -13.6 | 5.2 | 9.2 | 0.0312 | 0.5 |
| r1-7b | r3-9b | 44 | 3 | 9 | -13.6 | 7.7 | 13.6 | 0.146 | 0.625 |
| r1-7b | r4-7b-notes | 44 | 2 | 6 | -9.1 | 6.4 | 10.6 | 0.289 | 1 |
| r1-7b | r5-9b-notes | 44 | 3 | 9 | -13.6 | 7.7 | 13.6 | 0.146 | 0.625 |
| r2-4b | r3-9b | 44 | 3 | 3 | +0.0 | 5.6 | 10.0 | 1 | 1 |
| r2-4b | r4-7b-notes | 44 | 5 | 3 | +4.5 | 6.5 | 11.0 | 0.727 | 1 |
| r2-4b | r5-9b-notes | 44 | 3 | 3 | +0.0 | 5.6 | 10.0 | 1 | 1 |
| r3-9b | r4-7b-notes | 44 | 5 | 3 | +4.5 | 6.5 | 11.0 | 0.727 | 1 |
| r3-9b | r5-9b-notes | 44 | 3 | 3 | +0.0 | 5.6 | 10.0 | 1 | 1 |
| r4-7b-notes | r5-9b-notes | 44 | 3 | 5 | -4.5 | 6.5 | 11.0 | 0.727 | 1 |

**A - B matrix, group `all`** (row A, column B; points, `*` = McNemar p < 0.05)

| A \ B | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|
| r1-7b |  | -1.3 | -16.0* | -4.6* | -21.8* |
| r2-4b | +1.3 |  | -14.7* | -3.4 | -20.6* |
| r3-9b | +16.0* | +14.7* |  | +11.3* | -5.9 |
| r4-7b-notes | +4.6* | +3.4 | -11.3* |  | -17.2* |
| r5-9b-notes | +21.8* | +20.6* | +5.9 | +17.2* |  |

**A - B matrix, group `5.0`** (row A, column B; points, `*` = McNemar p < 0.05)

| A \ B | r1-7b | r2-4b | r3-9b | r4-7b-notes | r5-9b-notes |
|---|---|---|---|---|---|
| r1-7b |  | +0.0 | -3.4 | -3.4 | -10.1* |
| r2-4b | +0.0 |  | -3.4 | -3.4 | -10.1* |
| r3-9b | +3.4 | +3.4 |  | +0.0 | -6.7* |
| r4-7b-notes | +3.4 | +3.4 | +0.0 |  | -6.7 |
| r5-9b-notes | +10.1* | +10.1* | +6.7* | +6.7 |  |

## 5. Throughput and cost

Decode tok/s is `eval_count / eval_duration` per reply, warm-up excluded (it is in `warmups.jsonl`). A reply is truncated when it stopped on `num_predict`. Prompt tokens can undercount when Ollama reuses a cached prompt prefix; the warm-up figure (first request after load) is the full prompt.

| rung | model | replies | tok/s median | p10 | p90 | tokens/answer mean | prompt tokens median | truncated | generation wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| r1-7b | base-qwen2.5-coder-7b | 238 | 63.4 | 63.1 | 64.3 | 247 | 346 | 0 | 7.6 |
| r2-4b | base-qwen3.5-4b | 238 | 79.6 | 78.6 | 80.5 | 956 | 354 | 33 | 23.0 |
| r3-9b | base-qwen3.5-9b | 238 | 53.3 | 53.0 | 53.5 | 1037 | 354 | 64 | 37.1 |
| r4-7b-notes | base-qwen2.5-coder-7b | 238 | 62.5 | 62.2 | 63.1 | 223 | 1800 | 0 | 7.0 |
| r5-9b-notes | base-qwen3.5-9b | 238 | 52.9 | 52.7 | 53.0 | 992 | 1839 | 49 | 35.7 |

Total generation wall time: 1.84 h (plus 8 s of discarded warm-ups).

| rung | extraction path | think block | thinking chars | warm-up s | load s | warm-up prompt tokens | on GPU (bytes) | greedy repeats identical |
|---|---|---|---|---|---|---|---|---|
| r1-7b | {'fence': 238} | {'none': 238} | 0 | 1.334 | 34.707 | 238 | 5148633988 of 5148633988 | 0 of 0 |
| r2-4b | {'fence': 205, 'fence-unclosed': 33} | {'none': 238} | 0 | 1.119 | 21.224 | 246 | 3190974054 of 3190974054 | 0 of 0 |
| r3-9b | {'fence': 174, 'fence-unclosed': 64} | {'none': 238} | 0 | 1.538 | 39.042 | 246 | 5602059223 of 5602059223 | 0 of 0 |
| r4-7b-notes | {'fence': 238} | {'none': 238} | 0 | 1.907 | 32.901 | 1693 | 5148633988 of 5148633988 | 0 of 0 |
| r5-9b-notes | {'fence': 189, 'fence-unclosed': 49} | {'none': 238} | 0 | 2.358 | 37.877 | 1731 | 5602059223 of 5602059223 | 0 of 0 |

Measured size of the notes block in `r4-7b-notes`: 1455 prompt tokens (warm-up prompt minus the plain 7B's); the budget is 1500.

Measured size of the notes block in `r5-9b-notes`: 1485 prompt tokens (warm-up prompt minus the plain 7B's); the budget is 1500.
