# Stage 5 convergence study: s5-conv1

Dev pass@1 at T = 0 on the 99 cvbench dev tasks (77 in trained families, 22 in held-out families, 66 subset 5.0). SE is the prompt-cluster-robust one (tasks of one prompt share their answer). Selecting the best of four checkpoints on this set inflates its score: the test set (Stage 6) is the unbiased number.

## Checkpoints

| checkpoint | step | train loss (window) | probe loss | dev loss | pass@1 all (SE) | trained | held-out | 5.0 | timeless | cut at cap | fenced | train tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step0 | 0 | - | 0.3819 | 0.3186 | 45.5 (8.5) | 44.2 | 50.0 | 33.3 | 69.7 | 6/36 | 30/36 | - |
| r025@e0.5 | 23 | 0.3064 | 0.2285 | 0.1836 | 63.6 (8.2) | 67.5 | 50.0 | 45.5 | 100.0 | 7/36 | 29/36 | 1327 |
| r025@e1 | 45 | 0.2390 | 0.1865 | 0.1571 | 65.7 (8.2) | 74.0 | 36.4 | 53.0 | 90.9 | 1/36 | 35/36 | 1862 |
| r025@e1.5 | 68 | 0.1963 | 0.1656 | 0.1516 | 82.8 (6.1) | 92.2 | 50.0 | 74.2 | 100.0 | 2/36 | 34/36 | 1892 |
| r025@e2 **best** | 90 | 0.1780 | 0.1621 | 0.1497 | 82.8 (6.2) | 92.2 | 50.0 | 74.2 | 100.0 | 2/36 | 34/36 | 1872 |
| r050@e0.5 | 45 | 0.2715 | 0.1944 | 0.1568 | 69.7 (7.8) | 75.3 | 50.0 | 54.5 | 100.0 | 4/36 | 33/36 | 1540 |
| r050@e1 | 89 | 0.2140 | 0.1600 | 0.1417 | 66.7 (8.1) | 75.3 | 36.4 | 54.5 | 90.9 | 0/36 | 36/36 | 1926 |
| r050@e1.5 **best** | 134 | 0.1664 | 0.1415 | 0.1355 | 80.8 (6.7) | 89.6 | 50.0 | 71.2 | 100.0 | 2/36 | 34/36 | 1927 |
| r050@e2 | 178 | 0.1626 | 0.1372 | 0.1337 | 74.7 (7.5) | 81.8 | 50.0 | 62.1 | 100.0 | 1/36 | 35/36 | 1885 |
| r100@e0.5 | 89 | 0.2470 | 0.1741 | 0.1428 | 66.7 (8.0) | 75.3 | 36.4 | 54.5 | 90.9 | 0/36 | 36/36 | 1621 |
| r100@e1 **best** | 178 | 0.1899 | 0.1391 | 0.1295 | 82.8 (6.2) | 92.2 | 50.0 | 74.2 | 100.0 | 2/36 | 34/36 | 1787 |
| r100@e1.5 | 267 | 0.1462 | 0.1235 | 0.1260 | 79.8 (6.9) | 88.3 | 50.0 | 69.7 | 100.0 | 0/36 | 36/36 | 1808 |
| r100@e2 | 356 | 0.1413 | 0.1177 | 0.1236 | 66.7 (8.0) | 75.3 | 36.4 | 54.5 | 90.9 | 2/36 | 34/36 | 1794 |

The train-validation gap is probe loss (a fixed sample of the run's own training rows) against dev loss, at the same checkpoint.

## Paired against the untrained model (same dev tasks)

| checkpoint | diff (pts) | SE cluster | p (McNemar) | p (sign over prompts) | gained | lost |
|---|---|---|---|---|---|---|
| r025@e0.5 | 18.2 | 10.3 | 0.00643 | 0.118 | 29 | 11 |
| r025@e1 | 20.2 | 10.1 | 0.00166 | 0.0574 | 29 | 9 |
| r025@e1.5 | 37.4 | 8.7 | 7.84e-10 | 0.000519 | 39 | 2 |
| r025@e2 | 37.4 | 8.0 | 1.46e-11 | 6.1e-05 | 37 | 0 |
| r050@e0.5 | 24.2 | 8.5 | 8.43e-06 | 0.0117 | 27 | 3 |
| r050@e1 | 21.2 | 10.6 | 0.00191 | 0.0768 | 32 | 11 |
| r050@e1.5 | 35.4 | 8.1 | 5.82e-11 | 0.000244 | 35 | 0 |
| r050@e2 | 29.3 | 8.9 | 4.18e-07 | 0.00342 | 32 | 3 |
| r100@e0.5 | 21.2 | 9.1 | 0.000324 | 0.0225 | 27 | 6 |
| r100@e1 | 37.4 | 8.0 | 1.46e-11 | 6.1e-05 | 37 | 0 |
| r100@e1.5 | 34.3 | 8.0 | 1.16e-10 | 0.000244 | 34 | 0 |
| r100@e2 | 21.2 | 10.0 | 0.00107 | 0.0352 | 30 | 9 |

## Convergence rule (PLAN.md 4.3): stop growing when a doubling moves dev pass@1 by less than the noise

| from -> to | compared | diff (pts) | noise (SE, pts) | p (sign over prompts) | moves more than the noise |
|---|---|---|---|---|---|
| r025 -> r050 | best: r050@e1.5 vs r025@e2 | -2.0 | 1.4 | 0.5 | yes |
| r025 -> r050 | final: r050@e2 vs r025@e2 | -8.1 | 4.4 | 0.125 | yes |
| r050 -> r100 | best: r100@e1 vs r050@e1.5 | 2.0 | 1.4 | 0.5 | yes |
| r050 -> r100 | final: r100@e2 vs r050@e2 | -8.1 | 6.8 | 0.688 | yes |

The decision is Phil's (PLAN.md section 7): this table is the evidence, not the answer.
