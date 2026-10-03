# Writing a task family

A family is one module in this folder. It manufactures inputs *together with their
answers*, and it knows how to tell a right answer from a wrong one.
[`contour_area.py`](contour_area.py) is the worked example; [`../evaluate.py`](../evaluate.py)
grades any family through the four gates.

| name | what it is |
|---|---|
| `SPEC` | `TaskSpec`: the contract the model sees (`SPEC.to_prompt()`) |
| `PHRASINGS` | 3–5 rewordings of the task: prompt diversity, same contract |
| `make_scene(seed) -> Scene` | manufactures an input and its answer, deterministically |
| `inputs(scene) -> dict` | keyword arguments for `solve()` |
| `verify_manufactured(scene, out) -> Verdict` | recovers the manufactured answer within a *derived* tolerance |
| `transforms(scene) -> {name: inputs}` | the task's symmetries |
| `verify_metamorphic(base, variants) -> Verdict` | the invariants those symmetries imply |
| `REFERENCE`, `MUTANTS` | a correct solution plus plausible mistakes. Their gate matrix is the verifier's own test |

## Checklist (about one evening per family)

1. **Choose the answer first.** What quantity does the task produce? Build the scene *from* it.
2. **Derive the tolerance from the measurement physics** before writing code. Where does the error
   live (on the boundary? in sub-pixel localization? in the interpolation kernel?), and how does it
   scale? Write the scaling law in the docstring.
3. **Calibrate the constant on gauge blocks** (single-object scenes). Add a margin, and record the
   measured numbers in a comment next to the constant.
4. **List the symmetries** and what each implies for the output: invariant, covariant (shifts and
   rotates with the input), or neither.
5. **Write 5–8 mutants**: the mistakes you expect a model to make, especially the 4.x idioms in
   [`../../probes/results/diff_4.14_vs_5.0.md`](../../probes/results/diff_4.14_vs_5.0.md).
6. **Pin the gate matrix in a test.** If a mutant passes every gate, either it is harmless for this
   task (document it, like `rgb2gray`) or your verifier has a blind spot.

## Next families (yours to design)

These are ordered roughly by how hard the 4.14 → 5.0 diff hits them. The last column is the
question to answer *before* writing code. It is not the answer.

| family | 5.0 exposure (measured) | physics question for the verifier |
|---|---|---|
| `hough_segments`: detect line segments, return endpoints | `HoughLinesP` → `(N, 4)`, was `(N, 1, 4)` | Which metric: endpoint distance, or Hausdorff? How does the error scale with line thickness and with the angle step? Is a segment's direction covariant under rot90? |
| `chessboard_corners` | `findChessboardCorners`/`SB` → `(N, 2)` | Sub-pixel corner error vs blur σ and noise: what scaling law? |
| `perspective_rectify`: find the quad, warp to W×H | warp interpolation revised (max 4 gray levels, ~3 % of pixels) | Compare in which space: pixels (fragile), PSNR vs the manufactured texture, or recovered corner positions? |
| `feature_homography`: match keypoints, recover H | AKAZE / KAZE / BRISK / AGAST gone; ORB and SIFT remain | Error in corner reprojection, in px. How does it scale with inlier count? |
| `face_detect` | `CascadeClassifier` gone; `FaceDetectorYN` needs an ONNX model | Is this a *knowledge* family (API choice and model loading) more than a measurement one? |
| `histogram_stats` | `calcHist` → `(256,)`, was `(256, 1)` | Integer counts are exact, so the tolerance is zero; the trap is shape and dtype |
| `color_segmentation` (HSV `inRange`) | none | Unlike `contour_area`, this one *does* catch BGR/RGB swaps. Red hue wraps around 0/180: what symmetry does that break? |
| `text_layout` | `getTextSize` and glyph pixels changed | If the font changes, what stays invariant? Bounding-box containment, not pixels |
| `components_stats` | bool masks now accepted | Areas and centroids of pixel sets are exact; centroids are covariant under translation |
| `mser_regions` | **0 regions** with default params on 5.0.0 (7 on 4.14) | A verifier would have caught this library regression. Report it upstream with `probes/api_diff.py` as the minimal repro |
