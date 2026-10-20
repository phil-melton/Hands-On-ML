# upstream/: drafts of opencv/opencv issues for the project owner to file

**These are drafts for the owner to file.** Nothing was posted, commented, signed in or submitted anywhere. The research
used public web pages and the unauthenticated GitHub REST API only (reads), on 2026-10-20.

Six candidates found while building the benchmark were each reproduced on OpenCV 5.0.0.93 (`opencv-python-headless`) and
compared with 4.14.0.94, classified, and searched for duplicates in opencv/opencv issues and PRs plus a web search.
**Only one of the six is worth filing.** Three are already fixed or reported upstream, two are documented behaviour.

Environment of all measurements: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), Python 3.12.3,
numpy 2.4.6, opencv-python-headless 5.0.0.93 (`cv2.__version__` is `5.0.0`) and 4.14.0.94.

| # | Title | Classification | Duplicate search | Recommendation |
|---|---|---|---|---|
| [01](01-mser-default-params.md) | `MSER_create()` with default parameters returns 0 regions | Bug in 5.0.0; root cause is one changed line (`var < 0.f` became `var < wp.p.minDiversity`) | **Duplicate**: [#29652](https://github.com/opencv/opencv/issues/29652) (closed), fixed on `5.x` by [#29653](https://github.com/opencv/opencv/pull/29653) (2026-08-04); introduced by [#20170](https://github.com/opencv/opencv/pull/20170) | **Do not file** |
| [02](02-bitwise-not-bool.md) | `bitwise_not` on a bool array returns bytes 0xFE/0xFF | Bug (silent wrong result), arguably also a doc gap | **None found.** Related only: `CV_Bool` crash fixes [#29880](https://github.com/opencv/opencv/issues/29880), [#30153](https://github.com/opencv/opencv/issues/30153) | **File** (still present on a `5.x` build, checked in C++) |
| [03](03-remap-map2-dtype.md) | `remap` accepts a `map2` of any dtype when `map1` is float32 | Bug in 5.0.0 (4.14.0 rejects it); missing validation | **Fixed on `5.x`** by [#30286](https://github.com/opencv/opencv/pull/30286) (merged 2026-10-19); I found no issue for it | **Do not file** |
| [04](04-chessboard-corner-order.md) | `findChessboardCorners` returns corners reversed when the top-left square is white | Expected, long-standing; documentation gap. Identical on 4.14.0 | [#22083](https://github.com/opencv/opencv/issues/22083) (open); docs expanded by [#30208](https://github.com/opencv/opencv/pull/30208) (merged 2026-10-12, `5.x`) | **Do not file** |
| [05](05-ccl-label-order.md) | `connectedComponents` label order is not raster order at connectivity 8 | Documented behaviour ("SAUF ... row major ordering of labels while Spaghetti and BBDT do not"). Identical on 4.14.0 | Not needed; the docstring answers it | **Do not file** |
| [06](06-sift-keypoint-offset.md) | SIFT keypoints are offset by about +0.25 px | Known; documented opt-in `enable_precise_upscale=True` (off by default). Identical on 4.14.0 | **Duplicate**: [#23123](https://github.com/opencv/opencv/issues/23123) (open), [#23124](https://github.com/opencv/opencv/pull/23124), [#23270](https://github.com/opencv/opencv/pull/23270) | **Do not file** (at most a comment on #23123 asking whether the default will change in 5.x) |

Each draft file has: a summary table, what I found, the duplicate search with links and state, and then the draft in the
structure of the opencv/opencv `bug_report.yml` template (System Information, Detailed description, Steps to reproduce,
Issue submission checklist). For the five I recommend not filing, the draft is kept only so the text can be reused for a comment.

## Findings that change the project's records

- **MSER**: PLAN.md section 0 item 3 says "I found no upstream issue yet". The issue exists (#29652), it was fixed on
  `5.x` on 2026-08-04, and the cause is identified. The fix is in no release (latest: 5.0.0, tag 2026-06-06; PyPI 5.0.0.93,
  2026-07-02), so `drift_mser_default` stays valid for the pinned wheel and will stop being a gotcha on the next release.
- **Bool support is moving on `5.x`**. PLAN.md section 0 item 1 and `cvbench/families/README.md` list which functions reject
  bool in 5.0.0. Since then `connectedComponents*` (#29622, merged 2026-07-28), `SIFT.detect` masks (#29679),
  `matchTemplate` (#29677), `inpaint` (#29676) and `findTransformECC` (#29681) have been changed to accept `CV_Bool`, `inRange` now
  rejects it instead of crashing (#30154); morphology (`dilate`/`erode`) acceptance is still open (#29798, #29799). The per-function bool table is a
  statement about 5.0.0.93 only; re-run `probes/api_diff.py` when a newer release is pinned.
- **remap**: fixed on `5.x` (#30286). The gotcha `dtype_remap_maps` (float64 maps) is unaffected because both versions raise.
- **findChessboardCorners**: reversed order is a property of the board (the dark-square layout), not of 5.0. The family
  contract must state which end is first or compare up to the board's symmetry. Also measured: `findChessboardCornersSB` on a
  6x4 board with a white top-left square flipped order between 4.14.0 and 5.0.0 (same-parity pattern, so the order is
  ambiguous by the new `5.x` documentation).
- **SIFT**: the +0.246 px offset is the `enable_precise_upscale=False` default. `cv2.SIFT_create(enable_precise_upscale=True)`
  removes it on both versions (error below 0.03 px on an analytic blob). The reference pipeline of `feature_homography` can
  use it.
- **connectedComponents**: label order is only guaranteed (row major) for `CCL_SAUF`; the family's explicit tie-break is the right design.
- **Incidental**: `cv2.groupRectangles` exists on 4.14.0 and is missing on 5.0.0 (plausibly part of the cascade/HOG removal, not
  verified; it is not in the probe results or in `api_check`'s catalogue). Not investigated further.

## Not verified

- The `5.x` runs are C++ only (the Python binding was not built from `5.x`): `opencv_core`, `opencv_imgproc` and `opencv_features` built
  from `5.x` commit `9123a3c209` (2026-10-20, `5.1.0-dev`). Results: 02 still returns 254/255 bytes (`repro_02_bitwise_not_bool.cpp`);
  01 and 03 are fixed (`check_5x_fixes.cpp`: default MSER finds 2 and 22 regions, `remap` with a float64 or uint8 `map2` throws).
- Why 5.0.0 `remap` accepts the mismatched `map2` (which early path), and whether narrow dtypes read out of bounds (03: results for
  `uint8`/`int8` `map2` varied between calls in a separate loop, which would fit, but no memory checker was run).
- ORB's reported -0.1 px offset (06): not isolated, no claim made.
- answers.opencv.org returned HTTP 403, so forum threads on chessboard order were not read.
- The opencv/opencv issue templates carry the note "If you are an automated agent, we have a special process for you": add
  three robot emojis to the end of the issue title. These drafts were written by an AI assistant. Whether that applies to an
  issue the owner files himself after checking it is his call.
- The checklist item "I updated to the latest OpenCV version and the issue is still there" is true for the latest *release*
  (5.0.0) but false for `5.x` HEAD in 01 and 03; for 02 it is true on both (checked in C++ on `5.x`).

## Files

`NN-slug.md` is the draft for issue NN and `repro_NN_slug.py` its standalone repro (numpy + cv2 only, under a second each).
Each repro prints the version and what it observed against what it expected, and exits 1 when it reproduces the problem
(05 is a measurement and always exits 0). Two C++ files back the `5.x` checks: `repro_02_bitwise_not_bool.cpp` and
`check_5x_fixes.cpp`.
The outputs quoted in the drafts were recorded with:

```
~/venvs/cv5/bin/python upstream/repro_01_mser_default_params.py      # 5.0.0.93
~/venvs/cv4/bin/python upstream/repro_01_mser_default_params.py      # 4.14.0.94 (venv per README: requirements-cv4.txt)
```
