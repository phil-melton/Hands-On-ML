## OpenCV 5.0 notes (Python cv2 5.0.0; measured against 4.14.0)

Names listed as missing are absent from the 5.0 main wheel: do not write them.

### Missing in 5.0: write the replacement
- `cv2.CascadeClassifier`, and the Haar XML files (`cv2.data.haarcascades` exists but ships no XML): use `cv2.FaceDetectorYN.create` (YuNet, needs an ONNX model file).
- `cv2.HOGDescriptor`, `cv2.ximgproc`, `cv2.CV_16BF`: no confirmed replacement.
- `cv2.ml` (`KNearest_create`, `SVM_create`, `RTrees_create`): use scikit-learn or numpy.
- `cv2.gapi`: call the cv2 functions directly.
- `cv2.dnn.readNetFromCaffe`, `readNetFromDarknet`: convert the model to ONNX, use `cv2.dnn.readNetFromONNX`.
- `cv2.AKAZE_create`, `KAZE_create`, `BRISK_create`: use `cv2.ORB_create` or `cv2.SIFT_create`.
- `cv2.xfeatures2d` (for SURF), `cv2.SURF_create`: use `cv2.SIFT_create`.
- `cv2.AgastFeatureDetector_create`: use `cv2.FastFeatureDetector_create`.
- `cv2.BOWKMeansTrainer`: cluster descriptors with `cv2.kmeans`, build the histogram yourself.
- `cv2.TrackerGOTURN_create`, `TrackerKCF_create`, `TrackerCSRT_create`, `cv2.legacy`: use `cv2.TrackerMIL_create` (no model file), or `TrackerNano_create` / `TrackerVit_create` (need ONNX files).
- `cv2.createBackgroundSubtractorMOG`: use `cv2.createBackgroundSubtractorMOG2`.
- `cv2.estimateRigidTransform`: use `cv2.estimateAffinePartial2D` (fullAffine=False) or `cv2.estimateAffine2D` (fullAffine=True).
- `cv2.cv`, `cv2.CV_AA`: OpenCV 2 names; use current ones, e.g. `cv2.LINE_AA`.
- Still present: ORB, SIFT, FastFeatureDetector, MSER, BFMatcher, FlannBasedMatcher, QRCodeDetector, `cv2.aruco.ArucoDetector`, `cv2.dnn.readNetFromONNX`, the MOG2 and KNN background subtractors, CLAHE.

### Return shapes lost the middle axis: (N,1,k) becomes (N,k)
- `HoughLinesP` (N,4); `HoughLinesWithAccumulator` (N,3); `findNonZero` (N,2); `findChessboardCorners` and `findChessboardCornersSB` corners (N,2); `convexityDefects` (N,4); `minEnclosingTriangle` triangle (3,2); `calcHist` (256,); ArUco `ids` (N,).
- So `line[0]`, `pts[:, 0, 0]`, `defects[:, 0]` break. Write `.reshape(-1, k)` on the result: valid in 4.x and 5.0.
- Unchanged, do not "fix": `findContours` contours (N,1,2), `approxPolyDP`, `convexHull`, `goodFeaturesToTrack`, `HoughLines`, `HoughCircles`, `fitLine`, `kmeans` labels, `calcOpticalFlowPyrLK` status, `findHomography` mask.

### dtype acceptance changed per function, not globally
- bool arrays now accepted by `countNonZero`, `findNonZero`, `mask=` arguments (`cv2.bitwise_and(img, img, mask=bool_mask)`); `moments` only with `binaryImage=True`.
- bool arrays still raise in `findContours`, `connectedComponents*`, `threshold`, `dilate`/`erode`, the blurs, `cvtColor`, `warpAffine`, `add`: pass `mask.astype(np.uint8)`.
- `cv2.bitwise_not` accepts bool but returns all-true bytes 0xFE/0xFF, not the negation: use `~mask` in numpy.
- int64 (numpy's default int) is still silently narrowed to int32 with wraparound (`cv2.add` on `2**40` gives 0); `cv2.threshold` on int64 raises.

### Results changed without raising
- `MSER_create()` with default parameters returns 0 regions (22 to 0 on one texture image; 0 on all 7 images tried). `min_diversity=0` gave 170 regions on a text image.
- `warpAffine`, `warpPerspective` (uint8): up to 4 gray levels differ from 4.14 on about 3% of pixels.
- `putText` with Hershey fonts ignores `lineType`, always anti-aliases; 13% of the pixels of a 60x240 test image differ. `getTextSize` still returns `((w, h), baseline)`, but all three values changed and h is the font's line height, independent of the string.
- `INTER_NEAREST` mapping is unchanged (identical on 450 size pairs). `INTER_NEAREST_EXACT` changed on at least 5 of them and is not equivalent to it.
- `cap.get(prop)` returns -1.0 where 4.14 returned 0.0 (probe: unopened capture): `== 0` no longer detects it.

### Unchanged since 4.x, still true
- Drawing on `img[:, :, ::-1]` or `img.T` raises; an ROI slice writes through to the parent.
- `cv2.circle`/`cv2.line` with float coordinates raise; a tuple of numpy ints works. `fillPoly` accepts int64 points, rejects float32.
- `getPerspectiveTransform` needs float32 points; `perspectiveTransform` needs (N,1,2).
- `cv2.resize` dsize is (width, height). `imread`/`imdecode` return None on failure.
- `findChessboardCorners` returns corners reversed when the board's top-left square is white.
- Raise: `cvtColor(gray, COLOR_BGR2GRAY)`, `Canny` or `findContours` on float, `GaussianBlur` with even ksize, `cv2.imshow`/`cv2.waitKey` in headless builds.
