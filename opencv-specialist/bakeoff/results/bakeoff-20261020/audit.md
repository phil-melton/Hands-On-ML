# Taxonomy audit

Up to 3 randomly drawn failing replies per class (fixed seed). Read each: does the class match what went wrong? `also` lists the other signals that fired for the same reply.

## format (3 shown)

### r2-4b / shape_gftt_ctrl / sample 1

- rule `syntax-at-runtime`; also nothing
- evidence: exception: SyntaxError keyword argument repeated: maxCorners (solution.py, line 44)

```python
import cv2
import numpy as np

def solve(img, max_corners) -> dict:
    """
    Find the strongest corners in a grayscale image using OpenCV's Shi-Tomasi corner detection.
    
    Parameters:
        img: uint8, shape HxW, grayscale image
        max_corners: int, maximum number of corners to return
    
    Returns:
        dict with key 'corners': Nx2 array of float coordinates (x, y)
    """
    # Ensure the image is grayscale and converted to float for computation if needed
    # OpenCV functions usually accept uint8 or float, but we ensure it's correct
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform Shi-Tomasi corner detection
    # k: number of corners to detect
    # minEigVal: minimum eigenvalue threshold (0.01 * max_val for normalized images)
    # maxCorners: maximum number of corners to find
    # useGoodFeaturesToTrack: True (Shi-Tomasi uses this internally via goodFeaturesToTrack with specific params)
    # Actually, Shi-Tomasi is often done via goodFeaturesToTrack with specific criteria
    
    # Shi-Tomasi detection using goodFeaturesToTrack with specific parameters
    # Since OpenCV 5.0, goodFeaturesToTrack is still available and reliable
    # We set goodFeaturesToTrack parameters to enforce Shi-Tomasi behavior:
    # - method=0 (Shi-Tomasi)
    # - qualityLevel: relative threshold
    # - maxCorners: limit
    # - useGoodFeaturesToTrack: True (required for Shi-Tomasi in some contexts, but goodFeaturesToTrack with method=0 is sufficient)
    
    # According to OpenCV docs, Shi-Tomasi is accessed via goodFeaturesToTrack with method=0
    
    corners = cv2.goodFeaturesToTrack(
        gray,
```

### r3-9b / test-feature_homography-003 / sample 3

- rule `lint:syntax`; also ['format:syntax-at-runtime']
- evidence: line 68: [syntax] invalid syntax

```python
import cv2
import numpy as np

def solve(img_a, img_b) -> dict:
    """
    Registers img_a onto img_b using feature matching and returns the homography matrix.
    
    Parameters:
    img_a (np.ndarray): Source image (BGR uint8), shape HxWx3.
    img_b (np.ndarray): Target image (BGR uint8), shape HxWx3.
    
    Returns:
    dict: {'H': homography_matrix} where H is a 3x3 float array normalized to H[2,2]=1.
          Maps (x, y) in img_a to (x, y) in img_b.
    """
    
    # Convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing with Gaussian blur to reduce noise
    blur_val = 5.0
    blurred_a = cv2.GaussianBlur(gray_a, (9, 9), blur_val)
    blurred_b = cv2.GaussianBlur(gray_b, (9, 9), blur_val)
    
    # Initialize ORB detector (OpenCV 5.0 default)
    orb = cv2.ORB_create(nfeatures=2000) # Request many features, filtering will happen during matching
    
    # Detect and compute keypoints and descriptors
    kp_a, des_a = orb.detectAndCompute(blurred_a, None)
    kp_b, des_b = orb.detectAndCompute(blurred_b, None)
    
    # Perform feature matching
    # BFMatcher with Hamming distance (for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_a, des_b)
    
    # Sort matches by score (distance) and take the best ones
    matches.sort(key=lambda m: m.distance)
    
```

### r3-9b / test-hough_segments-000 / sample 0

- rule `lint:syntax`; also ['format:syntax-at-runtime']
- evidence: line 180: [syntax] '(' was never closed

```python
import cv2
import numpy as np

def solve(img):
    """
    Detects straight line segments in a binary-like image (bright strokes on dark background).
    
    Returns:
        dict: {'segments': list of [x1, y1, x2, y2]}
    """
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get bright strokes (adjust threshold if necessary, assuming dark background)
    # Since background is dark and strokes are bright (1-4px wide), a simple binary threshold works.
    # Assuming background is close to 0 and strokes are significantly brighter.
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Use Hough Line Transform (Probabilistic) to detect lines
    # Parameters: rho=1, theta=1 degree, threshold=min length of a line segment (min 40px)
    #   max_gap=0 (to connect gaps within a stroke)
    #   param1=1 (accumulator threshold for internal)
    #   param2=2 (accumulator threshold for external)
    lines = cv2.HoughLinesP(
        binary, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, # Adjust based on image brightness, but strokes are thin and distinct. 
                       # A lower threshold might detect noise, higher might miss faint parts. 
                       # Given "bright strokes", 100 is a reasonable starting point for uint8 images.
                       # However, we need to ensure we capture the *entire* stroke.
                       # Let's try a dynamic approach or standard values.
                       # For a robust solution on 'bright' strokes, we can start with a moderate threshold.
                       # But the problem says strokes are 1-4px wide and >=40px long.
                       # Let's set a reasonable threshold.
                       # To be safe against varying brightness, let's normalize or use a relative threshold?
                       # The prompt implies a simple "bright on dark". Let's stick to a fixed threshold 
                       # or try to infer from the image histogram if needed. 
                       # Given the constraints, a simple Otsu might be better for thresholding.
```

## api (3 shown)

### r2-4b / test-contour_area-002 / sample 3

- rule `cv2-attribute`; also nothing
- evidence: exec-only: exception on input 0: AttributeError module 'cv2' has no attribute 'undilate'

```python
import cv2
import numpy as np

def solve(img):
    """
    Measures the area of light objects in a BGR image.
    
    Args:
        img: uint8, shape HxWx3. BGR; bright solid shapes on a dark uniform background; 
             shapes touch neither each other nor the border.
    
    Returns:
        dict:
            - 'areas': float, shape N. px^2, one per shape, area inside the outer boundary, largest first
            - 'count': int, shape scalar. number of shapes
    """
    # Convert BGR to HSV for better thresholding on brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of bright colors (since shapes are bright and background is dark)
    # We'll use a simple threshold on Value channel since we have HSV
    lower_bright = np.array([0, 50, 50])
    upper_bright = np.array([180, 255, 255])
    
    # Threshold the HSV image to get bright regions
    mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Apply some morphological operations to ensure clean contours
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    eroded_mask = cv2.undilate(dilated_mask, kernel, iterations=2) # Wait, undilate doesn't exist, use erode
    
    # Correct morphological operations: Dilate then Erode (Closing)
    # Actually, for separating potential noise and keeping shapes:
    # Let's just use simple opening to remove small noise
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
```

### r2-4b / test-perspective_rectify-003 / sample 3

- rule `call-signature`; also nothing
- evidence: exec-only: exception on input 0: TypeError function takes at most 1 argument (2 given)

```python
import cv2
import numpy as np

def solve(img, out_w, out_h) -> dict:
    """
    Detects the quadrilateral outline of a bright card on a dark background,
    orders the corners (TL, TR, BR, BL), and warps the card to a rectangle.
    
    Args:
        img: uint8, shape HxWx3. BGR photo.
        out_w: int, width of the rectified image.
        out_h: int, height of the rectified image.
        
    Returns:
        dict with 'corners' (4x2 float) and 'warped' (out_hxout_wx3 uint8).
    """
    # Convert to grayscale and threshold to separate the bright card from the dark background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up noise and ensure a solid quadrilateral shape
    kernel = np.ones((5,5), np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)
    eroded_thresh = cv2.erode(dilated_thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback if no large contour found, though task implies card exists
        raise ValueError("No card contour detected.")
    
    # Get the largest contour (the card)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a quadrilateral
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Ensure we have 4 points
```

### r2-4b / test-perspective_rectify-014 / greedy 0

- rule `cv2-attribute`; also nothing
- evidence: exec-only: exception on input 0: AttributeError module 'cv2' has no attribute 'undilate'

```python
import cv2
import numpy as np

def solve(img, out_w, out_h):
    """
    Detects a rectangular card in a dark background image, finds its four corners,
    and returns the corners and a perspective-corrected (rectified) image.
    
    Parameters:
        img: uint8, shape HxWx3. BGR photo.
        out_w: int, width of the rectified image.
        out_h: int, height of the rectified image.
    
    Returns:
        dict with:
            - 'corners': float, shape 4x2. Card corners (x, y) in order TL, TR, BR, BL.
            - 'warped': uint8, shape out_hxout_wx3. Rectified card image.
    """
    # Convert to HSV to segment the bright card
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of bright colors (card is bright on dark background)
    # Adjust lower and upper bounds based on typical bright objects on dark backgrounds
    lower_bright = np.array([0, 50, 50])
    upper_bright = np.array([180, 255, 255])
    
    # Threshold the HSV image to get bright components
    mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    eroded_mask = cv2.undilate(dilated_mask, kernel, iterations=2) # Note: cv2.erode is standard, 'undilate' is not a function. Corrected below.
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming the card is the largest bright object)
    if not contours:
```

## lint (3 shown)

### r3-9b / call_imdecode_none / sample 1

- rule `lint:broad-except`; also nothing
- evidence: line 43: [broad-except] `except Exception` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(buf) -> dict:
    """
    Decodes an encoded image bytes (PNG) into a colour image using OpenCV 5.0.
    If decoding fails or the bytes are not a valid image, returns a dict with 
    shape [0, 0, 0] and mean 0.0.
    
    Parameters:
        buf (bytes): Bytes of an encoded image file (e.g., PNG).
        
    Returns:
        dict:
            - 'shape': list[int] of [height, width, channels] or [0, 0, 0]
            - 'mean': float, mean of all pixel values, or 0.0
    """
    try:
        # Attempt to decode the buffer as a PNG image
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Check if decoding was successful (img will be None on failure)
        if img is None:
            return {
                'shape': [0, 0, 0],
                'mean': 0.0
            }
        
        # Ensure the image is in BGR format (OpenCV default for color images)
        # The problem asks for 'colour image', which in OpenCV terms is BGR.
        # However, for calculating the mean, the channel order does not matter 
        # as long as we include all channels. We assume the request implies 
        # returning the OpenCV decoded image (BGR).
        
        # Calculate mean across all pixels and channels
        mean_val = float(np.mean(img))
        
        return {
            'shape': list(img.shape),
            'mean': mean_val
```

### r3-9b / shape_houghlinesp / sample 2

- rule `lint:broad-except`; also nothing
- evidence: line 38: [broad-except] `except Exception` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(edges: np.ndarray) -> dict:
    """
    Detects straight line segments in a binary edge image and returns their endpoints.
    
    Parameters:
    edges (np.ndarray): A binary edge image (uint8) where 0 represents background and 255 represents edges.
    
    Returns:
    dict: A dictionary with the key 'segments' mapping to a NumPy array of shape (N, 4) where each row 
          represents a line segment as [x1, y1, x2, y2].
    """
    try:
        # Ensure the input is uint8
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        
        # Use HoughLinesP to detect line segments
        rho = 1  # Distance resolution in pixels
        theta = np.pi / 180  # Angular resolution in radians
        threshold = 100  # Minimum number of votes to consider a segment
        minLineLength = 100  # Minimum line length as specified in the problem
        maxLineGap = 10  # Maximum gap between points to be considered part of the same line
        
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        
        # If no lines are detected, return an empty array
        if lines is None:
            return {'segments': np.array([])}
        
        # Convert the detected lines to a NumPy array of shape (N, 4)
        segments = np.array(lines).reshape(-1, 4)
        
        return {'segments': segments}
    
    except Exception as e:
        # In case of any unexpected error, return an empty segments array
        return {'segments': np.array([])}
```

### r4-7b-notes / call_imdecode_none / sample 1

- rule `lint:broad-except`; also nothing
- evidence: line 25: [broad-except] `except Exception` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(buf) -> dict:
    try:
        # Decode the image bytes
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            # Buffer is not a valid image
            return {
                'shape': [0, 0, 0],
                'mean': 0.0
            }
        
        # Get the shape and mean of the image
        shape = img.shape
        mean = np.mean(img)
        
        return {
            'shape': shape,
            'mean': mean
        }
    
    except Exception as e:
        # If any exception occurs, return invalid shape and mean
        return {
            'shape': [0, 0, 0],
            'mean': 0.0
        }
```

## timeout (1 shown)

### r2-4b / test-text_layout-000 / sample 4

- rule `timeout`; also nothing
- evidence: exec-only: timeout: no result after 60s; process group killed

```python
import cv2

def solve(img, box, text, face, scale, thickness, color) -> dict:
    """
    Draws text centered in the specified box on a copy of the image.
    If the text exceeds the box dimensions at the requested scale, it shrinks
    the font scale until it fits. A 10% tolerance is allowed for the final
    scale check before adjustment.
    
    Parameters:
        img: uint8, shape HxWx3. BGR image.
        box: tuple or list of 4 ints (x, y, w, h).
        text: str, the text to draw.
        face: int, OpenCV font face constant.
        scale: float, initial font scale.
        thickness: int, stroke thickness.
        color: tuple of 3 ints, BGR color.
    
    Returns:
        dict with keys:
            'img': the modified image.
            'scale': the scale actually used.
            'text_size': (width, height) of the text at that scale.
    """
    x, y, w_box, h_box = box
    H, W, _ = img.shape
    
    # Create a copy of the image to avoid modifying the original
    img_copy = img.copy()
    
    # Initial check: calculate text size at requested scale
    text_w, text_h = cv2.getTextSize(text, face, scale, thickness)[0]
    
    # Define the center of the box
    center_x = x + w_box // 2
    center_y = y + h_box // 2
    
    # While the text does not fit within the box, reduce the scale
    # Fit condition:
    # 1. text_w <= w_box
```

## convention (3 shown)

### r2-4b / call_cvtcolor_gray / sample 0

- rule `TypeError`; also nothing
- evidence: exception: TypeError unsupported operand type(s) for |: 'str' and 'NoneType'

```python
def solve(img: "numpy.ndarray" | None) -> dict:
    """
    Converts the input image to a single-channel grayscale image.
    
    If the input is already grayscale (shape HxW), it returns it unchanged.
    If the input is color (shape HxWx3 or HxWxC), it converts it to grayscale.
    
    Args:
        img (numpy.ndarray): Input image of shape HxW (grayscale) or HxWx3 (BGR).
        
    Returns:
        dict: A dictionary containing the key 'gray' with the grayscale image as the value.
    """
    if img is None:
        raise ValueError("Input image cannot be None")
    
    # Check if the input is already grayscale
    # Grayscale images in OpenCV have shape (H, W)
    # Color images have shape (H, W, C) where C=3
    if len(img.shape) == 2:
        gray = img.copy()
    else:
        # Convert BGR to grayscale using cv2.cvtColor
        # OpenCV 5.0 still uses the same conversion flags as previous versions
        # cv2.COLOR_BGR2GRAY is the correct flag for BGR input
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return {'gray': gray}
```

### r3-9b / test-histogram_stats-000 / greedy 0

- rule `TypeError`; also nothing
- evidence: exec-only: exception on input 0: TypeError only 0-dimensional arrays can be converted to Python scalars

```python
import cv2
import numpy as np

def solve(img: np.ndarray) -> dict:
    """
    Computes the 256-bin gray-level histogram and statistics (mean, std, median)
    for a BGR image using ITU-R BT.601 luma coefficients.
    
    Parameters:
        img (np.ndarray): uint8, shape (H, W, 3), BGR color image.
        
    Returns:
        dict:
            - 'hist': np.ndarray of shape (256,), integer counts.
            - 'mean': float, arithmetic mean of gray levels.
            - 'std': float, population standard deviation.
            - 'median': int, lower median (smallest v where count <= v is >= N/2).
    """
    # Convert BGR to Gray using cv2.COLOR_BGR2GRAY (which uses BT.601 coefficients)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure we are working with float for mean/std calculations to avoid integer truncation issues
    gray_float = gray.astype(np.float32)
    
    # Calculate statistics
    mean_val = np.mean(gray_float)
    std_val = np.std(gray_float, ddof=0)  # Population standard deviation
    
    # Calculate histogram
    # cv2.calcHist expects a list of arrays and a list of channels.
    # Since we have a single channel (gray), we pass it directly.
    # We specify the range as [0, 256] to get 256 bins (0 to 255).
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # cv2.calcHist returns float64 by default, convert to int64 or int32 as per requirement "int"
    hist = hist.astype(np.int64)
    
    # Calculate median
    # Sort the gray values to find the median
    sorted_gray = np.sort(gray)
    n_pixels = len(sorted_gray)
```

### r4-7b-notes / draw_transposed_view / sample 4

- rule `cv2-assertion`; also nothing
- evidence: exception on input 0: error OpenCV(5.0.0) :-1: error: (-5:Bad argument) in function 'circle' > Overload resolution failed: > - Layout of the output array img is incompatible with cv::Mat > - Expected Ptr<cv::UMat> for argument 'img'

```python
import cv2
import numpy as np

def solve(img_t, x, y, radius) -> dict:
    H, W = img_t.shape
    img = img_t.T
    cv2.circle(img, (x, y), radius, 255, -1)
    return {'img': img.T}
```

## algorithm (3 shown)

### r1-7b / test-feature_homography-001 / sample 3

- rule `verifier-value`; also nothing
- evidence: manufactured: scene 4004040: inset-corner error 203.62 px exceeds 3.42 px (N = 307 true correspondences)

```python
import cv2
import numpy as np

def solve(img_a, img_b) -> dict:
    # Convert images to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_a, descriptors_b)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points_a = np.zeros((len(matches), 2), dtype=np.float32)
    points_b = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_a[i, :] = keypoints_a[match.queryIdx].pt
        points_b[i, :] = keypoints_b[match.trainIdx].pt

    # Define the 8 points around the corners of img_a
    corner_points_a = np.array([
        [16, 16],
        [16, img_a.shape[1] - 16],
        [img_a.shape[0] - 16, img_a.shape[1] - 16],
        [img_a.shape[0] - 16, 16]
    ], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(corner_points_a, points_b[:4], cv2.RANSAC)

    # Normalize H so that H[2, 2] = 1
```

### r2-4b / dtype_uint32_add / sample 1

- rule `gotcha-value`; also nothing
- evidence: item 0: 21 of 48 saturating sums are wrong

```python
def solve(a, b) -> dict:
    """
    Computes the saturating sum of two uint32 images.
    
    The addition saturates at the maximum uint32 value (2**32 - 1) instead of wrapping around.
    
    Parameters:
        a (numpy.ndarray): uint32 array of shape HxW.
        b (numpy.ndarray): uint32 array of shape HxW.
        
    Returns:
        dict: A dictionary containing:
            - 'total': The saturated sum as a uint32 numpy array.
    """
    import numpy as np
    
    # Ensure inputs are uint32
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    
    # Calculate the maximum uint32 value
    MAX_UINT32 = 2**32 - 1
    
    # Perform the addition
    result = a + b
    
    # Saturate the result
    # Since numpy operations with uint32 will wrap around, we use a mask
    # to ensure the result does not exceed MAX_UINT32.
    # We can use np.minimum to clamp the values.
    result = np.minimum(result, MAX_UINT32)
    
    return {'total': result}
```

### r4-7b-notes / test-text_layout-002 / greedy 0

- rule `verifier-value`; also nothing
- evidence: manufactured: scene 4008080: ink centre is off the box centre by (+0.5, +7.5) px, allowed (±3.1, ±6.8)

```python
import cv2
import numpy as np

def solve(img, box, text, face, scale, thickness, color) -> dict:
    x, y, w, h = box
    H, W, _ = img.shape
    
    # Ensure the box is within the image boundaries
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    
    # Calculate the initial text size
    text_size, _ = cv2.getTextSize(text, face, scale, thickness)
    
    # Adjust scale if the text does not fit within the box
    while text_size[0] > w or text_size[1] > h:
        scale *= 0.9
        text_size, _ = cv2.getTextSize(text, face, scale, thickness)
    
    # Calculate the position to center the text
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h - text_size[1]) // 2 + text_size[1]
    
    # Create a copy of the image to draw on
    img_copy = img.copy()
    
    # Draw the text on the image
    cv2.putText(img_copy, text, (text_x, text_y), face, scale, color, thickness, cv2.LINE_AA)
    
    return {
        'img': img_copy,
        'scale': scale,
        'text_size': text_size
    }
```

## other (3 shown)

### r1-7b / dtype_persp_nx2 / greedy 0

- rule `exception:NameError`; also nothing
- evidence: exception on input 0: NameError name 'cv2' is not defined

```python
def solve(pts, H) -> dict:
    pts_homogeneous = cv2.convertPointsToHomogeneous(pts)
    mapped_homogeneous = H @ pts_homogeneous.T
    mapped_homogeneous = mapped_homogeneous.T
    mapped = cv2.convertPointsFromHomogeneous(mapped_homogeneous)
    return {'mapped': mapped}
```

### r4-7b-notes / call_resize_order / sample 2

- rule `exception:NameError`; also nothing
- evidence: exception on input 0: NameError name 'cv2' is not defined

```python
def solve(src, ref) -> dict:
    resized = cv2.resize(src, (ref.shape[1], ref.shape[0]))
    return {'resized': resized}
```

### r4-7b-notes / test-histogram_stats-001 / sample 3

- rule `exception:NameError`; also nothing
- evidence: exec-only: exception on input 0: NameError name 'cv2' is not defined

```python
def solve(img) -> dict:
    # Convert BGR to BT.601 luma
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    
    # Calculate the mean and standard deviation
    mean, std = cv2.meanStdDev(gray)
    mean = float(mean)
    std = float(std)
    
    # Calculate the median
    hist_cumsum = hist.cumsum()
    hist_cumsum = hist_cumsum / hist_cumsum[-1]
    median = np.interp(0.5, hist_cumsum, np.arange(256))
    median = int(np.round(median))
    
    return {
        'hist': hist,
        'mean': mean,
        'std': std,
        'median': median
    }
```
