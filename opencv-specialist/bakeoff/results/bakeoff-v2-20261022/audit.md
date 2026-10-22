# Taxonomy audit

Up to 3 randomly drawn failing replies per class (fixed seed). Read each: does the class match what went wrong? `also` lists the other signals that fired for the same reply.

## format (3 shown)

### r2-4b / test-chessboard_corners-005 / greedy 0

- rule `syntax-at-runtime`; also nothing
- evidence: exec-only: exception: SyntaxError keyword argument repeated: maxCorners (solution.py, line 55)

```python
import cv2
import numpy as np

def solve(img, squares_x, squares_y) -> dict:
    """
    Detects inner corners of a chessboard with sub-pixel accuracy.
    
    Parameters:
    - img: uint8, shape HxWx3. BGR photo of one chessboard.
    - squares_x: int, number of squares along the top edge.
    - squares_y: int, number of squares along the left edge.
    
    Returns:
    - dict with key 'corners': float, shape Nx2. Row-major order of inner corners.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    
    # Collect lines into two sets: horizontal and vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
    
    # Sort horizontal lines by y-coordinate (top to bottom)
    horizontal_lines.sort(key=lambda l: l[1])
```

### r3-9b / test-text_layout-015 / greedy 0

- rule `lint:syntax`; also ['format:syntax-at-runtime']
- evidence: line 173: [syntax] expected an indented block after function definition on line 147

```python
import cv2
import numpy as np

def solve(img, box, text, face, scale, thickness, color):
    """
    Draws text centered in a specified rectangle on an image,
    automatically scaling down the font if necessary to fit within the box.
    
    Returns a dictionary with the modified image, the actual scale used,
    and the text size at that scale.
    """
    
    # Unpack box coordinates
    x, y, w, h = box
    
    # Ensure the color is in BGR format (it is passed as such, but let's be safe if it's a tuple/list)
    if isinstance(color, int):
        color_bgr = (color, color, color)
    else:
        color_bgr = tuple(color)
        
    # Initial scale
    current_scale = scale
    
    # We need to find the largest scale that fits.
    # Strategy: Start with the requested scale. If it doesn't fit, reduce it.
    # Since cv2.putText doesn't have a direct "fit" function, we simulate fitting.
    # However, cv2.getTextSize gives us the bounding box for a given scale.
    # We can binary search or simply reduce iteratively. Given the nature of font scaling,
    # a simple iterative reduction or binary search is efficient.
    # Let's use a binary search approach for precision, or a simple loop if we assume
    # the scale won't need to be reduced by a huge factor (which is typical).
    # Actually, a simple loop reducing by a small factor or checking specific points is safer.
    # But the most robust way is to check the bounding box returned by getTextSize.
    
    # Get initial text size at the requested scale
    # getTextSize returns (width, height, baseline)
    # Note: getTextSize in OpenCV 5.0 returns (width, height, baseline)
    # The width and height are in pixels.
    
```

### r5-9b-notes / shape_houghlines_acc / greedy 0

- rule `lint:syntax`; also nothing
- evidence: line 130: [syntax] expected an indented block after 'else' statement on line 32

```python
import cv2
import numpy as np

def solve(edges: np.ndarray, min_votes: int) -> dict:
    """
    Run the standard Hough line transform at 1 px / 1 degree resolution.
    
    Args:
        edges: uint8, shape HxW. 0/255 edge map.
        min_votes: int, scalar. Discard lines with fewer votes than this.
        
    Returns:
        dict with 'lines': float, shape Nx3. One row (rho, theta, votes) per line,
        strongest first. Rho in px, theta in radians.
    """
    # HoughLinesP parameters for 1px/1deg resolution:
    # rho = 1, theta = pi/180
    rho = 1.0
    theta = np.pi / 180.0
    
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=min_votes,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None:
        lines = np.empty((0, 3), dtype=np.float32)
    else:
        # HoughLinesP returns (N, 4) in 4.x/5.0: (x1, y1, x2, y2)
        # We need to compute rho and theta for each line.
        # However, the standard HoughLinesP implementation in OpenCV already
        # returns lines in the Hough space if we use the standard parameters?
        # Actually, no. HoughLinesP returns lines in image coordinates.
        # The task asks for (rho, theta, votes).
        # To get rho and theta from (x1, y1, x2, y2):
        # rho = x1*cos(theta) + y1*sin(theta)
```

## api (3 shown)

### r2-4b / test-feature_homography-017 / greedy 0

- rule `cv2-attribute`; also nothing
- evidence: exec-only: exception on input 0: AttributeError module 'cv2' has no attribute 'ORB_SCORE_HARRIS'

```python
import cv2
import numpy as np

def solve(img_a, img_b) -> dict:
    """
    Computes the 3x3 homography H mapping img_a to img_b using matched local features.
    
    Args:
        img_a: uint8, shape HxWx3, BGR, reference picture.
        img_b: uint8, shape HxWx3, BGR, wider view.
        
    Returns:
        dict with key 'H': float, shape 3x3, normalized so H[2, 2] = 1.
    """
    # Convert to grayscale for feature detection
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    
    # Detect features using ORB (fast, robust to scale and illumination)
    # Parameters tuned for textured surfaces
    orb_a = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, cornerThreshold=31,
                           fastThreshold=20, edgetype=0, scoreType=cv2.ORB_SCORE_HARRIS)
    orb_b = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, cornerThreshold=31,
                           fastThreshold=20, edgetype=0, scoreType=cv2.ORB_SCORE_HARRIS)
    
    # Detect keypoints and compute descriptors
    kp_a, des_a = orb_a.detectAndCompute(gray_a, None)
    kp_b, des_b = orb_b.detectAndCompute(gray_b, None)
    
    if kp_a is None or kp_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        raise ValueError("No features detected in either image.")
    
    # Match descriptors using BFMatcher with TM_SQDIFF (good for texture)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_a, des_b, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
```

### r2-4b / test-perspective_rectify-008 / greedy 0

- rule `cv2-attribute`; also nothing
- evidence: exec-only: exception on input 0: AttributeError module 'cv2' has no attribute 'undilate'

```python
import cv2
import numpy as np

def solve(img, out_w, out_h) -> dict:
    """
    Finds the four corners of the bright card on the dark surface and warps the card 
    to an upright image of size out_w x out_h.
    
    Parameters:
        img (np.ndarray): uint8, shape HxWx3. BGR photo; one bright card on a dark background.
        out_w (int): width of the rectified image in px.
        out_h (int): height of the rectified image in px.
    
    Returns:
        dict with:
            - 'corners': float, shape 4x2. Card corners as (x, y) in the photo.
              Row order: TL, TR, BR, BL based on IMAGE position.
              - TL: smallest x+y
              - TR: smallest y-x
              - BR: largest x+y
              - BL: largest y-x
            - 'warped': uint8, shape out_hxout_wx3. BGR; the card rectified.
    """
    # Convert to HSV to segment the bright card
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of bright colors (adjust thresholds if necessary, but generic bright works)
    # Assuming the card is significantly brighter than the background
    lower_bright = np.array([0, 50, 50])
    upper_bright = np.array([180, 255, 255])
    
    # Threshold the HSV image to get bright components
    mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    eroded_mask = cv2.undilate(dilated_mask, kernel, iterations=2) # Note: cv2.erode is standard
    
    # Corrected erosion
```

### r2-4b / test-perspective_rectify-011 / greedy 0

- rule `cv2-attribute`; also nothing
- evidence: exec-only: exception on input 0: AttributeError module 'cv2' has no attribute 'undilate'

```python
import cv2
import numpy as np

def solve(img, out_w, out_h) -> dict:
    """
    Detects the four corners of a rectangular card in an image and returns
    the corners and a rectified (warped) image.
    
    Parameters:
        img: uint8, shape HxWx3. BGR photo.
        out_w: int, width of the rectified image.
        out_h: int, height of the rectified image.
    
    Returns:
        dict with:
            - 'corners': float, shape 4x2. Card corners (x, y) in original image coordinates.
              Order: TL, TR, BR, BL based on IMAGE position.
            - 'warped': uint8, shape out_hxout_wx3. Rectified BGR image.
    """
    # Convert to HSV to segment the bright card on dark background
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of bright colors (card is bright, background is dark)
    # Adjust lower and upper bounds based on typical bright objects on dark backgrounds
    lower_bright = np.array([0, 50, 50])
    upper_bright = np.array([180, 255, 255])
    
    # Threshold the HSV image to get bright components
    mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Apply morphological operations to clean up noise and fill gaps
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    eroded_mask = cv2.undilate(dilated_mask, kernel, iterations=2) # Note: cv2.erode is standard, 'undilate' is not a standard function name in OpenCV. Corrected below.
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming one card)
```

## lint (3 shown)

### r3-9b / call_imdecode_none / greedy 0

- rule `lint:broad-except`; also nothing
- evidence: line 32: [broad-except] `except Exception` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(buf: bytes) -> dict:
    """
    Decodes the encoded image bytes as a colour image.
    If decoding fails or the buffer is not a valid image, returns shape [0, 0, 0] and mean 0.0.
    
    Args:
        buf (bytes): Bytes of an encoded image file (PNG) or arbitrary bytes.
        
    Returns:
        dict: Contains 'shape' (list of 3 ints) and 'mean' (float).
    """
    try:
        # OpenCV expects BGR for color images when reading from bytes
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            # Decoding failed (invalid image)
            return {
                'shape': [0, 0, 0],
                'mean': 0.0
            }
        
        # Ensure the image is in the expected format (H, W, C)
        # OpenCV returns uint8, which is correct for mean calculation
        return {
            'shape': img.shape,
            'mean': float(np.mean(img))
        }
    except Exception:
        # Catch any unexpected errors during decoding
        return {
            'shape': [0, 0, 0],
            'mean': 0.0
        }
```

### r4-7b-notes / call_imdecode_none / greedy 0

- rule `lint:broad-except`; also nothing
- evidence: line 10: [broad-except] bare `except:` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(buf) -> dict:
    try:
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {'shape': [0, 0, 0], 'mean': 0.0}
        return {'shape': img.shape, 'mean': np.mean(img)}
    except:
        return {'shape': [0, 0, 0], 'mean': 0.0}
```

### r5-9b-notes / test-text_layout-015 / greedy 0

- rule `lint:broad-except`; also ['convention:ValueError']
- evidence: line 62: [broad-except] bare `except:` hides failures from the verifier

```python
import cv2
import numpy as np

def solve(img, box, text, face, scale, thickness, color) -> dict:
    """
    Draws text centered in a rectangle on an image, scaling down the font if necessary.
    
    Args:
        img: uint8, shape HxWx3 (BGR).
        box: tuple/list of 4 ints (x, y, w, h).
        text: str.
        face: int (cv2.FONT_HERSHEY_*).
        scale: float (requested font scale).
        thickness: int (stroke thickness).
        color: tuple/list of 3 ints (BGR).
        
    Returns:
        dict with keys:
            'img': The modified image.
            'scale': The actual scale used.
            'text_size': Tuple (width, height) of the text bounding box.
    """
    
    # Unpack box
    x, y, w, h = box
    
    # Ensure inputs are in the expected numpy types for OpenCV operations
    img = np.array(img, dtype=np.uint8)
    color = np.array(color, dtype=np.uint8)
    
    # We need to find the largest scale <= requested_scale such that the text fits.
    # OpenCV's getTextSize returns (width, height, baseline).
    # The width and height returned are the bounding box dimensions.
    # We must ensure:
    #   x + width <= x + w  => width <= w
    #   y + height <= y + h => height <= h
    
    # Start with the requested scale.
    current_scale = scale
    
```

## timeout (0 shown)

(none)

## convention (3 shown)

### r2-4b / shape_chessboard / greedy 0

- rule `stage:contract`; also nothing
- evidence: item 0: 'corners' should be float, got int32

```python
import cv2
import numpy as np

def solve(img, cols, rows) -> dict:
    """
    Finds the inner corners of a chessboard using OpenCV's corner detection.
    
    Parameters:
    - img: uint8, shape HxW, grayscale image of a flat chessboard.
    - cols: int, number of inner corners per row.
    - rows: int, number of inner corners per column.
    
    Returns:
    - dict with key 'corners': float, shape Nx2, containing (x, y) of every inner corner.
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur the image to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Detect corners using goodFeaturesToTrack
    # The 'maxCorners' parameter should be set to the total number of corners we expect
    # Total corners = cols * rows
    max_corners = cols * rows
    
    # goodFeaturesToTrack returns a list of corners, each as [x, y]
    # We use a high quality level and a large window size to ensure we get the inner corners
    # The 'qualityLevel' and 'maxCorners' are tuned for chessboard patterns
    corners = cv2.goodFeaturesToTrack(
        img_blur,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=33
    )
    
    # If no corners are found, return an empty list
    if corners is None:
        return {'corners': np.array([]).reshape(0, 2)}
```

### r3-9b / dtype_persp_points / greedy 0

- rule `cv2-assertion`; also nothing
- evidence: exception on input 0: error OpenCV(5.0.0) /io/opencv/modules/geometry/src/geometry.cpp:901: error: (-215:Assertion failed) src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4 in function 'getPerspectiveTransform'

```python
import cv2
import numpy as np

def solve(src, dst) -> dict:
    """
    Computes the 3x3 perspective transform matrix H that maps src corners to dst corners.
    
    Parameters:
    src (np.ndarray): 4x2 array of source corner coordinates (x, y).
    dst (np.ndarray): 4x2 array of destination corner coordinates (x, y).
    
    Returns:
    dict: {'H': np.ndarray} where H is the 3x3 transformation matrix.
    """
    # Ensure inputs are float64 for calculation
    src = np.array(src, dtype=np.float64)
    dst = np.array(dst, dtype=np.float64)
    
    # OpenCV's getPerspectiveTransform expects 4x2 arrays and returns a 3x3 matrix
    H = cv2.getPerspectiveTransform(src, dst)
    
    return {'H': H}
```

### r4-7b-notes / test-histogram_stats-003 / greedy 0

- rule `verifier-contract`; also nothing
- evidence: manufactured: scene 4005120: contract: 'hist' should be int, got float32; 'median' should be an int, got 65.0

```python
import cv2
import numpy as np

def solve(img) -> dict:
    # Convert BGR to BT.601 luma
    luma = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    
    # Round to nearest integer
    gray = np.round(luma).astype(np.uint8)
    
    # Calculate the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    
    # Calculate mean, standard deviation, and median
    mean = np.mean(gray)
    std = np.std(gray, ddof=0)  # Population standard deviation
    median = np.median(gray)
    
    return {
        'hist': hist,
        'mean': mean,
        'std': std,
        'median': median
    }
```

## algorithm (3 shown)

### r1-7b / test-text_layout-014 / greedy 0

- rule `verifier-value`; also nothing
- evidence: manufactured: scene 4008561: 686 changed pixels outside the box (26, 45, 514, 55); ink bbox x 16..353, y 42..80

```python
import cv2
import numpy as np

def solve(img, box, text, face, scale, thickness, color) -> dict:
    x, y, w, h = box
    (text_width, text_height), baseline = cv2.getTextSize(text, face, scale, thickness)
    
    # Calculate the center of the box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate the top-left corner of the text box
    text_x = center_x - text_width // 2
    text_y = center_y - text_height // 2
    
    # Ensure the text fits within the box
    while text_width > w or text_height > h:
        scale *= 0.9
        (text_width, text_height), baseline = cv2.getTextSize(text, face, scale, thickness)
    
    # Draw the text on the image
    cv2.putText(img, text, (text_x, text_y + text_height), face, scale, color, thickness)
    
    return {
        'img': img,
        'scale': scale,
        'text_size': (text_width, text_height)
    }
```

### r3-9b / test-hough_segments-021 / greedy 0

- rule `verifier-value`; also nothing
- evidence: manufactured: scene 4001843: 4 rows for 5 strokes (strokes were missed)

```python
import cv2
import numpy as np

def solve(img: np.ndarray) -> dict:
    """
    Detects straight bright strokes on a dark background and returns their endpoints.
    
    Args:
        img: uint8, shape HxWx3. BGR; bright straight strokes 1 to 4 px wide, 
             each at least 40 px long, on a dark uniform background.
    
    Returns:
        dict with key 'segments': a list of [x1, y1, x2, y2] for each stroke.
        Sorting: by x1 ascending, then y1 ascending.
        Ordering of points: (x1, y1) has smaller x (or smaller y if x equal).
    """
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate bright strokes against dark background
    # Assuming background is dark and strokes are bright. 
    # A simple threshold should work. Let's use Otsu's thresholding after Gaussian filtering.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up noise and ensure strokes are solid
    kernel = np.ones((3,3), np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)
    eroded_thresh = cv2.erode(dilated_thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segments = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 5.0  # Adjust based on stroke width (1-4px), 5 should be safe for 40px+ length
        approx = cv2.approxPolyDP(contour, epsilon, True)
```

### r5-9b-notes / test-hough_segments-021 / greedy 0

- rule `verifier-value`; also nothing
- evidence: manufactured: scene 4001841: 2 rows for 3 strokes (strokes were missed)

```python
import cv2
import numpy as np

def solve(img: np.ndarray) -> dict:
    """
    Detects straight bright strokes on a dark background and returns their endpoints.
    
    Args:
        img: uint8, shape HxWx3 (BGR). Bright strokes 1-4px wide, >=40px long.
             Dark uniform background. Strokes don't touch/cross, >=20px apart, clear of border.
    
    Returns:
        dict with key 'segments': list of [x1, y1, x2, y2] for each stroke.
        (x1, y1) has smaller x (or smaller y if x equal). Sorted by x1, then y1.
    """
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate bright strokes. 
    # Since background is dark and strokes are bright, a simple threshold works.
    # We use Otsu's thresholding after Gaussian blurring to handle noise if any,
    # but given the description, a fixed high threshold relative to max might be safer.
    # However, Otsu is robust for bimodal histograms (dark bg, bright strokes).
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to connect broken parts of strokes and remove noise
    # Strokes are straight and at least 40px long.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Dilate to connect gaps within a stroke
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    # Erode to restore approximate original width (though width is ambiguous)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segments = []
    
```

## other (3 shown)

### r1-7b / test-chessboard_corners-013 / greedy 0

- rule `exception:ValueError`; also nothing
- evidence: exec-only: exception on input 0: ValueError Chessboard not found in the image

```python
import cv2
import numpy as np

def solve(img, squares_x, squares_y) -> dict:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if not ret:
        raise ValueError("Chessboard not found in the image")
    
    # Refine the corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Convert the corners to the required format
    corners = corners.reshape(-1, 2)
    
    # Return the result as a dictionary
    return {'corners': corners}
```

### r1-7b / test-chessboard_corners-019 / greedy 0

- rule `exception:ValueError`; also nothing
- evidence: exec-only: exception on input 0: ValueError Chessboard not found in the image

```python
import cv2
import numpy as np

def solve(img, squares_x, squares_y) -> dict:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
    
    if not ret:
        raise ValueError("Chessboard not found in the image")
    
    # Refine the corner locations to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Convert the corners to the required format
    corners = corners.reshape(-1, 2).astype(np.float)
    
    return {'corners': corners}
```

### r5-9b-notes / test-chessboard_corners-018 / greedy 0

- rule `exception:ValueError`; also nothing
- evidence: exec-only: exception on input 0: ValueError Chessboard corners not found.

```python
import cv2
import numpy as np

def solve(img, squares_x, squares_y) -> dict:
    """
    Detects chessboard corners with sub-pixel accuracy and returns them in row-major order.
    
    Args:
        img (np.ndarray): uint8, shape (H, W, 3), BGR image.
        squares_x (int): Number of squares along the board's top edge (width).
        squares_y (int): Number of squares along the board's left edge (height).
        
    Returns:
        dict:
            'corners' (np.ndarray): float32, shape (N, 2), where N = (squares_x - 1) * (squares_y - 1).
            Coordinates are in row-major order starting from the board's top-left corner.
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    # Note: findChessboardCorners returns corners in row-major order starting from the top-left.
    # However, the documentation states that if the top-left square is white, the order is reversed.
    # The problem statement says "its top-left square may be black or white".
    # We need to handle the reversal if the top-left square is white.
    
    ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
    
    if not ret:
        raise ValueError("Chessboard corners not found.")
    
    # Refine corners to sub-pixel accuracy
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
    
    # Check the color of the top-left square to determine if the order needs to be reversed.
    # The corners array contains the centers of the squares.
    # The top-left corner in the image corresponds to the first corner in the array if the top-left square is black.
    # If the top-left square is white, the order is reversed.
    
```
