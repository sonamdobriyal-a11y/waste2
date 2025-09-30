import cv2
import numpy as np


def detect_utensil_ellipse(frame, utensil_hint='auto', min_radius=60, max_radius=400, debug=False):
    """
    Detect a circular/elliptical utensil region.
    Returns ellipse as ((cx,cy), (MA,ma), angle) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)

    # Try HoughCircles first (aimed at plates/bowls)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
                               param1=120, param2=40, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None and len(circles)>0:
        circles = np.uint16(np.around(circles))
        # choose the circle closest to center, largest radius as tiebreaker
        h, w = gray.shape[:2]
        cx, cy = w/2, h/2
        def score(c):
            x,y,r = c
            return np.hypot(x-cx, y-cy) - 0.1*r
        best = sorted(circles[0], key=score)[0]
        x,y,r = best
        ellipse = ((float(x), float(y)), (float(2*r), float(2*r)), 0.0)
        return ellipse

    # Fallback: contour with high circularity, fit ellipse
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ell = None
    best_score = -1
    for c in contours:
        if len(c) < 20:
            continue
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4*np.pi*area/(peri*peri)
        if circularity < 0.5:
            continue
        try:
            ell = cv2.fitEllipse(c)
        except cv2.error:
            continue
        # prefer larger, more circular shapes
        score = area * circularity
        if score > best_score:
            best_score = score
            best_ell = ell

    if debug and best_ell is None:
        print('Utensil detection: no suitable ellipse found')
    return best_ell


def _ellipse_mask(shape, ellipse):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
    return mask


def _sample_base_color_lab(img, ellipse, ring_frac=0.12):
    # Sample a ring near the inner rim to estimate utensil interior color
    (cx, cy), (MA, ma), angle = ellipse
    inner = ((cx, cy), (MA*(1-0.02), ma*(1-0.02)), angle)
    outer = ((cx, cy), (MA*(1-0.02), ma*(1-0.02)), angle)

    # Create masks for full interior and a ring near edge
    full_mask = _ellipse_mask(img.shape, inner)

    # Make a thinner inner ellipse to form a ring
    inner2 = ((cx, cy), (MA*(1-2*ring_frac), ma*(1-2*ring_frac)), angle)
    inner_mask = _ellipse_mask(img.shape, inner2)

    ring = cv2.subtract(full_mask, inner_mask)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ring_pixels = lab[ring > 0]
    if len(ring_pixels) < 100:
        ring_pixels = lab[full_mask > 0]
    med = np.median(ring_pixels, axis=0)
    return med.astype(np.float32), full_mask


def segment_food_in_utensil(img, ellipse, debug=False):
    """
    Returns binary mask of food region inside the utensil ellipse.
    Uses Lab color delta from base and refines with GrabCut.
    """
    base_lab, full_mask = _sample_base_color_lab(img, ellipse)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute simple Euclidean distance in Lab space
    delta = np.linalg.norm(lab - base_lab, axis=2)

    # Only consider interior
    delta_inside = np.zeros_like(delta, dtype=np.float32)
    delta_inside[full_mask > 0] = delta[full_mask > 0]

    # Adaptive threshold: Otsu on interior deltas
    vals = delta_inside[full_mask > 0].astype(np.uint8)
    if vals.size < 100:
        return None, {}
    # Otsu returns (ret, thresh). We need the scalar ret.
    ret, _th_img = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    init_food = np.zeros_like(delta_inside, dtype=np.uint8)
    init_food[(delta_inside >= ret) & (full_mask > 0)] = 1

    # Prepare GrabCut mask: 0=bg,1=fg,2=prob bg,3=prob fg
    gc_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    gc_mask[full_mask == 0] = 0  # definite background
    gc_mask[(full_mask > 0) & (init_food == 0)] = 2  # probable background (plate)
    gc_mask[(full_mask > 0) & (init_food == 1)] = 3  # probable foreground (food)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut constrained to the ellipse ROI
    rect = cv2.boundingRect(full_mask)
    try:
        cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Fall back to threshold-only
        seg = np.zeros_like(gc_mask, dtype=np.uint8)
        seg[(full_mask > 0) & (init_food == 1)] = 255
        return seg, {"delta": delta_inside, "init": init_food}

    seg = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)

    # Keep only inside utensil
    seg[full_mask == 0] = 0

    return seg, {"delta": delta_inside, "init": init_food}


def estimate_area_and_volume(ellipse, seg_mask, diameter_mm, utensil_hint, assumed_height_mm):
    """
    Compute percent fill by area and an approximate volume for plates (ml).
    - percent_fill: ratio of food pixels to utensil interior pixels
    - est_volume_ml: if plate and diameter_mm provided, compute area in mm^2 times assumed height
    """
    (cx, cy), (MA, ma), angle = ellipse
    full_mask = _ellipse_mask(seg_mask.shape + (1,), ellipse)
    interior_px = int(np.count_nonzero(full_mask))
    food_px = int(np.count_nonzero(seg_mask))
    if interior_px == 0:
        return None, None
    percent_fill = 100.0 * food_px / interior_px

    est_volume_ml = None
    # If a diameter and a height are provided, compute a heuristic volume
    # for any utensil type using area * height (mm^3 -> ml).
    if diameter_mm is not None and assumed_height_mm is not None and assumed_height_mm > 0:
        major_axis_px = max(MA, ma)
        if major_axis_px > 0:
            mm_per_px = diameter_mm / major_axis_px
            food_area_mm2 = food_px * (mm_per_px ** 2)
            volume_mm3 = food_area_mm2 * assumed_height_mm
            est_volume_ml = volume_mm3 / 1000.0
    return percent_fill, est_volume_ml
