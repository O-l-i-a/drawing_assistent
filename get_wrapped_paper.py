from dataclasses import dataclass
from itertools import combinations

from sklearn.cluster import KMeans

import cv2 as cv
import numpy as np


@dataclass
class PaperDetectionConfig:
    min_contour_area_ratio: float = 0.10
    approx_epsilon_ratio: float = 0.02
    approx_epsilon_candidates: tuple[float, ...] = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
    min_rectangularity: float = 0.70
    threshold_bias: int = 10
    max_saturation: int = 60
    min_brightness: int = 150
    border_margin: int = 5
    illumination_blur_size: int = 51
    normalized_min_brightness: int = 160
    grabcut_max_dimension: int = 1200
    grabcut_border_ratio: float = 0.08
    grabcut_iterations: int = 3
    hough_angle_tolerance_degrees: float = 18.0
    hough_distance_weight: float = 4.0
    hough_angle_weight: float = 20.0
    debug: bool = False


@dataclass
class PaperDetectionResult:
    overlay: np.ndarray
    collage: np.ndarray | None
    warped: np.ndarray | None
    corners: np.ndarray | None

@dataclass
class PaperState:
    last_corners: np.ndarray | None = None
    last_best_lines: list | None = None
    validation_fail_count: int = 0
    validation_count: int = 0
    MAX_FAILS: int = 10
    MIN_VALID: int = 50 

def x_intersect(rho, theta):
    # Schnittpunkt mit y = 0
    return rho / np.cos(theta)

def y_intersect(rho, theta):
    # Schnittpunkt mit x = 0
    return rho / np.sin(theta)

def sort_corners(corners):
    center = np.mean(corners, axis = 0)
    angles = np.arctan2(corners[:,1]- center[1], corners[:,0] - center[0])
    idx = np.argsort(angles)
    c_sorted = corners[idx]
    tl_index = np.argmin(c_sorted[:,0] + c_sorted[:,1])
    c_sorted = np.roll(c_sorted, -tl_index, axis=0)
    return c_sorted


def detect_corners(frame, last_corners):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 40, 120)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv.contourArea)

    peri = cv.arcLength(largest, True)
    approx = cv.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        corners = sort_corners(corners)
        return corners, None
    
    hull = cv.convexHull(largest)
    if len(hull) >= 4:
        rect = cv.minAreaRect(hull)
        box = cv.boxPoints(rect).astype(np.float32)
        corners = sort_corners(box)
        return corners, None
    
    return last_corners, None


def validate_geometry(
    new_corners: np.ndarray | None,
    old_corners: np.ndarray | None,
    max_ratio_change: float = 0.25,
    max_angle_change: float = 20.0,
    alpha: float = 0.25,
) -> tuple[bool, np.ndarray | None]:
    
    if new_corners is None:
        return False,old_corners
    new_corners = sort_corners(new_corners)
    if old_corners is None:
        return True,new_corners
    
    #edge length > 0
    lengths = [
        np.linalg.norm(new_corners[1] - new_corners[0]),
        np.linalg.norm(new_corners[2] - new_corners[1]),
        np.linalg.norm(new_corners[3] - new_corners[2]),
        np.linalg.norm(new_corners[0] - new_corners[3]),
    ]

    if any(L < 5 for L in lengths):
        return False, old_corners
    
    #area > 0
    def polygon_area(c):
        x = c[:, 0]
        y = c[:, 1]
        return 0.5 * abs(
            x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]
            - y[0]*x[1] - y[1]*x[2] - y[2]*x[3] - y[3]*x[0]
        )
    
    if polygon_area(new_corners) < 50:
        return False, old_corners

    #convexity
    def is_convex(c):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        signs = []
        for i in range(4):
            o = c[i]
            a = c[(i+1) % 4]
            b = c[(i+2) % 4]
            signs.append(cross(o, a, b))

        return all(s > 0 for s in signs) or all(s < 0 for s in signs)
    if not is_convex(new_corners):
        return False, old_corners
    
    #no overlapping edges
    def segments_intersect(a, b, c, d):
        def ccw(p, q, r):
            return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    if segments_intersect(new_corners[0], new_corners[1], new_corners[2], new_corners[3]):
        return False, old_corners
    if segments_intersect(new_corners[1], new_corners[2], new_corners[3], new_corners[0]):
        return False, old_corners

    #angles
    def angle(a, b, c):
        ab = a - b
        cb = c - b
        cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    angles = [
        angle(new_corners[3], new_corners[0], new_corners[1]),
        angle(new_corners[0], new_corners[1], new_corners[2]),
        angle(new_corners[1], new_corners[2], new_corners[3]),
        angle(new_corners[2], new_corners[3], new_corners[0]),
    ]

    if any(a < 60 or a > 120 for a in angles):
        return False, old_corners
    
    #DIN-A4 check
    w1 = np.linalg.norm(new_corners[1] - new_corners[0])
    w2 = np.linalg.norm(new_corners[2] - new_corners[3])
    h1 = np.linalg.norm(new_corners[3] - new_corners[0])
    h2 = np.linalg.norm(new_corners[2] - new_corners[1])

    width = (w1 + w2) / 2
    height = (h1 + h2) / 2

    if width == 0 or height == 0:
        return False, old_corners

    ratio = max(width, height) / min(width, height)

    if not (1.2 < ratio < 1.6):
        return False, old_corners
   
    #stability
    score = 0
    
    old_w = np.linalg.norm(old_corners[1]-old_corners[0])
    new_w = np.linalg.norm(new_corners[1]-new_corners[0])
    if abs(new_w - old_w) < old_w * max_ratio_change:
        score += 1
    
    old_w = np.linalg.norm(old_corners[2]-old_corners[3])
    new_w = np.linalg.norm(new_corners[2]-new_corners[3])
    if abs(new_w - old_w) < old_w * max_ratio_change:
        score += 1
    
    old_h = np.linalg.norm(old_corners[3]-old_corners[0])
    new_h = np.linalg.norm(new_corners[3]-new_corners[0])
    if abs(new_h - old_h) < old_h * max_ratio_change:
        score += 1
    
    old_h = np.linalg.norm(old_corners[2]-old_corners[1])
    new_h = np.linalg.norm(new_corners[2]-new_corners[1])
    if abs(new_h - old_h) < old_h * max_ratio_change:
        score += 1

    for i in range(4):
        old_angle = np.degrees(np.arctan2(old_corners[-3+i][0]-old_corners[i][0],old_corners[-3+i][1] - old_corners[i][1]))
        new_angle = np.degrees(np.arctan2(new_corners[-3+i][0]-new_corners[i][0],new_corners[-3+i][1] - new_corners[i][1]))
        if abs(new_angle - old_angle) < max_angle_change:
            score += 1

    if score >= 7:
        return True, new_corners
    elif score >= 4:
        return True, alpha * new_corners + (1-alpha) * old_corners
    else:
        return False, old_corners

    
    
def get_wrapped_paper(
    frame: np.ndarray,
    state: PaperState | None = None,
    config: PaperDetectionConfig | None = None,
) -> PaperDetectionResult:
    """Detect a paper sheet and return its warped top-down view."""
    if config is None:
        config = PaperDetectionConfig()
    if state is None:
        state = PaperState()

    h, w = frame.shape[:2]
    crop_height = 80
    frame = frame[crop_height:h,0:w]

    overlay = frame.copy()

    
    #corners_raw, updated_lines = detect_corners_hough_kmeans(frame, state.last_best_lines)
    corners_raw, _ = detect_corners(frame, state.last_corners)
    if corners_raw is None: 
        return PaperDetectionResult(overlay, None, None, state.last_corners)
    is_valid, corners = validate_geometry(corners_raw,state.last_corners)
    """ if is_valid and updated_lines is not None:
        state.last_best_lines = updated_lines """
    warped = None
    if corners is not None:
        if is_valid and state.validation_count < state.MIN_VALID:
            state.validation_fail_count = 0
            state.validation_count += 1

        elif state.validation_count < state.MIN_VALID:
            state.validation_fail_count += 1
            state.validation_count = 0
            if state.validation_fail_count > state.MAX_FAILS:
                corners = corners_raw
        draw_paper_outline(overlay, corners)
        warped = warp_paper(frame, corners)
        # crop frame to eliminate potential background
        h,w = warped.shape[:2]
        crop_w = int(w*0.05)
        crop_h = int(h* 0.05)
        warped = warped[crop_h : h - crop_h, crop_w : w - crop_w]
    else:
        warped = None
    
    collage = None
    if config.debug:
        collage = build_stage_collage(
            original=frame,
            gray=to_grayscale(frame),
            blurred=reduce_noise(to_grayscale(frame)),
            paper_mask=np.zeros(frame.shape[:2], dtype=np.uint8),
            edges=np.zeros(frame.shape[:2], dtype=np.uint8),
            contour_view=overlay,
            overlay=overlay,
            warped=warped,
        )

    return PaperDetectionResult(
        overlay=overlay,
        collage=collage,
        warped=warped,
        corners=corners,
    )


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def reduce_noise(gray: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(gray, (5, 5), 0)


def draw_contour_candidates(
    frame: np.ndarray,
    contours: list[np.ndarray],
    selected: np.ndarray | None,
) -> np.ndarray:
    contour_view = frame.copy()
    for contour in contours[:10]:
        cv.drawContours(contour_view, [contour], -1, (120, 120, 120), 1)

    if selected is not None:
        selected_int = selected.astype(np.int32)
        highlighted = contour_view.copy()
        cv.fillPoly(highlighted, [selected_int], (0, 255, 0))
        contour_view = cv.addWeighted(highlighted, 0.22, contour_view, 0.78, 0)
        cv.drawContours(contour_view, [selected_int], -1, (0, 255, 255), 6)

    return contour_view


def draw_paper_outline(overlay: np.ndarray, corners: np.ndarray) -> None:
    polygon = corners.astype(np.int32).reshape((-1, 1, 2))
    highlighted = overlay.copy()
    cv.fillPoly(highlighted, [polygon], (0, 255, 0))
    cv.addWeighted(highlighted, 0.18, overlay, 0.82, 0, dst=overlay)
    cv.polylines(overlay, [polygon], True, (0, 255, 255), 8, cv.LINE_AA)
    cv.polylines(overlay, [polygon], True, (0, 0, 0), 2, cv.LINE_AA)

    for index, point in enumerate(corners.astype(np.int32)):
        cv.circle(overlay, tuple(point), 10, (255, 255, 255), -1)
        cv.circle(overlay, tuple(point), 7, (0, 0, 255), -1)
        cv.putText(
            overlay,
            str(index + 1),
            tuple(point + np.array([10, -10])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            str(index + 1),
            tuple(point + np.array([10, -10])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )


def warp_paper(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = corners

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    max_width = max(int(width_top), int(width_bottom), 1)
    max_height = max(int(height_left), int(height_right), 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv.getPerspectiveTransform(corners, destination)
    return cv.warpPerspective(frame, matrix, (max_width, max_height))


def annotate_stage(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    tile = cv.resize(image, size, interpolation=cv.INTER_AREA)
    cv.rectangle(tile, (0, 0), (size[0], 30), (0, 0, 0), -1)
    cv.putText(
        tile,
        label,
        (10, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return tile


def build_stage_collage(
    original: np.ndarray,
    gray: np.ndarray,
    blurred: np.ndarray,
    paper_mask: np.ndarray,
    edges: np.ndarray,
    contour_view: np.ndarray,
    overlay: np.ndarray,
    warped: np.ndarray | None,
) -> np.ndarray:
    tile_size = (420, 280)
    fallback = np.zeros_like(original)
    tiles = [
        annotate_stage(original, "Original", tile_size),
        annotate_stage(gray, "Grayscale", tile_size),
        annotate_stage(blurred, "Blurred", tile_size),
        annotate_stage(paper_mask, "Paper Mask", tile_size),
        annotate_stage(edges, "Edges", tile_size),
        annotate_stage(contour_view, "Contour Candidate", tile_size),
        annotate_stage(overlay, "Detected Paper", tile_size),
        annotate_stage(warped if warped is not None else fallback, "Warped Paper", tile_size),
    ]

    rows = []
    for index in range(0, len(tiles), 2):
        rows.append(np.hstack((tiles[index], tiles[index + 1])))

    return np.vstack(rows)


def jpeg_to_paper_detection_result(source: str | np.ndarray) -> PaperDetectionResult:
    """Load a JPEG file or accept an image array and return a PaperDetectionResult.

    This is for cases where the image is already a cropped, top-down photo of the paper.
    The returned `warped` image will be the loaded image and `corners` will be
    the full-image rectangle corners so downstream code (e.g. `preprocess_drawing`)
    can use `result.warped` directly.

    Args:
        source: path to an image file or a BGR image numpy array.

    Returns:
        PaperDetectionResult with `overlay`, `collage`=None, `warped`, and `corners` set.
    """
    if isinstance(source, str):
        image = cv.imread(source, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {source}")
    else:
        image = source.copy()

    h, w = image.shape[:2]
    overlay = image.copy()
    warped = image.copy()
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    return PaperDetectionResult(overlay=overlay, collage=None, warped=warped, corners=corners)
