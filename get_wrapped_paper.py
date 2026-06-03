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

def x_intersect(rho, theta):
    # Schnittpunkt mit y = 0
    return rho / np.cos(theta)

def y_intersect(rho, theta):
    # Schnittpunkt mit x = 0
    return rho / np.sin(theta)

def detect_corners_hough_kmeans(frame, last_best_lines=None):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    overlay = frame.copy()

    blur = cv.GaussianBlur(img, (5,5), 0)
    edges = cv.Canny(blur, 30, 90)
    edges = cv.dilate(edges, None, iterations=1)

    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    if lines is None:
        return None, last_best_lines
    
    horizontal = []
    vertical = []
    vertical_rho = []
    horizontal_rho = [] 

    for line in lines:
        rho, theta = line[0]
        abs_rho = abs(rho)
        if(((theta <= np.pi*0.25 and theta >=0)  or (theta > np.pi*1.75 )) or (theta > np.pi*0.75 and theta <= np.pi*1.25)):
            vertical.append(line)
            vertical_rho.append(abs_rho)
        else:
            horizontal.append(line)
            horizontal_rho.append(abs_rho)

    if len(vertical) < 2 or len(horizontal) < 2:
        return None, last_best_lines
        
    vertical_rho = np.array(vertical_rho, dtype=np.float32).reshape(-1, 1)
    horizontal_rho = np.array(horizontal_rho, dtype=np.float32).reshape(-1, 1)

    try:
        vert_labels = KMeans(n_clusters=2, n_init=1).fit_predict(vertical_rho)
        hor_labels = KMeans(n_clusters=2, n_init=1).fit_predict(horizontal_rho)
    except:
        return None, last_best_lines
    
    categorized = [[], [], [], []]

    for i,l in enumerate(vert_labels):
        if(l == 0):
            categorized[0].append(vertical[i])
        else:
            categorized[1].append(vertical[i])

    for i,l in enumerate(hor_labels):
        if(l == 0):
            categorized[2].append(horizontal[i])
        else:
            categorized[3].append(horizontal[i])
    
    best_lines = [None, None, None, None] #[leftVert, rightVert, topHorz, bottomHorz]

    for i, group in enumerate(categorized):
        if len(group) == 0:
            continue

        mean_rho = np.mean([g[0][0] for g in group])
        best_line = min(group, key=lambda g: abs(g[0][0] - mean_rho))
        best_lines[i] = best_line

        rho_b = best_line[0][0]
        theta_b = best_line[0][1]

        if(i < 2):
            #vertical
            xi = x_intersect(rho_b, theta_b)

            if best_lines[0] is None:
                best_lines[0] = best_line
            else:
                # Vergleiche x-Schnittpunkte
                rho0 = best_lines[0][0][0]
                theta0 = best_lines[0][0][1]
                xi0 = x_intersect(rho0, theta0)

                if xi < xi0:
                    best_lines[1] = best_lines[0]
                    best_lines[0] = best_line
                else:
                    best_lines[1] = best_line
        else:
            #horizontal
            yi = y_intersect(rho_b, theta_b)

            if best_lines[2] is None:
                best_lines[2] = best_line
            else:
                rho2 = best_lines[2][0][0]
                theta2 = best_lines[2][0][1]
                yi0 = y_intersect(rho2, theta2)

                if yi < yi0:
                    best_lines[3] = best_lines[2]
                    best_lines[2] = best_line
                else:
                    best_lines[3] = best_line


    # fallback to last frame
    if last_best_lines is not None:
        for i in range(4):
            if best_lines[i] is None and last_best_lines[i] is not None:
                best_lines[i] = last_best_lines[i]

    # compute intersections
    def intersect(l1, l2):
        rho1, th1 = l1
        rho2, th2 = l2
        A = np.array([
            [np.cos(th1), np.sin(th1)],
            [np.cos(th2), np.sin(th2)]
        ])
        b = np.array([rho1, rho2])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, b)
    
    L = [(l[0][0], l[0][1]) for l in best_lines]

    left, right, top, bottom = L

    tl = intersect(left, top)
    tr = intersect(right, top)
    br = intersect(right, bottom)
    bl = intersect(left, bottom)

    if any(c is None for c in (tl, tr, br, bl)):
        return None, best_lines 

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return corners, best_lines



def get_wrapped_paper(
    frame: np.ndarray,
    last_corners: np.ndarray | None = None,
    config: PaperDetectionConfig | None = None,
) -> PaperDetectionResult:
    """Detect a paper sheet and return its warped top-down view."""
    if config is None:
        config = PaperDetectionConfig()

    h, w = frame.shape[:2]
    crop_height = 80
    frame = frame[crop_height:h,0:w]

    overlay = frame.copy()

    corners, updated_lines = detect_corners_hough_kmeans(frame, last_corners)

    warped = None
    if corners is not None:
        draw_paper_outline(overlay, corners)
        warped = warp_paper(frame, corners)
    
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
