from dataclasses import dataclass
from itertools import combinations

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


def get_wrapped_paper(
    frame: np.ndarray,
    last_corners: np.ndarray | None = None,
    config: PaperDetectionConfig | None = None,
) -> PaperDetectionResult:
    """Detect a paper sheet and return its warped top-down view."""
    if config is None:
        config = PaperDetectionConfig()

    gray = to_grayscale(frame)
    blurred = reduce_noise(gray)
    paper_mask = build_paper_mask(frame, blurred, config)
    edges = detect_edges(blurred)
    contours = find_external_contours(paper_mask)
    candidate = select_paper_contour(contours, frame.shape, config)

    if candidate is None:
        paper_mask = build_grabcut_paper_mask(frame, config)
        contours = find_external_contours(paper_mask)
        candidate = select_paper_contour(contours, frame.shape, config)

    corners = None
    if candidate is not None:
        corners = order_corners(candidate.reshape(4, 2))
        corners = refine_corners_with_edges(corners, edges, frame.shape, config)
    elif last_corners is not None:
        corners = last_corners

    overlay = frame.copy()
    contour_view = draw_contour_candidates(frame, contours, candidate)
    warped = None

    if corners is not None:
        draw_paper_outline(overlay, corners)
        warped = warp_paper(frame, corners)

    collage = None
    if config.debug:
        collage = build_stage_collage(
            original=frame,
            gray=gray,
            blurred=blurred,
            paper_mask=paper_mask,
            edges=edges,
            contour_view=contour_view,
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


def detect_edges(blurred: np.ndarray) -> np.ndarray:
    edges = cv.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)


def build_paper_mask(
    frame: np.ndarray,
    blurred: np.ndarray,
    config: PaperDetectionConfig,
) -> np.ndarray:
    otsu_threshold, _ = cv.threshold(
        blurred,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU,
    )
    threshold_value = max(0, int(otsu_threshold) - config.threshold_bias)
    _, brightness_mask = cv.threshold(
        blurred,
        max(config.min_brightness, threshold_value),
        255,
        cv.THRESH_BINARY,
    )

    blur_size = max(3, config.illumination_blur_size | 1)
    illumination = cv.GaussianBlur(blurred, (blur_size, blur_size), 0)
    normalized = cv.divide(blurred, illumination, scale=255)
    normalized = cv.GaussianBlur(normalized, (5, 5), 0)
    _, normalized_mask = cv.threshold(
        normalized,
        config.normalized_min_brightness,
        255,
        cv.THRESH_BINARY,
    )

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, saturation_mask = cv.threshold(
        saturation,
        config.max_saturation,
        255,
        cv.THRESH_BINARY_INV,
    )

    brightness_union = cv.bitwise_or(brightness_mask, normalized_mask)
    mask = cv.bitwise_and(brightness_union, saturation_mask)

    kernel_close = np.ones((9, 9), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open, iterations=1)
    return mask


def build_grabcut_paper_mask(
    frame: np.ndarray,
    config: PaperDetectionConfig,
) -> np.ndarray:
    """Fallback segmentation for low-contrast paper/background scenes."""
    height, width = frame.shape[:2]
    scale = min(1.0, config.grabcut_max_dimension / max(height, width))

    if scale < 1.0:
        resized = cv.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv.INTER_AREA,
        )
    else:
        resized = frame.copy()

    small_height, small_width = resized.shape[:2]
    border_x = max(10, int(small_width * config.grabcut_border_ratio))
    border_y = max(10, int(small_height * config.grabcut_border_ratio))
    rect = (
        border_x,
        border_y,
        max(1, small_width - (2 * border_x)),
        max(1, small_height - (2 * border_y)),
    )

    mask = np.full((small_height, small_width), cv.GC_PR_BGD, dtype=np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv.grabCut(
        resized,
        mask,
        rect,
        background_model,
        foreground_model,
        config.grabcut_iterations,
        cv.GC_INIT_WITH_RECT,
    )

    binary_mask = np.where(
        (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel_close = np.ones((9, 9), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel_close, iterations=2)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel_open, iterations=1)

    if scale < 1.0:
        binary_mask = cv.resize(binary_mask, (width, height), interpolation=cv.INTER_NEAREST)

    return binary_mask


def find_external_contours(edges: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv.contourArea, reverse=True)


def select_paper_contour(
    contours: list[np.ndarray],
    frame_shape: tuple[int, ...],
    config: PaperDetectionConfig,
) -> np.ndarray | None:
    frame_area = frame_shape[0] * frame_shape[1]
    min_area = frame_area * config.min_contour_area_ratio

    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area:
            continue

        hull = cv.convexHull(contour)
        candidate = build_quadrilateral_candidate(hull, config)

        if len(candidate) != 4:
            continue

        candidate_x, candidate_y, candidate_width, candidate_height = cv.boundingRect(
            candidate.astype(np.int32)
        )
        touched_borders = sum(
            (
                candidate_x <= config.border_margin,
                candidate_y <= config.border_margin,
                candidate_x + candidate_width >= frame_shape[1] - config.border_margin,
                candidate_y + candidate_height >= frame_shape[0] - config.border_margin,
            )
        )
        if touched_borders > 1:
            continue

        rect_area = cv.contourArea(candidate.astype(np.float32))
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        if rectangularity < config.min_rectangularity:
            continue

        return candidate

    return None


def build_quadrilateral_candidate(
    hull: np.ndarray,
    config: PaperDetectionConfig,
) -> np.ndarray:
    perimeter = cv.arcLength(hull, True)
    hull_area = cv.contourArea(hull)
    best_candidate = None
    best_score = -np.inf

    for epsilon_ratio in config.approx_epsilon_candidates:
        approx = cv.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        candidate = None

        if len(approx) == 4 and cv.isContourConvex(approx):
            candidate = approx
        elif 4 < len(approx) <= 6:
            candidate = simplify_polygon_to_quad(approx, hull_area)

        if candidate is None:
            continue

        rect_area = cv.contourArea(candidate.astype(np.float32))
        if rect_area <= 0:
            continue

        rectangularity = hull_area / rect_area
        score = rectangularity - (epsilon_ratio * 0.1)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate is not None:
        return best_candidate

    rect = cv.minAreaRect(hull)
    return cv.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2)


def simplify_polygon_to_quad(
    polygon: np.ndarray,
    reference_area: float,
) -> np.ndarray | None:
    points = polygon.reshape(-1, 2).astype(np.float32)
    best_candidate = None
    best_score = -np.inf

    indices = range(len(points))
    for subset in combinations(indices, 4):
        candidate = points[list(subset)]
        candidate = order_corners(candidate).reshape(-1, 1, 2)
        if not cv.isContourConvex(candidate.astype(np.int32)):
            continue

        candidate_area = cv.contourArea(candidate.astype(np.float32))
        if candidate_area <= 0:
            continue

        top_angle = segment_angle_degrees(candidate[0, 0], candidate[1, 0])
        right_angle = segment_angle_degrees(candidate[1, 0], candidate[2, 0])
        bottom_angle = segment_angle_degrees(candidate[3, 0], candidate[2, 0])
        left_angle = segment_angle_degrees(candidate[0, 0], candidate[3, 0])
        parallel_penalty = angular_distance_degrees(top_angle, bottom_angle)
        parallel_penalty += angular_distance_degrees(right_angle, left_angle)

        area_score = candidate_area / reference_area
        edge_fit_penalty = 0.0
        ordered_candidate = candidate.reshape(4, 2)
        for point in points:
            edge_fit_penalty += min(
                point_to_line_distance(
                    point,
                    ordered_candidate[index],
                    ordered_candidate[(index + 1) % 4],
                )
                for index in range(4)
            )

        score = area_score - (parallel_penalty * 0.01) - (edge_fit_penalty * 0.0002)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate


def refine_corners_with_edges(
    corners: np.ndarray,
    edges: np.ndarray,
    frame_shape: tuple[int, ...],
    config: PaperDetectionConfig,
) -> np.ndarray:
    if not should_refine_quadrilateral(corners):
        return corners

    diagonal = float(np.hypot(frame_shape[0], frame_shape[1]))
    min_line_length = max(80, int(diagonal * 0.12))
    max_line_gap = max(20, int(diagonal * 0.03))
    segments = cv.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if segments is None:
        return corners

    ordered = order_corners(corners)
    refined_lines = []

    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        hough_line = select_side_hough_line(start, end, segments, config)

        if hough_line is None:
            refined_lines.append(points_to_line(start, end))
            continue

        refined_lines.append(segment_to_line(hough_line))

    refined_corners = []
    for index in range(4):
        point = intersect_lines(refined_lines[index - 1], refined_lines[index])
        if point is None:
            return ordered
        refined_corners.append(point)

    refined = order_corners(np.array(refined_corners, dtype=np.float32))
    return clamp_corners_to_frame(refined, frame_shape)


def should_refine_quadrilateral(corners: np.ndarray) -> bool:
    ordered = order_corners(corners)
    top_angle = segment_angle_degrees(ordered[0], ordered[1])
    right_angle = segment_angle_degrees(ordered[1], ordered[2])
    bottom_angle = segment_angle_degrees(ordered[3], ordered[2])
    left_angle = segment_angle_degrees(ordered[0], ordered[3])

    horizontal_mismatch = angular_distance_degrees(top_angle, bottom_angle)
    vertical_mismatch = angular_distance_degrees(right_angle, left_angle)
    return max(horizontal_mismatch, vertical_mismatch) >= 12.0


def select_side_hough_line(
    side_start: np.ndarray,
    side_end: np.ndarray,
    segments: np.ndarray,
    config: PaperDetectionConfig,
) -> np.ndarray | None:
    side_angle = segment_angle_degrees(side_start, side_end)
    side_length = float(np.hypot(side_end[0] - side_start[0], side_end[1] - side_start[1]))
    max_distance = max(40.0, side_length * 0.08)
    best_segment = None
    best_score = 0.0

    for segment in segments[:, 0]:
        x1, y1, x2, y2 = segment.astype(np.float32)
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        angle = segment_angle_degrees(np.array([x1, y1]), np.array([x2, y2]))
        angle_delta = angular_distance_degrees(angle, side_angle)
        if angle_delta > config.hough_angle_tolerance_degrees:
            continue

        distance = point_to_line_distance(midpoint, side_start, side_end)
        if distance > max_distance:
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        score = (
            length
            - (config.hough_distance_weight * distance)
            - (config.hough_angle_weight * angle_delta)
        )
        if score > best_score:
            best_score = score
            best_segment = segment

    return best_segment


def points_to_line(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.array([start[0], start[1], end[0] - start[0], end[1] - start[1]], dtype=np.float32)


def segment_to_line(segment: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = segment.astype(np.float32)
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    x1, y1, vx1, vy1 = first
    x2, y2, vx2, vy2 = second
    matrix = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None

    offset = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    parameter, _ = np.linalg.solve(matrix, offset)
    return np.array([x1 + (parameter * vx1), y1 + (parameter * vy1)], dtype=np.float32)


def segment_angle_degrees(start: np.ndarray, end: np.ndarray) -> float:
    return float((np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])) + 180.0) % 180.0)


def angular_distance_degrees(first: float, second: float) -> float:
    delta = abs(first - second) % 180.0
    return min(delta, 180.0 - delta)


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    delta = line_end - line_start
    length = float(np.hypot(delta[0], delta[1]))
    if length <= 1e-6:
        return float("inf")

    numerator = abs(
        delta[0] * (line_start[1] - point[1])
        - (line_start[0] - point[0]) * delta[1]
    )
    return float(numerator / length)


def clamp_corners_to_frame(corners: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    clamped = corners.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
    return clamped


def order_corners(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


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
