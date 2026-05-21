from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass
class PaperDetectionConfig:
    min_contour_area_ratio: float = 0.10
    approx_epsilon_ratio: float = 0.02
    min_rectangularity: float = 0.70
    threshold_bias: int = 10
    max_saturation: int = 60
    min_brightness: int = 150
    border_margin: int = 5
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

    corners = None
    if candidate is not None:
        corners = order_corners(candidate.reshape(4, 2))
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

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, saturation_mask = cv.threshold(
        saturation,
        config.max_saturation,
        255,
        cv.THRESH_BINARY_INV,
    )

    mask = cv.bitwise_and(brightness_mask, saturation_mask)

    kernel_close = np.ones((9, 9), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open, iterations=1)
    return mask


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

        x, y, width, height = cv.boundingRect(contour)
        if (
            x <= config.border_margin
            or y <= config.border_margin
            or x + width >= frame_shape[1] - config.border_margin
            or y + height >= frame_shape[0] - config.border_margin
        ):
            continue

        hull = cv.convexHull(contour)
        perimeter = cv.arcLength(hull, True)
        approx = cv.approxPolyDP(
            hull,
            config.approx_epsilon_ratio * perimeter,
            True,
        )

        if not cv.isContourConvex(approx):
            continue

        candidate = approx
        if len(candidate) != 4:
            rect = cv.minAreaRect(hull)
            candidate = cv.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2)

        if len(candidate) != 4:
            continue

        rect_area = cv.contourArea(candidate.astype(np.float32))
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        if rectangularity < config.min_rectangularity:
            continue

        return candidate

    return None


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
