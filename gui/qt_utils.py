import cv2 as cv
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def cv_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert a BGR (or grayscale) OpenCV image to a QPixmap."""
    if bgr.ndim == 2:
        bgr = cv.cvtColor(bgr, cv.COLOR_GRAY2BGR)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def render_shape_icon(points: np.ndarray, size: int = 64, margin: int = 8) -> QPixmap:
    """Render a template's polygon points into a small square icon."""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    pts = points.astype(np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-3)
    scale = (size - 2 * margin) / extent.max()
    centered = (pts - mins - extent / 2) * scale
    shifted = centered + size / 2
    poly = shifted.astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(canvas, [poly], isClosed=True, color=(60, 60, 60), thickness=2, lineType=cv.LINE_AA)
    return cv_to_qpixmap(canvas)
