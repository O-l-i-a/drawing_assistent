import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

try:
    from .figure_item import PlacedFigureItem
    from .figure_sidebar import FIGURE_MIME_TYPE
    from .qt_utils import cv_to_qpixmap
except ImportError:
    from figure_item import PlacedFigureItem
    from figure_sidebar import FIGURE_MIME_TYPE
    from qt_utils import cv_to_qpixmap

DEFAULT_SHORT_SIDE = 700
A4_RATIO = 1.4142  # long side / short side


def canvas_dimensions(orientation: str, short_side: int = DEFAULT_SHORT_SIDE) -> tuple[int, int]:
    """Pixel size of the canonical paper canvas for a given orientation."""
    long_side = round(short_side * A4_RATIO)
    if orientation == "landscape":
        return long_side, short_side
    return short_side, long_side


def fit_to_canvas(image, box_w: int, box_h: int, pad_color=(200, 200, 200)):
    """Scale `image` to fit inside (box_w, box_h) without distorting it, letterboxing
    the remainder with `pad_color` so the result is always exactly (box_h, box_w, 3)."""
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

    canvas = np.full((box_h, box_w, 3), pad_color, dtype=np.uint8)
    x0 = (box_w - new_w) // 2
    y0 = (box_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, x0, y0, scale


class PaperCanvasView(QGraphicsView):
    """Shows the live segmented-paper feed and hosts drag-and-dropped target figures."""

    def __init__(self, sidebar, orientation: str = "portrait", short_side: int = DEFAULT_SHORT_SIDE, parent=None):
        self.canvas_width, self.canvas_height = canvas_dimensions(orientation, short_side)
        # Keep a Python reference to the scene: QGraphicsView does not own it,
        # so without this it can be garbage-collected out from under the view.
        self._scene = QGraphicsScene(0, 0, self.canvas_width, self.canvas_height)
        super().__init__(self._scene, parent)
        self._sidebar = sidebar
        self.setAcceptDrops(True)
        self.setFixedSize(self.canvas_width + 4, self.canvas_height + 4)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._background_item = QGraphicsPixmapItem()
        self._background_item.setZValue(-100)
        self._scene.addItem(self._background_item)

    def set_background(self, canonical_bgr: np.ndarray) -> None:
        self._background_item.setPixmap(cv_to_qpixmap(canonical_bgr))

    def placed_items(self) -> list[PlacedFigureItem]:
        return [it for it in self._scene.items() if isinstance(it, PlacedFigureItem)]

    def placed_targets(self) -> list[dict]:
        return [{"name": it.name, "points": it.scene_contour()} for it in self.placed_items()]

    def lock_all(self) -> None:
        for it in self.placed_items():
            it.lock()

    def unlock_all(self) -> None:
        for it in self.placed_items():
            it.unlock()

    def clear_figures(self) -> None:
        for it in self.placed_items():
            self._scene.removeItem(it)

    # --- drag & drop from the sidebar ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(FIGURE_MIME_TYPE):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(FIGURE_MIME_TYPE):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        # We accept figure drags in dragEnterEvent without forwarding to the
        # scene, so QGraphicsView's own drag-tracking state is never armed;
        # its default dragLeaveEvent then logs a spurious "leave before
        # enter" warning. Handling it ourselves silences that.
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in list(self._scene.selectedItems()):
                if isinstance(item, PlacedFigureItem):
                    self._scene.removeItem(item)
            event.accept()
            return
        super().keyPressEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime.hasFormat(FIGURE_MIME_TYPE):
            super().dropEvent(event)
            return
        name = bytes(mime.data(FIGURE_MIME_TYPE)).decode("utf-8")
        points = self._sidebar.template_points(name)
        scene_pos = self.mapToScene(event.position().toPoint())
        item = PlacedFigureItem(name, points)
        item.setPos(scene_pos)
        self._scene.addItem(item)
        event.acceptProposedAction()
