import cv2 as cv
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ..get_wrapped_paper import PaperDetectionConfig, PaperState, get_wrapped_paper
    from ..preprocess_drawing import preprocess_drawing
    from .figure_sidebar import FigureSidebar
    from .paper_canvas import A4_RATIO, DEFAULT_SHORT_SIDE, PaperCanvasView, fit_to_canvas
    from .qt_utils import cv_to_qpixmap
    from .scoring import score_targets
except ImportError:
    from get_wrapped_paper import PaperDetectionConfig, PaperState, get_wrapped_paper
    from preprocess_drawing import preprocess_drawing
    from figure_sidebar import FigureSidebar
    from paper_canvas import A4_RATIO, DEFAULT_SHORT_SIDE, PaperCanvasView, fit_to_canvas
    from qt_utils import cv_to_qpixmap
    from scoring import score_targets

SIDEBAR_RESERVED_WIDTH = 190  # sidebar's own fixed width + layout spacing
CONTROLS_RESERVED_HEIGHT = 90  # controls bar height + layout spacing
SCREEN_MARGIN = 0.94  # leave a little breathing room below the true available size


def _compute_short_side(orientation: str) -> int:
    """Pick a canvas short-side length so both panels (stacked for landscape,
    side by side for portrait) fit within the available screen, at that
    orientation's shape, instead of a size that can overflow the screen."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return DEFAULT_SHORT_SIDE
    geo = screen.availableGeometry()
    budget_w = (geo.width() - SIDEBAR_RESERVED_WIDTH) * SCREEN_MARGIN
    budget_h = (geo.height() - CONTROLS_RESERVED_HEIGHT) * SCREEN_MARGIN

    if orientation == "landscape":
        # two panels stacked vertically: each gets half the height, full width
        short_by_height = budget_h / 2
        short_by_width = budget_w / A4_RATIO
    else:
        # two panels side by side: each gets half the width, full height
        short_by_width = budget_w / 2
        short_by_height = budget_h / A4_RATIO

    return max(200, int(min(short_by_height, short_by_width)))


class MainWindow(QMainWindow):
    """sidebar | live camera feed | segmented paper canvas, plus Start/Compare controls.

    `capture` only needs to duck-type `cv.VideoCapture`'s `isOpened`/`set`/`read`/
    `grab`/`release`, so a static-image source can be swapped in for `--source image`.

    `orientation` ("portrait" or "landscape") sets the shape of the paper canvas
    (and, to match it, the live-feed panel) to the physical paper's orientation.
    Frames are always fit into that shape preserving aspect ratio (letterboxed,
    never stretched) regardless of the source video/image's own resolution.
    """

    def __init__(self, capture, frame_skip: int = 3, orientation: str = "portrait"):
        super().__init__()
        self.setWindowTitle("Drawing Assistant")

        self._config = PaperDetectionConfig()
        self._state = PaperState()
        self._frame_skip = max(1, frame_skip)
        self._frame_count = 0
        self._last_canonical = None  # last canonical-sized warped BGR frame, used by Compare

        self._cap = capture
        if not self._cap.isOpened():
            raise RuntimeError("Could not open the given capture source")

        short_side = _compute_short_side(orientation)
        self._sidebar = FigureSidebar()
        self._canvas = PaperCanvasView(self._sidebar, orientation=orientation, short_side=short_side)

        self._live_label = QLabel("Waiting for camera...")
        self._live_label.setFixedSize(self._canvas.canvas_width, self._canvas.canvas_height)
        self._live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_label.setScaledContents(False)
        self._live_label.setStyleSheet("background-color: #202020; color: white;")

        self._start_button = QPushButton("Start")
        self._stop_button = QPushButton("Stop")
        self._compare_button = QPushButton("Compare")
        self._segments_checkbox = QCheckBox("Show segments")
        self._score_label = QLabel("Score: --")
        self._start_button.clicked.connect(self._on_start)
        self._stop_button.clicked.connect(self._on_stop)
        self._compare_button.clicked.connect(self._on_compare)

        controls = QHBoxLayout()
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._compare_button)
        controls.addWidget(self._segments_checkbox)
        controls.addWidget(self._score_label)
        controls.addStretch(1)

        # Landscape paper -> canvas above the live feed (stacked). Portrait -> canvas left of the live feed.
        images_layout = QVBoxLayout() if orientation == "landscape" else QHBoxLayout()
        images_layout.addWidget(self._canvas)
        images_layout.addWidget(self._live_label)

        panels = QHBoxLayout()
        panels.addWidget(self._sidebar)
        panels.addLayout(images_layout)

        root = QVBoxLayout()
        root.addLayout(panels)
        root.addLayout(controls)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(30)

    def _on_tick(self) -> None:
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = self._cap.read()
        self._cap.grab()
        if not ret:
            return

        self._frame_count += 1
        if self._frame_count % self._frame_skip != 0:
            return

        result = get_wrapped_paper(frame.copy(), self._state, self._config)
        self._state.last_corners = result.corners
        self._set_live_pixmap(result.overlay)

        if result.warped is not None:
            canonical = fit_to_canvas(result.warped, self._canvas.canvas_width, self._canvas.canvas_height)
            self._last_canonical = canonical
            display = self._segments_overlay(canonical) if self._segments_checkbox.isChecked() else canonical
            self._canvas.set_background(display)

    def _segments_overlay(self, canonical):
        """Red overlay of the region-growing segments, for live visual feedback only
        (scoring always re-runs `preprocess_drawing` on the clean frame at Compare time)."""
        _, regions = preprocess_drawing(canonical)
        overlay = canonical.copy()
        for region in regions:
            overlay[region["mask"] == 255] = (0, 0, 255)
        return cv.addWeighted(overlay, 0.5, canonical, 0.5, 0)

    def _set_live_pixmap(self, bgr) -> None:
        pixmap = cv_to_qpixmap(bgr)
        scaled = pixmap.scaled(
            self._live_label.width(),
            self._live_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._live_label.setPixmap(scaled)

    def _on_start(self) -> None:
        self._canvas.lock_all()

    def _on_stop(self) -> None:
        self._canvas.unlock_all()

    def _on_compare(self) -> None:
        if self._last_canonical is None:
            self._score_label.setText("Score: no paper detected yet")
            return
        targets = self._canvas.placed_targets()
        if not targets:
            self._score_label.setText("Score: place a figure first")
            return
        results, overall = score_targets(self._last_canonical, targets)
        parts = [f"{r['name']}={r['score']:.0f}%" for r in results]
        self._score_label.setText(f"Score: {overall:.0f}%  (" + ", ".join(parts) + ")")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._cap.release()
        super().closeEvent(event)
