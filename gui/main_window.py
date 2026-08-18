import cv2 as cv
import numpy as np
from dataclasses import dataclass
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
    from ..match_shapes import TemplateState, match_shapes, line_by_line
    from .figure_sidebar import FigureSidebar
    from .paper_canvas import A4_RATIO, DEFAULT_SHORT_SIDE, PaperCanvasView, fit_to_canvas
    from .qt_utils import cv_to_qpixmap
    from .scoring import score_targets
except ImportError:
    from get_wrapped_paper import PaperDetectionConfig, PaperState, get_wrapped_paper
    from preprocess_drawing import preprocess_drawing
    from match_shapes import TemplateState, match_shapes, line_by_line
    from gui.figure_sidebar import FigureSidebar
    from gui.paper_canvas import A4_RATIO, DEFAULT_SHORT_SIDE, PaperCanvasView, fit_to_canvas
    from gui.qt_utils import cv_to_qpixmap
    from gui.scoring import score_targets

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
        self._next_line_button = QPushButton("Next line")
        self._segments_checkbox = QCheckBox("Show segments")
        self._score_label = QLabel("Score: --")
        self._start_button.clicked.connect(self._on_start)
        self._stop_button.clicked.connect(self._on_stop)
        self._compare_button.clicked.connect(self._on_compare)
        self._next_line_button.clicked.connect(self._on_next_line)
        self._next_line_button.setEnabled(False)

        self._do_match_overlay = False
        # Off by default: advancing only happens when you press "Next line".
        # Set to True (e.g. `window._auto_advance_line = True`) to also
        # auto-advance once a matching stroke is detected for a few stable
        # frames; the button itself always works regardless of this setting.
        self._auto_advance_line = False
        # "Move here next" arrows between lines (currently only for A/tri,
        # see TRANSITION_ARROW_SHAPES). Set to False (e.g.
        # `window._show_transition_arrows = False`) to hide them.
        self._show_transition_arrows = True
        self._auto_scored = False  # guards against re-triggering Compare every tick once complete
        self.template_states = {}

        controls = QHBoxLayout()
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._compare_button)
        controls.addWidget(self._next_line_button)
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
    
    def process_frame_with_overlay(self, result, canonical):
        if result.warped is None or result.corners is None:
            return result.overlay, canonical
        # Segment the canonical (canvas-space) image, not the raw warped
        # paper, so region masks share pixel coordinates with the placed
        # figures (which live in canvas/scene space) -- required for
        # line_by_line's ink-diff tracking to line up with the template.
        _, regions = preprocess_drawing(canonical)
        segments = [r for r in regions if "mask" in r]

        canonical_overlay = canonical.copy()

        h_warp, w_warp = result.warped.shape[:2]
        h_can, w_can = canonical.shape[:2]

        scale_x = (w_can / w_warp)
        scale_y = (h_can / h_warp) 
        scale = min(scale_x,scale_y)
        new_w = w_warp * scale
        new_h = h_warp * scale
        x0 = (w_can - new_w) / 2
        y0 = (h_can - new_h) / 2

        dst = np.array([[0, 0], [w_warp - 1, 0], [w_warp - 1, h_warp - 1], [0, h_warp - 1]], dtype=np.float32)
        src_corners = result.corners.astype(np.float32)
        M = cv.getPerspectiveTransform(dst, src_corners)

        for name, state in self.template_states.items():
            if name != "":
                canonical_overlay = line_by_line(
                    canonical_overlay, segments, state,
                    auto_advance=self._auto_advance_line,
                    show_transition_arrows=self._show_transition_arrows,
                )

        warped_overlay = np.zeros_like(result.warped)
        paper_roi = canonical_overlay[int(y0):int(y0 + new_h), int(x0):int(x0 + new_w)]
        if paper_roi.size > 0:
            paper_resized = cv.resize(paper_roi, (w_warp, h_warp), interpolation=cv.INTER_LINEAR)
            warped_overlay = paper_resized

        match_back = cv.warpPerspective(warped_overlay, M, (result.overlay.shape[1], result.overlay.shape[0]))

        overlay = result.overlay.copy()
        mask = (match_back.sum(axis=2) > 0)
        overlay[mask] = cv.addWeighted(overlay[mask], 0.4, match_back[mask], 0.6, 0)
        
        return overlay, canonical_overlay


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

        if result.warped is not None:
            clean_canonical, x0, y0, scale = fit_to_canvas(result.warped, self._canvas.canvas_width, self._canvas.canvas_height)
            canonical = clean_canonical
            self._last_canonical = clean_canonical
            self._last_fit = (x0, y0, scale)

            if self._do_match_overlay:

                current_names = {item.name for item in self._canvas.placed_items()}
                stored_names = set(self.template_states.keys())

                for name in stored_names - current_names:
                    del self.template_states[name]
                for item in self._canvas.placed_items():
                    if item.name not in self.template_states:
                        self.template_states[item.name] = TemplateState()
                        self.template_states[item.name].name = item.name
                        self.template_states[item.name].item = item
                        self.template_states[item.name].view = self._canvas

                for item in self._canvas.placed_items():
                    name = item.name
                    if name not in self.template_states:
                        self.template_states[name] = TemplateState()
                    state = self.template_states[name]
                    state.name = name
                    state.item = item
                    state.view = self._canvas
                    
                    rect = item.mapRectToScene(item.boundingRect())

                    px_x = int(rect.x())
                    px_y = int(rect.y())
                    px_w = int(rect.width())
                    px_h = int(rect.height())

                    state.box = (px_x, px_y, px_w, px_h)

                processed, canonical = self.process_frame_with_overlay(result, canonical)
                self._set_live_pixmap(processed)

                all_complete = bool(self.template_states) and all(s.complete for s in self.template_states.values())
                self._next_line_button.setEnabled(not all_complete)
                if all_complete and not self._auto_scored:
                    self._on_compare()
                    self._auto_scored = True
            else:
                self._set_live_pixmap(result.overlay)

            # Segments are always detected from the clean, un-annotated paper
            # image -- never the line-by-line guide overlay, whose own drawn
            # contours would otherwise get picked up as if they were ink --
            # but drawn onto whatever's currently on the canvas (the guide
            # included), so checking "Show segments" doesn't hide the
            # line-by-line guidance on the left panel.
            display = self._segments_overlay(clean_canonical, canonical) if self._segments_checkbox.isChecked() else canonical
            self._canvas.set_background(display)

    def _segments_overlay(self, clean_canonical, display_canonical):
        """Red overlay of every region-growing segment detected on the clean
        paper (scoring always re-runs `preprocess_drawing` on the clean
        frame at Compare time), drawn on top of `display_canonical` so any
        line-by-line guide already on screen stays visible."""
        _, regions = preprocess_drawing(clean_canonical)
        overlay = display_canonical.copy()
        for region in regions:
            overlay[region["mask"] == 255] = (0, 0, 255)
        return cv.addWeighted(overlay, 0.5, display_canonical, 0.5, 0)

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
        self._do_match_overlay = True
        self._canvas.lock_all()
        self._state.corners_locked = True
        self._next_line_button.setEnabled(True)
        self._auto_scored = False
        # Re-checkpoint ink tracking now, so anything drawn before this Start
        # doesn't count as "the line you just drew" for the first advance.
        for state in self.template_states.values():
            state.last_ink_mask = None

    def _on_stop(self) -> None:
        self._do_match_overlay = False
        self._canvas.unlock_all()
        self._state.corners_locked = False
        self._next_line_button.setEnabled(False)

    def _on_next_line(self) -> None:
        """Manually advance every placed figure's current stroke: fit its
        template to whatever was drawn since Start (or the last Next line)
        and reveal the following stroke. Always honored, independent of
        `self._auto_advance_line`."""
        for state in self.template_states.values():
            state.manual_trigger = True

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
