from PySide6.QtCore import QMimeData, QSize, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QListWidget, QListWidgetItem

try:
    from .qt_utils import render_shape_icon
    from .templates import get_figure_templates
except ImportError:
    from qt_utils import render_shape_icon
    from templates import get_figure_templates

FIGURE_MIME_TYPE = "application/x-drawing-assistant-figure"


class FigureSidebar(QListWidget):
    """Sidebar listing the available reference shapes, draggable onto the paper canvas.

    Populated dynamically from `templates.get_figure_templates()`, so adding a
    shape to match_shapes.py's template lists makes it appear here automatically.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setIconSize(QSize(56, 56))
        self.setSpacing(6)
        self.setFixedWidth(160)
        self._templates = {t["name"]: t["points"] for t in get_figure_templates()}
        self._populate()

    def _populate(self) -> None:
        for name, points in self._templates.items():
            item = QListWidgetItem(render_shape_icon(points), name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.addItem(item)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        mime = QMimeData()
        mime.setData(FIGURE_MIME_TYPE, name.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(self.iconSize()))
        drag.exec(Qt.DropAction.CopyAction)

    def template_points(self, name: str):
        return self._templates[name]
