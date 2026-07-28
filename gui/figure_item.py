import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem

MIN_SCALE = 0.25
MAX_SCALE = 4.0


class PlacedFigureItem(QGraphicsPolygonItem):
    """A template shape dropped onto the paper canvas.

    Movable, rotatable (top handle), and resizable (corner handle) while
    unlocked. `lock()` freezes it in place once the user presses Start,
    turning it into a static reference outline for tracing.
    """

    def __init__(self, name: str, points: np.ndarray, target_size: float = 140.0):
        super().__init__()
        self.name = name

        pts = points.astype(float)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-6)
        scale = target_size / extent.max()
        centered = (pts - mins - extent / 2.0) * scale

        self._base_polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in centered])
        self.setPolygon(self._base_polygon)

        rect = self._base_polygon.boundingRect()
        self._half_width = rect.width() / 2.0
        self._half_height = rect.height() / 2.0
        self._handle_offset = 24.0
        self._handle_radius = 6.0
        self._locked = False

        self._rotating = False
        self._rotate_start_angle = 0.0
        self._rotate_start_rotation = 0.0

        self._resizing = False
        self._resize_start_scale = 1.0
        self._resize_start_distance = 1.0

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setTransformOriginPoint(0.0, 0.0)
        self._apply_style()

    def _apply_style(self) -> None:
        if self._locked:
            pen = QPen(Qt.PenStyle.NoPen)
            brush = QBrush(Qt.BrushStyle.NoBrush)
        else:
            pen = QPen(QColor(255, 210, 0), 3)
            brush = QBrush(QColor(255, 210, 0, 60))
        self.setPen(pen)
        self.setBrush(brush)

    def lock(self) -> None:
        self._locked = True
        self.setSelected(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self._apply_style()
        self.update()

    def unlock(self) -> None:
        self._locked = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._apply_style()
        self.update()

    def scene_contour(self) -> np.ndarray:
        """Return this figure's outline in scene (canonical canvas) pixel coordinates."""
        poly = self.mapToScene(self.polygon())
        return np.array([[p.x(), p.y()] for p in poly], dtype=np.float32)

    def _rotate_handle_center(self) -> QPointF:
        return QPointF(0.0, -(self._half_height + self._handle_offset))

    def _resize_handle_center(self) -> QPointF:
        return QPointF(self._half_width, self._half_height)

    def _delete_handle_center(self) -> QPointF:
        return QPointF(-self._half_width, self._half_height)

    def boundingRect(self):
        rect = self.polygon().boundingRect()
        top_extra = self._handle_offset + self._handle_radius + 4.0
        corner_extra = self._handle_radius + 4.0
        return rect.adjusted(-corner_extra, -top_extra, corner_extra, corner_extra)

    def shape(self):
        path = QPainterPath()
        path.addPolygon(self.polygon())
        if not self._locked:
            path.addEllipse(self._rotate_handle_center(), self._handle_radius + 4, self._handle_radius + 4)
            path.addEllipse(self._resize_handle_center(), self._handle_radius + 4, self._handle_radius + 4)
            path.addEllipse(self._delete_handle_center(), self._handle_radius + 4, self._handle_radius + 4)
        return path

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.isSelected() and not self._locked:
            painter.save()
            handle_brush = QBrush(QColor(40, 120, 255))
            dashed_pen = QPen(QColor(40, 120, 255), 1, Qt.PenStyle.DashLine)
            handle_pen = QPen(QColor(40, 120, 255), 2)

            rotate_center = self._rotate_handle_center()
            painter.setPen(dashed_pen)
            painter.drawLine(QPointF(0, -self._half_height), rotate_center)
            painter.setPen(handle_pen)
            painter.setBrush(handle_brush)
            painter.drawEllipse(rotate_center, self._handle_radius, self._handle_radius)

            resize_center = self._resize_handle_center()
            painter.setPen(handle_pen)
            painter.setBrush(handle_brush)
            painter.drawRect(
                resize_center.x() - self._handle_radius,
                resize_center.y() - self._handle_radius,
                self._handle_radius * 2,
                self._handle_radius * 2,
            )

            delete_center = self._delete_handle_center()
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(220, 40, 40)))
            painter.drawEllipse(delete_center, self._handle_radius, self._handle_radius)
            r = self._handle_radius * 0.55
            painter.drawLine(
                QPointF(delete_center.x() - r, delete_center.y() - r),
                QPointF(delete_center.x() + r, delete_center.y() + r),
            )
            painter.drawLine(
                QPointF(delete_center.x() - r, delete_center.y() + r),
                QPointF(delete_center.x() + r, delete_center.y() - r),
            )
            painter.restore()

    def _handle_hit(self, local_pos: QPointF, center: QPointF) -> bool:
        dx = local_pos.x() - center.x()
        dy = local_pos.y() - center.y()
        return math.hypot(dx, dy) <= self._handle_radius + 6

    def mousePressEvent(self, event):
        if not self._locked and self.isSelected():
            if self._handle_hit(event.pos(), self._delete_handle_center()):
                scene = self.scene()
                if scene is not None:
                    scene.removeItem(self)
                event.accept()
                return
            if self._handle_hit(event.pos(), self._rotate_handle_center()):
                self._rotating = True
                vec = event.scenePos() - self.pos()
                self._rotate_start_angle = math.degrees(math.atan2(vec.y(), vec.x()))
                self._rotate_start_rotation = self.rotation()
                event.accept()
                return
            if self._handle_hit(event.pos(), self._resize_handle_center()):
                self._resizing = True
                vec = event.scenePos() - self.pos()
                self._resize_start_distance = max(1e-3, math.hypot(vec.x(), vec.y()))
                self._resize_start_scale = self.scale()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rotating:
            vec = event.scenePos() - self.pos()
            angle_now = math.degrees(math.atan2(vec.y(), vec.x()))
            delta = angle_now - self._rotate_start_angle
            self.setRotation(self._rotate_start_rotation + delta)
            event.accept()
            return
        if self._resizing:
            vec = event.scenePos() - self.pos()
            distance = math.hypot(vec.x(), vec.y())
            factor = distance / self._resize_start_distance
            new_scale = min(MAX_SCALE, max(MIN_SCALE, self._resize_start_scale * factor))
            self.setScale(new_scale)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rotating:
            self._rotating = False
            event.accept()
            return
        if self._resizing:
            self._resizing = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
