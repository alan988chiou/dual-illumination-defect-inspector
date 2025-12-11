# sync_image_viewer.py
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget


class SyncImageViewer(QWidget):
    """
    Image display widget:
    - Left mouse drag: pan
    - Mouse wheel: zoom (centered on cursor position)
    - Multiple viewers share shared_state for synchronized pan/zoom
    - mouse_info emits the mapped image coordinates for the cursor
    """
    mouse_info = Signal(str, float, float, bool)     # view_key, img_x, img_y, inside
    transform_changed = Signal()                     # Emitted when pan/zoom changes

    def __init__(self, view_key, shared_state, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.view_key = view_key
        self.shared_state = shared_state
        self.pixmap = None

        self.setMouseTracking(True)
        self.last_mouse_pos = None

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap:
            return

        scale = self.shared_state.get("scale", 1.0)
        cx = self.shared_state.get("center_x", 0.0)
        cy = self.shared_state.get("center_y", 0.0)

        w = self.width()
        h = self.height()

        painter.translate(w / 2.0, h / 2.0)
        painter.scale(scale, scale)
        painter.translate(-cx, -cy)
        painter.drawPixmap(0, 0, self.pixmap)

    def _widget_pos_to_image_pos(self, pos):
        if not self.pixmap:
            return 0.0, 0.0

        scale = self.shared_state.get("scale", 1.0)
        cx = self.shared_state.get("center_x", 0.0)
        cy = self.shared_state.get("center_y", 0.0)

        w = self.width()
        h = self.height()

        px = pos.x()
        py = pos.y()

        img_x = (px - w / 2.0) / scale + cx
        img_y = (py - h / 2.0) / scale + cy
        return img_x, img_y

    def wheelEvent(self, event):
        if not self.pixmap:
            return

        angle = event.angleDelta().y()
        if angle == 0:
            return

        factor = 1.25 if angle > 0 else 0.8
        self._zoom_at(event.position(), factor)
        event.accept()

    def _zoom_at(self, pos, factor):
        if not self.pixmap:
            return

        old_scale = self.shared_state.get("scale", 1.0)
        new_scale = old_scale * factor
        new_scale = max(0.05, min(20.0, new_scale))  # Limit zoom range

        if abs(new_scale - old_scale) < 1e-6:
            return

        w = self.width()
        h = self.height()
        cx = self.shared_state.get("center_x", 0.0)
        cy = self.shared_state.get("center_y", 0.0)

        px = pos.x()
        py = pos.y()

        img_x_before = (px - w / 2.0) / old_scale + cx
        img_y_before = (py - h / 2.0) / old_scale + cy

        cx_new = img_x_before - (px - w / 2.0) / new_scale
        cy_new = img_y_before - (py - h / 2.0) / new_scale

        self.shared_state["scale"] = new_scale
        self.shared_state["center_x"] = cx_new
        self.shared_state["center_y"] = cy_new

        self.transform_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if not self.pixmap:
            return

        pos = event.position()

        if event.buttons() & Qt.LeftButton and self.last_mouse_pos is not None:
            # Pan
            delta = pos - self.last_mouse_pos
            self.last_mouse_pos = pos

            scale = self.shared_state.get("scale", 1.0)
            dx_img = -delta.x() / scale
            dy_img = -delta.y() / scale

            self.shared_state["center_x"] = self.shared_state.get("center_x", 0.0) + dx_img
            self.shared_state["center_y"] = self.shared_state.get("center_y", 0.0) + dy_img

            self.transform_changed.emit()

        # Report cursor position
        img_x, img_y = self._widget_pos_to_image_pos(pos)
        pm_w = self.pixmap.width()
        pm_h = self.pixmap.height()
        inside = (0 <= img_x < pm_w) and (0 <= img_y < pm_h)

        self.mouse_info.emit(self.view_key, img_x, img_y, inside)
        event.accept()
