# sync_image_viewer.py
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
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
    roi_changed = Signal()

    def __init__(self, view_key, shared_state, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.view_key = view_key
        self.shared_state = shared_state
        self.pixmap = None
        self.roi_state = None

        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self._roi_action = None
        self._roi_start_pos = None
        self._roi_start_rect = None

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def set_roi_state(self, roi_state):
        self.roi_state = roi_state
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

        roi_rect = self._get_roi_rect()
        if roi_rect:
            x, y, rw, rh = roi_rect
            pen = QPen(QColor(0, 255, 255))
            pen.setWidthF(2.0 / max(scale, 1e-6))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, rw, rh)

            handle_size = 8.0 / max(scale, 1e-6)
            half = handle_size / 2.0
            painter.setBrush(QBrush(QColor(0, 255, 255)))
            for hx, hy in self._roi_handles(roi_rect):
                painter.drawRect(hx - half, hy - half, handle_size, handle_size)

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

    def _get_roi_rect(self):
        if not self.roi_state:
            return None
        if not self.roi_state.get("enabled"):
            return None
        rect = self.roi_state.get("rect")
        if not rect:
            return None
        return rect

    def _roi_handles(self, rect):
        x, y, w, h = rect
        return [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h),
        ]

    def _hit_test_roi_handle(self, img_x, img_y):
        rect = self._get_roi_rect()
        if not rect:
            return None
        scale = self.shared_state.get("scale", 1.0)
        tol = 6.0 / max(scale, 1e-6)
        handles = self._roi_handles(rect)
        for idx, (hx, hy) in enumerate(handles):
            if abs(img_x - hx) <= tol and abs(img_y - hy) <= tol:
                return idx
        return None

    def _point_inside_rect(self, img_x, img_y, rect):
        x, y, w, h = rect
        return x <= img_x <= x + w and y <= img_y <= y + h

    def _clamp_roi_rect(self, rect):
        if not self.pixmap:
            return rect
        x, y, w, h = rect
        min_size = 5.0
        w = max(w, min_size)
        h = max(h, min_size)
        max_w = self.pixmap.width()
        max_h = self.pixmap.height()
        x = max(0.0, min(x, max_w - w))
        y = max(0.0, min(y, max_h - h))
        return (x, y, w, h)

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
            img_x, img_y = self._widget_pos_to_image_pos(event.position())
            rect = self._get_roi_rect()
            handle_idx = self._hit_test_roi_handle(img_x, img_y)
            if rect and handle_idx is not None:
                self._roi_action = ("resize", handle_idx)
                self._roi_start_pos = (img_x, img_y)
                self._roi_start_rect = rect
                self.setCursor(Qt.SizeAllCursor)
                event.accept()
                return
            if rect and self._point_inside_rect(img_x, img_y, rect):
                self._roi_action = ("move", None)
                self._roi_start_pos = (img_x, img_y)
                self._roi_start_rect = rect
                self.setCursor(Qt.SizeAllCursor)
                event.accept()
                return
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._roi_action is not None:
                self._roi_action = None
                self._roi_start_pos = None
                self._roi_start_rect = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if not self.pixmap:
            return

        pos = event.position()

        if event.buttons() & Qt.LeftButton and self._roi_action is not None:
            img_x, img_y = self._widget_pos_to_image_pos(pos)
            rect = self._roi_start_rect
            if rect:
                dx = img_x - self._roi_start_pos[0]
                dy = img_y - self._roi_start_pos[1]
                x, y, w, h = rect
                action, handle_idx = self._roi_action
                if action == "move":
                    new_rect = (x + dx, y + dy, w, h)
                else:
                    if handle_idx == 0:
                        new_x = x + dx
                        new_y = y + dy
                        new_rect = (new_x, new_y, w - dx, h - dy)
                    elif handle_idx == 1:
                        new_y = y + dy
                        new_rect = (x, new_y, w + dx, h - dy)
                    elif handle_idx == 2:
                        new_x = x + dx
                        new_rect = (new_x, y, w - dx, h + dy)
                    else:
                        new_rect = (x, y, w + dx, h + dy)
                    min_size = 5.0
                    new_x, new_y, new_w, new_h = new_rect
                    if handle_idx in (0, 2) and new_w < min_size:
                        new_x = new_x + new_w - min_size
                        new_w = min_size
                    if handle_idx in (0, 1) and new_h < min_size:
                        new_y = new_y + new_h - min_size
                        new_h = min_size
                    if handle_idx in (1, 3) and new_w < min_size:
                        new_w = min_size
                    if handle_idx in (2, 3) and new_h < min_size:
                        new_h = min_size
                    new_rect = (new_x, new_y, new_w, new_h)
                new_rect = self._clamp_roi_rect(new_rect)
                if self.roi_state is not None:
                    self.roi_state["rect"] = new_rect
                    self.roi_changed.emit()
                    self.update()
        elif event.buttons() & Qt.LeftButton and self.last_mouse_pos is not None:
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
