# aoi_main_window.py
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QCheckBox, QFrame, QSizePolicy, QGroupBox,
    QDialog, QMessageBox
)

from PySide6.QtCore import Qt, Slot, QTimer, QSettings
from PySide6.QtGui import QImage, QPixmap, QResizeEvent

from sync_image_viewer import SyncImageViewer
from bright_field_processor import process_bright_field
from dark_field_processor import process_dark_field
from load_image_dialog import LoadImageDialog

BACKGROUND_COLOR_CSS = "background: rgb(128, 128, 128);"


def read_image_unicode(path: str, flags):
    if not path:
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_image_unicode(path: str, image: np.ndarray) -> bool:
    if image is None or not path:
        return False
    ext = os.path.splitext(path)[1] or ".bmp"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        encoded.tofile(path)
    except (FileNotFoundError, OSError):
        return False
    return True


class AOIInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI Defect Inspector")
        self.resize(1200, 850)

        self.settings = QSettings("DualIllumination", "AOIInspector")

        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Cursor: -")

        # Original BF / DF grayscale images (split from source)
        self.img_bf_original = None
        self.img_df_original = None

        # References used during computation (currently point to original)
        self.current_bf_gray = None
        self.current_df_gray = None

        # Grayscale images after preprocessing but before binarization
        self.current_bf_processed = None
        self.current_df_processed = None

        # Pixmaps used by viewers
        self.pixmap_bf = None
        self.pixmap_df = None
        self.pixmap_res = None

        # Latest BGR images for saving annotated views
        self.last_view_bf_bgr = None
        self.last_view_df_bgr = None
        self.last_view_res_bgr = None

        # Source filename and extension (used for output naming)
        self.last_input_name = "Output"
        self.last_input_ext = ".bmp"

        # Displayed image size
        self.disp_w = None
        self.disp_h = None

        # Ratio between displayed coordinates and original coordinates
        self.coord_scale_x = 1.0
        self.coord_scale_y = 1.0

        # Shared pan/zoom state
        self.view_state = {
            "scale": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }

        self._bf_from_slider = False
        self._df_from_slider = False

        self.spin_delay_timer = QTimer()
        self.spin_delay_timer.setSingleShot(True)
        self.spin_delay_timer.setInterval(500)
        self.spin_delay_timer.timeout.connect(self.update_result)

        # Timer used to restore status text after temporary states
        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.set_status_info)

        self.reset_view_next_update = False

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left side: three synchronized image viewers
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(2)

        def create_image_viewer(title_text, view_key):
            lbl_title = QLabel(title_text)
            lbl_title.setMaximumHeight(20)
            lbl_title.setStyleSheet("font-weight: bold;")
            self.image_layout.addWidget(lbl_title)

            viewer = SyncImageViewer(view_key, self.view_state, parent=self)
            viewer.setStyleSheet(BACKGROUND_COLOR_CSS)
            viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            viewer.mouse_info.connect(self.on_mouse_info)
            viewer.transform_changed.connect(self.on_transform_changed)

            self.image_layout.addWidget(viewer)
            return viewer

        self.viewer_bf = create_image_viewer("Bright Field (BF)", "BF")
        self.viewer_df = create_image_viewer("Dark Field (DF)", "DF")
        self.viewer_res = create_image_viewer(
            "Result (Red:Defect, Green:Particle, Yellow:Overlap)", "RES"
        )

        main_layout.addWidget(self.image_container, 1)

        # Right side: control panel
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setFixedWidth(260)

        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(15)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(60)
        control_layout.addWidget(self.status_label)

        # Operations group
        group_op = QGroupBox("Operations")
        layout_op = QVBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_image)
        layout_op.addWidget(self.btn_load)

        # Save buttons
        self.btn_save_bfdf = QPushButton("Save BF/DF Image")
        self.btn_save_bfdf.setMinimumHeight(40)
        self.btn_save_bfdf.clicked.connect(self.save_bf_df)
        layout_op.addWidget(self.btn_save_bfdf)

        self.btn_save_result = QPushButton("Save Result")
        self.btn_save_result.setMinimumHeight(40)
        self.btn_save_result.clicked.connect(self.save_result)
        layout_op.addWidget(self.btn_save_result)

        group_op.setLayout(layout_op)
        control_layout.addWidget(group_op)

        # BF Settings
        group_bf = QGroupBox("Bright Field Settings")
        layout_bf = QVBoxLayout()

        self.chk_bf = QCheckBox("Show Mask")
        self.chk_bf.setChecked(True)
        self.chk_bf.stateChanged.connect(self.update_result)
        layout_bf.addWidget(self.chk_bf)

        layout_bf.addSpacing(5)
        self.chk_bf_blur = QCheckBox("Enable Blur")
        self.chk_bf_blur.setChecked(False)
        self.chk_bf_blur.stateChanged.connect(self.update_result)
        layout_bf.addWidget(self.chk_bf_blur)

        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Kernel Size:"))
        self.spin_bf_ksize = QSpinBox()
        self.spin_bf_ksize.setRange(1, 31)
        self.spin_bf_ksize.setSingleStep(2)
        self.spin_bf_ksize.setValue(3)
        self.spin_bf_ksize.setFixedWidth(60)
        self.spin_bf_ksize.valueChanged.connect(self.delay_spin_update)
        blur_layout.addWidget(self.spin_bf_ksize)
        layout_bf.addLayout(blur_layout)

        layout_bf.addSpacing(5)
        layout_bf.addWidget(QLabel("Threshold:"))

        self.chk_bf_inverse = QCheckBox("Inverse Threshold")
        self.chk_bf_inverse.setChecked(False)
        self.chk_bf_inverse.stateChanged.connect(self.update_result)
        layout_bf.addWidget(self.chk_bf_inverse)

        bf_input_layout = QHBoxLayout()
        self.slider_bf = QSlider(Qt.Horizontal)
        self.slider_bf.setRange(0, 255)
        self.slider_bf.setValue(200)

        self.spin_bf = QSpinBox()
        self.spin_bf.setRange(0, 255)
        self.spin_bf.setValue(200)
        self.spin_bf.setFixedWidth(60)

        self.slider_bf.valueChanged.connect(self.on_slider_bf_changed)
        self.spin_bf.valueChanged.connect(self.on_spin_bf_changed)

        self.slider_bf.sliderReleased.connect(self.update_result)
        self.spin_bf.editingFinished.connect(self.delay_spin_update)

        bf_input_layout.addWidget(self.slider_bf)
        bf_input_layout.addWidget(self.spin_bf)
        layout_bf.addLayout(bf_input_layout)

        group_bf.setLayout(layout_bf)
        control_layout.addWidget(group_bf)

        # DF Settings
        group_df = QGroupBox("Dark Field Settings")
        layout_df = QVBoxLayout()

        self.chk_df = QCheckBox("Show Mask")
        self.chk_df.setChecked(True)
        self.chk_df.stateChanged.connect(self.update_result)
        layout_df.addWidget(self.chk_df)

        layout_df.addSpacing(5)
        layout_df.addWidget(QLabel("Threshold:"))

        self.chk_df_inverse = QCheckBox("Inverse Threshold")
        self.chk_df_inverse.setChecked(False)
        self.chk_df_inverse.stateChanged.connect(self.update_result)
        layout_df.addWidget(self.chk_df_inverse)

        df_input_layout = QHBoxLayout()
        self.slider_df = QSlider(Qt.Horizontal)
        self.slider_df.setRange(0, 255)
        self.slider_df.setValue(10)

        self.spin_df = QSpinBox()
        self.spin_df.setRange(0, 255)
        self.spin_df.setValue(10)
        self.spin_df.setFixedWidth(60)

        self.slider_df.valueChanged.connect(self.on_slider_df_changed)
        self.spin_df.valueChanged.connect(self.on_spin_df_changed)

        self.slider_df.sliderReleased.connect(self.update_result)
        self.spin_df.editingFinished.connect(self.delay_spin_update)

        df_input_layout.addWidget(self.slider_df)
        df_input_layout.addWidget(self.spin_df)
        layout_df.addLayout(df_input_layout)

        layout_df.addSpacing(8)
        layout_df.addWidget(QLabel("Dilated (DF only):"))

        dilate_layout = QHBoxLayout()
        lbl_ksize = QLabel("Kernel:")
        self.spin_df_ksize = QSpinBox()
        self.spin_df_ksize.setRange(1, 99)
        self.spin_df_ksize.setValue(3)
        self.spin_df_ksize.setFixedWidth(50)

        lbl_iter = QLabel("Iter:")
        self.spin_df_iter = QSpinBox()
        self.spin_df_iter.setRange(0, 20)
        self.spin_df_iter.setValue(1)
        self.spin_df_iter.setFixedWidth(50)

        self.spin_df_ksize.valueChanged.connect(self.delay_spin_update)
        self.spin_df_iter.valueChanged.connect(self.delay_spin_update)

        dilate_layout.addWidget(lbl_ksize)
        dilate_layout.addWidget(self.spin_df_ksize)
        dilate_layout.addWidget(lbl_iter)
        dilate_layout.addWidget(self.spin_df_iter)
        layout_df.addLayout(dilate_layout)

        group_df.setLayout(layout_df)
        control_layout.addWidget(group_df)

        control_layout.addStretch()
        main_layout.addWidget(control_panel, 0)

        # Initialize with Info status and enable buttons based on whether images are loaded
        self.set_status_info()

    # ---------- Status display & button state helpers ----------

    def update_buttons_state(self, info_state: bool):
        """
        info_state = True  : Info status (Ready, etc.), enable buttons based on image availability
        info_state = False : Non-info status (Loading / Processing / Saving / Error), disable all buttons
        """
        # Guard against early calls before init_ui creates buttons
        if not hasattr(self, "btn_load"):
            return

        if not info_state:
            # Non-info status: disable all buttons
            self.btn_load.setEnabled(False)
            self.btn_save_bfdf.setEnabled(False)
            self.btn_save_result.setEnabled(False)
            return

        # Info status: Load is always enabled
        self.btn_load.setEnabled(True)

        has_img = self.img_bf_original is not None and self.img_df_original is not None
        # Save buttons disabled until images are loaded
        self.btn_save_bfdf.setEnabled(has_img)

        has_result_view = (
            has_img
            and self.last_view_bf_bgr is not None
            and self.last_view_df_bgr is not None
            and self.last_view_res_bgr is not None
        )
        self.btn_save_result.setEnabled(has_result_view)

    def set_status_info(self, text="Ready"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "background-color: #D7F5C3;"
            "color: #111;"
            "font-weight: bold;"
            "font-size: 24px;"
            "padding: 6px;"
        )
        self.update_buttons_state(info_state=True)

    def set_status_warn(self, text):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "background-color: #FAFAD2;"
            "color: #222;"
            "font-weight: bold;"
            "font-size: 24px;"
            "padding: 6px;"
        )
        self.update_buttons_state(info_state=False)
        self.status_timer.start(1000)  # Automatically revert to Ready after 1 second

    def set_status_error(self, text="Error"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "background-color: #FAFAD2;"
            "color: red;"
            "font-weight: bold;"
            "font-size: 24px;"
            "padding: 6px;"
        )
        self.update_buttons_state(info_state=False)
        self.status_timer.start(3000)

    def load_settings(self):
        self.slider_bf.setValue(self.settings.value("thresholds/bf", 200, int))
        self.slider_df.setValue(self.settings.value("thresholds/df", 10, int))
        self.chk_bf_inverse.setChecked(self.settings.value("thresholds/bf_inverse", False, bool))
        self.chk_df_inverse.setChecked(self.settings.value("thresholds/df_inverse", False, bool))
        self.spin_df_ksize.setValue(self.settings.value("mask/df_ksize", 3, int))
        self.spin_df_iter.setValue(self.settings.value("mask/df_iter", 1, int))
        self.chk_bf.setChecked(self.settings.value("mask/show_bf", True, bool))
        self.chk_df.setChecked(self.settings.value("mask/show_df", True, bool))
        self.chk_bf_blur.setChecked(self.settings.value("mask/bf_blur_enabled", False, bool))
        self.spin_bf_ksize.setValue(self.settings.value("mask/bf_blur_ksize", 3, int))

        self.view_state["scale"] = self.settings.value("view/scale", self.view_state["scale"], float)
        self.view_state["center_x"] = self.settings.value("view/center_x", self.view_state["center_x"], float)
        self.view_state["center_y"] = self.settings.value("view/center_y", self.view_state["center_y"], float)

    def save_settings(self):
        self.settings.setValue("thresholds/bf", self.slider_bf.value())
        self.settings.setValue("thresholds/df", self.slider_df.value())
        self.settings.setValue("thresholds/bf_inverse", self.chk_bf_inverse.isChecked())
        self.settings.setValue("thresholds/df_inverse", self.chk_df_inverse.isChecked())
        self.settings.setValue("mask/df_ksize", self.spin_df_ksize.value())
        self.settings.setValue("mask/df_iter", self.spin_df_iter.value())
        self.settings.setValue("mask/show_bf", self.chk_bf.isChecked())
        self.settings.setValue("mask/show_df", self.chk_df.isChecked())
        self.settings.setValue("mask/bf_blur_enabled", self.chk_bf_blur.isChecked())
        self.settings.setValue("mask/bf_blur_ksize", self.spin_bf_ksize.value())
        self.settings.setValue("view/scale", self.view_state.get("scale", 1.0))
        self.settings.setValue("view/center_x", self.view_state.get("center_x", 0.0))
        self.settings.setValue("view/center_y", self.view_state.get("center_y", 0.0))

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    # ---------- Slider / SpinBox ----------

    @Slot(int)
    def on_slider_bf_changed(self, v):
        self._bf_from_slider = True
        self.spin_bf.setValue(v)
        self._bf_from_slider = False

    @Slot(int)
    def on_spin_bf_changed(self, v):
        if not self._bf_from_slider:
            self.delay_spin_update()

    @Slot(int)
    def on_slider_df_changed(self, v):
        self._df_from_slider = True
        self.spin_df.setValue(v)
        self._df_from_slider = False

    @Slot(int)
    def on_spin_df_changed(self, v):
        if not self._df_from_slider:
            self.delay_spin_update()

    def delay_spin_update(self):
        self.spin_delay_timer.start()

    # ---------- Load Image (3 modes) ----------

    def get_default_image_dir(self):
        images_dir = os.path.join(os.getcwd(), "images")
        if os.path.isdir(images_dir):
            return images_dir
        return os.getcwd()

    def set_last_input_file(self, path: str):
        """Record input filename and extension for saving results."""
        if not path:
            self.last_input_name = "Output"
            self.last_input_ext = ".bmp"
            return
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        self.last_input_name = name if name else "Output"
        self.last_input_ext = ext if ext else ".bmp"

    def load_image(self):
        dlg = LoadImageDialog(self, self.settings, default_dir=self.get_default_image_dir())
        if dlg.exec() != QDialog.Accepted:
            return

        dlg.save_settings()

        cfg = dlg.get_config()
        mode = cfg["mode"]

        self.set_status_warn("Loading...")
        QApplication.processEvents()

        img_bf_gray = None
        img_df_gray = None
        input_path_for_naming = None

        if mode == "time":
            path = cfg["file"]
            img_gray = read_image_unicode(path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                self.set_status_error("Load failed")
                return
            h, w = img_gray.shape
            mid = h // 2
            if cfg["bf_position"] == "Upper half":
                img_bf_gray = img_gray[0:mid, :]
                img_df_gray = img_gray[mid:h, :]
            else:
                img_bf_gray = img_gray[mid:h, :]
                img_df_gray = img_gray[0:mid, :]
            input_path_for_naming = path

        elif mode == "multi":
            path = cfg["file"]
            img_color = read_image_unicode(path, cv2.IMREAD_COLOR)
            if img_color is None:
                self.set_status_error("Load failed")
                return

            b, g, r = cv2.split(img_color)

            def ch_to_img(name):
                if name == "Red":
                    return r
                elif name == "Green":
                    return g
                else:
                    return b

            img_bf_gray = ch_to_img(cfg["bf_channel"])
            img_df_gray = ch_to_img(cfg["df_channel"])
            input_path_for_naming = path

        elif mode == "separate":
            path_bf = cfg["file_bf"]
            path_df = cfg["file_df"]

            img_bf_gray = read_image_unicode(path_bf, cv2.IMREAD_GRAYSCALE)
            img_df_gray = read_image_unicode(path_df, cv2.IMREAD_GRAYSCALE)

            if img_bf_gray is None or img_df_gray is None:
                self.set_status_error("Load failed")
                return

            if img_bf_gray.shape != img_df_gray.shape:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "BF and DF images have different sizes. Please select matched images.",
                )
                self.set_status_error("Size mismatch")
                return

            # Use the BF filename as the naming base
            input_path_for_naming = path_bf

        # Record input filename info
        self.set_last_input_file(input_path_for_naming)

        # Set input images and refresh displays
        self.img_bf_original = img_bf_gray
        self.img_df_original = img_df_gray

        self.reset_view_next_update = True
        self.update_result()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)

    # ---------- Core processing: invoke bright/dark field modules ----------

    def perform_calculation(self):
        if self.img_bf_original is None or self.img_df_original is None:
            return

        thresh_bf = self.spin_bf.value()
        thresh_df = self.spin_df.value()
        show_bf_mask = self.chk_bf.isChecked()
        show_df_mask = self.chk_df.isChecked()
        inverse_bf = self.chk_bf_inverse.isChecked()
        inverse_df = self.chk_df_inverse.isChecked()
        blur_bf = self.chk_bf_blur.isChecked()
        blur_ksize = self.spin_bf_ksize.value()

        ksize = self.spin_df_ksize.value()
        iters = self.spin_df_iter.value()

        self.set_status_warn("Processing...")
        QApplication.processEvents()

        self.current_bf_gray = self.img_bf_original
        self.current_df_gray = self.img_df_original
        self.current_bf_processed = None
        self.current_df_processed = None

        img_bf = self.current_bf_gray
        img_df = self.current_df_gray

        bf_processed, mask_bf, view_bf = process_bright_field(
            img_bf,
            thresh_bf=thresh_bf,
            show_mask=show_bf_mask,
            blur_enabled=blur_bf,
            blur_ksize=blur_ksize,
            inverse_threshold=inverse_bf,
        )

        self.current_bf_processed = bf_processed

        df_processed, mask_df_raw, mask_df_dilated, view_df = process_dark_field(
            img_df,
            thresh_df=thresh_df,
            show_mask=show_df_mask,
            ksize=ksize,
            iters=iters,
            inverse_threshold=inverse_df,
        )

        self.current_df_processed = df_processed

        view_res = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)

        if mask_bf is None or mask_df_dilated is None:
            self.update_display_pixmaps(view_bf, view_df, view_res)
            self.set_status_info()
            return

        if show_bf_mask and show_df_mask:
            mask_defect = cv2.bitwise_and(mask_bf, cv2.bitwise_not(mask_df_dilated))
            mask_common = cv2.bitwise_and(mask_bf, mask_df_dilated)
            mask_df_only = cv2.bitwise_and(mask_df_dilated, cv2.bitwise_not(mask_bf))

            view_res[mask_defect == 255] = [0, 0, 255]
            view_res[mask_df_only == 255] = [0, 255, 0]
            view_res[mask_common == 255] = [0, 255, 255]
        elif show_bf_mask and not show_df_mask:
            view_res[mask_bf == 255] = [0, 0, 255]
        elif not show_bf_mask and show_df_mask:
            view_res[mask_df_dilated == 255] = [0, 255, 0]

        self.update_display_pixmaps(view_bf, view_df, view_res)
        self.set_status_info()

    @Slot()
    def update_result(self):
        self.spin_delay_timer.stop()

        if self.img_bf_original is None:
            self.set_status_info()
            return
        self.perform_calculation()

    # ---------- Display / synchronized zoom ----------

    def update_display_pixmaps(self, view_bf_bgr, view_df_bgr, view_res_bgr):
        if view_bf_bgr is None or view_df_bgr is None or view_res_bgr is None:
            return

        self.disp_h, self.disp_w, _ = view_bf_bgr.shape

        # Keep BGR copies for saving annotated images
        self.last_view_bf_bgr = view_bf_bgr.copy()
        self.last_view_df_bgr = view_df_bgr.copy()
        self.last_view_res_bgr = view_res_bgr.copy()

        if self.current_bf_gray is not None:
            src_h, src_w = self.current_bf_gray.shape
            self.coord_scale_x = src_w / float(self.disp_w)
            self.coord_scale_y = src_h / float(self.disp_h)
        else:
            self.coord_scale_x = 1.0
            self.coord_scale_y = 1.0

        self.pixmap_bf = self.convert_cv_qt(view_bf_bgr)
        self.pixmap_df = self.convert_cv_qt(view_df_bgr)
        self.pixmap_res = self.convert_cv_qt(view_res_bgr)

        if self.reset_view_next_update:
            vw = self.viewer_bf.width()
            vh = self.viewer_bf.height()
            if vw > 0 and vh > 0:
                fit_scale = min(vw / float(self.disp_w), vh / float(self.disp_h))
            else:
                fit_scale = 1.0

            self.view_state["center_x"] = self.disp_w / 2.0
            self.view_state["center_y"] = self.disp_h / 2.0
            self.view_state["scale"] = fit_scale
            self.reset_view_next_update = False

        if self.viewer_bf is not None:
            self.viewer_bf.set_pixmap(self.pixmap_bf)
        if self.viewer_df is not None:
            self.viewer_df.set_pixmap(self.pixmap_df)
        if self.viewer_res is not None:
            self.viewer_res.set_pixmap(self.pixmap_res)

        self.on_transform_changed()

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        if not cv_img.flags['C_CONTIGUOUS']:
            cv_img = np.ascontiguousarray(cv_img)
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qt_img.copy())

    @Slot()
    def on_transform_changed(self):
        if self.viewer_bf:
            self.viewer_bf.update()
        if self.viewer_df:
            self.viewer_df.update()
        if self.viewer_res:
            self.viewer_res.update()

    @Slot(str, float, float, bool)
    def on_mouse_info(self, view_key, img_x, img_y, inside):
        if not inside:
            return
        if self.current_bf_gray is None:
            return

        src_x = int(img_x * self.coord_scale_x)
        src_y = int(img_y * self.coord_scale_y)

        if view_key == "BF":
            gray_img = (
                self.current_bf_processed
                if self.current_bf_processed is not None
                else self.current_bf_gray
            )
        elif view_key == "DF":
            gray_img = (
                self.current_df_processed
                if self.current_df_processed is not None
                else self.current_df_gray
            )
        else:
            gray_img = self.current_bf_gray

        if gray_img is None:
            return

        h, w = gray_img.shape
        if not (0 <= src_x < w and 0 <= src_y < h):
            return

        val = int(gray_img[src_y, src_x])
        zoom = self.view_state.get("scale", 1.0)

        self._status_bar.showMessage(
            f"View: {view_key} | X: {src_x}  Y: {src_y}  Gray: {val}  | Zoom: {zoom:.2f}x"
        )

    # ---------- Save images ----------

    def ensure_result_dir(self):
        result_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def save_bf_df(self):
        """
        Save the original split grayscale BF / DF images.
        Filenames: [input]_BF.[ext] and [input]_DF.[ext]
        """
        if self.img_bf_original is None or self.img_df_original is None:
            self.set_status_error("No image")
            return

        result_dir = self.ensure_result_dir()
        base = self.last_input_name
        ext = self.last_input_ext

        bf_path = os.path.join(result_dir, f"{base}_BF{ext}")
        df_path = os.path.join(result_dir, f"{base}_DF{ext}")

        save_image_unicode(bf_path, self.img_bf_original)
        save_image_unicode(df_path, self.img_df_original)

        self.set_status_warn("Saving...")

    def save_result(self):
        """
        Save three annotated images:
          1. BF viewer overlay: [input]_BF_Result.[ext]
          2. DF viewer overlay: [input]_DF_Result.[ext]
          3. Combined result overlay: [input]_Result.[ext]
        """
        if (
            self.last_view_bf_bgr is None
            or self.last_view_df_bgr is None
            or self.last_view_res_bgr is None
        ):
            self.set_status_error("No result image")
            return

        result_dir = self.ensure_result_dir()
        base = self.last_input_name
        ext = self.last_input_ext

        bf_mark_path = os.path.join(result_dir, f"{base}_BF_Result{ext}")
        df_mark_path = os.path.join(result_dir, f"{base}_DF_Result{ext}")
        res_path = os.path.join(result_dir, f"{base}_Result{ext}")

        save_image_unicode(bf_mark_path, self.last_view_bf_bgr)
        save_image_unicode(df_mark_path, self.last_view_df_bgr)
        save_image_unicode(res_path, self.last_view_res_bgr)

        self.set_status_warn("Saving...")
