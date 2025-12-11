from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QDialog, QTabWidget, QComboBox, QLineEdit, QMessageBox
)


class LoadImageDialog(QDialog):
    """
    Load Image 設定視窗：
    3 種模式：
      1. Time-division：一張圖垂直分上下半部
      2. Multispectral：一張彩色圖依通道分明/暗場
      3. Separate：明/暗場各讀一張圖
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Image Mode")
        self.resize(520, 260)

        self.tab_widget = QTabWidget(self)

        # --- Time-division tab ---
        self.tab_time = QWidget()
        self._init_tab_time()

        # --- Multispectral tab ---
        self.tab_multi = QWidget()
        self._init_tab_multi()

        # --- Separate tab ---
        self.tab_separate = QWidget()
        self._init_tab_separate()

        self.tab_widget.addTab(self.tab_time, "Time-division")
        self.tab_widget.addTab(self.tab_multi, "Multispectral")
        self.tab_widget.addTab(self.tab_separate, "Separate")

        # Bottom buttons
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept_clicked)
        btn_cancel.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)
        main_layout.addLayout(btn_layout)

    # ---------- Time-division tab ----------
    def _init_tab_time(self):
        layout = QVBoxLayout(self.tab_time)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Bright Field position:"))
        self.combo_time_bf_pos = QComboBox()
        self.combo_time_bf_pos.addItems(["Upper half", "Lower half"])
        mode_layout.addWidget(self.combo_time_bf_pos)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Image file:"))
        self.edit_time_file = QLineEdit()
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_time_file)
        file_layout.addWidget(self.edit_time_file)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        layout.addStretch()

    def browse_time_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open Time-division Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)"
        )
        if fn:
            self.edit_time_file.setText(fn)

    # ---------- Multispectral tab ----------
    def _init_tab_multi(self):
        layout = QVBoxLayout(self.tab_multi)

        # Channel selection
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("BF Channel:"))
        self.combo_multi_bf_ch = QComboBox()
        # 顯示 R/G/B，實際會對應到 OpenCV 的 BGR index
        self.combo_multi_bf_ch.addItems(["Red", "Green", "Blue"])

        ch_layout.addWidget(self.combo_multi_bf_ch)

        ch_layout.addSpacing(10)
        ch_layout.addWidget(QLabel("DF Channel:"))
        self.combo_multi_df_ch = QComboBox()
        self.combo_multi_df_ch.addItems(["Green", "Red", "Blue"])
        ch_layout.addWidget(self.combo_multi_df_ch)
        ch_layout.addStretch()
        layout.addLayout(ch_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Image file:"))
        self.edit_multi_file = QLineEdit()
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_multi_file)
        file_layout.addWidget(self.edit_multi_file)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        layout.addStretch()

    def browse_multi_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open Multispectral Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)"
        )
        if fn:
            self.edit_multi_file.setText(fn)

    # ---------- Separate tab ----------
    def _init_tab_separate(self):
        layout = QVBoxLayout(self.tab_separate)

        # BF file
        bf_layout = QHBoxLayout()
        bf_layout.addWidget(QLabel("BF Image:"))
        self.edit_sep_bf = QLineEdit()
        btn_browse_bf = QPushButton("Browse...")
        btn_browse_bf.clicked.connect(self.browse_sep_bf)
        bf_layout.addWidget(self.edit_sep_bf)
        bf_layout.addWidget(btn_browse_bf)
        layout.addLayout(bf_layout)

        # DF file
        df_layout = QHBoxLayout()
        df_layout.addWidget(QLabel("DF Image:"))
        self.edit_sep_df = QLineEdit()
        btn_browse_df = QPushButton("Browse...")
        btn_browse_df.clicked.connect(self.browse_sep_df)
        df_layout.addWidget(self.edit_sep_df)
        df_layout.addWidget(btn_browse_df)
        layout.addLayout(df_layout)

        layout.addStretch()

    def browse_sep_bf(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open BF Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)"
        )
        if fn:
            self.edit_sep_bf.setText(fn)

    def browse_sep_df(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open DF Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)"
        )
        if fn:
            self.edit_sep_df.setText(fn)

    # ---------- OK / 取得設定 ----------

    def accept_clicked(self):
        # 簡單檢查必填欄位
        mode = self.current_mode()
        if mode == "time":
            if not self.edit_time_file.text():
                QMessageBox.warning(self, "Warning", "Please select an image file.")
                return
        elif mode == "multi":
            if not self.edit_multi_file.text():
                QMessageBox.warning(self, "Warning", "Please select an image file.")
                return
        elif mode == "separate":
            if not self.edit_sep_bf.text() or not self.edit_sep_df.text():
                QMessageBox.warning(self, "Warning", "Please select BF and DF image files.")
                return

        self.accept()

    def current_mode(self):
        idx = self.tab_widget.currentIndex()
        if idx == 0:
            return "time"
        elif idx == 1:
            return "multi"
        else:
            return "separate"

    def get_config(self):
        """
        回傳 dict:
          mode: "time" / "multi" / "separate"
          依模式附上對應參數
        """
        mode = self.current_mode()
        if mode == 0 or mode == "time":
            mode = "time"
        elif mode == 1 or mode == "multi":
            mode = "multi"
        else:
            mode = "separate"

        if mode == "time":
            return {
                "mode": "time",
                "file": self.edit_time_file.text(),
                "bf_position": self.combo_time_bf_pos.currentText(),  # "Upper half" or "Lower half"
            }
        elif mode == "multi":
            return {
                "mode": "multi",
                "file": self.edit_multi_file.text(),
                "bf_channel": self.combo_multi_bf_ch.currentText(),  # "Red"/"Green"/"Blue"
                "df_channel": self.combo_multi_df_ch.currentText(),
            }
        else:
            return {
                "mode": "separate",
                "file_bf": self.edit_sep_bf.text(),
                "file_df": self.edit_sep_df.text(),
            }