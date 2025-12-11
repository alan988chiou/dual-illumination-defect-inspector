# main.py
import sys
from PySide6.QtWidgets import QApplication
from aoi_main_window import AOIInspector

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AOIInspector()
    window.show()
    sys.exit(app.exec())
