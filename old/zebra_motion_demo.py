import sys
from PyQt6 import QtWidgets
from zebrafish.models.fish import ZebrafishLarva
from zebrafish.rendering.qt_renderer import FishRenderer

app=QtWidgets.QApplication(sys.argv)
fish=ZebrafishLarva()
win=FishRenderer(fish)
win.show()
sys.exit(app.exec())
