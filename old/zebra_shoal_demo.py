import sys
from PyQt6 import QtWidgets
from zebrafish.models.shoal import Shoal
from zebrafish.rendering.shoal_renderer import ShoalRenderer

app = QtWidgets.QApplication(sys.argv)

shoal = Shoal(N=12)
win = ShoalRenderer(shoal)
win.show()

sys.exit(app.exec())
