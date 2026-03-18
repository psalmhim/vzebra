import sys
import math
from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

# -----------------------------------------------------
# Zebrafish Motor Agent
# -----------------------------------------------------
class Zebrafish:
    def __init__(self, x=0, y=0, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading   # radians
        self.speed = 1.0         # mm per step
        self.turn_rate = 0.15    # radians per key press

    def forward(self):
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)

    def turn_left(self):
        self.heading += self.turn_rate

    def turn_right(self):
        self.heading -= self.turn_rate


# -----------------------------------------------------
# Qt Window for Visualization
# -----------------------------------------------------
class Window(pg.GraphicsLayoutWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Zebrafish Rotation Test")

        self.view = self.addPlot()
        self.view.setAspectLocked(True)
        self.scatter = pg.ScatterPlotItem(size=15, brush=pg.mkBrush(50, 200, 255))
        self.view.addItem(self.scatter)

        self.fish = Zebrafish(x=0, y=0, heading=0.0)
        self.keys = set()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update_loop)
        timer.start(30)

    # -------------------------------------------------
    # Handle keys
    # -------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        self.keys.add(key)

    def keyReleaseEvent(self, event):
        key = event.key()
        if key in self.keys:
            self.keys.remove(key)

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def update_loop(self):

        # W → forward
        if QtCore.Qt.Key.Key_W in self.keys:
            self.fish.forward()

        # A → turn left
        if QtCore.Qt.Key.Key_A in self.keys:
            self.fish.turn_left()

        # D → turn right
        if QtCore.Qt.Key.Key_D in self.keys:
            self.fish.turn_right()

        # Update graphic
        self.scatter.setData([self.fish.x], [self.fish.y])


# -----------------------------------------------------
# Run
# -----------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())

