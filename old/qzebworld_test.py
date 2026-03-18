import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QWidget
from PyQt6.QtCore import QTimer, QRectF, QRect
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor

from qzeb_basic import *
from qzeb_qviz import *
from qzeb_zebra import *
from qzeb_seaworld import *  # Assuming this is converted too
WIDTH, HEIGHT = 1600, 1200


# Main Window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sea World")
        self.setGeometry(100, 100, WIDTH, HEIGHT)

        # Create a QGraphicsView
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, WIDTH, HEIGHT)

        # Create the SeaWorld scene
        self.scene = SeaWorld(WIDTH, HEIGHT)
        self.view.setScene(self.scene)

        # Load plants and planktons
        self.scene.load_sea_plants()
        self.scene.load_planktons()
        self.scene.add_zebrafishes()

        # Set up a timer to update the scene
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(60)  # Approximately 60 FPS

    def update_scene(self):
        self.scene.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())