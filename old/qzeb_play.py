import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView
from PyQt6.QtCore import QTimer
from qzeb_basic import *
from qzeb_zebra import *
from qzeb_seaworld import *  # Assuming this is converted too

WIDTH=1600
HEIGHT=800
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sea World with PyQt6")
        self.setGeometry(100, 100, WIDTH, HEIGHT)

        # Initialize the QGraphicsView and the QGraphicsScene (SeaWorld)
        self.view = QGraphicsView(self)
        self.setCentralWidget(self.view)
        self.ocean_scene = SeaWorld(WIDTH, HEIGHT)
        self.view.setScene(self.ocean_scene)

        # Load sea plants and planktons into the scene
        self.ocean_scene.load_sea_plants()
        self.ocean_scene.load_planktons()
        self.ocean_scene.add_zebrafishes(3)

        # Timer for updating the scene
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(16)  # ~60 FPS

    def resizeEvent(self, event):
        new_size = self.size()
        width, height = new_size.width(), new_size.height()
        self.ocean_scene.update_window(width,height)

        # Call the parent class's resizeEvent to ensure default behavior
        super().resizeEvent(event)

    def update_scene(self):
        self.ocean_scene.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
