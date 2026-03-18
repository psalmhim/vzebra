import sys
import random
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsLineItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QTimer, QLineF, QRectF
from PyQt6.QtGui import QPen, QFont, QPixmap, QPainter, QColor
from qzeb_physio import ZPhysio

# Example usage in a PyQt application
class MainWindow(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Neural Firing Time Series")
        self.setGeometry(100, 100, 800, 600)

        # Create the neural activity scene with groups
        num_neurons = 50
        num_groups = 6  # Number of groups, each with a different color
        self.scene = ZPhysio(800, 600, num_neurons=num_neurons, duration=1000, num_groups=num_groups)
        self.setScene(self.scene)

        # Timer for updating the scene every 1 ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scene.simulate_firing)
        self.timer.start(1)  # Update every 1 ms


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
