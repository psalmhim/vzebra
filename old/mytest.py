import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsItem, QGraphicsScene, QGraphicsEllipseItem
from PyQt6.QtGui import QPixmap, QBrush, QTransform, QPen, QPainter
from PyQt6.QtCore import QRectF, Qt, QPointF, QTimer
import random

class Zebrafish(QGraphicsItem):
    def __init__(self, x, y, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.head_radius = 20 * self.scale_factor
        self.body_length = 100 * self.scale_factor
        self.body_width = 20 * self.scale_factor
        self.tail_length = 50 * self.scale_factor
        self.tail_width = 10 * self.scale_factor

        # Set the initial position of the fish
        self.position = QPointF(x, y)
        self.setPos(self.position)

    def boundingRect(self) -> QRectF:
        # Define the bounding rectangle of the entire zebrafish, centered around (0, 0)
        # Make sure it includes the head circle
        total_length = self.head_radius * 2 + self.body_length + self.tail_length
        return QRectF(
            -self.head_radius,                # Extend to the left of the center to include the head
            -self.head_radius,                # Extend above the center to include the head
            total_length,                     # The total length of the fish including the head and tail
            self.head_radius * 2              # The total height including the head's diameter
        )

    def paint(self, painter: QPainter, option, widget=None):
        # Draw the head centered around (0, 0)
        painter.setBrush(QBrush(Qt.GlobalColor.blue))
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QPointF(0, 0), self.head_radius, self.head_radius)

        # Draw the body (rectangle)
        body_rect = QRectF(
            self.head_radius, 
            -self.body_width / 2, 
            self.body_length, 
            self.body_width
        )
        painter.setBrush(QBrush(Qt.GlobalColor.gray))
        painter.drawRect(body_rect)

        # Draw the tail (triangle)
        tail_points = [
            QPointF(self.head_radius + self.body_length, 0),
            QPointF(self.head_radius + self.body_length + self.tail_length, -self.tail_width / 2),
            QPointF(self.head_radius + self.body_length + self.tail_length, self.tail_width / 2)
        ]
        painter.setBrush(QBrush(Qt.GlobalColor.darkGray))
        painter.drawPolygon(tail_points)

    def advance(self, phase):
        if not phase:
            return
        # Update the position, movement, or any other dynamics
        # Simple horizontal movement example
        self.setPos(self.pos().x() + 1, self.pos().y())


class SeaWorld(QGraphicsScene):
    def __init__(self, width, height, scale_factor=0.2):
        super().__init__(0, 0, width, height)
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.plants_rotation_mode = "random"
        self.plants_global_angle = 0
        self.plants_global_direction = 1

        # Load the background image as a QPixmap
        background_pixmap = QPixmap("seaworld/background.png").scaled(width, height)

        # Set the background brush using the QPixmap wrapped in a QBrush
        self.setBackgroundBrush(QBrush(background_pixmap))

        self.sea_plants = []
        self.planktons = []
        self.bubbles = []
        self.zebrafishes = []

        # Calculate the center of the scene
        center_x = self.width / 2
        center_y = self.height / 2

        # Add a test circle to the scene at the center
        self.add_test_circle(center_x, center_y)

    def add_test_circle(self, center_x, center_y):
        # Add a simple circle to the scene to test if it appears
        circle = TestCircle(center_x, center_y, radius=50)
        self.addItem(circle)
        print(f"Test circle added at center ({center_x}, {center_y}).")

    def load_sea_plants(self):
        sea_plant_images = [f"seaworld/seaplant-{i}.png" for i in range(1, 20)]
        for _ in range(19):
            image = random.choice(sea_plant_images)
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            scale_factor = random.uniform(0.2, 0.5)  # Random scale factor between 0.2 and 0.5
            sea_plant = SeaPlant(x, y, image, scale_factor=scale_factor)
            self.addItem(sea_plant)
            self.sea_plants.append(sea_plant)

    def load_planktons(self):
        plankton_images = [f"seaworld/plankton-{i}.png" for i in range(1, 16)]
        for _ in range(15):
            image = random.choice(plankton_images)
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            scale_factor = random.uniform(0.1, 0.3)  # Random scale factor between 0.1 and 0.3
            plankton = Plankton(x, y, image, scale_factor=scale_factor)
            self.addItem(plankton)
            self.planktons.append(plankton)

    def create_bubble(self):
        x = random.randint(0, int(self.width))
        y = random.randint(int(self.height / 2), int(self.height))  # Start the bubble at the bottom of the screen
        bubble = Bubble(x, y, scale_factor=random.uniform(0.1, 0.3))
        self.addItem(bubble)
        self.bubbles.append(bubble)

    def add_zebrafish(self, x, y):
        fish = Zebrafish(x, y, scale_factor=0.5)
        self.addItem(fish)
        self.zebrafishes.append(fish)

    def update(self):
        # Optionally create new bubbles
        if random.random() < 0.01:  # 1% chance to create a new bubble each frame
            self.create_bubble()

        # Update all items in the scene
        for item in self.items():
            if isinstance(item, Zebrafish):
                item.advance(1)  # Call advance on Zebrafish to move it
            else:
                item.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the view and the scene
        self.view = QGraphicsView()
        self.setCentralWidget(self.view)
        self.scene = SeaWorld(800, 600)
        self.view.setScene(self.scene)

        # Set window size
        self.resize(800, 600)

        # Set up a timer to update the scene
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(16)  # Approximately 60 FPS

    def update_scene(self):
        self.scene.advance()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
