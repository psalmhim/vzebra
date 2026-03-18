import sys
import random
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import QTimer, QRectF, Qt, QPointF
from PyQt6.QtGui import QPixmap, QBrush, QTransform, QPen, QPainter, QColor, qAlpha
from PyQt6.QtWidgets import QGraphicsEllipseItem
from qzeb_basic import *
from qzeb_qviz import *
from qzeb_zebra import *
from PyQt6.QtCore import QRectF



class IObject(QGraphicsPixmapItem):
    def __init__(self, x=0, y=0, image=None, scale_factor=0.3):
        super().__init__()
        self.scale_factor = scale_factor
        self.original_pixmap = QPixmap(image)
        self.angle = 0
        self.x=x
        self.y=y
        self.id="IO0"
        self.update_pixmap()
        self.setPos(x, y)


    def update_pixmap(self):
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.scale_factor), 
            int(self.original_pixmap.height() * self.scale_factor), 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.setPixmap(scaled_pixmap)
        self.setTransformOriginPoint(self.pixmap().width() / 2, self.pixmap().height())
        self.rect=self.boundingRect()

    def boundingRect(self):
        scaled_pixmap = self.pixmap()
        return QRectF(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

    def update(self):
        pass


class Bubble(QGraphicsEllipseItem):
    def __init__(self, x, y, scale_factor=1.0, initial_radius=100):
        self.radius = initial_radius * scale_factor
        self.opacity = 1.0  # Start fully opaque

        # Set the initial bounding rect centered at (0, 0)
        super().__init__(QRectF(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius))

        # Set the pen and brush for the bubble
        self.setPen(QPen(Qt.GlobalColor.white, 2))  # Bubble outline
        self.setBrush(QBrush(Qt.GlobalColor.lightGray))  # Bubble fill
        
        # Position the bubble at the given (x, y) coordinates
        self.setPos(QPointF(x, y))

        # Velocity of the bubble (rising upwards)
        self.velocity = QPointF(0, -random.uniform(1, 5))

        #print(f"Bubble created at ({x}, {y}) with radius {self.radius}")

    def update(self):
        # Move the bubble upwards
        self.setPos(self.pos() + self.velocity)

        # Gradually fade out and shrink as it rises
        self.opacity -= 0.002  # Decrease opacity more slowly
        self.radius *= 0.995  # Gradually shrink the bubble more slowly

        # Update the bounding rect based on the new radius
        self.prepareGeometryChange()
        self.setRect(QRectF(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius))

        # Apply the opacity
        self.setOpacity(self.opacity)

        # Debugging output to track the bubble's state
        #print(f"Bubble updated: position=({self.pos().x()}, {self.pos().y()}), radius={self.radius}, opacity={self.opacity}")

        # Remove the bubble if it becomes invisible or too small
        if self.opacity <= 0 or self.radius < 5:
            #print("Bubble removed")
            self.scene().removeItem(self)  # Remove from the scene


# Example usage in SeaPlant or Plankton
class SeaPlant(IObject):
    def __init__(self, x, y, image, rotation_mode="random", scale_factor=0.3):
        super().__init__(x, y, image, scale_factor)
        self.rotation_mode = rotation_mode
        self.angle_direction = 1
        self.global_angle = 0
        self.global_direction = 1
        self.x=x
        self.y=y
        self.rect=QRectF(x,y,self.pixmap().width(),self.pixmap().height())
        self.id="SP0"

    def update(self):
        if self.rotation_mode == "random":
            self.angle += self.angle_direction * 0.5
            if abs(self.angle) >= 15:
                self.angle_direction *= -1
        elif self.rotation_mode == "synchronized":
            self.angle = self.global_angle

        transform = QTransform()
        transform.translate(self.transformOriginPoint().x(), self.transformOriginPoint().y())
        transform.rotate(self.angle)
        transform.translate(-self.transformOriginPoint().x(), -self.transformOriginPoint().y())
        self.setTransform(transform)

# Plankton class
class Plankton(IObject):
    def __init__(self, x, y, image, scale_factor=0.3):
        super().__init__(x, y, image, scale_factor)
        self.initial_position = QPointF(x, y)
        self.position = QPointF(x, y)
        self.velocity = QPointF(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        self.id="PL0"

    def update(self):
        # Random walk: add a small random vector to the current velocity
        self.velocity += QPointF(random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
        self.position += self.velocity
        distance_from_initial = (self.position - self.initial_position).manhattanLength()

        # Keep plankton within a range of 50 pixels from the initial position
        if distance_from_initial > 50:
            current_angle = math.atan2(self.velocity.y(), self.velocity.x())
            new_angle = current_angle + math.pi + math.radians(random.uniform(-45, 45))
            self.velocity.setX(math.cos(new_angle) * math.sqrt(self.velocity.x() ** 2 + self.velocity.y() ** 2))
            self.velocity.setY(math.sin(new_angle) * math.sqrt(self.velocity.x() ** 2 + self.velocity.y() ** 2))
            self.position += self.velocity  # Correct the position

        # Random rotation
        self.angle += random.uniform(-5, 5)
        transform = QTransform()
        transform.rotate(self.angle)
        self.setTransform(transform)
        self.setPos(self.position)


# SeaWorld class
class SeaWorld(QGraphicsScene):
    def __init__(self, width, height, scale_factor=0.2):
        super().__init__(0, 0, width, height)
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.plants_rotation_mode = "random"
        self.plants_global_angle = 0
        self.plants_global_direction = 1
        self.background_image = QPixmap("seaworld/background.png")
        self.sea_plants = []
        self.planktons = []
        self.bubbles = []
        self.zebrafishes = []
        self.update_background()

        self.display_area = None
        self.current_displayed_item = None
        self.create_display_area()

    def resizeEvent(self, event):
        new_size = event.size()
        self.width = new_size.width()
        self.height = new_size.height()
        self.update_background()
        super().resizeEvent(event)

    def update_background(self):
        background_pixmap = self.background_image.scaled(self.width, self.height, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.setBackgroundBrush(QBrush(background_pixmap))

    def create_display_area(self):
        """
        Create or update a rectangle at the top-right corner of the scene,
        with a width of 1/10th of the scene's width and a height proportional to the width.
        """
        # Remove the previous display area if it exists
        if self.display_area:
            self.removeItem(self.display_area)
        
        display_width = self.width / 10
        display_height = display_width  # Square area for now, change if needed
        display_x = self.width - display_width  # Top-right corner
        display_y = 0  # Top of the scene

        # Create a rectangle item for the display area
        self.display_area = QGraphicsRectItem(display_x, display_y, display_width, display_height)
        self.display_area.setBrush(QBrush(QColor(255, 255, 255, 150)))  # Semi-transparent white
        self.display_area.setPen(QColor(Qt.GlobalColor.black))  # Border color
        self.display_area.setZValue(10)  # Ensure it's above the background and other items

        self.addItem(self.display_area)
    
    def resize_event(self, new_width, new_height):
        """
        Handle the resizing of the scene.
        """
        # Update the scene's dimensions
        self.width = new_width
        self.height = new_height

        # Recreate the display area based on the new dimensions
        self.create_display_area()

        # Optionally, reposition or rescale the currently displayed item
        if self.current_displayed_item:
            self.reposition_displayed_item()

    def reposition_displayed_item(self):
        """
        Reposition the current displayed item to fit the new display area size.
        """
        display_rect = self.display_area.rect()

        # Recalculate position and scale for the displayed item
        display_x = display_rect.left()
        display_y = display_rect.top()

        self.current_displayed_item.setPos(display_x, display_y)

        scale_factor = min(display_rect.width() / self.current_displayed_item.boundingRect().width(),
                           display_rect.height() / self.current_displayed_item.boundingRect().height())
        self.current_displayed_item.setScale(scale_factor)

    def add_random_item_to_display_area(self):
        """
        Choose a random item from the sea plants or planktons and display it in the top-right display area.
        Remove the previous item displayed in the area.
        """
        if not self.sea_plants and not self.planktons:
            return  # No items to display

        # Remove the previous item from the display area, if it exists
        try:
            if self.current_displayed_item is not None:
                self.removeItem(self.current_displayed_item)
                self.current_displayed_item = None
        except Exception as e:
            print(f"Error removing item: {e}")

        # Choose a random item from either sea plants or planktons
        all_items = self.sea_plants + self.planktons
        selected_item = random.choice(all_items)

        # Get the bounding rectangle of the display area
        display_rect = self.display_area.rect()

        # Position the item inside the display area
        display_x = display_rect.left()
        display_y = display_rect.top()

        # Create a clone of the selected item (e.g., a QGraphicsPixmapItem)
        clone_item = QGraphicsPixmapItem(selected_item.pixmap())
        clone_item.setPos(display_x, display_y)

        # Scale the item to fit the display area (optional, depending on your use case)
        scale_factor = min(display_rect.width() / selected_item.boundingRect().width(),
                           display_rect.height() / selected_item.boundingRect().height())
        clone_item.setScale(scale_factor)

        # Set Z-value to ensure the cloned item is drawn on top
        clone_item.setZValue(11)

        # Add the cloned item to the scene
        self.addItem(clone_item)

        # Update the reference to the currently displayed item
        self.current_displayed_item = clone_item

    def load_sea_plants(self):
        sea_plant_images = [f"seaworld/seaplant-{i}.png" for i in range(1, 20)]
        for i in range(19):
            image = random.choice(sea_plant_images)
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            scale_factor = random.uniform(0.2, 0.5)  # Random scale factor between 0.2 and 0.5
            sea_plant = SeaPlant(x, y, image)#, scale_factor=scale_factor)
            sea_plant.id=f"SP{i}"
            self.addItem(sea_plant)
            self.sea_plants.append(sea_plant)

    def load_planktons(self):
        plankton_images = [f"seaworld/plankton-{i}.png" for i in range(1, 16)]
        for i in range(15):
            image = random.choice(plankton_images)
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            scale_factor = random.uniform(0.1, 0.3)  # Random scale factor between 0.1 and 0.3
            plankton = Plankton(x, y, image) # scale_factor=scale_factor)
            plankton.id=f"PL{i}"
            self.addItem(plankton)
            self.planktons.append(plankton)

    def create_bubble(self):
        x = random.randint(0, int(self.width))
        y = random.randint(int(self.height/2), int(self.height)) # Start the bubble at the bottom of the screen
        bubble = Bubble(x, y, scale_factor=random.uniform(0.1, 0.3))
        self.addItem(bubble)
        self.bubbles.append(bubble)

    def add_zebrafishes(self,number=3):
        for i in range(number):
            fish=ZFish(self,
                (random.randint(0, 0), random.randint(0,0)),
                (random.uniform(-1, 1), random.uniform(-1, 1)))
                #[random.uniform(1, 1), random.uniform(0, 0)],
                #[random.uniform(-1, -1), random.uniform(0, 0)], #left 
                #[random.uniform(0, 0), random.uniform(1, 1)], #left 
            id=f"ZID{i}"    
            fish.update_orig((self.width/2,self.height/2))  
            fish.update_property("ID",id)
            self.addItem(fish)
            self.zebrafishes.append(fish)

    def update_window(self,width,height):
        dx=width-self.width
        dy=height-self.height
        self.width = width
        self.height = height
        #self.scale_factor = self.scale_factor
        for zb in self.zebrafishes:
            zb.update_property("viz.orig",(self.width/2,self.height/2))
    
    def update(self):
        # Optionally create new bubbles
        if random.random() < 0.01:  # 5% chance to create a new bubble each frame
            self.create_bubble()

        for item in self.items():
            if isinstance(item, ZFish):
                item.advance(1)  # Call advance on Zebrafish to move it
            else:
                item.update()
