import sys
import random
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsLineItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QTimer, QLineF, QRectF
from PyQt6.QtGui import QPen, QFont, QPixmap, QPainter, QColor

class ZPhysio(QGraphicsScene):
    def __init__(self, width, height, num_neurons, duration, num_groups):
        super().__init__(0, 0, width, height)
        self.width = width
        self.height = height
        self.num_neurons = num_neurons
        self.duration = duration
        self.num_groups = num_groups
        self.group_colors = self.assign_group_colors()
        self.update_scene()
        # Initialize real-time firing data
        self.current_time = 0  # Track the current time in the simulation
        self.current_x = 0  # Track the current x position for drawing spikes
        self.add_static_labels()  # Add title, x-axis, and y-axis labels


    def resizeEvent(self, event):
        new_size = event.size()
        self.width = new_size.width()
        self.height = new_size.height()
        self.update_scene()
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.IgnoreAspectRatio)


    def update_scene(self):
        self.setSceneRect(0, 0, self.width, self.height)
        self.setBackgroundBrush(Qt.GlobalColor.white)
        self.pixmap = QPixmap(self.width, self.height)  # To hold the spiking plot
        self.pixmap.fill(Qt.GlobalColor.white)
        self.neuron_height = self.height  / self.num_neurons
        self.spike_length = self.neuron_height / 2

    def assign_group_colors(self):
        """
        Assign a color to each neuron group.
        """
        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                  QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]
        group_colors = [colors[i % len(colors)] for i in range(self.num_groups)]
        return group_colors

    def get_group_color(self, neuron_id):
        """
        Get the color for the neuron based on its group.
        """
        group_id = neuron_id % self.num_groups
        return self.group_colors[group_id]

    def add_static_labels(self):
        """
        Add a title, x-axis label, and y-axis neuron labels to the scene.
        """
        # Add title
        title = QGraphicsTextItem("Real-Time Neural Firing Time Series")
        title.setDefaultTextColor(Qt.GlobalColor.white)
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setPos(self.width / 2 - title.boundingRect().width() / 2, -40)  # Position above the plot
        self.addItem(title)

        # Add x-axis label
        x_label = QGraphicsTextItem("Time (ms)")
        x_label.setDefaultTextColor(Qt.GlobalColor.white)
        x_label.setFont(QFont("Arial", 12))
        x_label.setPos(self.width / 2 - x_label.boundingRect().width() / 2, self.height + 10)  # Position below the plot
        self.addItem(x_label)

        # Add y-axis labels for neuron numbers
        font = QFont("Arial", 10)
        for neuron_id in range(self.num_neurons):
            y = neuron_id * self.neuron_height + self.neuron_height / 2
            label = QGraphicsTextItem(f"{neuron_id + 1}")
            label.setDefaultTextColor(Qt.GlobalColor.white)
            label.setFont(font)
            label.setPos(-30, y - label.boundingRect().height() / 2)  # Adjust position
            self.addItem(label)

    def simulate_firing(self):
        """
        Simulate random firing for each neuron in real-time.
        """
        self.current_time += 1  # Advance time by 1 ms

        # Check if we need to shift the pixmap or draw at the current position
        if self.current_x < self.width:
            # Draw spikes at the current position
            painter = QPainter(self.pixmap)
            try:
                for neuron_id in range(self.num_neurons):
                    # Each neuron has a small chance of firing (e.g., 1% chance per ms)
                    if random.random() < 0.1:
                        color = self.get_group_color(neuron_id)
                        painter.setPen(QPen(color))
                        self.draw_spike(painter, neuron_id, self.current_x)
            finally:
                painter.end()                
            painter.end()
            self.current_x += 1  # Move to the next x position
        else:
            # Shift the pixmap content to the left
            self.shift_pixmap_left()

            # Draw new spikes in the rightmost column
            painter = QPainter(self.pixmap)
            try:
                for neuron_id in range(self.num_neurons):
                    if random.random() < 0.01:
                        color = self.get_group_color(neuron_id)
                        painter.setPen(QPen(color))
                        self.draw_spike(painter, neuron_id, self.width - 1)
            finally:
                painter.end()             
            painter.end()

        # Update the scene with the new pixmap
        self.addPixmap(self.pixmap)

    def shift_pixmap_left(self):
        """
        Shift the pixmap content to the left by 1 pixel.
        """
        painter = QPainter(self.pixmap)
        try:
            target_rect = QRectF(0, 0, self.width - 1, self.height)
            source_rect = QRectF(1, 0, self.width - 1, self.height)
            painter.drawPixmap(target_rect, self.pixmap, source_rect)
            painter.fillRect(QRectF(self.width - 1, 0, 1, self.height), Qt.GlobalColor.black)  # Clear the rightmost column
        finally:    
            painter.end()
        painter.end()    

    def draw_spike(self, painter, neuron_id, x_pos):
        """
        Draw a spike on the pixmap for the given neuron at the specified x position.
        """
        y = neuron_id * self.neuron_height + self.neuron_height / 2  # Vertical position of the neuron
        # Draw a vertical line to represent the spike
        painter.drawLine(QLineF(x_pos, y - self.spike_length / 2, x_pos, y + self.spike_length / 2))

    def update(self):
        self.simulate_firing()
