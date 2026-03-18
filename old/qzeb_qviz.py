from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QBrush, QPen, QPolygonF
from PyQt6.QtCore import Qt, QRectF, QPointF
import sys
import numpy as np
from qzeb_basic import *

class QViz:
    def __init__(self):
        pass

    @staticmethod
    def rect_to_qrectf(rect):
        """Convert a tuple (x, y, width, height) to QRectF."""
        x, y, width, height = rect
        return QRectF(x, y, width, height)

    @staticmethod
    def get_rotated_point(point, center, angle_rad):
        """Rotate a point around a center by the given angle in radians."""
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        translated_point = point - center

        rotated_x = translated_point.x() * cos_theta - translated_point.y() * sin_theta
        rotated_y = translated_point.x() * sin_theta + translated_point.y() * cos_theta

        return QPointF(rotated_x + center.x(), rotated_y + center.y())

    @staticmethod
    def get_half_ellipse_points(center, width, height, upper=True, num_points=50, angle=0):
        """Generate points for half an ellipse (upper or lower) with rotation."""
        points = []
        a = width / 2  # Semi-major axis (half of the width)
        b = height / 2  # Semi-minor axis (half of the height)
        theta_range = np.linspace(0, np.pi, num_points) if upper else np.linspace(np.pi, 2 * np.pi, num_points)
        
        angle_rad = np.radians(angle)  # Convert angle to radians

        for theta in theta_range:
            x = center.x() + a * np.cos(theta)
            y = center.y() - b * np.sin(theta) if upper else center.y() - b * np.sin(theta)
            point = QPointF(x, y)
            # Apply rotation to each point
            rotated_point = QViz.get_rotated_point(point, center, angle_rad)
            points.append(rotated_point)

        return points

    @staticmethod
    def get_ellipse_arc_points(center, width, height, start_angle, end_angle, num_points=50, rotation_angle=0):
        """Generate points along an elliptical arc with rotation."""
        points = []
        a = width / 2  # Semi-major axis (half of the width)
        b = height / 2  # Semi-minor axis (half of the height)

        # Convert angles from degrees to radians
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)
        rotation_rad = np.radians(rotation_angle)

        # Generate points along the arc
        for theta in np.linspace(start_rad, end_rad, num_points):
            x = center.x() + a * np.cos(theta)
            y = center.y() + b * np.sin(theta)
            point = QPointF(x, y)
            # Apply rotation to each point
            rotated_point = QViz.get_rotated_point(point, center, rotation_rad)
            points.append(rotated_point)

        return points

    @staticmethod
    def draw_rotated_ellipse(painter, rect, width, height, angle, line_color=Qt.GlobalColor.black, fill_color=Qt.GlobalColor.yellow, line_width=3):
        """Draw an ellipse centered within the rect, rotated by the specified angle."""
        # Convert the tuple rect to QRectF
        qrect = QViz.rect_to_qrectf(rect)
        
        # Calculate the center of the rectangle
        center = QPointF(qrect.center())

        # Create the QRectF for the ellipse centered in the rect
        ellipse_rect = QRectF(
            center.x() - width / 2,
            center.y() - height / 2,
            width,
            height
        )

        # Apply rotation around the center
        painter.translate(center)  # Move the origin to the center of the ellipse
        painter.rotate(angle)      # Rotate the coordinate system
        painter.translate(-center) # Move back to the original coordinate system

        # Set pen and brush for the ellipse
        painter.setPen(QPen(line_color, line_width))
        painter.setBrush(QBrush(fill_color))

        # Draw the ellipse
        painter.drawEllipse(ellipse_rect)

    @staticmethod
    def draw_rotated_elliptical_polygon(painter, rect, width, upper_height, lower_height, angle, line_color=Qt.GlobalColor.black, fill_color=Qt.GlobalColor.green, line_width=3):
        """Draw a closed polygon composed of two half ellipsoids with the same width but different heights for upper and lower parts, rotated by the specified angle."""
        # Convert the tuple rect to QRectF
        qrect = QViz.rect_to_qrectf(rect)
        
        # Calculate the center of the rectangle
        center = QPointF(qrect.center())

        # Generate the upper half ellipse points
        upper_points = QViz.get_half_ellipse_points(center, width, upper_height, upper=True, angle=angle)

        # Generate the lower half ellipse points
        lower_points = QViz.get_half_ellipse_points(center, width, lower_height, upper=False, angle=angle)

        # Combine the points to form a closed polygon
        points = upper_points + lower_points

        # Create a polygon from these points
        polygon = QPolygonF(points)

        # Set pen and brush for the polygon
        painter.setPen(QPen(line_color, line_width))
        painter.setBrush(QBrush(fill_color))

        # Draw the polygon
        painter.drawPolygon(polygon)

    @staticmethod
    def draw_elliptical_arc(painter, rect, width, height, start_angle, end_angle, rotation_angle=0, line_color=Qt.GlobalColor.black, fill_color=Qt.GlobalColor.green, line_width=3):
        """Draw a partial ellipse (arc) defined by start and end angles with rotation."""
        # Convert the tuple rect to QRectF
        qrect = QViz.rect_to_qrectf(rect)
        
        # Calculate the center of the rectangle
        center = QPointF(qrect.center())

        # Generate the points along the ellipse arc
        arc_points = QViz.get_ellipse_arc_points(center, width, height, start_angle, end_angle, rotation_angle=rotation_angle)

        # Create a polygon from these points
        polygon = QPolygonF(arc_points)

        # Set pen and brush for the polygon
        painter.setPen(QPen(line_color, line_width))
        painter.setBrush(QBrush(fill_color))

        # Draw the arc (polygon)
        painter.drawPolygon(polygon)



