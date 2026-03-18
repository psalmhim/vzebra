import sys
import numpy as np
import random
import math

from PyQt6.QtGui import QColor, QPolygonF
from PyQt6.QtCore import QPointF, QRectF

class Color:
    COLORS = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "lightblue": (173, 216, 230),
        "oceanblue": (28, 107, 160),
    }

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    LIGHTBLUE = (173, 216, 230)
    OCEANBLUE = (28, 107, 160)

    @staticmethod
    def to_rgb(color):
        if isinstance(color, tuple) and len(color) == 3:
            return tuple(int(c * 255) if isinstance(c, float) else c for c in color)
        elif isinstance(color, str):
            return Color.COLORS.get(color.lower(), Color.COLORS["black"])
        else:
            return Color.COLORS["black"]

    @staticmethod
    def add_alpha(col, alpha):
        """Add an alpha value to an RGB tuple or QColor."""
        if isinstance(col, tuple) and len(col) == 3:
            # If it's an RGB tuple, return a QColor with the added alpha
            return QColor(col[0], col[1], col[2], alpha)
        elif isinstance(col, QColor):
            # If it's already a QColor, just set the alpha channel
            col.setAlpha(alpha)
            return col
        else:
            # Default to black if the input is not recognized
            return QColor(0, 0, 0, alpha)

    @staticmethod
    def to_name(rgb):
        rgb = tuple(int(c * 255) if isinstance(c, float) else c for c in rgb)
        for name, value in Color.COLORS.items():
            if value == rgb:
                return name
        return "unknown"
    
    @staticmethod
    def to_qcolor(color):
        """Convert a color name, RGB tuple, or hex string to a QColor."""
        if isinstance(color, QColor):
            return color
        elif isinstance(color, tuple):
            return QColor(*color)
        elif isinstance(color, str):
            if color.lower() in Color.COLORS:
                return QColor(*Color.COLORS[color.lower()])
            else:
                return QColor(color)  # This handles hex strings like "#RRGGBB"
        else:
            return QColor(*Color.COLORS["black"])

    @staticmethod
    def from_qcolor(qcolor):
        """Convert a QColor to an RGB tuple."""
        return (qcolor.red(), qcolor.green(), qcolor.blue())

class Viz:
    def __init__(
        self,
        linecolor="black",
        facecolor=None,
        linewidth=1,
        linetype="-",
        scale=2,
        orig=(800, 600),
        on=True
    ):
        self.linecolor = Color.to_qcolor(linecolor)
        self.facecolor = Color.to_qcolor(facecolor) if facecolor else QColor(0, 0, 0, 0)  # Transparent by default
        self.linewidth = linewidth
        self.linetype = linetype
        self.orig = orig
        self.scale = scale
        self.on = on
        
    def copy(self):
        # Create a shallow copy of the instance
        new_copy = Viz(self.linecolor, self.facecolor, self.linewidth, self.linetype, self.scale, self.orig, self.on)
        return new_copy

    @staticmethod
    def get_arc_points_rect(rect, start_angle, end_angle, angle, num_points=50):
        start_angle = np.radians(start_angle)
        end_angle = np.radians(end_angle)
        angle = np.radians(angle)
        points = []
        # Calculate the center of the ellipse
        x_center = rect.left() + rect.width() / 2
        y_center = rect.top() + rect.height() / 2

        # Calculate the horizontal and vertical radii
        rwidth = rect.width() / 2  # Horizontal radius
        rheight = rect.height() / 2  # Vertical radius

        return Viz.get_arc_points(
            (x_center, y_center),
            rwidth,
            rheight,
            start_angle,
            end_angle,
            angle,
            num_points,
        )

    @staticmethod
    def get_arc_points(center, rwidth, rheight, start_angle, end_angle, angle=None, num_points=50):
        angle = angle if angle else (start_angle + end_angle) / 2
        start_angle = np.radians(start_angle)
        end_angle = np.radians(end_angle)
        angle = np.radians(angle)
        points = []
        # Calculate the center of the ellipse
        x_center = center[0]
        y_center = center[1]

        # Iterate over the angle range
        for theta in np.linspace(start_angle, end_angle, num_points):
            x = rwidth * np.cos(theta)
            y = rheight * np.sin(theta)

            # Rotate the point around the ellipse center
            rotated_x = x * np.cos(angle) - y * np.sin(angle)
            rotated_y = x * np.sin(angle) + y * np.cos(angle)

            # Translate to the ellipse center
            point = (x_center + rotated_x, y_center + rotated_y)
            points.append(point)

        return points

    @staticmethod
    def rotate_point(point, angle, center):
        """Rotates a point around a given center."""
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x, y = point[0] - center[0], point[1] - center[1]
        x_new = x * cos_angle - y * sin_angle
        y_new = x * sin_angle + y * cos_angle
        return (x_new + center[0], y_new + center[1])

    @staticmethod
    def get_ellipse_points(rect, num_points=100, angle=0):
        """Generates points around the perimeter of an ellipse within a rectangle."""
        points = []
        center = (rect.left + rect.width / 2, rect.top + rect.height / 2)
        a = rect.width / 2  # Semi-major axis (half of the rect width)
        b = rect.height / 2  # Semi-minor axis (half of the rect height)
        for theta in np.linspace(0, 2 * np.pi, num_points):
            x = center[0] + a * np.cos(theta)
            y = center[1] + b * np.sin(theta)
            points.append((x, y))
        if angle:
            points = [Viz.rotate_point(point, angle, center) for point in points]
    
        return points



class Morphology:
    def __init__(self, type=None, pos=(0, 0), direction=[1, 0], width=0, height=0, radius=0, nodes=[]):
        self.pos = pos
        self.direction = direction
        self.type = type
        self.width = width
        self.height = height
        self.nodes = nodes
        self.radius = radius
        self.angle=0
        

    def norm(self, direction=None):
        self.direction = direction if direction else self.direction
        self.direction = Morphology.normalize(self.direction)
  
    @staticmethod     
    def normalize(v):
        v_length = np.sqrt(v[0] ** 2 + v[1] ** 2)
        return v if v_length == 1 else (v[0] / v_length, v[1] / v_length)

    @staticmethod    
    def negative_direction(v):
        return (-v[0], -v[1])

    @staticmethod
    def calculate_angle(v):
        dx, dy = v
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def left_vertical_direction(v):
        dx, dy = v
        return (-dy, dx)

    @staticmethod
    def right_vertical_direction(v):
        dx, dy = v
        return (dy, -dx)

    @staticmethod
    def perpendicular_points(pos, direction, offset, offpos=0):
        v = Morphology.normalize(direction)
        offz = (pos[0] + v[0] * offpos, pos[1] + v[1] * offpos)
        pv1 = (-v[1], v[0])
        pv2 = (v[1], -v[0])
        point1 = (offz[0] + pv1[0] * offset, offz[1] + pv1[1] * offset)
        point2 = (offz[0] + pv2[0] * offset, offz[1] + pv2[1] * offset)
        return point1, point2

    @staticmethod
    def point_along_direction(rpos, direction, length, anglerad=0):
        v = Morphology.normalize(direction)
        if anglerad == 0:
            rv = (
                v[0] * np.cos(anglerad) - v[1] * np.sin(anglerad),
                v[0] * np.sin(anglerad) + v[1] * np.cos(anglerad),
            )
        else:
            rv = v
        return (rpos[0] + rv[0] * length, rpos[1] + rv[1] * length)

    @staticmethod
    def rotate_point(point, anglerad, center):
        """Rotate a point around a center by a given angle."""
        px, py = point
        cx, cy = center
        s, c = np.sin(anglerad), np.cos(anglerad)
        px -= cx
        py -= cy
        new_x = px * c - py * s
        new_y = px * s + py * c
        return (new_x + cx, new_y + cy)
    
    @staticmethod
    def rotate_vector(v, anglerad):
        cos_alpha = np.cos(anglerad)
        sin_alpha = np.sin(anglerad)
        vx, vy = v
        rotated_vx = vx * cos_alpha - vy * sin_alpha
        rotated_vy = vx * sin_alpha + vy * cos_alpha
        return (rotated_vx, rotated_vy)


class State:
    def __init__(self, state=0):
        self.state = state


class Object:
    def __init__(self, parent=None,morph=None, viz=None, state=None):
        self.morph = morph if morph else Morphology()
        self.viz = viz if viz else Viz()
        self.state = state if state else State()

        self.children=[]
        self.parent=parent
        self.verbose=True
        self.ID=''

    #update_property("viz.orig", orig)
    #update_property("ID", id)
    def update_property(self, propname, propval): 
        if '.' in propname:
            props = propname.split('.')
            obj = self
            for p in props[:-1]:  # Go through all but the last element
                obj = getattr(obj, p)
            setattr(obj, props[-1], propval)
        else:
            setattr(self, propname, propval)

        for child in self.children:
            child.update_property(propname,propval)

    def update_orig(self, new_orig):
        self.viz.orig = new_orig
        for child in self.children:
            child.update_orig(new_orig)

    def update_id(self, ID):
        self.ID = ID
        for child in self.children:
            child.update_id(ID)

    def update_verbose(self, flag):
        self.verbose = False
        for child in self.children:
            child.verbose(flag)

    @staticmethod
    def center(pos1, pos2):
        return ((pos1[0]+pos2[0])/2,(pos1[1]+pos2[1])/2)
    
    def get_angle(self):
        dx, dy = self.morph.direction
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad)
    
    def center_iscreen(self, pos1, pos2):
        return self.center(self.pos_to_iscreen(pos1), self.pos_to_iscreen(pos2))
    
    def pos_to_iscreen(self, pos=None, orig=None):
        pos = pos if pos is not None else self.morph.pos
        orig = orig if orig is not None else self.viz.orig
        return (
            pos[0] * self.viz.scale + orig[0],
            -pos[1] * self.viz.scale + orig[1],
        )
    
    def rel_pos_to_iscreen(self, pos, rpos):
        return (
            (pos[0]-rpos[0]) * self.viz.scale ,
            (rpos[1]-pos[1]) * self.viz.scale
        )
    
    def points_to_iscreen(self, pts=None, orig=None):
        orig = orig if orig else self.viz.orig
        pts = pts if pts else [self.morph.pos]
        return [
            (
                pos[0] * self.viz.scale + orig[0],
                -pos[1] * self.viz.scale + orig[1],
            )
            for pos in pts
        ]

    def direction_to_iscreen(self, direction=None, orig=None):
        direction = direction if direction else self.morph.direction
        v = [direction[0] * self.viz.scale, -direction[1] * self.viz.scale]  # Scale the direction vector
        return self.morph.normalize(v)
    
    def posx_to_iscreen(self, posx=None, orig=None):
        orig = orig if orig else self.viz.orig[0]
        if isinstance(posx, (list, np.ndarray)):
            return [px * self.viz.scale + orig for px in posx]
        else:
            posx = posx if posx else self.morph.pos[0]
            return posx * self.viz.scale + orig

    def posy_to_iscreen(self, posy=None, orig=None):
        orig = orig if orig else self.viz.orig[1]
        if isinstance(posy, (list, np.ndarray)):
            return [-py * self.viz.scale + orig for py in posy]
        else:
            posy = posy if posy else self.morph.pos[1]
            return -posy * self.viz.scale + orig

    def rect_to_iscreen(self, pos=None, width=None, height=None, orig=None):
        pos = pos if pos else self.morph.pos
        width = width if width else self.morph.width
        height = height if height else self.morph.height
        orig = orig if orig else self.viz.orig
        return (
            pos[0] * self.viz.scale + orig[0],
            -pos[1] * self.viz.scale + orig[1],
            width * self.viz.scale,
            height * self.viz.scale,
        )

    def scale_to_iscreen(self, scale):
        return scale * self.viz.scale

    def draw(self, painter):
        pass

    def draw_direction_arrow(self, painter, length=20):
        pos1=self.pos_to_iscreen(self.morph.pos)
        dir=self.direction_to_iscreen()
        pos2=(pos1[0]+dir[0]*length, pos1[1]+dir[1]*length)
        self.viz.draw_arrow(painter,Color.RED,pos1,pos2,10,2)


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def scaled_sigmoid(x, fmin, fmax):
        x_normalized = (x - (fmin + fmax) / 2) / ((fmax - fmin) / 2)
        return Object.sigmoid(x_normalized)
