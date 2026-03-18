import pygame
import numpy as np
import random
from scipy.interpolate import splprep, splev


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
    OCEANBLUE=(28, 107, 160)

    @staticmethod
    def to_rgb(color):
        if isinstance(color, tuple) and len(color) == 3:
            return tuple(int(c * 255) if isinstance(c, float) else c for c in color)
        elif isinstance(color, str):
            return Color.COLORS.get(color.lower(), Color.COLORS["black"])
        else:
            return Color.COLORS["black"]
    @staticmethod        
    def add_alpha(col,alpha):
        return (col[0],col[1],col[2],alpha)
    
    @staticmethod
    def to_name(rgb):
        rgb = tuple(int(c * 255) if isinstance(c, float) else c for c in rgb)
        for name, value in Color.COLORS.items():
            if value == rgb:
                return name
        return "unknown"


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
        self.linecolor = Color().to_rgb(linecolor)
        self.facecolor = Color().to_rgb(facecolor)
        self.linewidth = linewidth
        self.linetype = linetype
        self.orig = orig
        self.scale = scale
        self.on = on
        
    def copy(self):
        # Create a shallow copy of the instance
        new_copy = Viz(self.linecolor,self.facecolor,self.linewidth,self.linetype,self.scale,self.orig,self.on)
        return new_copy
    
    @staticmethod
    def get_arc_points_rect(rect, start_angle, end_angle, angle, num_points=50):
        start_angle = np.radians(start_angle)
        end_angle = np.radians(end_angle)
        angle = np.radians(angle)
        points = []
        # Calculate the center of the ellipse
        x_center = rect.left + rect.width / 2
        y_center = rect.top + rect.height / 2

        # Calculate the horizontal and vertical radii
        rwidth = rect.width / 2  # Horizontal radius
        rheight = rect.height / 2  # Vertical radius

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
        angle=angle if angle else (start_angle+end_angle)/2
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

    @staticmethod
    def draw_ellipse(screen, rect, angle, facecolor=None, linecolor=None, linewidth=0):
        """Fills a rotated ellipse on the Pygame screen."""
        rotated_points = Viz.get_ellipse_points(rect,angle=angle)
        int_points = [(int(x), int(y)) for x, y in rotated_points]
        # Draw the filled polygon
        if facecolor:
            pygame.draw.polygon(screen, facecolor, int_points)

        if linewidth > 1:
            linecolor = linecolor if linecolor else Color.BLACK
            pygame.draw.lines(screen, linecolor, True, int_points, linewidth)

    @staticmethod
    def draw_arrow(screen, color, start, end, arrow_length=10, arrow_width=5, line_width=2):
        pygame.draw.line(screen, color, start, end, line_width)
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return  # Avoid division by zero
        direction = direction / length  # Normalize the direction vector
        perpendicular = np.array([-direction[1], direction[0]])
        left_arrow_point = end - arrow_length * direction + arrow_width * perpendicular
        right_arrow_point = end - arrow_length * direction - arrow_width * perpendicular
        pygame.draw.line(screen, color, end, left_arrow_point, line_width)
        pygame.draw.line(screen, color, end, right_arrow_point, line_width)    

class Morphology:
    def __init__(self, type=None, pos=(0, 0), direction=[1,0],width=0, height=0, radius=0, nodes=[]):
        self.pos = pos
        self.direction=direction
        self.type = type
        self.width = width
        self.height = height
        self.nodes = nodes
        self.radius = radius

    def norm(self,direction=None):
        self.direction = direction if direction else self.direction
        self.direction=Morphology.normalize(self.direction)
  
    @staticmethod     
    def normalize(v):
        v_length = np.sqrt(
            v[0] ** 2 + v[1] ** 2
        )
        return v if v_length==1 else (v[0] / v_length,v[1] / v_length)       

    @staticmethod    
    def negative_direction(v):
        return (-v[0],-v[1])

    @staticmethod
    def calculate_angle(v):
        dx, dy = v
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def left_vertical_direction(v):
        dx, dy = v
        return (-dy,dx)

    @staticmethod
    def right_vertical_direction(v):
        dx, dy = v
        return (dy, -dx)

    @staticmethod
    def perpendicular_points(pos, direction, offset, offpos=0):
        v = Morphology.normalize(direction)
        offz = (
            pos[0] + v[0] * offpos,
            pos[1] + v[1] * offpos,
        )
        pv1 = (-v[1], v[0])
        pv2 = (v[1], -v[0])
        point1 = (
            offz[0] + pv1[0] * offset,
            offz[1] + pv1[1] * offset,
        )
        point2 = (
            offz[0] + pv2[0] * offset,
            offz[1] + pv2[1] * offset,
        )
        return point1, point2

    @staticmethod
    def point_along_direction(rpos, direction, length, anglerad=0):
        v = Morphology.normalize(direction)
        if anglerad==0:
            rv = (
                v[0] * np.cos(anglerad)
                - v[1] * np.sin(anglerad),
                v[0] * np.sin(anglerad)
                + v[1] * np.cos(anglerad),
            )  
        else:
            rv=v
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
    
    @staticmethod
    def get_rotated_rect_bounding_box(rect, anglerad):
        """Calculate the axis-aligned bounding box of a rotated rectangle."""
        cx, cy = rect.center
        corners = [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft]
        rotated_corners = [
            Object.rotate_point(corner, anglerad, (cx, cy)) for corner in corners
        ]
        min_x = min(rotated_corners, key=lambda p: p[0])[0]
        max_x = max(rotated_corners, key=lambda p: p[0])[0]
        min_y = min(rotated_corners, key=lambda p: p[1])[1]
        max_y = max(rotated_corners, key=lambda p: p[1])[1]
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    @staticmethod
    def rect_to_points(rect):
        return (
            (rect.topleft[0], rect.topleft[1]),
            (rect.topright[0], rect.topright[1]),
            (rect.bottomright[0], rect.bottomright[1]),
            (rect.bottomleft[0], rect.bottomleft[1]),
        )


class State:
    def __init__(self, state=0):
        self.state = state

class Object:
    def __init__(self, morph=None, viz=None, state=None):
        self.morph = morph if morph else Morphology()
        self.viz = viz if viz else Viz()
        self.state = state if state else State()

    @staticmethod
    def center(pos1,pos2):
        return ((pos1[0]+pos2[0])/2,(pos1[1]+pos2[1])/2)
    
    def get_angle(self):
        dx, dy = self.morph.direction
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad)
    
    def center_iscreen(self,pos1,pos2):
        return self.center(self.pos_to_iscreen(pos1),self.pos_to_iscreen(pos2))
    
    def pos_to_iscreen(self, pos=None, orig=None):
        pos = pos if pos is not None else self.morph.pos
        orig = orig if orig is not None else self.viz.orig
        return (
            int(pos[0] * self.viz.scale + orig[0]),
            int(-pos[1] * self.viz.scale + orig[1]),
        )
    def points_to_iscreen(self, pts=None, orig=None):
        orig = orig if orig else self.viz.orig
        pts = pts if pts else [self.morph.pos]
        return [
            (
                int(pos[0] * self.viz.scale + orig[0]),
                int(-pos[1] * self.viz.scale + orig[1]),
            )
            for pos in pts
        ]

    def direction_to_iscreen(self, direction=None, orig=None):
        direction = direction if direction else self.morph.direction
        v=[direction[0] * self.viz.scale,-direction[1] * self.viz.scale] #in case the sclae is different for x,y
        return self.morph.normalize(v)
    
    def posx_to_iscreen(self, posx=None, orig=None):
        orig = orig if orig else self.viz.orig[0]
        if isinstance(posx, (list, np.ndarray)):
            # If posx is a list or array, apply the transformation to each element
            return (int(px * self.viz.scale + orig) for px in posx)
        else:
            # If posx is a single value, apply the transformation directly
            posx = posx if posx else self.morph.pos[0]
            return int(posx * self.viz.scale + orig)

    def posy_to_iscreen(self, posy=None, orig=None):
        orig = orig if orig else self.viz.orig[1]
        if isinstance(posy, (list, np.ndarray)):
            # If posy is a list or array, apply the transformation to each element
            return (int(-py * self.viz.scale + orig) for py in posy)
        else:
            # If posy is a single value, apply the transformation directly
            posy = posy if posy else self.morph.pos[1]
            return int(-posy * self.viz.scale + orig)

    def rect_to_iscreen(self, pos=None, width=None, height=None, orig=None):
        pos = pos if pos else self.morph.pos
        width = width if width else self.morph.width
        height = height if height else self.morph.height
        orig = orig if orig else self.viz.orig
        return (
            int(pos[0] * self.viz.scale + orig[0]),
            int(-pos[1] * self.viz.scale + orig[1]),
            int(width * self.viz.scale),
            int(height * self.viz.scale),
        )

    def scale_to_iscreen(self, scale):
        return int(scale * self.viz.scale)

    def draw(self, screen):
        pass

    def draw_direction_arrow(self,screen,length=20):
        pos1=self.pos_to_iscreen(self.morph.pos)
        dir=self.direction_to_iscreen()
        pos2=(pos1[0]+dir[0]*length, pos1[1]+dir[1]*length)
        self.viz.draw_arrow(screen,Color.RED,pos1,pos2,10,2)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def scaled_sigmoid(x, fmin, fmax):
        x_normalized = (x - (fmin + fmax) / 2) / ((fmax - fmin) / 2)
        return Object.sigmoid(x_normalized)


# Base Object class
""" 
class IObject(pygame.sprite.Sprite, Object):
    def __init__(self, x=0, y=0, image=None, morph=None, viz=None, state=None):
        pygame.sprite.Sprite.__init__(self)

        Object.__init__(self, morph, viz, state)

        self.original_image = image
        self.image = image
        self.rect = self.image.get_rect(center=(x, y)) if image else None
        self.size = self.rect.size if self.rect else (0, 0)
        self.angle = 0

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)
"""