from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPolygonItem, QGraphicsLineItem
from PyQt6.QtGui import QPainterPath, QPolygonF, QBrush, QColor, QPen, QTransform,QPainter
from PyQt6.QtCore import Qt, QPointF, QTimer
import numpy as np
import random
from scipy.interpolate import splprep, splev
from qzeb_basic import *

class Joint(Object):
    def __init__(self, parent, pos=(0, 0), angle=0, radius=1.5):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.morph.pos = pos
        self.morph.angle = angle
        self.morph.radius = radius
        self.viz.facecolor=QColor("gray")

    def draw(self, painter,ref=None):
        painter.setBrush(QBrush(self.viz.facecolor))
        painter.setPen(Qt.PenStyle.NoPen)
        if ref is None:
            rpos1=self.pos_to_iscreen(self.morph.pos)
        else:
            rpos1=self.rel_pos_to_iscreen(self.morph.pos,ref)
        painter.drawEllipse(QPointF(*rpos1), self.scale_to_iscreen(self.morph.radius), self.scale_to_iscreen(self.morph.radius))

class Neuron(Object):
    def __init__(self, parent, firing=0, pos=(0, 0), radius=0.8, linewidth=0.1):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.firing = firing
        self.morph.pos = pos
        self.morph.radius = radius
        self.viz.linewidth = linewidth
        self.viz.facecolor = QColor("green")

    def update_position(self, rpos=None):
        self.morph.pos = rpos or self.morph.pos

    def draw(self, painter,ref=None):
        color = QColor("red") if self.firing else QColor("blue")
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        if ref is None:
            rpos1=self.pos_to_iscreen(self.morph.pos)
        else:
            rpos1=self.rel_pos_to_iscreen(self.morph.pos,ref)
        painter.drawEllipse(QPointF(*rpos1), self.scale_to_iscreen(self.morph.radius), self.scale_to_iscreen(self.morph.radius))

class Eye(Object):
    def __init__(self, parent, offset=5, radius=2.5, hemi=0):
        super().__init__()
        self.parent = parent
        self.viz = self.parent.viz.copy()
        self.viz.facecolor = QColor("white")
        self.hemi = hemi
        self.offset = offset
        self.offset_eyeball=offset*1.2
        self.offpos = -3
        self.gaze_angle = 0
        self.offset_angle = 15
        self.morph.radius = radius
        # eye muscle neuron
        self.neurons = [Neuron(self), Neuron(self)]
        self.neurons[0].update_position()
        self.neurons[1].update_position()
        self.update_position()
        self.children.extend(self.neurons)

    def update_position(self):
        v = self.parent.morph.direction
        pos = self.parent.morph.pos
        rpos = (pos[0] + v[0] * self.offpos, pos[1] + v[1] * self.offpos)
        if self.hemi:  # right eye
            dv = self.morph.right_vertical_direction(self.parent.morph.direction)
        else:  # left
            dv = self.morph.left_vertical_direction(self.parent.morph.direction)
        self.morph.direction = dv
        self.morph.pos = (rpos[0] + dv[0] * self.offset, rpos[1] + dv[1] * self.offset)
        self.eyeball_pos = (rpos[0] + dv[0] * self.offset_eyeball, rpos[1] + dv[1] * self.offset_eyeball)
        self.morph.norm()

        if self.hemi:
            anglerad = np.radians(self.offset_angle - 90)
            self.eyeball_direction = self.morph.rotate_vector(self.parent.morph.direction, anglerad)
        else:
            anglerad = np.radians(-self.offset_angle + 90)
            self.eyeball_direction = self.morph.rotate_vector(self.parent.morph.direction, anglerad)

        # eye muscle neuron part
        npos = self.morph.perpendicular_points(self.morph.pos, dv, 2, -3)
        if self.hemi:
            self.neurons[0].morph.pos = npos[1]
            self.neurons[0].firing = 1
            self.neurons[1].morph.pos = npos[0]
        else:
            self.neurons[0].morph.pos = npos[0]
            self.neurons[0].firing = 1
            self.neurons[1].morph.pos = npos[1]


    def draw(self, painter,ref=None):
        painter.setBrush(QBrush(QColor("black")))
        painter.setPen(QPen(QColor("black"), 1))

        rheight = 6
        rwidth2 = 3
        rwidth1 = 2

        if self.hemi:
            angle_dir = self.parent.get_angle() + self.offset_angle
            angle = angle_dir + self.gaze_angle
            lpts = self.viz.get_arc_points(self.morph.pos, rwidth1, rheight, -90, 90, angle - 90, 25)
            rpts = self.viz.get_arc_points(self.morph.pos, rwidth2, rheight, 90, 270, angle - 90, 25)
        else:
            angle_dir = self.parent.get_angle() - self.offset_angle
            angle = angle_dir + self.gaze_angle
            lpts = self.viz.get_arc_points(self.morph.pos, rwidth2, rheight, -90, 90, angle - 90, 25)
            rpts = self.viz.get_arc_points(self.morph.pos, rwidth1, rheight, 90, 270, angle - 90, 25)

        all_points = lpts + rpts

        if ref is None:
            rpos1=self.pos_to_iscreen(self.eyeball_pos)
        else:
            rpos1=self.rel_pos_to_iscreen(self.eyeball_pos,ref)

        all_points=[self.rel_pos_to_iscreen(p,ref) for p in all_points]
        all_points=[QPointF(float(p[0]), float(p[1])) for p in all_points]
        # Draw the polygon for the eyeball
        polygon = QPolygonF(all_points)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawPolygon(polygon)
        
        # Draw the retina (circle)
        radius = 1.8
        painter.setBrush(self.viz.facecolor)
        painter.drawEllipse(QPointF(*rpos1), self.scale_to_iscreen(radius), self.scale_to_iscreen(radius))

        # Optionally draw the outline of the eyeball
        linewidth = 1
        if linewidth > 0:
            painter.setPen(QColor(0, 0, 0))
            painter.drawPolyline(polygon)

        self.neurons[0].draw(painter,ref)
        self.neurons[1].draw(painter,ref)

        return polygon.boundingRect()


    
class Brain(Object):
    def __init__(self, parent, num_neurons=10):
        super().__init__()
        self.parent = parent
        self.num_neurons = num_neurons
        self.neurons = [Neuron(parent) for _ in range(num_neurons)]
        self.children.extend(self.neurons)

    def update_position(self, rpos=None, rdirection=None):
        self.morph.pos = rpos or self.parent.morph.pos
        self.morph.direction = rdirection or self.parent.morph.direction

    def draw(self, painter,ref=None):
        # Assuming neurons need to be drawn; otherwise, this could be left empty
        for neuron in self.neurons:
            neuron.draw(painter,ref)


class Head(Object):
    def __init__(self, parent, num_neurons=10):
        super().__init__()
        self.parent = parent
        self.viz = self.parent.viz
        self.eye_offset = 5
        self.joint_offset = 12

        self.eyes = [
            Eye(self, self.eye_offset, hemi=0),
            Eye(self, self.eye_offset, hemi=1),
        ]
        self.brain = Brain(self, num_neurons)
        self.joint = [Joint(self), Joint(self)]
        self.update_position()
        self.children.extend(self.eyes)
        self.children.append(self.brain)
        self.children.extend(self.joint)

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.morph.pos
        self.morph.direction = direction or self.parent.morph.direction
        self.morph.norm()
        self.joint[0].pos = self.morph.pos
        self.joint[1].pos = [
            self.morph.pos[0] - self.joint_offset * self.morph.direction[0],
            self.morph.pos[1] - self.joint_offset * self.morph.direction[1],
        ]
        self.eyes[0].update_position()
        self.eyes[1].update_position()
        self.brain.update_position()

    def draw(self, painter,ref=None):
        if self.verbose: print(f"{self.ID}>Head pos:{self.morph.pos}, dir:{self.morph.direction}")    
        rect0=self.eyes[0].draw(painter,ref)
        rect1=self.eyes[1].draw(painter,ref)    
        self.brain.draw(painter,ref)
        return rect1.united(rect0)

class Bone(Object):
    def __init__(self, parent, angle=0, bone_length=12, linewidth=3):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.angle = angle
        self.length = bone_length
        self.neuron_offset = linewidth * 0.6
        self.neuron_offpos = 5
        self.viz.linewidth = linewidth

        self.joint = [
            Joint(self, radius=linewidth / 4),
            Joint(self, radius=linewidth / 4),
        ]
        self.joint[0].viz.facecolor = QColor("gray")
        self.joint[1].viz.facecolor = QColor("gray")
        self.neurons = [Neuron(self), Neuron(self)]
        self.update_position()

        self.children.extend(self.joint)
        self.children.extend(self.neurons)

    def update_position(self, angle=None):
        self.angle = angle if angle is not None else self.angle
        self.morph.pos = self.parent.joint[1].pos
        self.morph.direction = self.morph.rotate_vector(self.parent.morph.direction, np.radians(self.angle))
        ndirection = self.morph.negative_direction(self.morph.direction)
        self.joint[0].pos = self.parent.joint[1].pos
        self.joint[1].pos = self.morph.point_along_direction(self.joint[0].pos, ndirection, self.length)
        # Neurons
        self.neurons[0].morph.pos, self.neurons[1].morph.pos = self.morph.perpendicular_points(
            self.joint[0].pos, ndirection, self.neuron_offset, self.neuron_offpos
        )

    def draw(self, painter,ref=None):
        pen = QPen(self.viz.linecolor, self.scale_to_iscreen(self.viz.linewidth))
        painter.setPen(pen)
        if ref is None:
            painter.drawLine(QPointF(*self.pos_to_iscreen(self.joint[0].pos)), QPointF(*self.pos_to_iscreen(self.joint[1].pos)))
        else:
            rpos1=self.rel_pos_to_iscreen(self.joint[0].pos,ref)
            rpos2=self.rel_pos_to_iscreen(self.joint[1].pos,ref)
            painter.drawLine(QPointF(*rpos1), QPointF(*rpos2))
        # Draw the joints and neurons
        self.joint[0].draw(painter,ref)
        self.neurons[0].draw(painter,ref)
        self.neurons[1].draw(painter,ref)

class Spine(Object):
    def __init__(self, parent, num_bones=4):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.morph.pos = self.parent.morph.pos
        self.morph.direction = self.parent.morph.direction
        self.joint = [Joint(self), Joint(self)]
        self.joint[0].pos = self.morph.pos
        self.joint[1].pos = self.morph.pos
        self.num_bones = num_bones
        self.bones = []
        bone_widths = [2.5, 2, 1.5, 1.2, 1]

        for i in range(num_bones):
            b = Bone(self.bones[i - 1] if i > 0 else self, linewidth=bone_widths[i])
            b.update_position()
            self.bones.append(b)
            self.children.append(b)

        self.children.extend(self.joint)


    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.morph.pos
        self.morph.direction = direction or self.parent.morph.direction
        self.joint[0].pos = self.morph.pos
        self.joint[1].pos = self.morph.pos
        for i in range(len(self.bones)):
            left = self.bones[i].neurons[0].firing
            right = self.bones[i].neurons[1].firing
            angle = (left - right) * 3
            self.bones[i].update_position(angle)

    def draw(self, painter,ref=None):
        if self.verbose: print(f"{self.ID}>Spine pos:{self.morph.pos}, dir:{self.morph.direction}")    
        for bone in self.bones:
            bone.draw(painter,ref)
        if self.verbose: print(f"{self.ID}>Bone pos:{self.bones[0].morph.pos}, dir:{self.bones[0].morph.direction}")


class Heart(Object):
    def __init__(self, parent, size=1.5, heartrate=120):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.size = size
        self.offpos = 5

        self.heartrate = heartrate
        self.beat_phase = 0
        self.beat_period = 60 / self.heartrate  # 60 seconds divided by heart rate
        t = np.arange(0, 2 * np.pi, 0.1)
        self.x = 16 * np.sin(t) ** 3
        self.y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        self.angle = 0
        self.update_position()


    def rotate_coordinates(self, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rotated = cos_angle * self.x - sin_angle * self.y
        y_rotated = sin_angle * self.x + cos_angle * self.y
        return x_rotated, y_rotated

    def update_position(self):
        self.morph.pos = self.morph.perpendicular_points(self.parent.morph.pos, self.parent.morph.direction, self.offpos)[0]
        self.angle = self.morph.calculate_angle(self.parent.morph.direction)

    def beat(self):
        self.beat_phase = (self.beat_phase + 1) % (self.beat_period * 60)
        if self.beat_phase < (self.beat_period * 30):
            scale = 1 + (self.beat_phase / (self.beat_period * 30)) * 0.5
            color_intensity = self.beat_phase / (self.beat_period * 30)
        else:
            scale = (
                1.5
                - (
                    (self.beat_phase - (self.beat_period * 30))
                    / (self.beat_period * 30)
                )
                * 0.5
            )
            color_intensity = ((self.beat_period * 60) - self.beat_phase) / (
                self.beat_period * 30
            )
        self.size_scaled = self.size * scale
        self.viz.color = QColor(255, 0, 0, int(color_intensity * 255))

    def draw(self, painter,ref=None):
        x, y = self.rotate_coordinates(np.radians(self.angle))
        pos = QPointF(*self.rel_pos_to_iscreen(self.morph.pos,ref))
        x_scaled = pos.x() + self.size_scaled * x / max(abs(x))
        y_scaled = pos.y() + self.size_scaled * y / max(abs(y))

        heart_shape = QPolygonF([QPointF(xi, yi) for xi, yi in zip(x_scaled, y_scaled)])
        painter.setBrush(QBrush(self.viz.color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(heart_shape)


class Body(Object):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.viz = parent.viz.copy()
        self.morph.pos = self.parent.head.joint[1].pos
        self.morph.direction = self.parent.morph.direction
        self.num_bones = 4
        self.spine = Spine(self, self.num_bones)
        self.children.append(self.spine)
        self.heartrate = 120
        self.heart = Heart(self, heartrate=self.heartrate)
        self.children.append(self.heart)

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.head.joint[1].pos
        self.morph.direction = direction or self.parent.morph.direction
        self.spine.update_position()
        self.heart.update_position()

    def draw(self, painter,ref=None):
        if self.verbose: print(f"{self.ID}>Body pos:{self.morph.pos}, dir:{self.morph.direction}")
        self.spine.draw(painter,ref)
        self.heart.beat()
        self.heart.draw(painter,ref)

        #top_left_x = rpos1[0] - radius
        #top_left_y = rpos1[1] - radius
        #return QRectF(top_left_x, top_left_y, 2 * radius, 2 * radius)


class ZFish(Object,QGraphicsItem):
    def __init__(self, parent=None, pos=(0, 0), direction=(0, 1)):
        QGraphicsItem.__init__(self)
        Object.__init__(self)
        self.parent = parent
        self.curpos=pos
        self.position=(0,0)
        self.morph.pos = pos
        self.morph.direction = direction
        self.num_neurons = 20
        self.tail_prepos = (0, 0)
        self.speed = 1
        self.speedfactor = 0.5
        self.speed_baseline = 0.5
        self.smoothfactor = 10
        self.bounding_area = (-300, -300, 600, 600)
        self.rect = QRectF(pos[0],pos[1],0,0)
        self.head = Head(self)  # head pos = zfish.pos
        self.children.append(self.head)
        self.head.update_position()
        self.body = Body(self)
        self.children.append(self.body)
        self.body.update_position()
        self.update_position()

    def __repr__(self):
        pass

    def update_random_firing(self):
        muscle_neuron_values = [
            [random.randint(0, 3), random.randint(0, 3)] for _ in range(self.body.spine.num_bones)
        ]
        for i, (left, right) in enumerate(muscle_neuron_values):
            self.body.spine.bones[i].neurons[0].firing = left
            self.body.spine.bones[i].neurons[1].firing = right

        self.update_position()

    def calculate_direction(self):
        positions = []
        positions.append(self.head.morph.pos)
        positions.append(self.head.joint[1].pos)
        pos = [bone.joint[1].pos for bone in self.body.spine.bones]
        positions.extend(pos)
        points = np.array(positions[::-1])
        tck, u = splprep([points[:, 0], points[:, 1]], s=self.smoothfactor)
        dx, dy = splev(0, tck, der=1)
        direction_vector = np.array([dx, dy])
        direction = direction_vector / np.linalg.norm(direction_vector)
        self.morph.direction = direction.tolist()

    def calculate_remained_energy(self):
        pass
    
    def get_tail_pos(self):
        return self.body.spine.bones[-1].joint[1].pos
    
    def calculate_force(self):
        c = self.get_tail_pos()
        d = [c[0] - self.tail_prepos[0], c[1] - self.tail_prepos[1]]
        f = np.linalg.norm(d)
        s = self.scaled_sigmoid(f, 0, 100)
        self.speed = self.speed_baseline + self.speedfactor * s

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.morph.pos
        self.morph.direction = direction or self.morph.direction
        #self.update_head_body()

        self.calculate_force()
        if direction is None:
            self.calculate_direction()
            self.position = (self.position[0] + self.speed * self.morph.direction[0],
                              self.position[1] + self.speed * self.morph.direction[1])

        self.tail_prepos = self.body.spine.bones[-1].joint[1].pos
        self.update_head_body()

    def update_head_body(self):
        self.head.update_position()
        self.body.update_position()


    def draw_skin(self, painter):

        pointsL = [self.head.eyes[0].morph.pos]
        pointsR = [self.head.eyes[1].morph.pos]

        offset_bone = [7, 3, 4, 3, 2, 1]
        offset_tend = 10
        bone = self.body.spine.bones[0]
        p1, p2 = self.morph.perpendicular_points(bone.joint[0].pos, bone.morph.direction, offset_bone[0])
        pointsL.append(p1)
        pointsR.append(p2)

        for bone, off in zip(self.body.spine.bones, offset_bone[1:]):
            p1, p2 = self.morph.perpendicular_points(bone.joint[1].pos, bone.morph.direction, off)
            pointsL.append(p1)
            pointsR.append(p2)

        bone = self.body.spine.bones[-1]
        tendpos = (
            bone.joint[1].pos[0] - offset_tend * bone.morph.direction[0],
            bone.joint[1].pos[1] - offset_tend * bone.morph.direction[1],
        )
        ptail1, ptail2 = self.morph.perpendicular_points(tendpos, bone.morph.direction, 0.5)
        pointsL.append(ptail1)
        pointsR.append(ptail2)

        tailend = [
            bone.joint[1].pos[0] - 1.5 * offset_tend * bone.morph.direction[0],
            bone.joint[1].pos[1] - 1.5 * offset_tend * bone.morph.direction[1],
        ]

        point_tailend = [self.pos_to_iscreen(tailend)]

        angle = self.morph.calculate_angle(self.head.morph.direction)
        center = self.center(self.head.eyes[0].morph.pos, self.head.eyes[1].morph.pos)
        scale = self.head.eye_offset
        arc_points_ = self.head.eyes[0].viz.get_arc_points(center, scale, scale, -90, 90, angle)

        arc_points = [self.pos_to_iscreen(p) for p in arc_points_]

        # Generate left and right skin points using spline interpolation
        points = np.array([(point[0], point[1]) for point in pointsL])  # Convert QPointF to (x, y) tuples
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        x_fine, y_fine = splev(np.linspace(0, 1, 50), tck)
        points_fine_left = list(zip(map(self.posx_to_iscreen, x_fine), map(self.posy_to_iscreen, y_fine)))

        points = np.array([(point[0], point[1]) for point in pointsR])  # Convert QPointF to (x, y) tuples
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        x_fine, y_fine = splev(np.linspace(0, 1, 50), tck)
        points_fine_right = list(zip(map(self.posx_to_iscreen, x_fine), map(self.posy_to_iscreen, y_fine)))

     # Combine all points in the correct order for polygon filling
        # Determine which end of arc_points is closer to the first point of points_fine_left
        dist_end_to_left = np.linalg.norm(
            np.array(arc_points[-1]) - np.array(points_fine_left[0])
        )
        dist_start_to_left = np.linalg.norm(
            np.array(arc_points[0]) - np.array(points_fine_left[0])
        )
        linewidth = 1
        # If the end of arc_points is closer, we keep the order, otherwise, we reverse arc_points
        if dist_end_to_left < dist_start_to_left:
            all_points = arc_points + points_fine_left + point_tailend + points_fine_right[::-1]
        else:
            all_points = arc_points[::-1] + points_fine_left + point_tailend + points_fine_right[::-1]
  
        refpos=self.pos_to_iscreen(self.morph.pos)
        
        all_points = [(x-refpos[0],y-refpos[1]) for x, y in all_points]
        qpoints = [QPointF(x, y) for x, y in all_points]
        color = QColor(173, 216, 230)  # Light Blue color equivalent to Pygame's Color.LIGHTBLUE

        path = QPainterPath()
        path.addPolygon(QPolygonF(qpoints))
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(0, 0, 0), 1))  # Black color for outline
        painter.drawPath(path)

        # Calculate and update the bounding box for the entire fish
        min_x = min(point.x() for point in qpoints)
        max_x = max(point.x() for point in qpoints)
        min_y = min(point.y() for point in qpoints)
        max_y = max(point.y() for point in qpoints)
        self.rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        #self.rect = path.boundingRect()
        if self.verbose: print(f"{self.ID}>Zeb range:{self.rect} ref:{refpos}")    


    def draw(self, painter,refpos=None):
        if refpos is None:
            refpos=self.morph.pos
        self.draw_skin(painter)    
        self.body.draw(painter,refpos)
        recth=self.head.draw(painter,refpos)
        self.rect=self.rect.united(recth)
        
    
    def boundingRect(self) -> QRectF:
        return self.rect
        
    def paint(self, painter: QPainter, option, widget=None):
        self.draw(painter)

    def advance(self, phase):
        if not phase:
            return
        self.prepareGeometryChange()
        self.update_random_firing()
        rpos=self.pos_to_iscreen(self.position)
        self.setPos(QPointF(*rpos))
        if random.random() < 0.1:
            self.parent.add_random_item_to_display_area()
            #self.morph.direction = self.morph.rotate_vector(self.morph.direction, np.radians(random.randint(-30, 30)))

        if self.verbose: print(f"{self.ID}>Zeb pos:{self.morph.pos}, real pos:{rpos}")    
