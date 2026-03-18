import pygame
import numpy as np
import random
from scipy.interpolate import splprep, splev
from vzt_basic import *
import copy


class Joint:
    def __init__(self, parent, pos=(0, 0), angle=0, radius=1.5):
        self.parent=parent
        self.viz=parent.viz.copy()
        self.pos = pos
        self.angle = angle
        self.radius = radius

    def pos_to_iscreen(self, pos=None, orig=None):
        pos = pos if pos else self.pos
        orig = orig if orig else self.viz.orig
        return (
            int(pos[0] * self.viz.scale + orig[0]),
            int(pos[1] * self.viz.scale + orig[1]),
        )

    def scale_to_iscreen(self, scale):
        return int(scale * self.viz.scale)

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            self.viz.facecolor,
            self.pos_to_iscreen(),
            self.scale_to_iscreen(self.radius),
        )


class Neuron(Object):
    def __init__(self, parent, firing=0, pos=(0, 0), radius=0.8, linewidth=0.1):
        super().__init__()
        self.parent=parent
        self.viz=parent.viz.copy()
        self.firing = firing
        self.morph.pos = pos
        self.morph.radius = radius
        self.viz.linewidth = linewidth
        self.viz.facecolor = Color.GREEN

    def update_position(self, rpos=None):
        self.morph.pos = rpos or self.morph.pos


    def draw(self, screen):
        color = Color.RED if self.firing else Color.BLUE
        pygame.draw.circle(
            screen,
            color,
            self.pos_to_iscreen(),
            self.scale_to_iscreen(self.morph.radius),
        )


class Eye(Object):
    def __init__(self, parent, offset=5, radius=2.5, hemi=0):
        super().__init__()
        self.parent=parent
        self.viz=self.parent.viz.copy()
        self.viz.facecolor=Color.WHITE
        self.hemi = hemi
        self.offset = offset
        self.offpos = -3
        self.gaze_angle = 0
        self.offset_angle = 15
        self.morph.radius = radius

        #eye muscle neuron
        self.neuron = [Neuron(self), Neuron(self)]
        self.neuron[0].update_position()
        self.neuron[1].update_position()

        self.update_position()

    def update_position(self):
        v=self.parent.morph.direction
        pos=self.parent.morph.pos
        rpos = (pos[0] + v[0] * self.offpos,pos[1] + v[1] * self.offpos)
        if self.hemi: #right eye
            dv = self.morph.right_vertical_direction(self.parent.morph.direction)
        else: #left
            dv = self.morph.left_vertical_direction(self.parent.morph.direction)    
        self.morph.direction=dv 
        self.morph.pos = (rpos[0] + dv[0] * self.offset,rpos[1] + dv[1] * self.offset)
        self.morph.norm()

        if self.hemi:
            anglerad=np.radians(self.offset_angle-90)
            self.eyeball_direction = self.morph.rotate_vector(self.parent.morph.direction,anglerad)
        else:
            anglerad=np.radians(-self.offset_angle+90)
            self.eyeball_direction = self.morph.rotate_vector(self.parent.morph.direction,anglerad)

        # eye muscle neuron part
        # neuron
        npos = self.morph.perpendicular_points(self.morph.pos, dv, 2, -3)
        if self.hemi:
            self.neuron[0].morph.pos = npos[1]
            self.neuron[0].firing = 1
            self.neuron[1].morph.pos = npos[0]
        else:
            self.neuron[0].morph.pos = npos[0]
            self.neuron[0].firing = 1
            self.neuron[1].morph.pos = npos[1]


    def draw(self, screen):
        #eyeball
        rheight=6;rwidth2=3;rwidth1=2
        if self.hemi:
            angle_dir=self.parent.get_angle()+self.offset_angle
            angle=angle_dir+self.gaze_angle
            lpts=self.viz.get_arc_points(self.morph.pos,rwidth1,rheight,-90,90,angle-90,25)
            rpts=self.viz.get_arc_points(self.morph.pos,rwidth2,rheight,90,270,angle-90,25)
        else:
            angle_dir=self.parent.get_angle()-self.offset_angle
            angle=angle_dir+self.gaze_angle
            lpts=self.viz.get_arc_points(self.morph.pos,rwidth2,rheight,-90,90,angle-90,25)
            rpts=self.viz.get_arc_points(self.morph.pos,rwidth1,rheight,90,270,angle-90,25)

        all_points=self.points_to_iscreen(lpts+rpts)
        pygame.draw.polygon(screen, Color.BLACK, all_points)
        linewidth=1
        if linewidth > 0:
            pygame.draw.lines(screen, Color.BLACK, True, all_points, linewidth)

        #retina 
        radius=1.5
        pygame.draw.circle(
                screen,
                self.viz.facecolor,
                self.pos_to_iscreen(),
                self.scale_to_iscreen(radius),
            )
        #self.draw_direction_arrow(screen)
        # eye muscle neuron part
        self.neuron[0].draw(screen)
        self.neuron[1].draw(screen)
        #print(f"Eye pos:{self.morph.pos}, dir:{self.morph.direction}")    


class Brain(Object):
    def __init__(self, parent, num_neurons=10):
        super().__init__()
        self.parent=parent
        self.num_neurons = num_neurons
        self.neurons = [Neuron(parent) for _ in range(num_neurons)]

    def update_position(self, rpos=None, rdirection=None):
        self.morph.pos = rpos or self.parent.morph.pos
        self.morph.direction = rdirection or self.parent.morph.direction
        pass

    def draw(self, screen):
        pass


class Head(Object):
    def __init__(self, parent, num_neurons=10):
        super().__init__()
        self.parent=parent
        self.viz=self.parent.viz
        self.eye_offset = 5
        self.joint_offset = 12
 
        self.eyes = [
            Eye(self, self.eye_offset, hemi=0),
            Eye(self, self.eye_offset, hemi=1),
        ]
        self.brain = Brain(self, num_neurons)
        self.joint = [Joint(self),Joint(self)]
        self.update_position()

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.morph.pos
        self.morph.direction = direction or self.parent.morph.direction
        self.morph.norm()
        self.joint[0].pos=self.morph.pos
        self.joint[1].pos = [
            self.morph.pos[0] - self.joint_offset * self.morph.direction[0],
            self.morph.pos[1] - self.joint_offset * self.morph.direction[1],
        ]
        self.eyes[0].update_position()
        self.eyes[1].update_position()
        self.brain.update_position()

    def draw(self, screen):
        print(f"Head pos:{self.morph.pos}, dir:{self.morph.direction}")    
        for eye in self.eyes:
            eye.draw(screen)
        self.brain.draw(screen)
        


class Bone(Object):
    def __init__(self, parent, angle=0, bone_length=12, linewidth=3):
        super().__init__()
        self.parent=parent
        self.viz=parent.viz.copy()
        self.angle = angle
        self.length = bone_length
        self.neuron_offset = linewidth * 0.6
        self.neuron_offpos=5
        self.viz.linewidth = linewidth

        self.joint = [
            Joint(self,radius=linewidth / 4),
            Joint(self,radius=linewidth / 4),
        ]
        self.joint[0].viz.facecolor=Color.GRAY
        self.joint[1].viz.facecolor=Color.GRAY

        self.neuron = [Neuron(self), Neuron(self)]
        self.update_position()

    def update_position(self, angle=None):
        self.angle = angle if angle is not None else self.angle
        self.morph.pos=self.parent.joint[1].pos
        self.morph.direction=self.morph.rotate_vector(self.parent.morph.direction,np.radians(self.angle))
        ndirection=self.morph.negative_direction(self.morph.direction)
        self.joint[0].pos = self.parent.joint[1].pos
        self.joint[1].pos = self.morph.point_along_direction(self.joint[0].pos, ndirection, self.length)
        # neurons
        self.neuron[0].morph.pos, self.neuron[1].morph.pos = self.morph.perpendicular_points(self.joint[0].pos,ndirection,self.neuron_offset, self.neuron_offpos)


    def draw(self, screen):
        pygame.draw.line(
            screen,
            self.viz.linecolor,
            self.pos_to_iscreen(self.joint[0].pos),
            self.pos_to_iscreen(self.joint[1].pos),
            self.scale_to_iscreen(self.viz.linewidth),
        )
        self.joint[0].draw(screen)

        self.neuron[0].draw(screen)
        self.neuron[1].draw(screen)
        #print(f"Bone pos:{self.morph.pos}, dir:{self.morph.direction}")    

class Spine(Object):
    def __init__(self, parent, num_bones=4):
        super().__init__()
        self.parent=parent
        self.viz=parent.viz.copy()
        self.morph.pos=self.parent.morph.pos
        self.morph.direction=self.parent.morph.direction
        self.joint=[Joint(self),Joint(self)]
        self.joint[0].pos=self.morph.pos
        self.joint[1].pos=self.morph.pos
        self.num_bones = num_bones
        self.bones = []
        bone_widths = [2.5, 2, 1.5, 1.2, 1]
        
        for i in range(num_bones):
            b=Bone(self.bones[i - 1] if i > 0 else self,linewidth=bone_widths[i])
            b.update_position()
            self.bones.append(b)

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.morph.pos
        self.morph.direction = direction or self.parent.morph.direction
        self.joint[0].pos=self.morph.pos
        self.joint[1].pos=self.morph.pos
        # firing induced changes
        for i in range(len(self.bones)):
            # check neuron firing 
            left = self.bones[i].neuron[0].firing
            right = self.bones[i].neuron[1].firing
            angle = (left - right) * 3
            self.bones[i].update_position(angle)

    def draw(self, screen):
        print(f"Spine pos:{self.morph.pos}, dir:{self.morph.direction}")    
        for bone in self.bones:
            bone.draw(screen)
        print(f"Bone pos:{self.bones[0].morph.pos}, dir:{self.bones[0].morph.direction}")     


class Heart(Object):
    def __init__(self, parent, size=1.5, heartrate=120):
        super().__init__()
        self.parent=parent
        self.viz=self.parent.viz
        self.size = size
        self.offpos=5

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
        self.color = (255, 0, 0, int(color_intensity * 255))

    def draw(self, screen):
        x, y = self.rotate_coordinates(np.radians(self.angle))
        pos = self.pos_to_iscreen(self.morph.pos)
        x_scaled = pos[0] + self.size_scaled * x / max(abs(x))
        y_scaled = pos[1] + self.size_scaled * y / max(abs(y))

        heart_shape = np.vstack((x_scaled, y_scaled)).T
        pygame.draw.polygon(screen, self.color, heart_shape)


class Body(Object):
    def __init__(self, parent):
        super().__init__()
        self.parent=parent
        self.viz=parent.viz.copy()
        self.morph.pos=self.parent.head.joint[1].pos
        self.morph.direction=self.parent.morph.direction
        self.num_bones=4
        self.spine = Spine(self, self.num_bones)
        self.heartrate = 120
        self.heart = Heart(self,heartrate=self.heartrate)

    def update_position(self, pos=None, direction=None):
        self.morph.pos = pos or self.parent.head.joint[1].pos
        self.morph.direction = direction or self.parent.morph.direction
        self.spine.update_position()
        self.heart.update_position()

    def draw(self, screen):
        print(f"Body pos:{self.morph.pos}, dir:{self.morph.direction}")
        self.spine.draw(screen)
        self.heart.beat()
        self.heart.draw(screen)
        
        #self.draw_direction_arrow(screen)

class ZFish(Object):
    def __init__(self, pos=(0, 0), direction=(0, 1)):
        super().__init__()
        self.morph.pos = pos 
        self.morph.direction=direction
        self.num_neurons = 20
        self.tail_prepos = [0, 0]
        self.speed = 1
        self.speedfactor = 0.5
        self.speed_baseline = 0.5
        self.smoothfactor = 10
        self.bounding_area = (-300, -300, 600, 600)
        self.rect = None
        self.head = Head(self) #head pos = zfish.pos 
        self.head.update_position()
        self.body = Body(self)
        self.body.update_position()
        self.update_position()

    def update_random_firing(self):
        muscle_neuron_values = [
            [random.randint(0, 3), random.randint(0, 3)] for _ in range(self.body.spine.num_bones)
        ]
        for i, (left, right) in enumerate(muscle_neuron_values):
            self.body.spine.bones[i].neuron[0].firing = left
            self.body.spine.bones[i].neuron[1].firing = right

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
        self.update_head_body()

        if 1:
            self.calculate_force()
            if direction is None:
                self.calculate_direction()
                self.morph.pos = (self.morph.pos[0] + self.speed * self.morph.direction[0],
                    self.morph.pos[1] + self.speed * self.morph.direction[1])
            self.tail_prepos = self.body.spine.bones[-1].joint[1].pos
        self.update_head_body()

    def update_head_body(self):
        self.head.update_position()
        self.body.update_position()

    def draw_skin(self, screen, dispflag=True):
        pointsL =[self.head.eyes[0].morph.pos]
        pointsR =[self.head.eyes[1].morph.pos]
        #print(f"left eye:{self.head.eyes[0].morph.pos}")
        #print(f"right eye:{self.head.eyes[1].morph.pos}")
        
        #offset_bone = [7, 3, 4, 3, 2, 1]
        #posoff= (self.head.joint_offset)/2
        #p1, p2 = self.morph.perpendicular_points(self.head.morph.pos, self.head.morph.direction, offset_bone[0],posoff)
        #print(f"leftp:{p1} rightp:{p2}")
        #pointsL.append(p1)
        #pointsR.append(p2)

        offset_bone = [7, 3, 4, 3, 2, 1]
        offset_tend = 10
        bone = self.body.spine.bones[0]
        p1, p2 = self.morph.perpendicular_points(bone.joint[0].pos, bone.morph.direction, offset_bone[0])
        #print(f"leftp:{p1} rightp:{p2}")
        pointsL.append(p1)
        pointsR.append(p2)

        for bone, off in zip(self.body.spine.bones, offset_bone[1:]):
            p1, p2 = self.morph.perpendicular_points(bone.joint[1].pos,bone.morph.direction, off)
            #print(f"leftp:{p1} rightp:{p2}")
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
                bone.joint[1].pos[0] - 1.5*offset_tend * bone.morph.direction[0],
                bone.joint[1].pos[1] - 1.5*offset_tend * bone.morph.direction[1],
        ]
        point_tailend=[self.pos_to_iscreen(tailend)]

        angle = self.morph.calculate_angle(self.head.morph.direction)
        # Calculate points along the arc of the head
        center = self.center(self.head.eyes[0].morph.pos,self.head.eyes[1].morph.pos)
        scale = self.head.eye_offset
        arc_points = self.head.eyes[0].viz.get_arc_points(center,scale,scale,-90,90,angle)
        arc_points = self.points_to_iscreen(arc_points)

        # Generate left and right skin points using spline interpolation
        points = np.array(pointsL)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        x_fine, y_fine = splev(np.linspace(0, 1, 50), tck)
        points_fine_left = list(zip(map(self.posx_to_iscreen, x_fine), map(self.posy_to_iscreen, y_fine)))

        points = np.array(pointsR)
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
            #pygame.draw.lines(screen, Color.GREEN, False, arc_points, linewidth)
            #pygame.draw.lines(screen, Color.YELLOW, False, points_fine_left, linewidth)
            #pygame.draw.lines(screen, Color.BLUE, False, points_fine_right, linewidth)
        else:
            all_points = arc_points[::-1] + points_fine_left + point_tailend + points_fine_right[::-1]
            #pygame.draw.lines(screen, Color.GREEN, False, arc_points, linewidth)
            #pygame.draw.lines(screen, Color.YELLOW, False, points_fine_left, linewidth)
            #pygame.draw.lines(screen, Color.BLUE, False, points_fine_right, linewidth)

        # Create a mask from the filled polygon for collision detection
        # polygon_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        # pygame.draw.polygon(polygon_surface, (255, 255, 255, 255), all_points)
        # self.mask = pygame.mask.from_surface(polygon_surface)

        color = Color.BLACK
        
        if dispflag:
            pygame.draw.polygon(screen, Color.LIGHTBLUE, all_points)
            if linewidth > 0:
                pygame.draw.lines(screen, Color.BLACK, True, all_points, linewidth)


        # Calculate and update the bounding box for the entire fish
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)

        self.rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def check_collision(self, other_object):
        if hasattr(other_object, "mask"):
            # If the other object has a mask, use mask-based collision detection
            offset = (
                other_object.rect.left - self.rect.left,
                other_object.rect.top - self.rect.top,
            )
            collision = self.mask.overlap(other_object.mask, offset)
            return (
                collision is not None
            )  # Returns True if there's a collision, otherwise False
        else:
            # If the other object doesn't have a mask, use rectangle collision detection
            return self.rect.colliderect(other_object.rect)

    def draw(self, screen):
        print(f"Fish pos:{self.morph.pos}, dir:{self.morph.direction}")
        self.draw_skin(screen)
        self.body.draw(screen)
        self.head.draw(screen)
        #self.draw_direction_arrow(screen)

    