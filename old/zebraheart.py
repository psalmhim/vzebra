import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from scipy.interpolate import splprep, splev

class Viz:
    def __init__(self, linecolor='black', facecolor=None, linewidth=1, linetype='-', vison=True):
        self.linecolor = linecolor
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.linetype = linetype
        self.on = vison

class Morphology:
    def __init__(self, type=None, center=(0, 0), joint=None, width=0, height=0, nodes=[]):
        self.center = center
        self.type = type
        self.joint = joint if joint else [None, None]
        self.width = width
        self.height = height
        self.nodes = nodes

class State:
    def __init__(self, state=0, pos=(0, 0)):
        self.state = state
        self.pos = pos

class Object:
    def __init__(self, morph=None, viz=None, state=None):
        self.morph = morph if morph else Morphology()
        self.viz = viz if viz else Viz()
        self.state = state if state else State()

    def draw(self, ax=plt.gca()):
        pass

    def calculate_angle(self, direction):
        dx, dy = direction
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def calculate_perpendicular_points(self, pos, direction, width, offpos=0):
        direction_length = np.sqrt(direction[0]**2 + direction[1]**2)
        direction_unit = (direction[0] / direction_length, direction[1] / direction_length)
        offpos_point = (pos[0] + direction_unit[0] * offpos, pos[1] + direction_unit[1] * offpos)
        perp_vector1 = (-direction_unit[1], direction_unit[0])
        perp_vector2 = (direction_unit[1], -direction_unit[0])
        perp_vector1_scaled = (perp_vector1[0] * width, perp_vector1[1] * width)
        perp_vector2_scaled = (perp_vector2[0] * width, perp_vector2[1] * width)
        point1 = (offpos_point[0] + perp_vector1_scaled[0], offpos_point[1] + perp_vector1_scaled[1])
        point2 = (offpos_point[0] + perp_vector2_scaled[0], offpos_point[1] + perp_vector2_scaled[1])
        return point2, point1

    def calculate_second_point(self, rpos, direction, alpha, length):
        direction_length = np.sqrt(direction[0]**2 + direction[1]**2)
        direction_unit = (direction[0] / direction_length, direction[1] / direction_length)
        alpha_rad = np.radians(alpha)
        rotated_vector = (
            direction_unit[0] * np.cos(alpha_rad) - direction_unit[1] * np.sin(alpha_rad),
            direction_unit[0] * np.sin(alpha_rad) + direction_unit[1] * np.cos(alpha_rad)
        )
        scaled_vector = (rotated_vector[0] * length, rotated_vector[1] * length)
        new_pos = (rpos[0] + scaled_vector[0], rpos[1] + scaled_vector[1])
        return new_pos

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def scaled_sigmoid(self, x, fmin, fmax):
        x_normalized = (x - (fmin + fmax) / 2) / ((fmax - fmin) / 2)
        return self.sigmoid(x_normalized)

class Joint:
    def __init__(self, pos=(0, 0), angle=0, radius=1.5, viz=None):
        self.angle = angle
        self.pos = pos
        self.radius = radius
        self.viz = viz if viz else Viz()

    def draw(self, ax=plt.gca(), radius=None):
        self.radius = radius if radius is not None else self.radius
        circle = patches.Circle(self.pos, self.radius, edgecolor=self.viz.linecolor, facecolor=self.viz.facecolor, linewidth=self.viz.linewidth)
        ax.add_patch(circle)

class Neuron:
    def __init__(self, firing=False, pos=(0, 0), radius=0.8, linewidth=0.1):
        self.firing = firing
        self.pos = pos
        self.linewidth = linewidth
        self.radius = radius

    def draw(self, ax=plt.gca()):
        color = 'red' if self.firing else 'blue'
        circle = patches.Circle(self.pos, self.radius, edgecolor=None, facecolor=color, linewidth=self.linewidth)
        ax.add_patch(circle)

class Eyeball(Object):
    def __init__(self, rpos=(0, 0), direction=(0, 1), angle=0, hemi=0, width=1.5, height=3, linewidth=1):
        super().__init__()
        self.hemi = hemi
        self.rpos = rpos
        self.direction = direction
        self.pos = (0, 0)
        self.angle = angle
        self.morph.height = height
        self.morph.width = width
        self.viz.linewidth = linewidth
        self.offset = 2
        self.angle_offset = 30
        self.update_position()

    def update_position(self, rpos=None, direction=None, angle=None, offset=None):
        self.angle = angle if angle is not None else self.angle
        self.offset = offset if offset is not None else self.offset
        self.rpos = rpos or self.rpos
        self.direction = direction or self.direction
        angle = (self.angle + self.angle_offset)
        if self.hemi:
            self.pos = self.calculate_second_point(self.rpos, self.direction, angle, self.offset)
        else:
            self.pos = self.calculate_second_point(self.rpos, self.direction, -angle, self.offset)

    def draw(self, ax=plt.gca()):
        color = 'yellow'
        if self.hemi:
            angle = self.calculate_angle(self.direction) + (self.angle + self.angle_offset)
        else:
            angle = self.calculate_angle(self.direction) - (self.angle + self.angle_offset)
        ellipse = patches.Ellipse(self.pos, width=self.morph.width, height=self.morph.height, angle=angle, edgecolor='black', facecolor=color, linewidth=self.viz.linewidth, zorder=3)
        ax.add_patch(ellipse)

class Eye(Object):
    def __init__(self, rpos=(0, 0), direction=(0, 1), offset=5, radius=2.5, hemi=0):
        super().__init__()
        self.hemi = hemi
        self.offset = offset
        self.offpos = 0
        self.rpos = rpos
        self.direction = direction
        self.pos = (0, 0)
        self.morph.width = radius
        self.neuron = [Neuron(), Neuron()]
        viz = Viz(facecolor=(0.5, 0.2, 0.2))
        self.eyeball = Eyeball(self.rpos, self.direction, hemi=self.hemi)
        self.eyeball.viz = viz
        self.eyeball.update_position()
        self.update_position()

    def update_position(self, rpos=None, direction=None):
        self.rpos = rpos or self.rpos
        self.direction = direction or self.direction

        direction_length = np.sqrt(self.direction[0]**2 + self.direction[1]**2)
        direction_unit = (self.direction[0] / direction_length, self.direction[1] / direction_length)

        offpos_point = (self.rpos[0] + direction_unit[0] * self.offpos, self.rpos[1] + direction_unit[1] * self.offpos)

        if self.hemi:
            perp_vector1 = (-direction_unit[1], direction_unit[0])
            perp_vector1_scaled = (perp_vector1[0] * self.offset, perp_vector1[1] * self.offset)
            self.pos = (offpos_point[0] + perp_vector1_scaled[0], offpos_point[1] + perp_vector1_scaled[1])
            self.eyeball.update_position(self.pos, perp_vector1_scaled)
            pos = self.calculate_perpendicular_points(self.pos, perp_vector1_scaled, 2, -3)
            self.neuron[0].pos = pos[1]
            self.neuron[0].firing = True
            self.neuron[1].pos = pos[0]
        else:
            perp_vector2 = (direction_unit[1], -direction_unit[0])
            perp_vector2_scaled = (perp_vector2[0] * self.offset, perp_vector2[1] * self.offset)
            self.pos = (offpos_point[0] + perp_vector2_scaled[0], offpos_point[1] + perp_vector2_scaled[1])
            self.eyeball.update_position(self.pos, perp_vector2_scaled)
            pos = self.calculate_perpendicular_points(self.pos, perp_vector2_scaled, 2, -3)
            self.neuron[0].pos = pos[0]
            self.neuron[0].firing = True
            self.neuron[1].pos = pos[1]

    def draw(self, ax=plt.gca()):
        eye = patches.Circle(self.pos, radius=self.morph.width, edgecolor=self.viz.linecolor, facecolor=self.viz.facecolor, zorder=3)
        ax.add_patch(eye)
        self.eyeball.draw(ax)
        self.neuron[0].draw(ax)
        self.neuron[1].draw(ax)

class Brain(Object):
    def __init__(self, rpos=(0, 0), rdirection=(0, 1), num_neurons=10):
        super().__init__()
        self.num_neurons = num_neurons
        self.rpos = rpos
        self.rdirection = rdirection
        self.neurons = [Neuron() for _ in range(num_neurons)]

    def update_position(self, rpos=None, rdirection=None):
        self.rpos = rpos or self.rpos
        self.rdirection = rdirection or self.rdirection
        pass

    def draw(self, ax=plt.gca()):
        pass

class Head(Object):
    def __init__(self, pos=(0, 0), direction=(0, 1), num_neurons=10):
        super().__init__()
        self.pos = pos
        self.direction = direction
        self.eye_offset = 5
        self.joint_offset = 8
        self.eyes = [Eye(pos, direction, self.eye_offset, hemi=0), Eye(pos, direction, self.eye_offset, hemi=1)]
        self.brain = Brain(pos, direction, num_neurons)
        self.joint = Joint()
        self.update_position()

    def update_position(self, pos=None, direction=None):
        self.pos = pos or self.pos
        self.direction = direction or self.direction
        direction_length = np.sqrt(self.direction[0]**2 + self.direction[1]**2)
        self.direction = (self.direction[0] / direction_length, self.direction[1] / direction_length)
        self.joint.pos = self.pos[0] + self.joint_offset * self.direction[0], self.pos[1] + self.joint_offset * self.direction[1]
        self.eyes[0].update_position(self.pos, self.direction)
        self.eyes[1].update_position(self.pos, self.direction)
        self.brain.update_position(self.pos, self.direction)

    def draw(self, ax=plt.gca()):
        for eye in self.eyes:
            eye.draw(ax)
        self.brain.draw(ax)
        angle = self.calculate_angle(self.direction)
        head_arc = patches.Arc(self.pos, width=self.eye_offset*2, height=self.eye_offset*2, angle=angle+90, theta1=0, theta2=180, edgecolor='black', facecolor='none', linewidth=1, zorder=2)
        ax.add_patch(head_arc)
        

class Bone(Object):
    def __init__(self, pos=(0, 0), rdirection=(0, 1), angle=0, bone_length=12, linewidth=3):
        super().__init__()
        self.pos = pos
        self.rdirection = rdirection
        self.direction = (0, 0)
        self.angle = angle
        self.length = bone_length
        self.neuron_offset = linewidth * 0.6
        self.viz.linewidth = linewidth
        viz = Viz()
        viz.facecolor = (0.5, 0.5, 0.5)
        self.joint = [Joint(pos, viz=viz, radius=linewidth/4), Joint(viz=viz, radius=linewidth/4)]
        self.neuron = [Neuron(), Neuron()]
        self.update_position()

    def update_position(self, pos=None, rdirection=None, angle=None):
        self.pos = pos or self.pos
        self.rdirection = rdirection or self.rdirection
        self.angle = angle if angle is not None else self.angle
        self.joint[0].pos = self.pos
        self.joint[1].pos = self.calculate_second_point(self.pos, self.rdirection, self.angle, self.length)
        self.direction = (self.joint[1].pos[0] - self.joint[0].pos[0], self.joint[1].pos[1] - self.joint[0].pos[1])
        direction_length = np.sqrt(self.direction[0]**2 + self.direction[1]**2)
        self.direction = (self.direction[0] / direction_length, self.direction[1] / direction_length)
        self.morph.center = ((self.joint[0].pos[0] + self.joint[1].pos[0]) / 2, (self.joint[0].pos[1] + self.joint[1].pos[1]) / 2)
        self.neuron[0].pos, self.neuron[1].pos = self.calculate_perpendicular_points(self.joint[0].pos, self.direction, self.neuron_offset, offpos=5)

    def draw(self, ax=plt.gca()):
        line = patches.FancyArrowPatch(self.joint[0].pos, self.joint[1].pos, linewidth=self.viz.linewidth, color=self.viz.linecolor, arrowstyle=self.viz.linetype)
        ax.add_patch(line)
        self.joint[0].draw(ax)
        # self.joint[1].draw(ax)
        self.neuron[0].draw(ax)
        self.neuron[1].draw(ax)

class Spine(Object):
    def __init__(self, pos=(0, 0), direction=(0, 1), num_bones=4):
        super().__init__()
        self.pos = pos
        self.direction = direction
        self.num_bones = num_bones
        self.bones = []
        widths = [2.5, 2, 1.5, 1.2, 1]
        self.bones = [Bone(linewidth=widths[i]) for i in range(num_bones)]
        for i in range(len(self.bones)):
            self.bones[i].update_position(self.bones[i - 1].joint[1].pos if i > 0 else self.pos, self.bones[i - 1].direction if i > 0 else self.direction, self.bones[i].angle)

    def update_position(self, pos=None, direction=None):
        self.pos = pos or self.pos
        self.direction = direction or self.direction

        head_angle = np.arctan2(self.direction[1], self.direction[0])
        angles = []
        for i, bone in enumerate(self.bones):
            left = bone.neuron[0].firing
            right = bone.neuron[1].firing
            angle = 10 if left != right else 0
            angles.append(angle if right else -angle)
        for i in range(len(self.bones)):
            if i == 0:
                self.bones[i].angle = head_angle + angles[i]
            else:
                self.bones[i].angle = self.bones[i - 1].angle + angles[i]
            self.bones[i].update_position(self.bones[i - 1].joint[1].pos if i > 0 else self.pos, self.bones[i - 1].direction if i > 0 else self.direction, self.bones[i].angle)

    def draw(self, ax=plt.gca()):
        for bone in self.bones:
            bone.draw(ax)

class Heart(Object):
    def __init__(self, pos=(0, 0), size=1.5, heartrate=120):
        super().__init__()
        self.pos = pos
        self.size = size
        self.heartrate = heartrate
        self.beat_phase = 0
        self.beat_period = 60 / self.heartrate  # 60 seconds divided by heart rate
        t = np.arange(0, 2 * np.pi, 0.1)
        self.x = (16 * np.sin(t)**3)
        self.y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        self.angle=0

    def rotate_coordinates(self, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rotated = cos_angle * self.x - sin_angle * self.y
        y_rotated = sin_angle * self.x + cos_angle * self.y
        return x_rotated, y_rotated

    def update_position(self, pos,direction):
        self.pos=self.calculate_perpendicular_points(pos, direction, 5)[0]
        self.angle=self.calculate_angle(direction)

    def beat(self):
        self.beat_phase = (self.beat_phase + 1) % (self.beat_period * 60)
        if self.beat_phase < (self.beat_period * 30):
            scale = 1 + (self.beat_phase / (self.beat_period * 30)) * 0.5
            color_intensity = self.beat_phase / (self.beat_period * 30)
        else:
            scale = 1.5 - ((self.beat_phase - (self.beat_period * 30)) / (self.beat_period * 30)) * 0.5
            color_intensity = ((self.beat_period * 60) - self.beat_phase) / (self.beat_period * 30)
        self.size_scaled = self.size * scale
        self.color = (1, 0, 0, color_intensity)

    def draw(self, ax=plt.gca()):
        x,y=self.rotate_coordinates(np.radians(self.angle))
        x_scaled = self.pos[0] + self.size_scaled * x / max(abs(x))
        y_scaled = self.pos[1] + self.size_scaled * y / max(abs(y))
        heart_shape = np.vstack((x_scaled, y_scaled)).T
        polygon = patches.Polygon(heart_shape, closed=True, facecolor=self.color, edgecolor='black')
        ax.add_patch(polygon)



class Body(Object):
    def __init__(self, pos=(0, 0), direction=(0, 1), num_bones=4):
        super().__init__()
        self.pos = pos
        self.direction = direction
        self.spine = Spine(pos, direction, num_bones)
        self.heartrate=120
        self.heart = Heart(self.heartrate)

    def update_position(self, pos=None, direction=None):
        self.pos = pos or self.pos
        self.direction = direction or self.direction
        self.spine.update_position(self.pos, self.direction)
        self.heart.update_position(self.spine.bones[0].joint[0].pos,self.spine.bones[0].direction)

    def update_energy_consumption(self):
        pass

    def draw(self, ax=plt.gca()):
        self.spine.draw(ax)
        self.heart.beat()
        self.heart.draw(ax)

class ZFish(Object):
    def __init__(self, pos=[0, 0], direction=(0, 1)):
        super().__init__()
        self.num_bones = 4
        self.num_neurons = 20
        self.pos = pos
        self.tail_prepos = [0, 0]
        self.speed = 2
        self.speedfactor = 2
        self.speed_baseline = 2
        self.smoothfactor = 10
        self.direction = direction
        self.head = Head(self.pos, self.direction, self.num_neurons)
        self.head.update_position()
        self.body = Body(self.head.joint.pos, self.direction, self.num_bones)
        self.body.update_position()

    def update_random_firing(self):
        muscle_neuron_values = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(self.num_bones)]
        for i, (left, right) in enumerate(muscle_neuron_values):
            self.body.spine.bones[i].neuron[0].firing = bool(left)
            self.body.spine.bones[i].neuron[1].firing = bool(right)
        self.update_position()

    def calculate_direction(self):
        positions = []
        positions.append(self.head.pos)
        positions.append(self.head.joint.pos)
        pos = [bone.joint[1].pos for bone in self.body.spine.bones]
        positions.extend(pos)
        points = np.array(positions)
        tck, u = splprep([points[:, 0], points[:, 1]], s=self.smoothfactor)
        dx, dy = splev(0, tck, der=1)
        direction_vector = np.array([dx, dy])
        direction = direction_vector / np.linalg.norm(direction_vector)
        self.direction = direction.tolist()

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
        self.pos = pos or self.pos
        self.direction = direction or self.direction
        self.calculate_force()
        if direction is None:
            self.calculate_direction()
            self.pos[0] = self.pos[0] - self.speed * self.direction[0]
            self.pos[1] = self.pos[1] - self.speed * self.direction[1]
        self.tail_prepos = self.body.spine.bones[-1].joint[1].pos
        self.head.update_position(self.pos, self.direction)
        self.body.update_position(self.head.joint.pos, self.head.direction)

    def draw_skin(self, ax):
        linewidth = 1
        color = 'black'
        pointsL = []
        pointsR = []
        pointsL.append(self.head.eyes[0].pos)
        pointsR.append(self.head.eyes[1].pos)

        offset = [7, 5, 3, 2, 1]
        bone = self.body.spine.bones[0]
        p1, p2 = self.calculate_perpendicular_points(bone.joint[0].pos, bone.direction, offset[0])
        pointsL.append(p1)
        pointsR.append(p2)

        for bone, off in zip(self.body.spine.bones, offset[1:]):
            p1, p2 = self.calculate_perpendicular_points(bone.joint[1].pos, bone.direction, off)
            pointsL.append(p1)
            pointsR.append(p2)

        bone = self.body.spine.bones[-1]
        offset = 10
        tendpos = (bone.joint[1].pos[0] + offset * bone.direction[0], bone.joint[1].pos[1] + offset * bone.direction[1])
        p1, p2 = self.calculate_perpendicular_points(tendpos, bone.direction, 0.5)
        pointsL.append(p1)
        pointsR.append(p2)

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)

        points = np.array(pointsL)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        x_fine, y_fine = splev(np.linspace(0, 1, 100), tck)
        ax.plot(x_fine, y_fine, color=color, linewidth=linewidth)

        points = np.array(pointsR)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        x_fine, y_fine = splev(np.linspace(0, 1, 100), tck)
        ax.plot(x_fine, y_fine, color=color, linewidth=linewidth)

    def draw(self, ax=plt.gca()):
        self.body.draw(ax)
        self.head.draw(ax)
        self.draw_skin(ax)

class Renderer:
    def __init__(self, zfishes, ax):
        self.ax = ax
        self.zfishes = zfishes
        self.scope_size = 300
        self.center = [0, 0]
        self.init()

    def init(self):
        self.ax.clear()
        self.ax.set_xlim(self.center[0] - self.scope_size / 2, self.center[0] + self.scope_size / 2)
        self.ax.set_ylim(self.center[1] - self.scope_size / 2, self.center[1] + self.scope_size / 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':')

    def render(self):
        self.init()
        for index, zfish in enumerate(self.zfishes):
            zfish.update_random_firing()
            if index == 0:
                x_center1, y_center1 = zfish.head.pos
                if abs(x_center1 - self.center[0]) > self.scope_size / 2 or abs(y_center1 - self.center[1]) > self.scope_size / 2:
                    self.center[0] = x_center1
                    self.center[1] = y_center1
            zfish.draw(self.ax)

        plt.draw()
        plt.pause(0.1)



class SeaWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.paths = []
        self.plants = []
        self.food_sources = []
        self.rocks = []
        self.load_areas = []

    def add_path(self, start, end):
        """Add a path from start to end."""
        self.paths.append((start, end))
        self._draw_path(start, end, value=1)

    def add_rock(self, location):
        """Add a rock at the specified location."""
        self.rocks.append(location)
        self.grid[location[1], location[0]] = 2

    def add_plant(self, location):
        """Add a sea plant at the specified location."""
        self.plants.append(location)
        self.grid[location[1], location[0]] = 3

    def add_food_source(self, location, food_type):
        """Add a food source at the specified location."""
        self.food_sources.append((location, food_type))
        self.grid[location[1], location[0]] = 4 + food_type

    def add_load_area(self, top_left, bottom_right):
        """Define a load area within the specified rectangle."""
        self.load_areas.append((top_left, bottom_right))
        self._fill_rectangle(top_left, bottom_right, value=10)

    def _draw_path(self, start, end, value):
        """Draw a path on the grid."""
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            self.grid[y1, x1] = value
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _fill_rectangle(self, top_left, bottom_right, value):
        """Fill a rectangle area with the specified value."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        self.grid[y1:y2+1, x1:x2+1] = value

    def visualize(self):
        """Visualize the environment."""
        cmap = plt.cm.get_cmap('viridis', 12)
        plt.imshow(self.grid, cmap=cmap, origin='lower')
        plt.colorbar(ticks=range(11), label='Element')
        plt.clim(-0.5, 10.5)
        plt.show()

# Create the sea world
sea_world = SeaWorld(1000, 1000)

# Add paths
sea_world.add_path((10, 10), (990, 990))
sea_world.add_path((990, 10), (10, 990))

# Add rocks in the first quadrant
for _ in range(50):
    sea_world.add_rock((random.randint(0, 499), random.randint(0, 499)))

# Add sea plants in the second quadrant
for _ in range(50):
    sea_world.add_plant((random.randint(500, 999), random.randint(0, 499)))

# Add food sources in the third and fourth quadrants
food_types = ['zooplankton', 'insects', 'phytoplankton', 'filamentous algae', 'vascular plant material']
for i, food in enumerate(food_types):
    for _ in range(10):
        sea_world.add_food_source((random.randint(0, 499), random.randint(500, 999)), i)
    for _ in range(10):
        sea_world.add_food_source((random.randint(500, 999), random.randint(500, 999)), i)

# Add load areas
sea_world.add_load_area((200, 200), (300, 300))
sea_world.add_load_area((700, 700), (800, 800))

# Visualize the sea world
sea_world.visualize()

# Define ZFish and related classes (Head, Body, Spine, etc.) from your original code here

class Renderer:
    def __init__(self, zfishes, ax):
        self.ax = ax
        self.zfishes = zfishes
        self.scope_size = 200
        self.center = [0, 0]
        self.init()

    def init(self):
        self.ax.clear()
        self.ax.set_xlim(self.center[0] - self.scope_size / 2, self.center[0] + self.scope_size / 2)
        self.ax.set_ylim(self.center[1] - self.scope_size / 2, self.center[1] + self.scope_size / 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':')

    def render(self):
        self.init()
        for index, zfish in enumerate(self.zfishes):
            zfish.update_random_firing()
            if index == 0:
                x_center1, y_center1 = zfish.head.pos
                if abs(x_center1 - self.center[0]) > self.scope_size / 2 or abs(y_center1 - self.center[1]) > self.scope_size / 2:
                    self.center[0] = x_center1
                    self.center[1] = y_center1
            zfish.draw(self.ax)

        plt.draw()
        plt.pause(0.1)

# Number of fishes
N = 10

# Create N zebrafish with random starting positions and directions
zfishes = []
for _ in range(N):
    start_pos = [random.randint(0, 999), random.randint(0, 999)]
    direction = (random.uniform(-1, 1), random.uniform(-1, 1))
    zfishes.append(ZFish(start_pos, direction))

# Create the figure and axis for plotting
fig, ax = plt.subplots()

# Turn on interactive plotting
plt.ion()

# Create the renderer with the list of fishes
renderer = Renderer(zfishes, ax)

# Function to handle key press events
def on_key(event):
    if event.key == 'q':
        plt.close(fig)

# Connect the key press event handler to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

# Main loop for rendering the environment and fishes
while plt.fignum_exists(fig.number):
    renderer.render()
