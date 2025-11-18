import math
import numpy as np

from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt


# ======================================================================
# WorldRenderer: draws fish, food, obstacles, boundary
# ======================================================================

class WorldRenderer(QtWidgets.QGraphicsView):
    def __init__(self, world, fish_list, body_list, physics_list,
                 width=900, height=700, scale=2.0):
        super().__init__()

        self.world = world
        self.fishes = fish_list          # list of brain outputs (decoded motor etc)
        self.bodies = body_list          # list of FishBody
        self.physics = physics_list      # list of FishPhysics
        
        self.scale_factor = scale

        # Qt Scene
        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        self.setFixedSize(width, height)
        self.setSceneRect(-width/2, -height/2, width, height)

        # drawing items
        self.fish_items = []     # list of QGraphicsPathItem for fish
        self.eye_items = []      # list of [left_eye_item, right_eye_item]
        self.food_items = []
        self.obstacle_items = []

        # setup initial drawing
        self.init_world()
        self.init_fish()


    # ==================================================================
    # Initialize world objects
    # ==================================================================
    def init_world(self):
        # foods
        for (fx, fy) in self.world.foods:
            item = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6)
            item.setBrush(QtGui.QColor(255, 100, 100))
            item.setPos(fx * self.scale_factor, -fy * self.scale_factor)
            self.scene.addItem(item)
            self.food_items.append(item)

        # obstacles
        for obs in self.world.obstacles:
            x, y, r = obs["x"], obs["y"], obs["r"]
            item = QtWidgets.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            item.setBrush(QtGui.QColor(80, 80, 80))
            item.setPos(x * self.scale_factor, -y * self.scale_factor)
            self.scene.addItem(item)
            self.obstacle_items.append(item)


    # ==================================================================
    # Initialize fish graphical objects
    # ==================================================================
    def init_fish(self):
        for _ in self.fishes:
            fish_path = QtWidgets.QGraphicsPathItem()
            fish_path.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1.8))
            fish_path.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
            self.scene.addItem(fish_path)

            # eyes: left & right
            left_eye = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6)
            right_eye = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6)
            left_eye.setBrush(QtGui.QColor(255, 255, 0))
            right_eye.setBrush(QtGui.QColor(255, 255, 0))
            self.scene.addItem(left_eye)
            self.scene.addItem(right_eye)

            self.fish_items.append(fish_path)
            self.eye_items.append([left_eye, right_eye])


    # ==================================================================
    # Update world objects (food, etc.)
    # ==================================================================
    def update_world(self):
        # update foods
        for idx, item in enumerate(self.food_items):
            if idx >= len(self.world.foods):
                item.hide()
                continue
            fx, fy = self.world.foods[idx]
            item.setPos(fx * self.scale_factor, -fy * self.scale_factor)


    # ==================================================================
    # Draw a single fish
    # ==================================================================
    def draw_fish(self, idx):
        body = self.bodies[idx]
        phys = self.physics[idx]

        # skeleton
        skel = body.compute_skeleton(phys.x, phys.y, phys.heading)

        # body edges
        left_edge, right_edge = body.compute_body_polylines(skel)

        # head polygon (rotated)
        head = body.head_pts.copy()
        c = math.cos(phys.heading)
        s = math.sin(phys.heading)
        for i in range(len(head)):
            x, y = head[i]
            gx = phys.x + x*c - y*s
            gy = phys.y + x*s + y*c
            head[i] = [gx, gy]

        # Build QPainterPath
        path = QtGui.QPainterPath()
        # start at head
        hx, hy = head[0]
        path.moveTo(hx * self.scale_factor, -hy * self.scale_factor)

        for i in range(1, len(head)):
            x, y = head[i]
            path.lineTo(x * self.scale_factor, -y * self.scale_factor)

        # left edge
        for (x, y) in left_edge:
            path.lineTo(x * self.scale_factor, -y * self.scale_factor)

        # right edge (reverse)
        for (x, y) in reversed(right_edge):
            path.lineTo(x * self.scale_factor, -y * self.scale_factor)

        # close head
        path.lineTo(hx * self.scale_factor, -hy * self.scale_factor)

        self.fish_items[idx].setPath(path)

        # draw eyes
        left_eye_item, right_eye_item = self.eye_items[idx]

        # eye positions relative to head
        exL = phys.x + (-5 * c - -3 * s)
        eyL = phys.y + (-5 * s + -3 * c)

        exR = phys.x + (5 * c - -3 * s)
        eyR = phys.y + (5 * s + -3 * c)

        left_eye_item.setPos(exL * self.scale_factor, -eyL * self.scale_factor)
        right_eye_item.setPos(exR * self.scale_factor, -eyR * self.scale_factor)


    # ==================================================================
    # Main update function for all fish
    # ==================================================================
    def update(self):
        self.update_world()

        for i in range(len(self.fishes)):
            self.draw_fish(i)

        self.scene.update()
