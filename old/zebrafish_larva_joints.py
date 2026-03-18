import sys, math, random
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt    # ★★★ 반드시 추가 ★★★

# ======================================================
# Smooth Bezier Helper
# ======================================================
def quad_bezier(p0, p1, p2, n=20):
    pts = []
    for t in np.linspace(0,1,n):
        x = (1-t)**2 * p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        pts.append([x,y])
    return np.array(pts)


# ======================================================
# Main Fish Model
# ======================================================
class ZebrafishLarva:
    def __init__(self):
        # position
        self.x = 300
        self.y = 300
        self.heading = 0.0
        self.speed = 2.0

        # tail wave
        self.t = 0
        self.wave_speed = 0.20
        self.wave_freq  = 0.25
        self.base_amp   = 4.0

        # body joints
        self.jY = np.array([0, -15, -30, -45, -60, -72, -88], dtype=float)
        self.jW = np.array([14,8,7,5,3,2,1], dtype=float)

        self.noise_gain = 0.003
        self.saccade_interval = (130,280)
        self.next_sacc = random.randint(*self.saccade_interval)

    # --------------------------------------------------
    # Head using Bezier (smooth curved head)
    # --------------------------------------------------
    def head_path(self):
        # upper curve
        p0 = np.array([0,6])
        p1 = np.array([-7,-3])
        p2 = np.array([0,-3])

        # mirrored
        p3 = np.array([7,-3])

        left = quad_bezier(p0,p1,p2,n=20)
        right = quad_bezier(p2,p3,p0,n=20)

        return np.vstack([left, right])

    # --------------------------------------------------
    # Body outline using joints + spline-like smoothing
    # --------------------------------------------------
    def body_outline(self):
        pts = []
        # left side
        for i in range(len(self.jY)):
            yi = self.jY[i]
            amp = self.base_amp * (i/(len(self.jY)-1))**1.3
            phase = self.wave_freq*yi + self.wave_speed*self.t
            offset = amp * math.sin(phase)
            w = self.jW[i]/2
            pts.append([offset - w, yi])

        # right side reversed
        for i in reversed(range(len(self.jY))):
            yi = self.jY[i]
            amp = self.base_amp * (i/(len(self.jY)-1))**1.3
            phase = self.wave_freq*yi + self.wave_speed*self.t
            offset = amp * math.sin(phase)
            w = self.jW[i]/2
            pts.append([offset + w, yi])

        pts = np.array(pts)
        return pts

    # --------------------------------------------------
    # Eyes (with shading)
    # --------------------------------------------------
    def eyes(self):
        return np.array([
            [-8, -2],
            [ 8, -2]
        ], dtype=float)

    # --------------------------------------------------
    # Pectoral fins (animated)
    # --------------------------------------------------
    def pectoral_fins(self):
        # small symmetric triangles
        flap = 4.0 * math.sin(self.t * 0.25)

        left_fin = np.array([
            [-4, -10],
            [-12 + flap, -20],
            [-4, -18],
        ])

        right_fin = np.array([
            [4, -10],
            [12 - flap, -20],
            [4, -18],
        ])

        return left_fin, right_fin

    # --------------------------------------------------
    # Pigmentation dots
    # --------------------------------------------------
    def pigment(self):
        # random speckle positions (in body coordinates)
        spots = []
        for i in range(12):
            y = random.uniform(-40,-5)
            x = random.uniform(-3,3)
            spots.append([x,y])
        return np.array(spots)

    # --------------------------------------------------
    # Movement update
    # --------------------------------------------------
    def update(self):
        self.t += 1

        self.heading += random.gauss(0,self.noise_gain)
        if self.t >= self.next_sacc:
            self.heading += random.uniform(-0.3,0.3)
            self.next_sacc = self.t + random.randint(*self.saccade_interval)

        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)

        if not (80 < self.x < 520): self.heading += math.pi*0.6
        if not (80 < self.y < 520): self.heading += math.pi*0.6

    # --------------------------------------------------
    # Rotate + Translate
    # --------------------------------------------------
    def transform(self, pts):
        ang = self.heading - math.pi/2
        c,s = math.cos(ang), math.sin(ang)
        R = np.array([[c,-s],[s,c]])
        pts2 = pts @ R.T
        pts2[:,0] += self.x
        pts2[:,1] += self.y
        return pts2


# ======================================================
# QGraphicsScene-based Renderer
# ======================================================
class FishWindow(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()

        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        self.scene = QtWidgets.QGraphicsScene(0,0,600,600)
        self.setScene(self.scene)

        # items
        self.head_item = QtWidgets.QGraphicsPathItem()
        self.body_item = QtWidgets.QGraphicsPathItem()
        self.fin_left_item = QtWidgets.QGraphicsPathItem()
        self.fin_right_item = QtWidgets.QGraphicsPathItem()
        self.eye_items = [QtWidgets.QGraphicsEllipseItem(),
                          QtWidgets.QGraphicsEllipseItem()]
        self.spots_items = []

        # styling
        black_pen = QtGui.QPen(QtGui.QColor(20,20,20), 2)
        for item in [self.head_item, self.body_item,
                     self.fin_left_item, self.fin_right_item]:
            item.setPen(black_pen)
            item.setBrush(QtGui.QBrush(Qt.BrushStyle.NoBrush))
            self.scene.addItem(item)

        for eye in self.eye_items:
            eye.setBrush(QtGui.QBrush(QtGui.QColor(50,50,50)))
            eye.setPen(QtGui.QPen(QtGui.QColor(20,20,20)))
            self.scene.addItem(eye)

        self.fish = ZebrafishLarva()

        # pigment spots
        self.spot_positions = self.fish.pigment()
        self.spots_items = []
        for s in self.spot_positions:
            dot = QtWidgets.QGraphicsEllipseItem()
            dot.setRect(0,0,4,4)
            dot.setBrush(QtGui.QBrush(QtGui.QColor(30,30,30)))
            dot.setPen(QtGui.QPen(QtGui.QColor(20,20,20)))
            self.scene.addItem(dot)
            self.spots_items.append(dot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30)

    def update_loop(self):
        f = self.fish
        f.update()

        # head
        head = f.transform(f.head_path())
        path = QtGui.QPainterPath()
        path.moveTo(head[0,0], head[0,1])
        for i in range(1,len(head)):
            path.lineTo(head[i,0], head[i,1])
        self.head_item.setPath(path)

        # body
        body = f.transform(f.body_outline())
        bpath = QtGui.QPainterPath()
        bpath.moveTo(body[0,0], body[0,1])
        for i in range(1,len(body)):
            bpath.lineTo(body[i,0], body[i,1])
        self.body_item.setPath(bpath)

        # fins
        lf, rf = f.pectoral_fins()
        lf = f.transform(lf)
        rf = f.transform(rf)

        lfpath = QtGui.QPainterPath()
        lfpath.moveTo(lf[0,0], lf[0,1])
        lfpath.lineTo(lf[1,0], lf[1,1])
        lfpath.lineTo(lf[2,0], lf[2,1])
        lfpath.closeSubpath()
        self.fin_left_item.setPath(lfpath)

        rfpath = QtGui.QPainterPath()
        rfpath.moveTo(rf[0,0], rf[0,1])
        rfpath.lineTo(rf[1,0], rf[1,1])
        rfpath.lineTo(rf[2,0], rf[2,1])
        rfpath.closeSubpath()
        self.fin_right_item.setPath(rfpath)

        # eyes
        eyes = f.transform(f.eyes())
        r = 6
        for i,eye in enumerate(self.eye_items):
            eye.setRect(eyes[i,0]-r, eyes[i,1]-r, 2*r,2*r)

        # pigment
        for i,s in enumerate(self.spot_positions):
            p = f.transform(np.array([s]))[0]
            self.spots_items[i].setRect(p[0]-2, p[1]-2, 4,4)


# run
app = QtWidgets.QApplication(sys.argv)
win = FishWindow()
win.show()
sys.exit(app.exec())
