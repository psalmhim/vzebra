import sys
import numpy as np
import math
from PyQt6 import QtWidgets, QtGui, QtCore


# ============================================================
# UTILITY
# ============================================================
def transform(pts, x, y, heading):
    ang = heading - math.pi/2
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    out = pts @ R.T
    out[:, 0] += x
    out[:, 1] += y
    return out


# ============================================================
# PHYSICS
# ============================================================
class HydroPhysics:
    def __init__(self):
        self.vx = 0
        self.vy = 0
        self.drag = 0.04

    def update(self, heading, thrust):
        ax = thrust * math.cos(heading)
        ay = thrust * math.sin(heading)

        self.vx += ax
        self.vy += ay

        self.vx *= (1 - self.drag)
        self.vy *= (1 - self.drag)

        return self.vx, self.vy


# ============================================================
# SINGLE ZEBRAFISH LARVA
# ============================================================
class ZebrafishLarva:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.t = 0

        self.jY = np.array([0, -15, -30, -45, -60, -75, -90])
        self.jW = np.array([14, 8, 7, 5, 3, 2, 1], dtype=float)

        self.physics = HydroPhysics()

        self.head_poly = np.array([
            [0, 6],
            [-7, -3],
            [0, -3],
            [7, -3],
            [0, 6]
        ], dtype=float)

        self.eyes = np.array([
            [-8, -2],
            [8, -2]
        ], dtype=float)

    def update(self):
        self.t += 1

        X = []
        for i, y in enumerate(self.jY):
            amp = 4 * (i / (len(self.jY) - 1)) ** 1.1
            phase = 0.22 * y + 0.15 * self.t
            X.append(amp * math.sin(phase))
        X = np.array(X)
        Y = self.jY

        left = np.stack([X - self.jW/2, Y], axis=1)
        right = np.stack([X + self.jW/2, Y], axis=1)
        self.body = np.vstack([left, right[::-1]])

        thrust = np.mean(np.abs(np.diff(X))) * 0.25

        vx, vy = self.physics.update(self.heading, thrust)
        self.x += vx
        self.y += vy


# ============================================================
# SHOAL (Reynolds + Couzin)
# ============================================================
class Shoal:
    def __init__(self, N=12):
        self.N = N
        self.fish = [ZebrafishLarva() for _ in range(N)]

        for f in self.fish:
            f.x = np.random.uniform(-200, 200)
            f.y = np.random.uniform(-200, 200)
            f.heading = np.random.uniform(-math.pi, math.pi)

        self.ZOR = 25
        self.ZOA = 60
        self.ZOAT = 120

        self.w_sep = 3.0
        self.w_align = 0.8
        self.w_coh = 0.5

        self.steer_gain = 0.03

    def step(self):
        pos = np.array([[f.x, f.y] for f in self.fish])
        head = np.array([f.heading for f in self.fish])

        for i, f in enumerate(self.fish):
            p = pos[i]

            sep = np.zeros(2)
            coh = np.zeros(2)
            align_list = []

            for j in range(self.N):
                if i == j:
                    continue
                p2 = pos[j]
                v = p2 - p
                d = np.linalg.norm(v)
                if d < 1e-6:
                    continue

                if d < self.ZOR:
                    sep -= v / (d**2)
                elif d < self.ZOA:
                    align_list.append(head[j])
                elif d < self.ZOAT:
                    coh += v / d

            sep_dir = math.atan2(sep[1], sep[0]) if np.linalg.norm(sep) > 1e-5 else 0
            coh_dir = math.atan2(coh[1], coh[0]) if np.linalg.norm(coh) > 1e-5 else 0
            
            if align_list:
                avg = np.mean(align_list)
                align_dir = math.atan2(math.sin(avg - f.heading),
                                       math.cos(avg - f.heading))
            else:
                align_dir = 0

            steer = self.w_sep*sep_dir + self.w_align*align_dir + self.w_coh*coh_dir
            f.heading += self.steer_gain * steer
            f.update()


# ============================================================
# RENDERER
# ============================================================
class ShoalRenderer(QtWidgets.QGraphicsView):
    def __init__(self, shoal):
        super().__init__()
        self.shoal = shoal

        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Huge scene area
        self.scene = QtWidgets.QGraphicsScene(-2000, -2000, 4000, 4000)
        self.setScene(self.scene)
        self.setSceneRect(-2000, -2000, 4000, 4000)

        # No transform distortion
        self.setTransform(QtGui.QTransform())

        self.items = []
        for f in shoal.fish:
            body = QtWidgets.QGraphicsPathItem()
            body.setPen(QtGui.QPen(QtGui.QColor(20,20,20), 2))
            body.setBrush(QtGui.QBrush(QtGui.QColor(200,200,200,60)))

            head = QtWidgets.QGraphicsPathItem()
            head.setPen(QtGui.QPen(QtGui.QColor(20,20,20), 2))
            head.setBrush(QtGui.QBrush(QtGui.QColor(200,200,200,80)))

            eyeL = QtWidgets.QGraphicsEllipseItem()
            eyeR = QtWidgets.QGraphicsEllipseItem()
            for e in [eyeL, eyeR]:
                e.setBrush(QtGui.QBrush(QtGui.QColor(30,30,30)))
                e.setPen(QtGui.QPen(QtGui.QColor(20,20,20)))

            self.scene.addItem(body)
            self.scene.addItem(head)
            self.scene.addItem(eyeL)
            self.scene.addItem(eyeR)

            self.items.append((body, head, eyeL, eyeR, f))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30)

    def update_loop(self):
        self.shoal.step()

        xs = []
        ys = []

        for body_item, head_item, eyeL, eyeR, f in self.items:
            xs.append(f.x)
            ys.append(f.y)

            H = transform(f.head_poly, f.x, f.y, f.heading)
            hp = QtGui.QPainterPath()
            hp.moveTo(H[0,0], H[0,1])
            for i in range(1, len(H)):
                hp.lineTo(H[i,0], H[i,1])
            head_item.setPath(hp)

            B = transform(f.body, f.x, f.y, f.heading)
            bp = QtGui.QPainterPath()
            bp.moveTo(B[0,0], B[0,1])
            for i in range(1, len(B)):
                bp.lineTo(B[i,0], B[i,1])
            bp.closeSubpath()
            body_item.setPath(bp)

            E = transform(f.eyes, f.x, f.y, f.heading)
            r = 4
            eyeL.setRect(E[0,0]-r, E[0,1]-r, 2*r, 2*r)
            eyeR.setRect(E[1,0]-r, E[1,1]-r, 2*r, 2*r)

        # Camera follow shoal center
        cx = np.mean(xs)
        cy = np.mean(ys)
        self.centerOn(cx, cy)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    shoal = Shoal(N=12)
    win = ShoalRenderer(shoal)
    win.show()
    sys.exit(app.exec())
