from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np
from ..core.transform import transform
from ..core.shaders import body_brush

class ShoalRenderer(QtWidgets.QGraphicsView):
    def __init__(self, shoal):
        super().__init__()
        self.shoal = shoal

        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        # Enlarged Scene Area (1000x1000)
        self.scene = QtWidgets.QGraphicsScene(0, 0, 1000, 1000)
        self.setScene(self.scene)
        self.setSceneRect(0, 0, 1000, 1000)

        self.resetTransform()
        self.scale(1.0, 1.0)

        # ----------------------------
        # ★ VIEW FIT SCENE ★
        # 모든 물고기가 화면에 보이도록 강제
        # ----------------------------
        self.fitInView(
            self.scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )

        # items for each fish
        self.items = []
        for f in shoal.fish_list:
            body = QtWidgets.QGraphicsPathItem()
            body.setPen(QtGui.QPen(QtGui.QColor(20,20,20),2))
            body.setBrush(body_brush())

            head = QtWidgets.QGraphicsPathItem()
            head.setPen(QtGui.QPen(QtGui.QColor(20,20,20),2))
            head.setBrush(body_brush())

            eyeL = QtWidgets.QGraphicsEllipseItem()
            eyeR = QtWidgets.QGraphicsEllipseItem()
            for e in [eyeL, eyeR]:
                e.setBrush(QtGui.QBrush(QtGui.QColor(50,50,50)))
                e.setPen(QtGui.QPen(QtGui.QColor(20,20,20)))

            self.scene.addItem(body)
            self.scene.addItem(head)
            self.scene.addItem(eyeL)
            self.scene.addItem(eyeR)

            self.items.append((body, head, eyeL, eyeR, f))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30)
        QtCore.QTimer.singleShot(50, self.apply_fit)

    def apply_fit(self):
        self.fitInView(
            self.scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
    def update_loop(self):
        self.shoal.step()

        for body_item, head_item, eyeL, eyeR, f in self.items:

            # head
            H = transform(f.head, f.x, f.y, f.heading)
            hpath = QtGui.QPainterPath()
            hpath.moveTo(H[0,0], H[0,1])
            for i in range(1, len(H)):
                hpath.lineTo(H[i,0], H[i,1])
            head_item.setPath(hpath)

            # body
            B = transform(f.body, f.x, f.y, f.heading)
            bpath = QtGui.QPainterPath()
            bpath.moveTo(B[0,0], B[0,1])
            for i in range(1, len(B)):
                bpath.lineTo(B[i,0], B[i,1])
            body_item.setPath(bpath)

            # eyes (first 2 points of head)
            eyes = transform(f.head[:2,:], f.x, f.y, f.heading)
            r = 7
            eyeL.setRect(eyes[0,0]-r, eyes[0,1]-r, 2*r, 2*r)
            eyeR.setRect(eyes[1,0]-r, eyes[1,1]-r, 2*r, 2*r)
