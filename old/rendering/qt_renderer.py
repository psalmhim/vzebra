from PyQt6 import QtWidgets, QtGui, QtCore
from ..core.transform import transform
from ..core.shaders import body_brush, belly_brush

class FishRenderer(QtWidgets.QGraphicsView):
    def __init__(self, fish):
        super().__init__()
        self.fish=fish

        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        self.scene=QtWidgets.QGraphicsScene(0,0,600,600)
        self.setScene(self.scene)

        # add items
        self.body_item=QtWidgets.QGraphicsPathItem()
        self.body_item.setPen(QtGui.QPen(QtGui.QColor(20,20,20),2))
        self.body_item.setBrush(body_brush())
        self.scene.addItem(self.body_item)

        self.head_item=QtWidgets.QGraphicsPathItem()
        self.head_item.setPen(QtGui.QPen(QtGui.QColor(20,20,20),2))
        self.head_item.setBrush(body_brush())
        self.scene.addItem(self.head_item)

        self.fins_items=[
            QtWidgets.QGraphicsPathItem(),
            QtWidgets.QGraphicsPathItem()
        ]
        for f in self.fins_items:
            f.setBrush(QtGui.QBrush(QtGui.QColor(200,200,200,80)))
            f.setPen(QtGui.QPen(QtGui.QColor(30,30,30),1))
            self.scene.addItem(f)

        self.eyeL=QtWidgets.QGraphicsEllipseItem()
        self.eyeR=QtWidgets.QGraphicsEllipseItem()
        for e in [self.eyeL,self.eyeR]:
            e.setBrush(QtGui.QBrush(QtGui.QColor(50,50,50)))
            e.setPen(QtGui.QPen(QtGui.QColor(20,20,20)))
            self.scene.addItem(e)

        self.spot_items=[]

        timer=QtCore.QTimer()
        timer.timeout.connect(self.update_loop)
        timer.start(30)

    def update_loop(self):
        f=self.fish
        f.update()

        # head
        H=transform(f.head,f.x,f.y,f.heading)
        hpath=QtGui.QPainterPath()
        hpath.moveTo(H[0,0],H[0,1])
        for i in range(1,len(H)):
            hpath.lineTo(H[i,0],H[i,1])
        self.head_item.setPath(hpath)

        # body
        B=transform(f.body,f.x,f.y,f.heading)
        bpath=QtGui.QPainterPath()
        bpath.moveTo(B[0,0],B[0,1])
        for i in range(1,len(B)):
            bpath.lineTo(B[i,0],B[i,1])
        self.body_item.setPath(bpath)

        # fins
        lf,rf=f.fins
        lf=transform(lf,f.x,f.y,f.heading)
        rf=transform(rf,f.x,f.y,f.heading)
        lp=QtGui.QPainterPath()
        lp.moveTo(lf[0,0],lf[0,1])
        lp.lineTo(lf[1,0],lf[1,1])
        lp.lineTo(lf[2,0],lf[2,1])
        lp.closeSubpath()
        self.fins_items[0].setPath(lp)

        rp=QtGui.QPainterPath()
        rp.moveTo(rf[0,0],rf[0,1])
        rp.lineTo(rf[1,0],rf[1,1])
        rp.lineTo(rf[2,0],rf[2,1])
        rp.closeSubpath()
        self.fins_items[1].setPath(rp)

        # eyes
        eyes = transform(f.head[:2,:], f.x, f.y, f.heading)
        r=7
        self.eyeL.setRect(eyes[0,0]-r,eyes[0,1]-r,2*r,2*r)
        self.eyeR.setRect(eyes[1,0]-r,eyes[1,1]-r,2*r,2*r)

