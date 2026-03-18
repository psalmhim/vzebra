import numpy as np
from ..core.physics import HydroPhysics
from ..core.kinematics import SpineModel
from ..core.morphology import Morphology
from ..core.transform import transform

class ZebrafishLarva:
    def __init__(self):
        self.x=300
        self.y=300
        self.heading=0.0

        self.jY=[0,-15,-30,-45,-60,-72,-88]
        self.jW=[14,8,7,5,3,2,1]

        self.spine_model=SpineModel(self.jY,self.jW)
        self.morph=Morphology(self.jW)
        self.physics=HydroPhysics()

        self.pigment_pts=self.morph.pigment()

        self.t=0

    def update(self):
        self.t+=1
        # compute spine centerline
        X,Y=self.spine_model.spine(self.t)
        spline=self.spine_model.spline_spine(X,Y)
        # compute thrust
        thrust=self.physics.compute_thrust(X,Y,self.t,amp=4.0)
        vx,vy=self.physics.update(self.heading,thrust)
        # move fish
        self.x += vx
        self.y += vy
        # soft wall
        if self.x<80 or self.x>520: self.heading+=np.pi*0.6
        if self.y<80 or self.y>520: self.heading+=np.pi*0.6
        # outputs for render
        self.spline=spline
        self.body=self.morph.body_outline(spline,self.jW)
        self.head=self.morph.head()
        self.fins=self.morph.fins(self.t)
        self.pig=self.pigment_pts
