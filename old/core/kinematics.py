import numpy as np

def catmull_rom(p0, p1, p2, p3, n=12):
    """Cubic Catmull-Rom spline segment."""
    pts = []
    for t in np.linspace(0,1,n):
        t2 = t*t
        t3 = t2*t
        x = 0.5*((2*p1[0]) +
                 (-p0[0]+p2[0])*t +
                 (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                 (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y = 0.5*((2*p1[1]) +
                 (-p0[1]+p2[1])*t +
                 (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                 (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        pts.append([x,y])
    return np.array(pts)

class SpineModel:
    def __init__(self, jY, jW):
        self.jY = np.array(jY)
        self.jW = np.array(jW)
        self.n = len(self.jY)

    def spine(self, t, base_amp=4.0):
        """Compute center spine x coords for traveling wave."""
        X = []
        Y = self.jY
        for i,y in enumerate(Y):
            phase = 0.25*y + 0.20*t
            amp_i = base_amp*(i/(self.n-1))**1.3
            X.append(amp_i*np.sin(phase))
        return np.array(X), Y

    def spline_spine(self, X, Y):
        """Create smooth full-body spine using Catmull-Rom."""
        pts = np.stack([X,Y], axis=1)
        spline = []
        # pad endpoints
        P = np.vstack([pts[0], pts, pts[-1]])
        for i in range(len(pts)-1):
            seg = catmull_rom(P[i], P[i+1], P[i+2], P[i+3])
            spline.append(seg)
        return np.vstack(spline)

