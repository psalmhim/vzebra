import numpy as np

class Morphology:
    def __init__(self, jW):
        self.jW = jW

    # curved head
    def head(self):
        return np.array([
            [  0,   6],
            [ -7,  -3],
            [  0,  -3],
            [  7,  -3],
            [  0,   6]
        ], dtype=float)

    def fins(self, t):
        flap = 4.0*np.sin(0.25*t)
        left = np.array([
            [-4,-10],
            [-12+flap,-20],
            [-4,-18]
        ])
        right = np.array([
            [4,-10],
            [12-flap,-20],
            [4,-18]
        ])
        return left, right

    def pigment(self, n=15):
        spots=[]
        for _ in range(n):
            y = np.random.uniform(-50,-5)
            x = np.random.uniform(-3,3)
            spots.append([x,y])
        return np.array(spots)

    def body_outline(self, spline, jW):
        """Envelope around spine."""
        body=[]
        # left side
        for i,(x,y) in enumerate(spline):
            w = jW[min(i,len(jW)-1)]/2
            body.append([x-w, y])
        # right side reversed
        for i,(x,y) in reversed(list(enumerate(spline))):
            w = jW[min(i,len(jW)-1)]/2
            body.append([x+w, y])
        return np.array(body)
