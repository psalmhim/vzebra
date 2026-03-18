import numpy as np

def transform(pts, x, y, heading):
    ang = heading - np.pi/2
    c,s = np.cos(ang), np.sin(ang)
    R = np.array([[c,-s],[s,c]])
    out = pts @ R.T
    out[:,0] += x
    out[:,1] += y
    return out
