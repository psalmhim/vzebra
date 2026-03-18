# GPU fallback (optional)
try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    cp = None
    GPU_ENABLED = False

def xp(array):
    """Return Cupy array if GPU enabled, else Numpy."""
    if GPU_ENABLED:
        return cp.asarray(array)
    import numpy as np
    return np.asarray(array)

