import torch
import numpy as np
import math

def cast_ray(world, origin, dx, dy, max_dist=200):
    return world.sample_direction(origin, dx, dy, max_dist=max_dist)

def sample_retina_binocular(position, heading, world, device="cpu"):
    """
    Retina:
        Left  eye: 20 x 20 = 400
        Right eye: 20 x 20 = 400
        Total RET = 800
    """
    AZ = np.linspace(-67.5, 67.5, 20)
    EL = np.linspace(-20.0, 20.0, 20)

    retL = torch.zeros(1, 400, device=device)
    retR = torch.zeros(1, 400, device=device)

    # ===========================================================
    # LEFT EYE
    # ===========================================================
    idx = 0
    for el in EL:
        elev = math.radians(el)
        for az in AZ:
            az_rad = math.radians(az - 45)     # left eye orientation
            angle = heading + az_rad

            dx = math.cos(angle)
            dy = math.sin(angle) + 0.3 * math.sin(elev)

            retL[0, idx] = cast_ray(world, position, dx, dy)
            idx += 1

    # ===========================================================
    # RIGHT EYE
    # ===========================================================
    idx = 0
    for el in EL:
        elev = math.radians(el)
        for az in AZ:
            az_rad = math.radians(az + 45)     # right eye orientation
            angle = heading + az_rad

            dx = math.cos(angle)
            dy = math.sin(angle) + 0.3 * math.sin(elev)

            retR[0, idx] = cast_ray(world, position, dx, dy)
            idx += 1

    # clamp to valid range
    retL = torch.clamp(retL, 0, 1)
    retR = torch.clamp(retR, 0, 1)

    return retL, retR
