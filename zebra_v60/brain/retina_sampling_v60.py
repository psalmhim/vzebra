import numpy as np
import torch
import math

def cast_ray(world, pos, angle, max_dist=200):
    dx = math.cos(angle)
    dy = math.sin(angle)
    return world.sample_direction(pos, dx, dy, max_dist=max_dist)

def retina_foveation_gain(alpha, sigma=25 * np.pi/180):
    return math.exp(-(alpha**2)/(2*sigma*sigma))

def retina_distance_gain(d):
    return 1.0 / math.log(1 + d + 1e-6)

def sample_retina_binocular_v60(position, heading, world, device="cpu"):
    az = np.linspace(-90, 90, 20)
    el = np.linspace(-20, 20, 20)

    retL = torch.zeros(1, 400, device=device)
    retR = torch.zeros(1, 400, device=device)

    idx = 0
    for elev in el:
        for azim in az:
            angle_L = heading + math.radians(azim - 90.0)
            inten = cast_ray(world, position, angle_L)
            fov = retina_foveation_gain(math.radians(azim))
            intensity = inten * fov
            retL[0, idx] = float(intensity)
            idx += 1

    idx = 0
    for elev in el:
        for azim in az:
            angle_R = heading + math.radians(azim + 90.0)
            inten = cast_ray(world, position, angle_R)
            fov = retina_foveation_gain(math.radians(azim))
            intensity = inten * fov
            retR[0, idx] = float(intensity)
            idx += 1

    return retL.clamp(0,1), retR.clamp(0,1)
