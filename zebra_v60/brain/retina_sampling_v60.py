# retina_sampling_v60.py
import torch
import numpy as np
import math

from zebra_v60.world.world_env import (
    ENTITY_NONE, ENTITY_FOOD, ENTITY_ENEMY, ENTITY_COLLEAGUE,
    ENTITY_BOUNDARY, ENTITY_OBSTACLE, ENTITY_PREY,
)

# Type-encoding channel: maximally distinct values per entity for classification
ENTITY_TYPE_ENCODING = {
    ENTITY_NONE:      0.0,
    ENTITY_FOOD:      1.0,
    ENTITY_ENEMY:     0.5,
    ENTITY_COLLEAGUE: 0.25,
    ENTITY_BOUNDARY:  0.12,
    ENTITY_OBSTACLE:  0.88,
    ENTITY_PREY:      0.38,
}


def cast_ray(world, pos, angle, max_dist=200):
    """Cast ray and return (intensity, entity_type)."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    return world.sample_direction(pos, dx, dy, max_dist=max_dist)


def cast_ray_with_depth(world, pos, angle, max_dist=200, depth_scale=80.0):
    """Cast ray with depth-dependent intensity falloff.

    Returns (intensity, entity_type) where intensity is attenuated by
    exp(-dist/depth_scale). Type channel is unaffected by distance.
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    intensity, entity_type, dist = world.sample_direction_with_depth(
        pos, dx, dy, max_dist=max_dist)
    intensity *= math.exp(-dist / depth_scale)
    return intensity, entity_type


# Sharpened foveation gain (12° sigma for better center-periphery contrast)
def retinal_foveation(alpha, sigma=np.radians(12)):
    return math.exp(-(alpha ** 2) / (2 * sigma * sigma))


def sample_retina_binocular_v60(position, heading, world, device="cpu",
                                 eye_offset=np.radians(45),
                                 depth_shading=False, depth_scale=80.0,
                                 max_dist=200):
    """
    v60 binocular retina sampling with 2-channel output:
      Channel 0 ([:400]): intensity (foveation-weighted brightness)
      Channel 1 ([400:]): type encoding (distinct value per entity type)

    Left eye: centered at heading - eye_offset
    Right eye: centered at heading + eye_offset
    Each eye: 20×20 grid = 400 pixels × 2 channels = 800 values

    Total output: L=[1,800], R=[1,800]

    When depth_shading=True, intensity is attenuated by exp(-dist/depth_scale).
    Type channel is unaffected (entity identity doesn't fade with distance).
    """
    half_fov = np.radians(80)
    az = np.linspace(-half_fov, half_fov, 20)
    el = np.linspace(-20, 20, 20) * np.pi / 180

    if depth_shading:
        def ray_fn(w, p, a):
            return cast_ray_with_depth(w, p, a, max_dist=max_dist,
                                       depth_scale=depth_scale)
    else:
        def ray_fn(w, p, a):
            return cast_ray(w, p, a, max_dist=max_dist)

    retL = torch.zeros(1, 800, device=device)
    retR = torch.zeros(1, 800, device=device)

    # NOTE: retL/retR are computational labels used throughout the SNN pipeline.
    # In world coords (y-up), heading-offset = CW = fish's right side,
    # heading+offset = CCW = fish's left side.  The SNN weights were trained
    # with retL=heading-offset.  The vision strip display swaps them so the
    # user sees anatomically correct L/R labels.
    heading_L = heading - eye_offset
    idx = 0
    for elev in el:
        for azim in az:
            ray_angle = heading_L + azim
            intensity, entity_type = ray_fn(world, position, ray_angle)
            fov_gain = retinal_foveation(azim)
            retL[0, idx] = float(intensity * fov_gain)
            # Type channel: NO foveation — spectral/type info available across full retina
            retL[0, idx + 400] = float(ENTITY_TYPE_ENCODING.get(entity_type, 0.0))
            idx += 1

    # Right eye (computational label)
    heading_R = heading + eye_offset
    idx = 0
    for elev in el:
        for azim in az:
            ray_angle = heading_R + azim
            intensity, entity_type = ray_fn(world, position, ray_angle)
            fov_gain = retinal_foveation(azim)
            retR[0, idx] = float(intensity * fov_gain)
            retR[0, idx + 400] = float(ENTITY_TYPE_ENCODING.get(entity_type, 0.0))
            idx += 1

    return retL.clamp(0, 1), retR.clamp(0, 1)
