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
    ENTITY_OBSTACLE:  0.75,
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


def compute_binocular_depth(typeL, typeR, intL, intR,
                             eye_offset=np.radians(45)):
    """Estimate depth to entities using binocular disparity.

    The two eyes are separated by 2*eye_offset (90°). The frontal overlap
    zone spans approximately ±35° from heading. Within this zone, the same
    entity may appear in both eyes at different azimuthal positions.

    For each entity type, we compare pixel counts and intensity-weighted
    positions between eyes to estimate depth.

    Uses monocular intensity cue: brighter = closer (inverse-square falloff).

    Args:
        typeL, typeR: numpy [400] — entity type channels
        intL, intR: numpy [400] — intensity channels
        eye_offset: float — half interocular angle (default 45°)

    Returns:
        dict with per-entity depth estimates and stereo metrics
    """
    n_az = 20
    half_fov = np.radians(80)
    az = np.linspace(-half_fov, half_fov, n_az)

    # Overlap zone: left eye indices where ray also falls in right eye FOV
    # Left eye ray at azim a_L points at heading - eye_offset + a_L
    # Right eye ray at azim a_R points at heading + eye_offset + a_R
    # For overlap: heading - eye_offset + a_L = heading + eye_offset + a_R
    #            → a_L - a_R = 2 * eye_offset
    # Left eye overlap: a_L > 2*eye_offset - half_fov = 90° - 80° = 10°
    # → indices where azim > 10° ≈ indices 11..19

    results = {}
    type_vals = {
        "food": 1.0, "enemy": 0.5, "obstacle": 0.75,
        "colleague": 0.25,
    }

    for name, tval in type_vals.items():
        tol = 0.1
        mask_L = np.abs(typeL - tval) < tol
        mask_R = np.abs(typeR - tval) < tol
        px_L = float(np.sum(mask_L))
        px_R = float(np.sum(mask_R))

        if px_L < 1 and px_R < 1:
            results[f"{name}_depth"] = None
            continue

        # Monocular depth from intensity: mean intensity of matching pixels
        # Higher intensity = closer (rays attenuate with distance)
        int_L_mean = float(np.mean(intL[mask_L])) if px_L > 0 else 0.0
        int_R_mean = float(np.mean(intR[mask_R])) if px_R > 0 else 0.0
        mean_int = (int_L_mean * px_L + int_R_mean * px_R) / (px_L + px_R + 1e-8)

        # Binocular cue: pixel count ratio between eyes indicates angular
        # position. Entity in frontal overlap → both eyes see it.
        # Entity to one side → only one eye sees it.
        stereo_overlap = min(px_L, px_R) / (max(px_L, px_R) + 1e-8)

        # Depth estimate: combine intensity (closer=brighter) with
        # stereo overlap (frontal=closer for zebrafish prey capture)
        # Map intensity [0,1] → depth [200, 10]
        mono_depth = max(10.0, 200.0 * (1.0 - mean_int))
        # Stereo correction: high overlap → reduce depth estimate
        # (entity in frontal zone is typically closer)
        stereo_correction = 1.0 - 0.3 * stereo_overlap
        estimated_depth = mono_depth * stereo_correction

        results[f"{name}_depth"] = estimated_depth
        results[f"{name}_px"] = px_L + px_R
        results[f"{name}_stereo"] = stereo_overlap

    return results
