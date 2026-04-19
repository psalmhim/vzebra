"""
Shared geometric sensory bridge: projects env objects onto retinal arrays.
Used by demo, comparison tests, and training scripts.

Key improvements over initial version:
  - Wider FoV (200° vs 140°) reduces rear blind spot
  - Larger food effective radius (12px vs 8px) for better detection
  - Symmetry-breaking flee kick for centered predators
"""
import math
import numpy as np


def inject_sensory(env, world_config=None):
    """
    Compute retinal and sensory inputs from environment geometry and inject
    them as env attributes that ZebrafishBrainV2.step() expects.

    Args:
        env: environment with fish/predator/food/rock state
        world_config: optional WorldConfig for visual physics params

    Sets:
      env.brain_L / env.brain_R  — 800-element retinal arrays
      env._enemy_pixels_total    — estimated predator pixels in retina
    """
    arena_w = getattr(env, 'arena_w', 800)
    arena_h = getattr(env, 'arena_h', 600)
    fish_x = getattr(env, 'fish_x', arena_w / 2)
    fish_y = getattr(env, 'fish_y', arena_h / 2)
    fish_heading = getattr(env, 'fish_heading', 0.0)
    gaze_offset = getattr(env, 'gaze_offset', 0.0)
    # Gaze offset from saccade module: shifts visual sampling without body rotation
    effective_heading = fish_heading + gaze_offset  # eye direction ≠ body heading

    # Visual physics from config (or defaults)
    if world_config is not None:
        _fov_deg = world_config.fov_degrees
        _max_dist = world_config.max_vision_distance
        _intensity_scale = world_config.retinal_intensity_scale
        _r_pred = world_config.visual_radius_predator
        _r_food = world_config.visual_radius_food
        _r_rock = world_config.visual_radius_rock
        _tc_pred = world_config.type_code_predator
        _tc_food = world_config.type_code_food
        _tc_rock = world_config.type_code_rock
    else:
        _fov_deg = 200.0
        _max_dist = 500.0
        _intensity_scale = 15.0
        _r_pred = 20.0
        _r_food = 12.0
        _r_rock = 30.0
        _tc_pred = 0.5
        _tc_food = 0.8
        _tc_rock = 0.3

    fov_rad = math.radians(_fov_deg)

    brain_L = np.zeros(800, dtype=np.float32)
    brain_R = np.zeros(800, dtype=np.float32)

    def _project(ox, oy, radius, type_val):
        dx = ox - fish_x
        dy = oy - fish_y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        if dist > _max_dist:
            return
        angle_to = math.atan2(dy, dx)
        rel = math.atan2(math.sin(angle_to - effective_heading),
                         math.cos(angle_to - effective_heading))
        if abs(rel) > fov_rad / 2:
            return
        ang_hw = math.atan(radius / dist)
        col_c = int((rel + fov_rad / 2) / fov_rad * 400)
        col_hw = max(1, int(ang_hw / fov_rad * 400))
        intensity = min(1.0, (radius / dist) * _intensity_scale)
        for c in range(max(0, col_c - col_hw), min(400, col_c + col_hw)):
            if rel <= 0:
                brain_L[c] = max(brain_L[c], intensity)
                brain_L[400 + c] = type_val
            else:
                brain_R[c] = max(brain_R[c], intensity)
                brain_R[400 + c] = type_val

    # Project predator
    _project(getattr(env, 'pred_x', -9999),
             getattr(env, 'pred_y', -9999), _r_pred, _tc_pred)

    # Project food items
    for food in getattr(env, 'foods', []):
        _project(food[0], food[1], _r_food, _tc_food)

    # Project rocks/obstacles
    for rock in getattr(env, 'rock_formations', []):
        _project(rock['cx'], rock['cy'], rock.get('radius', _r_rock), _tc_rock)

    # Count enemy pixels
    enemy_px = (int(np.sum(np.abs(brain_L[400:] - _tc_pred) < 0.1)) +
                int(np.sum(np.abs(brain_R[400:] - _tc_pred) < 0.1)))

    env.brain_L = brain_L
    env.brain_R = brain_R
    env._enemy_pixels_total = enemy_px
