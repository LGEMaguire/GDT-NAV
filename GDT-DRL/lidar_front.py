"""Front-180° LiDAR sampling matching pygame_env_rosv5.py.

Sim uses 20 rays over π radians:
    angle_i = -π/2 + i * (π / 20)   # i = 0..19  →  -90° … +81°
"""

import math
import numpy as np

N_BEAMS = 20
HALF_FOV = math.pi / 2.0
SEGMENT_ANGLE = math.pi / N_BEAMS
BEAM_ANGLES = np.array(
    [-math.pi / 2.0 + i * SEGMENT_ANGLE for i in range(N_BEAMS)],
    dtype=np.float64,
)


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def crop_scan_to_front(ranges, angle_min, angle_increment, intensities=None):
    """Keep beams with |angle| <= 90°, sorted from -90° to +90°."""
    ranges = np.asarray(ranges, dtype=np.float64)
    n = len(ranges)
    if n == 0 or angle_increment == 0:
        return None

    angles = wrap_pi(angle_min + np.arange(n) * angle_increment)
    mask = np.abs(angles) <= HALF_FOV + 1e-6
    if not np.any(mask):
        return None

    ang_f = angles[mask]
    rng_f = ranges[mask]
    order = np.argsort(ang_f)
    ang_f = ang_f[order]
    rng_f = rng_f[order]

    intens_f = None
    if intensities is not None and len(intensities) == n:
        intens_f = np.asarray(intensities)[mask][order]

    n_f = len(ang_f)
    increment = float((ang_f[-1] - ang_f[0]) / (n_f - 1)) if n_f > 1 else float(angle_increment)

    return {
        'ranges': rng_f,
        'angles': ang_f,
        'angle_min': float(ang_f[0]),
        'angle_max': float(ang_f[-1]),
        'angle_increment': increment,
        'intensities': intens_f,
    }


def sample_sim_beams(ranges, angle_min, angle_increment, max_range=10.0):
    """Nearest-neighbour sample of the front 180° scan at the 20 sim beam angles."""
    out = np.full(N_BEAMS, max_range, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.float64)
    n = len(ranges)
    if n == 0:
        return out

    angles = wrap_pi(angle_min + np.arange(n) * angle_increment)
    front = np.abs(angles) <= HALF_FOV + 1e-6
    if not np.any(front):
        return out

    for i, target in enumerate(BEAM_ANGLES):
        delta = np.abs(wrap_pi(angles - target))
        delta = np.where(front, delta, np.inf)
        idx = int(np.argmin(delta))
        if not np.isfinite(delta[idx]):
            continue
        r = ranges[idx]
        if not np.isfinite(r) or r < 0.0:
            r = max_range
        out[i] = min(float(r), max_range)
    return out
