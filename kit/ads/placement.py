"""Where advertising and pitch-side furniture go. Pure numpy.

- Perimeter board runs: 4 m behind the touchlines and 5.5 m behind the goal lines (inside the
  run-off), with a gap on the main-stand side for the dugouts and tunnel.
- 3D carpets: the ground footprint of a virtual upright board as seen from the main TV camera.
  Every vertex of the virtual board is projected from the camera onto the ground, so from that
  camera the flat carpet is indistinguishable from a standing board (and visibly flat from
  anywhere else, as on a real broadcast).
- TV weight: how well each placement faces and how close it is to the main camera (0-1), a
  starting point for pricing ad inventory.
"""
import math

import numpy as np

from ..stands.geometry import StraightRun

BOARD_SIDE_OFFSET = 4.0
BOARD_END_OFFSET = 5.5
BOARD_HEIGHT = 0.9


def main_camera(pitch, stands=None):
    """The main broadcast camera: stand gantry if there is a main stand, else a default tower."""
    if stands and stands.get("camera_mount"):
        return np.array(stands["camera_mount"]), np.array([0.0, 6.0, 0.0])
    return np.array([0.0, -(pitch["width"] / 2 + 32.0), 24.0]), np.array([0.0, 4.0, 0.0])


def board_runs(pitch, gap=0.0, side_offset=BOARD_SIDE_OFFSET, end_offset=BOARD_END_OFFSET):
    """Straight runs (anticlockwise, outward normal away from the pitch). gap: opening on the south
    (main stand) side centred on halfway, for the dugouts and tunnel."""
    L, W = pitch["length"], pitch["width"]
    yS, xE = W / 2 + side_offset, L / 2 + end_offset
    runs = []
    if gap > 0:
        runs.append(StraightRun("south_a", (-L / 2, -yS), (1, 0), L / 2 - gap / 2))
        runs.append(StraightRun("south_b", (gap / 2, -yS), (1, 0), L / 2 - gap / 2))
    else:
        runs.append(StraightRun("south", (-L / 2, -yS), (1, 0), L))
    runs.append(StraightRun("east", (xE, -W / 2 + 2.0), (0, 1), W - 4.0))
    runs.append(StraightRun("north", (L / 2, yS), (-1, 0), L))
    runs.append(StraightRun("west", (-xE, W / 2 - 2.0), (0, -1), W - 4.0))
    return runs


def tv_weight(centre, facing, cam):
    """0-1: facing the camera and near it scores high."""
    d = np.asarray(cam, float) - np.asarray(centre, float)
    dist = np.linalg.norm(d)
    f = np.asarray(facing, float)
    if f.shape[0] == 2:
        f = np.array([f[0], f[1], 0.0])
    f = f / np.linalg.norm(f)
    face = max(0.0, float(np.dot(f, d / dist)))
    return round(face * 1.0 / (1.0 + (dist / 120.0) ** 2), 3)


def carpet(cam, base_centre, width, height, nu=32, nv=10, ground=lambda x, y: 0.0):
    """Anamorphic carpet: returns ground verts (N,3), quads, uvs (per vertex, u 0..1 across, v 0..1 up).

    The virtual board stands on `base_centre` facing the camera (in plan); each of its grid points is
    projected along the camera ray onto the ground.
    """
    c = np.asarray(cam, float)
    b = np.asarray(base_centre, float)
    to_cam = c[:2] - b[:2]
    to_cam /= np.linalg.norm(to_cam)
    across = np.array([-to_cam[1], to_cam[0]])
    V, UV = [], []
    for j in range(nv + 1):
        for i in range(nu + 1):
            u, v = i / nu, j / nv
            p = np.array([*(b[:2] + across * (u - 0.5) * width), v * height])
            t = c[2] / (c[2] - p[2])                    # ray from the camera through p hits z = 0 at t
            g = c + (p - c) * t
            g[2] = ground(g[0], g[1]) + 0.008
            V.append(g)
            UV.append((u, v))                            # `across` is the camera's right-hand direction
    F = [(j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i)
         for j in range(nv) for i in range(nu)]
    return np.array(V), np.array(F), np.array(UV)


def carpet_spots(pitch, width=5.0, behind=0.8):
    """Base centres for carpets: both sides of each goal, just behind the goal line."""
    L = pitch["length"]
    out = []
    for s in (1, -1):
        for side in (1, -1):
            out.append((s, side, np.array([s * (L / 2 + behind), side * (3.66 + 1.6 + width / 2), 0.0])))
    return out


def place_carpet(cam, base, width, height, pitch, limit_x, margin=0.3):
    """Keep a carpet off the field of play: push it back until its whole footprint lies behind the
    goal line (the virtual board faces the camera, so its base runs diagonally and would otherwise
    cross the line), and inside the touchline extended; then shrink it to fit in front of the end
    boards. Returns (base, height)."""
    L, W = pitch["length"], pitch["width"]
    s = 1.0 if base[0] > 0 else -1.0
    base = np.array(base, float)
    h = height
    for _ in range(8):
        h = fit_carpet_height(cam, base, width, height, limit_x)
        V, _, _ = carpet(cam, base, width, h, 4, 2)
        inner = (s * V[:, 0]).min() - (L / 2 + margin)
        side = np.abs(V[:, 1]).max() - (W / 2 - 0.5)
        if inner >= -1e-6 and side <= 1e-6:
            break
        if inner < 0:
            base[0] += s * (-inner)
        if side > 0:
            base[1] -= np.sign(base[1]) * side
    return base, h


def fit_carpet_height(cam, base, width, height, limit_x):
    """Shrink the virtual board until its footprint stays in front of the end boards."""
    h = height
    for _ in range(30):
        V, _, _ = carpet(cam, base, width, h, 4, 2)
        if np.abs(V[:, 0]).max() <= limit_x:
            return h
        h *= 0.92
    return h
