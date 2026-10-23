"""Render review shots of the kit.

blender -b -P tools/review.py -- <shot> [--layout layouts/x.json] [--samples N] [--res WxH] [--out DIR]

Shots: broadcast, goalmouth, grass_macro, goal_<style>, corner_flags, all
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

from kit.assembler import assemble  # noqa: E402
from kit.core import render as R  # noqa: E402
from kit.core.scene import OUT, reset  # noqa: E402


def load_layout(path):
    return json.loads(Path(path).read_text()) if path else {}


def scene(layout, sun=(38, 215)):
    reset()
    R.setup_cycles(SAMPLES, RES)
    if NIGHT:
        layout = dict(layout, lighting="night")
        R.setup_night()
    else:
        R.setup_sky(*sun)
    R.surroundings()
    st = assemble(layout)
    fl = st.get("floodlights")
    if NIGHT and fl and fl["lux_avg"] < 1500:
        # expose like a camera would: a 150 lux grassroots pitch is ~3.7 stops darker than 2000 lux
        bpy.context.scene.view_settings.exposure = math.log2(1500 / fl["lux_avg"])
    return st


def shot_broadcast(layout, out):
    st = scene(layout)
    L, W = st["pitch_params"]["length"], st["pitch_params"]["width"]
    mount = (st.get("stands") or {}).get("camera_mount")
    if mount:  # main camera gantry on the front of the south stand's tier nearest 16 m
        cam = R.camera("Broadcast", mount, (0, 6, 0), lens=21)
    else:
        cam = R.camera("Broadcast", (0, -(W / 2 + 32), 24), (0, 4, 0), lens=21)
    return [R.render(cam, out / f"{NAME}_broadcast.png")]


def shot_high_behind_goal(layout, out):
    st = scene(layout)
    L = st["pitch_params"]["length"]
    cam = R.camera("BehindGoal", (-(L / 2 + 30), -6, 26), (0, 0, 0), lens=24)
    return [R.render(cam, out / f"{NAME}_behind_goal.png")]


def shot_goalmouth(layout, out):
    st = scene(layout, sun=(32, 240))
    p = st["pitch_params"]
    L = p["length"]
    cam_pos = (L / 2 - 13.5, -5.5, 1.55)
    from kit.assets import pitch as P
    P.add_blades(p, st["pitch"], st["lines"], (L / 2 - 16, L / 2 + 1.0, -12, 4), cam_pos[:2] + (0.0,),
                 density=18000, near=2.0, far=14.0)
    cam = R.camera("Goalmouth", cam_pos, (L / 2 + 0.6, 0.6, 1.1), lens=32, dof=8.0)
    return [R.render(cam, out / f"{NAME}_goalmouth.png")]


def shot_grass_macro(layout, out):
    st = scene(layout, sun=(24, 250))
    p = st["pitch_params"]
    L = p["length"]
    x0 = L / 2 - 16.5 + 0.06
    cam_pos = (x0 - 2.6, -6.5, 0.32)
    from kit.assets import pitch as P
    P.add_blades(p, st["pitch"], st["lines"], (x0 - 3.5, x0 + 6, -9, 0), cam_pos[:2] + (0.0,),
                 density=60000, near=1.0, far=7.0)
    cam = R.camera("Macro", cam_pos, (x0 + 0.5, -4.0, 0.0), lens=35, dof=4.0)
    return [R.render(cam, out / f"{NAME}_grass_macro.png")]


def shot_goal_closeup(layout, out):
    st = scene(layout, sun=(30, 140))
    L = st["pitch_params"]["length"]
    p = st["pitch_params"]
    from kit.assets import pitch as P
    cam_pos = (L / 2 - 6.0, 7.0, 1.7)
    P.add_blades(p, st["pitch"], st["lines"], (L / 2 - 9, L / 2 + 4, -6, 9), cam_pos[:2] + (0.0,),
                 density=14000, near=2.0, far=12.0)
    cam = R.camera("GoalSide", cam_pos, (L / 2 + 0.7, -0.4, 1.0), lens=28, dof=11.0)
    return [R.render(cam, out / f"{NAME}_goal_side.png")]


def shot_corner_flag(layout, out):
    st = scene(layout, sun=(30, 140))
    p = st["pitch_params"]
    L, W = p["length"], p["width"]
    from kit.assets import pitch as P
    cx, cy = L / 2, W / 2
    cam_pos = (cx - 3.4, cy - 2.6, 1.05)
    P.add_blades(p, st["pitch"], st["lines"], (cx - 6, cx + 1.5, cy - 5, cy + 1.5), cam_pos[:2] + (0.0,),
                 density=30000, near=1.0, far=7.0)
    cam = R.camera("Corner", cam_pos, (cx - 0.05, cy - 0.05, 0.85), lens=40, dof=5.6)
    return [R.render(cam, out / f"{NAME}_corner_flag.png")]


def shot_aerial(layout, out):
    st = scene(layout, sun=(42, 225))
    p = st["pitch_params"]
    stands = st.get("stands")
    reach = p["length"] / 2 + 40
    if stands:
        out_s = max([f["line_s"] for f in stands.get("facades", {}).values()] + [30.0])
        reach = stands["A"] + out_s + 40
    cam = R.camera("Aerial", (-reach * 0.95, -reach * 1.25, reach * 0.95), (0, 4, 0), lens=30)
    return [R.render(cam, out / f"{NAME}_aerial.png")]


def shot_stand_rows(layout, out):
    """From an aisle in the lower tier of the north stand, looking along the rows to the pitch."""
    st = scene(layout, sun=(34, 160))
    stands = st["stands"]
    B = stands["B"]
    cam = R.camera("Rows", (6.0, B + 9.0, 5.2), (-12.0, B - 6.0, 0.5), lens=24, dof=None)
    return [R.render(cam, out / f"{NAME}_stand_rows.png")]


def shot_opposite_stand(layout, out):
    """From the centre circle up at the north stand: sightlines, tiers, mosaics."""
    st = scene(layout, sun=(38, 200))
    B = st["stands"]["B"]
    cam = R.camera("Opposite", (0, -10, 1.7), (0, B + 20, 12), lens=24)
    return [R.render(cam, out / f"{NAME}_opposite.png")]


def shot_seat_lineup(layout, out):
    """Every seat style on a 3-row stepped block, side by side."""
    reset()
    R.setup_cycles(SAMPLES, RES)
    R.setup_sky(30, 205)
    R.surroundings()
    from kit import catalog
    from kit.assets import seat as SEAT
    from kit.assets import stand as STAND
    from kit.core.scene import collection, link, mesh_object
    import numpy as np
    styles = list(catalog.load("seats"))
    conc = STAND.concrete()
    tread, riser, n = 0.85, 0.36, 5
    for k, style in enumerate(styles):
        spec = SEAT.resolve(style)
        col = collection(f"Lineup_{style}")
        variants = SEAT.build_style(style, col)
        w = spec["width"]
        x0 = k * (n * w + 1.4)
        for row in range(3):
            z = row * riser
            y = -row * tread
            mesh_object(f"step_{style}_{row}", *_box(x0 - 0.3, x0 + n * w + 0.3, y - tread, y, -0.5, z), col,
                        materials=[conc])
            for j in range(n):
                v = variants[SEAT.variant_index(spec, j, n)]
                o = link(bpy.data.objects.new(f"{style}_{row}_{j}", v.data), col)
                o.location = (x0 + (j + 0.5) * w, y - tread, z)
    span = len(styles) * (n * 0.5 + 1.4)
    paths = []
    cam = R.camera("Lineup", (span / 2 - 0.7, 17.5, 4.5), (span / 2 - 0.7, -1.2, 0.5), lens=24)
    paths.append(R.render(cam, out / "seat_lineup.png"))
    # three-quarter close-ups, two styles per frame
    xs = []
    x = 0.0
    for style in styles:
        w = SEAT.resolve(style)["width"]
        xs.append(x + n * w / 2)
        x += n * w + 1.4
    for k in range(0, len(styles), 2):
        mid = (xs[k] + xs[min(k + 1, len(styles) - 1)]) / 2
        cam = R.camera(f"Q{k}", (mid - 2.5, 4.2, 2.0), (mid, -1.0, 0.55), lens=30)
        paths.append(R.render(cam, out / f"seat_closeup_{styles[k]}.png"))
    return paths


def _box(x0, x1, y0, y1, z0, z1):
    import numpy as np
    V = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    F = np.array([(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)])
    return V, F


def shot_main_stand(layout, out):
    """From the pitch, looking at the south (main) stand."""
    st = scene(layout, sun=(30, 20))
    B = st["stands"]["B"]
    cam = R.camera("Main", (8.0, -B + 18.0, 1.7), (2.0, -B - 6.0, 2.5), lens=28)
    return [R.render(cam, out / f"{NAME}_main_stand.png")]


def shot_exterior(layout, out):
    """Outside the main (south) stand at street level, looking up at the facade."""
    st = scene(layout, sun=(34, 200))
    fc = st["stands"]["facades"].get("south") or next(iter(st["stands"]["facades"].values()))
    B = st["stands"]["B"]
    y = -(B + fc["line_s"])
    cam = R.camera("Exterior", (-38.0, y - 48.0, 1.7), (-6.0, y, fc["height"] * 0.45), lens=24)
    return [R.render(cam, out / f"{NAME}_exterior.png")]


def shot_carpets(layout, out):
    """The 3D-carpet illusion: zoomed from the broadcast camera (upright) and from pitch level (flat)."""
    st = scene(layout, sun=(40, 200))
    from kit.ads.placement import main_camera
    cam_pos, _ = main_camera(st["pitch_params"], st.get("stands"))
    L = st["pitch_params"]["length"]
    goal = (L / 2 + 1.5, 4.0, 0.6)
    a = R.camera("CarpetTV", tuple(cam_pos), goal, lens=150)
    b = R.camera("CarpetSide", (L / 2 - 6.0, 16.0, 1.6), (L / 2 + 1.0, 6.0, 0.0), lens=28)
    return [R.render(a, out / f"{NAME}_carpets_tv.png"), R.render(b, out / f"{NAME}_carpets_ground.png")]


def shot_dugouts(layout, out):
    st = scene(layout, sun=(36, 160))
    W = st["pitch_params"]["width"]
    cam = R.camera("Dugouts", (9.0, -W / 2 + 3.0, 1.7), (-2.0, -W / 2 - 5.5, 1.0), lens=24)
    return [R.render(cam, out / f"{NAME}_dugouts.png")]


def shot_screens(layout, out):
    """From near the centre circle, looking up at the screens."""
    st = scene(layout, sun=(40, 200))
    scr = [o for o in bpy.data.objects if o.name.startswith("Screen_") and o.type == "MESH"]
    if not scr:
        return []
    target = scr[0].matrix_world.translation
    if abs(target.x) < 25 and abs(target.y) < 25:      # centre-hung: look up at it from the penalty spot
        cam = R.camera("Screens", (-30.0, -14.0, 1.7), (0.0, 0.0, target.z), lens=30)
    else:
        cam = R.camera("Screens", (-target.x * 0.15, -target.y * 0.15 - 6.0, 1.7), tuple(target), lens=30)
    return [R.render(cam, out / f"{NAME}_screens.png")]


SHOTS = {
    "screens": shot_screens,
    "carpets": shot_carpets,
    "dugouts": shot_dugouts,
    "exterior": shot_exterior,
    "main_stand": shot_main_stand,
    "seat_lineup": shot_seat_lineup,
    "aerial": shot_aerial,
    "stand_rows": shot_stand_rows,
    "opposite": shot_opposite_stand,
    "broadcast": shot_broadcast,
    "behind_goal": shot_high_behind_goal,
    "goalmouth": shot_goalmouth,
    "goal_side": shot_goal_closeup,
    "grass_macro": shot_grass_macro,
    "corner_flag": shot_corner_flag,
}


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="+")
    ap.add_argument("--layout", default=None)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--res", default="1600x900")
    ap.add_argument("--out", default=str(OUT / "review"))
    ap.add_argument("--night", action="store_true")
    a = ap.parse_args(argv)
    SAMPLES = a.samples
    RES = tuple(int(v) for v in a.res.split("x"))
    NIGHT = a.night
    NAME = (Path(a.layout).stem if a.layout else "default") + ("_night" if a.night else "")
    layout = load_layout(a.layout)
    names = list(SHOTS) if a.shots == ["all"] else a.shots
    for s in names:
        t = time.time()
        paths = SHOTS[s](layout, Path(a.out))
        print(f"[review] {s}: {[str(x) for x in paths]} in {time.time() - t:.1f}s", flush=True)
