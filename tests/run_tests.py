"""Spec and pipeline tests. Run inside Blender:

blender -b --factory-startup -P tests/run_tests.py
"""
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from kit import catalog  # noqa: E402
from kit.assets import corner_flag, goal, pitch  # noqa: E402
from kit.core import mesh as M  # noqa: E402
from kit.core.export import export_glb, stats  # noqa: E402
from kit.core.scene import descendants, reset  # noqa: E402
from kit.core.sockets import sockets_of  # noqa: E402

MM = 0.001
RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def near(a, b, tol=2 * MM, what=""):
    assert abs(a - b) <= tol, f"{what}: {a:.4f} != {b:.4f} (tol {tol})"


def verts_world(obj):
    co = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def child(root, prefix):
    return [o for o in descendants(root) if o.name.startswith(prefix)]


# -- pitch ------------------------------------------------------------------------------------

@test
def pitch_markings_follow_law_1():
    p = pitch.resolve({"optional_marks": False})
    L, W, w = p["length"], p["width"], p["line_width"]
    assert w <= 0.12
    lines, discs = pitch.marking_paths(p)
    bound = lines[0][0]
    near(bound[:, 0].max() + w / 2, L / 2, what="touchline length (outer edge)")
    near(bound[:, 1].max() + w / 2, W / 2, what="goal line width (outer edge)")
    circle = lines[2][0]
    near(np.linalg.norm(circle, axis=1).max() + w / 2, 9.15, what="centre circle radius")
    pa = lines[3][0]  # first penalty area (east)
    near(L / 2 - (pa[1, 0] - w / 2), 16.5, what="penalty area depth")
    near(pa[0, 1] + w / 2, 7.32 / 2 + 16.5, what="penalty area half-width")
    ga = lines[4][0]
    near(L / 2 - (ga[1, 0] - w / 2), 5.5, what="goal area depth")
    spots = sorted(c[0] for c, _ in discs)
    near(L / 2 - spots[-1], 11.0, what="penalty mark distance")


@test
def pitch_lines_sit_on_crowned_surface():
    reset()
    root, p, lines = pitch.build({"crown": 0.1})
    v = verts_world(lines)
    zf = pitch.crown_fn(p)
    gap = v[:, 2] - zf(v[:, 0], v[:, 1])
    assert gap.min() > 0.0005, f"paint dips below the turf by {-gap.min():.4f} m"
    assert gap.max() < 0.02, f"paint floats {gap.max():.4f} m above the turf"


@test
def pitch_sockets():
    reset()
    root, p, _ = pitch.build({})
    bpy.context.view_layer.update()
    kinds = {}
    for s in sockets_of(root):
        kinds.setdefault(s["socket_kind"], []).append(s)
    assert len(kinds.get("goal", [])) == 2
    assert len(kinds.get("corner_flag", [])) == 4
    assert len(kinds.get("stand_front", [])) == 4
    for s in kinds["goal"]:
        near(abs(s.location.x), p["length"] / 2 - p["line_width"] / 2, what="goal socket on goal line")
        # local +Y must point at the centre mark
        fwd = s.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 1, 0))
        assert fwd.x * -math.copysign(1, s.location.x) > 0.99, "goal socket faces away from pitch"
    for s in kinds["corner_flag"]:
        fwd = s.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 1, 0))
        to_c = -s.location.copy()
        to_c.z = 0
        assert fwd.dot(to_c.normalized()) > 0.99, f"{s.name} does not face the centre mark"


@test
def every_grass_type_and_pattern_builds():
    for g in catalog.load("grass"):
        for pat in ("stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles", "none"):
            reset()
            pitch.build({"grass": g, "mow": {"pattern": pat}})


# -- goals ------------------------------------------------------------------------------------

def _frame_dims(root):
    v = verts_world(child(root, "Frame")[0])
    posts = v[v[:, 2] < 2.2]                      # post rings below the corner fillets
    inner_half = posts[posts[:, 0] > 0][:, 0].min()
    bar = v[(np.abs(v[:, 0]) < 3.5) & (v[:, 2] > 1.0)]  # crossbar span, clear of the corner fillets
    return 2 * inner_half, bar[:, 2].min(), posts[:, 1].max() - posts[:, 1].min(), \
        posts[posts[:, 0] > 0][:, 0].max() - inner_half


@test
def goal_frames_are_law_1_size():
    for style in catalog.load("goals"):
        reset()
        root = goal.build({"style": style})
        width, height, depth, post_w = _frame_dims(root)
        near(width, 7.32, what=f"{style} inner width")
        near(height, 2.44, what=f"{style} inner height")
        assert depth <= 0.12 + MM and post_w <= 0.12 + MM, f"{style} post {post_w:.3f} x {depth:.3f} > 12 cm"


@test
def goal_nets_hang_behind_the_line_and_above_ground():
    for style in catalog.load("goals"):
        reset()
        root = goal.build({"style": style})
        v = verts_world(child(root, "Net")[0])
        assert v[:, 1].max() < 0.0, f"{style}: net pokes in front of the goal line"
        assert v[:, 2].min() > -0.01, f"{style}: net sinks into the ground ({v[:, 2].min():.3f})"
        bulge = np.abs(v[:, 0]).max() - (3.66 + 0.12)
        print(f"    {style}: side bulge beyond posts {bulge * 100:.1f} cm")
        assert bulge < 0.25, f"{style}: net balloons {bulge:.2f} m sideways"
        p = goal.resolve({"style": style})
        assert v[:, 1].min() > -(p["net_bottom_depth"] + 0.4), f"{style}: net trails too far back"


@test
def goal_tint_slots():
    reset()
    st = stats(goal.build({"style": "classic", "net_pattern": "bands"}))
    for slot in ("TINT_frame", "TINT_net_a", "TINT_net_b"):
        assert slot in st["tint_slots"], f"missing {slot}"
    net = None
    for o in bpy.data.objects:
        if o.name.startswith("Net") and o.type == "MESH" and len(o.data.materials) == 2:
            net = o
    idx = np.zeros(len(net.data.polygons), np.int32)
    net.data.polygons.foreach_get("material_index", idx)
    assert set(np.unique(idx)) == {0, 1}, "banded net should use both colours"


# -- corner flags -----------------------------------------------------------------------------

@test
def corner_flags_follow_law_1():
    for preset in catalog.load("corner_flags"):
        reset()
        root = corner_flag.build({"preset": preset})
        tops = max(verts_world(o)[:, 2].max() for o in descendants(root) if o.type == "MESH")
        assert tops >= 1.5, f"{preset}: flagpost {tops:.3f} m < 1.5 m"
        assert child(root, "PoleCap"), f"{preset}: top must be rounded (non-pointed)"
        f = verts_world(child(root, "Flag")[0])
        assert f[:, 2].min() > 0.5, f"{preset}: flag droops to the ground"
        spread = f[:, 0].max() - f[:, 0].min()
        assert spread < 0.44, f"{preset}: flag did not drape (still rigid, {spread:.2f} m wide)"


# -- pipeline ---------------------------------------------------------------------------------

@test
def glb_round_trip_keeps_sockets_and_extras():
    reset()
    root, _, _ = pitch.build({})
    with tempfile.TemporaryDirectory() as d:
        path = export_glb(root, Path(d) / "pitch.glb")
        reset()
        bpy.ops.import_scene.gltf(filepath=str(path))
        names = [o.name for o in bpy.data.objects]
        assert sum(n.startswith("SOCKET:goal:") for n in names) == 2, names
        r = [o for o in bpy.data.objects if o.get("asset_family") == "pitch"]
        assert r and json.loads(r[0]["params"])["length"] == 105.0, "asset extras lost in export"


@test
def triangle_budgets():
    budgets = {"pitch": 60_000, "goal": 250_000, "corner_flag": 20_000}
    builders = [("pitch", lambda: pitch.build({})[0])] + \
        [("goal", lambda s=s: goal.build({"style": s})) for s in catalog.load("goals")] + \
        [("corner_flag", lambda s=s: corner_flag.build({"preset": s})) for s in catalog.load("corner_flags")]
    for fam, fn in builders:
        reset()
        st = stats(fn())
        assert st["triangles"] <= budgets[fam], f"{st['asset_id']}: {st['triangles']} > {budgets[fam]}"


@test
def sweep_caps_are_valid_quads():
    V, F, _ = M.sweep([(0, 0, 0), (0, 0, 1)], M.profile_ellipse(0.1, 0.12, 24))
    assert all(len(set(f)) >= 3 for f in F.tolist())
    for f in F:
        assert len(set(f.tolist())) == 4 or len(set(f.tolist())) == 3


if __name__ == "__main__":
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} passed", flush=True)
    sys.exit(1 if failed else 0)
