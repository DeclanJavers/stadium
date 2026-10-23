"""Phase 4 tests: ads, 3D carpets, screens, dugouts, tunnel. Run inside Blender:

blender -b --factory-startup -P tests/test_signage.py
"""
import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from kit.ads import placement as PL  # noqa: E402
from kit.assembler import assemble  # noqa: E402
from kit.assets import ads as ADS  # noqa: E402
from kit.assets import dugout as DUG  # noqa: E402
from kit.assets import pitch as PITCH  # noqa: E402
from kit.core.scene import reset, safe_name  # noqa: E402

TESTS = []
PITCH_P = PITCH.resolve({})


def test(fn):
    TESTS.append(fn)
    return fn


def layout(name):
    return json.loads((ROOT / "layouts" / f"{name}.json").read_text())


@test
def carpet_is_an_exact_projection_of_an_upright_board():
    cam = np.array([0.0, -60.0, 20.0])
    base = np.array([54.0, 8.0, 0.0])
    w, h = 5.0, 1.0
    V, F, UV = PL.carpet(cam, base, w, h, nu=8, nv=4)
    to_cam = (cam[:2] - base[:2]) / np.linalg.norm(cam[:2] - base[:2])
    across = np.array([-to_cam[1], to_cam[0]])
    for g, (u, v) in zip(V, UV):
        board = np.array([*(base[:2] + across * (u - 0.5) * w), v * h])
        g = g.copy()
        g[2] -= 0.008
        # camera, board point and carpet point must be collinear
        cross = np.cross(board - cam, g - cam)
        assert np.linalg.norm(cross) / np.linalg.norm(g - cam) < 1e-6, (u, v)


@test
def carpets_stay_clear_of_the_end_boards():
    cam, _ = PL.main_camera(PITCH_P)
    limit = PITCH_P["length"] / 2 + PL.BOARD_END_OFFSET - 0.4
    for s, side, base in PL.carpet_spots(PITCH_P):
        h = PL.fit_carpet_height(cam, base, 5.0, 1.1, limit)
        V, _, _ = PL.carpet(cam, base, 5.0, h)
        assert np.abs(V[:, 0]).max() <= limit + 1e-6
        assert h > 0.5, f"carpet squashed to {h:.2f} m"


@test
def board_text_reads_left_to_right_from_the_pitch():
    run = PL.board_runs(PITCH_P)[1]          # east end, runs +Y; seen from the pitch +Y is on the left
    V, F, UV = ADS._strip(run, run.t0, run.t1, 0.0, 0.1, 0.15, 0.95, 0.0, 0.01)
    # the leftmost point (largest y) must carry the smallest u
    left = UV[np.argmax(V[:, 1])][0]
    right = UV[np.argmin(V[:, 1])][0]
    assert left < right, (left, right)


@test
def boards_are_off_the_pitch_and_leave_room_for_dugouts():
    gap = DUG.board_gap({"style": "shell", "seats": 16})
    runs = PL.board_runs(PITCH_P, gap)
    L, W = PITCH_P["length"], PITCH_P["width"]
    for r in runs:
        for t in (r.t0, r.t1):
            p, _ = r.frame(t)
            assert abs(p[1]) >= W / 2 + 3.9 or abs(p[0]) >= L / 2 + 3.9, f"{r.name} board on the pitch"
    south = [r for r in runs if r.name.startswith("south")]
    inner = sorted(abs(r.frame(r.t1 if r.name.endswith("a") else r.t0)[0][0]) for r in south)
    dl = DUG.length({"style": "shell", "seats": 16})
    assert inner[0] >= DUG.GAP / 2 + dl, "boards overlap the dugouts"


def _built(name):
    reset()
    return assemble(layout(name))


@test
def premier_inventory_and_tv_weights_are_sane():
    st = _built("premier")
    inv = st["ads"]["inventory"]
    kinds = {s["kind"] for s in inv}
    assert {"perimeter_led", "carpet_3d", "led_ribbon"} <= kinds, kinds
    assert all(0.0 <= s["tv_weight"] <= 1.0 for s in inv)
    far = [s["tv_weight"] for s in inv if s["kind"] == "perimeter_led" and s["id"].startswith("north")]
    near = [s["tv_weight"] for s in inv if s["kind"] == "perimeter_led" and s["id"].startswith("south")]
    assert np.mean(far) > np.mean(near), "boards facing the main camera must be worth more"
    socks = [o for o in bpy.data.objects if o.get("socket_kind") == "ad_slot"]
    assert len(socks) == len(inv)


@test
def dugouts_and_tunnel_fit_between_pitch_and_stand():
    st = _built("premier")
    W = st["pitch_params"]["width"]
    B = st["stands"]["B"]
    for o in bpy.data.objects:
        if o.name.startswith("Dugout_") and o.type == "MESH":
            ys = [(o.matrix_world @ v.co).y for v in o.data.vertices]
            assert max(ys) <= -(W / 2 + 4.0), f"{o.name} encroaches on the run-off"
            assert min(ys) >= -B - 0.01, f"{o.name} cuts into the stand"
    t = st["tunnel"]
    assert abs(t["from_y"] + B) < 0.01 and t["to_y"] > t["from_y"]


@test
def screens_hang_clear_of_play_and_structure():
    st = _built("premier")
    for o in bpy.data.objects:
        if o.name.startswith("Screen_") and o.type == "MESH":
            zs = [(o.matrix_world @ v.co).z for v in o.data.vertices]
            assert min(zs) > 12.0, f"{o.name} bottom at {min(zs):.1f} m"
    st = _built("mega")
    for o in bpy.data.objects:
        if o.name.startswith("Screen_centre") and o.type == "MESH":
            zs = [(o.matrix_world @ v.co).z for v in o.data.vertices]
            assert 18.0 < min(zs) < 40.0, f"centre-hung bottom at {min(zs):.1f} m"


@test
def screen_face_reads_correctly():
    from kit.assets import screen as SCR
    reset()
    img = bpy.data.images.new("t", 4, 4)
    mat = ADS.led_material("t", img, 1.0, (4, 4))
    o = SCR.panel("P", 10.0, 5.0, None, mat)
    me = o.data
    me.calc_loop_triangles()
    face = [p for p in me.polygons if me.materials[p.material_index].name == "t"][0]
    assert face.normal.y > 0.99, "screen must face local +Y"
    uv = me.uv_layers["UVMap"].data
    for li in face.loop_indices:
        x = me.vertices[me.loops[li].vertex_index].co.x
        u = uv[li].uv[0]
        assert (u > 0.5) == (x < 0), "viewer's right (local -X) must show the right of the image"


@test
def every_layout_assembles_with_phase_4():
    for name in ("sunday_league", "classic", "premier", "mega"):
        reset()
        st = assemble(layout(name))
        assert st.get("dugouts") or name == "none"


@test
def object_names_never_carry_long_numeric_suffixes():
    assert safe_name("end-3.0200000000000005") == "end-3_0200000000000005"
    assert safe_name("Seat.001") == "Seat.001"


@test
def stadium_with_ads_exports():
    sys.path.insert(0, str(ROOT / "tools"))
    import export_stadium
    out = export_stadium.main(ROOT / "layouts" / "premier.json")
    assert out.stat().st_size > 5e6


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed", flush=True)
    sys.exit(1 if failed else 0)
