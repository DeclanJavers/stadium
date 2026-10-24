"""Phase 2 tests: sightlines, stand geometry, seats, patterns. Run inside Blender:

blender -b --factory-startup -P tests/test_stands.py
"""
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from kit import catalog  # noqa: E402
from kit.assembler import assemble  # noqa: E402
from kit.assets import seat as SEAT  # noqa: E402
from kit.core.scene import reset  # noqa: E402
from kit.stands import geometry as G  # noqa: E402
from kit.stands import patterns as PAT  # noqa: E402
from kit.stands import sightlines as S  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def profiles():
    return catalog.load("stand_profiles")


@test
def every_row_meets_its_sightline_and_rake_limits():
    for name, prof in profiles().items():
        for d in (8.0, 12.0):
            for t in S.solve_profile(prof, d):
                risers = np.diff(t["h"])
                assert (risers >= -1e-9).all(), f"{name}: a row is lower than the one in front"
                rake = math.degrees(math.atan(risers.max() / t["tread"])) if len(risers) else 0
                assert rake <= S.MAX_RAKE_DEG + 1e-6, f"{name} tier {t['index']}: rake {rake:.1f} deg"
                max_r = t["tread"] * math.tan(math.radians(S.MAX_RAKE_DEG))
                for r, c in zip(risers, t["c_achieved"]):
                    # every row clears the minimum; rows not held back by the rake limit reach the target
                    floor = min(S.C_MIN, t["c_target"])  # a profile may deliberately target less (grassroots)
                    assert c >= floor - 1e-6, f"{name} tier {t['index']} @d={d}: C {1000 * c:.0f} mm"
                    if t["index"] == 0 and r < max_r - 1e-6:
                        assert c >= t["c_target"] - 1e-6, f"{name} @d={d}: C {1000 * c:.0f} mm below target"


@test
def c_value_formula_matches_green_guide_worked_example():
    # Eye 1.15 m above tread, D = 20 m, N = 3 m, T = 0.8 m, C = 0.09 -> R = (3.09)(20.8)/20 - 3 = 0.2136
    t = S.solve_tier(2, 20.0 - 0.8 + S.SEATED_EYE_FWD - 10.0, 3.0 - S.SEATED_EYE, 10.0)
    assert abs((t["h"][1] - t["h"][0]) - 0.215) < 0.0051, t["h"]


@test
def upper_tiers_never_cut_into_the_tier_below():
    for name, prof in profiles().items():
        tiers = S.solve_profile(prof, 9.0)
        for lo, up in zip(tiers[:-1], tiers[1:]):
            # where the upper tier overhangs, its soffit must clear the lower tier's treads by the clearance
            for s, h in zip(lo["s"], lo["h"]):
                if s >= up["s0"] - S.WALL:
                    assert S.soffit_z(up, s) - h >= 2.4, f"{name}: only {S.soffit_z(up, s) - h:.2f} m headroom"


def _bowl(profile, corners="curved"):
    tiers = S.solve_profile(profiles()[profile], 9.0)
    runs = G.perimeter_runs(63.0, 43.0, {s: {} for s in ("south", "north", "east", "west")}, corners, 14.0)
    return tiers, [(n, r, G.build_run(r, tiers)) for n, r in runs]


@test
def seats_sit_on_treads_at_legal_spacing():
    tiers, runs = _bowl("three_tier_bowl")
    for name, run, out in runs:
        for a in out["anchors"]:
            t = tiers[a["tier"]]
            assert abs(a["pos"][2] - t["h"][a["row"]]) < 1e-6
            assert a["pitch"] >= t["seat_pitch"] - 1e-6, f"{name}: seat pitch {a['pitch']:.3f}"
            assert a["pitch"] <= 2 * t["seat_pitch"]
        rows = {}
        for a in out["anchors"]:
            rows.setdefault((a["tier"], a["row"], a["bay"]), []).append(a)
        for seg in rows.values():
            assert len(seg) <= 28 + 2, "more than 28 seats between aisles"


@test
def stand_prisms_are_closed_solids():
    _, runs = _bowl("two_tier_22_18", corners="open")
    for name, run, out in runs[:1]:
        V, F, tags, UV = out["mesh"]
        me = bpy.data.meshes.new("t")
        me.from_pydata(V.tolist(), [], [tuple(f) for f in F])
        bm = bmesh.new()
        bm.from_mesh(me)
        bad = [e for e in bm.edges if len(e.link_faces) != 2]
        assert not bad, f"{name}: {len(bad)} open/non-manifold edges"


@test
def capacities_are_in_the_right_class():
    for profile, lo, hi in (("three_tier_bowl", 50_000, 75_000), ("mega_bowl", 148_000, 160_000)):
        _, runs = _bowl(profile)
        n = sum(len(out["anchors"]) for _, _, out in runs)
        assert lo <= n <= hi, f"{profile}: {n} seats"


@test
def seats_fit_their_anchor_and_leave_a_seatway():
    for style, spec in catalog.load("seats").items():
        reset()
        for obj in SEAT.build_style(style):
            co = np.array([v.co[:] for v in obj.data.vertices])
            assert co[:, 1].min() > -0.03, f"{style}: pokes {-co[:, 1].min():.3f} m into the riser behind"
            seatway = spec["min_tread"] - co[:, 1].max()
            assert seatway >= 0.305 - 1e-3 or style == "rail_seat", f"{style}: seatway {seatway:.3f} m < 305 mm"
            assert co[:, 2].min() > -0.01, f"{style}: below the tread"
            assert abs(co[:, 0]).max() <= spec["width"] / 2 + 0.01, f"{style}: wider than its pitch"
            assert 0.4 < co[:, 2].max() < 1.3, f"{style}: implausible height {co[:, 2].max():.2f}"


@test
def text_mosaic_reads_from_the_pitch():
    # a straight run along +t: seen from the pitch (looking outward), reading order is decreasing t
    anchors = [{"run": "s", "tier": 0, "row": r, "t": 0.25 + 0.5 * c, "pitch": 0.5, "bay": 0}
               for r in range(14) for c in range(200)]
    cols = np.zeros((len(anchors), 3))
    n = PAT.apply_mosaic(cols, anchors, {"run": "s", "text": "LI", "color": "#ffffff"}, 100.0, [14])
    assert n > 0
    lit = [a for a, c in zip(anchors, cols) if c.sum() > 0]
    ts = np.array([a["t"] for a in lit])
    # 'L' has a full-height stem on its left; 'I' is narrower. The L's stem must be at the highest t.
    stem_t = ts.max()
    stem_rows = {a["row"] for a in lit if abs(a["t"] - stem_t) < 1e-9}
    assert len(stem_rows) >= 7, "first letter's stem should be at the reader's left (highest t)"


@test
def bowl_assembles_with_capacity_and_sockets():
    reset()
    st = assemble({"pitch": {}, "goals": None, "corner_flags": None,
                   "stands": {"corners": "curved", "sides": {s: {"profile": "single_tier_20"} for s in
                                                             ("north", "south", "east", "west")}}})
    cap = st["stands"]["capacity"]
    assert 12_000 < cap["seated"] < 25_000, cap
    kinds = {}
    for o in bpy.data.objects:
        if o.name.startswith("SOCKET:"):
            kinds[o["socket_kind"]] = kinds.get(o["socket_kind"], 0) + 1
    assert kinds.get("fascia", 0) > 20 and kinds.get("roof_anchor", 0) > 20, kinds


@test
def curved_corners_with_mixed_profiles_fall_back_to_open():
    reset()
    state = assemble({"pitch": {}, "goals": None, "corner_flags": None,
                      "stands": {"corners": "curved", "sides": {"north": {"profile": "single_tier_20"},
                                                                "south": {"profile": "two_tier_22_18"},
                                                                "east": {"profile": "single_tier_20"},
                                                                "west": {"profile": "single_tier_20"}}}})
    assert any("open corners" in w for w in state["warnings"]), state["warnings"]
    assert state["stands"]["corner_radius"] is None, "a mixed bowl is built with open corners"


@test
def stadium_glb_round_trips_every_seat():
    sys.path.insert(0, str(ROOT / "tools"))
    import export_stadium
    import json as _json
    layout = ROOT / "layouts" / "sunday_league.json"
    out = export_stadium.main(layout)
    expected = sorted(tuple(np.round(a, 3)) for a in _expected_positions(layout))
    reset()
    bpy.ops.import_scene.gltf(filepath=str(out))
    got = sorted(tuple(np.round(o.matrix_world.translation[:], 3)) for o in bpy.data.objects
                 if o.type == "MESH" and o.data.name.startswith("SEAT_") and not o.name.startswith("SEATS:")
                 and not o.name.startswith("DugoutSeat"))   # team benches are not spectator capacity
    assert len(got) == len(expected), f"{len(got)} seats imported, {len(expected)} generated"
    err = np.abs(np.array(got) - np.array(expected)).max()
    assert err < 0.002, f"seat positions drift by {err:.4f} m through glTF"


def _expected_positions(layout):
    import json as _json
    reset()
    st = assemble(_json.loads(layout.read_text()))
    return [a["pos"] for a in st["stands"]["anchors"]]


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
