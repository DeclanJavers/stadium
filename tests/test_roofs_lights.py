"""Phase 3 tests: roofs, facades, floodlights. Run inside Blender:

blender -b --factory-startup -P tests/test_roofs_lights.py
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

from kit import catalog  # noqa: E402
from kit.assembler import assemble  # noqa: E402
from kit.core.scene import reset  # noqa: E402
from kit.stands import facades as FC  # noqa: E402
from kit.stands import floodlights as FL  # noqa: E402
from kit.stands import geometry as G  # noqa: E402
from kit.stands import roofs as RF  # noqa: E402
from kit.stands import sightlines as S  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def profiles():
    return catalog.load("stand_profiles")


@test
def roofs_never_block_the_back_row_and_clear_every_tread():
    for pname, prof in profiles().items():
        tiers = S.solve_profile(prof, 9.0)
        for rname, rspec in catalog.load("roofs").items():
            r = RF.solve(tiers, rspec, 9.0, 68.0)
            assert r["clearance_to_sightline"] >= 0.29, f"{pname}/{rname}: roof edge in the view ({r['clearance_to_sightline']})"
            # underside above every tread it covers, with standing headroom
            for t in tiers:
                for s, h in zip(t["s"], t["h"]):
                    if r["s_front"] <= s <= r["s_back"]:
                        if r["type"] == "cantilever":
                            z = r["z_back"] + (r["z_front"] - r["z_back"]) * (r["s_back"] - s) / r["length"]
                        else:
                            z = r["z_back"] + (r["z_front"] - r["z_back"]) * (r["s_back"] - s) / r["length"]
                        assert z - h >= 2.2, f"{pname}/{rname}: {z - h:.2f} m headroom over a tread"


@test
def cantilevers_are_capped_and_rise_to_the_pitch():
    tiers = S.solve_profile(profiles()["mega_bowl"], 9.0)
    r = RF.solve(tiers, {"type": "cantilever"}, 9.0, 68.0)
    assert r["length"] <= 60.0 + 1e-6, r["length"]
    assert r["z_front"] > r["z_back"]


@test
def facade_encloses_stand_and_roof_tie():
    tiers = S.solve_profile(profiles()["three_tier_bowl"], 9.0)
    r = RF.solve(tiers, {"type": "cantilever"}, 9.0, 68.0)
    run = G.StraightRun("south", (-60, -45), (1, 0), 120)
    bays = G.bays(run, 30)
    rb = RF.build(run, r, bays)
    for kind in ("concrete_frame", "brick", "metal_bands", "glass", "etfe", "fins"):
        out = FC.build(run, tiers, bays, {"type": kind}, rb)
        assert out["line_s"] >= G.stand_depth(tiers) and out["line_s"] >= rb["tail_s"], kind
        V = out["mesh"][0]
        assert V[:, 2].min() >= -1e-6 and V[:, 2].max() <= out["height"] + 2.0, kind


def _design(layout_name):
    reset()
    st = assemble(json.loads((ROOT / "layouts" / f"{layout_name}.json").read_text()))
    return st["floodlights"]


@test
def elite_floodlights_meet_uefa_uniformity_and_target():
    for name, lux in (("premier", 2000), ("classic", 1000)):
        fl = _design(name)
        assert abs(fl["lux_avg"] - lux) < 1.0, (name, fl["lux_avg"])
        assert fl["u1"] >= 0.7 and fl["u2"] >= 0.8, f"{name}: U1 {fl['u1']:.2f} U2 {fl['u2']:.2f}"


@test
def grassroots_floodlights_meet_en12193_class_iii():
    fl = _design("sunday_league")
    assert fl["u2"] >= 0.5, f"U2 {fl['u2']:.2f}"


@test
def corner_masts_are_high_enough():
    for m in FL.pylons(105, 68, 65, 46, 10, 30.0, {}):
        elev = math.degrees(math.atan2(m["height"], math.hypot(*m["base"])))
        assert elev >= FL.MIN_ELEVATION, elev


@test
def analytic_illuminance_matches_inverse_square():
    # one 1 W (Blender) spot 10 m straight above a point: E = (1/4pi)/100 W/m2
    E = FL.illuminance([(0, 0, 10)], [(0, 0, 0)], 1.0, 60, 0.0, np.array([[0, 0, 0.0]]))
    assert abs(E[0] - 683 / (4 * math.pi * 100)) < 1e-6, E


@test
def roof_and_facade_sockets_exist():
    reset()
    assemble(json.loads((ROOT / "layouts" / "premier.json").read_text()))
    kinds = {}
    for o in bpy.data.objects:
        if o.name.startswith("SOCKET:"):
            kinds[o["socket_kind"]] = kinds.get(o["socket_kind"], 0) + 1
    assert kinds.get("roof_edge", 0) > 20 and kinds.get("facade_sign", 0) > 20, kinds


@test
def stadium_with_roofs_and_lights_exports():
    sys.path.insert(0, str(ROOT / "tools"))
    import export_stadium
    from kit.core.gltf_instancing import count_instances
    out = export_stadium.main(ROOT / "layouts" / "classic.json")
    assert out.stat().st_size > 1e6
    assert sum(count_instances(out).values()) > 10_000


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
