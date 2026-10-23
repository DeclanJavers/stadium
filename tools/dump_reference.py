"""Dump reference outputs of the Python generators for the Rust port's parity tests.

blender -b --factory-startup -P tools/dump_reference.py

Writes viewer/gen/tests/reference/*.json. Re-run whenever a generator changes on purpose.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from kit.ads import placement as PL  # noqa: E402
from kit.assets import pitch as PITCH  # noqa: E402
from kit.stands import facades as FC  # noqa: E402
from kit.stands import floodlights as FL  # noqa: E402
from kit.stands import geometry as G  # noqa: E402
from kit.stands import patterns as PAT  # noqa: E402
from kit.stands import roofs as RF  # noqa: E402
from kit.stands import sightlines as S  # noqa: E402

OUTDIR = ROOT / "viewer" / "gen" / "tests" / "reference"


def cat(name):
    d = json.loads((ROOT / "catalog" / f"{name}.json").read_text())
    d.pop("_doc", None)
    return d


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def write(name, data):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{name}.json").write_text(json.dumps(jsonable(data), separators=(",", ":")))
    print(f"[ref] {name}.json")


def tier_dump(t):
    keys = ("index", "rows", "kind", "tread", "c_target", "s0", "h", "s", "c_achieved", "s_back", "setback",
            "parapet", "glass", "slab", "soffit_front", "soffit_slope", "seat_pitch", "vomitories", "hospitality")
    return {k: t.get(k) for k in keys}


def mesh_dump(arrays):
    V, F, tags, UV = arrays
    V = np.asarray(V)
    return {"n_verts": len(V), "n_faces": len(F), "face_sizes": [len(f) for f in F],
            "tags": [int(t) for t in tags],
            "bounds": [V.min(0).tolist(), V.max(0).tolist()] if len(V) else [[0, 0, 0], [0, 0, 0]],
            "vsum": V.sum(0).tolist() if len(V) else [0, 0, 0],
            "uvsum": np.asarray(UV).sum(0).tolist() if len(UV) else [0, 0],
            "first_faces": [list(map(int, f)) for f in F[:6]]}


def run_dump(run):
    return {"name": run.name, "curved": run.curved, "t0": run.t0, "t1": run.t1}


def anchors_dump(anchors):
    keys = ("pos", "yaw", "tier", "row", "t", "pitch", "bay", "seat", "of", "riser")
    pick = anchors[:4] + anchors[len(anchors) // 2:len(anchors) // 2 + 2] + anchors[-4:]
    return {"count": len(anchors), "sample": [{k: a[k] for k in keys} for a in pick],
            "pos_sum": np.sum([a["pos"] for a in anchors], 0).tolist() if anchors else [0, 0, 0]}


def stands_scenario(name, sides, corners, R, gap=3.0):
    p = PITCH.resolve({})
    A = p["length"] / 2 + p["runoff_end"] + gap
    B = p["width"] / 2 + p["runoff_side"] + gap
    profiles = cat("stand_profiles")
    out = {"A": A, "B": B, "runs": []}
    d_focal = {"south": p["runoff_side"] + gap, "north": p["runoff_side"] + gap,
               "east": p["runoff_end"] + gap, "west": p["runoff_end"] + gap}
    any_side = next(iter(sides.values()))
    for rname, run in G.perimeter_runs(A, B, sides, corners, R):
        sc = sides.get(rname, any_side)
        df = min(d_focal.values()) if corners == "curved" else d_focal[rname]
        tiers = S.solve_profile(profiles[sc["profile"]], df)
        for t in tiers:
            t["seat_pitch"] = max(t["seat_pitch"], 0.5)
        r = G.build_run(run, tiers)
        out["runs"].append({
            "run": run_dump(run), "d_focal": df, "bay_list": r["bay_list"], "mesh": mesh_dump(r["mesh"]),
            "anchors": anchors_dump(r["anchors"]),
            "rails": {"count": len(r["rails"]), "points": sum(len(x[0]) for x in r["rails"]),
                      "psum": np.sum([x[0].sum(0) for x in r["rails"]], 0).tolist() if r["rails"] else [0, 0, 0]},
            "nosings": {"count": len(r["nosings"]),
                        "psum": np.sum([a + b for a, b in r["nosings"]], 0).tolist() if r["nosings"] else [0, 0, 0]},
            "barriers": {"count": len(r["barriers"]), "points": sum(len(b) for b in r["barriers"])},
            "sockets": [(k, i, list(loc), list(n), props) for k, i, loc, n, props in r["sockets"]],
            "stand_depth": G.stand_depth(tiers), "hull": G.section_hull(tiers),
        })
    return out


def main():
    profiles = cat("stand_profiles")
    write("sightlines", {name: {str(d): [tier_dump(t) for t in S.solve_profile(prof, d)] for d in (9.0, 13.5)}
                         for name, prof in profiles.items()})

    sides_bowl = {s: {"profile": "three_tier_bowl"} for s in ("south", "north", "east", "west")}
    sides_open = {"south": {"profile": "two_tier_22_18"}, "north": {"profile": "single_tier_20"},
                  "east": {"profile": "kop_terrace_40"}, "west": {"profile": "single_tier_20", "length": 60}}
    write("stands", {"bowl": stands_scenario("bowl", sides_bowl, "curved", 14.0),
                     "open": stands_scenario("open", sides_open, "open", 0.0, 2.5)})

    # roofs
    run = G.StraightRun("south", (-60.0, -45.0), (1, 0), 120.0)
    roofs = {}
    for pname in ("small_seated_8", "single_tier_20", "two_tier_22_18", "three_tier_bowl", "mega_bowl"):
        tiers = S.solve_profile(profiles[pname], 9.0)
        bay_list = G.bays(run, 20.0)
        for rname, rspec in cat("roofs").items():
            sol = RF.solve(tiers, rspec, 9.0, 68.0)
            b = RF.build(run, dict(sol), bay_list, True)
            keys = ("type", "s_back", "s_front", "z_back", "z_front", "depth_back", "depth_front", "length",
                    "back_span", "raised_for_sightline", "raised_for_headroom", "clearance_to_sightline", "s_post")
            roofs[f"{pname}/{rname}"] = {
                "solve": {k: sol.get(k) for k in keys},
                "mesh": mesh_dump(b["mesh"]),
                "tubes": {"count": len(b["tubes"]), "points": sum(len(t[0]) for t in b["tubes"]),
                          "psum": np.sum([t[0].sum(0) for t in b["tubes"]], 0).tolist(),
                          "rsum": float(sum(t[1] for t in b["tubes"]))},
                "edge_line": b["edge_line"], "tail_s": b["tail_s"], "tail_top": b["tail_top"],
                "sockets": len(b["sockets"])}
    write("roofs", roofs)

    # facades (straight and arc runs)
    arc = G.ArcRun("NE", (40.0, 20.0), 14.0, 0.0, math.pi / 2)
    tiers = S.solve_profile(profiles["three_tier_bowl"], 9.0)
    sol = RF.solve(tiers, {"type": "cantilever"}, 9.0, 68.0)
    rb = RF.build(run, dict(sol), G.bays(run, 20.0), True)
    roof_info = {"tail_s": rb["tail_s"], "tail_top": rb["tail_top"]}
    facades = {}
    for fname, spec in cat("facades").items():
        spec = dict(spec)
        for rr, ends in ((run, (True, True)), (arc, (False, False))):
            out = FC.build(rr, tiers, G.bays(rr, 20.0), spec, roof_info, ends)
            facades[f"{fname}/{rr.name}"] = {"mesh": mesh_dump(out["mesh"]), "line_s": out["line_s"],
                                             "height": out["height"], "tubes": len(out["tubes"]),
                                             "tube_points": sum(len(t[0]) for t in out["tubes"]),
                                             "sockets": len(out["sockets"])}
    write("facades", facades)

    # floodlights
    L, W = 105.0, 68.0
    masts = FL.pylons(L, W, 65.5, 43.0, 10.0, 25.0, {})
    heads = [FL.head_layout(m, {"rows": 5, "cols": 7})[0] for m in masts]
    pole_list = FL.poles(L, W, {"per_side": 3, "height": 20.0}, 7.2)
    pts = np.array([[x, y, 0.0] for x in (-40, 0, 30) for y in (-20, 10)])
    fx = [h[0] for h in heads][:4]
    tg = [np.array([0.0, 0.0, 0.0]), np.array([20.0, 5.0, 0.0]), np.array([-30.0, -10.0, 0.0]), np.array([0.0, 20.0, 0.0])]
    gx, gy = np.meshgrid(np.linspace(-L / 2, L / 2, 22), np.linspace(-W / 2, W / 2, 15))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cx, cy = np.meshgrid(np.linspace(-L / 2, L / 2, 15), np.linspace(-W / 2, W / 2, 9))
    cands = np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)])
    small = np.array([(x, s * (W / 2 + 7.2), 20.3) for s in (1, -1) for x in (-30.0, 0.0, 30.0)])
    targets, E = FL.optimise_aims(small, cands, grid, 90)
    write("floodlights", {
        "pylons": masts, "heads": heads, "poles": pole_list,
        "illuminance": FL.illuminance(fx, tg, 1000.0, 40.0, 0.9, pts),
        "ribbon": FL.roof_ribbon([np.array([[0, 0, 30.0], [10, 0, 30.0], [10, 25, 31.0]])], 2.4),
        "spot_angle": FL.spot_angle(160, L * W, 60.0), "lamp_power": FL.lamp_power(160, 40.0, 2000, L * W),
        "optimise": {"fixtures": small, "targets": targets, "E": E, "u": FL.uniformity(E)},
    })

    # ads placement
    cam = np.array([0.0, -70.0, 22.0])
    p = PITCH.resolve({})
    runs = PL.board_runs(p, 22.0)
    write("placement", {
        "board_runs": [{"name": r.name, "p0": r.p0, "dir": r.t, "length": r.t1} for r in runs],
        "carpet_spots": [(s, side, b) for s, side, b in PL.carpet_spots(p)],
        "carpet": dict(zip(("V", "F", "UV"), PL.carpet(cam, np.array([53.3, 8.0, 0.0]), 5.0, 1.0, 8, 4))),
        "fit": PL.fit_carpet_height(cam, np.array([53.3, 8.0, 0.0]), 5.0, 1.1, p["length"] / 2 + 5.1),
        "place": [PL.place_carpet(cam, b, 5.0, 1.1, p, p["length"] / 2 + 5.1) for _, _, b in PL.carpet_spots(p)],
        "end_offset": [PL.end_board_offset(c, p, 5.0, m) for c, m in
                       ((cam, 9.7), (np.array([0.0, -60.85, 14.93]), 9.7), (np.array([0.0, -76.05, 23.05]), 9.7),
                        (np.array([0.0, -60.85, 14.93]), 7.0))],
        "tv": [PL.tv_weight((x, y, 0.5), f, cam) for x, y, f in ((0, 38, (0, -1)), (0, -38, (0, 1)), (56, 0, (-1, 0)))],
    })

    # pitch markings
    out = {}
    for label, params in (("default", {}), ("sunday", {"length": 98, "width": 62, "optional_marks": False,
                                                       "hand_painted": 1.0, "crown": 0.12})):
        pp = PITCH.resolve(params)
        lines, discs = PITCH.marking_paths(pp)
        wob = [PITCH._wobble(pts_, closed, 0.035 * pp["hand_painted"], pp["seed"] * 100 + i)
               for i, (pts_, closed) in enumerate(lines)]
        out[label] = {"lines": [(l, c) for l, c in lines], "discs": discs, "wobbled": wob,
                      "crown_samples": [float(PITCH.crown_fn(pp)(x, y)) for x, y in ((0, 0), (10, 20), (50, -40))]}
    write("pitch", out)

    # patterns
    anchors = [{"run": "south", "tier": t, "row": r, "bay": b, "t": 0.25 + 0.5 * c, "pitch": 0.5}
               for t in (0, 1) for r in range(12) for b in range(2) for c in range(30)]
    tier_rows = [12, 12]
    pats = {}
    for spec in ({"pattern": "solid", "color": "#c8102e"},
                 {"pattern": "tiers", "colors": ["#c8102e", "#1d4ea8"]},
                 {"pattern": "rows", "colors": ["#c8102e", "#ffffff"], "every": 3},
                 {"pattern": "bays", "colors": ["#0b6e4f", "#f2f2ee"]},
                 {"pattern": "gradient", "colors": ["#000000", "#ffffff"]},
                 {"pattern": "random", "colors": ["#0b6e4f", "#f2f2ee", "#d4af37"], "weights": [6, 2, 1]}):
        pats[spec["pattern"]] = PAT.base_colors(anchors, spec, tier_rows, seed=17)
    cols = PAT.base_colors(anchors, {"pattern": "solid", "color": "#1d4ea8"}, tier_rows)
    painted = PAT.apply_mosaic(cols, anchors, {"run": "south", "tier": 0, "text": "KIT FC", "color": "#ffffff"},
                               15.0, tier_rows)
    write("patterns", {"base": pats, "mosaic": {"painted": painted, "colors": cols},
                       "finish": PAT.finish(pats["random"][:50], fade=0.4, seed=5),
                       "bitmap": PAT.text_bitmap("KIT FC 9").astype(int), "crc": [PAT_crc(s) for s in ("south", "NE")]})


def PAT_crc(s):
    from kit.stands.prng import crc32
    return crc32(s)


if __name__ == "__main__":
    main()
