"""Dump the Python training-ground generator's output for the Rust port's parity test
(viewer/gen/tests/site_parity.rs): every layout in viewer/gen/tests/sites/ ->
viewer/gen/tests/reference/site_<name>.json.

Plain Python (no Blender):  python3 tools/dump_site_reference.py   (numpy needed)
Re-run whenever the site generator changes on purpose.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit.site import site as S  # noqa: E402

SITES = ROOT / "viewer" / "gen" / "tests" / "sites"
OUT = ROOT / "viewer" / "gen" / "tests" / "reference"


def vsum(pts, n):
    s = [0.0] * n
    for p in pts:
        for i in range(n):
            s[i] += p[i]
    return s


def mesh_dump(m):
    V = m.verts
    return {"n_verts": len(V), "n_faces": len(m.faces), "face_sizes": [len(f) for f in m.faces], "tags": list(m.tags),
            "bounds": [[min(p[i] for p in V) for i in range(3)], [max(p[i] for p in V) for i in range(3)]] if V else [[0, 0, 0], [0, 0, 0]],
            "vsum": vsum(V, 3), "uvsum": vsum(m.uvs, 2), "first_faces": [list(f) for f in m.faces[:6]]}


def mat_dump(m):
    return {"key": m.key, "color": list(m.color) if m.color else None,
            "emission": [list(m.emission[0]), m.emission[1]] if m.emission else None,
            "texture": [m.texture[0], m.texture[1]] if m.texture else None}


def dump(site):
    parts = []
    for p in site.parts:
        d = {"name": p.name, "mats": [mat_dump(m) for m in p.mats], "smooth_deg": p.smooth_deg, "sides": p.sides}
        if p.tubes:
            pts = [q for line, _ in p.tubes for q in line]
            d["tubes"] = {"lines": len(p.tubes), "points": len(pts), "psum": vsum(pts, 3), "rsum": sum(r for _, r in p.tubes)}
        else:
            d["mesh"] = mesh_dump(p.mesh)
        parts.append(d)
    props = [{"key": ps.key, "tints": [[n, list(c)] for n, c in ps.tints], "count": len(ps.items),
              "pos": vsum([it.pos for it in ps.items], 3), "yaw": sum(it.yaw for it in ps.items),
              "tilt": sum(it.tilt for it in ps.items), "scale": vsum([it.scale for it in ps.items], 3)} for ps in site.props]
    pitches = [{"plot": p.plot, "params": p.params, "origin": list(p.origin), "yaw": p.yaw, "goals": p.goals, "both_goals": p.both_goals,
                "flags": p.flags, "lit": p.lit, "surface": p.surface} for p in site.pitches]
    spots = {"count": len(site.spots), "pos": vsum([s.pos for s in site.spots], 3), "dir": vsum([s.dir for s in site.spots], 3),
             "power": sum(s.power for s in site.spots), "angle": sum(s.angle_deg for s in site.spots)}
    sockets = [{"kind": s.kind, "id": s.id, "loc": list(s.loc), "n": list(s.n), "props": [[k, v] for k, v in s.props]} for s in site.sockets]
    r = site.report
    report = {"template": r.template, "plots": r.plots, "pitches": r.pitches, "grass": r.grass, "hybrid": r.hybrid,
              "artificial": r.artificial, "lit": r.lit, "replica": r.replica, "parking": r.parking, "area_m2": r.area_m2,
              "developed_m2": r.developed_m2, "walk_m": r.walk_m, "academy": r.academy, "warnings": r.warnings}
    return {"name": site.name, "night": site.night, "parts": parts, "props": props, "pitches": pitches, "spots": spots,
            "sockets": sockets, "camera": [list(site.camera[0]), list(site.camera[1])], "boundary": [list(p) for p in site.boundary],
            "report": report}


def main():
    cat = S.load_catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SITES.glob("*.json")):
        site = S.build(json.loads(f.read_text()), cat)
        (OUT / f"site_{f.stem}.json").write_text(json.dumps(dump(site), separators=(",", ":")))
        print(f"[ref] site_{f.stem}.json: {len(site.parts)} parts, {sum(len(p.items) for p in site.props)} props")


if __name__ == "__main__":
    main()
