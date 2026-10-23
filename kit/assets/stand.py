"""Stands: turns the pure geometry from kit.stands into Blender objects, materials and seat instances.

Layout section (all optional except sides):
{
  "front_gap": 3.0, "corners": "open" | "curved", "corner_radius": 14, "apron": "turf|concrete|track",
  "sides": {"north": {"profile": "two_tier_22_18", "seat": "tipup_classic", "colors": {...}, "length": 60}, ...},
  "colors": {"pattern": "solid", "color": "#c8102e"},     # default for every side
  "mosaics": [{"run": "south", "tier": 0, "text": "KIT FC", "color": "#ffffff"}],
  "hospitality_seat": "padded_vip", "seat_fade": 0.0, "fascia_color": "#1b1d20"
}
"""
import math
import zlib

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import NB, material, srgb
from ..core.scene import collection, empty, link, mesh_object, set_params
from ..core.sockets import socket
from ..stands import geometry as G
from ..stands import patterns as PAT
from ..stands import sightlines as S
from . import seat as SEAT

STANDING_PER_M = 1.8   # spectators per metre of terrace step (Green Guide 4.7/m2 on 0.38 m steps)


def _stable(text):
    """Deterministic hash (Python's str hash is salted per process)."""
    return zlib.crc32(str(text).encode())


# -- materials --------------------------------------------------------------------------------

def concrete(name="Stand_Concrete", color="#9d998f"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    geo = nb.geometry()
    pos, nrm = geo["Position"], geo["Normal"]
    up = nb.smooth(0.6, 0.9, nb.sep(nrm)[2])                       # treads (horizontal)
    tone = nb.noise(pos, 0.9, detail=5, rough=0.6)
    fine = nb.noise(pos, 60.0, detail=4, rough=0.7)
    base = nb.mix(nb.madd(tone, 0.6, nb.mul(fine, 0.4)), srgb(color), srgb("#7f7b72"))
    # rain streaks down vertical faces, dirt and wear on treads
    streak = nb.smooth(0.5, 0.75, nb.noise(nb.vmath("MULTIPLY", pos, (6.0, 6.0, 0.35)), 1.0, detail=3))
    base = nb.mix(nb.mul(nb.mul(streak, nb.sub(1.0, up)), 0.35), base, srgb("#5f5c55"))
    grime = nb.smooth(0.45, 0.8, nb.noise(pos, 3.0, detail=6))
    base = nb.mix(nb.mul(up, nb.madd(grime, 0.35, 0.15)), base, srgb("#6c6860"))
    pores = nb.smooth(0.7, 0.75, nb.noise(pos, 180.0, detail=1))
    bsdf = nb.principled(Base_Color=base, Roughness=nb.madd(up, -0.08, 0.88),
                         Normal=nb.bump(nb.add(nb.mul(fine, 0.5), nb.mul(pores, -0.5)), 0.25, 0.003))
    nb.output(bsdf)
    return m


def painted(name, color, rough=0.5):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    v = nb.noise(pos, 2.0, detail=4)
    col = nb.mix(nb.mul(v, 0.15), srgb(color), nb.mix(1.0, srgb(color), (0.8, 0.8, 0.8), blend="MULTIPLY"))
    nb.output(nb.principled(Base_Color=col, Roughness=rough, Normal=nb.bump(nb.noise(pos, 90.0), 0.05, 0.001)))
    return m


def glass(name="Balustrade_Glass"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#dfeee6"), Roughness=0.03, Transmission_Weight=1.0, IOR=1.5))
    return m


def apron_material(kind):
    if kind == "track":
        return painted("Apron_Track", "#9c3f2c", rough=0.85)
    if kind == "concrete":
        return concrete("Apron_Concrete", "#77746e")
    return bpy.data.materials.get("Pitch_Grass") or painted("Apron_Turf", "#3e6b2c", 0.8)


# -- helpers ----------------------------------------------------------------------------------

def _tubes_mesh(name, polylines, col, parent, mat, sides=8):
    vs, fs, off = [], [], 0
    for pts, r in polylines:
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        V, F, _ = M.tubes(pts[:-1], pts[1:], r, sides=sides, extend=r * 0.5)
        vs.append(V)
        fs.append(F + off)
        off += len(V)
    if not vs:
        return None
    return mesh_object(name, np.concatenate(vs), np.concatenate(fs), col, parent, smooth=True, materials=[mat])


def _nosings_mesh(name, segs, col, parent, mat):
    if not segs:
        return None
    V, F = [], []
    for k, (p0, p1) in enumerate(segs):
        d = p1 - p0
        n = np.array([d[1], -d[0], 0.0])
        n /= max(1e-9, np.linalg.norm(n))
        lift = np.array([0, 0, 0.002])
        V += [p0 + lift, p1 + lift, p1 + n * 0.055 + lift, p0 + n * 0.055 + lift]
        F.append((4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3))
    return mesh_object(name, np.array(V), np.array(F), col, parent, materials=[mat])


def _barrier_lines(barriers):
    out = []
    for pts in barriers:
        out.append((pts, 0.03))
        for p in pts[::2]:
            out.append((np.array([p - np.array([0, 0, 1.1]), p]), 0.03))
    return out


def _recalc(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def smooth_by_angle(obj, degrees=35.0):
    """Smooth curved surfaces (arcs, cushions) while keeping hard architectural edges sharp."""
    obj.data.shade_smooth()
    obj.data.set_sharp_from_angle(angle=math.radians(degrees))


def paving(name="Plaza_Paving"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    br = nb.node("ShaderNodeTexBrick", offset=0.5)
    nb.connect(pos, br.inputs["Vector"])
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.008
    br.inputs["Brick Width"].default_value = 0.6
    br.inputs["Row Height"].default_value = 0.3
    br.inputs["Color1"].default_value = srgb("#a7a399")
    br.inputs["Color2"].default_value = srgb("#958f84")
    br.inputs["Mortar"].default_value = srgb("#6e6a62")
    stain = nb.noise(pos, 0.15, detail=5)
    col = nb.mix(nb.mul(nb.smooth(0.5, 0.8, stain), 0.3), br.outputs["Color"], srgb("#6f6b63"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.8, Normal=nb.bump(nb.sub(1.0, br.outputs["Fac"]), 0.3, 0.004)))
    return m


def _plaza(A, B, reach_x, reach_y, R, col, parent):
    """Paved concourse around the outside of the stadium."""
    import numpy as np
    n = 256
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hx, hy, r = A + reach_x + 28.0, B + reach_y + 28.0, max(R, 20.0) + 28.0
    pts = []
    for a in ang:
        dx, dy = math.cos(a), math.sin(a)
        lo, hi = 0.0, 4 * (hx + hy)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            x, y = abs(dx * mid), abs(dy * mid)
            qx, qy = max(x - (hx - r), 0), max(y - (hy - r), 0)
            inside = x <= hx and y <= hy and qx * qx + qy * qy <= r * r
            lo, hi = (mid, hi) if inside else (lo, mid)
        pts.append((dx * lo, dy * lo, -0.012))
    V = np.array([(0.0, 0.0, -0.012)] + pts)
    j = np.arange(n)
    F = np.stack([np.zeros(n, int), 1 + j, 1 + (j + 1) % n, 1 + (j + 1) % n], 1)[:, :3]
    return mesh_object("Plaza", V, F, col, parent, materials=[paving()])


def _apron(p, A, B, R, corners, col, parent, kind):
    """Ground between the pitch run-off and the stand fronts."""
    ix, iy = p["length"] / 2 + p["runoff_end"], p["width"] / 2 + p["runoff_side"]
    n = 360
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)

    def rect_hit(hx, hy, r):
        # point on a rounded rectangle along each direction (ray from the centre)
        out = []
        for dx, dy in d:
            lo, hi = 0.0, 4 * (hx + hy)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                x, y = abs(dx * mid), abs(dy * mid)
                qx, qy = max(x - (hx - r), 0), max(y - (hy - r), 0)
                inside = (x <= hx and y <= hy) and (qx * qx + qy * qy <= r * r)
                lo, hi = (mid, hi) if inside else (lo, mid)
            out.append((dx * lo, dy * lo))
        return np.array(out)
    inner = rect_hit(ix, iy, 0.0)
    outer = rect_hit(A + 0.3, B + 0.3, R if corners == "curved" else 0.0)
    V = np.concatenate([np.column_stack([inner, np.full(n, -0.005)]), np.column_stack([outer, np.full(n, -0.005)])])
    j = np.arange(n)
    F = np.stack([j, (j + 1) % n, n + (j + 1) % n, n + j], 1)
    return mesh_object("Apron", V, F, col, parent, materials=[apron_material(kind)])


def _seat_points(name, anchors, colors, style, col, parent):
    spec = SEAT.resolve(style)
    var_col = collection(f"SeatModels_{style}")
    if not var_col.objects:
        SEAT.build_style(style, var_col)
    bpy.context.view_layer.layer_collection.children[var_col.name].exclude = True
    N = len(anchors)
    V = np.array([a["pos"] for a in anchors]) if N else np.zeros((0, 3))
    me = bpy.data.meshes.new(name)
    me.vertices.add(N)
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    rot = np.zeros((N, 3), np.float32)
    rot[:, 2] = [a["yaw"] for a in anchors]
    variant = np.array([SEAT.variant_index(spec, a["seat"], a["of"]) for a in anchors], np.int32)
    sx = np.array([a["pitch"] / spec["width"] if spec.get("stretch") else 1.0 for a in anchors], np.float32)
    me.attributes.new("rot", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", rot.ravel())
    me.attributes.new("variant", "INT", "POINT").data.foreach_set("value", variant)
    me.attributes.new("sx", "FLOAT", "POINT").data.foreach_set("value", sx)
    me.attributes.new("seat_color", "FLOAT_COLOR", "POINT").data.foreach_set(
        "color", np.asarray(colors, np.float32).ravel())
    obj = link(bpy.data.objects.new(name, me), col, parent)
    obj["seat_style"] = style
    obj["seats"] = N
    mod = obj.modifiers.new("Seats", "NODES")
    mod.node_group = _seat_instancer(var_col)
    return obj


def _seat_instancer(var_col):
    name = f"SeatInstancer_{var_col.name}"
    if bpy.data.node_groups.get(name):
        return bpy.data.node_groups[name]
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = NB(tree)
    gin, gout = nb.node("NodeGroupInput"), nb.node("NodeGroupOutput")
    m2p = nb.node("GeometryNodeMeshToPoints")
    nb.connect(gin.outputs[0], m2p.inputs["Mesh"])
    ci = nb.node("GeometryNodeCollectionInfo", transform_space="RELATIVE")
    ci.inputs["Collection"].default_value = var_col
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True

    def named(attr, dtype):
        n = nb.node("GeometryNodeInputNamedAttribute", data_type=dtype)
        n.inputs["Name"].default_value = attr
        return n.outputs["Attribute"]
    iop = nb.node("GeometryNodeInstanceOnPoints")
    nb.connect(m2p.outputs[0], iop.inputs["Points"])
    nb.connect(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    nb.connect(named("variant", "INT"), iop.inputs["Instance Index"])
    e2r = nb.node("FunctionNodeEulerToRotation")
    nb.connect(named("rot", "FLOAT_VECTOR"), e2r.inputs[0])
    nb.connect(e2r.outputs[0], iop.inputs["Rotation"])
    nb.connect(nb.comb(named("sx", "FLOAT"), 1.0, 1.0), iop.inputs["Scale"])
    nb.connect(iop.outputs[0], gout.inputs[0])
    return tree


# -- build ------------------------------------------------------------------------------------

def resolve_sides(cfg):
    profiles = catalog.load("stand_profiles")
    sides = {}
    for name, sc in (cfg.get("sides") or {}).items():
        if not sc or not sc.get("profile"):
            continue
        if sc["profile"] not in profiles:
            raise KeyError(f"unknown stand profile {sc['profile']!r}; options: {sorted(profiles)}")
        sides[name] = dict(sc, _profile=profiles[sc["profile"]])
    return sides


def build(cfg, pitch_params, col=None):
    cfg = cfg or {}
    p = pitch_params
    col = col or collection("Stands")
    root = empty("Stands", col, display="PLAIN_AXES", size=3.0)
    sides = resolve_sides(cfg)
    gap = cfg.get("front_gap", 3.0)
    corners = cfg.get("corners", "open")
    R = cfg.get("corner_radius", 14.0)
    A = p["length"] / 2 + p["runoff_end"] + gap
    B = p["width"] / 2 + p["runoff_side"] + gap
    if corners == "curved":
        profs = {s["profile"] for s in sides.values()}
        if len(sides) < 4 or len(profs) != 1:
            raise ValueError("curved corners need all four sides to share one profile (a bowl); "
                             f"got {sorted(profs)} on {sorted(sides)}")
    set_params(root, "stands", "stands", "stadium_origin", {k: v for k, v in cfg.items()})

    mats = [concrete(color=cfg.get("concrete", "#9d998f")),
            painted("TINT_fascia", cfg.get("fascia_color", "#1b1d20"), 0.4),
            painted("Stand_Soffit", "#d9d6cf", 0.8), concrete("Stand_Block", "#b3aea4")]
    rail_mat = SEAT.galvanised()
    nose_mat = painted("Step_Nosing", "#f2c200", 0.55)
    glass_mat = glass()

    run_defs = G.perimeter_runs(A, B, sides, corners, R)
    d_focal = {"south": p["runoff_side"] + gap, "north": p["runoff_side"] + gap,
               "east": p["runoff_end"] + gap, "west": p["runoff_end"] + gap}
    any_side = next(iter(sides.values())) if sides else None
    all_anchors, run_lengths, report, standing = [], {}, {}, 0.0
    camera_mount = None
    warnings = []
    roofs, facades = {}, {}
    runs_out = {}
    tier_rows_by_run, vom_top_by_run = {}, {}
    for name, run in run_defs:
        sc = sides.get(name, any_side)
        df = min(d_focal.values()) if corners == "curved" else d_focal[name]
        tiers = S.solve_profile(sc["_profile"], df)
        style = sc.get("seat", "tipup_classic")
        for t in tiers:
            st = cfg.get("hospitality_seat", "padded_vip") if t["hospitality"] else style
            t["seat_style"] = st
            spec = SEAT.resolve(st)
            t["seat_pitch"] = max(t["seat_pitch"], spec["width"])
            if t["kind"] == "seated" and t["tread"] < spec["min_tread"] - 1e-9:
                warnings.append(f"{name} tier {t['index']}: {st} needs {spec['min_tread']} m rows for a "
                                f"305 mm seatway; tier has {t['tread']} m")
        out = G.build_run(run, tiers)
        V, F, tags, UV = out["mesh"]
        obj = mesh_object(f"Stand_{name}", V, F, col, root, materials=mats, mat_index=tags, uvs=UV)
        _recalc(obj)
        far = p["width"] if name in ("north", "south") or run.curved else p["length"]
        roof_spec = sc.get("roof", cfg.get("roof"))
        from . import roof as ROOF
        rf = ROOF.build(run, tiers, out["bay_list"], roof_spec, df, far, col, root,
                        include_ends=not run.curved)
        if rf:
            roofs[name] = rf
        from . import facade as FACADE
        fc = FACADE.build(run, tiers, out["bay_list"], sc.get("facade", cfg.get("facade")), rf,
                          (corners != "curved",) * 2, col, root, night=cfg.get("_night", False))
        if fc:
            facades[name] = fc
        _glass(run, tiers, col, root, glass_mat, name)
        _tubes_mesh(f"Rails_{name}", out["rails"] + _barrier_lines(out["barriers"]), col, root, rail_mat)
        _nosings_mesh(f"Nosings_{name}", out["nosings"], col, root, nose_mat)
        for kind, ident, loc, n, props in out["sockets"]:
            socket(kind, ident, tuple(float(v) for v in loc), math.degrees(math.atan2(n[0], -n[1])),
                   parent=root, col=col, **props)
        for a in out["anchors"]:
            a["style"] = tiers[a["tier"]]["seat_style"]
            a["side"] = name if name in sides else "corner"
        all_anchors += out["anchors"]
        run_lengths[name] = (run.t1 - run.t0) * run.metric(0.0)
        tier_rows_by_run[name] = [t["rows"] for t in tiers]
        vom_top_by_run[name] = [sum(G.vom_rows(t)) if G.vom_rows(t) else None for t in tiers]
        for t in tiers:
            if t["kind"] == "standing":
                standing += STANDING_PER_M * t["rows"] * (run.t1 - run.t0) * run.metric(t["s0"]) * 0.88
        report[name] = S.report(tiers)
        runs_out[name] = (run, tiers)
        if name == "south":
            t = min(tiers, key=lambda t: abs(t["h"][0] - 16.0))
            pnt, n = run.frame(0.5 * (run.t0 + run.t1))
            q = pnt + n * (t["s0"] - 1.2)
            camera_mount = (float(q[0]), float(q[1]), t["h"][0] + t["parapet"] + 1.6)

    mosaics = []
    for mz in cfg.get("mosaics", []):
        mz = dict(mz)
        rows = tier_rows_by_run[mz["run"]][mz.get("tier", 0)]
        top = vom_top_by_run[mz["run"]][mz.get("tier", 0)]
        if "rows" not in mz and top is not None and rows - top >= 7:
            mz["rows"] = [top, rows - 1]  # keep letters clear of the vomitory mouths
        mosaics.append(mz)
    colors = _colours(dict(cfg, mosaics=mosaics), sides, all_anchors, run_lengths, tier_rows_by_run)
    seats = {}
    for style in sorted({a["style"] for a in all_anchors}):
        idx = [i for i, a in enumerate(all_anchors) if a["style"] == style]
        _seat_points(f"Seats_{style}", [all_anchors[i] for i in idx], colors[idx], style, col, root)
        seats[style] = len(idx)
    _apron(p, A, B, R, corners, col, root, cfg.get("apron", "turf"))
    if cfg.get("plaza", bool(facades)):
        reach_x = max([facades[k]["line_s"] for k in ("east", "west") if k in facades] + [20.0])
        reach_y = max([facades[k]["line_s"] for k in ("north", "south") if k in facades] + [20.0])
        _plaza(A, B, reach_x, reach_y, R if corners == "curved" else 0.0, col, root)
    capacity = {"seated": len(all_anchors), "standing": int(standing), "by_style": seats}
    capacity["total"] = capacity["seated"] + capacity["standing"]
    root["capacity"] = capacity["total"]
    for w in dict.fromkeys(warnings):
        print(f"[stands] warning: {w}", flush=True)
    return {"root": root, "capacity": capacity, "warnings": list(dict.fromkeys(warnings)), "seat_colors": colors, "report": report, "A": A, "B": B, "anchors": all_anchors,
            "tiers_by_run": tier_rows_by_run, "camera_mount": camera_mount,
            "roofs": roofs, "facades": facades, "runs": runs_out}


def _glass(run, tiers, col, parent, mat, name):
    acc = G.MeshAcc()
    for t in tiers:
        if not t.get("glass"):
            continue
        s, z = t["s0"] - G.WALL + 0.12, t["h"][0] + t["parapet"]
        poly = np.array([(s, z), (s + 0.02, z), (s + 0.02, z + t["glass"]), (s, z + t["glass"])])
        acc.add_prism(run, run.t0, run.t1, (poly, [0] * 4))
    V, F, _, UV = acc.arrays()
    if len(F):
        mesh_object(f"Glass_{name}", V, F, col, parent, materials=[mat])


def _colours(cfg, sides, anchors, run_lengths, tier_rows_by_run):
    default = cfg.get("colors", {"pattern": "solid", "color": "#c8102e"})
    out = np.zeros((len(anchors), 3))
    groups = {}
    for i, a in enumerate(anchors):
        groups.setdefault(a["run"], []).append(i)
    for run, idx in groups.items():
        side_cfg = sides.get(run, {})
        spec = side_cfg.get("colors", default)
        sub = [anchors[i] for i in idx]
        out[idx] = PAT.base_colors(sub, spec, tier_rows_by_run[run], seed=_stable(run) % 1000)
    for mz in cfg.get("mosaics", []):
        image = _load_image(mz["image"]) if mz.get("image") else None
        PAT.apply_mosaic(out, anchors, mz, run_lengths[mz["run"]], tier_rows_by_run[mz["run"]], image)
    return PAT.finish(out, cfg.get("seat_fade", 0.0))


def _load_image(path):
    from ..core.scene import ROOT
    img = bpy.data.images.load(str(ROOT / path))
    w, h = img.size
    # byte images expose their stored (sRGB) values; flip so row 0 is the top
    return np.array(img.pixels[:]).reshape(h, w, 4)[::-1, :, :3]
