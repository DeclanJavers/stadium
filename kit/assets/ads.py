"""Advertising: perimeter boards (LED or printed), 3D carpets, LED fascia ribbons.

Layout section:
  "ads": {
    "brands": ["northwind", ...],                         # default: every brand in catalog/brands.json
    "perimeter": {"type": "led" | "static" | "none"},
    "carpets": true | {"brands": [...], "width": 5.0},
    "ribbons": {"tiers": [1, 2]}                          # LED ribbons on upper-tier fascias
  }
Every placement is recorded as an `ad_slot` socket (kind, area m², TV weight, brand) and summed into
an inventory for the game's revenue model.
"""
import math
import zlib

import bpy
import numpy as np

from ..ads import placement as PL
from ..ads import studio as STUDIO
from ..core.nodes import material, srgb
from ..core.scene import collection, empty, mesh_object, set_params
from ..core.sockets import socket
from ..stands.geometry import MeshAcc, WALL
from .seat import galvanised
from .stand import _recalc, painted

CREATIVE_M = 8.0        # an 8:1 creative spans 8 m of a 1 m board


def _stable(text):
    """Deterministic hash (Python's str hash is salted per process)."""
    return zlib.crc32(str(text).encode())


def _image(path):
    img = bpy.data.images.load(str(path), check_existing=True)
    return img


def led_material(name, img, strength, px):
    """LED panel: emissive texture seen through a grid of LED pixels (one per texel)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", image=img, interpolation="Closest", extension="REPEAT")
    nb.connect(uv, tex.inputs["Vector"])
    cell = nb.vmath("FRACTION", nb.vmath("MULTIPLY", uv, (px[0], px[1], 1.0)))
    cx, cy, _ = nb.sep(cell)
    dx, dy = nb.sub(cx, 0.5), nb.sub(cy, 0.5)
    r = nb.math("SQRT", nb.add(nb.mul(dx, dx), nb.mul(dy, dy)))
    dot = nb.madd(nb.sub(1.0, nb.smooth(0.28, 0.5, r)), 0.7, 0.3)   # lit dots, dimmer gaps
    emit = nb.mix(1.0, tex.outputs["Color"], nb.comb(dot, dot, dot), blend="MULTIPLY")
    nb.output(nb.principled(Base_Color=srgb("#050505"), Roughness=0.25, Coat_Weight=0.6, Coat_Roughness=0.1,
                            Emission_Color=emit, Emission_Strength=strength))
    return m


def print_material(name, img, rough=0.55, cloth=False):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", image=img, interpolation="Linear", extension="REPEAT")
    nb.connect(uv, tex.inputs["Vector"])
    pos = nb.geometry()["Position"]
    weave = nb.noise(pos, 700.0 if cloth else 150.0, detail=1)
    nb.output(nb.principled(Base_Color=tex.outputs["Color"], Roughness=rough, Sheen_Weight=0.4 if cloth else 0.0,
                            Normal=nb.bump(weave, 0.25 if cloth else 0.05, 0.001)))
    return m


def _strip(run, t0, t1, s_bot, z_bot, s_top, z_top, u0, u_per_m, lift=0.004):
    """Quad strip between two section points along a run, offset toward the pitch."""
    d = np.array([s_top - s_bot, z_top - z_bot])
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)        # towards smaller s (the pitch)
    if nrm[0] > 0:
        nrm = -nrm
    sb, zb = s_bot + nrm[0] * lift, z_bot + nrm[1] * lift
    st, zt = s_top + nrm[0] * lift, z_top + nrm[1] * lift
    ts = run.samples(t0, t1, max(s_bot, s_top))
    V, UV = [], []
    for t in ts:
        p, n = run.frame(t)
        along = (t - run.t0) * run.metric(s_bot)
        for s, z, v in ((sb, zb, 0.0), (st, zt, 1.0)):
            q = p + n * s
            V.append((q[0], q[1], z))
            # seen from the pitch a run reads right-to-left, so the texture runs against t
            UV.append((u0 - along * u_per_m, v))
    F = [(2 * j, 2 * j + 2, 2 * j + 3, 2 * j + 1) for j in range(len(ts) - 1)]
    return np.array(V), np.array(F), np.array(UV)


def _mesh_with_vertex_uvs(name, V, F, UV, col, parent, mat):
    loop_uv = np.concatenate([UV[list(f)] for f in F]) if len(F) else np.zeros((0, 2))
    return mesh_object(name, V, F, col, parent, materials=[mat], uvs=loop_uv)


def build(cfg, pitch, stands, night, col=None, gap=0.0):
    if not cfg:
        return None
    col = col or collection("Ads")
    root = empty("Ads", col, size=1.5)
    set_params(root, "ads", "ads", "stadium_origin", cfg)
    brands_all = STUDIO.brand_list()
    keys = cfg.get("brands") or [k for k, b in brands_all.items() if b.get("tier", 1) > 0]
    atlas_path, atlas_keys = STUDIO.render_boards(keys)
    atlas = _image(atlas_path)
    n_atlas = len(atlas_keys)
    cam, _ = PL.main_camera(pitch, stands)
    strength = 1.2 if night else 4.0
    inventory = []

    def slot(kind, ident, centre, facing, area, brand):
        w = PL.tv_weight(centre, facing, cam)
        inventory.append({"kind": kind, "id": ident, "area_m2": round(area, 2), "tv_weight": w, "brand": brand})
        yaw = math.degrees(math.atan2(-facing[0], facing[1]))
        socket("ad_slot", ident, tuple(float(v) for v in centre), yaw, parent=root, col=col, ad_kind=kind,
               area_m2=round(area, 2), tv_weight=w, brand=brand)

    # -- perimeter boards
    per = cfg.get("perimeter", {"type": "led"})
    kind = per.get("type", "led")
    housing = painted("Board_Housing", "#111214", 0.5)
    if kind != "none":
        runs = PL.board_runs(pitch, gap)
        if kind == "led":
            mat = led_material("Ads_LED", atlas, strength, (atlas.size[0], atlas.size[1]))
            for run in runs:
                acc = MeshAcc()
                poly = np.array([(0.0, 0.0), (0.42, 0.0), (0.42, 0.92), (0.2, 1.0), (0.17, 0.97), (0.0, 0.1)])
                acc.add_prism(run, run.t0, run.t1, (poly, [0] * len(poly)), cap_tag=0)
                V, F, _, UV = acc.arrays()
                o = mesh_object(f"BoardHousing_{run.name}", V, F, col, root, materials=[housing], uvs=UV)
                _recalc(o)
                offset = (_stable(run.name) % n_atlas) / n_atlas
                V, F, UVv = _strip(run, run.t0, run.t1, 0.02, 0.12, 0.165, 0.95, offset,
                                   1.0 / (CREATIVE_M * n_atlas))
                _mesh_with_vertex_uvs(f"BoardLED_{run.name}", V, F, UVv, col, root, mat)
                length = (run.t1 - run.t0)
                for k in range(int(length // CREATIVE_M)):
                    p, n = run.frame(run.t0 + (k + 0.5) * CREATIVE_M)
                    slot("perimeter_led", f"{run.name}_{k}", (p[0], p[1], 0.5), -n, CREATIVE_M * 0.85, "rotation")
        else:
            panel = 7.2
            legs = []
            backs = MeshAcc()
            for run in runs:
                length = run.t1 - run.t0
                n_p = max(1, int(length // (panel + 0.2)))
                pad = (length - n_p * (panel + 0.2)) / 2
                for k in range(n_p):
                    ta = run.t0 + pad + k * (panel + 0.2)
                    b = keys[(k + _stable(run.name)) % len(keys)]
                    bi = atlas_keys.index(b)
                    img_mat = print_material("Ads_Print", atlas)
                    V, F, UVv = _strip(run, ta, ta + panel, 0.0, 0.05, 0.24, 0.92, 0.0, 0.0, lift=0.0)
                    # start of the panel (in run order) is the reader's right-hand end
                    UVv[:, 0] = np.where(np.arange(len(UVv)) // 2 == 0, (bi + 1) / n_atlas, bi / n_atlas)
                    _mesh_with_vertex_uvs(f"Board_{run.name}_{k}", V, F, UVv, col, root, img_mat)
                    # the panel's back: a printed board is a sheet on a frame, not a see-through quad
                    back = np.array([(0.02, 0.05), (0.035, 0.05), (0.275, 0.92), (0.26, 0.92)])
                    backs.add_prism(run, ta, ta + panel, (back, [0] * 4), cap_tag=0)
                    for t in (ta + 0.4, ta + panel / 2, ta + panel - 0.4):
                        p, nrm = run.frame(t)
                        top = np.array([*(p + nrm * 0.24), 0.9])
                        foot = np.array([*(p + nrm * 0.75), 0.0])
                        legs.append((np.array([top, foot]), 0.02))
                    p, nrm = run.frame(ta + panel / 2)
                    slot("perimeter_static", f"{run.name}_{k}", (p[0], p[1], 0.5), -nrm, panel * 0.9, b)
            from .stand import _tubes_mesh
            _tubes_mesh("BoardLegs", legs, col, root, galvanised(), sides=6)
            V, F, _, UV = backs.arrays()
            o = mesh_object("BoardBacks", V, F, col, root, materials=[housing], uvs=UV)
            _recalc(o)

    # -- 3D carpets
    cp = cfg.get("carpets")
    if cp:
        cp = cp if isinstance(cp, dict) else {}
        width = cp.get("width", 5.0)
        ckeys = cp.get("brands") or keys
        images = STUDIO.render_carpets()
        limit = pitch["length"] / 2 + PL.BOARD_END_OFFSET - 0.4
        from .pitch import crown_fn
        zf = crown_fn(pitch)
        for k, (s, side, base) in enumerate(PL.carpet_spots(pitch, width)):
            b = ckeys[k % len(ckeys)]
            h = PL.fit_carpet_height(cam, base, width, 1.1, limit)
            V, F, UVv = PL.carpet(cam, base, width, h, ground=lambda x, y: float(zf(x, y)))
            mat = print_material(f"Carpet_{b}", _image(images[b]), rough=0.85, cloth=True)
            _mesh_with_vertex_uvs(f"Carpet_{k}", V, F, UVv, col, root, mat)
            footprint = float(np.ptp(V[:, 0]) * np.ptp(V[:, 1]))
            to_cam = cam - base
            slot("carpet_3d", f"carpet_{k}", (base[0], base[1], 0.0), to_cam / np.linalg.norm(to_cam), width * h, b)

    # -- LED ribbons on upper-tier fascias
    rb = cfg.get("ribbons")
    if rb and stands:
        mat = led_material("Ads_Ribbon", atlas, strength * 0.8, (atlas.size[0], atlas.size[1]))
        tiers_wanted = set(rb.get("tiers", [1, 2, 3]))
        for name, (run, tiers) in stands["runs"].items():
            for t in tiers:
                if t["index"] not in tiers_wanted or t["index"] == 0:
                    continue
                top = t["h"][0] + t["parapet"]
                s_face = t["s0"] - WALL
                V, F, UVv = _strip(run, run.t0, run.t1, s_face, top - 0.98, s_face, top - 0.06,
                                   (_stable(name) % n_atlas) / n_atlas, 1.0 / (CREATIVE_M * n_atlas), lift=0.03)
                _mesh_with_vertex_uvs(f"Ribbon_{name}_T{t['index']}", V, F, UVv, col, root, mat)
                length = (run.t1 - run.t0) * run.metric(s_face)
                p, n = run.frame(0.5 * (run.t0 + run.t1))
                c = p + n * s_face
                slot("led_ribbon", f"{name}_T{t['index']}", (c[0], c[1], top - 0.5), -n, length * 0.92, "rotation")
    totals = {}
    for s in inventory:
        k = totals.setdefault(s["kind"], {"slots": 0, "area_m2": 0.0, "tv_weighted_m2": 0.0})
        k["slots"] += 1
        k["area_m2"] = round(k["area_m2"] + s["area_m2"], 1)
        k["tv_weighted_m2"] = round(k["tv_weighted_m2"] + s["area_m2"] * s["tv_weight"], 1)
    root["inventory"] = str(totals)
    print(f"[ads] inventory {totals}", flush=True)
    return {"root": root, "inventory": inventory, "totals": totals, "camera": cam}
