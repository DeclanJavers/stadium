"""Roof objects for a stand run: deck (profiled steel, translucent strip), fascia, steelwork.

Roof layout section (stands.roof, overridable per side):
  {"preset": "modern_cantilever"} or {"type": "cantilever"|"goalpost", ...solver params,
   "color": "#e3e4e2", "fascia_color": "#1b1d20", "steel_color": "#eef0f0", "deck": "metal|membrane"}
"""
import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import mesh_object
from ..core.sockets import socket
from ..stands import roofs as RF
from .stand import _recalc, _tubes_mesh, painted, smooth_by_angle


GOALPOST_MAX_HEIGHT = 25.0   # highest front girder a goalpost roof is built with


def resolve(spec):
    if spec is None or spec is False:
        return None
    spec = dict(spec)
    if "preset" in spec:
        spec = catalog.merge(catalog.preset("roofs", spec.pop("preset")), spec)
    return spec


def deck_material(color, membrane=False):
    name = f"TINT_roof{'_membrane' if membrane else ''}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.sep(nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0])
    pos = nb.geometry()["Position"]
    if membrane:
        weave = nb.noise(pos, 300.0, detail=1)
        bsdf = nb.principled(Base_Color=srgb(color), Roughness=0.55, Transmission_Weight=0.35, Sheen_Weight=0.3,
                             Normal=nb.bump(weave, 0.05, 0.001))
    else:
        # trapezoidal profiled sheet: ribs every 250 mm across the run
        rib = nb.smooth(0.25, 0.4, nb.math("PINGPONG", nb.div(uv[0], 0.25), 0.5))
        streak = nb.smooth(0.55, 0.8, nb.noise(nb.vmath("MULTIPLY", pos, (1.0, 1.0, 0.2)), 0.6, detail=3))
        col = nb.mix(nb.mul(streak, 0.25), srgb(color), srgb("#8f9192"))
        bsdf = nb.principled(Base_Color=col, Roughness=nb.madd(streak, 0.15, 0.32), Metallic=0.15,
                             Normal=nb.bump(rib, 0.6, 0.03))
    nb.output(bsdf)
    return m


def underside_material(color="#d8d8d4"):
    """The soffit of a sheeted roof: the same trapezoidal profile as the deck seen from below (ribs
    every 250 mm), the troughs a shade darker, faint condensation streaks."""
    name = "Roof_Underside"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    x = nb.sep(pos)[0]
    rib = nb.smooth(0.22, 0.42, nb.math("PINGPONG", nb.div(x, 0.25), 0.5))
    streak = nb.smooth(0.55, 0.85, nb.noise(nb.vmath("MULTIPLY", pos, (0.3, 1.0, 1.0)), 0.8, detail=3))
    col = nb.mix(nb.mul(nb.sub(1.0, rib), 0.12), srgb(color), srgb("#6f7276"))
    col = nb.mix(nb.mul(streak, 0.18), col, srgb("#8a8c86"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.62, Normal=nb.bump(rib, 0.8, 0.035)))
    return m


def translucent_material():
    name = "Roof_Polycarbonate"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.sep(nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0])
    rib = nb.smooth(0.25, 0.4, nb.math("PINGPONG", nb.div(uv[0], 0.25), 0.5))
    nb.output(nb.principled(Base_Color=srgb("#e9f1f2"), Roughness=0.22, Transmission_Weight=0.85, IOR=1.58,
                            Normal=nb.bump(rib, 0.5, 0.03)))
    return m


def build(run, tiers, bay_list, spec, d_focal, far_width, col, parent, include_ends=True):
    spec = resolve(spec)
    if not spec:
        return None
    roof = RF.solve(tiers, spec, d_focal, far_width)
    if roof["type"] == "goalpost" and roof["z_front"] > GOALPOST_MAX_HEIGHT:
        # goalpost roofs suit single low stands; over a tall stand the posts would be 30-60 m columns
        # in the front row, so such stands get a cantilever instead
        print(f"[roof] warning: {run.name}: a goalpost roof would stand over {GOALPOST_MAX_HEIGHT:.0f} m on this "
              "stand; using a cantilever", flush=True)
        spec = dict(spec, type="cantilever")
        roof = RF.solve(tiers, spec, d_focal, far_width)
    out = RF.build(run, roof, bay_list, include_ends)
    steel = MAT.painted_metal("Roof_Steel", spec.get("steel_color", "#eceeee"), rough=0.4, dirt=0.0, scuffs=0.1)
    mats = [deck_material(spec.get("color", "#e3e4e2"), spec.get("deck") == "membrane"),
            translucent_material(),
            painted("TINT_roof_fascia", spec.get("fascia_color", "#1b1d20"), 0.45),
            underside_material(spec.get("underside_color", "#d8d8d4")),
            steel]
    V, F, tags, UV = out["mesh"]
    if len(F):
        o = mesh_object(f"Roof_{run.name}", V, F, col, parent, materials=mats, mat_index=tags, uvs=UV)
        _recalc(o)
        smooth_by_angle(o)
    _tubes_mesh(f"RoofSteel_{run.name}", out["tubes"], col, parent, steel, sides=8)
    import math
    for kind, ident, loc, n, props in out["sockets"]:
        socket(kind, ident, loc, math.degrees(math.atan2(n[0], -n[1])), parent=parent, col=col, **props)
    zs = [float(np.max(V[:, 2]))] if len(V) else []
    zs += [float(np.max(np.asarray(t[0])[:, 2])) for t in out["tubes"] if len(t[0])]
    return {"roof": roof, "edge_line": out["edge_line"], "edge_top": out["edge_top"], "tail_s": out["tail_s"],
            "tail_top": out["tail_top"],
            "top": max(zs) if zs else out["tail_top"]}
