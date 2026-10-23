"""Facade objects for a stand run (see kit/stands/facades.py for the systems)."""
import math

import bpy

from .. import catalog
from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import mesh_object
from ..core.sockets import socket
from ..stands import facades as FC
from .stand import _recalc, _tubes_mesh, concrete, painted, smooth_by_angle


def resolve(spec):
    if spec is None or spec is False:
        return None
    spec = dict(spec)
    if "preset" in spec:
        spec = catalog.merge(catalog.preset("facades", spec.pop("preset")), spec)
    return spec


def brick(name="Facade_Brick", color="#8a3b2a"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    nrm = nb.geometry()["Normal"]
    nx, ny, _ = nb.sep(nrm)
    # project bricks onto the wall: horizontal coordinate along the wall, vertical = z
    along = nb.add(nb.mul(nb.sep(pos)[0], nb.math("ABSOLUTE", ny)), nb.mul(nb.sep(pos)[1], nb.math("ABSOLUTE", nx)))
    coord = nb.comb(along, nb.sep(pos)[2], 0.0)
    br = nb.node("ShaderNodeTexBrick", offset=0.5, offset_frequency=2, squash=1.0, squash_frequency=2)
    nb.connect(coord, br.inputs["Vector"])
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.01
    br.inputs["Brick Width"].default_value = 0.225
    br.inputs["Row Height"].default_value = 0.075
    var = nb.noise(pos, 8.0, detail=3)
    c1 = srgb(color)
    br.inputs["Color1"].default_value = c1
    br.inputs["Color2"].default_value = tuple(v * 0.75 for v in c1[:3]) + (1,)
    br.inputs["Mortar"].default_value = srgb("#a39d92")
    col = nb.mix(nb.mul(var, 0.3), br.outputs["Color"], srgb("#5d4a3c"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.85,
                            Normal=nb.bump(nb.sub(1.0, br.outputs["Fac"]), 0.5, 0.006)))
    return m


def dark_glass(name="Facade_DarkGlass"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#1c2428"), Roughness=0.06, Specular_IOR_Level=0.8, Coat_Weight=0.6))
    return m


def etfe(color, glow, strength):
    name = "TINT_facade_etfe"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    # milky, slightly translucent foil (printed frit), not clear glass
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=0.3, Transmission_Weight=0.25, IOR=1.4,
                            Subsurface_Weight=0.3, Subsurface_Radius=(0.2, 0.2, 0.2), Coat_Weight=0.4,
                            Emission_Color=srgb(glow or color), Emission_Strength=strength,
                            Normal=nb.bump(nb.noise(pos, 2.0), 0.05, 0.01)))
    return m


def materials(spec, night):
    kind = spec["type"]
    color = spec.get("color", "#b8b3a8")
    accent = spec.get("accent", color)
    trim = MAT.painted_metal("Facade_Trim", "#d9dad6", rough=0.4, dirt=0.2, scuffs=0.0)
    if kind == "brick":
        main, trim = brick(color=color), concrete("Facade_Stone", "#d6ccb9")
    elif kind == "concrete_frame":
        main = concrete("Facade_Concrete", color)
    else:
        main = painted("TINT_facade", color, 0.35)
    acc = painted("TINT_facade_b", accent, 0.35)
    glow = etfe(color, spec.get("glow"), 1.6 if night and spec.get("glow") else 0.0)
    return [main, acc, dark_glass(), trim, glow, concrete("Facade_Core", "#bdb8ad")]


def build(run, tiers, bay_list, spec, roof_info, ends, col, parent, night=False):
    spec = resolve(spec)
    if not spec:
        return None
    out = FC.build(run, tiers, bay_list, spec, roof_info, ends)
    V, F, tags, UV = out["mesh"]
    o = mesh_object(f"Facade_{run.name}", V, F, col, parent, materials=materials(spec, night),
                    mat_index=tags, uvs=UV)
    _recalc(o)
    smooth_by_angle(o)
    if out["tubes"]:
        _tubes_mesh(f"FacadeFrame_{run.name}", out["tubes"], col, parent,
                    MAT.painted_metal("Facade_Frame", "#f1f2f0", rough=0.35, dirt=0.0, scuffs=0.0), sides=6)
    for kind, ident, loc, n, props in out["sockets"]:
        socket(kind, ident, loc, math.degrees(math.atan2(n[0], -n[1])), parent=parent, col=col, **props)
    return {"line_s": out["line_s"], "height": out["height"]}
