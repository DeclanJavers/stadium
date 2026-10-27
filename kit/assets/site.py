"""Training grounds in Blender: realise a kit.site Site (plain-Python generator output) as a scene.

Parts become mesh objects with library materials (recoloured copies where a part gives a colour,
lawn materials from the pitch grass shader, mesh cards for fences and nets); each pitch is built by
the pitch generator with its goals and corner flags; props are built once per asset and placed as
linked duplicates, or (game export) as one node per asset carrying EXT_mesh_gpu_instancing arrays.

    root, info = realize(site, col, instanced=True)
"""
import hashlib
import math

import bpy
import numpy as np

from ..core import mesh as M
from ..core.scene import collection, empty, mesh_object
from ..core.sockets import attach, sockets_of
from . import corner_flag, goal, library, pitch, props


def _lin(hexs):
    c = np.array([int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)]) if isinstance(hexs, str) else np.asarray(hexs, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex(rgb):
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


class Materials:
    """Blender materials for the generator's material references (built once each)."""

    def __init__(self):
        self.cache = {}
        self.tints = {}          # material name -> (base key, sRGB hex) to apply after the game bake
        self.lawns = {}          # lawn material name -> tileable size
        self.cells = {}          # mesh-card material name -> metres per cell (UVs are metres)

    def get(self, m):
        tex = m.texture
        if tex and tex[0] == "grass":
            return self._lawn(tex[1])
        if tex and tex[0] == "net":
            s = tex[1]
            c = s["colors"][0]
            name = f"Net_{s['mesh']}_{s['mesh_size_mm']}_{s['cord_tenths_mm']}_{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            if name not in self.cache:
                cell = s["mesh_size_mm"] / 1000.0
                self.cache[name] = props.mesh_image(name, cell, s["cord_tenths_mm"] / 10000.0, color="#%02x%02x%02x" % tuple(c))
                self.cells[name] = cell
            return self.cache[name]
        base = library.get(m.key)
        if base is None:
            base = props.mats()["grey"]
        if m.color is None and m.emission is None:
            return base
        hexs = _hex(m.color) if m.color is not None else "#ffffff"
        name = f"{m.key}#{hexs[1:]}" + ("+glow" if m.emission else "")
        if name not in self.cache:
            mat = base.copy()
            mat.name = name
            if m.emission:
                p = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
                if p is not None:
                    p.inputs["Emission Color"].default_value = (*_lin(m.emission[0]), 1.0)
                    p.inputs["Emission Strength"].default_value = m.emission[1]
            self.cache[name] = mat
            self.tints[name] = (m.key, hexs)
        return self.cache[name]

    def _lawn(self, spec):
        key = hashlib.md5(repr(sorted((k, str(v)) for k, v in spec.items() if k not in ("origin", "yaw"))).encode()).hexdigest()[:8]
        name = f"Lawn_{key}"
        if name in self.cache:
            return self.cache[name]
        params = pitch.resolve({"grass": "perennial_ryegrass" if spec["natural"] else "artificial_3g", "height_mm": spec["height_mm"],
                                "wear": spec["wear"], "mow": {"pattern": spec["pattern"], "width": spec["stripe_width"], "angle": 0.0, "offset": 0.0},
                                "length": 60.0, "width": 40.0})
        mat = pitch.grass_material(params, pitch.surface_group(params))
        mat.name = name
        self.cache[name] = mat
        self.lawns[name] = (8.0, False)
        return mat

    def apply_tints(self):
        """After the game bake: recolour each tinted copy relative to its library colour."""
        for name, (key, hexs) in self.tints.items():
            mat = bpy.data.materials.get(name)
            p = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None) if mat else None
            if p is None:
                continue
            want = _lin(hexs)
            inp = p.inputs["Base Color"]
            if inp.is_linked:
                src = inp.links[0].from_node
                if src.bl_idname == "ShaderNodeMix":            # TINT_: detail x colour factor
                    gain = mat.get("tint_gain", 1.0)
                    src.inputs[7].default_value = (*(want * gain), 1.0)
                    mat["tint_rgb"] = list(want)
                else:                                            # textured: scale by the colour ratio
                    bh = library.base_hex(key)
                    ratio = want / np.maximum(_lin(bh), 1e-4) if bh else np.ones(3)
                    mix = mat.node_tree.nodes.new("ShaderNodeMix")
                    mix.data_type = "RGBA"
                    mix.blend_type = "MULTIPLY"
                    mix.inputs[0].default_value = 1.0
                    mat.node_tree.links.new(src.outputs[0], mix.inputs[6])
                    mix.inputs[7].default_value = (*np.clip(ratio, 0, 4), 1.0)
                    mat.node_tree.links.new(mix.outputs[2], inp)
            else:
                inp.default_value = (*want, 1.0)


def _part_object(part, mats, col):
    if part.tubes:
        p0, p1, radii = [], [], []
        for pts, r in part.tubes:
            for a, b in zip(pts[:-1], pts[1:]):
                p0.append(a)
                p1.append(b)
                radii.append(r)
        V, F = [], []
        for a, b, r in zip(p0, p1, radii):
            v, q, _keep = M.tubes(np.array([a]), np.array([b]), r, max(part.sides, 3), extend=r * 0.5)
            base = len(V)
            V.extend(v.tolist())
            F.extend((np.asarray(q) + base).tolist())
        if not F:
            return None
        o = mesh_object(part.name, V, F, col, materials=[mats.get(part.mats[0])], smooth=True)
        return o
    m = part.mesh
    if not m.faces:
        return None
    bl = [mats.get(x) for x in part.mats]
    idx = [min(max(t, 0), len(bl) - 1) for t in m.tags]
    uvs = []
    k = 0
    for f, mi in zip(m.faces, idx):
        cell = mats.cells.get(bl[mi].name)
        for _ in f:
            u, v = m.uvs[k]
            uvs.append((u / cell, v / cell) if cell else (u, v))
            k += 1
    o = mesh_object(part.name, m.verts, m.faces, col, materials=bl, mat_index=idx, uvs=uvs)
    if part.smooth_deg:
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(part.smooth_deg))
    return o


def realize(site, col=None, instanced=False):
    """Build the site in the current scene. instanced: props as one node per asset and recolour
    (EXT_mesh_gpu_instancing data returned in info["instances"]) rather than linked duplicates."""
    col = col or collection("Site")
    root = empty("Site", col)
    mats = Materials()
    for part in site.parts:
        o = _part_object(part, mats, col)
        if o is not None:
            o.parent = root
    # pitches, each with its own turf, lines, goals and flags
    pitch_mats = {}
    for i, sp in enumerate(site.pitches):
        pc = collection(f"Pitch_{sp.plot}")
        proot, params, _lines = pitch.build(sp.params, pc)
        for o in pc.objects:
            for slot in o.material_slots:
                m = slot.material
                if m and m.name.startswith("Pitch_Grass") and not m.name.startswith("Pitch_Grass:"):
                    m.name = f"Pitch_Grass:{sp.plot}"
                    pitch_mats[m.name] = params
                elif m and m.name.startswith("TINT_Pitch_Paint") and ":" not in m.name:
                    m.name = f"TINT_Pitch_Paint:{sp.plot}"
        if sp.goals:
            built = None
            for sock in sockets_of(proot, "goal"):
                if not sp.both_goals and sock["socket_id"] == "west":
                    continue
                g = goal.build({"style": sp.goals}, collection(f"Goal_{sp.plot}_{sock['socket_id']}"), template=built)
                built = built or g
                attach(g, sock)
        if sp.flags:
            tpl = None
            for sock in sockets_of(proot, "corner_flag"):
                f = corner_flag.build({"preset": sp.flags}, collection(f"Flag_{sp.plot}_{sock['socket_id']}"), template=tpl)
                tpl = tpl or f
                attach(f, sock)
        proot.location = (sp.origin[0], sp.origin[1], 0.0)
        proot.rotation_euler = (0.0, 0.0, sp.yaw)
        proot.parent = root
    # props
    pcol = collection("SiteProps")
    protos = {}
    instances = {}
    lods = {}
    for ps in site.props:
        if ps.key not in protos:
            try:
                r = props.build(ps.key, collection(f"Proto_{ps.key}"))
            except KeyError:
                continue
            lod = None
            if f"{ps.key}.lod2" in props.keys():
                lod = props.build(f"{ps.key}.lod2", collection(f"Proto_{ps.key}.lod2"))
            protos[ps.key] = (r, lod)
        r, lod = protos[ps.key]
        tint = ps.tints[0][1] if ps.tints else None
        tag = f"#{_hex(tint)[1:]}" if tint else ""
        parts = [c for c in r.children if c.type == "MESH"]
        lparts = [c for c in lod.children if c.type == "MESH"] if lod else []
        for k, part in enumerate(parts):
            name = f"PROPI:{ps.key}{tag}:{k}"
            if instanced:
                o = bpy.data.objects.new(name, part.data)
                pcol.objects.link(o)
                o.parent = root
                g = instances.setdefault(name, {"pos": [], "yaw": [], "tilt": [], "scale": [], "color": [] if tint else None})
                for it in ps.items:
                    g["pos"].append(it.pos)
                    g["yaw"].append(it.yaw)
                    g["tilt"].append(it.tilt)
                    g["scale"].append(it.scale)
                    if tint:
                        g["color"].append(_lin(tint))
                if k < len(lparts):
                    lo = bpy.data.objects.new(name + ":LOD2", lparts[k].data)
                    pcol.objects.link(lo)
                    lo.parent = root
                    instances[name + ":LOD2"] = g
                    lods[name] = name + ":LOD2"
            else:
                for j, it in enumerate(ps.items):
                    o = bpy.data.objects.new(f"{name}:{j}", part.data)
                    pcol.objects.link(o)
                    o.location = it.pos
                    o.rotation_mode = "ZXY"
                    o.rotation_euler = (it.tilt, 0.0, it.yaw)
                    o.scale = it.scale
                    o.parent = root
    for r, lod in protos.values():
        for ob in [r, *r.children_recursive] + ([lod, *lod.children_recursive] if lod else []):
            bpy.data.objects.remove(ob)
    # lights (night) and sockets
    for i, s in enumerate(site.spots):
        L = bpy.data.lights.new(f"SiteSpot{i}", "SPOT")
        L.energy = s.power
        L.spot_size = math.radians(s.angle_deg)
        L.spot_blend = s.blend
        L.color = s.color
        lo = bpy.data.objects.new(f"SiteSpot{i}", L)
        col.objects.link(lo)
        lo.location = s.pos
        d = np.asarray(s.dir)
        lo.rotation_mode = "QUATERNION"
        from mathutils import Vector
        lo.rotation_quaternion = Vector(d).to_track_quat("-Z", "Y")
        lo.parent = root
    for s in site.sockets:
        e = empty(f"SOCKET:{s.kind}:{s.id}", col)
        e.location = s.loc
        e.rotation_euler = (0.0, 0.0, math.atan2(-s.n[0], s.n[1]))
        e["socket_kind"] = s.kind
        e["socket_id"] = s.id
        for k, v in s.props:
            e[k] = v
        e.parent = root
    return root, {"materials": mats, "pitches": pitch_mats, "instances": instances, "lods": lods}
