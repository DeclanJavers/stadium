"""Floodlight installations: fixtures (LED panel, metal-halide can), masts, mast heads, poles, lights.

Layout section:
  "floodlights": {"preset": "corner_pylons" | "roof_ribbon" | "grassroots_poles", "lux": 2000,
                  "on": true|false}   # "on" defaults to the scene's lighting (night => on)
"""
import math

import bpy
import numpy as np
from mathutils import Vector

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import material, srgb
from ..core.scene import collection, empty, link, mesh_object, set_params
from ..stands import floodlights as FL
from ..stands.geometry import stand_depth
from .seat import galvanised, rounded_box, side_plate, tube
from .stand import _tubes_mesh


ROOF_LIGHTS_MIN_HEIGHT = 10.0   # lowest roof edge a floodlight ribbon is mounted on (m)


def kelvin_rgb(k):
    """Approximate blackbody colour (Tanner Helland), linear-ish RGB normalised to max 1."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5 * math.log(t - 10) - 305.0)
    c = np.clip([r, g, b], 0, 255) / 255.0
    return tuple((c / c.max()) ** 2.2)


def lens(name, kelvin, strength):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    col = kelvin_rgb(kelvin)
    nb.output(nb.principled(Base_Color=srgb("#dfe3e6"), Roughness=0.15, Coat_Weight=1.0,
                            Emission_Color=col, Emission_Strength=strength))
    return m


def fixture_model(kind, col, kelvin, on):
    """One fixture; aims along local +Y. Returns a joined object (mesh shared by every instance)."""
    name = f"Fixture_{kind}"
    if bpy.data.objects.get(name):
        return bpy.data.objects[name]
    body = MAT.painted_metal("Fixture_Body", "#3a3d40", rough=0.45, dirt=0.0, scuffs=0.1)
    glass = lens("Fixture_Lens", kelvin, 40.0 if on else 0.0)
    parts = []
    if kind == "led":
        parts.append(rounded_box("housing", (0.62, 0.14, 0.46), (0, 0, 0), 0.02, col, body))
        parts.append(rounded_box("lens", (0.56, 0.01, 0.40), (0, 0.072, 0), 0.01, col, glass))
        for k in range(7):
            parts.append(rounded_box(f"fin{k}", (0.012, 0.08, 0.42), (-0.27 + k * 0.09, -0.1, 0), 0.002, col, body))
    else:
        V, F, UV = M.sweep([(0, -0.25, 0), (0, 0.22, 0)], M.profile_ellipse(0.62, 0.62, 24), (0, 0, 1))
        parts.append(mesh_object("can", V, F, col, smooth=True, materials=[body], uvs=UV))
        V, F, UV = M.sweep([(0, 0.22, 0), (0, 0.235, 0)], M.profile_ellipse(0.56, 0.56, 24), (0, 0, 1))
        parts.append(mesh_object("glass", V, F, col, smooth=True, materials=[glass], uvs=UV))
        V, F, UV = M.sweep([(0, 0.22, 0), (0, 0.42, 0)], M.profile_ellipse(0.66, 0.66, 24), (0, 0, 1), caps=False)
        visor = mesh_object("visor", V, F, col, smooth=True, materials=[body], uvs=UV)
        for v in visor.data.vertices:  # cut the visor to its top half
            if v.co.z < -0.05:
                v.co.z = -0.05
        parts.append(visor)
    for k, x in enumerate((-0.36, 0.36)):
        parts.append(side_plate(f"yoke_{k}", [(-0.05, -0.04), (0.05, -0.04), (0.05, 0.04), (-0.05, 0.04)],
                                x, 0.02, col, body, bevel=0))
    from ..core.scene import join
    obj = join(parts, name)
    return obj


def _place(model, pos, target, col, parent, name):
    o = link(bpy.data.objects.new(name, model.data), col, parent)
    o.location = Vector(pos)
    d = Vector(target) - Vector(pos)
    o.rotation_euler = d.to_track_quat("Y", "Z").to_euler()
    return o


def _light(pos, target, data, col, parent, name):
    o = link(bpy.data.objects.new(name, data), col, parent)
    o.location = Vector(pos)
    d = Vector(target) - Vector(pos)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return o


def _lattice(name, base, h, col, parent, mat):
    """Four-legged tapered lattice tower with X bracing."""
    bx, by = base
    w0, w1 = max(2.2, h / 14), 1.0
    lines = []
    segs = max(4, int(h / 4))
    zs = np.linspace(0, h, segs + 1)
    half = lambda z: (w0 + (w1 - w0) * z / h) / 2
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for cx, cy in corners:
        lines.append((np.array([(bx + cx * half(z), by + cy * half(z), z) for z in zs]), 0.09))
    for a, b in zip(zs[:-1], zs[1:]):
        for k in range(4):
            (c1x, c1y), (c2x, c2y) = corners[k], corners[(k + 1) % 4]
            p1a = (bx + c1x * half(a), by + c1y * half(a), a)
            p2b = (bx + c2x * half(b), by + c2y * half(b), b)
            p2a = (bx + c2x * half(a), by + c2y * half(a), a)
            p1b = (bx + c1x * half(b), by + c1y * half(b), b)
            lines.append((np.array([p1a, p2b]), 0.035))
            lines.append((np.array([p2a, p1b]), 0.035))
            lines.append((np.array([p1b, p2b]), 0.04))
    return _tubes_mesh(name, lines, col, parent, mat, sides=6)


def build(cfg, pitch, stands, night, col=None):
    if not cfg:
        return None
    cfg = dict(cfg)
    if "preset" in cfg:
        cfg = catalog.merge(catalog.preset("floodlights", cfg.pop("preset")), cfg)
    kind = cfg["type"]
    if kind == "roof" and stands and stands.get("roofs"):
        # a roof ribbon needs roof edges high enough to light the pitch from; under low roofs the
        # lamps would sit a few metres above the grass, so light it from corner pylons instead
        low = min(float(np.min(r["edge_line"][:, 2])) for r in stands["roofs"].values())
        if low < ROOF_LIGHTS_MIN_HEIGHT:
            print(f"[floodlights] warning: roof edges at {low:.1f} m are too low for roof floodlights; "
                  "using corner pylons", flush=True)
            kind = "corner_pylons"
            cfg = dict(cfg, type=kind, fixture="led")
    spec = dict(FL.DEFAULTS[kind], **cfg)
    on = cfg.get("on", night)
    col = col or collection("Floodlights")
    root = empty("Floodlights", col, size=2.0)
    set_params(root, f"floodlights.{kind}", "floodlights", "stadium_origin", spec)
    L, W = pitch["length"], pitch["width"]
    area = L * W
    model = fixture_model(spec["fixture"], col, spec["kelvin"], on)
    model.hide_render = True
    model["template"] = True
    model.hide_viewport = True
    steel = galvanised()
    fixtures = []          # (pos, mount)
    if kind == "corner_pylons":
        reach = 0.0
        roof_top = 0.0
        if stands:
            # how far out the structures go: facades, the stands themselves, roof back spans
            reach = max([f["line_s"] for f in stands.get("facades", {}).values()]
                        + [stand_depth(t) for _, t in stands.get("runs", {}).values()]
                        + [r["tail_s"] for r in stands.get("roofs", {}).values()] + [10.0])
            roof_top = max([r["tail_top"] for r in stands.get("roofs", {}).values()] + [0.0])
        masts = FL.pylons(L, W, stands["A"] if stands else L / 2 + 10, stands["B"] if stands else W / 2 + 8,
                          reach, roof_top, spec, corner_radius=(stands or {}).get("corner_radius"))
        for k, m in enumerate(masts):
            if spec["mast"] == "lattice":
                _lattice(f"Mast_{k}", m["base"], m["height"] - 1.5, col, root, steel)
            else:
                tube(f"Mast_{k}", [(m["base"][0], m["base"][1], 0), (m["base"][0], m["base"][1], m["height"] - 1)],
                     0.55, col, steel).parent = root
            pts, yaw, tilt = FL.head_layout(m, spec)
            lo, hi = pts.min(0), pts.max(0)
            frame = [(np.array([pts[i], pts[i + 1]]), 0.05) for i in range(len(pts) - 1) if (i + 1) % spec["cols"]]
            frame += [(np.array([pts[i], pts[i + spec['cols']]]), 0.05) for i in range(len(pts) - spec["cols"])]
            top = np.array([m["base"][0], m["base"][1], m["height"] - 1.5])
            frame += [(np.array([top, p]), 0.08) for p in (pts[0], pts[spec["cols"] - 1], pts[-1], pts[-spec["cols"]])]
            _tubes_mesh(f"MastHead_{k}", frame, col, root, steel, sides=6)
            fixtures += [(p, "mast") for p in pts]
    elif kind == "roof":
        lines = [r["edge_line"] for r in (stands or {}).get("roofs", {}).values()]
        if not lines:
            raise ValueError("roof floodlights need roofs on the stands")
        # a rail under the roof edge, braced back to the edge girder, carries the fixtures on short
        # hangers
        rail = []
        for line in lines:
            rail.append((np.asarray(line) - np.array([0, 0, 0.12]), 0.06))
            for p in FL.roof_ribbon([line], spec["spacing"]):
                r = p - np.array([0, 0, 0.12])
                out = FL.edge_normal(line, p)
                rail.append((np.array([r, r + np.array([out[0] * 0.4, out[1] * 0.4, 0.47])]), 0.04))
                rail.append((np.array([r, p - np.array([0, 0, 0.32])]), 0.035))
                fixtures.append((p - np.array([0, 0, 0.45]), "roof"))
        _tubes_mesh("RibbonRail", rail, col, root, steel, sides=6)
    elif kind == "poles":
        offset = pitch["runoff_side"] + 1.2
        for k, pole in enumerate(FL.poles(L, W, spec, offset)):
            bx, by = pole["base"]
            h = pole["height"]
            tube(f"Pole_{k}", [(bx, by, 0), (bx, by, h)], 0.16, col, steel).parent = root
            side = np.array([1.0, 0, 0])
            for j in range(spec["lamps"]):
                p = np.array([bx, by, h + 0.3]) + side * (j - (spec["lamps"] - 1) / 2) * 0.8
                fixtures.append((p - np.array([0, np.sign(by) * 0.5, 0]), "pole"))
            bar = max(1.2, 0.4 * (spec["lamps"] - 1) + 0.4)      # the crossbar spans every lamp
            tube(f"PoleBar_{k}", [(bx - bar, by, h), (bx + bar, by, h)], 0.05, col, steel).parent = root
    else:
        raise ValueError(f"unknown floodlight type {kind!r}")

    # design: try a few beam angles, re-aim each for uniformity, keep the most uniform, then set
    # lamp power so the computed average illuminance hits the target exactly
    gx, gy = np.meshgrid(np.linspace(-L / 2, L / 2, 22), np.linspace(-W / 2, W / 2, 15))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cx, cy = np.meshgrid(np.linspace(-L / 2, L / 2, 15), np.linspace(-W / 2, W / 2, 9))
    cands = np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)])
    beams = spec.get("beams", (70, 90, 120) if kind == "poles" else (30, 40, 50, 60))
    best = None
    for beam in beams:
        targets_b, E1 = FL.optimise_aims([f[0] for f in fixtures], cands, grid, beam)
        u1, u2 = FL.uniformity(E1)
        if best is None or u1 > best[2]:
            best = (beam, targets_b, u1, u2, E1)
    spot, targets, u1, u2, E1 = best
    n = len(fixtures)
    power = spec["lux"] / E1.mean()
    E = E1 * power
    light = None
    if on:
        light = bpy.data.lights.new("Floodlight", "SPOT")
        light.energy = power
        light.spot_size = math.radians(spot)
        light.spot_blend = 0.9
        light.shadow_soft_size = 0.25
        light.color = kelvin_rgb(spec["kelvin"])
    for k, ((p, mount), t) in enumerate(zip(fixtures, targets)):
        _place(model, p, t, col, root, f"Fixture_{k:03d}")
        if light:
            d = np.asarray(t) - np.asarray(p)
            _light(np.asarray(p) + 0.3 * d / np.linalg.norm(d), t, light, col, root, f"Lamp_{k:03d}")
    root["lux_avg"] = round(float(E.mean()))
    root["uniformity_u1"] = round(u1, 2)
    root["fixtures"] = n
    root["lux_target"] = spec["lux"]
    root["lamp_power_w"] = round(power, 1)
    root["beam_deg"] = round(spot, 1)
    print(f"[floodlights] {kind}: {n} fixtures, beam {spot:.0f} deg, {power:.0f} W each (Blender units), "
          f"target {spec['lux']} lux -> computed avg {E.mean():.0f} lux, U1 {u1:.2f}, U2 {u2:.2f}, "
          f"{'on' if on else 'off'}", flush=True)
    if on and stands and stands.get("roofs"):
        _bowl_lights(stands, col, root, spec["kelvin"])
    return {"root": root, "fixtures": n, "power": power, "beam": spot, "on": on, "lux_avg": float(E.mean()),
            "u1": u1, "u2": u2}


def _bowl_lights(stands, col, parent, kelvin):
    """House lights under the roof so the seating bowl reads at night (~150 lux on the seats)."""
    data = bpy.data.lights.new("BowlLight", "AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = 6.0, 1.0
    data.color = kelvin_rgb(kelvin)
    k = 0
    for run_name, rf in stands["roofs"].items():
        line = rf["edge_line"]
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(1, int(s[-1] / 12.0))
        for tt in (np.arange(n) + 0.5) * s[-1] / n:
            p = np.array([np.interp(tt, s, line[:, i]) for i in range(3)])
            outward = np.array([p[0], p[1], 0.0])
            outward /= np.linalg.norm(outward)
            q = p + outward * 14.0 - np.array([0, 0, 2.0])
            o = link(bpy.data.objects.new(f"BowlLight_{k:03d}", data), col, parent)
            o.location = Vector(q)
            o.rotation_euler = Vector((-outward[0] * 0.3, -outward[1] * 0.3, -1.0)).to_track_quat("-Z", "Y").to_euler()
            k += 1
    data.energy = 150 / 683 * (12.0 * 30.0) * 2.0  # ~150 lux over a 12 m x 30 m patch of seats
    return k
