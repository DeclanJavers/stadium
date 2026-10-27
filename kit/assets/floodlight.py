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


def _ccw_yz(poly):
    P = np.asarray(poly, float)
    a = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return [tuple(q) for q in (P if a > 0 else P[::-1])]


def _faces_facing(p, quads, mat, point, outward=True):
    """Add polygons, each wound so its normal points away from (or towards) `point`."""
    c0 = np.asarray(point, float)
    for q in quads:
        Q = np.asarray(q, float)
        n = np.zeros(3)
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]
            n += np.cross(a, b)
        d = Q.mean(0) - c0
        if (n @ d > 0) != outward:
            Q = Q[::-1]
        p.face([tuple(x) for x in Q], mat)


def _loft_solid(p, rings, mat, front_cap=True):
    """Skin closed rings into a convex-ish solid with outward faces and end caps."""
    R = [np.asarray(r, float) for r in rings]
    c = np.mean(np.concatenate(R), 0)
    m = len(R[0])
    quads = [[R[i][j], R[i + 1][j], R[i + 1][(j + 1) % m], R[i][(j + 1) % m]] for i in range(len(R) - 1) for j in range(m)]
    quads += ([R[0]] if front_cap else []) + [R[-1]]
    _faces_facing(p, quads, mat, c)


def _led_panel(kelvin, rows=7, cols=16, cell=64):
    """LED optics behind the front glass: modules of clear TIR collimators (faceted rings that
    catch the light, a crescent glint) each over a bright phosphor die; and the dies alone
    (emission). Linear RGB arrays, row 0 the bottom."""
    H, W = rows * cell, cols * cell
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx % cell + 0.5) / cell - 0.5
    v = (yy % cell + 0.5) / cell - 0.5
    r = np.sqrt(u * u + v * v)
    base = np.full((H, W, 3), 0.02)
    seam = ((xx // cell) % 4 == 0) & (xx % cell < 3)
    base[seam] = 0.07
    body = (r < 0.4).astype(float)
    rings = 0.5 + 0.5 * np.cos(r * 2 * np.pi * 6)
    lens = np.array([0.05, 0.055, 0.06])[None, None] + (rings * np.clip(r / 0.4, 0, 1) * 0.22)[..., None]
    glint = np.clip(1 - np.hypot(u + 0.12, v - 0.14) / 0.09, 0, 1) ** 2 * 0.6
    lens = lens + glint[..., None]
    rim = np.clip(1 - np.abs(r - 0.41) / 0.025, 0, 1) * 0.35
    col = base * (1 - body[..., None]) + lens * body[..., None] + rim[..., None]
    die = ((np.abs(u) < 0.07) & (np.abs(v) < 0.07)).astype(float)
    halo = np.clip(1 - r / 0.16, 0, 1) * 0.25
    phos = np.array([0.92, 0.82, 0.45])
    col = col + (halo * body)[..., None] * phos
    col = col * (1 - die[..., None]) + phos * die[..., None]
    emit = np.array(kelvin_rgb(kelvin)) * np.clip(die + 0.3 * halo / 0.25 * body, 0, 1)[..., None]
    # the cover glass: a broad sky sheen across it, and grime streaks running down from the visor
    gu, gv = xx / W, yy / H
    sheen = np.clip(1 - np.abs(gu * 0.8 - gv + 0.25) / 0.22, 0, 1) ** 2 * 0.12
    col = col + sheen[..., None] * np.array([0.8, 0.85, 0.9])
    from . import kit_detail as KD
    streak = (KD.pnoise(H, W, 1, 60, seed=9) > 0.62) * np.clip((gv - 0.55) / 0.45, 0, 1) * 0.35
    col = col * (1 - streak[..., None]) + np.array([0.12, 0.11, 0.09]) * streak[..., None]
    return np.clip(col, 0, 1), np.clip(emit, 0, 1)


def _halide_reflector(kelvin, w=320, h=256):
    """A metal-halide floodlight's asymmetric polished reflector, seen through the glass: a deep
    trough of 6 x 4 mirror facets, each catching a different, streaky slice of sky and ground (the
    viewer has no reflections, so they are painted in), dark seams between them."""
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx + 0.5) / w
    v = (yy + 0.5) / h
    fi = np.floor(u * 6).astype(int)
    fj = np.floor((v ** 0.8) * 4).astype(int)
    rng = np.random.default_rng(5)
    lev = rng.uniform(0.25, 1.0, (7, 5))
    tilt = rng.uniform(-1, 1, (7, 5))
    lu = u * 6 - fi
    lv = (v ** 0.8) * 4 - fj
    val = lev[fi, fj] * (0.55 + 0.45 * np.clip(0.5 + tilt[fi, fj] * (lu - 0.5) + 0.4 * (lv - 0.5), 0, 1))
    streak = 0.5 + 0.5 * np.sin((u * 31 + v * 7) * np.pi + fi * 1.7) * np.sin(v * 13 + fj)
    val = val * (0.75 + 0.35 * streak)
    sky = np.array([0.82, 0.88, 0.95])
    ground = np.array([0.42, 0.45, 0.36])
    mixv = np.clip(v * 1.3 - 0.15 + tilt[fi, fj] * 0.2, 0, 1)[..., None]
    col = val[..., None] * (ground * (1 - mixv) + sky * mixv)
    # polished sheet: bright horizontal highlight bands sweep across all facets (the trough's
    # curvature), so the facet edges read as folds, not tiles
    band = np.exp(-((v - 0.72) / 0.05) ** 2) * 0.9 + np.exp(-((v - 0.3) / 0.035) ** 2) * 0.6
    col = col * 0.8 + band[..., None] * np.array([0.95, 0.96, 0.98])
    seam = (lu < 0.012) | (lu > 0.988) | (lv < 0.015) | (lv > 0.985)
    col[seam] *= 0.6
    deep = np.clip(1 - np.hypot((u - 0.5) * 1.6, (v - 0.42) * 2.2), 0, 1)
    col = col * (1 - 0.45 * deep[..., None])
    emit = col * 0.6
    return np.clip(col, 0, 1), np.clip(emit, 0, 1)


def _label_image():
    from . import kit_detail as KD
    n_w, n_h = 256, 128
    img = np.full((n_h, n_w), 0.0)
    for k, text in enumerate(("IP66 2000W", "400V 50HZ", "MVF 403")):
        m = KD.lettering(text, 22)
        mh, mw = m.shape
        y0 = n_h - 14 - (k + 1) * 34
        x0 = 10
        img[max(y0, 0):max(y0, 0) + mh, x0:x0 + min(mw, n_w - x0)] = np.maximum(
            img[max(y0, 0):max(y0, 0) + mh, x0:x0 + min(mw, n_w - x0)], m[:, :n_w - x0])
    bg, fg = KD.to_lin("#d9d6cc"), KD.to_lin("#202020")
    return bg + (fg - bg) * img[..., None]


def _scale_image(n=256):
    """Aiming protractor: tick marks every 5 degrees round a half disc, on dull aluminium."""
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx + 0.5) / n * 2 - 1, (yy + 0.5) / n * 2 - 1
    r = np.sqrt(x * x + y * y)
    a = np.degrees(np.arctan2(y, x))
    col = np.full((n, n, 3), 0.42)
    tick = (np.abs(((a + 2.5) % 5) - 2.5) < 0.45) & (r > 0.78) & (r < 0.94)
    major = (np.abs(((a + 7.5) % 15) - 7.5) < 0.6) & (r > 0.64) & (r < 0.94)
    col[tick | major] = 0.05
    col[(r > 0.97)] = 0.3
    return col


# height of a fixture's trunnion axis (the housing centre) above its origin, the yoke's mounting
# plate: a mount that wants the old centred placement offsets the fixture by this along its local -Z
FIXTURE_PIVOT = {"led": 0.30, "halide": 0.33}


def fixture_model(kind, col, kelvin, on):
    """One sports floodlight on its U-shaped yoke; aims along local +Y. Origin: the yoke's mounting
    plate (bolted to a headframe, crossarm or hanger), the trunnion axis 0.3 m above it.
    led: Signify/Musco-type LED floodlight: die-cast housing with fine vertical cooling fins, a
    recessed lens panel of LED optics in modules, an asymmetric glare visor, driver box and gland.
    halide: metal-halide box luminaire (ArenaVision-type): tempered glass front in a gasketed
    bezel, faceted specular reflector and lamp capsule behind it, tapered finned rear housing,
    short visor. Both carry an aiming protractor at the trunnion. Returns a joined object."""
    name = f"Fixture_{kind}"
    if bpy.data.objects.get(name):
        return bpy.data.objects[name]
    from . import kit_detail as KD
    from .props import PM
    from .seat import flat
    body = flat("Fixture_Body", "#6e7073", rough=0.55, metallic=0.15)
    dark = flat("Fixture_Gasket", "#121314", rough=0.7)
    yoke_m = flat("Fixture_Yoke", "#5d6063", rough=0.55, metallic=0.25)
    bolt_m = flat("Fixture_Bolt", "#8d9194", rough=0.4, metallic=0.6)
    scale = KD.image_material("Fixture_Scale", albedo=KD.image("Fixture_Scale_img", _scale_image()),
                              rough=0.45, metallic=0.5, extension="EXTEND")
    p = PM(0.01)
    zc = FIXTURE_PIVOT[kind]
    hw, hh = (0.30, 0.22) if kind == "led" else (0.30, 0.25)
    glass_m = flat("Fixture_Lens_Glass", "#c3cccf", rough=0.05)
    glass_m.node_tree.nodes["Principled BSDF"].inputs["Transmission Weight"].default_value = 1.0
    screw = flat("Fixture_Screw", "#b4b8ba", rough=0.35, metallic=0.8)
    if kind == "led":
        alb, emi = _led_panel(kelvin)
        lens_m = KD.image_material(f"Fixture_Lens_LED{'_on' if on else ''}", albedo=KD.image("Fixture_LED_img", alb),
                                   rough=0.12, extension="EXTEND",
                                   emission=(KD.image("Fixture_LED_emit", emi), 40.0 if on else 0.35))
        p.r = 0.014
        p.box((-hw, -0.05, zc - hh), (hw, 0.045, zc + hh), body)                 # die-cast body
        p.r = 0.004
        bz = 0.022                                                               # bezel round the lens
        p.box((-hw, 0.045, zc + hh - bz), (hw, 0.07, zc + hh), body)
        p.box((-hw, 0.045, zc - hh), (hw, 0.07, zc - hh + bz), body)
        p.box((-hw, 0.045, zc - hh + bz), (-hw + bz, 0.07, zc + hh - bz), body)
        p.box((hw - bz, 0.045, zc - hh + bz), (hw, 0.07, zc + hh - bz), body)
        p.r = 0.0
        gw, gh = hw - bz, hh - bz
        p.decal((0, 0.046, zc), (0, 1, 0), (0, 0, 1), 2 * gw, 2 * gh, lens_m, lift=0.0)
        p.quad((-gw, 0.052, zc - gh), (gw, 0.052, zc - gh), (gw, 0.052, zc + gh), (-gw, 0.052, zc + gh), glass_m)
        for z in (zc - gh, zc + gh):                                             # gasket line
            p.box((-gw, 0.046, z - 0.003), (gw, 0.0535, z + 0.003), dark)
        for x in (-gw, gw):
            p.box((x - 0.003, 0.046, zc - gh), (x + 0.003, 0.0535, zc + gh), dark)
        p.r = 0.0
        for k in range(6):                                                       # bezel screws
            for z in (zc - hh + bz / 2, zc + hh - bz / 2):
                x = -hw + 0.05 + k * (2 * hw - 0.1) / 5
                p.cyl((x, 0.07, z), (x, 0.073, z), 0.0045, screw, sides=8)
        for x in (-hw + bz / 2, hw - bz / 2):
            for z in (zc - 0.08, zc + 0.08):
                p.cyl((x, 0.07, z), (x, 0.073, z), 0.0045, screw, sides=8)
        # glare visor: formed aluminium shield, returned front lip, side cheeks, two bolted brackets
        p.r = 0.003
        vz = zc + hh
        L = 0.17
        p.obox((0, 0.07 + L / 2 * math.cos(math.radians(12)), vz + 0.012 - L / 2 * math.sin(math.radians(12))),
               (2 * hw + 0.012, L, 0.005), body, pitch=math.radians(-12))
        fy, fz = 0.07 + L * math.cos(math.radians(12)), vz + 0.012 - L * math.sin(math.radians(12))
        p.box((-hw - 0.006, fy - 0.004, fz - 0.03), (hw + 0.006, fy + 0.002, fz + 0.002), body)
        for side in (-1, 1):
            x = side * (hw + 0.003)
            p.extrude_yz(_ccw_yz([(0.045, vz - 0.07), (fy, fz - 0.03), (fy, fz + 0.003), (0.045, vz + 0.013)]),
                         x - 0.003, x + 0.003, body)
        for x in (-hw * 0.6, hw * 0.6):
            p.box((x - 0.015, -0.03, vz - 0.002), (x + 0.015, 0.12, vz + 0.008), body)
        p.r = 0.0
        for k, yl in enumerate((0.1, 0.14, 0.18)):                               # glare louvres
            zl = vz - 0.015 - k * 0.012
            p.obox((0, yl, zl), (2 * hw, 0.035, 0.0025), body, pitch=math.radians(-35))
        cable = flat("Fixture_Cable", "#b9bcbe", rough=0.35, metallic=0.9)    # stainless safety cable
        p.tube([(hw * 0.8, -0.02, 0.05), (hw * 0.85, -0.09, zc), (hw * 0.75, -0.06, zc + hh + 0.01)], 0.0022, cable, sides=5)
        # rear: deep die-cast fins across the whole back (50 mm, 12 mm pitch) and down the sides
        p.r = 0.0                          # thin fins: no bevels (they were most of the triangles)
        nf = int((2 * hw - 0.03) / 0.012) + 1
        for k in range(nf):
            x = -hw + 0.015 + k * 0.012
            p.box((x - 0.0018, -0.1, zc - hh + 0.012), (x + 0.0018, -0.05, zc + hh - 0.012), body)
        for side in (-1, 1):
            for k in range(7):
                z = zc - hh + 0.05 + k * (2 * hh - 0.1) / 6
                p.box((side * hw - (0.022 if side > 0 else 0.0), -0.045, z - 0.0018),
                      (side * hw + (0.022 if side < 0 else 0.0), 0.03, z + 0.0018), body)
        p.r = 0.006
        p.box((-0.13, -0.155, zc - 0.08), (0.13, -0.1, zc + 0.08), body)
        p.r = 0.002
        p.cyl((0.08, -0.13, zc - 0.08), (0.08, -0.13, zc - 0.115), 0.012, dark, sides=10)
        p.cyl((0.08, -0.13, zc - 0.115), (0.08, -0.13, zc - 0.2), 0.006, dark, sides=8)
        label = KD.image_material("Fixture_Label", albedo=KD.image("Fixture_Label_img", _label_image()), rough=0.4,
                                  extension="EXTEND")
        p.decal((-0.03, -0.1555, zc + 0.02), (0, -1, 0), (0, 0, 1), 0.1, 0.05, label, lift=0.0008)
    else:
        alb, emi = _halide_reflector(kelvin)
        refl = KD.image_material(f"Fixture_Lens_Halide{'_on' if on else ''}", albedo=KD.image("Fixture_Halide_img", alb),
                                 rough=0.12, metallic=0.3, extension="EXTEND",
                                 emission=(KD.image("Fixture_Halide_emit", emi), 6.0 if on else 0.55))
        lamp = flat("Fixture_LampQuartz", "#ece6d6", rough=0.1)
        ceramic = flat("Fixture_Ceramic", "#e4e1d8", rough=0.6)
        green = flat("Fixture_GlassEdge", "#5f7f74", rough=0.1)
        # tapered rear housing (die-cast), lofted from the bezel back to the gear tray
        ring = lambda w, h, y, z0=zc: [(x, y, z0 + z) for x, z in
                                         [(w, -h), (w, h), (-w, h), (-w, -h)]]
        p.r = 0.012
        _loft_solid(p, [ring(hw, hh, 0.04), ring(hw - 0.02, hh - 0.02, -0.06), ring(hw - 0.08, hh - 0.07, -0.25),
                        ring(hw - 0.1, hh - 0.09, -0.29)], body, front_cap=False)
        # gasketed bezel, the reflector behind the glass, the lamp, toggle clips
        p.r = 0.004
        bz = 0.03
        p.box((-hw - 0.01, 0.04, zc + hh - bz), (hw + 0.01, 0.075, zc + hh + 0.01), body)
        p.box((-hw - 0.01, 0.04, zc - hh - 0.01), (hw + 0.01, 0.075, zc - hh + bz), body)
        p.box((-hw - 0.01, 0.04, zc - hh + bz), (-hw + bz, 0.075, zc + hh - bz), body)
        p.box((hw - bz, 0.04, zc - hh + bz), (hw + 0.01, 0.075, zc + hh - bz), body)
        p.r = 0.0
        gw, gh = hw - bz, hh - bz
        # the reflector: a deep asymmetric trough (130 mm) of flat polished facets, 6 strips x 6
        def trough(x, z):
            a, b = x / gw, (z - zc + 0.02) / gh
            return 0.04 - 0.13 * (1 - np.clip(b * (0.85 if b > 0 else 1.15), -1, 1) ** 2) * (1 - 0.35 * a * a)
        nxr, nzr = 6, 6
        for i in range(nxr):
            for j in range(nzr):
                xs4 = [-gw + 2 * gw * (i + di) / nxr for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))]
                zs4 = [zc - gh + 2 * gh * (j + dj) / nzr for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))]
                pts = [(x, trough(x, z), z) for x, z in zip(xs4, zs4)]
                uv = [((x + gw) / (2 * gw), (z - zc + gh) / (2 * gh)) for x, z in zip(xs4, zs4)]
                p.face(pts[::-1], refl, uv[::-1])
        ly = trough(0.0, zc - 0.02) + 0.06
        p.cyl((-0.065, ly, zc - 0.02), (0.065, ly, zc - 0.02), 0.012, lamp, sides=12)        # quartz arc tube
        for sx in (-1, 1):
            p.cyl((sx * 0.065, ly, zc - 0.02), (sx * 0.085, ly, zc - 0.02), 0.014, ceramic, sides=12)
            p.cyl((sx * 0.085, ly, zc - 0.02), (sx * 0.105, ly, zc - 0.02), 0.019, ceramic, sides=12)   # lamp holder
            yb = trough(sx * 0.095, zc - 0.02)
            p.box((sx * 0.095 - 0.01, yb - 0.005, zc - 0.035), (sx * 0.095 + 0.01, ly, zc - 0.005), ceramic)
        p.quad((-gw, 0.068, zc - gh), (gw, 0.068, zc - gh), (gw, 0.068, zc + gh), (-gw, 0.068, zc + gh), glass_m)
        for z in (zc - gh, zc + gh):
            p.box((-gw, 0.0675, z - 0.01), (gw, 0.0695, z + 0.002 if z < zc else z - 0.002), green)
            p.box((-gw, 0.066, z - 0.005), (gw, 0.072, z + 0.005), dark)
        for x in (-gw, gw):
            p.box((x - 0.005, 0.066, zc - gh), (x + 0.005, 0.072, zc + gh), dark)
        p.r = 0.002
        for x, z in ((-0.15, zc + hh), (0.15, zc + hh), (-0.15, zc - hh), (0.15, zc - hh), (-hw, zc), (hw, zc)):
            vert = abs(x) < hw - 0.01
            if vert:
                p.box((x - 0.012, 0.055, z - 0.022), (x + 0.012, 0.081, z + 0.022), screw)
            else:
                p.box((x - 0.022, 0.055, z - 0.012), (x + 0.022, 0.081, z + 0.012), screw)
        # cooling fins along the top and back, a visor with returns and rivets, gland, label
        p.r = 0.0
        top0, top1 = zc + hh - 0.02, zc + hh - 0.07          # the tapered top, front to back
        for k in range(9):
            x = -0.2 + k * 0.05
            p.extrude_yz([(-0.25, top1 - 0.01), (-0.06, top0 - 0.01), (-0.06, top0 + 0.022), (-0.25, top1 + 0.022)],
                         x - 0.003, x + 0.003, body)
            p.box((x - 0.003, -0.31, zc - 0.12), (x + 0.003, -0.27, zc + 0.12), body)
        p.r = 0.003
        p.obox((0, 0.13, zc + hh + 0.004), (2 * hw + 0.02, 0.13, 0.006), body, pitch=math.radians(-8))
        vy, vzf = 0.13 + 0.065 * math.cos(math.radians(8)), zc + hh + 0.004 - 0.065 * math.sin(math.radians(8))
        p.box((-hw - 0.01, vy - 0.005, vzf - 0.025), (hw + 0.01, vy + 0.001, vzf + 0.002), body)        # front return
        for side in (-1, 1):
            p.box((side * (hw + 0.01) - 0.003, 0.07, zc + hh - 0.03), (side * (hw + 0.01) + 0.003, vy, zc + hh + 0.01), body)
        p.r = 0.002
        p.cyl((0.12, -0.2, zc - hh + 0.06), (0.12, -0.2, zc - hh - 0.02), 0.013, dark, sides=10)
        p.cyl((0.12, -0.2, zc - hh - 0.02), (0.12, -0.2, zc - hh - 0.09), 0.007, dark, sides=8)
        label = KD.image_material("Fixture_Label", albedo=KD.image("Fixture_Label_img", _label_image()), rough=0.4,
                                  extension="EXTEND")
        p.decal((0.0, -0.2905, zc + 0.15), (0, -1, 0), (0, 0, 1), 0.1, 0.05, label, lift=0.0008)
    # yoke: trunnion bosses, two flat-bar arms, a base bar and the mounting plate
    xa = hw + 0.03
    p.r = 0.004
    for side in (-1, 1):
        p.cyl((side * hw, 0, zc), (side * (xa + 0.012), 0, zc), 0.042, body, sides=16)
        p.box((side * xa - 0.008, -0.04, 0.035), (side * xa + 0.008, 0.04, zc + 0.03), yoke_m)
        p.cyl((side * (xa + 0.007), 0, zc), (side * (xa + 0.02), 0, zc), 0.018, bolt_m, sides=6)
    p.box((-xa - 0.008, -0.04, 0.02), (xa + 0.008, 0.04, 0.05), yoke_m)
    p.r = 0.003
    p.box((-0.1, -0.1, 0.0), (0.1, 0.1, 0.02), yoke_m)                              # 200 x 200 plate
    p.r = 0.0
    for x in (-0.075, 0.075):
        for y in (-0.075, 0.075):
            p.cyl((x, y, 0.02), (x, y, 0.03), 0.011, bolt_m, sides=6)
    for x in (-0.055, 0.055):
        for y in (-0.055, 0.055):
            p.cyl((x, y, 0.05), (x, y, 0.058), 0.009, bolt_m, sides=6)
    p.r = 0.0
    for x in (-0.2, 0.2):                                                        # slotted mounting holes
        p.box((x - 0.025, -0.008, 0.0495), (x + 0.025, 0.008, 0.0505), dark)
    # aiming protractor on the right-hand arm
    p.r = 0.0
    p.cyl((xa + 0.007, 0, zc), (xa + 0.011, 0, zc), 0.075, yoke_m, sides=20)
    p.decal((xa + 0.011, 0, zc), (1, 0, 0), (0, 0, 1), 0.104, 0.104, scale, lift=0.0008)
    obj = p.obj(name, col)
    KD.finish(obj, distance=0.25, ground=False)

    def weather(P, N):
        up = np.clip(N[:, 2], 0, 1)
        out = 1 - (up ** 2 * 0.16)[:, None] * (1 - np.array([0.85, 0.8, 0.7]))       # grime on top faces
        behind = P[:, 1] < -0.05
        streak = behind & (np.sin(P[:, 0] / 0.012 * np.pi) > 0.6) & (P[:, 2] < zc)
        out[streak] *= 0.88                                                             # streaks below the fins
        return out
    KD.tint_vertices(obj, weather, materials={"Fixture_Body"})
    return obj


def fixture_lod2(kind, col, kelvin, on):
    """The coarse level of a fixture (about 600-900 triangles) for 40 m and beyond: the same
    silhouette, origin and materials (the front optics keep their image), the fins as a few
    plates, no screws, bevels or gaskets."""
    name = f"Fixture_{kind}_LOD2"
    if bpy.data.objects.get(name):
        return bpy.data.objects[name]
    fixture_model(kind, col, kelvin, on)          # makes the materials (shared by name)
    from . import kit_detail as KD
    from .props import PM
    M_ = bpy.data.materials
    body, dark, yoke_m = M_["Fixture_Body"], M_["Fixture_Gasket"], M_["Fixture_Yoke"]
    p = PM(0.0)
    zc = FIXTURE_PIVOT[kind]
    hw, hh = (0.30, 0.22) if kind == "led" else (0.30, 0.25)
    if kind == "led":
        lens_m = M_[f"Fixture_Lens_LED{'_on' if on else ''}"]
        p.box((-hw, -0.05, zc - hh), (hw, 0.07, zc + hh), body)
        bz = 0.022
        p.decal((0, 0.0705, zc), (0, 1, 0), (0, 0, 1), 2 * (hw - bz), 2 * (hh - bz), lens_m, lift=0.0)
        vz = zc + hh
        L = 0.17
        p.obox((0, 0.07 + L / 2 * math.cos(math.radians(12)), vz + 0.012 - L / 2 * math.sin(math.radians(12))),
               (2 * hw + 0.012, L, 0.005), body, pitch=math.radians(-12))
        for side in (-1, 1):
            x = side * (hw + 0.003)
            fy, fz = 0.07 + L * math.cos(math.radians(12)), vz + 0.012 - L * math.sin(math.radians(12))
            p.extrude_yz(_ccw_yz([(0.045, vz - 0.07), (fy, fz - 0.03), (fy, fz + 0.003), (0.045, vz + 0.013)]),
                         x - 0.003, x + 0.003, body)
        for k in range(33):                                   # fins, every other one of the real ones
            x = -hw + 0.02 + k * (2 * hw - 0.04) / 32
            p.box((x - 0.003, -0.1, zc - hh + 0.012), (x + 0.003, -0.05, zc + hh - 0.012), body)
        p.box((-0.13, -0.155, zc - 0.08), (0.13, -0.1, zc + 0.08), body)
    else:
        refl = M_[f"Fixture_Lens_Halide{'_on' if on else ''}"]
        ring = lambda w, h, y: [(x, y, zc + z) for x, z in [(w, -h), (w, h), (-w, h), (-w, -h)]]
        _loft_solid(p, [ring(hw + 0.01, hh + 0.01, 0.075), ring(hw, hh, -0.06), ring(hw - 0.08, hh - 0.07, -0.25),
                        ring(hw - 0.1, hh - 0.09, -0.29)], body)
        p.decal((0, 0.0755, zc), (0, 1, 0), (0, 0, 1), 2 * (hw - 0.03), 2 * (hh - 0.03), refl, lift=0.0)
        p.cyl((-0.065, 0.085, zc - 0.02), (0.065, 0.085, zc - 0.02), 0.012, M_["Fixture_LampQuartz"], sides=6)
        p.obox((0, 0.13, zc + hh + 0.004), (2 * hw + 0.02, 0.13, 0.006), body, pitch=math.radians(-8))
        top0, top1 = zc + hh - 0.02, zc + hh - 0.07
        for k in range(9):
            x = -0.2 + k * 0.05
            p.extrude_yz([(-0.25, top1 - 0.01), (-0.06, top0 - 0.01), (-0.06, top0 + 0.022), (-0.25, top1 + 0.022)],
                         x - 0.004, x + 0.004, body)
            p.box((x - 0.004, -0.31, zc - 0.12), (x + 0.004, -0.27, zc + 0.12), body)
    xa = hw + 0.03
    for side in (-1, 1):
        p.cyl((side * hw, 0, zc), (side * (xa + 0.012), 0, zc), 0.042, body, sides=12)
        p.box((side * xa - 0.008, -0.04, 0.035), (side * xa + 0.008, 0.04, zc + 0.03), yoke_m)
        p.cyl((side * (xa + 0.008), 0, zc), (side * (xa + 0.011), 0, zc), 0.075, yoke_m, sides=12)
    p.box((-xa - 0.008, -0.04, 0.02), (xa + 0.008, 0.04, 0.05), yoke_m)
    p.box((-0.08, -0.08, 0.0), (0.08, 0.08, 0.02), yoke_m)
    obj = p.obj(name, col, bevel=False)
    KD.finish(obj, distance=0.25, ground=False)
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
        lines.append((np.array([(bx + cx * half(z), by + cy * half(z), z) for z in zs]), 0.11))
    for a, b in zip(zs[:-1], zs[1:]):
        for k in range(4):
            (c1x, c1y), (c2x, c2y) = corners[k], corners[(k + 1) % 4]
            p1a = (bx + c1x * half(a), by + c1y * half(a), a)
            p2b = (bx + c2x * half(b), by + c2y * half(b), b)
            p2a = (bx + c2x * half(a), by + c2y * half(a), a)
            p1b = (bx + c1x * half(b), by + c1y * half(b), b)
            lines.append((np.array([p1a, p2b]), 0.03))
            lines.append((np.array([p2a, p1b]), 0.03))
            lines.append((np.array([p1b, p2b]), 0.035))
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
    # stadium floodlights are switched on for televised day matches too: roof rings and masts show lit
    # lenses by day (grassroots poles only when on)
    model = fixture_model(spec["fixture"], col, spec["kelvin"], on or kind != "poles")
    model.hide_render = True
    model["template"] = True
    model.hide_viewport = True
    steel = galvanised()
    fixtures = []          # (pos, mount)
    clusters = []          # roof fixtures: the lamps of each cluster
    plates = []            # MeshAcc: platforms, plinths, cabinets
    extra = []             # steel polylines: catwalks, ladders, frames
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
            st_, acc_ = FL.mast_parts(m, pts, spec["cols"], spec["mast"] == "lattice")
            extra += st_
            plates.append(acc_)
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
        # a catwalk slung under the roof edge carries the fixtures in clusters on its lamp bar
        st_, acc_, cl = FL.roof_gantry(lines, spec["spacing"])
        extra += st_
        plates.append(acc_)
        for c, members in cl:
            fixtures.append((c, "roof"))
            clusters.append(members)
    elif kind == "poles":
        offset = pitch["runoff_side"] + 1.2
        # poles on the main-stand side also stand clear of the stand's ends
        clear = FL.POLE_CLEAR_X
        if stands and "south" in (stands.get("runs") or {}):
            from ..stands.geometry import _world
            run = stands["runs"]["south"][0]
            clear = max(clear, max(abs(_world(run, run.t0, 0.0, 0.0)[0]), abs(_world(run, run.t1, 0.0, 0.0)[0])) + 4.0)
        for k, pole in enumerate(FL.poles(L, W, spec, offset, clear)):
            bx, by = pole["base"]
            h = pole["height"]
            tubes_, acc_ = FL.pole_parts(pole, spec["lamps"])
            for j, (a, b, r) in enumerate(tubes_):
                tube(f"Pole_{k}_{j}", [a, b], r, col, steel).parent = root
            plates.append(acc_)
            side = np.array([1.0, 0, 0])
            for j in range(spec["lamps"]):
                p = np.array([bx, by, h + 0.3]) + side * (j - (spec["lamps"] - 1) / 2) * 0.8
                fixtures.append((p - np.array([0, np.sign(by) * 0.5, 0]), "pole"))
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
    if extra:
        _tubes_mesh("FloodlightSteel", extra, col, root, steel, sides=6)
    _plates(plates, col, root, steel)
    scale = FL.LAMP_SCALE if kind == "roof" else 1.0       # roof floods are larger than the library fixture
    pivot = FIXTURE_PIVOT.get(spec["fixture"], 0.0) * scale
    for k, ((p, mount), t) in enumerate(zip(fixtures, targets)):
        for j, q in enumerate(clusters[k] if clusters else [p]):
            # the housing centre sits on the design point; the origin (yoke plate) below it
            o = _place(model, FL.mount_point(q, t, pivot), t, col, root, f"Fixture_{k:03d}_{j}")
            o.scale = (scale, scale, scale)
    if kind == "roof" and spec["fixture"] == "led":
        # the lit diffuser panels of the roof ring's floods (dimmer by day, when they are not on)
        lamps = [(q, t) for (p, _), t, members in zip(fixtures, targets, clusters) for q in members]
        V, F, _, UV = FL.lens_panels(lamps, scale).arrays()
        glow = lens("Floodlight_Glow", spec["kelvin"], 14.0 if on else 7.0)
        mesh_object("FloodlightLenses", V, F, col, root, materials=[glow], uvs=UV)
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


def _plates(accs, col, parent, steel):
    """Platforms, decks, plinths and cabinets (FL.TAG_STEEL / TAG_CONCRETE / TAG_DARK)."""
    from ..stands.geometry import MeshAcc
    from .stand import _recalc, concrete
    acc = MeshAcc()
    for a in accs:
        V, F, tags, UV = a.arrays()
        if not len(F):
            continue
        base = acc.n
        acc.verts.append(np.asarray(V))
        acc.n += len(V)
        acc.faces += [tuple(i + base for i in f) for f in F]
        acc.tags += list(tags)
        acc.uvs += [tuple(u) for u in UV]
    V, F, tags, UV = acc.arrays()
    if not len(F):
        return None
    body = MAT.painted_metal("Fixture_Body", "#3a3d40", rough=0.45, dirt=0.0, scuffs=0.1)
    o = mesh_object("FloodlightPlates", V, F, col, parent, materials=[steel, concrete("Stand_Concrete"), body],
                    mat_index=tags, uvs=UV)
    _recalc(o)
    return o


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
