"""Materials for training grounds (kit/site): ground (asphalt, gravel, dirt, rubble, sand), building
skins (corrugated steel, timber, panels, render, old brick, rusted sheet), roofs (slate, clay tile,
fibre cement, membrane, sedum, solar), and odds (water, rust, plywood, decking).

Patterns that belong on walls are laid out in wall coordinates: along the wall and up it, which on a
horizontal surface (the game-export bake plane, or a flat roof) become x and y. So one material reads
right on a facade in Blender and bakes to a tile that the generator's metric UVs (u along, v up) map
the same way.
"""
import bpy

from ..core import materials as MAT
from ..core.nodes import material, srgb
from . import facade, seat, stand


def _get(name):
    return bpy.data.materials.get(name)


def wall_coord(nb):
    """(along, up) in metres: along the wall and up it; x and y on horizontal faces."""
    geo = nb.geometry()
    pos, nrm = geo["Position"], geo["Normal"]
    x, y, z = nb.sep(pos)
    nx, ny, nz = nb.sep(nrm)
    ax, ay, az = nb.math("ABSOLUTE", nx), nb.math("ABSOLUTE", ny), nb.math("ABSOLUTE", nz)
    along = nb.add(nb.mul(x, nb.add(ay, az)), nb.mul(y, ax))
    up = nb.add(nb.mul(z, nb.sub(1.0, az)), nb.mul(y, az))
    return along, up, pos


def asphalt(name="Site_Asphalt", base="#464441", faded=False):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    agg = nb.voronoi(pos, 140.0, out="Distance")
    stones = nb.smooth(0.05, 0.25, agg)
    tone = nb.noise(pos, 0.6, detail=4)
    # warm grey (bitumen greys towards the stone as it wears; a cool grey reads navy under skylight)
    col = nb.mix(nb.mul(nb.sub(1.0, stones), 0.35), srgb(base), srgb("#8e8981"))
    col = nb.mix(nb.mul(nb.smooth(0.55, 0.8, tone), 0.25), col, srgb("#2e2c2a"))
    stain = nb.smooth(0.62, 0.8, nb.noise(pos, 0.25, detail=5))
    col = nb.mix(nb.mul(stain, 0.4), col, srgb("#242220"))
    rough = nb.madd(stones, -0.12, 0.9)
    if faded:
        crack = nb.voronoi(nb.vmath("MULTIPLY", pos, (1.0, 1.0, 0.0)), 0.9, feature="DISTANCE_TO_EDGE")
        lines = nb.sub(1.0, nb.smooth(0.0, 0.025, crack))
        col = nb.mix(nb.mul(lines, 0.75), col, srgb("#1c1d1d"))
        weeds = nb.mul(lines, nb.smooth(0.55, 0.7, nb.noise(pos, 3.0, detail=3)))
        col = nb.mix(nb.mul(weeds, 0.8), col, srgb("#3f5124"))
        patch = nb.smooth(0.6, 0.64, nb.noise(pos, 0.18, detail=2))
        col = nb.mix(nb.mul(patch, 0.5), col, srgb("#2b2c2e"))
        bump = nb.add(nb.mul(stones, 0.6), nb.mul(lines, -0.8))
    else:
        bump = nb.mul(stones, 0.6)
    nb.output(nb.principled(Base_Color=col, Roughness=rough, Normal=nb.bump(bump, 0.35, 0.004)))
    return m


def gravel(name="Site_Gravel"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    v = nb.node("ShaderNodeTexVoronoi")
    nb.connect(pos, v.inputs["Vector"])
    v.inputs["Scale"].default_value = 55.0
    cell = v.outputs["Color"]
    d = v.outputs["Distance"]
    shade = nb.sep(cell)[0]
    col = nb.mix(shade, srgb("#6e6960"), srgb("#a39d91"))
    col = nb.mix(nb.mul(nb.smooth(0.25, 0.5, d), 0.6), col, srgb("#4a463f"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.88, Normal=nb.bump(nb.sub(1.0, d), 0.7, 0.012)))
    return m


def dirt(name="Site_Dirt", color="#5a4632"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    clod = nb.noise(pos, 9.0, detail=6, rough=0.7)
    damp = nb.smooth(0.5, 0.75, nb.noise(pos, 0.3, detail=4))
    col = nb.mix(nb.mul(clod, 0.5), srgb(color), srgb("#7a6449"))
    col = nb.mix(nb.mul(damp, 0.5), col, srgb("#3a2c20"))
    stones = nb.smooth(0.82, 0.86, nb.noise(pos, 40.0, detail=1))
    col = nb.mix(nb.mul(stones, 0.6), col, srgb("#8d8678"))
    nb.output(nb.principled(Base_Color=col, Roughness=nb.madd(damp, -0.2, 0.92), Normal=nb.bump(clod, 0.6, 0.02)))
    return m


def rubble(name="Site_Rubble"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    v = nb.node("ShaderNodeTexVoronoi")
    nb.connect(pos, v.inputs["Vector"])
    v.inputs["Scale"].default_value = 9.0
    r, g, _ = nb.sep(v.outputs["Color"])
    concrete = nb.mix(g, srgb("#8f8b84"), srgb("#b3aea6"))
    col = nb.mix(nb.smooth(0.72, 0.75, r), concrete, srgb("#8d4a35"))        # brick fragments
    col = nb.mix(nb.mul(nb.smooth(0.2, 0.45, v.outputs["Distance"]), 0.5), col, srgb("#4a443c"))
    dust = nb.noise(pos, 0.5, detail=3)
    col = nb.mix(nb.mul(dust, 0.25), col, srgb("#a29a8c"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.9, Normal=nb.bump(nb.sub(1.0, v.outputs["Distance"]), 0.8, 0.04)))
    return m


def sand(name="Site_Sand"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    n = nb.noise(pos, 30.0, detail=5)
    col = nb.mix(nb.mul(n, 0.4), srgb("#c2a878"), srgb("#a88f63"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.95, Normal=nb.bump(n, 0.2, 0.004)))
    return m


def water(name="Site_Water"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    ripple = nb.noise(nb.vmath("MULTIPLY", pos, (1.0, 2.5, 1.0)), 3.0, detail=4)
    nb.output(nb.principled(Base_Color=srgb("#1c2a25"), Roughness=0.04, Specular_IOR_Level=0.6,
                            Normal=nb.bump(ripple, 0.08, 0.02)))
    return m


def rust(name="Site_Rust"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    n = nb.noise(pos, 6.0, detail=6, rough=0.65)
    col = nb.mix(nb.smooth(0.4, 0.7, n), srgb("#5a2e1a"), srgb("#a2552a"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.78, Metallic=0.25, Normal=nb.bump(n, 0.4, 0.004)))
    return m


def car_scrap(name="Car_Scrap"):
    """An abandoned car's skin: sun-faded red paint, blotches and runs of rust, grime."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    x, y, z = nb.sep(pos)
    blot = nb.noise(pos, 2.2, detail=8, rough=0.7)
    rust_m = nb.smooth(0.56, 0.64, blot)
    runs = nb.noise(nb.vmath("MULTIPLY", pos, (9.0, 1.2, 1.0)), 3.0, detail=4)
    rust_m = nb.math("MAXIMUM", rust_m, nb.mul(nb.smooth(0.62, 0.7, runs), 0.6))
    fade = nb.noise(pos, 0.8, detail=3)
    paint = nb.mix(nb.smooth(0.3, 0.7, fade), srgb("#6e2a22"), srgb("#9a5a4c"))
    rust = nb.mix(nb.smooth(0.4, 0.7, nb.noise(pos, 14.0, detail=4)), srgb("#4a2414"), srgb("#8a4a24"))
    col = nb.mix(rust_m, paint, rust)
    nb.output(nb.principled(Base_Color=col, Roughness=nb.add(0.55, nb.mul(rust_m, 0.3)),
                            Normal=nb.bump(nb.mul(rust_m, nb.noise(pos, 30.0, detail=3)), 0.3, 0.002)))
    return m


def scrap(name="Site_Scrap"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    v = nb.node("ShaderNodeTexVoronoi")
    nb.connect(pos, v.inputs["Vector"])
    v.inputs["Scale"].default_value = 4.0
    r, g, b = nb.sep(v.outputs["Color"])
    painted = nb.comb(nb.madd(r, 0.4, 0.15), nb.madd(g, 0.35, 0.15), nb.madd(b, 0.35, 0.15))
    rusty = nb.mix(nb.noise(pos, 8.0, detail=4), srgb("#5a2e1a"), srgb("#93502a"))
    col = nb.mix(nb.smooth(0.3, 0.6, nb.noise(pos, 3.0, detail=4)), painted, rusty)
    nb.output(nb.principled(Base_Color=col, Roughness=0.7, Metallic=0.4,
                            Normal=nb.bump(nb.sub(1.0, v.outputs["Distance"]), 1.0, 0.05)))
    return m


def planks(name, color, board=0.15, along_x=True, rough=0.7):
    """Boards: decking on the ground (along x) or vertical cladding (along the wall)."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    # board index from the along coordinate; grain runs up the board
    k = nb.math("FLOOR", nb.div(along, board))
    seam = nb.smooth(0.0, 0.03, nb.math("FRACT", nb.div(along, board)))
    rnd = nb.math("FRACT", nb.mul(nb.math("SINE", nb.mul(k, 12.9898)), 43758.5453))
    grain = nb.noise(nb.comb(nb.mul(along, 30.0), nb.mul(up, 1.5), k), 3.0, detail=5, distortion=0.3)
    c = srgb(color)
    col = nb.mix(nb.madd(rnd, 0.35, nb.mul(grain, 0.25)), c, tuple(v * 0.7 for v in c[:3]) + (1,))
    col = nb.mix(nb.sub(1.0, seam), col, srgb("#1b1612"))
    col, r = MAT.weathering(nb, col, nb.madd(grain, 0.15, rough), 0.6)
    nb.output(nb.principled(Base_Color=col, Roughness=r, Normal=nb.bump(nb.add(seam, nb.mul(grain, 0.2)), 0.4, 0.004)))
    return m


def corrugated(name, color, rusty=0.0, pitch=0.076, box=False):
    """Profiled steel sheet: vertical corrugations on walls, weathered or rusting. `box`: the
    trapezoidal (box) profile of modern cladding instead, crowns a third of the pitch wide
    with sloping flanks, in a smooth factory finish (sheet-to-sheet tone only)."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    c = srgb(color)
    if box:
        u = nb.math("FRACT", nb.div(along, pitch))
        wave = nb.mul(nb.smooth(0.0, 0.09, u), nb.sub(1.0, nb.smooth(0.33, 0.42, u)))
        # a faint, broad tone drift (the sheet repeats every tile, so nothing sheet-by-sheet)
        tone = nb.noise(pos, 0.6, detail=2)
        col = nb.mix(nb.mul(tone, 0.06), c, tuple(v * 0.9 for v in c[:3]) + (1,))
        # the valleys sit in the crowns' shade
        col = nb.mix(nb.mul(nb.sub(1.0, wave), 0.18), col, tuple(v * 0.55 for v in c[:3]) + (1,))
        col, rough = MAT.weathering(nb, col, 0.38, 0.3)
        nb.output(nb.principled(Base_Color=col, Roughness=rough, Metallic=0.3, Normal=nb.bump(wave, 0.9, 0.03)))
        return m
    wave = nb.math("SINE", nb.mul(along, 6.2832 / pitch))
    tone = nb.noise(pos, 1.5, detail=4)
    col = nb.mix(nb.mul(tone, 0.2), c, tuple(v * 0.8 for v in c[:3]) + (1,))
    if rusty > 0:
        rmask = nb.smooth(0.42, 0.62, nb.noise(pos, 1.2, detail=6, rough=0.65))
        streak = nb.smooth(0.5, 0.8, nb.noise(nb.comb(nb.mul(along, 3.0), nb.mul(up, 0.15), 0.0), 2.0, detail=3))
        mask = nb.clamp01(nb.mul(nb.add(rmask, nb.mul(streak, 0.5)), rusty))
        rcol = nb.mix(nb.noise(pos, 7.0, detail=4), srgb("#5a2e1a"), srgb("#9a4f28"))
        col = nb.mix(mask, col, rcol)
        rough = nb.madd(mask, 0.35, 0.4)
    else:
        rough = 0.4
    col, rough = MAT.weathering(nb, col, rough, 0.7)
    nb.output(nb.principled(Base_Color=col, Roughness=rough, Metallic=0.35, Normal=nb.bump(wave, 0.5, 0.012)))
    return m


def panels(name, color, pw=1.2, ph=3.0):
    """Rainscreen panels with dark joints."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    ja = nb.smooth(0.0, 0.012, nb.math("FRACT", nb.div(along, pw)))
    ju = nb.smooth(0.0, 0.005, nb.math("FRACT", nb.div(up, ph)))
    joint = nb.sub(1.0, nb.mul(ja, ju))
    k = nb.add(nb.math("FLOOR", nb.div(along, pw)), nb.mul(nb.math("FLOOR", nb.div(up, ph)), 17.0))
    rnd = nb.math("FRACT", nb.mul(nb.math("SINE", nb.mul(k, 12.9898)), 43758.5453))
    c = srgb(color)
    col = nb.mix(nb.mul(rnd, 0.08), c, tuple(v * 0.85 for v in c[:3]) + (1,))
    col = nb.mix(joint, col, srgb("#2a2b2d"))
    col, r = MAT.weathering(nb, col, 0.32, 0.4)
    nb.output(nb.principled(Base_Color=col, Roughness=r, Specular_IOR_Level=0.5, Normal=nb.bump(joint, -0.3, 0.004)))
    return m


def render(name, color):
    """Smooth painted render: trowel texture, dirt washing down from sills and copings."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    trowel = nb.noise(pos, 6.0, detail=6, rough=0.6)
    c = srgb(color)
    col = nb.mix(nb.mul(trowel, 0.12), c, tuple(v * 0.85 for v in c[:3]) + (1,))
    col, r = MAT.weathering(nb, col, nb.madd(trowel, 0.1, 0.7), 0.9)
    nb.output(nb.principled(Base_Color=col, Roughness=r, Normal=nb.bump(trowel, 0.15, 0.002)))
    return m


def tiles(name, color, w, h, curved=False, var=0.25):
    """Roof tiles or slates in courses: rows of overlapping units with a shadow at each lap."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    row = nb.math("FLOOR", nb.div(up, h))
    shift = nb.mul(nb.math("MODULO", row, 2.0), 0.5 * w)
    a = nb.add(along, shift)
    k = nb.add(nb.math("FLOOR", nb.div(a, w)), nb.mul(row, 31.0))
    rnd = nb.math("FRACT", nb.mul(nb.math("SINE", nb.mul(k, 12.9898)), 43758.5453))
    fu = nb.math("FRACT", nb.div(up, h))
    fa = nb.math("FRACT", nb.div(a, w))
    lap = nb.smooth(0.0, 0.12, fu)                                 # shadow under the course above
    gap = nb.smooth(0.0, 0.04, nb.math("PINGPONG", fa, 0.5))
    height = nb.mul(lap, gap)
    if curved:
        height = nb.mul(height, nb.math("SINE", nb.mul(fa, 3.1416)))
    c = srgb(color)
    col = nb.mix(nb.mul(rnd, var), c, tuple(v * 0.7 for v in c[:3]) + (1,))
    col = nb.mix(nb.sub(1.0, lap), col, srgb("#1b1b1c"))
    lichen = nb.smooth(0.62, 0.78, nb.noise(pos, 1.5, detail=5))
    col = nb.mix(nb.mul(lichen, 0.35), col, srgb("#6f7454"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.72, Normal=nb.bump(height, 0.6, 0.012)))
    return m


def membrane(name="Roof_Flat"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    seam = nb.sub(1.0, nb.smooth(0.0, 0.01, nb.math("FRACT", nb.div(along, 2.0))))
    tone = nb.noise(pos, 0.8, detail=4)
    col = nb.mix(nb.mul(tone, 0.2), srgb("#3c3f42"), srgb("#55585c"))
    col = nb.mix(nb.mul(seam, 0.6), col, srgb("#2c2e30"))
    puddle = nb.smooth(0.7, 0.8, nb.noise(pos, 0.3, detail=3))
    nb.output(nb.principled(Base_Color=col, Roughness=nb.madd(puddle, -0.5, 0.75), Normal=nb.bump(seam, 0.4, 0.004)))
    return m


def sedum(name="Roof_Sedum"):
    """A sedum blanket: mats of mixed species in drifts a metre or two across (green, lime, olive and
    the red-bronze of stressed sedum album), each drift a carpet of rosette clumps with the dark
    substrate showing between them."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    # drifts: two broad noises pick the species mix
    d1 = nb.noise(pos, 0.45, detail=3, rough=0.55)
    d2 = nb.noise(nb.vmath("ADD", pos, (17.0, 5.0, 0.0)), 0.6, detail=3, rough=0.55)
    col = nb.mix(nb.smooth(0.44, 0.56, d1), srgb("#55752f"), srgb("#8d9b45"))
    col = nb.mix(nb.mul(nb.smooth(0.58, 0.66, d2), 0.75), col, srgb("#7e4c33"))
    col = nb.mix(nb.mul(nb.smooth(0.4, 0.46, d1), nb.sub(1.0, nb.smooth(0.5, 0.56, d1))), col, srgb("#71723a"))
    # clumps and rosettes, a little lighter at the tips
    clump = nb.noise(pos, 6.0, detail=4, rough=0.6)
    v = nb.voronoi(pos, 35.0, out="Distance")
    col = nb.mix(nb.mul(nb.smooth(0.25, 0.6, v), 0.35), col, nb.mix(0.5, col, (1.0, 1.0, 0.8, 1.0), blend="MULTIPLY"))
    col = nb.mix(nb.mul(nb.smooth(0.45, 0.6, clump), 0.25), col, nb.mix(1.0, col, (1.2, 1.15, 0.9, 1.0), blend="MULTIPLY"))
    # substrate between the clumps (thin spots), more of it in the drier drifts
    gap = nb.smooth(0.62, 0.72, nb.sub(1.0, clump))
    col = nb.mix(nb.mul(gap, 0.7), col, srgb("#3b3328"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.85,
                            Normal=nb.bump(nb.add(nb.mul(nb.sub(1.0, v), 0.7), nb.mul(clump, 0.6)), 0.6, 0.02)))
    return m


# -- interiors (kit/site/medical.py) ---------------------------------------------------------------

def _grid(nb, pos, size):
    """(fx, fy) in [0, 1) across each size x size cell, and a per-cell random in [0, 1)."""
    x, y, _ = nb.sep(pos)
    fx, fy = nb.math("FRACT", nb.div(x, size)), nb.math("FRACT", nb.div(y, size))
    k = nb.add(nb.math("FLOOR", nb.div(x, size)), nb.mul(nb.math("FLOOR", nb.div(y, size)), 37.0))
    rnd = nb.math("FRACT", nb.mul(nb.math("SINE", nb.mul(k, 12.9898)), 43758.5453))
    return fx, fy, rnd


def _seams(nb, fx, fy, w):
    """1 on a cell's edges (w: the seam's share of the cell), 0 inside."""
    ex = nb.sub(1.0, nb.mul(nb.smooth(0.0, w, fx), nb.smooth(0.0, w, nb.sub(1.0, fx))))
    ey = nb.sub(1.0, nb.mul(nb.smooth(0.0, w, fy), nb.smooth(0.0, w, nb.sub(1.0, fy))))
    return nb.math("MAXIMUM", ex, ey)


def vinyl(name="Floor_Vinyl"):
    """Safety vinyl sheet (Altro class): grey-blue with dark and light flecks and the carborundum
    grit that makes it slip-resistant, heat-welded seams every 2 m, scuffs in the traffic."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col = nb.mix(nb.mul(nb.noise(pos, 1.2, detail=3), 0.15), srgb("#b7bfc4"), srgb("#a3abb1"))
    v = nb.voronoi(pos, 220.0, out="Distance")
    col = nb.mix(nb.mul(nb.smooth(0.08, 0.0, v), 0.7), col, srgb("#5d666d"))
    v2 = nb.voronoi(nb.vmath("ADD", pos, (3.1, 7.7, 0.0)), 180.0, out="Distance")
    col = nb.mix(nb.mul(nb.smooth(0.07, 0.0, v2), 0.6), col, srgb("#eef1f2"))
    fx, fy, _ = _grid(nb, pos, 2.0)
    seam = nb.sub(1.0, nb.mul(nb.smooth(0.0, 0.0025, fx), nb.smooth(0.0, 0.0025, nb.sub(1.0, fx))))
    col = nb.mix(nb.mul(seam, 0.5), col, srgb("#8d969c"))
    col = nb.mix(nb.mul(nb.smooth(0.55, 0.8, nb.noise(pos, 0.6, detail=4)), 0.12), col, srgb("#8a8d8a"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.62, Normal=nb.bump(nb.add(nb.smooth(0.05, 0.0, v), nb.mul(seam, -0.5)), 0.15, 0.001)))
    return m


def rubber_floor(name="Floor_Rubber"):
    """Gym rubber tiles (1 m, 10 mm): black with grey and blue EPDM flecks, tight joints."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col = nb.mix(nb.mul(nb.noise(pos, 3.0, detail=3), 0.2), srgb("#1e1f21"), srgb("#28292c"))
    v = nb.voronoi(pos, 140.0, out="Distance")
    col = nb.mix(nb.mul(nb.smooth(0.12, 0.0, v), 0.8), col, srgb("#6f7378"))
    v2 = nb.voronoi(nb.vmath("ADD", pos, (5.3, 1.1, 0.0)), 70.0, out="Distance")
    col = nb.mix(nb.mul(nb.smooth(0.07, 0.0, v2), 0.8), col, srgb("#2f5a8a"))
    fx, fy, _ = _grid(nb, pos, 1.0)
    seam = _seams(nb, fx, fy, 0.004)
    col = nb.mix(nb.mul(seam, 0.7), col, srgb("#0b0b0c"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.9, Normal=nb.bump(nb.add(nb.mul(seam, -1.0), nb.mul(nb.noise(pos, 60.0), 0.3)), 0.2, 0.002)))
    return m


def ceramic_tile(name="Floor_Tile"):
    """200 mm white ceramic tiles in stack bond with mid-grey grout (pool halls, recovery suites,
    wet-room walls); a slight tile-to-tile shade."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    fx, fy, rnd = _grid(nb, pos, 0.2)
    grout = _seams(nb, fx, fy, 0.03)
    col = nb.mix(nb.mul(rnd, 0.08), srgb("#ecedea"), srgb("#d9dbd8"))
    col = nb.mix(grout, col, srgb("#a7a9a4"))
    nb.output(nb.principled(Base_Color=col, Roughness=nb.madd(grout, 0.5, 0.2), Normal=nb.bump(nb.mul(grout, -1.0), 0.3, 0.002)))
    return m


def carpet_tiles(name="Floor_Carpet"):
    """500 mm loop-pile carpet tiles laid quarter-turned (offices, staff and sleep rooms): a grey
    with a fine tufted texture whose direction alternates tile to tile."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    fx, fy, rnd = _grid(nb, pos, 0.5)
    tuft = nb.noise(nb.vmath("MULTIPLY", pos, (1.0, 6.0, 1.0)), 90.0, detail=2)
    tuft2 = nb.noise(nb.vmath("MULTIPLY", pos, (6.0, 1.0, 1.0)), 90.0, detail=2)
    x, y, _ = nb.sep(pos)
    chk = nb.math("FRACT", nb.mul(nb.add(nb.math("FLOOR", nb.div(x, 0.5)), nb.math("FLOOR", nb.div(y, 0.5))), 0.5))
    t = nb.mix(nb.smooth(0.2, 0.3, chk), tuft, tuft2)
    col = nb.mix(nb.mul(t, 0.35), srgb("#6c737c"), srgb("#5a6068"))
    col = nb.mix(nb.mul(rnd, 0.1), col, srgb("#787f88"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.97, Normal=nb.bump(t, 0.25, 0.002)))
    return m


def interior_paint(name="TINT_interior", color="#eceeee"):
    """Emulsion on plaster (recoloured per room): a faint roller texture, nothing else."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col = nb.mix(nb.mul(nb.noise(pos, 40.0, detail=3), 0.04), srgb(color), tuple(v * 0.94 for v in srgb(color)[:3]) + (1,))
    nb.output(nb.principled(Base_Color=col, Roughness=0.85, Normal=nb.bump(nb.noise(pos, 120.0), 0.04, 0.0005)))
    return m


def ceiling_grid(name="Ceiling_Grid"):
    """A suspended ceiling: 600 mm mineral-fibre tiles with a fissured face in a white T-bar grid."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    fx, fy, rnd = _grid(nb, pos, 0.6)
    bar = _seams(nb, fx, fy, 0.02)
    fiss = nb.smooth(0.6, 0.7, nb.noise(pos, 70.0, detail=3))
    col = nb.mix(nb.mul(fiss, 0.25), srgb("#ecebe6"), srgb("#c9c8c2"))
    col = nb.mix(nb.mul(rnd, 0.04), col, srgb("#e2e0da"))
    col = nb.mix(bar, col, srgb("#f6f6f4"))
    nb.output(nb.principled(Base_Color=col, Roughness=nb.madd(bar, -0.5, 0.92), Normal=nb.bump(nb.add(bar, nb.mul(fiss, -0.3)), 0.3, 0.002)))
    return m


def solar(name="Solar_Panel"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    along, up, pos = wall_coord(nb)
    cell = 0.165
    ca = nb.smooth(0.0, 0.04, nb.math("PINGPONG", nb.math("FRACT", nb.div(along, cell)), 0.5))
    cu = nb.smooth(0.0, 0.04, nb.math("PINGPONG", nb.math("FRACT", nb.div(up, cell)), 0.5))
    grid = nb.sub(1.0, nb.mul(ca, cu))
    col = nb.mix(grid, srgb("#141c33"), srgb("#a7acb3"))
    nb.output(nb.principled(Base_Color=col, Roughness=nb.madd(grid, 0.4, 0.12), Metallic=nb.mul(grid, 0.8),
                            Specular_IOR_Level=0.7, Coat_Weight=0.8, Coat_Roughness=0.05))
    return m


def opening(name="Bldg_Opening"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#0d0d0e"), Roughness=0.95))
    return m


def lit_glass(name="Bldg_Glass_Lit"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#2a2620"), Roughness=0.1, Emission_Color=srgb("#ffd8a6"),
                            Emission_Strength=1.6))
    return m


def old_brick(name="Bldg_Brick_Old", color="#6a3a2c"):
    if _get(name):
        return _get(name)
    m = facade.brick(name, color)
    nb_tree = m.node_tree
    # soot and streaks over the brick
    from ..core.nodes import NB
    nb = NB(nb_tree, clear=False)
    p = next(n for n in nb_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    base = p.inputs["Base Color"].links[0].from_socket
    pos = nb.geometry()["Position"]
    soot = nb.smooth(0.45, 0.75, nb.noise(pos, 0.6, detail=5))
    col = nb.mix(nb.mul(soot, 0.55), base, srgb("#231a16"))
    nb.connect(col, p.inputs["Base Color"])
    return m


# name -> (builder, the sRGB colour it is baked with: the editor recolours relative to it)
MATERIALS = {
    "Site_Asphalt": (lambda: asphalt(), "#464441"),
    "Site_Asphalt_Old": (lambda: asphalt("Site_Asphalt_Old", "#5a5853", faded=True), "#5a5853"),
    "Site_Gravel": (lambda: gravel(), None),
    "Site_Dirt": (lambda: dirt(), "#5a4632"),
    "Site_Rubble": (lambda: rubble(), None),
    "Site_Sand": (lambda: sand(), "#c2a878"),
    "Site_Kerb": (lambda: stand.concrete("Site_Kerb", "#b9b5ac"), "#b9b5ac"),
    "Site_Water": (lambda: water(), None),
    "Site_Rust": (lambda: rust(), None),
    "Site_Scrap": (lambda: scrap(), None),
    "Site_Wood": (lambda: planks("Site_Wood", "#7a5a3c", board=0.14), "#7a5a3c"),
    "Site_Plywood": (lambda: planks("Site_Plywood", "#b48a59", board=1.22, rough=0.8), "#b48a59"),
    "TINT_road_paint": (lambda: stand.painted("TINT_road_paint", "#ecebe4", 0.6), "#ecebe4"),
    "TINT_fence": (lambda: MAT.painted_metal("TINT_fence", "#2b4433", rough=0.4, dirt=0.3, scuffs=0.2), "#2b4433"),
    "TINT_trim": (lambda: MAT.painted_metal("TINT_trim", "#2a2c30", rough=0.35, dirt=0.2, scuffs=0.1), "#2a2c30"),
    "TINT_door": (lambda: stand.painted("TINT_door", "#c8102e", 0.4), "#c8102e"),
    "TINT_club": (lambda: stand.painted("TINT_club", "#c8102e", 0.4), "#c8102e"),
    "TINT_timber": (lambda: planks("TINT_timber", "#b38455", board=0.15), "#b38455"),
    "TINT_corrugated": (lambda: corrugated("TINT_corrugated", "#59616a", pitch=1.52 / 6.0, box=True), "#59616a"),
    "TINT_panel": (lambda: panels("TINT_panel", "#e9ebec"), "#e9ebec"),
    "TINT_render": (lambda: render("TINT_render", "#f0ebe0"), "#f0ebe0"),
    "TINT_court": (lambda: stand.painted("TINT_court", "#5e7356", 0.8), "#5e7356"),
    "Bldg_Rusted": (lambda: corrugated("Bldg_Rusted", "#7c5a46", rusty=1.0), "#7c5a46"),
    "Bldg_Brick_Old": (lambda: old_brick(), "#6a3a2c"),
    "Bldg_Opening": (lambda: opening(), None),
    "Bldg_Glass_Lit": (lambda: lit_glass(), None),
    "Roof_Cement": (lambda: corrugated("Roof_Cement", "#8f8c85", rusty=0.0, pitch=0.146), "#8f8c85"),
    "Roof_Flat": (lambda: membrane(), None),
    "Roof_Slate": (lambda: tiles("Roof_Slate", "#3a3f46", 0.3, 0.25, var=0.15), "#3a3f46"),
    "Roof_Terracotta": (lambda: tiles("Roof_Terracotta", "#b5562f", 0.24, 0.32, curved=True, var=0.3), "#b5562f"),
    "Roof_Sedum": (lambda: sedum(), None),
    "Solar_Panel": (lambda: solar(), None),
    "Floor_Vinyl": (lambda: vinyl(), None),
    "Floor_Rubber": (lambda: rubber_floor(), None),
    "Floor_Tile": (lambda: ceramic_tile(), None),
    "Floor_Carpet": (lambda: carpet_tiles(), None),
    "TINT_interior": (lambda: interior_paint(), "#eceeee"),
    "Ceiling_Grid": (lambda: ceiling_grid(), None),
}

# large surfaces baked as tileable sets for games (size in metres, exact pattern multiple)
TILEABLE = {
    "Site_Asphalt": (4.0, False), "Site_Asphalt_Old": (4.0, False), "Site_Gravel": (2.0, False),
    "Site_Dirt": (4.0, False), "Site_Rubble": (4.0, False), "Site_Sand": (2.0, False), "Site_Kerb": (2.0, False),
    "Site_Wood": (2.1, True), "Site_Plywood": (2.44, True), "TINT_timber": (2.4, True),
    "TINT_corrugated": (1.52, True), "TINT_panel": (3.6, True), "TINT_render": (4.0, False),
    "TINT_court": (4.0, False), "Bldg_Rusted": (3.04, True), "Bldg_Brick_Old": (3.6, True),
    "Roof_Cement": (2.92, True), "Roof_Flat": (4.0, True), "Roof_Slate": (3.0, True), "Roof_Terracotta": (2.88, True),
    "Roof_Sedum": (12.0, False), "Solar_Panel": (1.98, True), "Site_Scrap": (4.0, False), "Site_Rust": (2.0, False),
    "Car_Scrap": (2.0, False), "Floor_Vinyl": (2.0, True), "Floor_Rubber": (2.0, True), "Floor_Tile": (2.0, True),
    "Floor_Carpet": (2.0, True), "TINT_interior": (2.0, False), "Ceiling_Grid": (2.4, True),
}
KEEP = ("Site_Water", "Bldg_Glass_Lit", "Bldg_Opening")


def build(name):
    """A site material by name (None for library materials made elsewhere)."""
    entry = MATERIALS.get(name)
    return entry[0]() if entry else None


__all__ = ["MATERIALS", "TILEABLE", "KEEP", "build", "seat"]
