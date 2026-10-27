"""Press, broadcast and content-production kit for a club's media building (kit/site/interior.py):
the press conference room (sponsor backdrop, roller banners, the top table with its microphone
cluster, rows of press seating), cameras (a tripod ENG camera, a studio pedestal camera with its
teleprompter), studio lighting (softboxes, a ceiling lighting grid), the TV studio set (presenter
desk, curved LED backdrop), the gallery (multiview wall, vision-mixing desk), edit desks, the
podcast table, acoustic panels, ring lights, a green screen, the flash-interview set, equipment
racking, the make-up station and its mirror. Real sizes; origin on the floor at the footprint
centre, +Y the front (the user's side; a camera's and a light's lens; wall units: the back
against the wall at y = 0, facing +y).
"""
import math

import numpy as np

from .texpaint import Canvas, decal, noise, project
from .props import PM, _paint, _plastic, clear_glass, tyre
from .props_amenity import _tube
from .props_medical import _monitor, _white
from .props_hospitality import _mirror_glass
from .props_office import _arm_monitors, _desk_frame, _display, _keyboard, _mug, _place, _task_chair


# -- shared bits ------------------------------------------------------------------------------------

def _sponsor_canvas(name, w, h, ppm=200, seed=1):
    """A step-and-repeat print: the club name and partner logos alternating on white."""
    cv = Canvas(name, 1024, 1024, col="#ffffff", rough=0.5)
    r = cv.alloc(w, h, ppm)
    r.fill(col="#f6f6f4")
    step_x, step_y = 0.6, 0.4
    logos = ("UNITED", "PARTNER", "AIRLINE", "BANK", "UNITED", "KIT CO")
    j = 0
    y = 0.15
    while y < h:
        x = 0.15 + (0.3 if j % 2 else 0.0)
        i = 0
        while x < w - 0.1:
            t = logos[(i + j * 2 + seed) % len(logos)]
            if t == "UNITED":
                r.poly([(x - 0.06, y + 0.11), (x + 0.06, y + 0.11), (x + 0.06, y + 0.02), (x, y - 0.04), (x - 0.06, y + 0.02)], col="#1d4f91")
                r.text("U", x, y + 0.02, 0.06, weight="black", align="c", col="#ffffff")
            else:
                c = ("#c8202a", "#2a2c30", "#e3a020", "#2a8a4a")[(i + j) % 4]
                r.rect(x - 0.13, y - 0.02, x + 0.13, y + 0.1, r=0.02, col=c)
                r.text(t, x, y + 0.02, 0.05, weight="black", align="c", col="#ffffff", max_w=0.24)
            x += step_x
            i += 1
        y += step_y
        j += 1
    return cv, r


def _tripod(p, M, h, spread=0.45):
    """A three-legged tripod with a mid-level spreader, its head at height h."""
    p.r = 0.0
    for k in range(3):
        a = math.pi / 2 + k * 2 * math.pi / 3
        foot = (spread * math.cos(a), spread * math.sin(a), 0.0)
        p.cyl(foot, (0.03 * math.cos(a), 0.03 * math.sin(a), h - 0.08), 0.014, M["black"], 8)
        p.cyl((foot[0] * 0.45, foot[1] * 0.45, h * 0.3), (0.0, 0.0, h * 0.35), 0.006, M["alu"], 6)
        p.cyl(foot, (foot[0], foot[1], 0.02), 0.02, M["rubber"], 8)
    p.cyl((0, 0, h - 0.1), (0, 0, h), 0.05, M["black"], 14)


def _light_stand(p, M, h):
    """A folding light stand (three splayed legs, a telescopic column) up to height h."""
    p.r = 0.0
    for k in range(3):
        a = k * 2 * math.pi / 3
        p.cyl((0.4 * math.cos(a), 0.4 * math.sin(a), 0.0), (0.0, 0.0, 0.45), 0.01, M["black"], 6)
    p.cyl((0, 0, 0.0), (0, 0, h * 0.55), 0.018, M["black"], 8)
    p.cyl((0, 0, h * 0.55), (0, 0, h), 0.013, M["alu"], 8)


def _camera_body(p, M, z, lens=0.35, prompter=False, cv=None):
    """A broadcast camera on its head at height z, its lens along +y: body, lens with hood,
    viewfinder on top, pan bars back; a teleprompter hood over the lens if asked."""
    body = _paint("Med_CamBody", "#2a2c30", 0.45)
    p.r = 0.01
    p.box((-0.07, -0.12, z + 0.02), (0.07, 0.04, z + 0.06), M["black"])
    p.box((-0.08, -0.25, z + 0.06), (0.08, 0.15, z + 0.26), body)
    p.r = 0.0
    p.cyl((0, 0.15, z + 0.16), (0, 0.15 + lens, z + 0.16), 0.06, M["black"], 16, r2=0.065)
    p.cyl((0, 0.15 + lens, z + 0.16), (0, 0.15 + lens + 0.06, z + 0.16), 0.075, M["black"], 16, r2=0.085)
    p.r = 0.006
    p.box((-0.07, -0.05, z + 0.26), (0.07, 0.12, z + 0.36), M["black"])
    p.r = 0.0
    p.box((-0.065, -0.052, z + 0.27), (0.065, -0.051, z + 0.35), M["glass"])
    for s in (-1, 1):
        _tube(p, [(s * 0.05, -0.15, z + 0.04), (s * 0.25, -0.6, z - 0.05), (s * 0.3, -0.75, z - 0.1)], 0.012, M["alu"], 6)
        p.cyl((s * 0.3, -0.75, z - 0.1), (s * 0.3, -0.85, z - 0.13), 0.02, M["black"], 8)
    p.cyl((0.0, -0.25, z + 0.24), (0.1, -0.35, z + 0.4), 0.005, M["black"], 5)
    if prompter:
        p.r = 0.006
        p.box((-0.22, 0.15 + lens * 0.4, z + 0.0), (0.22, 0.15 + lens + 0.12, z + 0.04), M["black"])
        p.box((-0.22, 0.15 + lens + 0.08, z + 0.04), (0.22, 0.15 + lens + 0.12, z + 0.42), M["black"])
        p.box((-0.22, 0.15 + lens * 0.4, z + 0.38), (0.22, 0.15 + lens + 0.12, z + 0.42), M["black"])
        p.r = 0.0
        p.face([(-0.2, 0.15 + lens * 0.4, z + 0.05), (0.2, 0.15 + lens * 0.4, z + 0.05), (0.2, 0.15 + lens + 0.08, z + 0.37), (-0.2, 0.15 + lens + 0.08, z + 0.37)], clear_glass())
        if cv is not None:
            r = cv.alloc(0.38, 0.26, 600)
            r.fill(col="#050505", emit="#050505")
            for i in range(6):
                r.text(("GOOD EVENING AND", "WELCOME TO THE", "BUILD-UP. TONIGHT", "THE TEAM NEWS IS", "IN AND THERE ARE", "TWO CHANGES...")[i], 0.19, 0.22 - i * 0.04, 0.025,
                       weight="bold", align="c", col="#f0f0f0", emit="#f0f0f0")
            pm = cv.material("Med_PrompterText", emission=1.0)
            decal(p, (0.0, 0.15 + lens * 0.6, z + 0.03), (0, 0, 1), (0, 1, 0), 0.38, 0.26, pm, r, lift=0.002)


# -- the press room ---------------------------------------------------------------------------------

def press_backdrop(col, M):
    """The press conference backdrop (a 4.0 x 2.4 m step-and-repeat of the club crest and partner
    logos on a free-standing frame with weighted feet). Thin; its print faces +y."""
    cv, r = _sponsor_canvas("Med_Backdrop", 4.0, 2.4, seed=1)
    mat = cv.material("Med_BackdropPrint")
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-2.0, -0.01, 0.15), (2.0, 0.0, 2.55), mat)
    project(p, i0, r, (2.0, 0.0, 0.15), (-1, 0, 0), (0, 0, 1))
    for x in (-2.0, 0.0, 2.0):
        p.cyl((x, -0.03, 0.0), (x, -0.03, 2.6), 0.015, M["alu"], 8)
        p.r = 0.006
        p.box((x - 0.05, -0.35, 0.0), (x + 0.05, 0.3, 0.03), M["black"])
        p.r = 0.0
    p.cyl((-2.0, -0.03, 2.58), (2.0, -0.03, 2.58), 0.012, M["alu"], 8)
    return p.obj("press_backdrop", col)


def roller_banner(col, M):
    """A pull-up roller banner (0.85 x 2.0 m) in the club's colours on its cassette base. Front +y."""
    cv = Canvas("Med_Roller", 512, 1024, col="#ffffff", rough=0.5)
    r = cv.alloc(0.85, 2.0, 400)
    r.fill(col="#ffffff")
    r.rect(0.0, 0.9, 0.85, 2.0, col="#1d4f91")
    r.poly([(0.0, 0.9), (0.85, 1.25), (0.85, 0.9)], col="#c8202a")
    r.poly([(0.3, 1.75), (0.55, 1.75), (0.55, 1.55), (0.425, 1.42), (0.3, 1.55)], col="#ffffff")
    r.text("U", 0.425, 1.56, 0.12, weight="black", align="c", col="#1d4f91")
    r.text("UNITED", 0.425, 1.3, 0.1, weight="black", align="c", col="#ffffff")
    r.text("PRESS", 0.425, 0.6, 0.09, weight="black", align="c", col="#1d4f91")
    r.text("CONFERENCE", 0.425, 0.48, 0.06, weight="bold", align="c", col="#1d4f91")
    mat = cv.material("Med_RollerPrint")
    p = PM(0.004)
    p.r = 0.01
    p.box((-0.44, -0.08, 0.0), (0.44, 0.08, 0.1), M["alu"])
    p.r = 0.0
    i0 = len(p.F)
    p.box((-0.425, -0.003, 0.1), (0.425, 0.003, 2.1), mat)
    project(p, i0, r, (0.425, 0.003, 0.1), (-1, 0, 0), (0, 0, 1))
    p.cyl((0, -0.02, 0.1), (0, -0.02, 2.1), 0.008, M["alu"], 6)
    p.cyl((-0.43, 0.0, 2.1), (0.43, 0.0, 2.1), 0.01, M["alu"], 8)
    for x in (-0.3, 0.3):
        p.cyl((x, -0.1, 0.0), (x, -0.3, 0.0), 0.015, M["alu"], 6)
    return p.obj("roller_banner", col)


def press_table(col, M):
    """The top table (2.4 x 0.75 m under a floor-length skirt, tinted: the club colour): three
    chairs behind it, name plates, a cluster of broadcasters' microphones with flags on a stand
    and recorders laid in front, bottled water and glasses. Front +y (the press)."""
    cv = Canvas("Med_TopTable", 512, 512, col="#ffffff", rough=0.5)
    plates = []
    for t in ("MANAGER", "PLAYER", "MEDIA"):
        r = cv.alloc(0.3, 0.1, 900)
        r.fill(col="#ffffff")
        r.text(t, 0.15, 0.035, 0.035, weight="black", align="c", col="#141a28")
        plates.append(r)
    pmat = cv.material("Med_NamePlates")
    scv = Canvas("Med_Skirt", 256, 256, col="#ffffff", rough=0.8)
    sr = scv.alloc(1.0, 1.0, 100)
    sr.fill(col="#ffffff")
    sr.field(lambda X, Y: (np.sin(X * 60.0) * 0.5 + 0.5) * 0.12, col="#909090")
    skirt = scv.tint_material("TINT_club_skirt", "#1d4f91", rough=0.8)
    p = PM(0.004)
    p.r = 0.006
    p.box((-1.2, -0.37, 0.72), (1.2, 0.37, 0.75), skirt)
    p.r = 0.0
    p.box((-1.2, 0.37, 0.0), (1.2, 0.375, 0.75), skirt)
    for x in (-1.2, 1.195):
        p.box((x, -0.37, 0.0), (x + 0.005, 0.37, 0.75), skirt)
    for j, x in enumerate((-0.8, 0.0, 0.8)):
        p.r = 0.0
        p.face([(x - 0.15, 0.2, 0.85), (x + 0.15, 0.2, 0.85), (x + 0.15, 0.25, 0.75), (x - 0.15, 0.25, 0.75)], _white("Med_Plate"))
        decal(p, (x, 0.229, 0.8), (0, math.cos(math.radians(27)), math.sin(math.radians(27))), (0, -math.sin(math.radians(27)), math.cos(math.radians(27))), 0.3, 0.1, pmat, plates[j], lift=0.002)
        p.cyl((x + 0.25, -0.1, 0.75), (x + 0.25, -0.1, 0.97), 0.033, clear_glass(), 10)
        p.cyl((x + 0.25, -0.1, 0.97), (x + 0.25, -0.1, 1.0), 0.015, M["blue"], 8)
        p.cyl((x - 0.25, -0.05, 0.75), (x - 0.25, -0.05, 0.85), 0.035, clear_glass(), 10)
        _place(p, _press_chair(M), x, -0.6, 0.0)
    # microphone cluster on a stand in the middle, mic flags of the broadcasters
    p.r = 0.0
    p.cyl((0.0, 0.05, 0.75), (0.0, 0.05, 0.77), 0.08, M["black"], 14)
    rng = np.random.default_rng(3)
    for k in range(7):
        a = -0.6 + k * 0.2
        tip = (0.25 * math.sin(a), 0.05 - 0.22, 0.95 + 0.04 * (k % 2))
        p.cyl((0.0, 0.05, 0.77), tip, 0.005, M["black"], 5)
        p.sphere(tip, 0.025, M["black"], 8, 6)
        c = str(rng.choice(["#c8202a", "#2457a6", "#e3a020", "#2a8a4a", "#f2f2f2", "#7a2a9a"]))
        p.r = 0.004
        p.box((tip[0] - 0.03, tip[1] + 0.03, tip[2] - 0.09), (tip[0] + 0.03, tip[1] + 0.07, tip[2] - 0.04), _plastic("Med_Flag" + c[1:], c, 0.4))
        p.r = 0.0
    for k in range(4):
        p.r = 0.004
        p.obox((-0.35 + k * 0.22, 0.28, 0.76), (0.05, 0.12, 0.02), M["black"], yaw=0.3 * (k - 1.5))
    return p.obj("press_table", col)


def _press_chair(M):
    """A conference chair facing +y: a black frame, a padded seat and back."""
    p = PM(0.004)
    p.r = 0.0
    for x in (-0.22, 0.22):
        _tube(p, [(x, 0.22, 0.0), (x, 0.2, 0.45), (x, -0.22, 0.45), (x, -0.24, 0.0)], 0.011, M["black"], 6, bend=0.03)
        _tube(p, [(x, -0.2, 0.45), (x, -0.25, 0.92)], 0.011, M["black"], 6)
    p.r = 0.03
    p.box((-0.24, -0.21, 0.45), (0.24, 0.23, 0.51), _plastic("Med_PressSeat", "#2a2c30", 0.6))
    p.obox((0, -0.24, 0.74), (0.46, 0.05, 0.36), _plastic("Med_PressSeat", "#2a2c30", 0.6), pitch=math.radians(-8))
    return p


def press_seating(col, M):
    """A row of six linked conference chairs for the press (3.3 m), notepads and a recorder on
    the odd seat, a laptop bag at the end. Long axis along x; the seats face +y."""
    p = PM(0.004)
    for k in range(6):
        x = -1.375 + k * 0.55
        _place(p, _press_chair(M), x, 0.0, 0.0)
        if k in (1, 4):
            p.r = 0.002
            p.obox((x + 0.05, 0.0, 0.512), (0.15, 0.21, 0.01), _paint("Med_Notepad", "#f2e98a", 0.6), yaw=0.2)
    p.r = 0.03
    p.box((1.75, -0.1, 0.0), (2.05, 0.05, 0.3), _plastic("Med_LaptopBag", "#1a1c1f", 0.7))
    return p.obj("press_seating", col)


# -- cameras and lights -----------------------------------------------------------------------------

def tv_camera(col, M):
    """A broadcast ENG camera (shoulder-mount class) on a fluid head and carbon tripod with
    spreader, its pan bar back, the lens to +y at 1.5 m. Front +y (the lens)."""
    p = PM(0.004)
    _tripod(p, M, 1.38)
    _camera_body(p, M, 1.38, lens=0.22)
    return p.obj("tv_camera", col)


def studio_camera(col, M):
    """A studio camera on a pneumatic pedestal: a steering ring over a three-wheeled skirted base,
    the column, a box lens, a large viewfinder, pan bars with zoom and focus demands, a
    teleprompter hood with its glass and the script reflected. Front +y (the lens)."""
    cv = Canvas("Med_Studio", 512, 512, col="#000000", rough=0.4)
    p = PM(0.004)
    p.r = 0.02
    p.cyl((0, 0, 0.06), (0, 0, 0.22), 0.42, _paint("Med_PedBase", "#3a3d42", 0.4), 3)
    for k in range(3):
        a = k * 2 * math.pi / 3 + math.pi / 2
        p.r = 0.0
        tyre(p, (0.36 * math.cos(a), 0.36 * math.sin(a), 0.06), 0.06, 0.05, 0.03, M["rubber"], M["grey"], sides=14, axis="x")
    p.cyl((0, 0, 0.22), (0, 0, 1.15), 0.08, _paint("Med_PedCol", "#5a5e64", 0.35), 16)
    _tube(p, [(0.4 * math.cos(t), 0.4 * math.sin(t), 0.95) for t in np.linspace(0, 2 * math.pi, 25)], 0.018, M["black"], 8, closed=True)
    for k in range(3):
        a = k * 2 * math.pi / 3
        p.cyl((0.08 * math.cos(a), 0.08 * math.sin(a), 0.9), (0.4 * math.cos(a), 0.4 * math.sin(a), 0.95), 0.01, M["black"], 6)
    _camera_body(p, M, 1.2, lens=0.45, prompter=True, cv=cv)
    p.r = 0.006
    p.box((-0.15, -0.22, 1.46), (0.15, 0.0, 1.66), M["black"])
    return p.obj("studio_camera", col)


def softbox_light(col, M):
    """A studio softbox (0.9 x 0.6 m, a bi-colour LED head behind its diffuser) on a stand at
    2.0 m, tilted down. Front +y (the light)."""
    p = PM(0.004)
    _light_stand(p, M, 1.9)
    tilt = math.radians(20)
    p.r = 0.004
    p.obox((0, -0.05, 2.0), (0.92, 0.32, 0.62), M["black"], pitch=-tilt)
    n = (0.0, math.cos(tilt), -math.sin(tilt))
    c = (0.0, -0.05 + 0.161 * math.cos(tilt), 2.0 - 0.161 * math.sin(tilt))
    p.r = 0.0
    p.face([(c[0] - 0.44, c[1] + 0.29 * math.sin(tilt), c[2] + 0.29 * math.cos(tilt)), (c[0] + 0.44, c[1] + 0.29 * math.sin(tilt), c[2] + 0.29 * math.cos(tilt)),
            (c[0] + 0.44, c[1] - 0.29 * math.sin(tilt), c[2] - 0.29 * math.cos(tilt)), (c[0] - 0.44, c[1] - 0.29 * math.sin(tilt), c[2] - 0.29 * math.cos(tilt))],
           _plastic("Med_Diffuser", "#fffaf0", 0.2))
    return p.obj("softbox_light", col)


def lighting_grid(col, M):
    """A studio lighting grid (3.6 x 2.4 m of box truss, hung 2.55 m up from the soffit on wires):
    six Fresnel lanterns on hook clamps with barn doors, their cables along the truss. Front +y."""
    truss = _paint("Med_Truss", "#2a2c30", 0.4)
    p = PM(0.004)
    z = 2.55
    p.r = 0.0
    for y in (-1.2, 1.2):
        for dz in (0.0, 0.2):
            for dy in (-0.1, 0.1):
                p.cyl((-1.8, y + dy, z + dz), (1.8, y + dy, z + dz), 0.012, truss, 6)
        for k in range(19):
            x = -1.8 + k * 0.2
            p.cyl((x, y - 0.1, z), (x + 0.1, y + 0.1, z + 0.2), 0.005, truss, 4)
    for x in (-1.8, 1.8):
        for dz in (0.0, 0.2):
            p.cyl((x, -1.2, z + dz), (x, 1.2, z + dz), 0.012, truss, 6)
    for x in (-1.8, 1.8):
        for y in (-1.2, 1.2):
            p.cyl((x, y, z + 0.2), (x, y, 3.3), 0.002, M["chrome"], 4)
    for x in (-1.2, 0.0, 1.2):
        for y in (-1.2, 1.2):
            p.r = 0.0
            p.cyl((x, y, z), (x, y, z - 0.12), 0.01, M["black"], 6)
            aim = (0.0, -0.35 if y > 0 else 0.35, -0.6)
            ln = math.sqrt(aim[1] ** 2 + aim[2] ** 2)
            d = (0.0, aim[1] / ln, aim[2] / ln)
            a = (x, y, z - 0.18)
            b = (x + d[0] * 0.3, y + d[1] * 0.3, z - 0.18 + d[2] * 0.3)
            p.cyl(a, b, 0.11, M["black"], 14, r2=0.12)
            p.cyl(b, (b[0], b[1] + d[1] * 0.005, b[2] + d[2] * 0.005), 0.1, _plastic("Med_Fresnel", "#fff3dc", 0.2), 14)
    return p.obj("lighting_grid", col)


# -- the studio set and the gallery -----------------------------------------------------------------

def studio_backdrop(col, M):
    """The studio's curved LED backdrop (4.8 m across, 2.6 m high in five panels on a shallow arc,
    showing the club's matchday graphics) on its black frame. The back against the wall at y = 0;
    front +y."""
    big = Canvas("Med_LedBack2", 2048, 1024, col="#050505", rough=0.3)
    whole = big.alloc(4.8, 2.4, 200)
    _display(whole, "graphics", 7)
    mat = big.material("Med_LedBackScreen", emission=1.3)
    p = PM(0.004)
    R = 6.0
    for k in range(5):
        a0, a1 = (k - 2.5) * 0.96 / R, (k - 1.5) * 0.96 / R
        x0, y0 = R * math.sin(a0), R - R * math.cos(a0) + 0.15
        x1, y1 = R * math.sin(a1), R - R * math.cos(a1) + 0.15
        p.r = 0.0
        # (seen from the front the image reads right to left along +x)
        ua, ub = 4.8 - (k + 1) * 0.96, 4.8 - k * 0.96
        p.face([(x1, y1, 0.2), (x0, y0, 0.2), (x0, y0, 2.6), (x1, y1, 2.6)], mat,
               [whole.uv(ua, 0.0), whole.uv(ub, 0.0), whole.uv(ub, 2.4), whole.uv(ua, 2.4)])
        p.r = 0.004
        p.box((x0 - 0.03, y0 - 0.12, 0.0), (x0 + 0.03, y0 - 0.02, 2.65), M["black"])
    p.box((-2.45, 0.0, 0.0), (2.45, 0.3, 0.2), M["black"])
    return p.obj("studio_backdrop", col)


def studio_desk(col, M):
    """A presenter desk (2.4 m, a curved front of back-lit acrylic with the club's colour band,
    tinted, an oak top): three high chairs behind it, tablets, notes, glasses of water. Front +y
    (the cameras); the presenters sit at -y."""
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    glow = _plastic("TINT_club_desk", "#1d4f91", 0.3)
    p = PM(0.006)
    pts = [(1.2 * math.sin(t), 0.25 * math.cos(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, 13)]
    for k in range(len(pts) - 1):
        (xa, ya), (xb, yb) = pts[k], pts[k + 1]
        p.r = 0.0
        p.face([(xa, ya, 0.05), (xb, yb, 0.05), (xb, yb, 0.98), (xa, ya, 0.98)][::-1], _plastic("Med_Acrylic", "#e8eef4", 0.2))
        p.face([(xa, ya + 0.002, 0.4), (xb, yb + 0.002, 0.4), (xb, yb + 0.002, 0.55), (xa, ya + 0.002, 0.55)][::-1], glow)
    p.r = 0.01
    p.box((-1.25, -0.4, 0.98), (1.25, 0.3, 1.02), oak)
    p.box((-1.2, -0.4, 0.0), (1.2, -0.35, 0.98), M["black"])
    for j, x in enumerate((-0.75, 0.0, 0.75)):
        p.r = 0.003
        p.obox((x + 0.1, -0.15, 1.03), (0.24, 0.17, 0.01), M["black"], yaw=0.1)
        p.r = 0.0
        p.cyl((x - 0.2, 0.0, 1.02), (x - 0.2, 0.0, 1.14), 0.03, clear_glass(), 10)
        c = PM(0.004)
        c.cyl((0, 0, 0.0), (0, 0, 0.02), 0.24, M["black"], 16)
        c.cyl((0, 0, 0.02), (0, 0, 0.72), 0.03, M["chrome"], 10)
        _tube(c, [(0.2 * math.cos(t), 0.2 * math.sin(t), 0.3) for t in np.linspace(0, 2 * math.pi, 17)], 0.008, M["chrome"], 6, closed=True)
        c.r = 0.04
        c.box((-0.22, -0.2, 0.72), (0.22, 0.22, 0.8), _plastic("Off_Leather", "#1d1e20", 0.45))
        c.obox((0, -0.22, 1.0), (0.42, 0.05, 0.35), _plastic("Off_Leather", "#1d1e20", 0.45), pitch=math.radians(-8))
        _place(p, c, x, -0.75, 0.0)
    return p.obj("studio_desk", col)


def multiview_wall(col, M):
    """The gallery's monitor wall (3.0 x 1.4 m): two 55 inch multiviewers of camera feeds with
    tally borders and a programme and preview monitor pair, on a black frame from 1.0 m. The back
    against the wall at y = 0; front +y."""
    cv = Canvas("Med_Multiview", 2048, 1024, col="#050505", rough=0.3)
    regs = []
    for k in range(4):
        r = cv.alloc(1.2, 0.68, 330)
        r.fill(col="#0a0a0a", emit="#0a0a0a")
        for i in range(2):
            for j in range(2):
                x0, y0 = 0.01 + i * 0.6, 0.01 + j * 0.34
                sub_kind = ("camfeed", "match", "camfeed", "graphics")[(i + j * 2 + k) % 4]
                if sub_kind == "match":
                    from .props_office import _broadcast
                    _broadcast(r, x0, y0 + 0.03, x0 + 0.58, y0 + 0.32, 5 + k * 4 + i + j)
                else:
                    r.rect(x0, y0 + 0.03, x0 + 0.58, y0 + 0.32, col="#1d4f91" if sub_kind == "graphics" else "#0a1530", emit="#1d4f91" if sub_kind == "graphics" else "#0a1530")
                    r.circle(x0 + 0.22, y0 + 0.2, 0.04, col="#c89a78", emit="#c89a78")
                    r.circle(x0 + 0.36, y0 + 0.2, 0.04, col="#c89a78", emit="#c89a78")
                    r.rect(x0 + 0.1, y0 + 0.03, x0 + 0.48, y0 + 0.12, col="#e8eaee", emit="#e8eaee")
                tc = "#e0302a" if (i + j + k) % 5 == 0 else ("#30d040" if (i + j + k) % 5 == 1 else "#3a3d42")
                r.frame(x0, y0 + 0.03, x0 + 0.58, y0 + 0.32, 0.008, col=tc, emit=tc)
                r.rect(x0, y0, x0 + 0.58, y0 + 0.03, col="#202020", emit="#202020")
                r.text(f"CAM {i + j * 2 + 1 + k * 4}", x0 + 0.29, y0 + 0.006, 0.02, weight="bold", align="c", col="#f0f0f0", emit="#f0f0f0")
        regs.append(r)
    mat = cv.material("Med_MultiviewScreens", emission=1.2)
    p = PM(0.004)
    p.r = 0.004
    for x in (-1.5, 0.0, 1.5):
        p.box((x - 0.03, 0.0, 0.95), (x + 0.03, 0.05, 2.45), M["black"])
    for k, (cx, cz, w, h) in enumerate(((-0.75, 2.05, 1.21, 0.68), (0.75, 2.05, 1.21, 0.68), (-0.75, 1.35, 1.21, 0.68), (0.75, 1.35, 1.21, 0.68))):
        p.r = 0.002
        p.box((cx - w / 2 - 0.01, 0.05, cz - h / 2 - 0.01), (cx + w / 2 + 0.01, 0.09, cz + h / 2 + 0.01), M["black"])
        p.r = 0.0
        decal(p, (cx, 0.09, cz), (0, 1, 0), (0, 0, 1), w - 0.005, h - 0.005, mat, regs[k], lift=0.001)
    return p.obj("multiview_wall", col)


def gallery_desk(col, M):
    """The gallery production desk (3.0 x 0.9 m, a raked black console): a vision mixer of lit
    crosspoint buttons and a T-bar, an audio desk of faders, talkback panels, small preview
    monitors, keyboards; three task chairs. Front +y (the operators' side)."""
    cv = Canvas("Med_Gallery", 1024, 512, col="#101010", rough=0.4)
    vm = cv.alloc(0.9, 0.35, 700)
    vm.fill(col="#1a1c20")
    rng = np.random.default_rng(23)
    for row in range(4):
        for k in range(16):
            x, y = 0.04 + k * 0.05, 0.04 + row * 0.075
            c = "#e0302a" if (row == 0 and k == 3) else ("#30d040" if (row == 1 and k == 5) else ("#e0b040" if rng.random() < 0.1 else "#3a3d42"))
            vm.rect(x, y, x + 0.035, y + 0.05, r=0.004, col=c, emit=c if c != "#3a3d42" else None)
    ad = cv.alloc(0.9, 0.35, 700)
    ad.fill(col="#1c1e22")
    for k in range(16):
        x = 0.05 + k * 0.052
        ad.rect(x + 0.012, 0.03, x + 0.018, 0.22, col="#050505")
        y = 0.05 + 0.15 * rng.random()
        ad.rect(x, y, x + 0.03, y + 0.025, r=0.003, col="#d8d8d8")
        for j in range(5):
            ad.circle(x + 0.015, 0.25 + j * 0.02, 0.006, col="#5a5e64")
    vmat = cv.material("Med_GalleryPanels", emission=0.8)
    mons = []
    for k in range(3):
        r = cv.alloc(0.4, 0.23, 600)
        _display(r, ("camfeed", "match", "edit")[k], 30 + k)
        mons.append(r)
    mmat = cv.material("Med_GalleryMonitors", emission=1.0)
    body = _paint("Med_Console", "#1e2024", 0.5)
    p = PM(0.006)
    p.r = 0.01
    p.box((-1.5, -0.45, 0.0), (1.5, 0.45, 0.72), body)
    p.r = 0.0
    tilt = math.radians(12)
    for k, (x, reg) in enumerate(((-0.75, vm), (0.75, ad))):
        p.r = 0.0
        # (raked: low at the operators' edge, rising away from them)
        p.obox((x, 0.05, 0.75), (0.95, 0.75, 0.04), body, pitch=-tilt)
        decal(p, (x, 0.05, 0.772), (0, math.sin(tilt), math.cos(tilt)), (0, -math.cos(tilt), math.sin(tilt)), 0.9, 0.35, vmat, reg, lift=0.002)
    p.r = 0.0
    p.cyl((-0.15, 0.25, 0.77), (-0.1, 0.25, 0.92), 0.01, M["chrome"], 8)
    for k, x in enumerate((-1.0, 0.0, 1.0)):
        _monitor(p, (x, -0.3, 0.8), 0.0, 0.4, 0.23, mmat, mons[k], M)
        _place(p, _task_chair(M), x, 0.85, math.pi)
    return p.obj("gallery_desk", col)


# -- edit, podcast, content ---------------------------------------------------------------------------

def edit_desk(col, M):
    """An edit suite workstation (1.8 x 0.8 m, black sit-stand desk): two 32 inch monitors on an arm
    (the edit timeline and the playback), a pair of nearfield speakers on isolation pads, a colour
    grading panel with trackballs, a jog-shuttle, headphones, a drive dock with drives; a task
    chair. Front +y (the editor's side)."""
    cv = Canvas("Med_Edit", 1024, 1024, col="#202020", rough=0.4)
    top = _paint("Med_EditTop", "#1e2024", 0.4)
    frame = _paint("Off_FrameBlack", "#25272a", 0.4)
    p = PM(0.004)
    _desk_frame(p, M, 1.8, 0.8, top, frame, pedestal=False, sitstand=True)
    _arm_monitors(p, M, ("edit", "camfeed"), cv, 41, w=0.71, h=0.4, y=-0.24)
    for s in (-1, 1):
        x = s * 0.78
        p.r = 0.004
        p.box((x - 0.09, -0.32, 0.745), (x + 0.09, -0.14, 0.77), _plastic("Med_IsoPad", "#3a3d42", 0.9))
        p.obox((x, -0.23, 0.95), (0.17, 0.2, 0.36), M["black"], yaw=-s * 0.3)
        p.r = 0.0
        p.cyl((x - s * 0.03, -0.12, 0.88), (x - s * 0.03, -0.11, 0.88), 0.06, _plastic("Med_Cone", "#d8d8d4", 0.4), 16)
        p.cyl((x - s * 0.03, -0.12, 1.05), (x - s * 0.03, -0.11, 1.05), 0.02, M["chrome"], 10)
    _keyboard(p, M, -0.2, 0.15, 0.745, 0.4)
    p.r = 0.008
    p.box((0.2, -0.02, 0.745), (0.62, 0.22, 0.79), M["black"])
    for k in range(3):
        p.r = 0.0
        p.sphere((0.27 + k * 0.14, 0.1, 0.79), 0.035, M["grey"], 10, 6, z_min=0.0)
    p.r = 0.006
    p.box((-0.7, 0.0, 0.745), (-0.5, 0.2, 0.8), M["black"])
    p.r = 0.002
    for k in range(3):
        p.box((-0.68 + k * 0.06, 0.04, 0.8), (-0.64 + k * 0.06, 0.16, 0.86), _plastic("Med_Drive", "#c8202a" if k == 1 else "#2a2c30", 0.4))
    _place(p, _task_chair(M), -0.05, 0.68, math.pi)
    return p.obj("edit_desk", col)


def podcast_table(col, M):
    """A podcast table (1.2 m round, oak) for three: broadcast microphones on boom arms with pop
    shields, closed-back headphones, a rodecaster-style mixer, a laptop, mugs; three upholstered
    swivel chairs. Front +y."""
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.004)
    p.r = 0.0
    p.cyl((0, 0, 0.0), (0, 0, 0.03), 0.35, M["black"], 20)
    p.cyl((0, 0, 0.03), (0, 0, 0.72), 0.05, M["black"], 14)
    p.cyl((0, 0, 0.72), (0, 0, 0.76), 0.6, oak, 32)
    for k in range(3):
        a = math.pi / 2 + k * 2 * math.pi / 3
        ca, sa = math.cos(a), math.sin(a)
        ex, ey = 0.52 * ca, 0.52 * sa
        p.cyl((ex, ey, 0.76), (ex, ey, 0.85), 0.012, M["black"], 8)
        mid = (0.45 * ca, 0.45 * sa, 1.05)
        mic = (0.35 * ca, 0.35 * sa, 0.98)
        _tube(p, [(ex, ey, 0.85), mid, mic], 0.008, M["black"], 6)
        p.cyl((mic[0], mic[1], mic[2]), (mic[0] - 0.12 * ca, mic[1] - 0.12 * sa, mic[2] - 0.02), 0.025, M["black"], 12)
        p.cyl((mic[0] - 0.15 * ca, mic[1] - 0.15 * sa, mic[2] - 0.02), (mic[0] - 0.155 * ca, mic[1] - 0.155 * sa, mic[2] - 0.02), 0.05, _plastic("Med_PopShield", "#1a1a1a", 0.9), 14)
        hx, hy = 0.25 * ca, 0.25 * sa
        _tube(p, [(hx - 0.08 * sa, hy + 0.08 * ca, 0.78), (hx, hy, 0.86), (hx + 0.08 * sa, hy - 0.08 * ca, 0.78)], 0.01, M["black"], 6, bend=0.05)
        _mug(p, M, 0.4 * math.cos(a + 0.5), 0.4 * math.sin(a + 0.5), 0.76, ("#1d4f91", "#f0f0ee", "#c8202a")[k])
        c = PM(0.004)
        c.cyl((0, 0, 0.0), (0, 0, 0.05), 0.28, M["black"], 16)
        c.cyl((0, 0, 0.05), (0, 0, 0.42), 0.03, M["chrome"], 10)
        c.r = 0.05
        c.box((-0.26, -0.24, 0.42), (0.26, 0.26, 0.52), _plastic("Med_PodChair", "#3a3f48", 0.8))
        c.obox((0, -0.26, 0.78), (0.5, 0.08, 0.42), _plastic("Med_PodChair", "#3a3f48", 0.8), pitch=math.radians(-8))
        _place(p, c, 0.95 * ca, 0.95 * sa, a + math.pi / 2)
    p.r = 0.008
    p.box((-0.15, -0.1, 0.76), (0.15, 0.1, 0.8), M["black"])
    p.r = 0.0
    for k in range(4):
        p.box((-0.12 + k * 0.06, -0.05, 0.8), (-0.1 + k * 0.06, 0.05, 0.81), M["grey"])
    return p.obj("podcast_table", col)


def acoustic_panels(col, M):
    """Acoustic absorber panels on a wall (2.4 x 1.2 m: six fabric-wrapped slabs in two tones,
    tinted, with a gap between, 0.9 m up). The back against the wall at y = 0; front +y."""
    cv = Canvas("Med_AcFabric", 256, 256, col="#ffffff", rough=0.95)
    r = cv.alloc(0.8, 0.6, 300)
    r.fill(col="#ffffff")
    r.field(lambda X, Y: noise(X.shape, 50, 9, 2) * 0.15, col="#9a9a9a")
    fab = cv.tint_material("TINT_upholstery_panels", "#3a4450", rough=0.95)
    dark = _paint("Med_AcDark", "#2a2c30", 0.95)
    p = PM(0.004)
    for i in range(3):
        for j in range(2):
            x, z = -1.2 + i * 0.81, 0.9 + j * 0.61
            p.r = 0.015
            p.box((x, 0.0, z), (x + 0.79, 0.05, z + 0.59), fab if (i + j) % 2 else dark)
    return p.obj("acoustic_panels", col)


def ring_light(col, M):
    """A 19 inch LED ring light on its stand (1.7 m) holding a phone in the middle, a remote and a
    small tripod with a second phone beside it. Front +y (the light)."""
    p = PM(0.004)
    _light_stand(p, M, 1.55)
    p.r = 0.0
    _tube(p, [(0.24 * math.cos(t), 0.0, 1.7 + 0.24 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 33)], 0.025, _plastic("Med_RingLed", "#fffaf0", 0.2), 8, closed=True)
    p.cyl((0, 0, 1.55), (0, 0, 1.7), 0.01, M["black"], 6)
    p.r = 0.004
    p.box((-0.04, 0.0, 1.62), (0.04, 0.02, 1.78), M["black"])
    p.box((-0.035, 0.02, 1.625), (0.035, 0.021, 1.775), M["glass"])
    return p.obj("ring_light", col)


def green_screen(col, M):
    """A collapsible chroma-key screen (3.0 x 2.4 m, green, pulled taut on a frame with stands and
    sandbags) with a fall of green onto the floor in front. Front +y."""
    green = _paint("Med_Chroma", "#2aa84a", 0.9)
    p = PM(0.004)
    p.r = 0.0
    p.box((-1.5, 0.0, 0.05), (1.5, 0.01, 2.5), green)
    p.box((-1.5, 0.0, 0.0), (1.5, 1.2, 0.006), green)
    for x in (-1.55, 1.55):
        _place(p, _stand_pm(M, 2.6), x, -0.15, 0.0)
        p.r = 0.03
        p.box((x - 0.12, -0.35, 0.0), (x + 0.12, -0.15, 0.1), _paint("Med_Sandbag", "#c8202a", 0.8))
    p.r = 0.0
    p.cyl((-1.6, -0.02, 2.52), (1.6, -0.02, 2.52), 0.012, M["alu"], 8)
    return p.obj("green_screen", col)


def _stand_pm(M, h):
    p = PM(0.004)
    _light_stand(p, M, h)
    return p


def flash_interview(col, M):
    """A flash-interview set: a 2.0 x 2.2 m sponsor board on its frame (the interviewee stands in
    front of it), two LED panels on stands either side and a tripod camera facing it 2.2 m out.
    Front +y (the camera's side)."""
    cv, r = _sponsor_canvas("Med_Flash", 2.0, 2.2, seed=3)
    mat = cv.material("Med_FlashPrint")
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-1.0, 0.0, 0.2), (1.0, 0.02, 2.4), mat)
    project(p, i0, r, (1.0, 0.02, 0.2), (-1, 0, 0), (0, 0, 1))
    for x in (-1.0, 1.0):
        p.cyl((x, -0.02, 0.0), (x, -0.02, 2.45), 0.015, M["alu"], 8)
        p.r = 0.006
        p.box((x - 0.05, -0.35, 0.0), (x + 0.05, 0.3, 0.03), M["black"])
        p.r = 0.0
    for s in (-1, 1):
        lp = PM(0.004)
        _light_stand(lp, M, 1.7)
        lp.r = 0.004
        lp.obox((0, 0.0, 1.85), (0.6, 0.05, 0.4), M["black"])
        lp.r = 0.0
        lp.box((-0.28, 0.026, 1.67), (0.28, 0.028, 2.03), _plastic("Med_Diffuser", "#fffaf0", 0.2))
        _place(p, lp, s * 1.3, 1.3, math.pi + s * 0.6)
    cam = PM(0.004)
    _tripod(cam, M, 1.5)
    _camera_body(cam, M, 1.5, lens=0.22)
    _place(p, cam, 0.0, 2.3, math.pi)
    return p.obj("flash_interview", col)


# -- storage and the green room ---------------------------------------------------------------------

def equipment_racking(col, M):
    """Kit-room shelving (1.5 x 0.6 x 2.0 m, black steel): hard flight cases for cameras and
    lenses, lens bags, folded tripods and light stands leaning, cable drums, battery chargers with
    lights. Front +y."""
    steel = _paint("Med_KitSteel", "#2a2c30", 0.45)
    case = _plastic("Med_Case", "#1a1c1f", 0.6)
    p = PM(0.004)
    for x in (-0.74, 0.74):
        for y in (-0.28, 0.28):
            p.r = 0.002
            p.box((x - 0.015, y - 0.015, 0.0), (x + 0.015, y + 0.015, 2.0), steel)
    rng = np.random.default_rng(55)
    for k, z in enumerate((0.1, 0.6, 1.1, 1.6)):
        p.r = 0.002
        p.box((-0.74, -0.28, z), (0.74, 0.28, z + 0.02), steel)
        x = -0.7
        while x < 0.55:
            w = float(rng.choice([0.25, 0.35, 0.45]))
            h = float(rng.uniform(0.18, 0.36))
            p.r = 0.01
            p.box((x, -0.24, z + 0.02), (x + w - 0.02, 0.24, z + 0.02 + h), case if rng.random() < 0.7 else _plastic("Med_CaseY", "#e3a020", 0.5))
            p.r = 0.002
            for dx in (0.05, w - 0.09):
                p.box((x + dx, 0.24, z + 0.02 + h * 0.6), (x + dx + 0.04, 0.255, z + 0.02 + h * 0.75), M["chrome"])
            x += w
    p.r = 0.0
    for k in range(3):
        p.cyl((0.85 + k * 0.05, 0.1, 0.0), (0.82 + k * 0.05, 0.05, 1.3), 0.02, M["black"], 8)
    p.cyl((0.3, 0.0, 1.62), (0.3, 0.0, 1.85), 0.15, _plastic("Med_CableDrum", "#e3a020", 0.5), 18)
    p.r = 0.004
    p.box((-0.6, -0.1, 1.62), (-0.2, 0.15, 1.7), M["black"])
    for k in range(4):
        p.box((-0.58 + k * 0.1, 0.15, 1.66), (-0.55 + k * 0.1, 0.155, 1.68), _plastic("Med_ChargeLed", "#30d040", 0.2))
    return p.obj("equipment_racking", col)


def makeup_station(col, M):
    """A make-up station (1.6 m, the back against the wall at y = 0): a white counter with drawers,
    brushes in pots, palettes and a hairdryer, a hydraulic make-up chair. Front +y."""
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.8, 0.0, 0.0), (0.8, 0.5, 0.75), _white("Med_MakeupUnit"))
    p.box((-0.82, 0.0, 0.75), (0.82, 0.52, 0.78), _paint("Hos_Quartz", "#e8e6e2", 0.15))
    for k in range(3):
        p.box((-0.75 + k * 0.5, 0.5, 0.45), (-0.3 + k * 0.5, 0.51, 0.72), _white("Med_MakeupUnit"))
    p.r = 0.0
    for k in range(3):
        p.cyl((-0.6 + k * 0.08, 0.25, 0.78), (-0.6 + k * 0.08, 0.25, 0.9), 0.03, M["black"], 10)
        for j in range(4):
            p.cyl((-0.6 + k * 0.08 + 0.01 * (j - 1.5), 0.25, 0.85), (-0.6 + k * 0.08 + 0.02 * (j - 1.5), 0.25, 1.0), 0.004, _paint("Med_Brush", "#5a3a28", 0.5), 5)
    p.r = 0.004
    for k in range(2):
        p.box((-0.1 + k * 0.25, 0.15, 0.78), (0.1 + k * 0.25, 0.3, 0.8), M["black"])
    p.r = 0.0
    p.cyl((0.6, 0.3, 0.83), (0.75, 0.3, 0.83), 0.04, M["black"], 12)
    c = PM(0.004)
    c.cyl((0, 0, 0.0), (0, 0, 0.03), 0.28, M["chrome"], 18)
    c.cyl((0, 0, 0.03), (0, 0, 0.55), 0.04, M["chrome"], 12)
    c.r = 0.05
    c.box((-0.26, -0.22, 0.55), (0.26, 0.26, 0.66), M["black"])
    c.obox((0, -0.24, 1.0), (0.48, 0.08, 0.65), M["black"], pitch=math.radians(-12))
    c.r = 0.02
    c.box((-0.12, 0.2, 0.25), (0.12, 0.35, 0.28), M["chrome"])
    _place(p, c, 0.0, 1.0, math.pi)
    return p.obj("makeup_station", col)


def makeup_mirror(col, M):
    """A Hollywood make-up mirror over the station (1.6 x 1.0 m, 1.0 m up) ringed with globe
    bulbs. The back against the wall at y = 0; front +y."""
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.82, 0.0, 0.98), (0.82, 0.04, 2.02), _white("Med_MakeupUnit"))
    p.r = 0.0
    p.box((-0.72, 0.04, 1.08), (0.72, 0.045, 1.92), _mirror_glass())
    bulb = _plastic("Med_Globe", "#fff6e0", 0.1)
    for k in range(7):
        p.sphere((-0.6 + k * 0.2, 0.07, 1.97), 0.03, bulb, 10, 6)
    for k in range(5):
        for s in (-1, 1):
            p.sphere((s * 0.77, 0.07, 1.15 + k * 0.18), 0.03, bulb, 10, 6)
    return p.obj("makeup_mirror", col)


BUILDERS = {
    "prop.press_backdrop": press_backdrop, "prop.roller_banner": roller_banner, "prop.press_table": press_table,
    "prop.press_seating": press_seating, "prop.tv_camera": tv_camera, "prop.studio_camera": studio_camera,
    "prop.softbox_light": softbox_light, "prop.lighting_grid": lighting_grid, "prop.studio_backdrop": studio_backdrop,
    "prop.studio_desk": studio_desk, "prop.multiview_wall": multiview_wall, "prop.gallery_desk": gallery_desk,
    "prop.edit_desk": edit_desk, "prop.podcast_table": podcast_table, "prop.acoustic_panels": acoustic_panels,
    "prop.ring_light": ring_light, "prop.green_screen": green_screen, "prop.flash_interview": flash_interview,
    "prop.equipment_racking": equipment_racking, "prop.makeup_station": makeup_station, "prop.makeup_mirror": makeup_mirror,
}
