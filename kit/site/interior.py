"""Buildings with interiors (mirrors viewer/gen/src/site/interior.rs): the club's medical centre
and scouting building, planned from their rooms outwards and built so they can be seen into with
the roof off.

The plan: a corridor along the building's length with a band of rooms either side (6 m deep at the
front: clinical rooms, offices; the large rooms in the back: a rehab gym, a pool, a match-viewing
room); the entrance and reception mid-front; stairs and a lift at one end of the back band once there is an
upper floor. Which rooms are built comes from the quality tier's programme (catalog/facilities.json,
in priority order: a room is built if it still fits its band); the size option sets the footprint
and storeys, so a building can be big and basic or small and elite. Each room is furnished from
catalog/rooms.json (library props laid out against its walls). The facility's sizes, tiers (a
programme per band and floor, the name on the front, an ambulance bay) and options come from
catalog/facilities.json.

The shell: an outer leaf in the building style's wall material on a plinth, a lining in each room's
finish, 120 mm partitions finished each side for the room it faces, doors (leaves hinged open, lead
for X-ray, a lift's steel doors), windows to the room's glazing type, floors to the room's finish
(vinyl, rubber, tile, carpet, timber), suspended ceilings with LED panels, a hydrotherapy pool sunk
in its floor, a stair to the upper floor. Views: closed (a flat roof behind a parapet with plant and
the name on the front), dollhouse (no roof: the shown floor's walls full height, the floors above
removed) and cutaway (the shown floor's walls cut at 1.3 m). Rooms are reported as sockets.

Building local frame: centre of the footprint, front (the entrance) towards -y.
"""
import math

from . import buildings, kernel, mats
from .model import COVER_Z, LAWN_Z, num, option, text, tier_spec

STOREY = 3.6       # floor to floor
T_EXT = 0.3        # outer leaf 0.28 + lining 0.02
T_INT = 0.12       # partitions
CEIL = 2.7         # suspended ceiling
CUT = 1.3          # cutaway wall height
FLOORS = {"vinyl": "Floor_Vinyl", "rubber": "Floor_Rubber", "tile": "Floor_Tile", "carpet": "Floor_Carpet", "timber": "Site_Wood"}
PAINT = ("#e7e3d7", "#eceeee", "#f1f2f1", "#f4f4f2")       # walls by quality tier (magnolia to bright white)
GLAZING = {"window": (0.9, 2.3), "full": (0.1, 2.6), "high": (1.6, 2.3)}
HALF_PI = math.pi / 2


class Mats:
    """Material slots of one building by key, in first-use order."""

    def __init__(self):
        self.list, self.keys = [], {}

    def tag(self, key, make):
        if key not in self.keys:
            self.keys[key] = len(self.list)
            self.list.append(make())
        return self.keys[key]


class Room:
    __slots__ = ("kind", "spec", "x0", "x1", "y0", "y1", "band", "storey")

    def __init__(self, kind, spec, x0, x1, y0, y1, band, storey):
        self.kind, self.spec, self.x0, self.x1, self.y0, self.y1, self.band, self.storey = kind, spec, x0, x1, y0, y1, band, storey

    def door(self):
        """(width, kind, centre along the corridor wall in local u)."""
        dw = num(self.spec, "door", 0.9)
        dk = text(self.spec, "door_kind", "leaf")
        w = self.x1 - self.x0
        u = 0.0 if dk == "opening" and dw >= 2.0 else -w / 2.0 + 0.35 + dw / 2.0
        return dw, dk, u

    def frame_xy(self):
        """The room frame (origin at the middle of its corridor wall, +y into the room): (x, y, yaw)."""
        cx = (self.x0 + self.x1) / 2.0
        return (cx, self.y1, math.pi) if self.band == "front" else (cx, self.y0, 0.0)

    def depth(self):
        """(the room's depth in use, the depth of the plant space behind its cross wall or 0)."""
        d = self.y1 - self.y0
        md = num(self.spec, "max_d", 0.0)
        if md > 0.0 and d > md + 1.2:
            return md, d - md - T_INT
        return d, 0.0

    def u_to_x(self, u):
        cx = (self.x0 + self.x1) / 2.0
        return cx - u if self.band == "front" else cx + u


def fill(length, depth, kinds, rooms_cat):
    """The rooms of a band: kinds in priority order, each built if it still fits (its min_w along the
    band, its min_d into it), then all stretched to fill the band. [(kind, width)]."""
    chosen, used = [], 0.0
    for k in kinds:
        r = rooms_cat.get(k)
        if not isinstance(r, dict):
            continue
        mw, md = num(r, "min_w", 4.0), num(r, "min_d", 3.0)
        if used + mw <= length + 1e-9 and md <= depth + 1e-9:
            chosen.append(k)
            used += mw
    if not chosen:
        return []
    s = length / used
    return [(k, num(rooms_cat[k], "min_w", 4.0) * s) for k in chosen]


def plan(cat, t, w, d, storeys):
    """Rooms per storey and the corridor: ([Room], (y_c0, y_c1))."""
    rc = cat["rooms"]
    xi0, xi1 = -w / 2.0 + T_EXT, w / 2.0 - T_EXT
    yi0, yi1 = -d / 2.0 + T_EXT, d / 2.0 - T_EXT
    inner_d = yi1 - yi0
    corr = 1.4 if inner_d < 10.0 else 1.8
    df = min(6.0, (inner_d - corr) * 0.6)
    yc0, yc1 = yi0 + df, yi0 + df + corr
    out = []
    for s in range(storeys):
        pre = "" if s == 0 else "upper_"
        front = [k for k in t.get(pre + "front", []) if isinstance(k, str)]
        back = [k for k in t.get(pre + "back", []) if isinstance(k, str)]
        if storeys > 1:
            back = ["stairs", "lift"] + back
        for band, kinds, y0, y1 in (("front", front, yi0, yc0), ("back", back, yc1, yi1)):
            rs = fill(xi1 - xi0, y1 - y0, kinds, rc)
            # (the entrance, a room glazed as one, mid-front)
            if band == "front" and s == 0 and any(text(rc[k], "glazing", "") == "entrance" for k, _ in rs):
                e = next(i for i, (k, _) in enumerate(rs) if text(rc[k], "glazing", "") == "entrance")
                ent = rs.pop(e)
                rs.insert(len(rs) // 2, ent)
            # (room extents are the clear space: half a partition off each shared boundary)
            ry0, ry1 = (y0, y1 - T_INT / 2.0) if band == "front" else (y0 + T_INT / 2.0, y1)
            x = xi0
            for i, (k, rw) in enumerate(rs):
                a = x + (T_INT / 2.0 if i > 0 else 0.0)
                b = x + rw - (T_INT / 2.0 if i + 1 < len(rs) else 0.0)
                out.append(Room(k, rc[k], a, b, ry0, ry1, band, s))
                x += rw
    return out, (yc0, yc1)


def wbox(m, x0, y0, z0, x1, y1, z1, side, top):
    """A wall piece: an axis-aligned box in `side`, its top face (a cut section) in `top`."""
    kernel.box(m, x0, y0, z0, x1, y1, z1, side)
    m.tags[len(m.tags) - 5] = top


def wall(m, axis, c0, c1, a0, a1, zb, zt, openings, side, top):
    """A wall along x (axis 0: thickness c0..c1 in y, length a0..a1 in x) or along y (axis 1), from
    zb to zt, with openings [(a0, a1, z0, z1)] cut through it; pieces under and over an opening keep
    `side` on top unless their top is the wall's (the cut)."""
    def piece(p0, p1, z0, z1):
        if p1 - p0 < 1e-6 or z1 - z0 < 1e-6:
            return
        tp = top if z1 >= zt - 1e-9 else side
        if axis == 0:
            wbox(m, p0, c0, z0, p1, c1, z1, side, tp)
        else:
            wbox(m, c0, p0, z0, c1, p1, z1, side, tp)
    ops = sorted([o for o in openings if o[1] > a0 + 1e-6 and o[0] < a1 - 1e-6], key=lambda o: o[0])
    a = a0
    for o0, o1, oz0, oz1 in ops:
        o0, o1 = max(o0, a0), min(o1, a1)
        piece(a, o0, zb, zt)
        piece(o0, o1, zb, min(oz0, zt))
        piece(o0, o1, max(oz1, zb), zt)
        a = o1
    piece(a, a1, zb, zt)


def build(o, cat, p, tier, style, opts, fac):
    from .facilities import forecourt, place
    t = tier_spec(cat, fac, tier) or {}
    fs = cat["facilities"].get(fac, {})
    sizes = fs.get("sizes") if isinstance(fs.get("sizes"), list) else []
    sv = option(cat, fac, tier, opts, "size")
    si = min(max(int(sv) if isinstance(sv, (int, float)) and not isinstance(sv, bool) else 2, 1), max(len(sizes), 1))
    sz = sizes[si - 1] if sizes else {"w": 28.8, "d": 15.6, "storeys": 1}
    storeys = int(num(sz, "storeys", 1.0))
    pl = place(p, num(sz, "w", 28.8), num(sz, "d", 15.6), 16.0)
    f, w, d = pl.f, pl.w, pl.d
    forecourt(o, p, pl)
    club = o.club
    vv = option(cat, fac, tier, opts, "view")
    view = vv if vv in ("closed", "dollhouse", "cutaway") else "closed"
    fv = option(cat, fac, tier, opts, "floor")
    top = storeys - 1 if view == "closed" else min(max(int(fv) if isinstance(fv, (int, float)) and not isinstance(fv, bool) else storeys - 1, 0), storeys - 1)
    sp = buildings.styled(cat, style, w, d, storeys, club.primary, o.night, opts, p.seed)
    P = buildings.PLINTH
    rooms, (yc0, yc1) = plan(cat, t, w, d, storeys)
    M = Mats()
    q = min(max(tier, 1), 4)
    paint = PAINT[q - 1]

    def floor_tag(kind):
        key = FLOORS.get(kind, "Floor_Vinyl")
        return M.tag("floor:" + key, lambda: mats.named(key))

    def finish_tag(spec, feature=False):
        """The wall finish facing a room; an accent room has one feature wall in the club colour
        (its right-hand side wall, seen from the door), the rest painted."""
        k = text(spec, "walls", "paint")
        if k == "tile":
            return M.tag("wall:tile", lambda: mats.named("Floor_Tile"))
        if k == "accent" and q >= 3 and feature:
            return M.tag("wall:accent", lambda: mats.tint("TINT_interior", club.primary))
        return M.tag("wall:paint", lambda: mats.tint("TINT_interior", paint))

    def room_floor(spec):
        k = text(spec, "floor", "vinyl")
        tf = spec.get("timber_from")
        if isinstance(tf, (int, float)) and not isinstance(tf, bool) and q >= int(tf):
            k = "timber"
        return floor_tag(k)
    t_fac = M.tag("facade", lambda: sp.mats[buildings.T_WALL])
    t_base = M.tag("base", lambda: sp.mats[buildings.T_BASE])
    t_sec = M.tag("section", lambda: mats.tint("TINT_trim", "#2f3236"))
    t_trim = M.tag("trim", lambda: sp.mats[buildings.T_TRIM])
    t_glass = M.tag("glass", lambda: mats.named("Balustrade_Glass"))
    t_ceil = M.tag("ceiling", lambda: mats.named("Ceiling_Grid"))
    t_led = M.tag("led", lambda: mats.glow("TINT_trim", "#f6f8fa", 3.0 if o.night else 0.9))
    t_corr_f = floor_tag("timber" if q >= 4 else "vinyl")
    t_corr_w = M.tag("wall:paint", lambda: mats.tint("TINT_interior", paint))
    t_door = M.tag("door", lambda: mats.tint("TINT_door", club.primary if q >= 3 else "#c9b48a"))
    t_lead = M.tag("door:lead", lambda: mats.tint("TINT_door", "#4a4d52"))
    t_steel = M.tag("steel", lambda: mats.named("Galvanised"))
    t_warn = M.tag("warn", lambda: mats.glow("TINT_trim", "#e0302a", 2.0))
    t_water = M.tag("water", lambda: mats.named("Site_Water"))
    t_tile = M.tag("wall:tile", lambda: mats.named("Floor_Tile"))
    t_accent = M.tag("accent", lambda: sp.mats[buildings.T_ACCENT])
    t_roof = M.tag("roof", lambda: mats.named("Roof_Flat"))
    m = kernel.Mesh()
    xo0, xo1, yo0, yo1 = -w / 2.0, w / 2.0, -d / 2.0, d / 2.0
    xi0, xi1, yi0, yi1 = xo0 + T_EXT, xo1 - T_EXT, yo0 + T_EXT, yo1 - T_EXT

    def zf(s):
        return P + s * STOREY

    def height(s):
        if view == "closed" or s < top:
            return STOREY
        return STOREY - 0.3 if view == "dollhouse" else CUT
    # -- openings in the outside walls: windows per room, the entrance, the corridor ends ---------
    ent = next((r for r in rooms if r.storey == 0 and text(r.spec, "glazing", "") == "entrance"), None)
    ycm = (yc0 + yc1) / 2.0
    for s in range(top + 1):
        z0, h = zf(s), height(s)
        zt = z0 + h
        ops = {0: [], 1: [], 2: [], 3: []}        # front, back (along x); left, right (along y)
        for r in rooms:
            if r.storey != s:
                continue
            g = text(r.spec, "glazing", "window")
            side = 0 if r.band == "front" else 1
            if g == "entrance":
                ops[side].append((((r.x0 + r.x1) / 2.0 - 1.1), ((r.x0 + r.x1) / 2.0 + 1.1), z0, z0 + 2.5))
                for k in (-1.0, 1.0):
                    a = (r.x0 + r.x1) / 2.0 + k * ((r.x1 - r.x0) / 2.0 - 1.4) - 0.8
                    if (r.x1 - r.x0) >= 6.4:
                        ops[side].append((a, a + 1.6, z0 + 0.1, z0 + 2.6))
            elif g in GLAZING and r.depth()[1] == 0.0:
                sill, head = GLAZING[g]
                ww = min(r.x1 - r.x0 - 0.9, 7.2)
                c = (r.x0 + r.x1) / 2.0
                if ww > 0.6:
                    ops[side].append((c - ww / 2.0, c + ww / 2.0, z0 + sill, z0 + head))
        for side, a0 in ((2, 0), (3, 1)):
            if s == 0:
                ops[side].append((ycm - 0.6, ycm + 0.6, z0, z0 + 2.2))
            else:
                ops[side].append((ycm - 0.6, ycm + 0.6, z0 + 0.9, z0 + 2.3))
        # outer leaves (the plinth below the ground floor in the base material)
        if s == 0:
            for axis, c0, c1, a0_, a1_ in ((0, yo0, yo0 + T_EXT - 0.02, xo0, xo1), (0, yo1 - T_EXT + 0.02, yo1, xo0, xo1),
                                           (1, xo0, xo0 + T_EXT - 0.02, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02),
                                           (1, xo1 - T_EXT + 0.02, xo1, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02)):
                wall(m, axis, c0, c1, a0_, a1_, 0.0, P, [], t_base, t_base)
        for side, (axis, c0, c1, a0_, a1_) in enumerate(((0, yo0, yo0 + T_EXT - 0.02, xo0, xo1), (0, yo1 - T_EXT + 0.02, yo1, xo0, xo1),
                                                        (1, xo0, xo0 + T_EXT - 0.02, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02),
                                                        (1, xo1 - T_EXT + 0.02, xo1, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02))):
            wall(m, axis, c0, c1, a0_, a1_, z0, zt, ops[side], t_fac, t_sec)
            # glazing and frames in each opening
            for a, b, oz0, oz1 in ops[side]:
                mid = (c0 + c1) / 2.0
                if axis == 0:
                    kernel.quad(m, (a, mid, oz0), (b, mid, oz0), (b, mid, min(oz1, zt)), (a, mid, min(oz1, zt)), t_glass)
                    for (p0, p1, q0, q1) in ((a, a + 0.06, oz0, min(oz1, zt)), (b - 0.06, b, oz0, min(oz1, zt))):
                        wbox(m, p0, mid - 0.04, q0, p1, mid + 0.04, q1, t_trim, t_trim)
                    if oz1 <= zt:
                        wbox(m, a, mid - 0.04, oz1 - 0.06, b, mid + 0.04, oz1, t_trim, t_trim)
                    if oz0 > z0 + 0.05:
                        wbox(m, a, mid - 0.04, oz0, b, mid + 0.04, oz0 + 0.06, t_trim, t_trim)
                    n = int(math.floor((b - a) / 1.5))
                    for k in range(1, n + 1):
                        x = a + (b - a) * k / (n + 1)
                        wbox(m, x - 0.03, mid - 0.04, oz0, x + 0.03, mid + 0.04, min(oz1, zt), t_trim, t_trim)
                else:
                    kernel.quad(m, (mid, a, oz0), (mid, b, oz0), (mid, b, min(oz1, zt)), (mid, a, min(oz1, zt)), t_glass)
                    for (p0, p1, q0, q1) in ((a, a + 0.06, oz0, min(oz1, zt)), (b - 0.06, b, oz0, min(oz1, zt))):
                        wbox(m, mid - 0.04, p0, q0, mid + 0.04, p1, q1, t_trim, t_trim)
                    if oz1 <= zt:
                        wbox(m, mid - 0.04, a, oz1 - 0.06, mid + 0.04, b, oz1, t_trim, t_trim)
        # linings: each room's finish on its stretch of the outside wall
        for r in rooms:
            if r.storey != s:
                continue
            ft = finish_tag(r.spec)
            side = 0 if r.band == "front" else 1
            c0, c1 = (yi0 - 0.02, yi0) if side == 0 else (yi1, yi1 + 0.02)
            wall(m, 0, c0, c1, r.x0, r.x1, z0, zt, ops[side], ft, t_sec)
            # (a room's right-hand wall, seen from its door, is its x0 side in the front band)
            if r.x0 <= xi0 + 1e-6:
                wall(m, 1, xi0 - 0.02, xi0, r.y0, r.y1, z0, zt, [o_ for o_ in ops[2] if o_[1] > r.y0 and o_[0] < r.y1], finish_tag(r.spec, r.band == "front"), t_sec)
            if r.x1 >= xi1 - 1e-6:
                wall(m, 1, xi1, xi1 + 0.02, r.y0, r.y1, z0, zt, [o_ for o_ in ops[3] if o_[1] > r.y0 and o_[0] < r.y1], finish_tag(r.spec, r.band == "back"), t_sec)
        for x0_, x1_ in ((xi0 - 0.02, xi0), (xi1, xi1 + 0.02)):
            wall(m, 1, x0_, x1_, yc0, yc1, z0, zt, ops[2] if x0_ < 0 else ops[3], t_corr_w, t_sec)
        # partitions between rooms (each half finished for the room it faces) and the corridor walls
        for r in rooms:
            if r.storey != s or r.x1 >= xi1 - 1e-6:
                continue
            nxt = next((n_ for n_ in rooms if n_.storey == s and n_.band == r.band and abs(n_.x0 - r.x1 - T_INT) < 1e-6), None)
            if nxt is None:
                continue
            # (into the outside wall's lining, three-quarters through the corridor wall: never
            # flush with a face that is seen)
            ya, yb = (r.y0 - 0.02, r.y1 + T_INT * 0.75) if r.band == "front" else (r.y0 - T_INT * 0.75, r.y1 + 0.02)
            wall(m, 1, r.x1, r.x1 + T_INT / 2.0, ya, yb, z0, zt, [], finish_tag(r.spec, r.band == "back"), t_sec)
            wall(m, 1, r.x1 + T_INT / 2.0, nxt.x0, ya, yb, z0, zt, [], finish_tag(nxt.spec, nxt.band == "front"), t_sec)
        # cross walls closing deep rooms, mechanical plant behind (an air-handling unit, its duct)
        for r in rooms:
            if r.storey != s:
                continue
            de, dp = r.depth()
            if dp == 0.0:
                continue
            yw0, yw1 = (r.y1 - de - T_INT, r.y1 - de) if r.band == "front" else (r.y0 + de, r.y0 + de + T_INT)
            wall(m, 0, yw0, yw1, r.x0 - T_INT * 0.25, r.x1 + T_INT * 0.25, z0, zt, [], finish_tag(r.spec), t_sec)
            yp = (r.y0 + yw0) / 2.0 if r.band == "front" else (yw1 + r.y1) / 2.0
            xc = (r.x0 + r.x1) / 2.0
            aw = min(r.x1 - r.x0 - 0.8, 2.4)
            if aw > 0.8 and dp > 1.6:
                kernel.box(m, xc - aw / 2.0, yp - 0.5, z0, xc + aw / 2.0, yp + 0.5, z0 + min(1.8, zt - z0 - 0.05), t_steel)
                if zt - z0 > 2.2:
                    kernel.box(m, xc - 0.3, yp - 0.3, z0 + 1.8, xc + 0.3, yp + 0.3, zt, t_steel)
        for band, ycw in (("front", yc0), ("back", yc1)):
            doors = []
            for r in rooms:
                if r.storey != s or r.band != band:
                    continue
                dw, dk, u = r.door()
                x = r.u_to_x(u)
                if dw > 0.0:
                    doors.append((x - dw / 2.0, x + dw / 2.0, z0, z0 + (2.1 if dw < 2.0 else 2.6)))
            rr = [r for r in rooms if r.storey == s and r.band == band]
            for r in rr:
                c0, c1 = (ycw - T_INT / 2.0, ycw) if band == "front" else (ycw, ycw + T_INT / 2.0)
                wall(m, 0, c0, c1, r.x0 - (0.01 if r.x0 <= xi0 + 1e-6 else 0.0), r.x1 + (0.01 if r.x1 >= xi1 - 1e-6 else 0.0),
                     z0, zt, doors, finish_tag(r.spec), t_sec)
            c0, c1 = (ycw, ycw + T_INT / 2.0) if band == "front" else (ycw - T_INT / 2.0, ycw)
            wall(m, 0, c0, c1, xi0 - 0.01, xi1 + 0.01, z0, zt, doors, t_corr_w, t_sec)
        # floors, the corridor, ceilings with LED panels
        zfl = z0 + 0.002
        for r in rooms:
            if r.storey != s:
                continue
            ft = room_floor(r.spec)
            if r.spec.get("pool") is True:
                continue
            if r.kind == "stairs" and s > 0:
                # the stair well: open over the flight, a glass balustrade along its edge
                fx = r.x1 - 1.45 if r.band == "back" else r.x0 + 1.45
                xa, xb = (r.x0, fx) if r.band == "back" else (fx, r.x1)
                kernel.ground(m, kernel.rect(xb - xa, r.y1 - r.y0, (xa + xb) / 2.0, (r.y0 + r.y1) / 2.0), zfl, ft)
                wbox(m, fx - 0.012, r.y0 + 1.4, z0, fx + 0.012, r.y1, z0 + 1.1, t_glass, t_trim)
                continue
            kernel.ground(m, kernel.rect(r.x1 - r.x0, r.y1 - r.y0, (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0), zfl, ft)
        kernel.ground(m, kernel.rect(xi1 - xi0, yc1 - yc0, (xi0 + xi1) / 2.0, ycm), zfl, t_corr_f)
        if view == "closed" or s < top:
            zc = z0 + CEIL
            areas = [(r.x0, r.x1, r.y0, r.y1, r) for r in rooms if r.storey == s and r.depth()[1] == 0.0]
            for r in rooms:
                if r.storey == s and r.depth()[1] > 0.0:
                    de = r.depth()[0]
                    areas.append((r.x0, r.x1, r.y1 - de, r.y1, r) if r.band == "front" else (r.x0, r.x1, r.y0, r.y0 + de, r))
            areas.append((xi0, xi1, yc0, yc1, None))
            for x0_, x1_, y0_, y1_, r in areas:
                if r is not None and r.kind == "stairs":
                    continue
                kernel.quad(m, (x0_, y0_, zc), (x0_, y1_, zc), (x1_, y1_, zc), (x1_, y0_, zc), t_ceil)
                nx, ny = max(int((x1_ - x0_) / 2.4), 1), max(int((y1_ - y0_) / 2.4), 1)
                for i in range(nx):
                    for j in range(ny):
                        cx = x0_ + (x1_ - x0_) * (i + 0.5) / nx
                        cy = y0_ + (y1_ - y0_) * (j + 0.5) / ny
                        kernel.quad(m, (cx - 0.3, cy - 0.3, zc - 0.01), (cx - 0.3, cy + 0.3, zc - 0.01), (cx + 0.3, cy + 0.3, zc - 0.01), (cx + 0.3, cy - 0.3, zc - 0.01), t_led)
        # doors: leaves hinged open into the room, frames; a lead door shut with its warning light;
        # the lift's steel doors shut
        for r in rooms:
            if r.storey != s:
                continue
            dw, dk, u = r.door()
            if dw <= 0.0 or dk == "opening":
                continue
            cx, cy, yaw = r.frame_xy()
            rf = kernel.Frame.rad(cx, cy, yaw)
            dm = kernel.Mesh()
            ua, ub = u - dw / 2.0, u + dw / 2.0
            # (clipped to the cut: nothing stands above the walls)
            def cbox(x0, y0, za, x1, y1, zb, tag):
                if za < zt - 1e-6:
                    kernel.box(dm, x0, y0, za, x1, y1, min(zb, zt), tag)
            for (p0, p1, q0, q1) in ((ua - 0.05, ua, 0.0, 2.15), (ub, ub + 0.05, 0.0, 2.15), (ua - 0.05, ub + 0.05, 2.1, 2.15)):
                cbox(p0, -T_INT - 0.01, z0 + q0, p1, 0.01, z0 + q1, t_trim)
            if dk == "leaf":
                cbox(ua, 0.02, z0 + 0.01, ua + 0.045, 0.02 + dw - 0.02, z0 + 2.09, t_door)
                cbox(ua - 0.005, 0.25, z0 + 1.1, ua + 0.05, 0.35, z0 + 1.6, t_glass)
            else:
                cbox(ua + 0.01, -T_INT / 2.0 - 0.025, z0 + 0.01, ub - 0.01, -T_INT / 2.0 + 0.025, z0 + 2.09, t_lead if dk == "lead" else t_steel)
                if dk == "lead":
                    cbox(u - 0.15, -T_INT - 0.06, z0 + 2.25, u + 0.15, -T_INT - 0.01, z0 + 2.4, t_warn)
            dm.verts = [(rf.pt(v_[0], v_[1])[0], rf.pt(v_[0], v_[1])[1], v_[2]) for v_ in dm.verts]
            m.append(dm)
        # the pool: tiled basin with steps, the water 120 mm below the deck, the floor round it
        for r in rooms:
            if r.storey != s or r.spec.get("pool") is not True:
                continue
            cx, cy, yaw = r.frame_xy()
            rf = kernel.Frame.rad(cx, cy, yaw)
            W, D = r.x1 - r.x0, r.y1 - r.y0
            pu0, pu1, pv0, pv1 = -W / 2.0 + 1.2, W / 2.0 - 1.2, 1.8, D - 1.0
            pm = kernel.Mesh()
            ft = room_floor(r.spec)
            for (a0_, a1_, b0_, b1_) in ((-W / 2.0, W / 2.0, 0.0, pv0), (-W / 2.0, W / 2.0, pv1, D), (-W / 2.0, pu0, pv0, pv1), (pu1, W / 2.0, pv0, pv1)):
                kernel.ground(pm, kernel.rect(a1_ - a0_, b1_ - b0_, (a0_ + a1_) / 2.0, (b0_ + b1_) / 2.0), zfl, ft)
            zb = z0 - 1.3
            kernel.quad(pm, (pu0, pv0, zb), (pu1, pv0, zb), (pu1, pv0, z0), (pu0, pv0, z0), t_tile)
            kernel.quad(pm, (pu1, pv1, zb), (pu0, pv1, zb), (pu0, pv1, z0), (pu1, pv1, z0), t_tile)
            kernel.quad(pm, (pu0, pv1, zb), (pu0, pv0, zb), (pu0, pv0, z0), (pu0, pv1, z0), t_tile)
            kernel.quad(pm, (pu1, pv0, zb), (pu1, pv1, zb), (pu1, pv1, z0), (pu1, pv0, z0), t_tile)
            kernel.ground(pm, kernel.rect(pu1 - pu0, pv1 - pv0, (pu0 + pu1) / 2.0, (pv0 + pv1) / 2.0), zb, t_tile)
            for k in range(4):
                kernel.box(pm, pu1 - 1.2, pv0 + k * 0.3, zb, pu1 - 0.01, pv0 + (k + 1) * 0.3, z0 - 0.3 - k * 0.25, t_tile)
            kernel.ground(pm, kernel.rect(pu1 - pu0, pv1 - pv0, (pu0 + pu1) / 2.0, (pv0 + pv1) / 2.0), z0 - 0.12, t_water)
            pm.verts = [(rf.pt(v_[0], v_[1])[0], rf.pt(v_[0], v_[1])[1], v_[2]) for v_ in pm.verts]
            m.append(pm)
            o.prop_at(f.sub(cx, cy, yaw), "prop.pool_hoist", pu0 + 0.5, pv0 - 0.35, z0, 0.0)
        # stairs: a straight flight up to the next floor along the far side of the stair room, a glass
        # balustrade round the void above
        for r in rooms:
            if r.storey != s or r.kind != "stairs" or s + 1 >= storeys:
                continue
            cx, cy, yaw = r.frame_xy()
            rf = kernel.Frame.rad(cx, cy, yaw)
            W = r.x1 - r.x0
            sm = kernel.Mesh()
            n = 20
            rise, going = STOREY / n, 0.25
            ua, ub = W / 2.0 - 1.35, W / 2.0 - 0.05
            tread = room_floor(r.spec)
            for k in range(n):
                kernel.box(sm, ua, 0.4 + k * going, z0, ub, 0.4 + (k + 1) * going, z0 + (k + 1) * rise, tread)
            sm.verts = [(rf.pt(v_[0], v_[1])[0], rf.pt(v_[0], v_[1])[1], v_[2]) for v_ in sm.verts]
            m.append(sm)
            rail = [(ua - 0.04, 0.4, z0 + 0.9), (ua - 0.04, 0.4 + n * going, z0 + STOREY + 0.9)]
            o.tubes_at(f.sub(cx, cy, yaw), f"{p.id}:stair_rail{s}", [(rail, 0.025)], mats.named("Galvanised"), 8)
        # furniture from the room catalogue
        for r in rooms:
            if r.storey != s:
                continue
            furnish(o, f, r, q, z0, club, view == "cutaway" and s == top)
        # room sockets
        for i, r in enumerate(rooms):
            if r.storey != s:
                continue
            de = r.depth()[0]
            yc = r.y1 - de / 2.0 if r.band == "front" else r.y0 + de / 2.0
            o.socket_at(f, "room", f"{p.id}:{r.kind}:{i}", (r.x0 + r.x1) / 2.0, yc, z0, (0.0, 1.0),
                        [("width", r.x1 - r.x0), ("depth", de), ("storey", float(s))])
    # -- the roof (closed view): a flat roof behind a parapet, plant, the name on the front ---------
    if view == "closed":
        zr = zf(storeys)
        kernel.quad(m, (xo0, yo0, zr), (xo1, yo0, zr), (xo1, yo1, zr), (xo0, yo1, zr), t_roof)
        for axis, c0, c1, a0_, a1_ in ((0, yo0, yo0 + 0.25, xo0, xo1), (0, yo1 - 0.25, yo1, xo0, xo1),
                                       (1, xo0, xo0 + 0.25, yo0 + 0.25, yo1 - 0.25), (1, xo1 - 0.25, xo1, yo0 + 0.25, yo1 - 0.25)):
            wall(m, axis, c0, c1, a0_, a1_, zr, zr + 0.9, [], t_fac, t_trim)
        for k in range(int(max(w / 12.0, 1.0))):
            x = xo0 + 4.0 + k * 12.0
            kernel.box(m, x, yo1 - 4.0, zr, x + 2.4, yo1 - 2.6, zr + 1.3, t_steel)
    # -- outside: entrance canopy, steps and ramp, the name, an ambulance bay (medical) ---------------
    if ent is not None:
        ex = (ent.x0 + ent.x1) / 2.0
        # (the canopy and the name only where the ground floor's front is standing to their height)
        front_up = view == "closed" or top > 0
        if front_up or view == "dollhouse":
            kernel.box(m, ex - 2.4, yo0 - 2.6, P + 2.55, ex + 2.4, yo0, P + 2.8, t_trim)
        for k in range(3):
            kernel.box(m, ex - 1.6, yo0 - 1.0 + k * 0.33, 0.0, ex + 1.6, yo0 - 0.67 + k * 0.33, (k + 1) * 0.15, t_base)
        # the ramp alongside (1:15 up to the entrance level), its kerb and handrails
        ra, rb = ex + 1.6, ex + 1.6 + P * 15.0
        kernel.quad(m, (rb, yo0 - 1.6, 0.0), (rb, yo0 - 0.1, 0.0), (ra, yo0 - 0.1, P), (ra, yo0 - 1.6, P), t_base)
        kernel.quad(m, (rb, yo0 - 1.6, 0.0), (ra, yo0 - 1.6, P), (ra, yo0 - 1.6, 0.0), (rb, yo0 - 1.6, 0.0), t_base)
        rails = [([(rb, yo0 - 1.62, 0.9), (ra, yo0 - 1.62, P + 0.9), (ra - 0.3, yo0 - 1.62, P + 0.9)], 0.022)]
        for x in (ra, (ra + rb) / 2.0, rb):
            zz = P * (rb - x) / (rb - ra)
            rails.append(([(x, yo0 - 1.62, zz), (x, yo0 - 1.62, zz + 0.9)], 0.022))
        o.tubes_at(f, f"{p.id}:ramp_rail", rails, mats.named("Galvanised"), 8)
        sign = text(t, "sign", "")
        name = f"{club.name} {sign}" if sign else club.name
        if view == "closed":
            # (over the canopy on one storey; on two, high on the front above the first-floor windows;
            # not in the open views, where the wall it is fixed to is cut or gone)
            zs, hs = (P + 3.0, 0.42) if storeys == 1 else (zf(storeys) - 0.72, 0.5)
            buildings.letters(m, w, d, 0, name, w / 2.0 + ex, zs, min(w * 0.5, 16.0), t_accent, hs / 7.0)
        if t.get("ambulance_bay") is True:
            pm = kernel.Mesh()
            bx0, by0 = ex + 5.0, yo0 - 12.0
            r_ = kernel.rect(3.6, 8.0, bx0 + 1.8, by0 + 4.0)
            kernel.strip(pm, [r_[0], r_[1], r_[2], r_[3], r_[0]], 0.15, COVER_Z + 0.004, 0)
            for k in range(6):
                y = by0 + 0.6 + k * 1.3
                kernel.strip(pm, [(bx0 + 0.2, y), (bx0 + 3.4, y + 1.0)], 0.12, COVER_Z + 0.004, 0)
            o.part_at(f, f"{p.id}:ambulance_bay", pm, [mats.tint("TINT_road_paint", "#e3b92a")])
    frontage(o, f, w, d, pl.gap, (ent.x0 + ent.x1) / 2.0 if ent is not None else None, t.get("ambulance_bay") is True, q, club,
             text(t, "statue", "") or None)
    o.part_at(f, f"{p.id}:building", m, M.list)


def frontage(o, f, w, d, gap, ex, bay, q, club, statue=None):
    """Street furniture on the forecourt, kept off the entrance steps, ramp and ambulance bay:
    planters along the front, a litter bin by the door and a run of cycle stands (a bicycle at one);
    from quality 2 benches facing out and low path lights lining the walk to the door; from quality
    3 a row of bollards across the forecourt edge and three flagpoles (the club's colours) at a
    corner. Life: blown leaves at a basic one, pigeons from quality 2. The tier's statue (a
    generated prop), if it names one, stands beside the walk facing out."""
    yf, ye = -d / 2.0, -d / 2.0 - gap + 0.5
    clear = []
    if ex is not None:
        clear.append((ex - 2.9, ex + 1.6 + buildings.PLINTH * 15.0 + 0.4))
        if bay:
            clear.append((ex + 4.6, ex + 9.0))

    def free(x, half):
        return all(x + half < a or x - half > b for a, b in clear)
    n = 0
    x = -w / 2.0 + 2.0
    while x <= w / 2.0 - 2.0 + 1e-9:
        if free(x, 1.0):
            o.prop_at(f, "prop.planter", x, yf - 0.9, COVER_Z, 0.0)
            if q >= 2 and n % 2 == 1 and yf - 2.0 > ye + 1.0:
                o.prop_at(f, "prop.bench", x, yf - 2.1, COVER_Z, math.pi)
            n += 1
        x += 6.0
    o.prop_at(f, "prop.bin_litter", ex - 3.4 if ex is not None else 0.0, yf - 1.6, COVER_Z, 0.0, [("TINT_bin", "#2a2c2e")])
    rx = w / 2.0 - 3.5 if free(w / 2.0 - 3.5, 2.6) else -w / 2.0 + 3.5
    o.prop_at(f, "prop.bike_rack", rx, yf - 3.6, COVER_Z, 0.0)
    o.prop_at(f, "genprop.bicycle", rx - 1.85, yf - 3.6, COVER_Z, HALF_PI)
    bx = ex - 3.4 if ex is not None else 0.0
    if q == 1:
        o.prop_at(f, "genprop.leaf_litter", bx + 1.2, yf - 2.3, COVER_Z, 0.6)
    else:
        o.prop_at(f, "genprop.pigeon_pecking", bx - 1.6, yf - 2.6, COVER_Z, 1.3)
        if q >= 3:
            o.prop_at(f, "genprop.pigeon_standing", bx - 2.4, yf - 3.1, COVER_Z, 4.0)
    if statue is not None and ex is not None and free(ex - 6.5, 1.2) and abs(ex - 6.5 - rx) > 3.8:
        o.prop_at(f, statue, ex - 6.5, yf - 5.0, COVER_Z, math.pi)
    if ex is not None:
        if q >= 2:
            for k in range(3):
                y = yf - 3.0 - 4.0 * k
                if y > ye + 0.5:
                    for s in (-1.0, 1.0):
                        o.prop_at(f, "prop.path_light", ex + s * 2.2, y, COVER_Z, 0.0)
        if q >= 3:
            for k in range(-3, 4):
                o.prop_at(f, "prop.bollard", ex + k * 1.6, ye + 0.4, COVER_Z, 0.0)
    if q >= 3:
        side = -1.0 if rx > 0.0 else 1.0
        for i, s in enumerate((-1.0, 0.0, 1.0)):
            col = club.secondary if i == 1 else club.primary
            o.prop_at(f, "prop.flagpole", side * (w / 2.0 - 4.5) + s * 2.5, ye + 2.5, COVER_Z, 0.0, [("TINT_flag", col)])


def furnish(o, f, r, q, z0, club, cut=False):
    """Lay out a room's fixtures (catalog/rooms.json) in its frame."""
    cx, cy, yaw = r.frame_xy()
    rf = f.sub(cx, cy, yaw)
    W, D = r.x1 - r.x0, r.depth()[0]
    for fx in r.spec.get("fixtures", []):
        if not isinstance(fx, dict) or not isinstance(fx.get("key"), str):
            continue
        if num(fx, "min_w", 0.0) > W + 1e-9 or num(fx, "min_d", 0.0) > D + 1e-9 or num(fx, "min_tier", 1.0) > q or num(fx, "max_tier", 9.0) < q or num(fx, "max_w", 999.0) < W - 1e-9:
            continue
        if cut and fx.get("cut") is False:
            continue
        at = text(fx, "at", "centre")
        off = num(fx, "off", 0.0)
        rot = math.radians(num(fx, "rot", 0.0))
        tints = []
        tv = fx.get("tints") if isinstance(fx.get("tints"), dict) else {}
        for slot in sorted(tv):
            c = tv[slot]
            if isinstance(c, str):
                tints.append((slot, club.primary if c == "club" else c))
        if at == "centre":
            fracs = [None]
        elif isinstance(fx.get("repeat"), (int, float)) and not isinstance(fx.get("repeat"), bool):
            span = fx.get("span") if isinstance(fx.get("span"), list) and len(fx.get("span")) == 2 else [0.0, 1.0]
            s0, s1 = float(span[0]), float(span[1])
            ln = (s1 - s0) * (W if at in ("back", "door") else D)
            n = min(int(math.floor(ln / float(fx["repeat"]))), int(num(fx, "max", 99.0)))
            fracs = [s0 + (s1 - s0) * (k + 0.5) / n for k in range(n)]
        else:
            fracs = [num(fx, "along", 0.5)]
        for a in fracs:
            if at == "back":
                u, v, y = (a - 0.5) * W, D - off, math.pi
            elif at == "door":
                u, v, y = (a - 0.5) * W, off, 0.0
            elif at == "left":
                u, v, y = -W / 2.0 + off, a * D, -HALF_PI
            elif at == "right":
                u, v, y = W / 2.0 - off, a * D, HALF_PI
            else:
                u, v, y = (num(fx, "fu", 0.5) - 0.5) * W, num(fx, "fv", 0.5) * D, 0.0
            o.prop_at(rf, fx["key"], u, v, z0 + num(fx, "z", 0.0), y + rot, tints)
