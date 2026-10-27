"""Geometry kernel for the training-ground generators. Plain Python floats (no numpy, no Blender), so
the Rust port (viewer/gen/src/site/kernel.rs) reproduces every vertex bit for bit: same operations
in the same order.

A Mesh is polygons with per-face material tags and per-loop UVs (metres). Generators work in a
plot's local frame (origin at the plot centre, x along its width, y along its depth, z up) and a
Frame moves the result into the site.
"""
import math


class Mesh:
    __slots__ = ("verts", "faces", "tags", "uvs")

    def __init__(self):
        self.verts, self.faces, self.tags, self.uvs = [], [], [], []

    def add_face(self, idx, tag, uv):
        self.faces.append(list(idx))
        self.tags.append(tag)
        self.uvs.extend(uv)

    def add_verts(self, pts):
        base = len(self.verts)
        self.verts.extend(pts)
        return base

    def append(self, other):
        base = len(self.verts)
        self.verts.extend(other.verts)
        for f in other.faces:
            self.faces.append([i + base for i in f])
        self.tags.extend(other.tags)
        self.uvs.extend(other.uvs)


class Frame:
    """A plot's frame: local (u, v) -> site (x, y), rotated by yaw (degrees) about (x, y)."""
    __slots__ = ("x", "y", "yaw", "c", "s")

    def __init__(self, x, y, yaw_deg):
        self.x, self.y = x, y
        self.yaw = math.radians(yaw_deg)
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)

    @staticmethod
    def rad(x, y, yaw):
        """A frame from a yaw in radians."""
        f = Frame(x, y, 0.0)
        f.yaw = yaw
        f.c, f.s = math.cos(yaw), math.sin(yaw)
        return f

    def sub(self, u, v, yaw):
        """A frame inside this one: origin at local (u, v), turned by a further local yaw (radians)."""
        p = self.pt(u, v)
        return Frame.rad(p[0], p[1], self.yaw + yaw)

    def pt(self, u, v):
        return (self.x + u * self.c - v * self.s, self.y + u * self.s + v * self.c)

    def vec(self, u, v):
        return (u * self.c - v * self.s, u * self.s + v * self.c)

    def mesh(self, m):
        m.verts = [(self.x + p[0] * self.c - p[1] * self.s, self.y + p[0] * self.s + p[1] * self.c, p[2]) for p in m.verts]
        return m

    def tubes(self, lines):
        return [([(self.x + p[0] * self.c - p[1] * self.s, self.y + p[0] * self.s + p[1] * self.c, p[2]) for p in pts], r)
                for pts, r in lines]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def box(m, x0, y0, z0, x1, y1, z1, tag):
    """Axis-aligned box, outward quads, metric UVs on the two axes each face spans."""
    base = len(m.verts)
    for z in (z0, z1):
        for y in (y0, y1):
            for x in (x0, x1):
                m.verts.append((x, y, z))
    for f in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
        q = [base + i for i in f]
        pts = [m.verts[i] for i in q]
        n = _cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))
        n = (abs(n[0]), abs(n[1]), abs(n[2]))
        if n[0] >= n[1] and n[0] >= n[2]:
            uv = [(p[1], p[2]) for p in pts]
        elif n[1] >= n[2]:
            uv = [(p[0], p[2]) for p in pts]
        else:
            uv = [(p[0], p[1]) for p in pts]
        m.add_face(q, tag, uv)


def obox(m, cx, cy, z0, w, d, h, yaw, tag):
    """Box of size (w, d, h) standing on z0, centred on (cx, cy), turned by yaw (radians)."""
    sub = Mesh()
    box(sub, -w / 2, -d / 2, z0, w / 2, d / 2, z0 + h, tag)
    c, s = math.cos(yaw), math.sin(yaw)
    sub.verts = [(cx + p[0] * c - p[1] * s, cy + p[0] * s + p[1] * c, p[2]) for p in sub.verts]
    m.append(sub)


def quad(m, a, b, c, d, tag):
    """Planar quad a-b-c-d (counter-clockwise seen from its front); UVs: u along a->b."""
    base = m.add_verts([a, b, c, d])
    m.add_face([base, base + 1, base + 2, base + 3], tag, planar_uv([a, b, c, d]))


def tri(m, a, b, c, tag):
    base = m.add_verts([a, b, c])
    m.add_face([base, base + 1, base + 2], tag, planar_uv([a, b, c]))


def planar_uv(pts):
    """Metric UVs in a polygon's plane: u along its first edge."""
    a = pts[0]
    eu = _sub(pts[1], a)
    lu = _norm(eu)
    if lu < 1e-9:
        return [(0.0, 0.0)] * len(pts)
    eu = (eu[0] / lu, eu[1] / lu, eu[2] / lu)
    w = _sub(pts[-1], a)
    k = _dot(w, eu)
    ev = (w[0] - eu[0] * k, w[1] - eu[1] * k, w[2] - eu[2] * k)
    lv = _norm(ev)
    if lv < 1e-9:
        return [(_dot(_sub(p, a), eu), 0.0) for p in pts]
    ev = (ev[0] / lv, ev[1] / lv, ev[2] / lv)
    return [(_dot(_sub(p, a), eu), _dot(_sub(p, a), ev)) for p in pts]


def area2(pts):
    """Twice the signed area of a 2D polygon (positive = counter-clockwise)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        s += a[0] * b[1] - b[0] * a[1]
    return s


def ccw(pts):
    return list(pts) if area2(pts) >= 0 else list(reversed(pts))


def ground(m, pts, z, tag):
    """One flat polygon facing up (any simple polygon), UVs = site metres."""
    pts = ccw(pts)
    base = m.add_verts([(p[0], p[1], z) for p in pts])
    m.add_face(list(range(base, base + len(pts))), tag, [(p[0], p[1]) for p in pts])


def prism(m, pts, z0, z1, side_tag, top_tag, bottom_tag=None):
    """Vertical extrusion of a 2D polygon: walls, a top, optionally a bottom."""
    pts = ccw(pts)
    n = len(pts)
    s = 0.0
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ln = math.hypot(b[0] - a[0], b[1] - a[1])
        base = m.add_verts([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)])
        m.add_face([base, base + 1, base + 2, base + 3], side_tag, [(s, z0), (s + ln, z0), (s + ln, z1), (s, z1)])
        s += ln
    if top_tag is not None:
        base = m.add_verts([(p[0], p[1], z1) for p in pts])
        m.add_face(list(range(base, base + n)), top_tag, [(p[0], p[1]) for p in pts])
    if bottom_tag is not None:
        base = m.add_verts([(p[0], p[1], z0) for p in reversed(pts)])
        m.add_face(list(range(base, base + n)), bottom_tag, [(p[0], p[1]) for p in reversed(pts)])


def cylinder(m, cx, cy, z0, z1, r, n, tag, cap=True):
    s = 0.0
    step = 2 * math.pi * r / n
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        base = m.add_verts([(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)])
        m.add_face([base, base + 1, base + 2, base + 3], tag, [(s, z0), (s + step, z0), (s + step, z1), (s, z1)])
        s += step
    if cap:
        pts = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
        base = m.add_verts([(p[0], p[1], z1) for p in pts])
        m.add_face(list(range(base, base + n)), tag, [(p[0], p[1]) for p in pts])


def strip(m, path, width, z, tag, z_of=None):
    """Flat ribbon along an open 2D polyline (mitred joins), UVs (along, across) in metres.
    z_of(x, y) overrides the height per vertex."""
    n = len(path)
    if n < 2:
        return
    ns = []
    for i in range(n - 1):
        dx, dy = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        ln = math.hypot(dx, dy)
        ns.append((-dy / ln, dx / ln) if ln > 1e-12 else (0.0, 1.0))
    nrm = [ns[0]] + [None] * (n - 2) + [ns[-1]]
    for i in range(1, n - 1):
        mx, my = ns[i - 1][0] + ns[i][0], ns[i - 1][1] + ns[i][1]
        ml = math.hypot(mx, my)
        mx, my = (mx / ml, my / ml) if ml > 1e-12 else ns[i]
        k = max(mx * ns[i][0] + my * ns[i][1], 0.3)
        nrm[i] = (mx / k, my / k)
    h = width / 2
    base = len(m.verts)
    for i in range(n):
        p = path[i]
        for sgn in (-1.0, 1.0):
            x, y = p[0] + nrm[i][0] * h * sgn, p[1] + nrm[i][1] * h * sgn
            m.verts.append((x, y, z if z_of is None else z_of(x, y)))
    s = 0.0
    for i in range(n - 1):
        ln = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        a = base + 2 * i
        m.add_face([a, a + 2, a + 3, a + 1], tag, [(s, 0.0), (s + ln, 0.0), (s + ln, width), (s, width)])
        s += ln


def polyline_length(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))


def along(pts, spacing, offset=0.5):
    """Points every ~spacing metres along a 2D polyline (evenly divided), with the unit direction at
    each: [(x, y, dx, dy)]. offset 0.5 = centred in each division, 0 = from the start."""
    total = polyline_length(pts)
    if total <= 1e-9:
        return []
    n = max(1, int(total / spacing))
    out = []
    for k in range(n + (1 if offset == 0.0 else 0)):
        t = (k + offset) * total / n
        acc = 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ln = math.hypot(b[0] - a[0], b[1] - a[1])
            if t <= acc + ln or i == len(pts) - 2:
                f = 0.0 if ln <= 1e-12 else min(max((t - acc) / ln, 0.0), 1.0)
                out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                            (b[0] - a[0]) / ln if ln > 1e-12 else 1.0, (b[1] - a[1]) / ln if ln > 1e-12 else 0.0))
                break
            acc += ln
    return out


def rect(w, d, cx=0.0, cy=0.0):
    """Counter-clockwise rectangle corners."""
    return [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2), (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]


def inside(pts, x, y):
    """Point in polygon (even-odd)."""
    c = False
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if (a[1] > y) != (b[1] > y):
            xi = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < xi:
                c = not c
    return c


def seg_hit(o, d, a, b):
    """Ray o + t d against segment a-b: t >= 0 or None."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = d[0] * ey - d[1] * ex
    if abs(den) < 1e-12:
        return None
    wx, wy = a[0] - o[0], a[1] - o[1]
    t = (wx * ey - wy * ex) / den
    u = (wx * d[1] - wy * d[0]) / den
    if t >= 0.0 and -1e-9 <= u <= 1.0 + 1e-9:
        return t
    return None


def dist_to_polyline(p, pts):
    best = math.inf
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        l2 = ex * ex + ey * ey
        t = 0.0 if l2 <= 1e-12 else min(max(((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / l2, 0.0), 1.0)
        best = min(best, math.hypot(a[0] + ex * t - p[0], a[1] + ey * t - p[1]))
    return best
