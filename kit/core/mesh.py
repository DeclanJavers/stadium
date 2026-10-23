"""Pure-numpy geometry: profiles, sweeps, fillets, cord tubes, strips. Blender-free so it is unit-testable."""
import numpy as np


# -- profiles (closed 2D loops, counter-clockwise, in (n, b) frame coordinates) ---------------

def profile_ellipse(width, depth, k=24):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([0.5 * width * np.cos(t), 0.5 * depth * np.sin(t)], 1)


def profile_rounded_rect(width, depth, radius, k_corner=4):
    r = min(radius, 0.5 * width, 0.5 * depth)
    hx, hy = 0.5 * width - r, 0.5 * depth - r
    pts = []
    for cx, cy, a0 in ((hx, hy, 0), (-hx, hy, 90), (-hx, -hy, 180), (hx, -hy, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, k_corner + 1)):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def profile(kind, width, depth, k=24):
    if kind in ("elliptical", "round"):
        return profile_ellipse(width, depth, k)
    if kind in ("square", "rectangular"):
        return profile_rounded_rect(width, depth, 0.008, 3)
    raise ValueError(kind)


# -- paths ------------------------------------------------------------------------------------

def fillet(points, radius, steps=6):
    """Round every interior corner of a polyline with an arc of the given radius."""
    p = np.asarray(points, float)
    if len(p) < 3 or radius <= 0:
        return p
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, b, c = p[i - 1], p[i], p[i + 1]
        d0 = (a - b) / np.linalg.norm(a - b)
        d1 = (c - b) / np.linalg.norm(c - b)
        cosang = np.clip(np.dot(d0, d1), -1, 1)
        ang = np.arccos(cosang)
        if ang > np.pi - 1e-3:  # straight
            out.append(b)
            continue
        t = radius / np.tan(ang / 2)
        t = min(t, 0.45 * np.linalg.norm(a - b), 0.45 * np.linalg.norm(c - b))
        s, e = b + d0 * t, b + d1 * t
        # quadratic Bezier through the corner is a good arc stand-in at these radii
        for u in np.linspace(0, 1, steps):
            out.append((1 - u) ** 2 * s + 2 * (1 - u) * u * b + u ** 2 * e)
    out.append(p[-1])
    return np.array(out)


def resample(points, step):
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, i]) for i in range(p.shape[1])], 1)


def arc(center, radius, a0, a1, step=0.05, z=0.0):
    n = max(3, int(abs(a1 - a0) * radius / step) + 1)
    a = np.linspace(a0, a1, n)
    return np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), np.full(n, z)], 1)


# -- sweep ------------------------------------------------------------------------------------

def sweep(path, prof, binormal_hint=(0, 1, 0), caps=True):
    """Sweep a closed 2D profile along a 3D path.

    Profile x maps to N = B x T, profile y maps to B (the hint projected off the tangent).
    Returns verts (N*K,3), quads (M,4), uvs per loop (M*4,2).
    """
    P = np.asarray(path, float)
    Q = np.asarray(prof, float)
    n, k = len(P), len(Q)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    h = np.asarray(binormal_hint, float)
    B = h - (T @ h)[:, None] * T
    bad = np.linalg.norm(B, axis=1) < 1e-6
    if bad.any():
        alt = np.array([1.0, 0, 0]) if abs(h[0]) < 0.9 else np.array([0, 0, 1.0])
        B[bad] = alt - (T[bad] @ alt)[:, None] * T[bad]
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    N = np.cross(B, T)
    V = P[:, None, :] + Q[None, :, 0, None] * N[:, None, :] + Q[None, :, 1, None] * B[:, None, :]
    V = V.reshape(-1, 3)
    i = np.arange(n - 1)[:, None]
    j = np.arange(k)[None, :]
    a = i * k + j
    b = i * k + (j + 1) % k
    quads = np.stack([a, b, b + k, a + k], -1).reshape(-1, 4)
    # UVs: u around profile, v = arc length
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    uu = np.broadcast_to(j / k, (n - 1, k))
    uu1 = np.broadcast_to((j + 1) / k, (n - 1, k))
    vv = np.broadcast_to(s[:-1, None], (n - 1, k))
    vv1 = np.broadcast_to(s[1:, None], (n - 1, k))
    uvs = np.stack([np.stack([uu, vv], -1), np.stack([uu1, vv], -1),
                    np.stack([uu1, vv1], -1), np.stack([uu, vv1], -1)], 2).reshape(-1, 2)
    if caps:
        V, quads, uvs = _add_caps(V, quads, uvs, P, n, k)
    return V, quads, uvs


def _add_caps(V, quads, uvs, P, n, k):
    """Quad-fan caps (centre, r_j, r_j+1, r_j+2 for even j) so the whole mesh stays (M,4). k must be even."""
    assert k % 2 == 0
    extra_v, extra_f = [], []
    base = len(V)
    for ring, centre, flip in ((0, P[0], True), (n - 1, P[-1], False)):
        ci = base + len(extra_v)
        extra_v.append(centre)
        for j in range(0, k, 2):
            a, b, c = (ring * k + (j + o) % k for o in (0, 1, 2))
            extra_f.append((ci, c, b, a) if flip else (ci, a, b, c))
    V = np.concatenate([V, np.array(extra_v)])
    quads = np.concatenate([quads, np.array(extra_f)])
    uvs = np.concatenate([uvs, np.full((4 * len(extra_f), 2), 0.5)])
    return V, quads, uvs


def tubes(p0, p1, radius, sides=4, extend=None):
    """Independent prisms along segments (for net cords). Returns verts, quads."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d, axis=1, keepdims=True)
    keep = L[:, 0] > 1e-6
    p0, p1, d, L = p0[keep], p1[keep], d[keep], L[keep]
    d /= L
    ext = radius if extend is None else extend
    p0 = p0 - d * ext
    p1 = p1 + d * ext
    ref = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    n1 = np.cross(d, ref)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(d, n1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    off = radius * (np.cos(ang)[None, :, None] * n1[:, None, :] + np.sin(ang)[None, :, None] * n2[:, None, :])
    ring0 = p0[:, None, :] + off
    ring1 = p1[:, None, :] + off
    m = len(p0)
    V = np.concatenate([ring0, ring1], 1).reshape(-1, 3)
    base = (np.arange(m) * 2 * sides)[:, None]
    j = np.arange(sides)[None, :]
    a = base + j
    b = base + (j + 1) % sides
    quads = np.stack([a, b, b + sides, a + sides], -1).reshape(-1, 4)
    return V, quads, keep


def strip(points2d, width, closed=False, z=0.0):
    """Flat ribbon of constant width centred on a 2D polyline (mitred joins).

    Returns verts, quads, uvs (u = arc length m, v = 0..1 across).
    """
    p = np.asarray(points2d, float)[:, :2]
    if closed:
        p = np.concatenate([p, p[:1]])
    d = np.diff(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm_seg = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm = np.zeros_like(p)
    nrm[0] = nrm_seg[0]
    nrm[-1] = nrm_seg[-1]
    if len(p) > 2:
        m = nrm_seg[:-1] + nrm_seg[1:]
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        cos = np.sum(m * nrm_seg[1:], axis=1, keepdims=True)
        nrm[1:-1] = m / np.clip(cos, 0.3, None)
    if closed:
        m = nrm_seg[-1] + nrm_seg[0]
        m /= np.linalg.norm(m)
        m /= max(np.dot(m, nrm_seg[0]), 0.3)
        nrm[0] = nrm[-1] = m
    left = p + nrm * width / 2
    right = p - nrm * width / 2
    n = len(p)
    V = np.zeros((2 * n, 3))
    V[0::2, :2] = right
    V[1::2, :2] = left
    V[:, 2] = z
    i = np.arange(n - 1)
    quads = np.stack([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    uv = np.stack([
        np.stack([s[i], np.zeros(n - 1)], 1), np.stack([s[i + 1], np.zeros(n - 1)], 1),
        np.stack([s[i + 1], np.ones(n - 1)], 1), np.stack([s[i], np.ones(n - 1)], 1)], 1).reshape(-1, 2)
    return V, quads, uv


def disc(center, radius, k=32, z=0.0):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rim = np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), np.full(k, z)], 1)
    V = np.concatenate([[[center[0], center[1], z]], rim])
    j = np.arange(0, k, 2)
    quads = np.stack([np.zeros(len(j), int), 1 + j, 1 + (j + 1) % k, 1 + (j + 2) % k], 1)
    uv = np.tile([[0.5, 0.5]], (4 * len(j), 1))
    return V, quads, uv


def grid(nx, ny, sx, sy, z_fn=None):
    """Regular grid centred at origin, (nx+1)*(ny+1) verts. Returns verts, quads, uvs (metres)."""
    xs = np.linspace(-sx / 2, sx / 2, nx + 1)
    ys = np.linspace(-sy / 2, sy / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X) if z_fn is None else z_fn(X, Y)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    a = i * (ny + 1) + j
    quads = np.stack([a, a + ny + 1, a + ny + 2, a + 1], -1).reshape(-1, 4)
    uv = V[quads.ravel(), :2]
    return V, quads, uv


def smooth_noise_1d(t, seed=0, octaves=3, base_freq=0.15):
    """Deterministic smooth 1D noise in [-1, 1] (sum of random-phase sines; portable PRNG)."""
    from ..stands.prng import uniform
    out = np.zeros_like(np.asarray(t, float))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        f = base_freq * (2.1 ** o) * (0.8 + 0.4 * uniform(seed, 2 * o))
        out += amp * np.sin(2 * np.pi * f * t + 2 * np.pi * uniform(seed, 2 * o + 1))
        total += amp
        amp *= 0.5
    return out / total


def uv_sphere(center, radius, nu=12, nv=24, z_min=-1.0):
    """Sphere (or cap when z_min > -1, as a fraction of radius). Quads only."""
    lo = np.arccos(np.clip(z_min, -1, 1))
    u = np.linspace(0, lo, nu + 1)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    P = np.stack([np.sin(U) * np.cos(Vv), np.sin(U) * np.sin(Vv), np.cos(U)], -1).reshape(-1, 3)
    P = P * radius + np.asarray(center, float)
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a = i * nv + j
    b = i * nv + (j + 1) % nv
    quads = np.stack([a, b, b + nv, a + nv], -1).reshape(-1, 4)
    return P, quads


def densify(points, max_len):
    """Insert points along segments longer than max_len; existing points (fillets) are kept."""
    p = np.asarray(points, float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / max_len))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t)
    return np.array(out)
