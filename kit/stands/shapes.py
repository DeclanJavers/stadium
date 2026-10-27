"""Small solids the structure generators share. Pure numpy; ported to viewer/gen/src/mesh.rs.

  outward(line)                 horizontal unit normals along a polyline, away from the pitch centre
  sweep(acc, line, section)     a section (o outward, dz up) swept along a polyline (catwalks, gutters)
  obox(acc, c, ax, ay, hx..)    an oriented box (platforms, plinths, cabinets, doors)

UVs are metric (metres along and around), as MeshAcc.add_prism writes them.
"""
import math

import numpy as np


def outward(line):
    """Horizontal unit normal at every vertex of a polyline (central differences), pointing away
    from the pitch centre."""
    P = np.asarray(line, float)
    n = len(P)
    out = []
    for i in range(n):
        a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        L = math.hypot(tx, ty)
        tx, ty = (tx / L, ty / L) if L > 1e-12 else (1.0, 0.0)
        o = np.array([ty, -tx])
        if o[0] * P[i, 0] + o[1] * P[i, 1] < 0:
            o = -o
        out.append(o)
    return np.array(out)


def offset(line, normals, o, dz):
    """The polyline moved `o` outward and `dz` up."""
    P = np.asarray(line, float)
    return np.column_stack([P[:, 0] + normals[:, 0] * o, P[:, 1] + normals[:, 1] * o, P[:, 2] + dz])


def sweep(acc, line, section, tag, normals=None, caps=True):
    """Closed section [(o, dz), ...] swept along a polyline (one ring per vertex)."""
    P = np.asarray(line, float)
    O = outward(P) if normals is None else normals
    S = np.asarray(section, float)
    K, J = len(S), len(P)
    if J < 2:
        return
    base = acc.n
    ring = [(p[0] + o[0] * so, p[1] + o[1] * so, p[2] + dz) for p, o in zip(P, O) for so, dz in S]
    acc.verts.append(np.array(ring))
    acc.n += J * K
    along = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
    per = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([S, S[:1]]), axis=0), axis=1))])
    for j in range(J - 1):
        for i in range(K):
            a, b = base + j * K + i, base + j * K + (i + 1) % K
            acc.faces.append((a, b, b + K, a + K))
            acc.tags.append(tag)
            acc.uvs += [(along[j], per[i]), (along[j], per[i + 1]), (along[j + 1], per[i + 1]),
                        (along[j + 1], per[i])]
    if caps:
        for j, flip in ((0, True), (J - 1, False)):
            idx = [base + j * K + i for i in range(K)]
            acc.faces.append(tuple(reversed(idx)) if flip else tuple(idx))
            acc.tags.append(tag)
            uv = [tuple(q) for q in S]
            acc.uvs += list(reversed(uv)) if flip else uv


_BOX_FACES = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))


def obox(acc, c, ax, ay, hx, hy, hz, tag):
    """Box centred at c with half extents (hx, hy, hz) along ax, ay (horizontal or not) and
    az = ax x ay. Faces wind outward."""
    c, ax, ay = (np.asarray(v, float) for v in (c, ax, ay))
    az = np.cross(ax, ay)
    base = acc.n
    loc = [(x, y, z) for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)]
    acc.verts.append(np.array([c + ax * x + ay * y + az * z for x, y, z in loc]))
    acc.n += 8
    for f in _BOX_FACES:
        acc.faces.append(tuple(base + i for i in f))
        acc.tags.append(tag)
        q = [loc[i] for i in f]
        nrm = np.abs(np.cross(np.subtract(q[1], q[0]), np.subtract(q[2], q[0])))
        if nrm[0] >= nrm[1] and nrm[0] >= nrm[2]:
            acc.uvs += [(v[1], v[2]) for v in q]
        elif nrm[1] >= nrm[2]:
            acc.uvs += [(v[0], v[2]) for v in q]
        else:
            acc.uvs += [(v[0], v[1]) for v in q]


def zbox(acc, lo, hi, tag):
    """Axis-aligned box from corner lo to corner hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    obox(acc, c, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), h[0], h[1], h[2], tag)
