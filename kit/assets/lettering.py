"""Crisp lettering for signs and vinyls: a geometric sans stroke font (capitals, digits) drawn
anti-aliased from line segments and arcs at 3x and filtered down, so text reads as type at any
size (props.label's 5x7 pixel face reads as pixels close up).

    mask = text_mask("HOME TEAM", px_h=128)     # (H, W) coverage 0-1, row 0 = bottom
"""
import math

import numpy as np

# glyphs on a box 0-4 wide, 0-6 tall: lists of strokes; a stroke is a polyline of (x, y) points,
# or ("arc", cx, cy, rx, ry, a0, a1) in degrees (counter-clockwise from +x)
_G = {
    "A": [[(0, 0), (2, 6), (4, 0)], [(0.7, 2.1), (3.3, 2.1)]],
    "B": [[(0, 0), (0, 6), (2.6, 6)], [("arc", 2.6, 4.6, 1.4, 1.4, 90, -90)], [(0, 3.2), (2.8, 3.2)],
          [("arc", 2.8, 1.6, 1.4, 1.6, 90, -90)], [(2.8, 0), (0, 0)]],
    "C": [[("arc", 2.3, 3, 2.1, 3, 45, 315)]],
    "D": [[(0, 0), (0, 6), (1.6, 6)], [("arc", 1.6, 3, 2.4, 3, 90, -90)], [(1.6, 0), (0, 0)]],
    "E": [[(3.8, 6), (0, 6), (0, 0), (3.8, 0)], [(0, 3.1), (3.2, 3.1)]],
    "F": [[(3.8, 6), (0, 6), (0, 0)], [(0, 3.1), (3.2, 3.1)]],
    "G": [[("arc", 2.3, 3, 2.1, 3, 45, 330)], [(4.2, 1.3), (4.2, 2.7), (2.6, 2.7)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3.1), (4, 3.1)]],
    "I": [[(0.8, 0), (0.8, 6)]],
    "J": [[(3.4, 6), (3.4, 1.6)], [("arc", 1.8, 1.6, 1.6, 1.6, 0, -180)]],
    "K": [[(0, 0), (0, 6)], [(3.8, 6), (0, 2.2)], [(1.3, 3.4), (4, 0)]],
    "L": [[(0, 6), (0, 0), (3.6, 0)]],
    "M": [[(0, 0), (0, 6), (2.2, 1.2), (4.4, 6), (4.4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [[("arc", 2.3, 3, 2.3, 3, 0, 360)]],
    "P": [[(0, 0), (0, 6), (2.4, 6)], [("arc", 2.4, 4.4, 1.6, 1.6, 90, -90)], [(2.4, 2.8), (0, 2.8)]],
    "Q": [[("arc", 2.3, 3, 2.3, 3, 0, 360)], [(2.6, 1.4), (4.6, -0.3)]],
    "R": [[(0, 0), (0, 6), (2.4, 6)], [("arc", 2.4, 4.4, 1.6, 1.6, 90, -90)], [(2.4, 2.8), (0, 2.8)], [(2.0, 2.8), (4, 0)]],
    "S": [[("arc", 2.0, 4.5, 1.8, 1.5, 20, 270)], [("arc", 2.0, 1.5, 2.0, 1.5, 90, -160)]],
    "T": [[(0, 6), (4.2, 6)], [(2.1, 6), (2.1, 0)]],
    "U": [[(0, 6), (0, 2)], [("arc", 2, 2, 2, 2, 180, 360)], [(4, 2), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1.3, 0), (2.6, 4.6), (3.9, 0), (5.2, 6)]],
    "X": [[(0, 6), (4, 0)], [(0, 0), (4, 6)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0.2, 6), (4, 6), (0, 0), (4, 0)]],
    "0": [[("arc", 2, 3, 2, 3, 0, 360)]],
    "1": [[(0.8, 4.8), (2.2, 6), (2.2, 0)]],
    "2": [[("arc", 2, 4.1, 1.9, 1.9, 160, -40)], [(3.4, 2.9), (0, 0), (4, 0)]],
    "3": [[("arc", 1.9, 4.6, 1.7, 1.4, 160, -90)], [("arc", 1.9, 1.6, 2.0, 1.6, 90, -160)]],
    "4": [[(3, 0), (3, 6), (0, 1.6), (4.2, 1.6)]],
    "5": [[(3.8, 6), (0.4, 6), (0.2, 3.3)], [("arc", 2.0, 1.9, 2.0, 1.9, 135, -150)]],
    "6": [[("arc", 2.1, 1.9, 1.9, 1.9, 0, 360)], [(0.25, 2.3), (2.6, 6)]],
    "7": [[(0, 6), (4, 6), (1.4, 0)]],
    "8": [[("arc", 2, 4.6, 1.6, 1.4, 0, 360)], [("arc", 2, 1.6, 1.9, 1.6, 0, 360)]],
    "9": [[("arc", 1.9, 4.1, 1.9, 1.9, 0, 360)], [(3.75, 3.7), (1.4, 0)]],
    "-": [[(0.4, 3), (2.6, 3)]],
    ".": [[(0.5, 0.2), (0.5, 0.25)]],
    "&": [[(4, 0), (1, 4.2)], [("arc", 1.8, 4.9, 1.0, 1.1, -40, 220)], [(1.2, 4.4), (0.3, 2.4)],
          [("arc", 1.7, 1.6, 1.5, 1.6, 150, 330)], [(2.9, 0.6), (4, 2.2)]],
}
_W = {"I": 1.6, "M": 4.4, "W": 5.2, "1": 3.2, "-": 3.0, ".": 1.0, " ": 2.4, "O": 4.6, "Q": 4.6, "C": 4.2, "G": 4.4}


def _points(stroke):
    if stroke and stroke[0] and stroke[0][0] == "arc":
        _, cx, cy, rx, ry, a0, a1 = stroke[0]
        n = max(8, int(abs(a1 - a0) / 8))
        return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
                for a in np.linspace(a0, a1, n)]
    return stroke


def text_mask(text, px_h=128, weight=0.75, track=1.0, ss=3):
    """Coverage of `text` in capitals: px_h pixels from baseline to cap height (plus margins),
    strokes `weight` glyph units wide."""
    text = text.upper()
    unit = px_h / 8.0 * ss                          # cap height 6 units, a unit margin each side
    adv = [(_W.get(ch, 4.0) + track + weight) for ch in text]
    W = int(math.ceil((sum(adv) + 2) * unit))
    H = int(8 * unit)
    mask = np.zeros((H, W), np.float32)
    x0 = 1.0
    hw = weight / 2
    for ch, a in zip(text, adv):
        for st in _G.get(ch, []):
            P = _points(st)
            for (ax, ay), (bx, by) in zip(P, P[1:] + ([] if len(P) > 1 else P)):
                ax, ay, bx, by = (ax + x0 + hw) * unit, (ay + 1) * unit, (bx + x0 + hw) * unit, (by + 1) * unit
                r = hw * unit
                xa, xb = int(max(0, min(ax, bx) - r - 2)), int(min(W, max(ax, bx) + r + 3))
                ya, yb = int(max(0, min(ay, by) - r - 2)), int(min(H, max(ay, by) + r + 3))
                if xb <= xa or yb <= ya:
                    continue
                X, Y = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(ya, yb) + 0.5)
                ex, ey = bx - ax, by - ay
                t = np.clip(((X - ax) * ex + (Y - ay) * ey) / (ex * ex + ey * ey + 1e-9), 0, 1)
                d = np.hypot(X - ax - t * ex, Y - ay - t * ey)
                cov = np.clip(r - d + 0.5, 0, 1)
                mask[ya:yb, xa:xb] = np.maximum(mask[ya:yb, xa:xb], cov)
        x0 += a
    # filter down
    h, w = H // ss, W // ss
    mask = mask[:h * ss, :w * ss].reshape(h, ss, w, ss).mean((1, 3))
    return mask


def sign(text, px_h, fg, bg=None, pad=0.15, weight=0.75):
    """RGBA (linear) of text in fg over bg (None: transparent), with `pad` (fraction of height)."""
    from .foliage_paint import srgb_lin
    m = text_mask(text, px_h, weight)
    p = int(pad * m.shape[0])
    m = np.pad(m, ((p, p), (p, p)))
    f = srgb_lin(fg)
    out = np.zeros(m.shape + (4,))
    if bg is None:
        out[..., :3] = f
        out[..., 3] = m
    else:
        b = srgb_lin(bg)
        out[..., :3] = b + (f - b) * m[..., None]
        out[..., 3] = 1
    return out
