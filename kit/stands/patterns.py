"""Seat colour patterns and seat mosaics (text or image spelled out in seat colours). Pure numpy.

Every seat anchor gets an sRGB colour. Base patterns: solid, tiers, rows, bays, gradient, random.
Mosaics overlay text (5x7 pixel font) or an image on a straight run, sized to fit and mirrored so
they read correctly from the pitch.
"""
import numpy as np

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "&": ["01100", "10010", "10100", "01000", "10101", "10010", "01101"],
    "'": ["00100", "00100", "01000", "00000", "00000", "00000", "00000"],
}


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def text_bitmap(text):
    """(7, W) bool array for the text, one blank column between glyphs."""
    cols = []
    for k, ch in enumerate(text.upper()):
        g = FONT.get(ch, FONT[" "])
        if k:
            cols.append(np.zeros((7, 1), bool))
        cols.append(np.array([[c == "1" for c in row] for row in g]))
    return np.concatenate(cols, 1) if cols else np.zeros((7, 0), bool)


def base_colors(anchors, spec, tier_rows, seed=0):
    """anchors: list of dicts (tier, row, bay). Returns (N,3) sRGB."""
    n = len(anchors)
    rng = np.random.default_rng(seed)
    pat = spec.get("pattern", "solid")
    tier = np.array([a["tier"] for a in anchors])
    row = np.array([a["row"] for a in anchors])
    grow = row + np.array([sum(tier_rows[:t]) for t in tier])  # row counted from the bottom tier
    cols = [hex_rgb(c) for c in spec.get("colors", [spec.get("color", "#c8102e")])]
    if pat == "solid":
        out = np.tile(cols[0], (n, 1))
    elif pat == "tiers":
        out = np.array([cols[t % len(cols)] for t in tier])
    elif pat == "rows":
        every = spec.get("every", 1)
        out = np.array([cols[(r // every) % len(cols)] for r in grow])
    elif pat == "bays":
        out = np.array([cols[a["bay"] % len(cols)] for a in anchors])
    elif pat == "gradient":
        f = (grow / max(1, grow.max()))[:, None]
        out = cols[0] * (1 - f) + cols[-1] * f
    elif pat == "random":
        w = np.array(spec.get("weights", [1] * len(cols)), float)
        idx = rng.choice(len(cols), n, p=w / w.sum())
        out = np.array(cols)[idx]
    else:
        raise ValueError(f"unknown seat pattern {pat!r}")
    return out


def apply_mosaic(colors, anchors, mosaic, run_length, tier_rows, image=None):
    """Overlay a text/image mosaic onto the anchors of one straight run and tier, in place.

    image: optional (H, W, 3) sRGB array (row 0 = top) used instead of text.
    """
    tier = mosaic.get("tier", 0)
    idx = [i for i, a in enumerate(anchors) if a["run"] == mosaic["run"] and a["tier"] == tier]
    if not idx:
        return 0
    rows = tier_rows[tier]
    r_lo, r_hi = mosaic.get("rows", [1, rows - 1])
    r_hi = rows - 1 if r_hi is None else r_hi
    avail_rows = r_hi - r_lo + 1
    pitch = float(np.mean([anchors[i]["pitch"] for i in idx]))
    span = mosaic.get("span", [0.04, 0.96])
    t0, t1 = span[0] * run_length, span[1] * run_length
    avail_seats = (t1 - t0) / pitch
    if image is None:
        bmp = text_bitmap(mosaic["text"])
        rgb = np.where(bmp[..., None], hex_rgb(mosaic.get("color", "#ffffff")), np.nan)
        if mosaic.get("background"):
            rgb = np.where(bmp[..., None], rgb, hex_rgb(mosaic["background"]))
    else:
        rgb = np.asarray(image, float)
    H, Wd = rgb.shape[:2]
    q = max(1, avail_rows // H)                       # rows per pixel
    p = max(1, int(round(q * 1.6)))                   # seats per pixel (rows read taller than seats)
    while Wd * p > avail_seats and p > 1:
        p -= 1
    width_m = Wd * p * pitch
    centre = 0.5 * (t0 + t1)
    x_left = centre + width_m / 2                     # reads right-to-left in run order (from the pitch)
    top_row = r_lo + (avail_rows + H * q) // 2 - 1
    painted = 0
    for i in idx:
        a = anchors[i]
        c = int(np.floor((x_left - a["t"]) / (p * pitch)))
        r = (top_row - a["row"]) // q
        if 0 <= c < Wd and 0 <= r < H and not np.isnan(rgb[r, c, 0]):
            colors[i] = rgb[r, c]
            painted += 1
    return painted


def finish(colors, fade=0.0, variation=0.025, seed=0):
    """Per-seat manufacturing/UV-fade variation; returns linear RGBA for the renderer."""
    rng = np.random.default_rng(seed + 1)
    c = colors * (1 + rng.normal(0, variation, (len(colors), 1)))
    if fade:
        grey = c.mean(1, keepdims=True)
        c = c * (1 - 0.45 * fade) + (0.25 + 0.55 * grey) * 0.45 * fade + 0.08 * fade
    c = np.clip(c, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.concatenate([lin, np.ones((len(c), 1))], 1)
