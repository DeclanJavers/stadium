"""Ad studio: renders advertising creatives and screen graphics to images with Blender's Workbench
engine, in a scene of their own, so the boards, carpets, ribbons and screens can use real textures
(which export to glTF). Every brand is fictional (catalog/brands.json)."""
import math
from pathlib import Path

import bpy
import numpy as np

from .. import catalog
from ..core.scene import OUT

FONTS = ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
FONTS_NARROW = ["/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"]


def _font(paths):
    for p in paths:
        if Path(p).exists():
            return bpy.data.fonts.load(p, check_existing=True)
    return None  # Blender's built-in font


def _hex(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [1.0]


class Studio:
    def __init__(self, width_px, height_px):
        self.scene = bpy.data.scenes.new("AdStudio")
        sc = self.scene
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.light = "FLAT"
        sc.display.shading.color_type = "MATERIAL"
        sc.display.render_aa = "16"
        sc.view_settings.view_transform = "Standard"
        sc.render.resolution_x, sc.render.resolution_y = width_px, height_px
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        self.w, self.h = width_px / height_px, 1.0   # canvas units: height 1
        cam = bpy.data.cameras.new("StudioCam")
        cam.type = "ORTHO"
        cam.ortho_scale = max(self.w, self.h)
        self.cam = bpy.data.objects.new("StudioCam", cam)
        self.cam.location = (0, 0, 10)
        sc.collection.objects.link(self.cam)
        sc.camera = self.cam
        self.objs = []

    def _mat(self, color):
        key = f"Studio_{color}"
        m = bpy.data.materials.get(key) or bpy.data.materials.new(key)
        m.diffuse_color = _hex(color)
        return m

    def rect(self, x0, y0, x1, y1, color, z=0.0):
        me = bpy.data.meshes.new("r")
        me.from_pydata([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [], [(0, 1, 2, 3)])
        return self._add(me, color)

    def poly(self, pts, color, z=0.01):
        me = bpy.data.meshes.new("p")
        me.from_pydata([(x, y, z) for x, y in pts], [], [tuple(range(len(pts)))])
        return self._add(me, color)

    def disc(self, cx, cy, r, color, z=0.01, inner=0.0, k=64):
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        if inner <= 0:
            return self.poly([(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a], color, z)
        V = [(cx + r * math.cos(t), cy + r * math.sin(t), z) for t in a] + \
            [(cx + inner * math.cos(t), cy + inner * math.sin(t), z) for t in a]
        F = [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
        me = bpy.data.meshes.new("ring")
        me.from_pydata(V, [], F)
        return self._add(me, color)

    def text(self, body, x, y, size, color, max_w=None, align="CENTER", narrow=False, z=0.02):
        cu = bpy.data.curves.new("t", "FONT")
        cu.body = body
        f = _font(FONTS_NARROW if narrow else FONTS)
        if f:
            cu.font = f
        cu.size = size
        cu.align_x = align
        cu.align_y = "CENTER"
        o = bpy.data.objects.new("t", cu)
        o.location = (x, y, z)
        o.data.materials.append(self._mat(color))
        self.scene.collection.objects.link(o)
        self.objs.append(o)
        if max_w:
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            me = o.evaluated_get(dg).to_mesh()
            xs = [v.co.x for v in me.vertices]
            width = (max(xs) - min(xs)) * o.scale.x if xs else 0
            o.evaluated_get(dg).to_mesh_clear()
            if width > max_w:
                cu.size = size * max_w / width
        return o

    def _add(self, me, color):
        me.materials.append(self._mat(color))
        o = bpy.data.objects.new("s", me)
        self.scene.collection.objects.link(o)
        self.objs.append(o)
        return o

    def render(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True, scene=self.scene.name)
        return path

    def clear(self):
        for o in self.objs:
            bpy.data.objects.remove(o)
        self.objs = []

    def close(self):
        self.clear()
        bpy.data.objects.remove(self.cam)
        bpy.data.scenes.remove(self.scene)


def _logo(st, kind, cx, cy, s, color, bg):
    if kind == "disc":
        st.disc(cx, cy, s * 0.42, color)
    elif kind == "ring":
        st.disc(cx, cy, s * 0.42, color, inner=s * 0.27)
    elif kind == "chevron":
        for k in range(2):
            dx = k * s * 0.32
            st.poly([(cx - s * 0.35 + dx, cy + s * 0.38), (cx - s * 0.12 + dx, cy + s * 0.38), (cx + s * 0.18 + dx, cy),
                     (cx - s * 0.12 + dx, cy - s * 0.38), (cx - s * 0.35 + dx, cy - s * 0.38), (cx - s * 0.05 + dx, cy)], color)
    elif kind == "triangle":
        st.poly([(cx - s * 0.42, cy - s * 0.36), (cx + s * 0.42, cy - s * 0.36), (cx, cy + s * 0.4)], color)
    elif kind == "bars":
        for k in range(3):
            h = s * (0.35 + 0.22 * k)
            st.rect(cx - s * 0.4 + k * s * 0.3, cy - s * 0.38, cx - s * 0.2 + k * s * 0.3, cy - s * 0.38 + h, color, 0.01)
    elif kind == "wave":
        a = np.linspace(-1, 1, 40)
        top = [(cx + s * 0.45 * t, cy + s * 0.12 * math.sin(t * 5)) for t in a]
        bot = [(cx + s * 0.45 * t, cy - s * 0.18 + s * 0.12 * math.sin(t * 5)) for t in a[::-1]]
        st.poly(top + bot, color)


def board_creative(st, brand, tagline=True):
    """Wide board layout: logo at the left, wordmark, tagline at the right."""
    W, H = st.w, 1.0
    st.rect(-W / 2, -H / 2, W / 2, H / 2, brand["bg"])
    st.rect(-W / 2, -H / 2, W / 2, -H / 2 + 0.06, brand["accent"], 0.005)
    has_logo = brand.get("logo", "none") != "none"
    if has_logo:
        _logo(st, brand["logo"], -W / 2 + 0.75, 0.02, 0.9, brand["accent"], brand["bg"])
    x0 = -W / 2 + (1.5 if has_logo else 0.4)
    room = W - (1.5 if has_logo else 0.4) - (2.6 if tagline else 0.4)
    st.text(brand["name"], x0 + room / 2, 0.02, 0.62, brand["fg"], max_w=room)
    if tagline:
        st.text(brand["tagline"], W / 2 - 1.3, 0.02, 0.2, brand["fg"], max_w=2.3, narrow=True)


def brand_list():
    return catalog.load("brands")


def render_boards(keys=None, px_h=128, aspect=8.0, out=None):
    """One image per brand (aspect:1) plus a horizontal atlas of them, in `keys` order (the LED
    rotation). Returns (atlas path, keys)."""
    import zlib
    out = Path(out or OUT / "ads")
    allb = brand_list()
    keys = list(keys or allb)
    brands = {k: allb[k] for k in keys}
    st = Studio(int(px_h * aspect), px_h)
    paths = []
    for key, b in brands.items():
        st.clear()
        board_creative(st, b)
        paths.append(st.render(out / f"board_{key}.png"))
    st.close()
    tag = zlib.crc32(",".join(keys).encode())
    atlas = _atlas(paths, out / f"board_atlas_{tag:08x}.png")
    return atlas, keys


def render_carpets(px_h=256, aspect=6.0, out=None):
    out = Path(out or OUT / "ads")
    st = Studio(int(px_h * aspect), px_h)
    paths = {}
    for key, b in brand_list().items():
        st.clear()
        board_creative(st, b, tagline=False)
        paths[key] = st.render(out / f"carpet_{key}.png")
    st.close()
    return paths


def render_scoreboard(home="KIT FC", away="ROVERS", score=(2, 1), clock="67:12", sponsor="northwind", out=None,
                      home_color="#c8102e", away_color="#1d4ea8"):
    out = Path(out or OUT / "ads")
    st = Studio(1920, 1080)
    W, H = st.w, 1.0
    b = brand_list()[sponsor]
    st.rect(-W / 2, -H / 2, W / 2, H / 2, "#07090d")
    st.rect(-W / 2, 0.16, W / 2, 0.5, "#10141c", 0.002)
    st.rect(-W / 2 + 0.06, 0.2, -0.2, 0.44, home_color, 0.005)
    st.rect(0.2, 0.2, W / 2 - 0.06, 0.44, away_color, 0.005)
    st.text(home, -W / 4 - 0.05, 0.32, 0.16, "#ffffff", max_w=W / 2 - 0.5)
    st.text(away, W / 4 + 0.05, 0.32, 0.16, "#ffffff", max_w=W / 2 - 0.5)
    st.text(f"{score[0]}  -  {score[1]}", 0, -0.02, 0.36, "#ffffff")
    st.text(clock, 0, -0.27, 0.12, "#ffd400")
    st.rect(-W / 2, -H / 2, W / 2, -0.38, b["bg"], 0.003)
    st.text(b["name"], 0, -0.44, 0.08, b["fg"], max_w=W - 0.3)
    p = st.render(out / "screen_scoreboard.png")
    st.close()
    return p


def _atlas(paths, dest):
    imgs = [bpy.data.images.load(str(p)) for p in paths]
    w, h = imgs[0].size
    arr = np.concatenate([np.array(im.pixels[:]).reshape(h, w, 4) for im in imgs], 1)
    atlas = bpy.data.images.new(Path(dest).stem, w * len(imgs), h)
    atlas.pixels.foreach_set(arr.astype(np.float32).ravel())
    atlas.filepath_raw = str(dest)
    atlas.file_format = "PNG"
    atlas.save()
    for im in imgs:
        bpy.data.images.remove(im)
    bpy.data.images.remove(atlas)
    return Path(dest)
