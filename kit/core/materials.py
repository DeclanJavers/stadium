"""Shared physically-based materials. Names starting with TINT_ are recolourable by the game."""
import bpy
import numpy as np

from .nodes import NB, material, srgb


def _get(name):
    m = bpy.data.materials.get(name)
    return m


def ground_dirt(nb, base_color, height=0.18, amount=1.0, color="#4b4a2c"):
    """Grass/soil staining that creeps up from the ground, broken up by noise."""
    pos = nb.geometry()["Position"]
    z = nb.sep(pos)[2]
    ramp = nb.math("SUBTRACT", 1.0, nb.smooth(0.0, height, z))
    n = nb.noise(pos, 18.0, detail=6, rough=0.65)
    mask = nb.clamp01(nb.mul(nb.mul(ramp, nb.smooth(0.35, 0.75, n)), amount))
    return nb.mix(mask, base_color, srgb(color)), mask


def painted_metal(name, color="#f4f4f0", rough=0.32, dirt=0.6, scuffs=0.4):
    """Powder-coated steel/aluminium: satin paint, faint orange peel, grass stains at the foot."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col, dmask = ground_dirt(nb, srgb(color), amount=dirt)
    # handling scuffs: small grey marks
    sc = nb.smooth(0.72, 0.8, nb.noise(pos, 9.0, detail=8, rough=0.7))
    col = nb.mix(nb.mul(sc, scuffs), col, srgb("#b9b8b2"))
    peel = nb.noise(pos, 400.0, detail=2)
    r = nb.add(rough, nb.mul(dmask, 0.4))
    bsdf = nb.principled(Base_Color=col, Roughness=r, Coat_Weight=0.25, Coat_Roughness=0.25,
                         Normal=nb.bump(peel, 0.04, 0.001))
    nb.output(bsdf)
    return m


def aluminium(name, worn=0.6):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    # white paint worn through to bare aluminium in patches
    wear = nb.smooth(0.62, 0.7, nb.noise(pos, 6.0, detail=8, rough=0.7))
    wear = nb.mul(wear, worn)
    paint, dmask = ground_dirt(nb, srgb("#ecebe6"), amount=1.0, height=0.25)
    col = nb.mix(wear, paint, srgb("#a7a9ab"))
    metal = nb.mul(wear, 1.0)
    rough = nb.mixf(wear, 0.42, 0.3)
    bsdf = nb.principled(Base_Color=col, Metallic=metal, Roughness=nb.add(rough, nb.mul(dmask, 0.3)),
                         Normal=nb.bump(nb.noise(pos, 120.0, detail=4), 0.08, 0.002))
    nb.output(bsdf)
    return m


def plastic(name, color, rough=0.45, dirt=0.3, sss=0.0):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    col, dmask = ground_dirt(nb, srgb(color), amount=dirt, height=0.12)
    bsdf = nb.principled(Base_Color=col, Roughness=nb.add(rough, nb.mul(dmask, 0.3)),
                         Subsurface_Weight=sss, Subsurface_Radius=(0.02, 0.02, 0.02))
    nb.output(bsdf)
    return m


def rubber(name, color="#151515"):
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    bsdf = nb.principled(Base_Color=srgb(color), Roughness=0.75,
                         Normal=nb.bump(nb.noise(pos, 300.0, detail=3), 0.15, 0.001))
    nb.output(bsdf)
    return m


def net_cord(name, color="#f7f7f5"):
    """Braided polyester cord: rough, slightly translucent, faint sheen."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col, _ = ground_dirt(nb, srgb(color), amount=0.5, height=0.15)
    v = nb.noise(pos, 3.0, detail=3)
    col = nb.mix(nb.mul(v, 0.25), col, nb.mul(col, 0.8))
    bsdf = nb.principled(Base_Color=col, Roughness=0.7, Sheen_Weight=0.4, Sheen_Roughness=0.4,
                         Subsurface_Weight=0.15, Subsurface_Radius=(0.003, 0.003, 0.003))
    nb.output(bsdf)
    return m


def fabric_image(name, image, sheen=0.5, translucency=0.25):
    """Thin flag fabric from a generated image texture (exports to glTF)."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    tex = nb.node("ShaderNodeTexImage", image=image, interpolation="Closest" if image.size[0] <= 64 else "Linear")
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap")
    nb.connect(uv.outputs[0], tex.inputs["Vector"])
    col = tex.outputs["Color"]
    weave = nb.noise(nb.geometry()["Position"], 900.0, detail=1)
    bsdf = nb.principled(Base_Color=col, Roughness=0.62, Sheen_Weight=sheen, Sheen_Roughness=0.35,
                         Normal=nb.bump(weave, 0.08, 0.0005))
    trans = nb.node("ShaderNodeBsdfTranslucent")
    nb.connect(col, trans.inputs["Color"])
    mixs = nb.node("ShaderNodeMixShader")
    mixs.inputs[0].default_value = translucency
    nb.connect(bsdf.outputs[0], mixs.inputs[1])
    nb.connect(trans.outputs[0], mixs.inputs[2])
    nb.output(mixs)
    return m


def canvas(name, color="#8b7d5a"):
    """Woven sack fabric (sandbags)."""
    if _get(name):
        return _get(name)
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    col, _ = ground_dirt(nb, srgb(color), amount=0.8, height=0.2, color="#3d3a28")
    weave = nb.noise(pos, 600.0, detail=2)
    blotch = nb.noise(pos, 8.0, detail=4)
    col = nb.mix(nb.mul(blotch, 0.4), col, nb.mul(col, 0.7))
    bsdf = nb.principled(Base_Color=col, Roughness=0.9, Normal=nb.bump(weave, 0.3, 0.001))
    nb.output(bsdf)
    return m


def image_from_array(name, rgba):
    """Create a packed Blender image from an (H,W,4) float array in linear 0-1, row 0 = bottom."""
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = "sRGB"
    # store sRGB-encoded values since the image is tagged sRGB
    rgb = rgba[..., :3]
    enc = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.clip(rgb, 0, None), 1 / 2.4) - 0.055)
    out = np.concatenate([enc, rgba[..., 3:4]], -1).astype(np.float32)
    img.pixels.foreach_set(out.ravel())
    img.pack()
    return img
