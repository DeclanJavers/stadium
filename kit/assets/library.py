"""The library's named materials (what layouts and generated parts reference by key): the stadium
kit's, here, plus the training grounds' (site_materials.MATERIALS). Each entry: (builder, the sRGB
colour it is built with, or None) -- engines recolour relative to that colour.

    get("Facade_Brick") -> bpy material (built once)        base_hex("Facade_Brick") -> "#8a3b2a"
"""
import bpy

from ..core import materials as MAT
from . import dugout, facade, roof, seat, site_materials, stand, tunnel

# (material, the sRGB colour it is baked with) -- the editor recolours relative to it
MATERIALS = {
    "Stand_Concrete": (lambda: stand.concrete(), "#9d998f"),
    "Stand_Block": (lambda: stand.blockwork("Stand_Block", "#b3aea4"), "#b3aea4"),
    "Stand_Soffit": (lambda: stand.painted("Stand_Soffit", "#d9d6cf", 0.8), "#d9d6cf"),
    "TINT_fascia": (lambda: stand.painted("TINT_fascia", "#1b1d20", 0.4), "#1b1d20"),
    "Balustrade_Glass": (lambda: stand.glass(), None),
    "Galvanised": (lambda: seat.galvanised(), None),
    "Step_Nosing": (lambda: stand.painted("Step_Nosing", "#f2c200", 0.55), "#f2c200"),
    "Apron_Track": (lambda: stand.painted("Apron_Track", "#9c3f2c", 0.85), "#9c3f2c"),
    "Apron_Concrete": (lambda: stand.concrete("Apron_Concrete", "#77746e"), "#77746e"),
    "Plaza_Paving": (lambda: stand.paving(), None),
    "Plaza_Band": (lambda: stand.paving("Plaza_Band", "#6c6a66", "#62605c", "#4a4946"), None),
    "TINT_roof": (lambda: roof.deck_material("#d9dbdc"), "#d9dbdc"),
    "TINT_roof_membrane": (lambda: roof.deck_material("#f4f3ef", membrane=True), "#f4f3ef"),
    "Roof_Polycarbonate": (lambda: roof.translucent_material(), None),
    "TINT_roof_fascia": (lambda: stand.painted("TINT_roof_fascia", "#1b1d20", 0.45), "#1b1d20"),
    "Roof_Underside": (lambda: roof.underside_material(), "#d8d8d4"),
    "Roof_Steel": (lambda: MAT.painted_metal("Roof_Steel", "#eceeee", rough=0.4, dirt=0.0, scuffs=0.1), "#eceeee"),
    "Facade_Brick": (lambda: facade.brick(color="#8a3b2a"), "#8a3b2a"),
    "Facade_Stone": (lambda: stand.concrete("Facade_Stone", "#d6ccb9"), "#d6ccb9"),
    "Facade_Concrete": (lambda: stand.concrete("Facade_Concrete", "#b8b3a8"), "#b8b3a8"),
    "Facade_Trim": (lambda: MAT.painted_metal("Facade_Trim", "#d9dad6", rough=0.4, dirt=0.2, scuffs=0.0), "#d9dad6"),
    "TINT_facade": (lambda: stand.painted("TINT_facade", "#c7c9c9", 0.35), "#c7c9c9"),
    "TINT_facade_b": (lambda: stand.painted("TINT_facade_b", "#9e0c24", 0.35), "#9e0c24"),
    "Facade_DarkGlass": (lambda: facade.dark_glass(), None),
    "TINT_facade_etfe": (lambda: facade.etfe("#f2f2f0", None, 0.0), "#f2f2f0"),
    "Facade_Core": (lambda: stand.concrete("Facade_Core", "#bdb8ad"), "#bdb8ad"),
    "Facade_Frame": (lambda: MAT.painted_metal("Facade_Frame", "#f1f2f0", rough=0.35, dirt=0.0, scuffs=0.0), "#f1f2f0"),
    "Board_Housing": (lambda: stand.painted("Board_Housing", "#34363a", 0.5), "#34363a"),
    "Screen_Body": (lambda: stand.painted("Screen_Body", "#45484d", 0.5), "#45484d"),
    "Dugout_Slab": (lambda: stand.concrete("Dugout_Slab", "#8e8a82"), "#8e8a82"),
    "Dugout_Polycarbonate": (lambda: dugout.polycarbonate("#3d4954"), "#3d4954"),
    "Dugout_Frame": (lambda: MAT.painted_metal("Dugout_Frame", "#d8dadb", rough=0.35, dirt=0.2), "#d8dadb"),
    "Dugout_Brick": (lambda: facade.brick("Dugout_Brick", "#7a3426"), "#7a3426"),
    "TINT_dugout": (lambda: stand.painted("TINT_dugout", "#c8102e", 0.45), "#c8102e"),
    "Tunnel_Mouth": (lambda: MAT.painted_metal("Tunnel_Mouth", "#050506", rough=0.9, dirt=0.0, scuffs=0.0), "#050506"),
    "Tunnel_Trim": (lambda: stand.painted("Tunnel_Trim", "#1a1b1e", 0.4), "#1a1b1e"),
    "Tunnel_Building": (lambda: stand.concrete("Tunnel_Building", "#a8a399"), "#a8a399"),
    "TINT_tunnel": (lambda: tunnel.fabric("#c8102e"), "#c8102e"),
    "Tunnel_Mat": (lambda: MAT.rubber("Tunnel_Mat", "#1f2a22"), "#1f2a22"),
    "Tunnel_Hoop": (lambda: MAT.painted_metal("Tunnel_Hoop", "#d9dbdc", 0.35, 0.1, 0.0), "#d9dbdc"),
}




def _entry(name):
    return MATERIALS.get(name) or site_materials.MATERIALS.get(name)


def get(name):
    """The named library material (None if unknown)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    e = _entry(name)
    return e[0]() if e else None


def base_hex(name):
    e = _entry(name)
    return e[1] if e else None
