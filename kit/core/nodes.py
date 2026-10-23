"""A small builder for shader and geometry node trees, so material code reads like maths."""
import bpy

# ShaderNodeMix socket indices (Blender 5.x): factor float 0, A/B float 2/3, vector 4/5, colour 6/7.
_MIX_IN = {"FLOAT": (0, 2, 3, 0), "VECTOR": (0, 4, 5, 1), "RGBA": (0, 6, 7, 2)}


def srgb(hex_or_rgb):
    """sRGB hex or 0-1 tuple to linear RGBA."""
    if isinstance(hex_or_rgb, str):
        h = hex_or_rgb.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    else:
        rgb = list(hex_or_rgb)[:3]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return (*lin, 1.0)


class NB:
    def __init__(self, tree, clear=True):
        self.tree = tree
        if clear:
            tree.nodes.clear()
        self.nodes = tree.nodes
        self.links = tree.links

    # -- plumbing -------------------------------------------------------
    def connect(self, v, sock):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.links.new(v.outputs[0], sock)
        else:
            dv = sock.default_value
            if isinstance(v, (tuple, list)) and hasattr(dv, "__len__") and len(dv) == 4 and len(v) == 3:
                v = (*v, 1.0)
            sock.default_value = v

    def node(self, type_, inputs=None, **props):
        n = self.nodes.new(type_)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self.connect(v, n.inputs[k])
        return n

    # -- maths ----------------------------------------------------------
    def math(self, op, a, b=0.0, c=0.0, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self.connect(v, n.inputs[i])
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def madd(self, a, b, c): return self.math("MULTIPLY_ADD", a, b, c)

    def clamp01(self, a):
        return self.math("MULTIPLY", a, 1.0, clamp=True)

    def smooth(self, e0, e1, x):
        n = self.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP")
        self.connect(x, n.inputs["Value"])
        n.inputs["From Min"].default_value = e0
        n.inputs["From Max"].default_value = e1
        return n.outputs["Result"]

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.connect(a, n.inputs[0])
        if b is not None:
            self.connect(b, n.inputs[1])
        if scale is not None:
            self.connect(scale, n.inputs[3])
        scalar_ops = {"DOT_PRODUCT", "DISTANCE", "LENGTH"}
        return n.outputs[1] if op in scalar_ops else n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.connect(v, n.inputs[0])
        return n.outputs

    def comb(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.connect(v, n.inputs[i])
        return n.outputs[0]

    def mix(self, f, a, b, data_type="RGBA", blend="MIX", clamp=True):
        n = self.node("ShaderNodeMix", data_type=data_type)
        if data_type == "RGBA":
            n.blend_type = blend
            n.clamp_result = False
        n.clamp_factor = clamp
        fi, ai, bi, oi = _MIX_IN[data_type]
        self.connect(f, n.inputs[fi])
        self.connect(a, n.inputs[ai])
        self.connect(b, n.inputs[bi])
        return n.outputs[oi]

    def mixf(self, f, a, b):
        return self.mix(f, a, b, "FLOAT")

    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, dims="3D"):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        self.connect(vec, n.inputs["Vector"])
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Factor"]

    def voronoi(self, vec, scale, feature="F1", out="Distance"):
        n = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.connect(vec, n.inputs["Vector"])
        n.inputs["Scale"].default_value = scale
        return n.outputs[out]

    def ramp(self, fac, stops):
        """stops: [(pos, rgba)]"""
        n = self.node("ShaderNodeValToRGB")
        self.connect(fac, n.inputs[0])
        els = n.color_ramp.elements
        while len(els) > len(stops):
            els.remove(els[-1])
        while len(els) < len(stops):
            els.new(0.5)
        for el, (p, c) in zip(els, stops):
            el.position = p
            el.color = c if len(c) == 4 else (*c, 1.0)
        return n.outputs[0]

    def bump(self, height, strength=0.3, distance=0.01, normal=None):
        n = self.node("ShaderNodeBump")
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        self.connect(height, n.inputs["Height"])
        if normal is not None:
            self.connect(normal, n.inputs["Normal"])
        return n.outputs[0]

    def principled(self, **inputs):
        n = self.node("ShaderNodeBsdfPrincipled")
        for k, v in inputs.items():
            self.connect(v, n.inputs[k.replace("_", " ")])
        return n

    def output(self, shader, displacement=None):
        out = self.node("ShaderNodeOutputMaterial")
        self.connect(shader, out.inputs["Surface"])
        if displacement is not None:
            self.connect(displacement, out.inputs["Displacement"])
        return out

    def geometry(self):
        return self.node("ShaderNodeNewGeometry").outputs

    def attribute(self, name, kind="GEOMETRY"):
        return self.node("ShaderNodeAttribute", attribute_name=name, attribute_type=kind).outputs


def material(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    return m, NB(m.node_tree)
