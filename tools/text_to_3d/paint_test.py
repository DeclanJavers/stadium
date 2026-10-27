"""A/B the texture stage on props make_props.py already refined (Colab, Hi3D env).

    python paint_test.py meshes id [id ...]

Writes meshes/<id>/paint_full.glb (Hunyuan3D-Paint v2.0, not turbo) and paint_sr.glb (turbo with
each generated view upscaled 4x by the SD x4 upscaler before baking: the views come out at 512 px
and are otherwise stretched to the 2048 px bake).
"""
import sys
import time
from pathlib import Path

import torch
import trimesh
from PIL import Image

torch.backends.cuda.enable_cudnn_sdp(False)

from hy3dgen.shapegen.postprocessors import FaceReducer  # noqa: E402
from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: E402
from hy3dgen.texgen.utils.imagesuper_utils import Image_Super_Net  # noqa: E402

root, ids = Path(sys.argv[1]), sys.argv[2:]


class Upscaled:
    """Stands in for the multiview model and upscales whatever it returns."""

    def __init__(self, mv, sr):
        self.mv, self.sr = mv, sr

    def __call__(self, *a, **k):
        return [self.sr(im) for im in self.mv(*a, **k)]


def run(painter, tag):
    for pid in ids:
        d = root / pid
        mid = FaceReducer()(trimesh.load(d / "high.glb", force="mesh", process=False), max_facenum=40_000)
        t = time.time()
        painter(mid, image=Image.open(d / "cutout.png")).export(d / f"paint_{tag}.glb")
        print(f"{pid} {tag}: {time.time() - t:.0f} s", flush=True)


turbo = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
turbo.enable_model_cpu_offload()
turbo.models["multiview_model"] = Upscaled(turbo.models["multiview_model"], Image_Super_Net(turbo.config))
run(turbo, "sr")
del turbo
torch.cuda.empty_cache()

full = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0")
full.enable_model_cpu_offload()
run(full, "full")
