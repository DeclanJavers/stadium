"""Stage 2 (Colab, Hi3D env): image -> coarse mesh -> Hi3D refinement -> painted texture.

    python make_props.py --manifest generated_props.json --cutouts cutouts --out meshes
                         [--ids a,b] [--first a,b] [--follow] [--force]

Per prop, in meshes/<id>/:
  cutout.png    the reviewed cut-out it was made from
  coarse.glb    Hunyuan3D-2mini turbo shape
  high.glb      Hi3D refinement, reduced to 400k faces (the normal-map source)
  paint.glb     a 40k-face copy with Hunyuan3D-Paint's 2048 px texture (the colour source)
  status.json   timings, or the error if a stage failed
All models stay loaded, so each prop costs only its own inference.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/content/Hi3Dv1.0/demo")

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

# torch 2.5's cuDNN attention fails in the paint UNet ("No execution plans support the graph").
torch.backends.cuda.enable_cudnn_sdp(False)

from coarse_stage import generate_coarse_mesh, load_coarse_generator  # noqa: E402
from hy3dgen.shapegen.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover  # noqa: E402
import hy3dgen.texgen.pipelines as paint_pipelines  # noqa: E402
import multiprocessing  # noqa: E402
import queue  # noqa: E402
import xatlas  # noqa: E402
from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: E402
from sparc3d import Sparc3DPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--cutouts", required=True, help="make_cutouts.py output; reviewed before this runs")
ap.add_argument("--out", required=True)
ap.add_argument("--ids", default="")
ap.add_argument("--force", action="store_true")
ap.add_argument("--first", default="", help="ids to make before the rest, in this order")
ap.add_argument("--follow", action="store_true", help="wait for cut-outs that are still being made")
ap.add_argument("--hi3d", default="/content/models/hi3d")
args = ap.parse_args()

spec = json.loads(Path(args.manifest).read_text())
wanted = {s for s in args.ids.split(",") if s}
props = [p for p in spec["props"] if not wanted or p["id"] in wanted]
first = [s for s in args.first.split(",") if s]
props.sort(key=lambda p: first.index(p["id"]) if p["id"] in first else len(first))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


log("loading models")
coarse_gen = load_coarse_generator(device="cuda")
coarse_gen.to("cpu")
refiner = Sparc3DPipeline.from_pretrained(args.hi3d, local_files_only=True)
refiner.to("cuda")
painter = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
painter.enable_model_cpu_offload()
log("models ready")

def _parametrize(v, f, out):
    out.put(xatlas.parametrize(v, f))


def uv_wrap(mesh, timeout=180):
    """xatlas in a child process with a time limit: on thin layered meshes (a stack of bibs) it
    can grind for hours. Returns None on timeout so the caller can retry on a coarser mesh."""
    ctx = multiprocessing.get_context("fork")
    out = ctx.Queue()
    proc = ctx.Process(target=_parametrize, args=(mesh.vertices, mesh.faces, out))
    proc.start()
    try:
        vmapping, indices, uvs = out.get(timeout=timeout)
    except queue.Empty:
        proc.kill()
        proc.join()
        return None
    proc.join()
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs
    return mesh


def grid_wrap(mesh, px=2048):
    """A UV layout that cannot fail: every pair of triangles gets its own square cell of a grid
    (one in the lower-left half, one in the upper-right), with a texel of padding. Shapes are
    distorted and nothing is shared across edges, but the painted texture is only a colour source
    that finish_props.py re-bakes onto its own unwrap, so all that matters is even texel density
    and no overlaps. For meshes xatlas can't chart in time (thin layered fabric, stripes, fringes)."""
    import numpy as np
    f = np.asarray(mesh.faces)
    n = len(f)
    g = int(np.ceil(np.sqrt((n + 1) // 2)))
    s, pad = 1.0 / g, 1.5 / px
    k = np.arange(n)
    cell = k // 2
    x0, y0 = (cell % g) * s, (cell // g) * s
    x1, y1 = x0 + s, y0 + s
    lower = (k % 2) == 0
    uv = np.empty((n, 3, 2))
    uv[:, 0] = np.where(lower[:, None], np.stack([x0 + pad, y0 + pad], -1), np.stack([x1 - pad, y1 - pad], -1))
    uv[:, 1] = np.where(lower[:, None], np.stack([x1 - 2 * pad, y0 + pad], -1), np.stack([x0 + 2 * pad, y1 - pad], -1))
    uv[:, 2] = np.where(lower[:, None], np.stack([x0 + pad, y1 - 2 * pad], -1), np.stack([x1 - pad, y0 + 2 * pad], -1))
    mesh.vertices = np.asarray(mesh.vertices)[f.reshape(-1)]
    mesh.faces = np.arange(3 * n).reshape(n, 3)
    mesh.visual.uv = uv.reshape(-1, 2)
    return mesh


# The painter unwraps with no time limit; meshes reach it already unwrapped by uv_wrap.
paint_pipelines.mesh_uv_wrap = lambda mesh: mesh


def newer(a, b):
    """a exists and was written after b (or b is missing)."""
    return a.exists() and (not b.exists() or a.stat().st_mtime > b.stat().st_mtime)


def make(p, n):
    d = Path(args.out) / p["id"]
    src = Path(args.cutouts) / f"{p['id']}.png"
    d.mkdir(parents=True, exist_ok=True)
    seed = p.get("seed", 7)
    status, t0 = {"id": p["id"], "res": p["res"]}, time.time()
    try:
        cutout = Image.open(src)
        cutout.save(d / "cutout.png")

        if newer(d / "high.glb", src) and not args.force:  # resume: refinement is the slow part
            high = trimesh.load(d / "high.glb", force="mesh", process=False)
        else:
            coarse_gen.to("cuda")
            coarse = generate_coarse_mesh(coarse_gen, cutout, seed)
            coarse_gen.to("cpu")
            torch.cuda.empty_cache()
            coarse.export(d / "coarse.glb")
            status["coarse_s"] = round(time.time() - t0)

            t = time.time()
            high = refiner(cutout, coarse, resolution=p["res"], num_inference_steps=25, seed=seed).mesh
            high = DegenerateFaceRemover()(FloaterRemover()(high))
            status["high_faces"] = len(high.faces)
            high = FaceReducer()(high, max_facenum=400_000)
            high.export(d / "high.glb")
            status["refine_s"] = round(time.time() - t)
            torch.cuda.empty_cache()

        t = time.time()
        # xatlas first (proper charts); where it grinds, the grid atlas (instant) at 20k faces
        status["uv"] = "xatlas"
        for faces, limit in ((40_000, 120), (20_000, 60)):
            mid = uv_wrap(FaceReducer()(high.copy(), max_facenum=faces), timeout=limit)
            if mid is not None:
                break
        else:
            faces = 20_000
            mid = grid_wrap(FaceReducer()(high.copy(), max_facenum=faces))
            status["uv"] = "grid"
        status["paint_faces"] = faces
        painter(mid, image=cutout).export(d / "paint.glb")
        status["paint_s"] = round(time.time() - t)
        status["ok"] = True
    except Exception as exc:  # keep the batch going; the status file says what broke
        status["ok"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["trace"] = traceback.format_exc()
        torch.cuda.empty_cache()
    status["total_s"] = round(time.time() - t0)
    (d / "status.json").write_text(json.dumps(status, indent=1))
    log(f"[{n}/{len(props)}] {p['id']}: " + ("ok" if status["ok"] else status["error"]) + f" ({status['total_s']} s)")


def cutting():
    """make_cutouts.py is still running, so more cut-outs may arrive."""
    return subprocess.run(["pgrep", "-f", "make_cutouts.py"], capture_output=True).returncode == 0


# A prop is due when its cut-out is newer than its painted mesh (new, or re-cut since). Checked
# afresh each round, so cut-outs that land while the batch runs are picked up.
cut = {p["id"]: Path(args.cutouts) / f"{p['id']}.png" for p in props}
tried, n = set(), 0
while True:
    due = [p for p in props if p["id"] not in tried and cut[p["id"]].exists()
           and time.time() - cut[p["id"]].stat().st_mtime > 5
           and (args.force or newer(cut[p["id"]], Path(args.out) / p["id"] / "paint.glb"))]
    if not due:
        if args.follow and cutting():
            time.sleep(15)
            continue
        break
    tried.add(due[0]["id"])
    n += 1
    make(due[0], n)
missing = [p["id"] for p in props if not cut[p["id"]].exists()]
log("batch done" + (f"; no cut-out for {', '.join(missing)}" if missing else ""))
