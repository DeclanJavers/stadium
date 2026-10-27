"""Stage 1.5 (Colab, Hi3D env): background removal, reviewed before any 3D work.

    python make_cutouts.py --manifest generated_props.json --images images --out cutouts [--ids a,b] [--force]

BiRefNet (rembg's birefnet-general) keeps white and silver objects on the white backdrop that
the default u2net model eats, but it still drops some parts it reads as a separate object (a dark
plinth under a statue, a picture frame round a shirt). The references are studio shots on pure
white, so a flood fill from the border finds everything that is clearly not backdrop: regions it
finds that BiRefNet missed are added back when they contrast strongly with the backdrop (so soft
floor shadows stay out), and specks are dropped. Pale stone on white is still ambiguous, so the
prompts avoid it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove
from scipy import ndimage

BACKDROP_TOL = 28  # max channel difference from the backdrop that still counts as backdrop
SOLID = 60         # median difference an added region needs (a shadow is fainter than this)
SPECK = 0.02       # components smaller than this share of the largest are dropped


def hybrid_alpha(rgb, alpha):
    rgb = rgb.astype(np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    diff = np.abs(rgb - np.median(border, axis=0)).max(-1)
    lab, _ = ndimage.label(diff <= BACKDROP_TOL)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    backdrop = np.isin(lab, edge[edge > 0])
    found = ndimage.binary_opening(~backdrop, iterations=2)
    added, n = ndimage.label(found & (alpha < 128))
    keep = np.zeros_like(found)
    for i in range(1, n + 1):
        region = added == i
        if np.median(diff[region]) > SOLID:
            keep |= region
    out = np.maximum(alpha, keep.astype(np.uint8) * 255)
    comp, n = ndimage.label(out > 127)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
        small = [i + 1 for i, s in enumerate(sizes) if s < SPECK * sizes.max()]
        out[np.isin(comp, small)] = 0
    return out, int(keep.sum())


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ids", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

wanted = {s for s in args.ids.split(",") if s}
props = [p for p in json.loads(Path(args.manifest).read_text())["props"] if not wanted or p["id"] in wanted]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
session = new_session("birefnet-general")
for p in props:
    src, dst = Path(args.images) / f"{p['id']}.png", out / f"{p['id']}.png"
    if not src.exists() or (dst.exists() and not args.force and dst.stat().st_mtime > src.stat().st_mtime):
        continue
    t = time.time()
    image = Image.open(src).convert("RGB")
    cut = remove(image, session=session)
    alpha, added = hybrid_alpha(np.asarray(image), np.asarray(cut.getchannel("A")))
    rgba = image.copy()
    rgba.putalpha(Image.fromarray(alpha))
    rgba.save(dst)
    print(f"{p['id']}: {time.time() - t:.1f} s, {added} px restored", flush=True)
