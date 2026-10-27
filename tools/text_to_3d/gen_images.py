"""Stage 1 (Colab, MOSS env): one reference image per prop in catalog/generated_props.json.

    python gen_images.py --manifest generated_props.json --out images [--ids a,b] [--seed 11] [--force]

Writes images/<id>.png and images/<id>.json (the exact prompt and seed). Existing images are
kept unless --force, so a rerun only fills gaps.
"""
import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ids", default="")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

spec = json.loads(Path(args.manifest).read_text())
style = spec["style"]
wanted = {s for s in args.ids.split(",") if s}
props = [p for p in spec["props"] if not wanted or p["id"] in wanted]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT",
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")

for i, p in enumerate(props):
    png = out / f"{p['id']}.png"
    if png.exists() and not args.force:
        continue
    seed = args.seed if args.seed is not None else p.get("seed", 7)
    prompt = p["prompt"] + style["suffix"]
    negative = style["negative"] + (", " + p["negative"] if p.get("negative") else "")
    image = pipe(
        prompt,
        negative_prompt=negative,
        height=1024,
        width=1024,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        uncond_skip_layers=[22],
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    image.save(png)
    (out / f"{p['id']}.json").write_text(json.dumps({"prompt": prompt, "negative": negative, "seed": seed}, indent=1))
    print(f"[{i + 1}/{len(props)}] {p['id']} seed {seed}", flush=True)
