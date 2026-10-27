"""Compare background removers on reference images: cutout_test.py <model> <out dir> img.png ..."""
import sys
import time
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

model, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
session = new_session(model)
for p in map(Path, sys.argv[3:]):
    t = time.time()
    remove(Image.open(p).convert("RGB"), session=session).save(out / p.name)
    print(f"{p.stem}: {time.time() - t:.1f} s", flush=True)
