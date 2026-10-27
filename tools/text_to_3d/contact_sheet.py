"""A labelled grid of images for review: contact_sheet.py out.jpg img1.png img2.png ..."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CELL, COLS = 320, 6
dst, paths = sys.argv[1], [Path(p) for p in sys.argv[2:]]
rows = (len(paths) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + 22)), "white")
draw = ImageDraw.Draw(sheet)
for i, p in enumerate(paths):
    im = Image.open(p)
    if im.mode == "RGBA":  # cut-outs: show them on mid grey so the mask edge is visible
        bg = Image.new("RGB", im.size, (150, 150, 150))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    im = im.convert("RGB")
    im.thumbnail((CELL, CELL))
    x, y = (i % COLS) * CELL, (i // COLS) * (CELL + 22)
    sheet.paste(im, (x, y))
    draw.text((x + 4, y + CELL + 4), p.stem if p.stem not in ("cutout", "render") else p.parent.name, fill="black")
sheet.save(dst, quality=88)
print("wrote", dst)
