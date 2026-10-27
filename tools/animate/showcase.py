"""One labelled montage of every creature action's preview: out/review/anims/showcase.mp4.

    python3 tools/animate/showcase.py

Reads catalog/creature_anims.json for the order and the labels (creature, action, speed, one-shot)
and the per-action MP4s that review_anims.py --mp4 writes; needs ffmpeg on PATH.
"""
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "out" / "review" / "anims"


def label(pid, name, act):
    words = [pid.replace("_", " "), "-", name.replace("_", " ")]
    if act.get("one_shot"):
        words.append("(once)")
    return " ".join(words)


def main():
    cfg = json.loads((ROOT / "catalog" / "creature_anims.json").read_text())
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for pid, spec in cfg["creatures"].items():
        for name, act in spec["actions"].items():
            src = DIR / f"{pid}.{name}.mp4"
            if not src.exists():
                print("missing", src.name)
                continue
            out = tmp / f"{len(parts):03d}.mp4"
            text = label(pid, name, act).replace(":", r"\:")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf",
                            f"scale=640:640,drawtext=text='{text}':x=16:y=16:fontsize=26:fontcolor=white:"
                            "box=1:boxcolor=black@0.5:boxborderw=8",
                            "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
            parts.append(out)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    dst = DIR / "showcase.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(dst)], check=True)
    print(f"wrote {dst} from {len(parts)} clips")


if __name__ == "__main__":
    main()
