"""Build a concise labeled T1 + T2 submission reel."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mise_en_place import config as C


FPS = 24


def title_card(title: str, subtitle: str, seconds: float = 1.2) -> list[np.ndarray]:
    image = Image.new("RGB", (640, 480), (23, 31, 43))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=18)
    title_box = draw.textbbox((0, 0), title, font=font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=small)
    draw.text(((640 - (title_box[2] - title_box[0])) / 2, 195), title, font=font, fill=(244, 246, 248))
    draw.text(
        ((640 - (subtitle_box[2] - subtitle_box[0])) / 2, 245),
        subtitle,
        font=small,
        fill=(155, 174, 202),
    )
    frame = np.asarray(image)
    return [frame.copy() for _ in range(round(FPS * seconds))]


def labeled_frames(path: Path, label: str) -> list[np.ndarray]:
    output = []
    for raw in imageio.get_reader(path):
        image = Image.fromarray(raw).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 30), fill=(18, 24, 34))
        draw.text((12, 8), label, font=ImageFont.load_default(size=15), fill=(245, 247, 250))
        output.append(np.asarray(image))
    return output


def pick_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"none of these exist: {[str(p) for p in paths]}")


def main() -> None:
    t1 = pick_existing(C.ARTIFACTS / "demo_t1_best.mp4", C.ARTIFACTS / "demo_t1.mp4")
    t2 = pick_existing(C.ARTIFACTS / "demo_t2_best.mp4", C.ARTIFACTS / "demo_t2.mp4")
    hybrid = C.ARTIFACTS / "learned_hybrid_int8_seed0.mp4"

    frames = title_card(
        "Mise en Place",
        "Verified multimodal bimanual manipulation on Intel OpenVINO",
        1.6,
    )
    frames += title_card("T1 — Parallel table setting", "100/100 randomized seeds", 1.0)
    frames += labeled_frames(t1, "T1  |  parallel fork + spoon placement  |  100/100")
    frames += title_card("T2 — Right-to-left handoff", "50/50 seeds; zero arm-arm contacts", 1.0)
    frames += labeled_frames(t2, "T2  |  collision-free spoon handoff  |  50/50")
    if hybrid.exists():
        frames += title_card(
            "Hybrid learned control",
            "OpenVINO INT8 LoRA residual @ 50% authority",
            1.0,
        )
        frames += labeled_frames(hybrid, "Hybrid  |  geometric guard + INT8 learned residual")
    frames += title_card(
        "Validated autonomy stack",
        "SmolVLM proposal gate + geometric safety + OpenVINO INT8",
        1.6,
    )
    output = C.ARTIFACTS / "submission_demo.mp4"
    imageio.mimsave(output, frames, fps=FPS, macro_block_size=1)
    print(f"wrote {output} ({len(frames) / FPS:.1f}s, {len(frames)} frames)")


if __name__ == "__main__":
    main()
