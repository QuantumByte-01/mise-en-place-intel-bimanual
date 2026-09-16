"""Test SmolVLM visual sensitivity with simple yes/no counterfactuals."""

from __future__ import annotations

import json
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mise_en_place import config as C


DESCRIPTIONS = {
    "plate": "white circular plate",
    "fork": "silver metal fork",
    "spoon": "silver spoon with a rounded bowl",
    "mug": "blue mug",
}


def main() -> None:
    model_path = C.REPO_ROOT / "checkpoints" / "smolvlm2-256m"
    image_dir = C.ARTIFACTS / "vlm_grounding_images"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    results = []
    for obj, description in DESCRIPTIONS.items():
        for case, expected in (("all_visible", "YES"), (f"without_{obj}", "NO")):
            image_path = image_dir / f"{case}.png"
            image = Image.open(image_path).convert("RGB")
            question = (
                f"Look only at the image. Is a {description} visibly present? "
                "Answer exactly YES or NO."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            elapsed = time.perf_counter() - started
            generated_only = generated[:, inputs["input_ids"].shape[1] :]
            raw = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
            normalized = "YES" if raw.upper().startswith("YES") else (
                "NO" if raw.upper().startswith("NO") else "INVALID"
            )
            results.append(
                {
                    "object": obj,
                    "case": case,
                    "expected": expected,
                    "raw_output": raw,
                    "answer": normalized,
                    "correct": normalized == expected,
                    "inference_seconds": elapsed,
                }
            )
            print(f"{obj} {case}: answer={normalized} expected={expected} raw={raw!r}")

    accuracy = sum(result["correct"] for result in results) / len(results)
    report = {
        "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "binary_accuracy": accuracy,
        "results": results,
        "decision": (
            "eligible only as a non-safety-critical planner"
            if accuracy >= 0.875
            else "do not trust for grounding; retain deterministic perception fallback"
        ),
    }
    output = C.ARTIFACTS / "vlm_binary_eval.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
