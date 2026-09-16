"""Run a real compact vision-language planner on a rendered scene image."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mise_en_place import config as C
from mise_en_place.planner import plan_from_instruction


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "smolvlm2-256m",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=C.ARTIFACTS / "demo_t2_keyframes" / "overhead_start.png",
    )
    parser.add_argument(
        "--instruction",
        default="Set the table by placing the fork left of the plate and spoon right of the plate.",
    )
    parser.add_argument("--mode", choices=("plan", "verify"), default="plan")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if args.mode == "plan":
        prompt_text = f"""You are the visual planner for two robot arms.
Inspect the image, then obey this instruction: {args.instruction}
Return only JSON with this schema:
{{
  "scene": {{"plate": true, "fork": true, "spoon": true, "mug": true}},
  "subtasks": [
    {{"action": "pick_place", "object": "fork", "arm": "left", "target": "plate_left"}},
    {{"action": "pick_place", "object": "spoon", "arm": "right", "target": "plate_right"}}
  ],
  "reason": "short visual reason"
}}
Do not add markdown."""
    else:
        prompt_text = f"""You are an independent visual verifier for a robot task.
Inspect only the image and evaluate this requirement: {args.instruction}
Return only JSON with this schema:
{{
  "objects_visible": {{"plate": true, "fork": true, "spoon": true, "mug": true}},
  "requirement_satisfied": true,
  "evidence": "short spatial explanation",
  "confidence": 0.0
}}
Set confidence between 0 and 1. Do not add markdown."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    infer_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    infer_seconds = time.perf_counter() - infer_started
    generated_only = generated[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    parsed = extract_json(raw)
    fallback = parsed is None and args.mode == "plan"
    if parsed is None and args.mode == "plan":
        parsed = json.loads(plan_from_instruction(args.instruction).to_json())

    report = {
        "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "model_revision": "067788b187b95ebe7b2e040b3e4299e342e5b8fd",
        "image": str(args.image),
        "instruction": args.instruction,
        "mode": args.mode,
        "raw_output": raw,
        "parsed_plan": parsed,
        "strict_json_success": not fallback,
        "fallback_used": fallback,
        "load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "device": "CPU",
    }
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = C.ARTIFACTS / f"smolvlm_{args.mode}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
