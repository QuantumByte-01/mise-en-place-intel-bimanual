"""Compact SmolVLM proposal planner with strict semantic validation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class SmolVLMPlanner:
    model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    revision = "067788b187b95ebe7b2e040b3e4299e342e5b8fd"

    def __init__(self, model_path: Path) -> None:
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def propose(self, instruction: str, image_path: Path) -> dict:
        image = Image.open(image_path).convert("RGB")
        request = f"""Inspect the robot-table image and plan this instruction: {instruction}
Return only compact JSON:
{{
  "scene": {{"plate": true, "fork": true, "spoon": true, "mug": true}},
  "subtasks": [
    {{"action": "pick_place", "object": "fork", "arm": "left", "target": "plate_left"}},
    {{"action": "pick_place", "object": "spoon", "arm": "right", "target": "plate_right"}}
  ]
}}
Do not add markdown."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": request}],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )
        inference_seconds = time.perf_counter() - started
        generated_only = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )[0].strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        parsed = None
        if match:
            try:
                value = json.loads(match.group(0))
                parsed = value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
        return {
            "raw": raw,
            "parsed": parsed,
            "load_seconds": self.load_seconds,
            "inference_seconds": inference_seconds,
        }


def validate_t1_proposal(
    proposal: dict | None,
    *,
    present_objects: set[str],
) -> list[str]:
    """Reject malformed, hallucinated, unsafe or incomplete T1 plans."""
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]
    subtasks = proposal.get("subtasks")
    if not isinstance(subtasks, list):
        return ["subtasks must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    expected_arms = {"fork": "left", "spoon": "right"}
    for index, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            errors.append(f"subtask {index} is not an object")
            continue
        obj = subtask.get("object")
        arm = subtask.get("arm")
        action = subtask.get("action")
        if obj not in expected_arms:
            errors.append(f"subtask {index} has unsupported object {obj!r}")
            continue
        if obj not in present_objects:
            errors.append(f"subtask {index} references absent object {obj!r}")
        if obj in seen:
            errors.append(f"duplicate object {obj!r}")
        seen.add(obj)
        if arm != expected_arms[obj]:
            errors.append(f"unsafe arm assignment for {obj!r}: {arm!r}")
        if action != "pick_place":
            errors.append(f"unsupported action for {obj!r}: {action!r}")
    missing = set(expected_arms) - seen
    if missing:
        errors.append(f"missing required objects: {sorted(missing)}")
    return errors
