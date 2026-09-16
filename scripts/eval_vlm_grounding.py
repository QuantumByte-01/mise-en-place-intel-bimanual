"""Counterfactual visual grounding test for the compact SmolVLM planner."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mujoco
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mise_en_place import config as C
from mise_en_place.sim.env import DinnerTableEnv


OBJECTS = ("plate", "fork", "spoon", "mug")


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def create_counterfactuals(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    env = DinnerTableEnv()
    for hidden in (None, *OBJECTS):
        env.reset(seed=0)
        if hidden is not None:
            address = env.obj_qadr[hidden]
            env.data.qpos[address : address + 3] = (2.0, 2.0, -1.0)
            mujoco.mj_forward(env.model, env.data)
        name = "all_visible" if hidden is None else f"without_{hidden}"
        path = output_dir / f"{name}.png"
        Image.fromarray(env.render("overhead")).save(path)
        expected = {obj: obj != hidden for obj in OBJECTS}
        cases.append({"name": name, "image": path, "expected": expected})
    env.close()
    return cases


def main() -> None:
    model_path = C.REPO_ROOT / "checkpoints" / "smolvlm2-256m"
    image_dir = C.ARTIFACTS / "vlm_grounding_images"
    cases = create_counterfactuals(image_dir)

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    prompt_text = (
        "Inspect the image carefully. Which listed dinner-table objects are visibly present: "
        "white plate, metal fork, metal spoon, blue mug? An object moved outside the image is "
        "absent. Return only a JSON object with exactly the keys plate, fork, spoon, mug and "
        "a true or false value for each. Do not copy a default; use the pixels."
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    correct = 0
    total = 0
    exact = 0
    results = []
    for case in cases:
        image = Image.open(case["image"]).convert("RGB")
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=70, do_sample=False)
        elapsed = time.perf_counter() - started
        generated_only = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
        parsed = parse_json(raw)
        normalized = (
            {key: bool(parsed.get(key)) for key in OBJECTS}
            if parsed is not None and all(key in parsed for key in OBJECTS)
            else None
        )
        case_correct = 0
        if normalized is not None:
            case_correct = sum(
                normalized[key] == case["expected"][key] for key in OBJECTS
            )
            exact += int(case_correct == len(OBJECTS))
        correct += case_correct
        total += len(OBJECTS)
        results.append(
            {
                "case": case["name"],
                "image": str(case["image"]),
                "expected": case["expected"],
                "raw_output": raw,
                "parsed": normalized,
                "correct": case_correct,
                "inference_seconds": elapsed,
            }
        )
        print(
            f"{case['name']}: {case_correct}/{len(OBJECTS)} "
            f"parsed={normalized} expected={case['expected']}"
        )

    report = {
        "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "attribute_accuracy": correct / total,
        "exact_scene_accuracy": exact / len(cases),
        "cases": results,
        "decision": (
            "use as visual planner"
            if correct / total >= 0.9 and exact / len(cases) >= 0.8
            else "keep behind deterministic fallback; grounding is not reliable enough"
        ),
    }
    output = C.ARTIFACTS / "vlm_grounding_eval.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
