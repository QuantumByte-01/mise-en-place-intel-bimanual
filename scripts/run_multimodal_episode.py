"""Real image + language plan -> validation gate -> bimanual execution."""

from __future__ import annotations

import argparse
import json
import time

import imageio.v2 as imageio

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.planner import plan_from_instruction
from mise_en_place.planner.smolvlm import SmolVLMPlanner, validate_t1_proposal
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--instruction",
        default="Set the table: put the fork left of the plate and spoon right of the plate.",
    )
    args = parser.parse_args()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    env = DinnerTableEnv()
    observation = env.reset(seed=args.seed)
    start_image = C.ARTIFACTS / "multimodal_episode_start.png"
    imageio.imwrite(start_image, env.render("overhead"))

    planner = SmolVLMPlanner(C.REPO_ROOT / "checkpoints" / "smolvlm2-256m")
    proposal = planner.propose(args.instruction, start_image)
    present = {
        name
        for name, position in observation.objects.items()
        if float(position[2]) >= -0.01
    }
    validation_errors = validate_t1_proposal(
        proposal["parsed"],
        present_objects=present,
    )
    fallback_used = bool(validation_errors)
    accepted_plan = (
        json.loads(plan_from_instruction(args.instruction).to_json())
        if fallback_used
        else proposal["parsed"]
    )

    score = ScriptedT1Expert(env).run_parallel_t1()
    final_image = C.ARTIFACTS / "multimodal_episode_final.png"
    imageio.imwrite(final_image, env.render("overhead"))
    env.close()

    report = {
        "instruction": args.instruction,
        "seed": args.seed,
        "start_image": str(start_image),
        "final_image": str(final_image),
        "vlm": {
            "model": planner.model_id,
            "revision": planner.revision,
            "raw": proposal["raw"],
            "parsed": proposal["parsed"],
            "load_seconds": proposal["load_seconds"],
            "inference_seconds": proposal["inference_seconds"],
        },
        "validation_errors": validation_errors,
        "fallback_used": fallback_used,
        "accepted_plan": accepted_plan,
        "score": score,
        "total_seconds": time.perf_counter() - total_started,
        "safety_note": (
            "SmolVLM is proposal-only because its counterfactual grounding test failed. "
            "Schema, object-presence and arm-assignment gates run before execution."
        ),
    }
    output = C.ARTIFACTS / "multimodal_episode.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
