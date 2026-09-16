"""Evaluate the guarded right-to-left spoon handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mise_en_place import config as C
from mise_en_place.control.scripted_t2 import ScriptedT2Handoff
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    env = DinnerTableEnv()
    rows = []
    for index in range(args.episodes):
        seed = args.start_seed + index
        env.reset(seed=seed)
        score = ScriptedT2Handoff(env).run_t2()
        rows.append({"seed": seed, **score})
        print(
            f"seed={seed} success={int(score['success'])} "
            f"transfer={int(score['transferred'])} placed={int(score['placed_left'])} "
            f"error={score['place_error']:.4f}"
        )
    env.close()

    report = {
        "episodes": args.episodes,
        "success_rate": sum(row["success"] for row in rows) / args.episodes,
        "transfer_rate": sum(row["transferred"] for row in rows) / args.episodes,
        "place_rate": sum(row["placed_left"] for row in rows) / args.episodes,
        "rows": rows,
    }
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = args.output or C.ARTIFACTS / "eval_t2_handoff.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
