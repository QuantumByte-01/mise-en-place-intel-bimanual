"""10-seed T1 evaluation harness (rubric: robustness 15 pts)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.sim.env import DinnerTableEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rows = []
    env = DinnerTableEnv()
    for i in range(args.n):
        seed = args.start_seed + i
        env.reset(seed=seed)
        expert = ScriptedT1Expert(env)
        score = expert.run_parallel_t1()
        row = {
            "seed": seed,
            "success": int(score["success"]),
            "fork": int(score["per_object"]["fork"]),
            "spoon": int(score["per_object"]["spoon"]),
            "err_fork": round(score["errors"]["fork"], 4),
            "err_spoon": round(score["errors"]["spoon"], 4),
            "steps": score["steps"],
        }
        rows.append(row)
        print(row)

    env.close()
    csv_path = args.output or C.ARTIFACTS / f"eval_{args.n}seed.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n": args.n,
        "success_rate": sum(r["success"] for r in rows) / max(args.n, 1),
        "csv": str(csv_path),
    }
    csv_path.with_name(f"{csv_path.stem}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
