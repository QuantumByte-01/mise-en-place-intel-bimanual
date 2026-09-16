"""Fail fast if measured submission claims and required assets diverge."""

from __future__ import annotations

import json

from mise_en_place import config as C


def load(name: str) -> dict:
    return json.loads((C.ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> None:
    t1 = load("eval_100seed_summary.json")
    t2 = load("eval_t2_handoff.json")
    contact_t1 = load("contact_audit_t1.json")["t1"]
    contact_t2 = load("contact_audit.json")["t2"]
    inference = load("inference_matrix.json")
    hybrid = load("hybrid_precision_sweep.json")
    multimodal = load("multimodal_episode.json")
    vlm_binary = load("vlm_binary_eval.json")

    checks = {
        "t1_100_of_100": t1["n"] == 100 and t1["success_rate"] == 1.0,
        "t2_50_of_50": t2["episodes"] == 50 and t2["success_rate"] == 1.0,
        "t1_collision_free": contact_t1["collision_free_rate"] == 1.0
        and contact_t1["penetration_free_rate"] == 1.0,
        "t2_collision_free": contact_t2["episodes"] == 50
        and contact_t2["collision_free_rate"] == 1.0
        and contact_t2["penetration_free_rate"] == 1.0,
        "inference_matrix_complete": not inference["errors"],
        "static_int8_hybrid_50pct_10_of_10": hybrid["results"]["int8"]["0.50"][
            "success_rate"
        ]
        == 1.0,
        "multimodal_episode_succeeded": multimodal["score"]["success"],
        "vlm_not_overtrusted": vlm_binary["binary_accuracy"] < 0.875,
        "t1_video_present": (C.ARTIFACTS / "demo_t1.mp4").exists(),
        "t2_video_present": (C.ARTIFACTS / "demo_t2.mp4").exists(),
        "submission_reel_present": (C.ARTIFACTS / "submission_demo.mp4").exists(),
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "ready": not failures,
        "checks": checks,
        "failures": failures,
        "required_disclosures": [
            "NPU unavailable and unmeasured on this host",
            "pure BC is not the submitted control mode",
            "SmolVLM is proposal-only after failed grounding stress tests",
        ],
    }
    output = C.ARTIFACTS / "submission_readiness.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
