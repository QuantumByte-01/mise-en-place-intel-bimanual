"""OpenVINO device probe + placeholder latency table (Intel rubric path)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mise_en_place import config as C


def probe_openvino() -> dict:
    info: dict = {"ok": False, "devices": [], "error": None}
    try:
        from openvino import Core  # type: ignore

        core = Core()
        info["devices"] = list(core.available_devices)
        info["ok"] = True
        info["versions"] = {d: str(core.get_versions(d)) for d in info["devices"]}
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> None:
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ov = probe_openvino()
    # Placeholder bench rows — replace with real IR timing on Core Ultra / AI PC Cloud
    table = [
        {"precision": "FP32", "device": "CPU", "latency_ms": None, "notes": "fill after export"},
        {"precision": "FP16", "device": "GPU", "latency_ms": None, "notes": "iGPU when available"},
        {"precision": "INT8", "device": "NPU", "latency_ms": None, "notes": "Core Ultra NPU only"},
    ]
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "openvino": ov,
        "latency_table": table,
        "coordination_collapse": {
            "metric": "inter_arm_sync_error_ms / relative_pose_err",
            "status": "stub — populate when quantized policy exists",
            "hypothesis": "T2 handoff degrades under INT8 before T1 single-arm success",
        },
    }
    path = C.ARTIFACTS / "openvino_bench.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
