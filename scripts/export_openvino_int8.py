"""Export the LoRA BC student to ONNX/OpenVINO and run real INT8 PTQ."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch

from mise_en_place import config as C
from mise_en_place.learning import load_policy


def model_size_bytes(xml_path: Path) -> int:
    bin_path = xml_path.with_suffix(".bin")
    return xml_path.stat().st_size + (bin_path.stat().st_size if bin_path.exists() else 0)


def infer_batch(compiled: ov.CompiledModel, observations: np.ndarray) -> np.ndarray:
    result = compiled({compiled.input(0): observations.astype(np.float32)})
    return np.asarray(result[compiled.output(0)])


def benchmark(
    core: ov.Core,
    model_path: Path,
    device: str,
    samples: np.ndarray,
    iterations: int,
) -> dict:
    started = time.perf_counter()
    compiled = core.compile_model(model_path, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_ms = (time.perf_counter() - started) * 1000
    for index in range(min(20, iterations)):
        infer_batch(compiled, samples[index % len(samples) : index % len(samples) + 1])
    latencies = []
    for index in range(iterations):
        sample = samples[index % len(samples) : index % len(samples) + 1]
        started = time.perf_counter_ns()
        infer_batch(compiled, sample)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(latencies)
    return {
        "device": device,
        "compile_ms": round(compile_ms, 3),
        "mean_ms": round(float(values.mean()), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "throughput_fps": round(float(1000.0 / values.mean()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "lora_bc.pt",
    )
    parser.add_argument("--data", type=Path, default=C.ARTIFACTS / "expert_demos.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=C.REPO_ROOT / "checkpoints" / "openvino",
    )
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--devices", default="CPU,GPU,NPU")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    observations = data["observations"].astype(np.float32)
    expert_actions = data["actions"].astype(np.float32)
    calibration_count = min(args.calibration_samples, len(observations))

    torch_model = load_policy(args.checkpoint, use_lora=True)
    sample = torch.from_numpy(observations[:1])
    onnx_path = args.output_dir / "lora_bc.onnx"
    torch.onnx.export(
        torch_model,
        sample,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    core = ov.Core()
    ov_model = core.read_model(onnx_path)
    fp32_path = args.output_dir / "lora_bc_fp32.xml"
    ov.save_model(ov_model, fp32_path, compress_to_fp16=False)

    calibration_rows = [row[None, :] for row in observations[:calibration_count]]
    calibration_dataset = nncf.Dataset(calibration_rows)
    int8_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        subset_size=calibration_count,
        preset=nncf.QuantizationPreset.PERFORMANCE,
    )
    int8_path = args.output_dir / "lora_bc_int8.xml"
    ov.save_model(int8_model, int8_path, compress_to_fp16=False)

    compiled_fp32 = core.compile_model(fp32_path, "CPU")
    compiled_int8 = core.compile_model(int8_path, "CPU")
    comparison_count = min(1024, len(observations))
    comparison_obs = observations[:comparison_count]
    target = expert_actions[:comparison_count]
    fp32_actions = infer_batch(compiled_fp32, comparison_obs)
    int8_actions = infer_batch(compiled_int8, comparison_obs)
    fp32_relative = fp32_actions[:, :6] - fp32_actions[:, 6:]
    int8_relative = int8_actions[:, :6] - int8_actions[:, 6:]

    requested_devices = [item.strip().upper() for item in args.devices.split(",") if item.strip()]
    available = set(core.available_devices)
    benches: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for precision, path in (("FP32", fp32_path), ("INT8", int8_path)):
        for device in requested_devices:
            key = f"{precision}_{device}"
            if device not in available:
                failures[key] = "device unavailable"
                continue
            try:
                benches[key] = benchmark(
                    core,
                    path,
                    device,
                    observations,
                    args.iterations,
                )
            except Exception as exc:  # noqa: BLE001
                failures[key] = f"{type(exc).__name__}: {exc}"

    metrics = {
        "runtime_version": ov.get_version(),
        "available_devices": sorted(available),
        "checkpoint": str(args.checkpoint),
        "calibration_samples": calibration_count,
        "models": {
            "onnx": str(onnx_path),
            "fp32_ir": str(fp32_path),
            "int8_ir": str(int8_path),
            "fp32_bytes": model_size_bytes(fp32_path),
            "int8_bytes": model_size_bytes(int8_path),
        },
        "offline_accuracy": {
            "fp32_vs_expert_mae": float(np.mean(np.abs(fp32_actions - target))),
            "int8_vs_expert_mae": float(np.mean(np.abs(int8_actions - target))),
            "int8_vs_fp32_mae": float(np.mean(np.abs(int8_actions - fp32_actions))),
            "int8_vs_fp32_max_abs": float(np.max(np.abs(int8_actions - fp32_actions))),
            "coordination_relative_action_mae": float(
                np.mean(np.abs(int8_relative - fp32_relative))
            ),
        },
        "latency": benches,
        "skipped_or_failed": failures,
        "hardware_note": (
            "NPU measurements require a Core Ultra machine; this host exposes "
            f"{', '.join(sorted(available))}."
        ),
    }
    output_path = C.ARTIFACTS / "openvino_int8_results.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
