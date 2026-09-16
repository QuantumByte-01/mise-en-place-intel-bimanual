"""Compare Torch, ONNX Runtime and OpenVINO precision/performance modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import nncf
import numpy as np
import onnxruntime as ort
import openvino as ov
import torch

from mise_en_place import config as C
from mise_en_place.learning import load_policy


def latency_benchmark(call: Callable[[np.ndarray], np.ndarray], samples: np.ndarray, n: int) -> dict:
    for index in range(30):
        call(samples[index % len(samples) : index % len(samples) + 1])
    timings = []
    for index in range(n):
        sample = samples[index % len(samples) : index % len(samples) + 1]
        started = time.perf_counter_ns()
        call(sample)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(timings)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "sync_fps": float(1000.0 / values.mean()),
    }


def repeated_latency(
    call: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    iterations: int,
    repeats: int,
) -> dict:
    runs = [latency_benchmark(call, samples, iterations) for _ in range(repeats)]
    aggregate = {
        key: float(np.median([run[key] for run in runs]))
        for key in runs[0]
    }
    aggregate["runs"] = runs
    return aggregate


def ir_size(path: Path) -> int:
    return path.stat().st_size + path.with_suffix(".bin").stat().st_size


def save_compressed_variants(core: ov.Core, fp32_path: Path, output_dir: Path) -> dict[str, Path]:
    variants = {"fp32": fp32_path}
    fp16_path = output_dir / "lora_bc_fp16.xml"
    ov.save_model(core.read_model(fp32_path), fp16_path, compress_to_fp16=True)
    variants["fp16"] = fp16_path

    compression_modes = {
        "int8_weight": nncf.CompressWeightsMode.INT8_ASYM,
        "int4_weight": nncf.CompressWeightsMode.INT4_ASYM,
    }
    for name, mode in compression_modes.items():
        model = core.read_model(fp32_path)
        options = {"all_layers": True, "group_size": -1} if "int4" in name else {}
        compressed = nncf.compress_weights(model, mode=mode, **options)
        path = output_dir / f"lora_bc_{name}.xml"
        ov.save_model(compressed, path, compress_to_fp16=False)
        variants[name] = path

    static_int8 = output_dir / "lora_bc_int8.xml"
    if static_int8.exists():
        variants["int8_static"] = static_int8
    return variants


def openvino_result(
    core: ov.Core,
    path: Path,
    device: str,
    samples: np.ndarray,
    iterations: int,
    repeats: int,
) -> dict:
    compile_started = time.perf_counter()
    compiled = core.compile_model(path, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_ms = (time.perf_counter() - compile_started) * 1000
    input_port, output_port = compiled.input(0), compiled.output(0)

    def call(batch: np.ndarray) -> np.ndarray:
        return np.asarray(compiled({input_port: batch})[output_port])

    latency = repeated_latency(call, samples, iterations, repeats)

    throughput_compiled = core.compile_model(path, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue = ov.AsyncInferQueue(throughput_compiled)
    throughput_input = throughput_compiled.input(0)
    async_iterations = max(1000, iterations * 2)
    async_runs = []
    for _ in range(repeats):
        started = time.perf_counter()
        for index in range(async_iterations):
            sample = samples[index % len(samples) : index % len(samples) + 1]
            queue.start_async({throughput_input: sample})
        queue.wait_all()
        async_runs.append(async_iterations / (time.perf_counter() - started))
    return {
        **latency,
        "async_fps": float(np.median(async_runs)),
        "async_fps_runs": async_runs,
        "compile_ms": float(compile_ms),
        "optimal_infer_requests": int(
            throughput_compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="CPU,GPU")
    args = parser.parse_args()

    output_dir = C.REPO_ROOT / "checkpoints" / "openvino"
    fp32_path = output_dir / "lora_bc_fp32.xml"
    onnx_path = output_dir / "lora_bc.onnx"
    data = np.load(C.ARTIFACTS / "expert_demos.npz")
    observations = data["observations"].astype(np.float32)
    target = data["actions"].astype(np.float32)
    comparison = observations[:1024]
    comparison_target = target[:1024]
    core = ov.Core()
    variants = save_compressed_variants(core, fp32_path, output_dir)

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    torch_model = load_policy(C.REPO_ROOT / "checkpoints" / "lora_bc.pt")

    def torch_call(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return torch_model(torch.from_numpy(batch)).numpy()

    results["torch_cpu"] = repeated_latency(
        torch_call, observations, args.iterations, args.repeats
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def ort_call(batch: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(None, {input_name: batch})[0])

    results["onnxruntime_cpu"] = repeated_latency(
        ort_call, observations, args.iterations, args.repeats
    )

    devices = [value.strip().upper() for value in args.devices.split(",") if value.strip()]
    available = set(core.available_devices)
    accuracy: dict[str, dict] = {}
    fp32_compiled = core.compile_model(fp32_path, "CPU")
    reference = np.asarray(
        fp32_compiled({fp32_compiled.input(0): comparison})[fp32_compiled.output(0)]
    )

    for variant, path in variants.items():
        accuracy_compiled = core.compile_model(path, "CPU")
        prediction = np.asarray(
            accuracy_compiled(
                {accuracy_compiled.input(0): comparison}
            )[accuracy_compiled.output(0)]
        )
        accuracy[variant] = {
            "bytes": ir_size(path),
            "mae_vs_expert": float(np.mean(np.abs(prediction - comparison_target))),
            "mae_vs_fp32": float(np.mean(np.abs(prediction - reference))),
            "max_abs_vs_fp32": float(np.max(np.abs(prediction - reference))),
        }
        for device in devices:
            key = f"openvino_{variant}_{device.lower()}"
            if device not in available:
                errors[key] = "device unavailable"
                continue
            try:
                results[key] = openvino_result(
                    core,
                    path,
                    device,
                    observations,
                    args.iterations,
                    args.repeats,
                )
            except Exception as exc:  # noqa: BLE001
                errors[key] = f"{type(exc).__name__}: {exc}"

    report = {
        "runtime_versions": {
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "openvino": ov.get_version(),
            "nncf": nncf.__version__,
        },
        "available_devices": sorted(available),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "accuracy_and_size": accuracy,
        "performance": results,
        "errors": errors,
        "caution": (
            "This 28k-parameter policy is too small to guarantee INT8 speedups; "
            "Python/runtime overhead can dominate. Use p50/p95 and task success, "
            "not a single mean."
        ),
    }
    output = C.ARTIFACTS / "inference_matrix.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
