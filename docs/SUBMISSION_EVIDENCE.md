# Submission evidence — 2026-09-14

## Verdict

This is a strong, reproducible **verified hierarchical multimodal manipulation**
submission candidate. The reliable contribution is the combination of:

1. a real image-language proposal model,
2. a semantic validation and fallback boundary,
3. collision-free bimanual task execution,
4. a bounded LoRA behavior-cloning residual, and
5. measured OpenVINO precision/latency/closed-loop trade-offs.

Do not describe it as a fully learned end-to-end VLA policy. The pure BC policy
and compact VLM are not reliable enough to own the safety boundary.

## Independently re-run evidence

- T1 parallel placement: **100/100** seeds (95% Wilson lower bound: 96.3%).
- T2 right-to-left handoff: **50/50** seeds (95% Wilson lower bound: 92.9%).
- T1 final error: fork mean 4.11 mm / max 13.9 mm; spoon mean 3.90 mm /
  max 27.6 mm.
- T2 final placement error: mean 6.43 mm / max 40.85 mm.
- One T1 episode used the explicit sequential retry fallback; all 100 completed.
- Contact audit: 20/20 T1 and 50/50 T2 had zero inter-arm contact and zero
  penetration steps.
- Maximum plate displacement in the 50-episode T2 contact audit: 9.18 mm.
- No audited episode dropped an object below the table.

Source artifacts:

- `artifacts/eval_100seed.csv`
- `artifacts/eval_100seed_summary.json`
- `artifacts/eval_t2_handoff.json`
- `artifacts/contact_audit_t1.json`
- `artifacts/contact_audit.json` (50-episode T2)

## Visual evidence

- `artifacts/submission_demo.mp4` — 16.2-second labeled T1 + T2 reel.
- `artifacts/demo_t1.mp4` — fresh seed-7 T1 recording.
- `artifacts/demo_t2.mp4` — fresh seed-7 collision-free handoff.
- `artifacts/demo_t1_keyframes/overhead_final.png`
- `artifacts/demo_t2_keyframes/overhead_final.png`
- `artifacts/multimodal_episode_start.png`
- `artifacts/multimodal_episode_final.png`

Manual frame inspection confirmed both utensils are visibly lifted and moved,
the corrected T2 grippers remain separated during transfer, and final object
positions agree with the numeric scores. The contact audit found that the
original T2 implementation penetrated the opposing gripper; that implementation
was replaced with far-end grasping and measured TCP-to-object offset
compensation before these final results were recorded.

## OpenVINO inference decision

Recommended demo configuration:

- Intel iGPU: NNCF static INT8, 50% bounded authority.
- CPU-only fallback: OpenVINO FP32.
- Safety: deterministic gripper commands and residual clipping at 0.15 rad.

Repeated benchmark medians (three runs of 500 synchronous requests):

- OpenVINO FP32 CPU p50: 0.148 ms.
- ONNX Runtime CPU p50: 0.070 ms (fastest CPU runtime for this tiny MLP).
- OpenVINO FP32 iGPU median mean: 3.131 ms.
- OpenVINO static INT8 iGPU median mean: 1.192 ms (2.63x faster).
- iGPU asynchronous throughput: FP32 521 FPS; static INT8 1,189 FPS (2.28x).
- Static INT8 size: 102,205 B vs FP32 142,826 B (28.4% reduction).
- Weight-only INT8 size: 89,346 B; FP32 drift MAE 0.00106.
- Weight-only INT4 size: 75,159 B; FP32 drift MAE 0.01142.
- Static INT8 FP32 drift MAE: 0.01823.

Closed-loop, 10 matched seeds:

- 50% authority success: FP32 8/10, FP16 9/10, weight INT8 9/10,
  weight INT4 10/10, static INT8 10/10.
- 75% authority success: FP32 2/10, FP16 5/10, weight INT8 6/10,
  weight INT4 3/10, static INT8 8/10.

The static INT8 model is less accurate offline but more stable in this bounded
closed-loop sweep. Describe this as a **quantization-induced regularization
hypothesis** backed by this experiment, not as a universal property.

Source artifacts:

- `artifacts/inference_matrix.json`
- `artifacts/hybrid_precision_sweep.json`
- `artifacts/openvino_int8_results.json`

## Multimodal evidence and trust boundary

`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (Apache-2.0, revision
`067788b187b95ebe7b2e040b3e4299e342e5b8fd`) consumed a real overhead frame and
natural-language instruction. On seed 1000 it emitted strict task JSON, passed
the validator without fallback, and the resulting T1 execution succeeded with
1.17 mm fork and 4.65 mm spoon error.

However:

- Structured five-scene counterfactual grounding: 0% exact scene accuracy.
- Binary object-presence counterfactuals: 50%; the model always answered "No."
- CPU proposal latency: about 50.7 seconds.

Therefore the VLM remains proposal-only. The gate rejects invalid JSON,
unsupported actions, hallucinated/absent objects, duplicate objects, missing
subtasks, and unsafe arm assignments. The deterministic planner is the fallback.

Source artifacts:

- `artifacts/multimodal_episode.json`
- `artifacts/vlm_grounding_eval.json`
- `artifacts/vlm_binary_eval.json`

## Judge reproduction path

```powershell
pip install -e ".[all]"
python scripts/smoke.py
python scripts/eval_10seed.py --n 10 --start-seed 1000
python scripts/eval_t2_handoff.py --episodes 10 --start-seed 1000
python scripts/audit_contacts.py --episodes 10
python scripts/benchmark_inference_matrix.py --iterations 500 --repeats 3
python scripts/live_demo.py --task t1 --seed 7
python scripts/live_demo.py --task t2 --seed 7
```

The multimodal episode additionally requires the checkpoint under
`checkpoints/smolvlm2-256m`; see the README for the exact Hugging Face model ID.

## Remaining external requirement

This machine exposes Intel CPU and GPU only. NPU measurements remain unverified
and must be run on a Core Ultra machine or Intel AI PC Cloud. Never substitute
projected NPU numbers for measurements.
