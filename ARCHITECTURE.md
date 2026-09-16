# Architecture — Mise en Place

```
NL instruction
  + overhead RGB
    → SmolVLM2-256M proposal (real image-language inference)
    → schema + object-presence + arm-assignment gate
         invalid / hallucinated → deterministic planner fallback
    → task graph (T1 place | T2 handoff | T3 stretch)
    → safe bimanual executive
         geometric IK + guarded grasp/release
         collision-free far-end handoff + measured offset compensation
         + bounded LoRA BC residual (OpenVINO FP32/FP16/INT8/INT4)
    → DinnerTableEnv (MuJoCo, dual SO-101, 12-DoF)
    → live viewer / MP4 / score + contact audit + stress JSON/CSV
    → ONNX → OpenVINO IR → NNCF static / weight-only quantization
         CPU + Intel GPU measured locally
         NPU measured on Core Ultra
```

## Innovation angle

**Quantization-induced stability window:** static INT8 has worse offline action
MAE than FP32 but retained more closed-loop success at 50–75% learned authority
on the measured 10-seed sweep. Quantization appears to damp destabilizing action
residuals. This is reported as a measured hypothesis, not a general claim.

**Verified multimodal autonomy:** the compact VLM proposes task JSON, but
counterfactual tests show that it is not reliably grounded. Validation and a
deterministic fallback keep language-model errors outside the control boundary.

## Hardware note

Local Alder Lake has no NPU. CPU/iGPU results are measured; Core Ultra / NPU
remains an external hardware follow-up and must not be presented as measured.
