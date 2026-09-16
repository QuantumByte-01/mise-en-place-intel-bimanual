# Learned policy and Intel deployment

## What is trained

`LoRABCPolicy` is a 28k-parameter observation-to-action student. Its input is
joint state, TCP positions, object positions, holding flags and normalized
episode phase; its output is the 12 joint-position targets for both SO-101 arms.

Training has two explicit stages:

1. Train the dense MLP on randomized expert demonstrations.
2. Freeze all dense weights and adapt only rank-8 LoRA matrices on held-out
   domain-randomized episodes.

LoRA has 4,528 parameters (16.1% of this deliberately tiny student) and reduced
held-out MAE from 0.01325 to 0.01227 in the recorded run.

## What it is not

This student is not SmolVLA and has no image or language encoder. It validates
the complete data, low-rank adaptation, ONNX, OpenVINO and INT8 deployment
plumbing on local hardware. Language reasoning remains at the planner boundary;
a SmolVLA checkpoint can later implement the same 12-DoF policy interface.

## Deployment modes

- `scripted`: geometric executive only; submission fallback.
- `pure learned`: LoRA model controls all joints; currently 0/10 closed-loop.
- `hybrid`: model inference runs every step as a bounded joint residual;
  deterministic grasp/release timing stays with the executive.

This separation prevents offline imitation loss from being mistaken for robotic
task success.

## Quantization

`export_openvino_int8.py` performs:

1. PyTorch to ONNX export.
2. ONNX to FP32 OpenVINO IR.
3. NNCF static post-training quantization with 512 calibration samples.
4. FP32/INT8 output-drift and bimanual relative-action error.
5. Latency on every requested available Intel device.

The local machine exposes CPU and Intel GPU, but no NPU. Core Ultra NPU results
must be collected on Intel AI PC Cloud or another Core Ultra host.
