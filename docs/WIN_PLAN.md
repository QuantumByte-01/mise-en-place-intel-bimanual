# Hackathon A — Intel Bimanual VLA (Win Plan)

**Track:** LabLab AI Infra Summit — **Intel Bimanual VLA Manipulation with Multi-Modal Reasoning** (online only)  
**Deadline:** Wed **16 Sep 2026, 18:30 UTC** (≈ midnight IST into Sep 17)  
**Brief:** [Google Drive PDF](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view)  
**Codename:** **Mise en Place**

---

## Problem statement (FINAL)

Natural-language household manipulation with **two arms** fails when policies treat the scene as a single-arm script: one grasp can succeed while hand-offs, drawer state, and complementary actions (“hold mug / pour”) fail. In a **MuJoCo dual SO-101** dinner-table environment, build an end-to-end **Physical AI** system that (1) interprets multi-step NL instructions with a **VLM planner**, (2) reasons over camera observations, (3) coordinates both arms through drawer opening, utensil/plate retrieval, placement, and **at least one hand-off or complementary dual-arm action**, and (4) deploys control + planner with **OpenVINO** on **Intel Core Ultra Series 2/3**, reporting latency, device, and precision across **10 randomized seeds** — including whether **inter-arm coordination collapses under quantization** before single-arm success does.

---

## Why this wins the rubric (/100)

| Rubric | Pts | How Mise en Place scores |
|---|---|---|
| E2E + bimanual | 30 | Full table workflow; **forced hand-off (T2)** |
| VLA / multi-modal reasoning | 20 | Qwen2.5-VL planner: instruction + frame → JSON subtasks + arm assignment + verify/retry |
| OpenVINO + Core Ultra | 20 | `physical-ai-studio` export + NNCF INT8; bench on AI PC Cloud |
| Robustness (10 seeds) | 15 | Randomization harness = daily eval loop |
| Reproducibility | 10 | Docker, HF datasets/ckpts, `make eval` |
| Innovation | 5 | **Coordination-collapse under quant** (handover sync error FP32→INT8→INT4) |

**Win thesis:** Judges remember *two arms + language planner + Intel numbers + coordination-under-quant plot* — not “I ran SmolVLA once.”

---

## Literature / landscape (crowded vs open)

| Finding | Implication |
|---|---|
| Intel ships [`physical-ai-studio`](https://github.com/open-edge-platform/physical-ai-studio) + [`openvinotoolkit/physicalai`](https://github.com/openvinotoolkit/physicalai); π0.5-on-Core-Ultra article exists | OpenVINO export is **expected baseline**, not innovation |
| SO-101 MJCF exists ([TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)); public MuJoCo demos are **single-arm** | Dual-arm dinner scene + demos = OSS gap you fill |
| HoloQ-VLA, DyQ-VLA, ActQuant, QuantVLA, Mix-QVLA, EaqVLA | “We quantized a VLA” is **crowded**; all mostly **single-arm / NVIDIA** |
| **Open:** does quant break **inter-arm coordination** before single-arm success — on **OpenVINO/NPU**? | Your innovation + optional workshop follow-up |

---

## Locked architecture

```
NL command (+ optional Speechmatics)
        │
        ▼
┌─────────────────────────────┐
│ Layer 1 — Reason (20 pts)   │  Qwen2.5-VL-3B INT4 (OpenVINO GenAI / NPU)
│ JSON plan: subtasks, arms,  │  verify after each subtask → 1 retry
│ object targets              │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Layer 2 — Act (30 pts)      │  SmolVLA-450M fine-tuned (multi-cam)
│ dual SO-101 actions         │  Safety net: ACT skill checkpoints if SmolVLA weak
└──────────────┬──────────────┘
               ▼
        MuJoCo dual SO-101 dinner table
               │
               ▼
   OpenVINO IR + INT8 PTQ → Core Ultra bench
```

| Component | Choice | Notes |
|---|---|---|
| Planner | Qwen2.5-VL-3B-Instruct OV-INT4 | NPU-oriented HF builds exist |
| Policy | `lerobot/smolvla_base` (~450M) | Named in brief; SO-101 community |
| Safety net | ACT (~80M) per skill | Train from day 2 on same demos |
| Skip for train | Pi0.5 (~3B) | Too heavy for 6 days |
| Demos | Scripted mink IK + domain rand | No teleop hardware needed |
| Assets | Duplicate SO-101 MJCF with L/R prefixes | Table + drawer + dishware |

### Task tiers (cut from the bottom if late)

| Tier | Task | Priority |
|---|---|---|
| **T1** | Parallel place: fork left + spoon right beside plate | **Must work** |
| **T2** | Cross-table hand-off (object starts wrong side) | Showpiece for 30 pts |
| **T3** | Bimanual plate/tray carry | Stretch — cut freely |

---

## Hardware (critical)

- Local laptop **Alder Lake** (Intel CPU + Iris Xe): OK for OpenVINO **dev**, **not** Core Ultra / NPU.  
- **Tonight:** request [Intel AI PC Cloud / Tiber](https://ai.cloud.intel.com/) Core Ultra Series 2 instance.  
- Fallback: honest CPU+iGPU bench + cite Intel’s published Core Ultra π0.5 numbers; Discord mentors for partner access.  
- Training: Kaggle T4 / Colab / any cloud — Intel does **not** provide training GPUs.

---

## Day-by-day plan

| Day | Deliverable | Exit criterion |
|---|---|---|
| **Tonight Sep 10** | AI PC Cloud request; LabLab enroll; Discord; repo skeleton | Access request submitted |
| **Sep 11** | Dual SO-101 scene + mink single pick-place | Scene loads; one arm pick-place |
| **Sep 12** | Scripted T1 demos (300+) → HF; ACT training starts | Dataset on Hub |
| **Sep 13** | SmolVLA fine-tune; 10-seed harness; ACT smoke results | Both arms move on NL subtask |
| **Sep 14** | OpenVINO export + INT8; planner wiring; T2 if on track | E2E on laptop |
| **Sep 15** | Core Ultra bench; coordination-collapse plot; 10-seed videos; README+Docker | Rubric evidence pack |
| **Sep 16** | Buffer; optional Speechmatics; **submit early** | Before 18:30 UTC |

**Iron rule:** boring working **T1 by Sep 13** beats ambitious broken T2 on Sep 16.

---

## Deliverables checklist

1. Reproducible GitHub (setup, assets, train/eval/infer)  
2. MuJoCo dual-arm dinner scene + randomization  
3. Intel bench script (latency, throughput, device, precision)  
4. Demo video — **10 randomized seeds** + success rate  
5. Architecture README (planner + policy + OpenVINO map)  
6. Optional: Speechmatics voice → planner (bonus prize only)

### Demo video script (~90s)

1. NL command on screen  
2. Randomized initial scene  
3. Planner JSON flash (arm assignments)  
4. Dual-arm hand-off visible  
5. Final table state  
6. 10-seed success bar + OpenVINO latency table + coordination-collapse slide  

---

## Risks / mitigations

| Risk | Mitigation |
|---|---|
| No Core Ultra | AI PC Cloud tonight; Discord; transparent Alder Lake fallback |
| Dinner table too hard | Ship T1+one hand-off; cut utensils/T3 |
| SmolVLA fails | Language-routed ACT skills |
| Demo generation stuck | Start with fewer objects; script waypoints first |
| Time | Freeze assets end of Sep 11; no new objects after Sep 12 |

---

## Kill weak ideas (do not pursue)

- Training Pi0.5 from scratch this week  
- Physical SO-101 / onsite Intel robot track  
- Generic “quantize SmolVLA, report FPS” as the only innovation  
- Merging SAM grounding paper into this 6-day build
