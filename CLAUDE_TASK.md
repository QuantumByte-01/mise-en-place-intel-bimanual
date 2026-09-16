# Mise en Place — Intel Bimanual VLA Hackathon (WINNING IMPLEMENTATION)

You are implementing a **winning** submission for LabLab AI Infra Summit online track:
**Intel Bimanual VLA Manipulation with Multi-Modal Reasoning**.

Deadline: **Wed 16 Sep 2026, 18:30 UTC**. Today is **Sun 13 Sep 2026** — we are BEHIND. Prioritize a boring working **T1** end-to-end over perfection.

Read `docs/WIN_PLAN.md` in this repo first.

## Iron rules
1. Working T1 (parallel fork+spoon place) > broken fancy T2.
2. Dual simulated SO-101 in MuJoCo; no physical robot required.
3. Stack: VLM planner (Qwen2.5-VL or stub JSON planner first) → control (scripted mink IK first, then ACT/SmolVLA hooks) → eval harness (10 seeds) → OpenVINO bench stubs.
4. Ship reproducible `make` / README / Docker-friendly layout.
5. Do NOT wait for Core Ultra to build the sim+policy path; add OpenVINO bench scripts that run on CPU/iGPU with honest device reporting.
6. Iterate: implement → run smoke tests → fix → continue. If something fails, simplify and keep moving.

## Phase order (do in order; complete as much as possible this session)

### Phase 0 — Repo skeleton
- Python package `mise_en_place/`
- `pyproject.toml` / `requirements.txt` (mujoco, numpy, scipy, opencv, torch optional, mink if available)
- `README.md` with architecture, rubric map, how to run
- `Makefile` with `setup`, `sim-smoke`, `demo-t1`, `eval-10seed`, `bench-openvino`
- `.gitignore`

### Phase 1 — Dual SO-101 dinner scene (CRITICAL)
- Pull/adapt SO-101 MJCF from TheRobotStudio/SO-ARM100 Simulation/SO101 (or mujoco menagerie SO-ARM100)
- Duplicate arm with prefixes `left_` / `right_`
- Table + plate + fork + spoon + mug geoms; cameras: overhead + optional wrist
- Script: `python -m mise_en_place.sim.smoke` loads scene, steps physics, saves screenshot to `artifacts/`

### Phase 2 — Scripted expert (T1) without teleop
- Use mink differential IK or simple joint waypoint controller
- T1: left arm places fork beside plate; right arm places spoon (parallel)
- Record episodes to disk in a simple format (npzs or LeRobot-compatible if feasible without heavy deps)
- Domain randomization: object XY jitter, lighting optional

### Phase 3 — NL planner interface
- `planner.py`: given instruction string + optional image path → JSON plan `{subtasks:[{action, object, arm, target}]}`
- Start with **rule-based / template planner** that parses dinner-table commands (so E2E works offline)
- Add adapter stub for Qwen2.5-VL later (HF transformers optional)

### Phase 4 — Policy hooks
- Interface `Policy.act(obs) -> action`
- Implementations: `ScriptedExpertPolicy`, stub `ACTPolicy`, stub `SmolVLAPolicy` (load if checkpoint exists else skip)
- Runner that executes plan subtasks via policy

### Phase 5 — Eval harness (15 pts)
- `eval_10seed.py`: 10 randomized seeds, success criteria for T1, CSV + per-seed video/frames
- Metrics: success rate, episode length; placeholder for coordination sync error (T2)

### Phase 6 — OpenVINO / Intel bench stubs (20 pts path)
- Script that reports available OpenVINO devices (CPU/GPU/NPU)
- Dummy latency table writer for FP32/FP16/INT8 placeholders
- Doc note: run real export on AI PC Cloud / Core Ultra when available
- Coordination-collapse eval stub comparing FP32 vs quantized action MSE / sync error if policies exist

### Phase 7 — Docs for judges
- `ARCHITECTURE.md` ASCII diagram
- Rubric checklist in README
- Demo script notes for 90s video

## Success criteria for THIS session
Must have before you stop:
1. Scene loads + smoke screenshot
2. T1 scripted episode runs (even if imperfect grasps)
3. 10-seed eval produces a CSV
4. README explains how judges reproduce

Then keep improving T1 success rate and add T2 hand-off if time.

## Constraints
- Windows machine (PowerShell). Prefer cross-platform Python.
- May not have CUDA; make CPU sim work.
- No secrets in repo.
- Prefer small focused files; no huge unrelated refactors.

Start implementing NOW. Run smoke tests after each phase. Fix failures iteratively.
