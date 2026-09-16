param(
    [int]$Episodes = 40,
    [int]$EvalSeeds = 10
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

python scripts/smoke.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }

python scripts/eval_10seed.py --n $EvalSeeds
if ($LASTEXITCODE -ne 0) { throw "Scripted baseline failed" }

python scripts/eval_t2_handoff.py --episodes $EvalSeeds
if ($LASTEXITCODE -ne 0) { throw "T2 handoff evaluation failed" }

python scripts/audit_contacts.py --episodes $EvalSeeds
if ($LASTEXITCODE -ne 0) { throw "Contact audit failed" }

python scripts/record_demos.py --episodes $Episodes
if ($LASTEXITCODE -ne 0) { throw "Demonstration recording failed" }

python scripts/train_lora_bc.py --base-epochs 40 --lora-epochs 30
if ($LASTEXITCODE -ne 0) { throw "LoRA training failed" }

python scripts/export_openvino_int8.py
if ($LASTEXITCODE -ne 0) { throw "OpenVINO INT8 export failed" }

python scripts/benchmark_inference_matrix.py --iterations 500 --repeats 3
if ($LASTEXITCODE -ne 0) { throw "Inference matrix failed" }

python scripts/eval_hybrid_sweep.py --episodes $EvalSeeds
if ($LASTEXITCODE -ne 0) { throw "Hybrid evaluation failed" }

python scripts/record_demo.py --task t1 --seed 7
if ($LASTEXITCODE -ne 0) { throw "T1 video failed" }

python scripts/record_demo.py --task t2 --seed 7
if ($LASTEXITCODE -ne 0) { throw "T2 video failed" }

python scripts/build_submission_video.py
if ($LASTEXITCODE -ne 0) { throw "Submission reel failed" }

Write-Host "Pipeline complete. Results are in artifacts/ and checkpoints/openvino/."
