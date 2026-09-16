param(
    [int]$Episodes = 120,
    [int]$EvalSeeds = 10
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "=== 1/8 Record expert demonstrations ($Episodes episodes) ==="
python scripts/record_demos.py --episodes $Episodes
if ($LASTEXITCODE -ne 0) { throw "Demonstration recording failed" }

Write-Host "=== 2/8 Prepare augmented training set (+ optional HF) ==="
pip install -q datasets 2>$null
python scripts/prepare_training_data.py --augment-copies 2 --hf-samples 2048
if ($LASTEXITCODE -ne 0) { throw "Training data preparation failed" }

Write-Host "=== 3/8 Competition training (base + LoRA + joint fine-tune) ==="
python scripts/train_lora_bc.py --mode competition --base-epochs 60 --lora-epochs 40 --joint-epochs 30 --hidden-dim 256 --rank 16
if ($LASTEXITCODE -ne 0) { throw "Training failed" }

Write-Host "=== 4/8 Export OpenVINO INT8 ==="
python scripts/export_openvino_int8.py
if ($LASTEXITCODE -ne 0) { throw "OpenVINO export failed" }

Write-Host "=== 5/8 Hybrid closed-loop sweep ==="
python scripts/eval_hybrid_sweep.py --episodes $EvalSeeds --output artifacts/hybrid_competition_sweep.json
if ($LASTEXITCODE -ne 0) { throw "Hybrid evaluation failed" }

Write-Host "=== 6/8 Pure learned closed-loop eval ==="
python scripts/eval_learned_policy.py --runtime int8 --episodes $EvalSeeds
if ($LASTEXITCODE -ne 0) { throw "Learned policy evaluation failed" }

Write-Host "=== 7/8 Record hybrid learned MP4 ==="
python scripts/record_learned_demo.py --mode hybrid --blend 0.5 --runtime int8 --seed 100
if ($LASTEXITCODE -ne 0) { throw "Learned demo video failed" }

Write-Host "=== 8/8 Scripted baseline videos (submission) ==="
python scripts/record_demo.py --task t1 --seed 7
python scripts/record_demo.py --task t2 --seed 7
if ($LASTEXITCODE -ne 0) { throw "Baseline video failed" }

Write-Host "Competition training pipeline complete."
Write-Host "  checkpoints/lora_bc.pt"
Write-Host "  artifacts/training_data.npz"
Write-Host "  artifacts/hybrid_competition_sweep.json"
Write-Host "  artifacts/learned_hybrid_int8_seed100.mp4"
Write-Host "Run live viewer: python scripts/live_learned_vla.py --mode hybrid --blend 0.5 --no-vlm"
