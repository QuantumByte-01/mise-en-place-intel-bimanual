# Submission package

## Included in this repo

- Full source code (`mise_en_place/`, `scripts/`, `tests/`)
- Trained policy: `checkpoints/lora_bc.pt` + OpenVINO IRs in `checkpoints/openvino/`
- Evaluation artifacts: JSON/CSV under `artifacts/`
- Submission video: `artifacts/submission_demo.mp4`
- Slides: `artifacts/mise_en_place_slides.pdf`

## Download after clone

```powershell
pip install -e ".[all]"
python scripts/download_vlm.py
python scripts/record_demos.py --episodes 40   # optional: regenerate training data
```

## Quick verify

```powershell
python scripts/smoke.py
python scripts/verify_submission.py
python scripts/eval_10seed.py --n 10
```
