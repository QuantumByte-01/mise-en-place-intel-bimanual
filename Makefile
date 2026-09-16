.PHONY: scene smoke demo live live-t2 t2 video eval stress audit demos train export hybrid bench vlm-grounding multimodal install

install:
	pip install -r requirements.txt
	pip install -e .

scene:
	python -c "from mise_en_place.sim.scene_builder import build_scene; print(build_scene())"

smoke:
	python scripts/smoke.py

demo:
	python scripts/demo_t1.py --seed 0

live:
	python scripts/live_demo.py --seed 0

live-t2:
	python scripts/live_demo.py --task t2 --seed 0

t2:
	python scripts/eval_t2_handoff.py --episodes 10

video:
	python scripts/record_demo.py --task t1 --seed 0
	python scripts/record_demo.py --task t2 --seed 0

eval:
	python scripts/eval_10seed.py --n 10

stress:
	python scripts/eval_10seed.py --n 100
	python scripts/eval_t2_handoff.py --episodes 50

audit:
	python scripts/audit_contacts.py --tasks t1 --episodes 20 --output artifacts/contact_audit_t1.json
	python scripts/audit_contacts.py --tasks t2 --episodes 50

demos:
	python scripts/record_demos.py --episodes 40

train:
	python scripts/train_lora_bc.py --base-epochs 40 --lora-epochs 30

export:
	python scripts/export_openvino_int8.py

hybrid:
	python scripts/eval_hybrid_sweep.py --episodes 10

bench:
	python scripts/benchmark_inference_matrix.py --iterations 500 --repeats 3

vlm-grounding:
	python scripts/eval_vlm_grounding.py
	python scripts/eval_vlm_binary.py

multimodal:
	python scripts/run_multimodal_episode.py --seed 1000
