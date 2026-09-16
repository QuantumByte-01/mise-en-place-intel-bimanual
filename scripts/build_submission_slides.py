"""Build a detailed 10-slide LabLab submission deck with diagrams (PDF)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from mise_en_place import config as C

W, H = 10 * inch, 5.625 * inch
M = 0.5 * inch

INK = colors.HexColor("#1a2332")
MUTED = colors.HexColor("#5c6b7a")
ACCENT = colors.HexColor("#0071c5")
ACCENT2 = colors.HexColor("#00a3e0")
SOFT = colors.HexColor("#eef3f8")
SOFT2 = colors.HexColor("#f7f9fc")
LINE = colors.HexColor("#d8e0e8")
WHITE = colors.white
DARK = colors.HexColor("#1a2332")
BOX = colors.HexColor("#243044")


class Deck:
    def __init__(self, path: Path) -> None:
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.n = 0

    def new_page(self) -> None:
        if self.n:
            self.c.showPage()
        self.n += 1
        self.c.setFillColor(WHITE)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)
        self.c.setFillColor(SOFT)
        self.c.rect(0, H - 0.09 * inch, W, 0.09 * inch, fill=1, stroke=0)

    def footer(self) -> None:
        self.c.setFillColor(MUTED)
        self.c.setFont("Helvetica", 7.5)
        self.c.drawString(M, 0.24 * inch, "Mise en Place  |  Intel AI Infra Summit  |  LabLab")
        self.c.drawRightString(W - M, 0.24 * inch, f"Slide {self.n} / 10")

    def title_block(self, title: str, subtitle: str = "") -> float:
        self.c.setFillColor(INK)
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawString(M, H - 0.82 * inch, title)
        y = H - 1.05 * inch
        if subtitle:
            self.c.setFillColor(MUTED)
            self.c.setFont("Helvetica", 9.5)
            self.c.drawString(M, y, subtitle)
            y -= 0.14 * inch
        self.c.setStrokeColor(LINE)
        self.c.setLineWidth(0.6)
        self.c.line(M, y, W - M, y)
        return y - 0.14 * inch

    def bullets(self, items: list[str], x: float, y: float, w: float, size: float = 9.2) -> float:
        for item in items:
            self.c.setFillColor(ACCENT)
            self.c.circle(x + 3, y + 2, 2, fill=1, stroke=0)
            self.c.setFillColor(INK)
            self.c.setFont("Helvetica", size)
            words = item.split()
            line = ""
            for word in words:
                test = (line + " " + word).strip()
                if self.c.stringWidth(test, "Helvetica", size) > w - 10:
                    self.c.drawString(x + 9, y, line)
                    y -= 0.21 * inch
                    line = word
                else:
                    line = test
            if line:
                self.c.drawString(x + 9, y, line)
                y -= 0.24 * inch
        return y

    def box(self, x: float, y: float, w: float, h: float, text: str, sub: str = "", fill=SOFT) -> None:
        self.c.setFillColor(fill)
        self.c.roundRect(x, y, w, h, 5, fill=1, stroke=0)
        self.c.setFillColor(INK)
        self.c.setFont("Helvetica-Bold", 8.5)
        self.c.drawCentredString(x + w / 2, y + h - 0.2 * inch, text)
        if sub:
            self.c.setFillColor(MUTED)
            self.c.setFont("Helvetica", 7)
            words = sub.split()
            line = ""
            ty = y + h - 0.38 * inch
            for word in words:
                test = (line + " " + word).strip()
                if self.c.stringWidth(test, "Helvetica", 7) > w - 8:
                    self.c.drawCentredString(x + w / 2, ty, line)
                    ty -= 9
                    line = word
                else:
                    line = test
            if line:
                self.c.drawCentredString(x + w / 2, ty, line)

    def arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.c.setStrokeColor(ACCENT)
        self.c.setLineWidth(1.1)
        self.c.line(x1, y1, x2, y2)
        if x2 > x1:
            self.c.line(x2, y2, x2 - 5, y2 - 3)
            self.c.line(x2, y2, x2 - 5, y2 + 3)

    def save(self) -> None:
        self.c.save()


def slide1_title(d: Deck) -> None:
    d.new_page()
    d.c.setFillColor(ACCENT)
    d.c.rect(0, H - 0.15 * inch, W, 0.15 * inch, fill=1, stroke=0)
    d.c.setFillColor(INK)
    d.c.setFont("Helvetica-Bold", 30)
    d.c.drawString(M, H - 1.25 * inch, "Mise en Place")
    d.c.setFont("Helvetica", 12)
    d.c.setFillColor(MUTED)
    d.c.drawString(M, H - 1.62 * inch, "Verified hierarchical bimanual manipulation on Intel OpenVINO")
    d.c.setFillColor(INK)
    d.c.setFont("Helvetica", 10)
    lines = [
        "LabLab AI Infra Summit  |  Intel track  |  Online submission",
        "Platform: MuJoCo dual SO-101 (12-DoF) dinner-table simulation",
        "Stack: SmolVLM2-256M planning + validation gate + geometric executive + LoRA BC / INT8",
    ]
    for i, line in enumerate(lines):
        d.c.drawString(M, H - 2.15 * inch - i * 0.28 * inch, line)
    # mini pipeline preview
    y = 1.05 * inch
    labels = ["RGB+NL", "VLM", "Gate", "Exec", "OpenVINO"]
    bw = 1.35 * inch
    for i, lab in enumerate(labels):
        x = M + i * (bw + 0.18 * inch)
        d.box(x, y, bw, 0.55 * inch, lab, fill=SOFT if i % 2 == 0 else SOFT2)
        if i < len(labels) - 1:
            d.arrow(x + bw + 0.02 * inch, y + 0.27 * inch, x + bw + 0.16 * inch, y + 0.27 * inch)
    d.footer()


def slide2_problem(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Problem, tasks, and research question", "What are we trying to solve and measure?")
    left_w = (W - 2 * M - 0.2 * inch) * 0.52
    d.bullets([
        "Scene: fork, spoon, plate, mug on a dinner table; two SO-101 arms.",
        "T1: place fork (left arm) and spoon (right arm) on plate in parallel.",
        "T2: right arm grasps spoon, hands off to left arm, places spoon on left.",
        "Research Q: Can we combine real multimodal planning with Intel OpenVINO",
        "deployment while keeping a verifiable geometric safety boundary?",
    ], M, y, left_w)
    # task diagram
    x0 = M + left_w + 0.2 * inch
    d.c.setFillColor(MUTED)
    d.c.setFont("Helvetica-Bold", 8)
    d.c.drawString(x0, y - 0.05 * inch, "Task layout (top-down)")
    d.box(x0, y - 0.75 * inch, 1.5 * inch, 0.45 * inch, "Fork", "left arm", SOFT)
    d.box(x0 + 1.65 * inch, y - 0.75 * inch, 1.1 * inch, 0.45 * inch, "Plate", "center", SOFT2)
    d.box(x0 + 2.9 * inch, y - 0.75 * inch, 1.5 * inch, 0.45 * inch, "Spoon", "right arm", SOFT)
    d.box(x0 + 0.9 * inch, y - 1.35 * inch, 1.1 * inch, 0.4 * inch, "Mug", "", SOFT2)
    d.c.setFont("Helvetica", 7.5)
    d.c.setFillColor(ACCENT)
    d.c.drawString(x0, y - 1.55 * inch, "T2: spoon moves R -> L after handoff")
    d.footer()


def slide3_architecture(d: Deck) -> None:
    d.new_page()
    y = d.title_block("System architecture", "Four-layer pipeline from perception to actuation")
    # pipeline diagram
    dy = y - 1.55 * inch
    bh = 0.95 * inch
    specs = [
        ("1. Perception", "Overhead RGB frame\n+ NL instruction"),
        ("2. SmolVLM2-256M", "Proposes JSON task graph\n(pick_place subtasks)"),
        ("3. Validation gate", "Schema, object presence,\narm assignment check"),
        ("4. Geometric exec", "IK, parallel T1, guarded T2,\ngrasp/release timing"),
        ("5. LoRA BC / INT8", "Bounded 50% joint residual\nvia OpenVINO"),
    ]
    bw = (W - 2 * M - 0.5 * inch) / 5
    for i, (head, body) in enumerate(specs):
        x = M + i * (bw + 0.1 * inch)
        fill = SOFT if i % 2 == 0 else colors.HexColor("#e3f2fd")
        d.box(x, dy, bw, bh, head, body, fill=fill)
        if i < 4:
            d.arrow(x + bw + 0.01 * inch, dy + bh / 2, x + bw + 0.09 * inch, dy + bh / 2)
    d.bullets([
        "Invalid VLM plans trigger deterministic rule-based fallback (never sent to motors).",
        "Gripper open/close commands are never overridden by the learned policy.",
        "Joint residuals clipped to 0.15 rad max deviation from geometric target.",
    ], M, dy - 0.35 * inch, W - 2 * M, size=8.8)
    d.footer()


def slide4_simulation(d: Deck, img: Path) -> None:
    d.new_page()
    y = d.title_block("Simulation and robot setup", "MuJoCo environment, actuation, and sensing")
    col = (W - 2 * M - 0.22 * inch) * 0.5
    d.bullets([
        "Robot: 2x SO-101, 6 joints + gripper per arm (12-DoF total).",
        "Control: joint position targets at 50 Hz; weld-based grasp abstraction.",
        "Randomization: object XY pose, plate position, lighting per episode.",
        "Observation (42-dim): qpos, qvel, TCP poses, object poses, holding flags, phase.",
        "Action (12-dim): joint targets for left then right arm.",
        "Cameras: overhead (planner) + demo view (video recording).",
    ], M, y, col, size=8.6)
    if img.exists():
        d.c.drawImage(str(img), M + col + 0.22 * inch, 0.42 * inch,
                      width=W - M - col - 0.22 * inch - M, height=H - 1.35 * inch - 0.42 * inch,
                      preserveAspectRatio=True, anchor="c")
        d.c.setFillColor(MUTED)
        d.c.setFont("Helvetica", 7)
        d.c.drawCentredString(M + col + 0.22 * inch + (W - M - col - 0.22 * inch - M) / 2, 0.32 * inch,
                              "T1 final state (seed 3, overhead view)")
    d.footer()


def slide5_training(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Training data and LoRA fine-tuning", "Expert demos -> augmented dataset -> 3-stage training")
    # training flow
    dy = y - 1.2 * inch
    stages = [
        ("Record", "120 expert\nepisodes"),
        ("Augment", "183k samples\n3x noise"),
        ("Base", "60 epochs\n70% eps"),
        ("LoRA", "40 epochs\n15% eps"),
        ("Joint FT", "30 epochs\n85% eps"),
    ]
    bw = (W - 2 * M - 0.4 * inch) / 5
    for i, (h, b) in enumerate(stages):
        x = M + i * (bw + 0.08 * inch)
        d.box(x, dy, bw, 0.85 * inch, h, b, fill=SOFT if i < 2 else SOFT2)
        if i < 4:
            d.arrow(x + bw + 0.01 * inch, dy + 0.42 * inch, x + bw + 0.07 * inch, dy + 0.42 * inch)
    # metrics table
    ty = dy - 0.55 * inch
    headers = ["Metric", "Value", "Detail"]
    rows = [
        ["Expert success", "119/120", "1 failed episode excluded"],
        ["Holdout MAE", "0.0105", "Joint fine-tune best"],
        ["Model params", "97,132", "17% LoRA adapters"],
        ["HF datasets", "0 rows", "No compatible 42/12 dims"],
    ]
    cw = (W - 2 * M) / 3
    d.c.setFillColor(SOFT)
    d.c.rect(M, ty - 0.22 * inch, W - 2 * M, 0.22 * inch, fill=1, stroke=0)
    d.c.setFont("Helvetica-Bold", 8)
    d.c.setFillColor(INK)
    for i, h in enumerate(headers):
        d.c.drawString(M + i * cw + 0.06 * inch, ty - 0.15 * inch, h)
    ry = ty - 0.32 * inch
    for row in rows:
        d.c.setFont("Helvetica", 8)
        for i, cell in enumerate(row):
            d.c.drawString(M + i * cw + 0.06 * inch, ry, cell)
        ry -= 0.2 * inch
    d.footer()


def slide6_results(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Experimental results", "Stress-tested across randomized seeds with contact audit")
    boxes = [
        ("T1 parallel", "100/100", "fork+spoon placed"),
        ("T2 handoff", "50/50", "R->L transfer"),
        ("Collisions", "0", "70 audited eps"),
        ("Hybrid INT8", "10/10", "@ 50% authority"),
    ]
    bw = (W - 2 * M - 0.36 * inch) / 4
    by = y - 1.05 * inch
    for i, (title, val, sub) in enumerate(boxes):
        x = M + i * (bw + 0.12 * inch)
        d.box(x, by, bw, 0.95 * inch, title, f"{val}\n{sub}", fill=SOFT2)
    d.bullets([
        "T1 placement error: fork mean 4.1 mm, spoon mean 3.9 mm (100-seed eval).",
        "T2 placement error: mean 6.4 mm; max plate displacement 9.2 mm in contact audit.",
        "Pure BC closed-loop: 0/10 without geometric guard (reported separately).",
        "One T1 episode used sequential retry fallback; all 100 completed successfully.",
    ], M, by - 0.45 * inch, W - 2 * M, size=8.6)
    d.footer()


def slide7_openvino(d: Deck) -> None:
    d.new_page()
    y = d.title_block("OpenVINO deployment and quantization", "Export path: PyTorch -> ONNX -> IR -> NNCF INT8")
    # export diagram
    dy = y - 0.95 * inch
    path = ["PyTorch\nlora_bc.pt", "ONNX", "FP32 IR", "NNCF\nINT8 IR", "iGPU\ninfer"]
    bw = (W - 2 * M - 0.45 * inch) / 5
    for i, p in enumerate(path):
        x = M + i * (bw + 0.09 * inch)
        d.box(x, dy, bw, 0.7 * inch, p.split("\n")[0], "\n".join(p.split("\n")[1:]) if "\n" in p else "", fill=SOFT)
        if i < 4:
            d.arrow(x + bw + 0.01 * inch, dy + 0.35 * inch, x + bw + 0.08 * inch, dy + 0.35 * inch)
    # table
    ty = dy - 0.55 * inch
    rows = [
        ["FP32", "419 KB", "3.13 ms", "8/10 @ 50%"],
        ["FP16", "100 KB", "2.93 ms", "9/10 @ 50%"],
        ["Static INT8", "174 KB", "1.19 ms", "10/10 @ 50%"],
        ["Weight INT4", "75 KB", "5.06 ms", "10/10 @ 50%"],
    ]
    headers = ["Precision", "Size", "iGPU p50", "Hybrid success"]
    cw = (W - 2 * M) / 4
    d.c.setFillColor(SOFT2)
    d.c.rect(M, ty - 0.2 * inch, W - 2 * M, 0.2 * inch, fill=1, stroke=0)
    d.c.setFont("Helvetica-Bold", 7.5)
    d.c.setFillColor(INK)
    for i, h in enumerate(headers):
        d.c.drawString(M + i * cw + 0.05 * inch, ty - 0.13 * inch, h)
    ry = ty - 0.3 * inch
    for row in rows:
        d.c.setFont("Helvetica", 7.5)
        for i, cell in enumerate(row):
            d.c.drawString(M + i * cw + 0.05 * inch, ry, cell)
        ry -= 0.18 * inch
    d.c.setFillColor(MUTED)
    d.c.setFont("Helvetica", 7.5)
    d.c.drawString(M, ry - 0.05 * inch, "INT8 iGPU: 2.63x faster than FP32; async throughput 1,189 vs 521 FPS")
    d.footer()


def slide8_multimodal(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Multimodal planning and safety gate", "SmolVLM2-256M as proposal-only planner")
    left = (W - 2 * M - 0.2 * inch) * 0.55
    d.bullets([
        "Input: overhead PNG + text (e.g. 'put fork left, spoon right of plate').",
        "Output: JSON with scene object flags and pick_place subtask list.",
        "Gate checks: valid schema, objects present in sim, correct arm assignment.",
        "Measured episode: valid plan, gate passed, T1 executed without fallback.",
        "Inference: ~46 s per plan on CPU (256M model).",
    ], M, y, left, size=8.4)
    # gate diagram
    x0 = M + left + 0.15 * inch
    d.box(x0, y - 0.1 * inch, 1.55 * inch, 0.55 * inch, "VLM output", "JSON plan", SOFT)
    d.arrow(x0 + 1.57 * inch, y + 0.17 * inch, x0 + 1.72 * inch, y + 0.17 * inch)
    d.box(x0 + 1.74 * inch, y - 0.1 * inch, 1.35 * inch, 0.55 * inch, "Gate", "pass / fail", colors.HexColor("#e3f2fd"))
    d.arrow(x0 + 3.11 * inch, y + 0.17 * inch, x0 + 3.24 * inch, y + 0.17 * inch)
    d.box(x0 + 3.26 * inch, y - 0.1 * inch, 1.2 * inch, 0.55 * inch, "Execute", "or fallback", SOFT2)
    d.c.setFillColor(colors.HexColor("#c62828"))
    d.c.setFont("Helvetica-Bold", 8)
    d.c.drawString(x0, y - 0.85 * inch, "Failed grounding tests (reported honestly):")
    d.c.setFillColor(INK)
    d.c.setFont("Helvetica", 8)
    d.c.drawString(x0, y - 1.05 * inch, "Structured counterfactual: 0%  |  Binary yes/no: 50%")
    d.footer()


def slide9_hybrid(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Hybrid control mode", "How learned inference integrates without breaking safety")
    # blend diagram
    dy = y - 1.15 * inch
    d.box(M, dy, 2.1 * inch, 0.8 * inch, "Geometric target", "IK + scripted timing", SOFT)
    d.c.setFont("Helvetica-Bold", 14)
    d.c.setFillColor(ACCENT)
    d.c.drawString(M + 2.25 * inch, dy + 0.35 * inch, "+")
    d.box(M + 2.55 * inch, dy, 2.1 * inch, 0.8 * inch, "Learned residual", "OpenVINO INT8 LoRA", SOFT2)
    d.c.drawString(M + 4.75 * inch, dy + 0.35 * inch, "->")
    d.box(M + 5.05 * inch, dy, 2.0 * inch, 0.8 * inch, "Blended cmd", "50% authority", colors.HexColor("#e3f2fd"))
    d.bullets([
        "Formula: cmd = geo + blend * (learned - geo), blend=0.5, clip +/- 0.15 rad.",
        "Gripper joints (indices 5, 11) always copy geometric command exactly.",
        "Why hybrid: pure BC is 0/10; geometric alone is 10/10; INT8 hybrid is 10/10.",
        "Finding: static INT8 regularizes destabilizing residuals vs FP32 at high authority.",
    ], M, dy - 0.4 * inch, W - 2 * M, size=8.5)
    d.footer()


def slide10_close(d: Deck) -> None:
    d.new_page()
    y = d.title_block("Limitations, artifacts, and demo", "What we ship and what we do not claim")
    d.bullets([
        "Limitations: no NPU on host; not end-to-end VLA; VLM not visually grounded.",
        "Safety: geometric executive is always the boundary; learned model is residual only.",
        "Video: submission_demo.mp4 (T1 seed 3 + T2 seed 0 + hybrid INT8).",
        "Artifacts: eval_100seed.csv, contact_audit JSON, inference_matrix.json.",
        "Scripts: record_demos, train_lora_bc, export_openvino_int8, eval_hybrid_sweep.",
        "Thank you. Questions welcome.",
    ], M, y, W - 2 * M, size=9)
    d.c.setFillColor(ACCENT)
    d.c.setFont("Helvetica-Bold", 11)
    d.c.drawString(M, 0.55 * inch, "Mise en Place: verified multimodal manipulation on Intel OpenVINO")
    d.footer()


def main() -> None:
    out = C.ARTIFACTS / "mise_en_place_slides.pdf"
    downloads = Path.home() / "Downloads" / "mise_en_place_slides.pdf"
    img = C.ARTIFACTS / "demo_candidates" / "candidate_t1_seed3_final.png"
    if not img.exists():
        img = C.ARTIFACTS / "demo_t1_keyframes" / "overhead_final.png"

    d = Deck(out)
    slide1_title(d)
    slide2_problem(d)
    slide3_architecture(d)
    slide4_simulation(d, img)
    slide5_training(d)
    slide6_results(d)
    slide7_openvino(d)
    slide8_multimodal(d)
    slide9_hybrid(d)
    slide10_close(d)
    d.save()
    downloads.write_bytes(out.read_bytes())
    print(f"wrote {out}")
    print(f"wrote {downloads}")


if __name__ == "__main__":
    main()
