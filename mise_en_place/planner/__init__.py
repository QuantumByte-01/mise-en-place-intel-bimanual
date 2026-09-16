"""NL → JSON plan. Template planner first; VLM adapter stub for OpenVINO path."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass


@dataclass
class Subtask:
    action: str
    object: str
    arm: str
    target: str


@dataclass
class Plan:
    instruction: str
    subtasks: list[Subtask]

    def to_json(self) -> str:
        return json.dumps(
            {"instruction": self.instruction, "subtasks": [asdict(s) for s in self.subtasks]},
            indent=2,
        )


_DEFAULT_T1 = [
    Subtask("pick_place", "fork", "left", "plate_left"),
    Subtask("pick_place", "spoon", "right", "plate_right"),
]


def plan_from_instruction(instruction: str) -> Plan:
    """Rule-based dinner-table planner (offline, always available)."""
    text = instruction.lower().strip()
    subtasks: list[Subtask] = []

    # Hand-off pattern (T2)
    if "hand" in text and ("off" in text or "pass" in text or "transfer" in text):
        subtasks.append(Subtask("pick", "spoon", "right", "grasp"))
        subtasks.append(Subtask("handoff", "spoon", "both", "mid"))
        subtasks.append(Subtask("place", "spoon", "left", "plate_left"))
        return Plan(instruction, subtasks)

    # Default / explicit fork+spoon
    wants_fork = "fork" in text or "utensil" in text or "set the table" in text or "dinner" in text
    wants_spoon = "spoon" in text or "utensil" in text or "set the table" in text or "dinner" in text
    if not wants_fork and not wants_spoon:
        wants_fork = wants_spoon = True

    if wants_fork:
        arm = "right" if re.search(r"fork.*\bright\b", text) else "left"
        subtasks.append(Subtask("pick_place", "fork", arm, "plate_side"))
    if wants_spoon:
        arm = "left" if re.search(r"spoon.*\bleft\b", text) else "right"
        subtasks.append(Subtask("pick_place", "spoon", arm, "plate_side"))

    if not subtasks:
        subtasks = list(_DEFAULT_T1)
    return Plan(instruction, subtasks)


class QwenPlannerStub:
    """Placeholder for Qwen2.5-VL OpenVINO GenAI planner.

    Falls back to :func:`plan_from_instruction` until the OV model is wired.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_id = model_id
        self.ready = False

    def plan(self, instruction: str, image_path: str | None = None) -> Plan:
        # Future: run OV INT4 VLM → JSON. For now, template path keeps E2E green.
        return plan_from_instruction(instruction)
