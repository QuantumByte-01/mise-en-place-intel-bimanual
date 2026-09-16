from mise_en_place.planner.smolvlm import validate_t1_proposal


VALID_PLAN = {
    "subtasks": [
        {
            "action": "pick_place",
            "object": "fork",
            "arm": "left",
            "target": "plate_left",
        },
        {
            "action": "pick_place",
            "object": "spoon",
            "arm": "right",
            "target": "plate_right",
        },
    ]
}


def test_accepts_complete_safe_t1_plan() -> None:
    assert validate_t1_proposal(
        VALID_PLAN,
        present_objects={"plate", "fork", "spoon", "mug"},
    ) == []


def test_rejects_hallucinated_absent_object() -> None:
    errors = validate_t1_proposal(
        VALID_PLAN,
        present_objects={"plate", "fork", "mug"},
    )
    assert any("absent object 'spoon'" in error for error in errors)


def test_rejects_unsafe_arm_assignment() -> None:
    plan = {
        "subtasks": [
            {**VALID_PLAN["subtasks"][0], "arm": "right"},
            VALID_PLAN["subtasks"][1],
        ]
    }
    errors = validate_t1_proposal(
        plan,
        present_objects={"plate", "fork", "spoon", "mug"},
    )
    assert any("unsafe arm assignment" in error for error in errors)


def test_rejects_duplicate_and_missing_subtask() -> None:
    plan = {"subtasks": [VALID_PLAN["subtasks"][0], VALID_PLAN["subtasks"][0]]}
    errors = validate_t1_proposal(
        plan,
        present_objects={"plate", "fork", "spoon", "mug"},
    )
    assert any("duplicate object" in error for error in errors)
    assert any("missing required objects" in error for error in errors)


def test_rejects_non_json_proposal() -> None:
    assert validate_t1_proposal(
        None,
        present_objects={"plate", "fork", "spoon", "mug"},
    ) == ["proposal is not a JSON object"]
