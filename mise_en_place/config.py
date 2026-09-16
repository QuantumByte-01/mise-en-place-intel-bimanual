"""Scene layout + task constants for Mise en Place.

All coordinates are in the **table frame**: the table top surface is ``z = 0``,
``+x`` points away from the robots (into the scene), ``+y`` points toward the
left arm.  Both SO-101 arms are mounted at the near edge facing ``+x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
ASSETS = PKG_ROOT / "assets"
SO101_DIR = ASSETS / "so101"
SO101_XML = SO101_DIR / "so101_new_calib.xml"
GENERATED_DIR = ASSETS / "generated"
SCENE_XML = GENERATED_DIR / "dinner_scene.xml"
ARTIFACTS = REPO_ROOT / "artifacts"

# --------------------------------------------------------------------------
# robot
# --------------------------------------------------------------------------
ARMS = ("left", "right")

#: base placement of each arm, (x, y, z) in table frame
ARM_BASE = {
    "left": (-0.15, 0.16, 0.0),
    "right": (-0.15, -0.16, 0.0),
}

#: SO-101 joints, in actuator order. index 5 is the gripper.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
N_ARM_JOINTS = 5          # DoF excluding the gripper
N_JOINTS = len(JOINTS)    # per-arm actuators
N_ACTION = 2 * N_JOINTS   # full bimanual action vector

#: gripper actuator limits (rad). the SO-101 jaw opens toward +.
GRIPPER_OPEN = 1.25
GRIPPER_CLOSED = -0.05

#: neutral "ready" pose, arm tucked up and back out of the way
HOME_QPOS = (0.0, -0.55, 1.10, 0.75, 0.0, GRIPPER_OPEN)

#: TCP site offset inside the `gripper` body, chosen to sit between the jaws.
TCP_LOCAL_POS = (-0.0079, -0.0002, -0.0855)
TCP_LOCAL_QUAT = (0.707107, 0.0, 0.707107, 0.0)

# --------------------------------------------------------------------------
# table + objects
# --------------------------------------------------------------------------
TABLE_HALF = (0.42, 0.50, 0.02)   # box half-extents; top face sits at z = 0
TABLE_TOP_Z = 0.0

PLATE_POS = (0.055, 0.0)
PLATE_RADIUS = 0.055

#: free-body objects.  ``spawn`` is the nominal xy, ``jitter`` the +/- uniform
#: randomization applied per seed (Phase 2 domain randomization).
@dataclass(frozen=True)
class ObjectSpec:
    name: str
    spawn: tuple[float, float]
    jitter: tuple[float, float]
    z: float
    rgba: tuple[float, float, float, float]
    grasp_z: float          # TCP height used when picking this object
    handle_len: float       # half-length of the graspable handle

OBJECTS: tuple[ObjectSpec, ...] = (
    # Spawn further +x so XY jitter cannot pull utensils into near-base deadzone.
    ObjectSpec("fork", (0.070, 0.210), (0.025, 0.030), 0.010,
               (0.88, 0.90, 0.94, 1.0), grasp_z=0.028, handle_len=0.045),
    ObjectSpec("spoon", (0.070, -0.210), (0.025, 0.030), 0.010,
               (0.62, 0.66, 0.72, 1.0), grasp_z=0.028, handle_len=0.045),
    ObjectSpec("mug", (0.215, 0.0), (0.030, 0.045), 0.040,
               (0.20, 0.42, 0.72, 1.0), grasp_z=0.058, handle_len=0.035),
)
OBJECT_BY_NAME = {o.name: o for o in OBJECTS}

# --------------------------------------------------------------------------
# task targets
# --------------------------------------------------------------------------
#: where each utensil must end up for T1 to count as a success
PLACE_TARGET = {
    "fork": (0.055, 0.092),
    "spoon": (0.055, -0.092),
    "mug": (0.175, 0.105),
}

#: an object counts as placed when its xy is within this radius of the target
PLACE_TOLERANCE = 0.045

#: default arm assignment when the planner has no better information
DEFAULT_ARM = {"fork": "left", "spoon": "right", "mug": "right"}

# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
SIM_TIMESTEP = 0.002
CONTROL_HZ = 50
SIM_STEPS_PER_CONTROL = int(round((1.0 / CONTROL_HZ) / SIM_TIMESTEP))
MAX_EPISODE_STEPS = 1400     # control steps

RENDER_W, RENDER_H = 640, 480
CAMERAS = ("overhead", "demo", "left_wrist", "right_wrist")


@dataclass
class SceneRandomization:
    """Per-seed domain randomization knobs."""

    object_xy: bool = True
    plate_xy: bool = True
    lighting: bool = True
    plate_jitter: tuple[float, float] = (0.025, 0.030)
    light_range: tuple[float, float] = (0.55, 1.15)
    extras: dict = field(default_factory=dict)
