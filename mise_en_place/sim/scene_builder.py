"""Generate the dual SO-101 dinner-table MJCF.

The upstream SO-101 model (``assets/so101/so101_new_calib.xml``, vendored from
TheRobotStudio/SO-ARM100) is a single arm with unprefixed names.  We parse it
with ElementTree and emit **one self-contained scene** that contains the arm
twice, with every body/joint/site/actuator name prefixed ``left_`` / ``right_``.

Doing this programmatically (rather than hand-editing a copy) means an upstream
asset refresh is a re-run of ``python -m mise_en_place.sim.scene_builder``.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .. import config as C

# element attributes that name a *definition* and therefore need prefixing
_NAME_ATTRS = ("name",)
# element attributes that *reference* a prefixed definition
_REF_ATTRS = ("joint", "body1", "body2", "site", "body", "tendon")
# tags whose `name` lives in a namespace we must prefix (per-arm entities)
_PREFIX_TAGS = {"body", "joint", "site", "camera", "position", "motor", "general",
                "geom", "light", "freejoint"}


def _prefix_tree(elem: ET.Element, prefix: str) -> None:
    """Recursively prefix names/references on a cloned subtree, in place."""
    if elem.tag in _PREFIX_TAGS:
        for attr in _NAME_ATTRS:
            if attr in elem.attrib:
                elem.set(attr, prefix + elem.attrib[attr])
    for attr in _REF_ATTRS:
        if attr in elem.attrib:
            elem.set(attr, prefix + elem.attrib[attr])
    for child in list(elem):
        _prefix_tree(child, prefix)


def _fmt(vals) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _load_arm(src_xml: Path) -> tuple[ET.Element, list[ET.Element], ET.Element]:
    """Return (base body, actuator elements, asset element) from the source arm."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    base = root.find("./worldbody/body[@name='base']")
    if base is None:
        raise RuntimeError(f"no <body name='base'> in {src_xml}")
    actuators = list(root.find("actuator"))
    asset = root.find("asset")
    return base, actuators, asset


def _add_tcp_and_wrist_cam(base: ET.Element) -> None:
    """Insert the TCP site + wrist camera into the arm's `gripper` body.

    Done *before* prefixing so both copies pick them up.
    """
    gripper = base.find(".//body[@name='gripper']")
    if gripper is None:
        raise RuntimeError("no 'gripper' body in arm model")
    ET.SubElement(gripper, "site", {
        "name": "tcp",
        "pos": _fmt(C.TCP_LOCAL_POS),
        "quat": _fmt(C.TCP_LOCAL_QUAT),
        "size": "0.004",
        "rgba": "1 0.2 0.2 0.6",
        "group": "4",
    })
    # wrist camera looks down the approach axis toward the pinch point
    ET.SubElement(gripper, "camera", {
        "name": "wrist",
        "pos": "-0.012 -0.055 -0.035",
        "quat": "0.5 0.5 -0.5 -0.5",
        "fovy": "70",
    })


def _table_and_scene_assets(asset: ET.Element) -> None:
    """Add ground/table materials + a skybox to the shared asset block."""
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.32 0.38 0.48", "rgb2": "0.06 0.08 0.12",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "texture", {
        "name": "tabletex", "type": "2d", "builtin": "checker",
        "rgb1": "0.55 0.44 0.33", "rgb2": "0.48 0.38 0.28",
        "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "tablemat", "texture": "tabletex",
        "texrepeat": "6 6", "specular": "0.15", "shininess": "0.1",
    })
    ET.SubElement(asset, "material", {
        "name": "platemat", "rgba": "0.96 0.96 0.97 1", "specular": "0.4",
        "shininess": "0.5",
    })
    for spec in C.OBJECTS:
        ET.SubElement(asset, "material", {
            "name": f"{spec.name}mat", "rgba": _fmt(spec.rgba),
            "specular": "0.6", "shininess": "0.6",
        })


def _build_table(world: ET.Element) -> None:
    hx, hy, hz = C.TABLE_HALF
    ET.SubElement(world, "geom", {
        "name": "table", "type": "box", "pos": f"0 0 {-hz:.4g}",
        "size": _fmt(C.TABLE_HALF), "material": "tablemat",
        "friction": "1.0 0.02 0.001", "contype": "1", "conaffinity": "1",
    })
    # purely visual legs so renders read as a dinner table
    for sx in (-1, 1):
        for sy in (-1, 1):
            ET.SubElement(world, "geom", {
                "name": f"leg_{sx}_{sy}".replace("-", "n"),
                "type": "box", "contype": "0", "conaffinity": "0", "group": "2",
                "pos": f"{sx * (hx - 0.05):.4g} {sy * (hy - 0.05):.4g} -0.20",
                "size": "0.02 0.02 0.18", "rgba": "0.35 0.28 0.22 1",
            })


def _build_objects(world: ET.Element) -> None:
    """Plate + fork + spoon + mug as free bodies resting on the table."""
    # --- plate (free body, heavy enough to stay put; still fully dynamic) ---
    plate = ET.SubElement(world, "body", {
        "name": "plate", "pos": f"{C.PLATE_POS[0]:.4g} {C.PLATE_POS[1]:.4g} 0.006",
    })
    ET.SubElement(plate, "freejoint", {"name": "plate_free"})
    ET.SubElement(plate, "geom", {
        "name": "plate_geom", "type": "cylinder",
        "size": f"{C.PLATE_RADIUS:.4g} 0.006", "material": "platemat",
        "mass": "0.35", "friction": "1.2 0.02 0.001",
    })
    ET.SubElement(plate, "geom", {
        "name": "plate_rim", "type": "cylinder", "pos": "0 0 0.004",
        "size": f"{C.PLATE_RADIUS:.4g} 0.008", "material": "platemat",
        "contype": "0", "conaffinity": "0", "group": "2", "mass": "0.001",
    })

    # --- fork: handle box + flat head ---
    fork = C.OBJECT_BY_NAME["fork"]
    b = ET.SubElement(world, "body", {
        "name": "fork", "pos": f"{fork.spawn[0]:.4g} {fork.spawn[1]:.4g} {fork.z:.4g}",
    })
    ET.SubElement(b, "freejoint", {"name": "fork_free"})
    ET.SubElement(b, "geom", {
        "name": "fork_handle", "type": "box",
        "size": f"{fork.handle_len:.4g} 0.008 0.005", "material": "forkmat",
        "mass": "0.022", "friction": "1.4 0.05 0.002",
    })
    ET.SubElement(b, "geom", {
        "name": "fork_head", "type": "box", "pos": f"{fork.handle_len + 0.022:.4g} 0 0",
        "size": "0.022 0.013 0.003", "material": "forkmat",
        "mass": "0.010", "friction": "1.4 0.05 0.002",
    })

    # --- spoon: handle box + ellipsoid bowl ---
    spoon = C.OBJECT_BY_NAME["spoon"]
    b = ET.SubElement(world, "body", {
        "name": "spoon", "pos": f"{spoon.spawn[0]:.4g} {spoon.spawn[1]:.4g} {spoon.z:.4g}",
    })
    ET.SubElement(b, "freejoint", {"name": "spoon_free"})
    ET.SubElement(b, "geom", {
        "name": "spoon_handle", "type": "box",
        "size": f"{spoon.handle_len:.4g} 0.008 0.005", "material": "spoonmat",
        "mass": "0.022", "friction": "1.4 0.05 0.002",
    })
    ET.SubElement(b, "geom", {
        "name": "spoon_bowl", "type": "ellipsoid",
        "pos": f"{spoon.handle_len + 0.024:.4g} 0 0",
        "size": "0.024 0.016 0.005", "material": "spoonmat",
        "mass": "0.010", "friction": "1.4 0.05 0.002",
    })

    # --- mug: cylinder + arc handle (used by T3 / distractor) ---
    mug = C.OBJECT_BY_NAME["mug"]
    b = ET.SubElement(world, "body", {
        "name": "mug", "pos": f"{mug.spawn[0]:.4g} {mug.spawn[1]:.4g} {mug.z:.4g}",
    })
    ET.SubElement(b, "freejoint", {"name": "mug_free"})
    ET.SubElement(b, "geom", {
        "name": "mug_body", "type": "cylinder", "size": "0.034 0.040",
        "material": "mugmat", "mass": "0.14", "friction": "1.3 0.03 0.001",
    })
    ET.SubElement(b, "geom", {
        "name": "mug_handle", "type": "box", "pos": "0.044 0 0.006",
        "size": "0.008 0.005 0.018", "material": "mugmat", "mass": "0.02",
        "friction": "1.3 0.03 0.001",
    })


def _build_cameras_and_lights(world: ET.Element) -> None:
    ET.SubElement(world, "light", {
        "name": "keylight", "pos": "0.1 0.0 1.3", "dir": "0 0 -1",
        "directional": "true", "diffuse": "0.55 0.55 0.55",
        "specular": "0.12 0.12 0.12",
    })
    ET.SubElement(world, "light", {
        "name": "filllight", "pos": "0.6 -0.6 0.9", "dir": "-0.5 0.5 -1",
        "directional": "false", "diffuse": "0.3 0.3 0.34",
    })
    # top-down camera: the policy's primary observation
    ET.SubElement(world, "camera", {
        "name": "overhead", "pos": "0.06 0 0.78", "quat": "0.7071 0 0 -0.7071",
        "fovy": "64",
    })
    # cinematic camera for the demo video
    ET.SubElement(world, "camera", {
        "name": "demo", "pos": "0.62 -0.52 0.46",
        "xyaxes": "0.64 0.77 0 -0.35 0.29 0.89", "fovy": "52",
    })


def _build_equalities(world_root: ET.Element) -> None:
    """One (arm, object) weld per pair, inactive until the gripper closes.

    This is the attach-on-grasp abstraction: the weld is only *allowed* to
    activate when the TCP is within tolerance of the object and the jaw is
    commanded closed (enforced in :mod:`mise_en_place.sim.env`).  Objects stay
    fully dynamic free bodies, so they can still be dropped or knocked over.
    """
    eq = ET.SubElement(world_root, "equality")
    for arm in C.ARMS:
        for spec in C.OBJECTS:
            ET.SubElement(eq, "weld", {
                "name": f"{arm}_grasp_{spec.name}",
                "body1": f"{arm}_gripper",
                "body2": spec.name,
                "active": "false",
                "solref": "0.005 1",
                "solimp": "0.95 0.99 0.001",
                "torquescale": "1",
            })


def build_scene(src_xml: Path | None = None, out_xml: Path | None = None) -> Path:
    """Generate the dual-arm dinner scene and return the written path."""
    src_xml = Path(src_xml or C.SO101_XML)
    out_xml = Path(out_xml or C.SCENE_XML)
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    arm_base, arm_actuators, arm_asset = _load_arm(src_xml)
    _add_tcp_and_wrist_cam(arm_base)

    root = ET.Element("mujoco", {"model": "mise_en_place_dual_so101"})
    meshdir = Path("..") / src_xml.parent.name / "assets"
    ET.SubElement(root, "compiler", {
        "angle": "radian", "autolimits": "true", "meshdir": meshdir.as_posix(),
    })
    ET.SubElement(root, "option", {
        "timestep": str(C.SIM_TIMESTEP), "integrator": "implicitfast",
        "cone": "elliptic", "impratio": "10",
    })
    ET.SubElement(root, "visual").append(
        ET.Element("global", {"offwidth": str(C.RENDER_W), "offheight": str(C.RENDER_H)})
    )

    # defaults come straight from upstream (shared by both arms)
    src_root = ET.parse(src_xml).getroot()
    for dflt in src_root.findall("default"):
        root.append(copy.deepcopy(dflt))

    asset = copy.deepcopy(arm_asset)
    _table_and_scene_assets(asset)
    root.append(asset)

    world = ET.SubElement(root, "worldbody")
    _build_cameras_and_lights(world)
    _build_table(world)
    _build_objects(world)

    actuator = ET.SubElement(root, "actuator")
    for arm in C.ARMS:
        prefix = f"{arm}_"
        body = copy.deepcopy(arm_base)
        _prefix_tree(body, prefix)
        body.set("pos", _fmt(C.ARM_BASE[arm]))
        world.append(body)
        for act in arm_actuators:
            a = copy.deepcopy(act)
            _prefix_tree(a, prefix)
            actuator.append(a)

    _build_equalities(root)

    ET.indent(root, space="  ")
    out_xml.write_text(
        '<?xml version="1.0" ?>\n'
        "<!-- GENERATED by mise_en_place.sim.scene_builder - do not edit by hand. -->\n"
        "<!-- Arm model vendored from TheRobotStudio/SO-ARM100 Simulation/SO101. -->\n"
        + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return out_xml


def main() -> None:
    path = build_scene()
    print(f"wrote {path}")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} neq={model.neq} ncam={model.ncam}")


if __name__ == "__main__":
    main()
