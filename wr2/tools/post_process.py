#!/usr/bin/env python3
"""Build and validate the canonical WR2 MuJoCo model.

The Onshape export under ``onshape_export/`` is treated as immutable input.
This tool stages its mesh assets and scene beside the canonical ``wr2.xml``,
then applies the small amount of semantic information that is not represented
in CAD:

* servo defaults and IMU sensors;
* stable body names and actuator order;
* simple, named foot collision boxes and foot-center sites; and
* a zero-joint-angle standing pose.

Run from the repository root with MuJoCo available, for example:

    uv run --no-project --with 'mujoco>=3.3,<4' \
        python -m wr2.tools.post_process
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


ROBOT_DIR = Path(__file__).resolve().parents[1] / "descriptions" / "wr2"
RAW_DIR_NAME = "onshape_export"
MODEL_NAME = "wr2.xml"
SCENE_NAME = "scene.xml"
ROOT_BODY_NAME = "torso"
ROOT_JOINT_NAME = "torso_freejoint"
HOME_HEIGHT_M = 0.305
WALK_HOME_HEIGHT_M = 0.304
WALK_HOME_JOINTS = {
    "right_hip_pitch": 0.12,
    "right_knee_pitch": 0.24,
    "right_ankle_pitch": 0.12,
    "left_hip_pitch": -0.12,
    "left_knee_pitch": -0.24,
    "left_ankle_pitch": -0.12,
}

# Derived from the Onshape foot mesh's axis-aligned bounds in the foot body
# frame. MuJoCo box sizes are half-extents. In WR2's foot frame, +Y points
# approximately downward and Z runs heel-to-toe.
FOOT_BOX_POS = (0.0, 0.038500, -0.029060)
FOOT_BOX_SIZE = (0.029400, 0.007500, 0.060000)
FOOT_SITE_POS = (0.0, 0.046000, -0.029060)

EXPECTED_ACTUATORS = 17
EXPECTED_HINGES = 18
EXPECTED_SENSORS = {
    "torso_imu_gyro",
    "torso_imu_accel",
    "torso_imu_mag",
    "torso_imu_quat",
}

# A body is the link immediately downstream of its direct joint.  The joint
# name is the reliable side oracle; raw Onshape suffixes such as ``_2`` are not.
BODY_NAME_BY_JOINT = {
    "right_hip_pitch": "right_hip_pitch_link",
    "right_hip_roll": "right_upper_leg",
    "right_knee_pitch": "right_lower_leg",
    "right_ankle_pitch": "right_ankle_pitch_link",
    "right_ankle_roll": "right_foot",
    "left_hip_pitch": "left_hip_pitch_link",
    "left_hip_roll": "left_upper_leg",
    "left_knee_pitch": "left_lower_leg",
    "left_ankle_pitch": "left_ankle_pitch_link",
    "left_ankle_roll": "left_foot",
    "right_shoulder_pitch": "right_shoulder_pitch_link",
    "right_shoulder_roll": "right_upper_arm",
    "right_elbow_pitch": "right_forearm",
    "left_shoulder_pitch": "left_shoulder_pitch_link",
    "left_shoulder_roll": "left_upper_arm",
    "left_elbow_pitch": "left_forearm",
}


def _parse_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _read_actuator_order(path: Path) -> list[str]:
    order = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(order) != len(set(order)):
        raise ValueError(f"Duplicate entries in {path}")
    if len(order) != EXPECTED_ACTUATORS:
        raise ValueError(
            f"Expected {EXPECTED_ACTUATORS} actuators in {path}, found {len(order)}"
        )
    return order


def _inject_auxiliary_xml(root: ET.Element, robot_dir: Path) -> None:
    main_default = root.find("default")
    if main_default is None:
        main_default = ET.Element("default")
        root.insert(0, main_default)

    defaults = _parse_xml(robot_dir / "joints_properties.xml").getroot()
    existing_classes = {
        elem.get("class") for elem in root.findall(".//default") if elem.get("class")
    }
    for child in list(defaults):
        class_name = child.get("class")
        if class_name not in existing_classes:
            main_default.append(child)
            if class_name:
                existing_classes.add(class_name)

    sensors = _parse_xml(robot_dir / "sensors.xml").getroot()
    target = root.find("sensor")
    if target is None:
        root.append(sensors)
        return

    existing_names = {child.get("name") for child in target if child.get("name")}
    for child in list(sensors):
        if child.get("name") not in existing_names:
            target.append(child)


def _set_simulation_options(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        worldbody_index = next(
            (index for index, child in enumerate(root) if child.tag == "worldbody"),
            0,
        )
        root.insert(worldbody_index, option)
    option.set("timestep", "0.002")

    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")


def _normalize_body_names(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("The exported MJCF has no <worldbody>")

    bodies_by_joint: dict[str, ET.Element] = {}
    for body in worldbody.findall(".//body"):
        direct_joint = body.find("joint")
        if direct_joint is not None and direct_joint.get("name"):
            bodies_by_joint[direct_joint.get("name", "")] = body

    missing = sorted(set(BODY_NAME_BY_JOINT) - set(bodies_by_joint))
    if missing:
        raise ValueError(f"Cannot normalize bodies; joints are missing: {missing}")

    untouched_names = {
        body.get("name")
        for body in worldbody.findall(".//body")
        if body.get("name")
        and body not in {bodies_by_joint[name] for name in BODY_NAME_BY_JOINT}
    }
    duplicate_targets = sorted(set(BODY_NAME_BY_JOINT.values()) & untouched_names)
    if duplicate_targets:
        raise ValueError(
            f"Body rename would create duplicate names: {duplicate_targets}"
        )

    for joint_name, body_name in BODY_NAME_BY_JOINT.items():
        bodies_by_joint[joint_name].set("name", body_name)


def _replace_foot_collisions(root: ET.Element) -> None:
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_foot']")
        if body is None:
            raise ValueError(f"Missing normalized {side}_foot body")

        for geom in list(body.findall("geom")):
            if geom.get("class") == "collision":
                body.remove(geom)
        for site in list(body.findall("site")):
            if site.get("name") == f"{side}_foot_center":
                body.remove(site)

        body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{side}_foot_collision",
                    "type": "box",
                    "class": "collision",
                    "pos": _numbers(FOOT_BOX_POS),
                    "size": _numbers(FOOT_BOX_SIZE),
                    "friction": "1 0.005 0.0001",
                    "material": "foot_material",
                },
            )
        )
        body.append(
            ET.Element(
                "site",
                {
                    "name": f"{side}_foot_center",
                    "type": "sphere",
                    "pos": _numbers(FOOT_SITE_POS),
                    "size": "0.003",
                    "group": "3",
                    "rgba": "0.9 0.1 0.1 0.8",
                },
            )
        )


def _reorder_actuators(root: ET.Element, order: list[str]) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("The exported MJCF has no <actuator> block")

    children = list(actuator)
    by_name = {child.get("name"): child for child in children}
    if None in by_name or len(by_name) != len(children):
        raise ValueError("Every actuator must have a unique name")

    missing = [name for name in order if name not in by_name]
    extra = [name for name in by_name if name not in order]
    if missing or extra:
        raise ValueError(f"Actuator-order mismatch; missing={missing}, extra={extra}")

    actuator[:] = [by_name[name] for name in order]


def _set_home_poses(root: ET.Element, actuator_order: list[str]) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("The exported MJCF has no <worldbody>")
    torso = worldbody.find(f"body[@name='{ROOT_BODY_NAME}']")
    if torso is None:
        raise ValueError(f"Missing root body {ROOT_BODY_NAME!r}")
    if torso.find(f"freejoint[@name='{ROOT_JOINT_NAME}']") is None:
        raise ValueError(f"Missing root freejoint {ROOT_JOINT_NAME!r}")
    torso.set("pos", f"0 0 {HOME_HEIGHT_M:.6f}")

    joints = worldbody.findall(".//joint")
    unsupported = [
        joint.get("name")
        for joint in joints
        if joint.get("type", "hinge") not in {"hinge", "slide"}
    ]
    if unsupported:
        raise ValueError(f"Unsupported non-scalar joints in home pose: {unsupported}")

    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.get("name") in {"home", "walk_home"}:
            keyframe.remove(key)

    joint_names = [joint.get("name", "") for joint in joints]
    for key_name, root_height, joint_positions in (
        ("home", HOME_HEIGHT_M, {}),
        ("walk_home", WALK_HOME_HEIGHT_M, WALK_HOME_JOINTS),
    ):
        qpos = [0.0, 0.0, root_height, 1.0, 0.0, 0.0, 0.0]
        qpos.extend(joint_positions.get(name, 0.0) for name in joint_names)
        ctrl = [joint_positions.get(name, 0.0) for name in actuator_order]
        ET.SubElement(
            keyframe,
            "key",
            {
                "name": key_name,
                "qpos": _numbers(qpos),
                "ctrl": _numbers(ctrl),
            },
        )


def _write_mjx_variant(
    canonical_root: ET.Element,
    raw_scene: Path,
    robot_dir: Path,
) -> tuple[Path, Path]:
    """Write a sensor-free MJX model and matching scene.

    The first training model keeps MuJoCo position actuators. This lets us
    validate the environment and policy contract before replacing the actuator
    with an explicit torque-speed controller.
    """
    mjx_root = copy.deepcopy(canonical_root)
    sensors = mjx_root.find("sensor")
    if sensors is not None:
        mjx_root.remove(sensors)
    option = mjx_root.find("option")
    if option is None:
        raise ValueError("Canonical model is missing simulation options")
    option.set("iterations", "1")
    option.set("ls_iterations", "4")

    model_path = robot_dir / "wr2_mjx.xml"
    mjx_tree = ET.ElementTree(mjx_root)
    ET.indent(mjx_tree, space="  ", level=0)
    mjx_tree.write(model_path, encoding="utf-8", xml_declaration=True)

    scene_tree = _parse_xml(raw_scene)
    include = scene_tree.getroot().find("include")
    if include is None:
        raise ValueError(f"Scene has no model include: {raw_scene}")
    include.set("file", model_path.name)
    scene_path = robot_dir / "scene_mjx.xml"
    ET.indent(scene_tree, space="  ", level=0)
    scene_tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    return model_path, scene_path


def _validate_static_contract(root: ET.Element, order: list[str]) -> None:
    hinges = [
        joint
        for joint in root.findall("worldbody//joint")
        if joint.get("type", "hinge") == "hinge"
    ]
    if len(hinges) != EXPECTED_HINGES:
        raise ValueError(
            f"Expected {EXPECTED_HINGES} hinge joints, found {len(hinges)}"
        )

    actuator = root.find("actuator")
    actual_order = [] if actuator is None else [child.get("name") for child in actuator]
    if actual_order != order:
        raise ValueError(f"Canonical actuator order was not applied: {actual_order}")

    equality = root.find(
        "equality/joint[@joint1='waist_yaw_driven'][@joint2='waist_yaw_drive']"
    )
    if equality is None or equality.get("polycoef") != "0 1.0 0 0 0":
        raise ValueError("Missing expected 1:1 waist yaw equality constraint")

    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_foot']")
        geom = root.find(f".//geom[@name='{side}_foot_collision']")
        site = root.find(f".//site[@name='{side}_foot_center']")
        if body is None or geom is None or site is None:
            raise ValueError(f"Incomplete {side} foot model")
        if geom.get("type") != "box":
            raise ValueError(f"{side}_foot_collision is not a box")


def build_model(robot_dir: Path) -> tuple[Path, Path, Path, Path]:
    raw_dir = robot_dir / RAW_DIR_NAME
    raw_model = raw_dir / MODEL_NAME
    raw_scene = raw_dir / SCENE_NAME
    raw_assets = raw_dir / "assets"
    order_path = robot_dir / "actuator_order.txt"

    for required in (
        raw_model,
        raw_scene,
        raw_assets,
        order_path,
        robot_dir / "joints_properties.xml",
        robot_dir / "sensors.xml",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    canonical_assets = robot_dir / "assets"
    canonical_assets.mkdir(parents=True, exist_ok=True)
    stl_files = sorted(raw_assets.glob("*.stl"))
    if not stl_files:
        raise ValueError(f"No STL assets found in {raw_assets}")
    for source in stl_files:
        shutil.copy2(source, canonical_assets / source.name)

    tree = _parse_xml(raw_model)
    root = tree.getroot()
    order = _read_actuator_order(order_path)
    _inject_auxiliary_xml(root, robot_dir)
    _set_simulation_options(root)
    _normalize_body_names(root)
    _replace_foot_collisions(root)
    _reorder_actuators(root, order)
    _set_home_poses(root, order)
    _validate_static_contract(root, order)

    canonical_model = robot_dir / MODEL_NAME
    ET.indent(tree, space="  ", level=0)
    tree.write(canonical_model, encoding="utf-8", xml_declaration=True)

    canonical_scene = robot_dir / SCENE_NAME
    shutil.copy2(raw_scene, canonical_scene)
    mjx_model, mjx_scene = _write_mjx_variant(root, raw_scene, robot_dir)
    return canonical_model, canonical_scene, mjx_model, mjx_scene


def validate_model(
    model_path: Path,
    scene_path: Path,
    order_path: Path,
    *,
    print_mass_report: bool = False,
) -> None:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for validation. Run with: "
            "uv run --no-project --with 'mujoco>=3.3,<4' "
            "python -m wr2.tools.post_process"
        ) from exc

    robot_model = mujoco.MjModel.from_xml_path(str(model_path))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    expected_order = _read_actuator_order(order_path)

    # Ensure the hand-authored primitive still encloses the latest CAD foot.
    # The raw foot mesh remains as a visual geom on each foot body, so compare
    # its body-frame AABB with the generated collision box after every export.
    robot_data = mujoco.MjData(robot_model)
    mujoco.mj_forward(robot_model, robot_data)
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(
            robot_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot"
        )
        box_id = mujoco.mj_name2id(
            robot_model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
        )
        mesh_ids = [
            geom_id
            for geom_id in range(robot_model.ngeom)
            if int(robot_model.geom_bodyid[geom_id]) == body_id
            and int(robot_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(robot_model.geom_dataid[geom_id]) >= 0
        ]
        if body_id < 0 or box_id < 0 or len(mesh_ids) != 2:
            raise ValueError(f"Cannot identify {side} foot box and visual meshes")

        # Select the larger visual mesh (the foot rather than hip_pitch_link).
        mesh_geom_id = max(
            mesh_ids,
            key=lambda geom_id: int(
                robot_model.mesh_vertnum[int(robot_model.geom_dataid[geom_id])]
            ),
        )
        mesh_id = int(robot_model.geom_dataid[mesh_geom_id])
        vertex_start = int(robot_model.mesh_vertadr[mesh_id])
        vertex_count = int(robot_model.mesh_vertnum[mesh_id])
        vertices = robot_model.mesh_vert[vertex_start : vertex_start + vertex_count]
        geom_rotation = robot_data.geom_xmat[mesh_geom_id].reshape(3, 3)
        world_vertices = vertices @ geom_rotation.T + robot_data.geom_xpos[mesh_geom_id]
        body_rotation = robot_data.xmat[body_id].reshape(3, 3)
        body_vertices = (world_vertices - robot_data.xpos[body_id]) @ body_rotation
        lower = body_vertices.min(axis=0)
        upper = body_vertices.max(axis=0)
        mesh_center = 0.5 * (lower + upper)
        mesh_half_size = 0.5 * (upper - lower)
        if not np.allclose(robot_model.geom_pos[box_id], mesh_center, atol=1e-4):
            raise ValueError(
                f"{side} foot collision center is stale: "
                f"box={robot_model.geom_pos[box_id]}, mesh={mesh_center}"
            )
        if not np.allclose(robot_model.geom_size[box_id], mesh_half_size, atol=1e-4):
            raise ValueError(
                f"{side} foot collision size is stale: "
                f"box={robot_model.geom_size[box_id]}, mesh={mesh_half_size}"
            )

    actual_order = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    if actual_order != expected_order:
        raise ValueError(f"Compiled actuator order mismatch: {actual_order}")
    if model.nu != EXPECTED_ACTUATORS:
        raise ValueError(f"Expected nu={EXPECTED_ACTUATORS}, got {model.nu}")
    if model.njnt != EXPECTED_HINGES + 1:
        raise ValueError(
            f"Expected 1 free + {EXPECTED_HINGES} hinge joints, got {model.njnt}"
        )
    if model.neq != 1:
        raise ValueError(f"Expected one waist equality constraint, got {model.neq}")

    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    }
    if sensor_names != EXPECTED_SENSORS:
        raise ValueError(f"Sensor mismatch: {sorted(sensor_names)}")

    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    for side in ("left", "right"):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
        )
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_center"
        )
        if geom_id < 0 or site_id < 0 or int(model.geom_type[geom_id]) != box_type:
            raise ValueError(f"Compiled {side} foot contract is incomplete")

    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT_NAME)
    if root_joint_id < 0:
        raise ValueError("Compiled model is missing the root joint")
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    scalar_qpos_addresses = [
        int(model.jnt_qposadr[joint_id])
        for joint_id in range(model.njnt)
        if joint_id != root_joint_id
    ]

    total_mass = float(robot_model.body_mass[1:].sum())
    print(
        f"Compiled robot: nq={robot_model.nq}, nv={robot_model.nv}, nu={robot_model.nu}"
    )
    print(
        f"Joints/sensors/equalities: {robot_model.njnt}/{robot_model.nsensor}/{robot_model.neq}"
    )
    print(f"Total modeled mass: {total_mass:.6f} kg")

    settle_seconds = 2.0
    steps = round(settle_seconds / model.opt.timestep)
    for key_name in ("home", "walk_home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id < 0:
            raise ValueError(f"Compiled model is missing keyframe {key_name!r}")

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        initial_qpos = data.qpos.copy()
        initial_site_z = {}
        for side in ("left", "right"):
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_center"
            )
            initial_site_z[side] = float(data.site_xpos[site_id, 2])

        for _ in range(steps):
            mujoco.mj_step(model, data)

        root_xyz = data.qpos[root_qpos_adr : root_qpos_adr + 3].copy()
        root_quat = data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7].copy()
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, root_quat)
        tilt_deg = math.degrees(math.acos(float(np.clip(rotation[8], -1.0, 1.0))))
        max_joint_drift = max(
            abs(float(data.qpos[address] - initial_qpos[address]))
            for address in scalar_qpos_addresses
        )
        max_xy_drift = float(np.max(np.abs(root_xyz[:2] - initial_qpos[:2])))
        min_contact_distance = min(
            (float(data.contact[index].dist) for index in range(data.ncon)),
            default=0.0,
        )

        failures = []
        if not 0.295 <= float(root_xyz[2]) <= 0.310:
            failures.append(f"settled root Z is {root_xyz[2]:.6f} m")
        if max_xy_drift > 0.005:
            failures.append(f"horizontal drift is {max_xy_drift:.6f} m")
        if tilt_deg > 2.0:
            failures.append(f"torso tilt is {tilt_deg:.3f} deg")
        if max_joint_drift > 0.02:
            failures.append(f"joint drift is {max_joint_drift:.6f} rad")
        if data.ncon < 2:
            failures.append(f"only {data.ncon} settled contacts")
        if min_contact_distance < -0.005:
            failures.append(f"contact penetration is {min_contact_distance:.6f} m")
        if failures:
            raise ValueError(
                f"Standing validation failed for {key_name}: " + "; ".join(failures)
            )

        print(
            f"{key_name} initial foot-center Z: "
            + ", ".join(
                f"{side}={height:.6f} m" for side, height in initial_site_z.items()
            )
        )
        print(
            f"{key_name} standing check ({settle_seconds:.1f} s): "
            f"root_z={root_xyz[2]:.6f} m, xy_drift={max_xy_drift:.6f} m, "
            f"tilt={tilt_deg:.3f} deg, joint_drift={max_joint_drift:.6f} rad, "
            f"contacts={data.ncon}, "
            f"max_penetration={max(0.0, -min_contact_distance):.6f} m"
        )

    if print_mass_report:
        print("Body mass report:")
        for body_id in range(1, robot_model.nbody):
            body_name = mujoco.mj_id2name(
                robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            print(f"  {body_name}: {robot_model.body_mass[body_id]:.6f} kg")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build WR2's canonical MJCF from the raw Onshape export."
    )
    parser.add_argument(
        "--robot-dir",
        type=Path,
        default=ROBOT_DIR,
        help=f"WR2 description directory (default: {ROBOT_DIR})",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Build without compiling or running the standing test.",
    )
    parser.add_argument(
        "--mass-report",
        action="store_true",
        help="Print the per-body mass breakdown after validation.",
    )
    args = parser.parse_args()

    robot_dir = args.robot_dir.resolve()
    model_path, scene_path, mjx_model_path, mjx_scene_path = build_model(robot_dir)
    print(f"Staged canonical model: {model_path}")
    print(f"Staged canonical scene: {scene_path}")
    print(f"Staged MJX model: {mjx_model_path}")
    print(f"Staged MJX scene: {mjx_scene_path}")
    print(f"Staged STL assets: {robot_dir / 'assets'}")

    if not args.skip_validation:
        validate_model(
            model_path,
            scene_path,
            robot_dir / "actuator_order.txt",
            print_mass_report=args.mass_report,
        )
        mjx_model = __import__("mujoco").MjModel.from_xml_path(str(mjx_scene_path))
        if mjx_model.nsensor != 0:
            raise ValueError("MJX scene must not contain MJCF sensors")
        print(
            f"MJX scene compiled with MuJoCo: nq={mjx_model.nq}, "
            f"nv={mjx_model.nv}, nu={mjx_model.nu}"
        )
        print("WR2 model validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
