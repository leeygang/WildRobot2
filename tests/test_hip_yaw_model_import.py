"""Regression contract for the six-DOF CAD/model revision."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.rsl_rl_training import RSLPPOTrainer, _jax_policy_factory
from wr2.locomotion.walking_env import WR2WalkingEnv
from wr2.reference.walk_zmp import WR2ZMPReference
from wr2.reference.zmp_validation import apply_reference_frame, create_reference_context
from wr2.tools.view_legacy_zmp_reference import _draw_reference_overlay
from wr2.tools.verify_walking_symmetry import joint_reflection, verify_model_reference


class HipYawModelImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config("wr2/locomotion/configs/ppo_walking_xy.yaml")
        cls.env = WR2WalkingEnv(cls.config.environment, add_observation_noise=False)
        cls.robot = cls.env.robot
        cls.model = mujoco.MjModel.from_xml_path(
            str(cls.robot.directory / "scene_mjx.xml")
        )

    def test_model_counts_and_inactive_head_contract(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (29, 28, 21))
        self.assertEqual(self.env.action_size, 12)
        self.assertEqual(
            self.env.observation_size, {"state": (1005,), "privileged_state": (1740,)}
        )
        active = np.asarray(self.env._active_indices)
        self.assertEqual(
            {self.robot.actuator_names[i] for i in active},
            {
                f"{side}_{joint}"
                for side in ("left", "right")
                for joint in (
                    "hip_pitch",
                    "hip_roll",
                    "hip_yaw",
                    "knee_pitch",
                    "ankle_pitch",
                    "ankle_roll",
                )
            },
        )
        target = np.asarray(self.env._policy_target(np.ones(12)))
        for name in ("neck_yaw", "head_pitch", "waist_yaw_drive"):
            i = self.robot.actuator_names.index(name)
            self.assertEqual(target[i], self.robot.home_position_rad[i])

    def test_exported_ranges_and_head_inertia_are_not_fabricated(self):
        raw = ET.parse(self.robot.directory / "onshape_export/wr2.xml").getroot()
        canonical = ET.parse(self.robot.directory / "wr2.xml").getroot()
        for joint in raw.findall("worldbody//joint"):
            actual = canonical.find(f".//joint[@name='{joint.get('name')}']")
            np.testing.assert_allclose(
                np.fromstring(actual.get("range"), sep=" "),
                np.fromstring(joint.get("range"), sep=" "),
                atol=0,
                rtol=0,
            )
        for name in ("neck_yaw", "head_pitch"):

            def inertia(root):
                return next(
                    b.find("inertial").attrib
                    for b in root.findall("worldbody//body")
                    if b.find(f"joint[@name='{name}']") is not None
                )

            self.assertEqual(inertia(raw), inertia(canonical))
            self.assertGreater(float(inertia(canonical)["mass"]), 0.001)

    def test_xy_reference_mirroring_uses_opposite_lateral_command(self):
        source, signs = joint_reflection(self.model, self.robot.actuator_names)
        report = verify_model_reference(self.env, self.model, source, signs)
        # Small CAD asymmetry can require different IK joint coordinates;
        # the generated foot displacement, not those coordinates, must mirror.
        self.assertLess(report["reference_foot_displacement_error_mm"]["max_abs"], 0.1)
        self.assertLess(
            report["reference_foot_error_mm"]["max_abs"],
            report["home_foot_error_mm"]["max_abs"] + 0.1,
        )

    def test_home_ground_clearance_is_model_derived(self):
        d = mujoco.MjData(self.model)
        for key in ("home", "walk_home"):
            mujoco.mj_resetDataKeyframe(self.model, d, self.model.key(key).id)
            mujoco.mj_forward(self.model, d)
            bottoms = []
            for side in ("left", "right"):
                i = self.model.geom(f"{side}_foot_collision").id
                bottoms.append(
                    d.geom_xpos[i, 2]
                    - np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ self.model.geom_size[i]
                )
            self.assertAlmostEqual(min(bottoms), 0.001, places=8)

    def test_six_dof_ik_controls_foot_yaw_and_keeps_position(self):
        m, d = self.model, mujoco.MjData(self.model)
        home = np.asarray(m.key_qpos[m.key("walk_home").id])
        d.qpos[:] = home
        mujoco.mj_forward(m, d)
        for side in ("left", "right"):
            site = m.site(f"{side}_foot_center").id
            target_position = d.site_xpos[site].copy()
            target_orientation = Rotation.from_euler(
                "z", 8, degrees=True
            ).as_matrix() @ d.site_xmat[site].reshape(3, 3)
            names = [
                f"{side}_{s}"
                for s in (
                    "hip_pitch",
                    "hip_roll",
                    "hip_yaw",
                    "knee_pitch",
                    "ankle_pitch",
                    "ankle_roll",
                )
            ]
            ids = np.asarray([m.joint(n).id for n in names])
            q, position_error, orientation_error = WR2ZMPReference._solve_leg_ik(
                m,
                d,
                home_qpos=home,
                site_id=site,
                qpos_indices=m.jnt_qposadr[ids],
                dof_indices=m.jnt_dofadr[ids],
                lower=m.jnt_range[ids, 0],
                upper=m.jnt_range[ids, 1],
                target_position=target_position,
                target_orientation=target_orientation,
                damping=0.0002,
                max_iterations=120,
            )
            self.assertLess(position_error, 0.0001)
            self.assertLess(orientation_error, 0.0001)
            self.assertGreater(abs(q[2]), np.deg2rad(4))
            d.qpos[:] = home
            mujoco.mj_forward(m, d)

    def test_lipm_height_matches_actual_crouched_com_not_root_height(self):
        r, m = self.env._zmp_reference, self.model
        zero = np.argmin(np.linalg.norm(r._command_grid_numpy, axis=1))
        d = mujoco.MjData(m)
        d.qpos[:] = m.key_qpos[m.key("walk_home").id]
        d.qpos[2] = r.diagnostics.root_height_m
        d.qpos[np.asarray(self.env._joint_qpos_indices)] = np.asarray(
            r._joint_lookup[zero, 0]
        )
        mujoco.mj_forward(m, d)
        self.assertAlmostEqual(
            d.subtree_com[m.body("torso").id, 2], r.diagnostics.com_height_m, places=6
        )
        self.assertGreater(
            r.diagnostics.root_height_m - r.diagnostics.com_height_m, 0.05
        )

    def test_checkpoint_restore_rejects_changed_model_contract_even_with_same_shapes(
        self,
    ):
        c = replace(
            self.config,
            ppo=replace(
                self.config.ppo, num_envs=2, unroll_length=2, num_minibatches=1
            ),
            network=replace(
                self.config.network,
                policy_hidden_layer_sizes=(8, 4),
                value_hidden_layer_sizes=(8, 4),
            ),
        )
        e = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=2,
            num_actions=12,
            robot_contract={
                "model_sha256": "new-model",
                "actuator_names": list(self.robot.actuator_names),
            },
            get_observations=lambda: (
                torch.zeros(2, 1005),
                {"observations": {"critic": torch.zeros(2, 1740)}},
            ),
        )
        trainer = RSLPPOTrainer(
            e,
            c,
            output=Path("unused"),
            restore_checkpoint=None,
            progress_fn=lambda *a: None,
            policy_params_fn=lambda *a: None,
            evaluator=None,
        )
        state = trainer.checkpoint_state()
        state["robot_contract"] = {"model_sha256": "old-model"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "fresh training"):
                trainer.load(path)

    def test_viewer_draws_planned_com_not_pelvis_height(self):
        context = create_reference_context((0.0, 0.1), training_config=self.config)
        viewer = SimpleNamespace(
            lock=nullcontext,
            user_scn=SimpleNamespace(ngeom=0),
            cam=SimpleNamespace(lookat=np.zeros(3)),
        )
        for command in (0.0, 0.1):
            sample, root_x = apply_reference_frame(
                context, phase_rad=1.0, command_forward_m_s=command
            )
            with (
                patch("wr2.tools.view_legacy_zmp_reference._add_sphere") as sphere,
                patch("wr2.tools.view_legacy_zmp_reference._add_line"),
            ):
                _draw_reference_overlay(viewer, context, sample, root_x)
            planned = sphere.call_args_list[1].args[1]
            expected = (
                context.reference.diagnostics.com_height_m
                if command
                else context.data.subtree_com[0, 2]
            )
            self.assertAlmostEqual(planned[2], expected)
            self.assertGreater(sample.root_height_m - planned[2], 0.05)

    def test_legacy_policy_evaluation_fails_with_clear_shape_error(self):
        policy = _jax_policy_factory("elu")(((np.zeros((10, 855)), np.zeros(10)),))
        with self.assertRaisesRegex(ValueError, "fresh hip-yaw"):
            policy({"state": np.zeros((1, 1005))}, None)


if __name__ == "__main__":
    unittest.main()
