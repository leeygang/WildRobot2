"""Compare one-leg and bilateral changes with foot-pose-preserving leg IK.

Uses saved WR2 policies and the native environment. No training weights or
physical parameters change. ToddlerBot Paper provides the coordinated-reference
design: https://arxiv.org/html/2502.00893v2, Tables VI–VII.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.optimize import least_squares


def centered_reference_wave(reference, strength):
    """Zero-mean applied waveform over this diagnostic's actual time window."""
    reference = np.asarray(reference, dtype=float)
    strength = np.asarray(strength, dtype=float)[:, None]
    center = np.sum(reference * strength, axis=0) / strength.sum()
    shaped = strength * (reference - center)
    return shaped / np.max(np.abs(shaped), axis=0)


def reference_hip_targets(original, reference, strength, lower, upper, peak, center):
    """Blend toward a matched hip mean with a bounded, centered reference wave."""
    weight = np.asarray(strength)[:, None]
    center = np.asarray(center)
    shifted = original + weight * (center - original)
    wave = centered_reference_wave(reference, strength)
    room = np.full_like(wave, np.inf)
    positive, negative = wave > 1e-8, wave < -1e-8
    room[positive] = ((upper - shifted) / np.where(positive, wave, 1))[positive]
    room[negative] = ((shifted - lower) / np.where(negative, -wave, 1))[negative]
    peaks = np.minimum(peak, room.min(axis=0) * 0.999)
    if np.any(peaks <= 0):
        raise ValueError("Reference-centered hip target is outside the actor bounds")
    targets = shifted + wave * peaks
    targets[strength == 0] = original[strength == 0]
    return targets, center, peaks


def floor_wrenches(geom1, geom2, positions, forces, feet, floor, com):
    """World forces and moments on each foot, about the cached subtree CoM.

    MuJoCo's contact_force is the wrench on geom2; reversing the endpoint order
    therefore reverses its sign. Contact free moments are included.
    """
    import jax.numpy as jp

    wrenches = []
    for foot in feet:
        sign = (geom2 == foot) & (geom1 == floor)
        sign = sign.astype(forces.dtype) - (
            (geom1 == foot) & (geom2 == floor)
        ).astype(forces.dtype)
        force = forces[:, :3] * sign[:, None]
        moment = jp.cross(positions - com, force) + forces[:, 3:] * sign[:, None]
        wrenches.append(jp.concatenate([force.sum(axis=0), moment.sum(axis=0)]))
    return jp.stack(wrenches)


def placement_summary(trace, start, end, minimum, dt, scale):
    """Use sustained contact onsets entirely within the settled window."""
    healthy = trace["valid"].astype(bool) & ~trace["fall"].astype(bool)
    contact = trace["contact"].astype(bool)
    positions = trace["foot_geom_center_m"]
    w, x, y, z = trace["torso_quaternion"].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    output = {}
    for side, name in enumerate(("left", "right")):
        onsets = np.flatnonzero(contact[:, side] & ~trace["previous_contact"][:, side])
        onsets = np.asarray([
            tick for tick in onsets
            if start <= tick and tick + minimum <= end
            and healthy[tick:tick + minimum].all()
            and contact[tick:tick + minimum, side].all()
        ], dtype=int)
        lead = np.sum(
            (positions[onsets, side, :2] - positions[onsets, 1 - side, :2])
            * heading[onsets], axis=1,
        )
        advances = np.diff(positions[onsets, side, :2], axis=0)
        output[name] = {
            "landings": len(onsets),
            "lands_ahead": int((lead > 0).sum()),
            "mean_lead_physical_mm": float(lead.mean() * 1000) if len(lead) else None,
            "mean_world_x_stride_physical_mm": float(advances[:, 0].mean() * 1000) if len(advances) else None,
            "mean_xy_stride_physical_mm": float(np.linalg.norm(advances, axis=1).mean() * 1000) if len(advances) else None,
            "mean_cycle_TB_s": float(np.diff(onsets).mean() * dt / scale) if len(onsets) > 1 else None,
        }
    return output


def physics_trace(env, before, action):
    """Replay the unchanged four-substep controller and expose solver reactions."""
    import jax
    import jax.numpy as jp
    from wr2.tools.diagnose_walking import target_trace

    _, target, _, _, _ = target_trace(env, before, action)
    active = env._active_indices
    qids, vids = env._joint_qpos_indices, env._joint_qvel_indices
    noise = before.info["actuator_noise"]
    torso = env.sys.mj_model.body("torso").id

    def step(p, _):
        position, velocity = p.q[qids], p.qd[vids]
        error = target - position
        kp = env._servo_kp * noise["kp"]
        ratio = env._servo_passive_active_ratio * noise["passive_active_ratio"]
        kp = jp.where(p.qacc[vids] * error < 0, kp * ratio, kp)
        raw = kp * error - env._servo_kv * noise["kv"] * velocity
        torque = env._controller_torque(p, target, noise)
        maximum = env._servo_max_speed_rad_s * noise["max_speed"]
        plateau = jp.minimum(env._servo_peak_torque_speed_rad_s, maximum - 1e-6)
        fraction = jp.clip(
            (jp.abs(velocity) - plateau) / jp.maximum(maximum - plateau, 1e-6), 0, 1
        )
        stall = env._servo_torque_limit_nm * noise["torque_limit"]
        motoring = stall + fraction * (env._servo_torque_at_max_speed_nm - stall)
        braking = env._servo_brake_torque_limit_nm * noise["brake_torque"]
        limit = jp.where(torque * velocity >= 0, motoring, braking)
        after = env._pipeline.step(env.sys, p, torque, env._debug)
        impl = after._impl
        friction = (impl.efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)) | (
            impl.efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
        )
        contact = impl.efc_type >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        friction_torque = impl.efc_J.T @ (impl.efc_force * friction)
        contact_torque = impl.efc_J.T @ (impl.efc_force * contact)
        forces = env._contact_forces(after)
        com = after.subtree_com[torso]
        wrench = floor_wrenches(
            after.contact.geom1, after.contact.geom2, after.contact.pos,
            forces, env._foot_geom_ids, env._floor_geom_id, com,
        )
        return after, {
            "position_rad": position[active],
            "velocity_rad_s": velocity[active],
            "error_rad": error[active],
            "raw_torque_nm": raw[active],
            "torque_nm": torque[active],
            "torque_limit_nm": limit[active],
            "torque_clipped": (jp.abs(raw - torque) > 1e-6)[active],
            "friction_torque_nm": friction_torque[vids][active],
            "contact_joint_torque_nm": contact_torque[vids][active],
            "passive_torque_nm": after.qfrc_passive[vids][active],
            "bias_torque_nm": after.qfrc_bias[vids][active],
            "floor_wrench": wrench,
            "com_world_m": com,
            "actuator_error_nm": jp.max(jp.abs(after.actuator_force - torque)),
            "constraint_error": jp.max(jp.abs(
                impl.efc_J.T @ impl.efc_force - after.qfrc_constraint
            )),
            "constraint_scale": jp.max(jp.abs(after.qfrc_constraint)),
            "floor_vertical_error_n": jp.max(jp.abs(
                wrench[:, 2] - env._foot_vertical_forces(after, forces)
            )),
        }

    return jax.lax.scan(step, before.pipeline_state, None, length=env._n_frames)


class SagittalCompensator:
    """Keep commanded site height and sole pitch while fixing the hip target.

    Knee and ankle pitch are the two unknowns. All remaining joint positions,
    including the hip, are fixed. Bounds include the actor's reachable range.
    Full foot roll/yaw are measured, rather than assumed constrained by two DOFs.
    """

    def __init__(self, model, names, home, action_scale, height_scale_m,
                 side="left", free_hip_roll=False):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.actuator(name).trnid[0] for name in names]
        self.qpos_indices = model.jnt_qposadr[joints]
        self.hip = names.index(f"{side}_hip_pitch")
        self.ka = np.asarray(
            [names.index(f"{side}_knee_pitch"), names.index(f"{side}_ankle_pitch")]
        )
        self.full_unknown = np.asarray([
            names.index(f"{side}_{joint}")
            for joint in (
                ("hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
                if free_hip_roll else
                ("hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
            )
        ])
        bounds = model.jnt_range[joints]
        self.lower = np.maximum(bounds[:, 0], np.asarray(home) - action_scale)
        self.upper = np.minimum(bounds[:, 1], np.asarray(home) + action_scale)
        self.height_scale_m = height_scale_m
        self.site = model.site(f"{side}_foot_center").id
        self.geom = model.geom(f"{side}_foot_collision").id
        self.torso = model.body("torso").id
        self.data.qpos[:] = model.key("walk_home").qpos
        mujoco.mj_kinematics(model, self.data)
        rotation = self.data.geom_xmat[self.geom].reshape(3, 3)
        self.normal_axis = int(np.argmax(np.abs(rotation[2])))
        self.normal_sign = float(np.sign(rotation[2, self.normal_axis]))

    def _pose(self, heading):
        rotation = self.data.geom_xmat[self.geom].reshape(3, 3)
        normal = rotation[:, self.normal_axis] * self.normal_sign
        pitch = np.arctan2(normal[:2] @ heading, normal[2])
        return self.data.site_xpos[self.site, 2], pitch, normal.copy()

    def solve(self, qpos, baseline_targets, hip_target):
        baseline_targets = np.clip(baseline_targets, self.lower, self.upper)
        self.data.qpos[:] = qpos
        self.data.qpos[self.qpos_indices] = baseline_targets
        mujoco.mj_kinematics(self.model, self.data)
        w, x, y, z = self.data.xquat[self.torso]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        heading = np.asarray([np.cos(yaw), np.sin(yaw)])
        height, pitch, normal = self._pose(heading)
        position = self.data.site_xpos[self.site].copy()
        self.data.qpos[self.qpos_indices[self.hip]] = hip_target

        def residual(target):
            self.data.qpos[self.qpos_indices[self.ka]] = target
            mujoco.mj_kinematics(self.model, self.data)
            new_height, new_pitch, _ = self._pose(heading)
            angle = (new_pitch - pitch + np.pi) % (2 * np.pi) - np.pi
            return np.asarray([(new_height - height) / self.height_scale_m, angle])

        initial = np.asarray(baseline_targets)[self.ka]
        result = least_squares(
            residual,
            initial,
            bounds=(self.lower[self.ka], self.upper[self.ka]),
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=60,
        )
        error = residual(result.x)
        height_error = abs(error[0]) * self.height_scale_m
        pitch_error = abs(error[1])
        if height_error > 1e-5 or pitch_error > 1e-5:
            raise ValueError(
                f"No reachable compensation: height={height_error:.6g} m, "
                f"pitch={pitch_error:.6g} rad, knee/ankle={result.x.tolist()}"
            )
        _, _, new_normal = self._pose(heading)
        motion = self.data.site_xpos[self.site] - position
        return result.x, {
            "height_error_m": height_error,
            "pitch_error_rad": pitch_error,
            "full_normal_change_rad": float(
                np.arccos(np.clip(normal @ new_normal, -1, 1))
            ),
            "forward_change_m": float(motion[:2] @ heading),
            "lateral_change_m": float(
                motion[:2] @ np.asarray([-heading[1], heading[0]])
            ),
        }

    def solve_full(self, qpos, baseline_targets, hip_target, initial=None):
        """Preserve height and full foot orientation.

        Hip pitch is fixed. The legacy left-only comparison fixes hip roll;
        bilateral coordination uses all five remaining joints and also preserves
        lateral foot position. Forward foot position remains free.
        """
        baseline_targets = np.clip(baseline_targets, self.lower, self.upper)
        self.data.qpos[:] = qpos
        self.data.qpos[self.qpos_indices] = baseline_targets
        mujoco.mj_kinematics(self.model, self.data)
        position = self.data.site_xpos[self.site].copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, self.data.geom_xmat[self.geom])
        w, x, y, z = self.data.xquat[self.torso]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        forward = np.asarray([np.cos(yaw), np.sin(yaw)])
        lateral = np.asarray([-forward[1], forward[0]])
        unknown = self.full_unknown
        self.data.qpos[self.qpos_indices[self.hip]] = hip_target

        def residual(target):
            self.data.qpos[self.qpos_indices[unknown]] = target
            mujoco.mj_kinematics(self.model, self.data)
            offset = self.data.site_xpos[self.site] - position
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[self.geom])
            angular = np.zeros(3)
            mujoco.mju_subQuat(angular, quat, target_quat)
            if len(unknown) == 5:
                return np.r_[
                    offset[2] / self.height_scale_m,
                    offset[:2] @ lateral / self.height_scale_m,
                    angular,
                ]
            return np.r_[offset[2] / self.height_scale_m, angular]

        result = least_squares(
            residual,
            np.asarray(baseline_targets)[unknown] if initial is None else initial,
            bounds=(self.lower[unknown], self.upper[unknown]),
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=200 if len(unknown) == 5 else 60,
        )
        error = residual(result.x)
        spatial = 2 if len(unknown) == 5 else 1
        position_error = np.linalg.norm(error[:spatial]) * self.height_scale_m
        orientation_error = np.linalg.norm(error[spatial:spatial + 3])
        if position_error > 1e-5 or orientation_error > 1e-5:
            raise ValueError(
                f"No reachable full-pose compensation: position={position_error:.6g} m, "
                f"orientation={orientation_error:.6g} rad, nfev={result.nfev}, "
                f"status={result.status}, optimality={result.optimality:.6g}"
            )
        offset = self.data.site_xpos[self.site] - position
        return (
            unknown,
            result.x,
            {
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "solver_evaluations": int(result.nfev),
                "forward_change_m": float(offset[:2] @ forward),
                "lateral_change_m": float(offset[:2] @ lateral),
            },
        )


    def solve_reachable_full(self, qpos, baseline_targets, hip_target, initial=None):
        """Project an infeasible hip request toward its original bounded pose.

        Eight bisections resolve the hip request more finely than the servo's
        .24-degree command unit. This tests local reachability, not global IK.
        """
        try:
            unknown, angles, error = self.solve_full(
                qpos, baseline_targets, hip_target, initial
            )
            return hip_target, unknown, angles, {**error, "projection_fraction": 1.0}
        except ValueError as exc:
            rejected = str(exc)
        origin = np.clip(baseline_targets, self.lower, self.upper)[self.hip]
        unknown, angles, error = self.solve_full(qpos, baseline_targets, origin)
        low, high = 0.0, 1.0
        best = (origin, unknown, angles, error)
        for _ in range(8):
            fraction = (low + high) / 2
            candidate = origin + fraction * (hip_target - origin)
            try:
                unknown, angles, error = self.solve_full(
                    qpos, baseline_targets, candidate, best[2]
                )
            except ValueError:
                high = fraction
            else:
                low = fraction
                best = (candidate, unknown, angles, error)
        actual, unknown, angles, error = best
        return actual, unknown, angles, {
            **error, "projection_fraction": low, "rejected_goal_error": rejected,
        }


def run(run_dir: Path, output: Path, bilateral=False):
    from wr2.locomotion.train import (
        _configure_backend_logging,
        _filter_optional_backend_import_messages,
    )

    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        import jax.numpy as jp
        from brax.io import model as brax_model
        from brax.training import types
        from brax.training.agents.ppo import checkpoint
        from wr2.locomotion.configs import load_training_config
        from wr2.locomotion.ppo import make_network_factory
        import wr2.locomotion.walking_env as env_module
        from wr2.reference.paper_walk import paper_command_grid
        from wr2.tools.diagnose_walking import target_trace

    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed diagnostic already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "high")
    cfg = load_training_config(run_dir / "training_config.yaml")
    assert cfg.environment.recipe == "tb_paper" and not cfg.ppo.normalize_observations
    dt, scale = 0.02, cfg.environment.normalization_scale
    command = jp.asarray([0.1 * scale, 0.0, 0.0])
    grid = paper_command_grid(scale)
    nearest = int(jp.argmin(jp.linalg.norm(jp.asarray(grid) - command, axis=1)))
    reference_class = env_module.PaperWalkReference
    env_module.PaperWalkReference = partial(
        reference_class, command_vectors=grid[[nearest]]
    )
    nominal = replace(
        cfg.environment,
        command_resample_steps=1001,
        randomization=replace(cfg.environment.randomization, enabled=False),
        push_linear_velocity_m_s=0.0,
        push_angular_velocity_rad_s=0.0,
    )
    try:
        env = env_module.WR2WalkingEnv(
            nominal, add_observation_noise=False, reward_discounting=cfg.ppo.discounting
        )
    finally:
        env_module.PaperWalkReference = reference_class
    model = mujoco.MjModel.from_xml_path(str(env.robot.directory / "scene_mjx.xml"))
    active = np.asarray(env._active_indices)
    names = [env.robot.actuator_names[int(i)] for i in active]
    home = np.asarray(env._home_ctrl)[active]
    action_scale = nominal.policy_action_scale_rad
    solver = SagittalCompensator(
        model, names, home, action_scale, env.zmp_reference_diagnostics.com_height_m,
        free_hip_roll=bilateral,
    )
    starts = np.asarray([180, 190, 200])
    horizon, treatment_steps, ramp_steps = (252, 180, 12) if bilateral else (99, 60, 6)
    best_step = json.loads((run_dir / "best_checkpoint.json").read_text())["step"]
    labels = ["80M", f"{round(best_step / 1e6)}M"]
    paths = [run_dir / "checkpoints/000080035840", run_dir / "best_params"]
    params = [
        tuple(
            checkpoint.load(str(p.resolve()))
            if p.is_dir()
            else brax_model.load_params(p)
        )
        for p in paths
    ]
    stacked = jax.tree.map(lambda *v: jp.stack(v), *params)
    net = make_network_factory(cfg.network)(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
    )
    key = jax.random.split(jax.random.split(jax.random.PRNGKey(22102))[1], 8)[0]

    def baseline_rollout():
        state = jax.vmap(env.reset)(jp.stack([key] * 2), jp.stack([command] * 2))

        def step(before, _):
            logits = jax.vmap(lambda p, o: net.policy_network.apply(p[0], p[1], o))(
                stacked, before.obs
            )
            action = net.parametric_action_distribution.mode(logits)
            after = jax.vmap(env.step)(before, action)
            return after, {"before": before, "action": action, "done": after.done}

        return jax.lax.scan(step, state, jp.arange(starts.max() + horizon))[1]

    print("BUILD_BASELINES", flush=True)
    base = jax.jit(baseline_rollout)()
    assert not np.asarray(base["done"]).any()
    cp = np.repeat(np.arange(2), len(starts))
    start = np.tile(starts, 2)
    original = np.asarray(base["action"])[
        np.arange(horizon)[:, None] + start[None, :], cp[None, :]
    ]
    before_q = np.asarray(base["before"].pipeline_state.q)[
        np.arange(horizon)[:, None] + start[None, :], cp[None, :]
    ]
    times = (np.arange(horizon)[:, None] + start[None, :]) * dt
    refs = np.asarray(
        jax.vmap(jax.vmap(lambda t: env._zmp_reference.get_reference(t, command)[0]))(
            jp.asarray(times)
        )
    )
    k = np.arange(horizon)
    u = np.clip(
        np.minimum(k / ramp_steps, (treatment_steps + ramp_steps - k) / ramp_steps),
        0,
        1,
    )
    strength = u * u * (3 - 2 * u)
    requested = home + action_scale * original
    full_errors = []
    reference_centres = {}
    target_centres = {}
    desired_hips = {}
    if bilateral:
        right = SagittalCompensator(
            model, names, home, action_scale,
            env.zmp_reference_diagnostics.com_height_m, side="right", free_hip_roll=True,
        )
        left_only, both = original.copy(), original.copy()
        peaks = {}
        weighted = strength[:, None]
        forward_mean = np.sum(
            (requested[:, :, right.hip] - requested[:, :, solver.hip]) * weighted,
            axis=0,
        ) / (2 * weighted.sum())
        print("SOLVE_BILATERAL_TARGETS", flush=True)
        for side, leg in (("left", solver), ("right", right)):
            center = forward_mean * (-1 if side == "left" else 1)
            targets, centre, peak = reference_hip_targets(
                requested[:, :, leg.hip], refs[:, :, active[leg.hip]], strength,
                leg.lower[leg.hip], leg.upper[leg.hip], np.deg2rad(7), center,
            )
            # 14-degree hip swing approximates TB's learned forward gait.
            target_centres[side] = np.rad2deg(centre).tolist()
            reference_centres[side] = np.rad2deg(
                np.sum(refs[:, :, active[leg.hip]] * weighted, axis=0) / weighted.sum()
            ).tolist()
            peaks[side] = np.rad2deg(peak).tolist()
            desired_hips[side] = targets
            leg_tape = original.copy()
            for i in range(len(cp)):
                warm_start = None
                for tick in np.flatnonzero(strength):
                    hip_target = float(targets[tick, i])
                    try:
                        actual_hip, unknown, angles, error = leg.solve_reachable_full(
                            before_q[tick, i], requested[tick, i], hip_target,
                            initial=warm_start,
                        )
                    except ValueError as exc:
                        (output / "failed_target.json").write_text(json.dumps({
                            "checkpoint": labels[cp[i]], "start": int(start[i]),
                            "tick": int(tick), "side": side, "error": str(exc),
                            "qpos_before": before_q[tick, i].tolist(),
                            "baseline_targets_rad": requested[tick, i].tolist(),
                            "hip_target_rad": hip_target,
                            "lower_rad": leg.lower.tolist(),
                            "upper_rad": leg.upper.tolist(),
                        }, indent=2) + "\n")
                        raise
                    warm_start = angles
                    leg_tape[tick, i, leg.hip] = (
                        actual_hip - home[leg.hip]
                    ) / action_scale
                    leg_tape[tick, i, unknown] = (angles - home[unknown]) / action_scale
                    full_errors.append({
                        "checkpoint": labels[cp[i]], "start": int(start[i]),
                        "tick": int(tick), "side": side,
                        "hip_requested_rad": hip_target, "hip_applied_rad": actual_hip,
                        **error,
                    })
            changed = np.r_[leg.hip, leg.full_unknown]
            both[:, :, changed] = leg_tape[:, :, changed]
            if side == "left":
                left_only = leg_tape
        np.testing.assert_array_equal(left_only[:, :, :6], both[:, :, :6])
        np.testing.assert_array_equal(left_only[:, :, 6:], original[:, :, 6:])
        np.testing.assert_allclose(
            (desired_hips["right"] - desired_hips["left"]).mean(axis=0),
            (requested[:, :, right.hip] - requested[:, :, solver.hip]).mean(axis=0),
            rtol=0, atol=1e-6,
        )
        modes = ["baseline", "left_coordinated", "bilateral_coordinated", "baseline_duplicate"]
        controls = [original, left_only, both, original]
    else:
        shifted = requested[:, :, solver.hip] - np.deg2rad(3) * strength[:, None]
        shaped = centered_reference_wave(refs[:, :, active[solver.hip]], strength)
        room = np.full_like(shaped, np.inf)
        positive, negative = shaped > 1e-8, shaped < -1e-8
        room[positive] = (
            (home[solver.hip] + action_scale - shifted) / np.where(positive, shaped, 1)
        )[positive]
        room[negative] = (
            (shifted - (home[solver.hip] - action_scale)) / np.where(negative, -shaped, 1)
        )[negative]
        peak = np.minimum(np.deg2rad(5), room.min(axis=0) * 0.999)
        peaks = np.rad2deg(peak).tolist()
        hip_only = original.copy()
        hip_only[:, :, solver.hip] = (
            shifted + shaped * peak - home[solver.hip]
        ) / action_scale
        hip_only[strength == 0, :, solver.hip] = original[strength == 0, :, solver.hip]
        coordinated, full_pose = hip_only.copy(), hip_only.copy()
        errors = []
        print("SOLVE_COORDINATED_TARGETS", peaks, flush=True)
        for i in range(len(cp)):
            for tick in range(horizon):
                if hip_only[tick, i, solver.hip] == original[tick, i, solver.hip]:
                    continue
                hip_target = float(home[solver.hip] + action_scale * hip_only[tick, i, solver.hip])
                targets, error = solver.solve(before_q[tick, i], requested[tick, i], hip_target)
                coordinated[tick, i, solver.ka] = (targets - home[solver.ka]) / action_scale
                errors.append({
                    "checkpoint": labels[cp[i]], "start": int(start[i]),
                    "tick": tick, **error,
                })
                unknown, targets, error = solver.solve_full(
                    before_q[tick, i], requested[tick, i], hip_target
                )
                full_pose[tick, i, unknown] = (targets - home[unknown]) / action_scale
                full_errors.append({
                    "checkpoint": labels[cp[i]], "start": int(start[i]),
                    "tick": tick, **error,
                })
        (output / "ik_validation.json").write_text(json.dumps(errors, indent=2) + "\n")
        np.testing.assert_array_equal(hip_only[:, :, solver.hip], coordinated[:, :, solver.hip])
        np.testing.assert_array_equal(hip_only[:, :, solver.hip], full_pose[:, :, solver.hip])
        other = np.setdiff1d(np.arange(env.action_size), np.r_[solver.hip, solver.ka])
        np.testing.assert_array_equal(coordinated[:, :, other], original[:, :, other])
        np.testing.assert_array_equal(full_pose[:, :, 6:], original[:, :, 6:])
        np.testing.assert_array_equal(full_pose[:, :, 1], original[:, :, 1])
        modes = ["baseline", "hip_only", "coordinated", "full_pose", "baseline_duplicate"]
        controls = [original, hip_only, coordinated, full_pose, original]
    (output / "full_ik_validation.json").write_text(json.dumps(full_errors, indent=2) + "\n")
    assert all(np.abs(tape).max() <= 1 for tape in controls)
    tapes = jp.asarray(
        np.stack(controls, axis=2).reshape(horizon, -1, env.action_size)
    )
    cp_all, start_all = np.repeat(cp, len(modes)), np.repeat(start, len(modes))
    initial = jax.tree.map(lambda v: v[start_all, cp_all], base["before"])
    branch_params = jax.tree.map(lambda v: v[cp_all], stacked)
    reward_names = list(
        json.loads((run_dir / "run_config.json").read_text())["training_config"][
            "reward_weights"
        ]
    )
    feet = env._foot_geom_ids

    def values(state):
        return jax.vmap(lambda p, o: net.value_network.apply(p[0], p[2], o))(
            branch_params, state.obs
        )

    def rollout():
        def step(carry, tick):
            before, alive = carry
            action = tapes[tick]
            after = jax.vmap(env.step)(before, action)

            def sample(b, a, act):
                p, m = a.pipeline_state, a.metrics
                substeps = {}
                if bilateral:
                    p, substeps = physics_trace(env, b, act)
                _, delivered, low, high, slew = target_trace(env, b, act)
                forces = env._contact_forces(p)
                pair = ((p.contact.geom1 == feet[0]) & (p.contact.geom2 == feet[1])) | (
                    (p.contact.geom1 == feet[1]) & (p.contact.geom2 == feet[0])
                )
                pair &= (p.contact.dist < 0) & (p._impl.contact.efc_address >= 0)
                record = {
                    "qpos": p.q,
                    "qvel": p.qd,
                    "torso_quaternion": p.x.rot[env._root_link_index],
                    "foot_geom_center_m": p.geom_xpos[feet],
                    "foot_geom_rotation": p.geom_xmat[feet],
                    "contact": jp.asarray(
                        [
                            m["left_foot_contact_per_step"],
                            m["right_foot_contact_per_step"],
                        ]
                    )
                    > 0,
                    "previous_contact": b.info["previous_contact"],
                    "action": act,
                    "current_request_rad": home + action_scale * act,
                    "actual_rad": p.q[env._joint_qpos_indices][active],
                    "delivered_rad": delivered[active],
                    "command_m_s": command,
                    "target_clip": (low | high)[active],
                    "target_slew": slew[active],
                    "target_reconstruction_error_rad": jp.where(
                        a.done == 0,
                        jp.max(jp.abs(delivered - a.info["previous_target"])),
                        0.0,
                    ),
                    "velocity_m_s": jp.asarray(
                        [
                            m["forward_velocity_m_s_per_step"],
                            m["lateral_velocity_m_s_per_step"],
                        ]
                    ),
                    "yaw_rate_rad_s": m["yaw_rate_rad_s_per_step"],
                    "path_delta_m": p.q[:2] - b.pipeline_state.q[:2],
                    "foot_self_force_n": jp.max(
                        jp.where(pair, jp.linalg.norm(forces[:, :3], axis=1), 0.0)
                    ),
                    "foot_vertical_force_n": jp.asarray(
                        [
                            m["left_foot_vertical_force_n_per_step"],
                            m["right_foot_vertical_force_n_per_step"],
                        ]
                    ),
                    "rewards": jp.stack(
                        [m["paper_reward/" + name] for name in reward_names]
                    ),
                    "total_reward": a.reward,
                    "qualified_air_time_s": m["qualified_air_time_s_per_step"],
                    "rejected_air_time_s": m["rejected_air_time_s_per_step"],
                    "fall": m["fall"],
                    "nonfinite": m["nonfinite_state"],
                }
                if bilateral:
                    record.update({f"substep_{k}": v for k, v in substeps.items()})
                    record["physics_replay_error"] = jp.where(
                        a.done == 0,
                        jp.maximum(
                            jp.max(jp.abs(p.q - a.pipeline_state.q)),
                            jp.max(jp.abs(p.qd - a.pipeline_state.qd)),
                        ),
                        0.0,
                    )
                return record

            record = jax.vmap(sample)(before, after, action)
            finite = jp.ones(len(cp_all), bool)
            for v in (*record.values(), *after.obs.values()):
                finite &= jp.isfinite(v).reshape(len(cp_all), -1).all(axis=1)
            record["valid"] = alive & finite
            record["fall_event"] = alive & finite & (after.metrics["fall"] > 0)
            record["nonfinite_event"] = alive & (
                ~finite | (after.metrics["nonfinite_state"] > 0)
            )
            surviving = alive & finite & (after.done == 0)
            following = jax.tree.map(
                lambda new, old: jp.where(
                    surviving.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
                ),
                after,
                before,
            )
            return (following, surviving), record

        (last, alive), record = jax.lax.scan(
            step, (initial, jp.ones(len(cp_all), bool)), jp.arange(horizon)
        )
        return record, values(initial), values(last), alive

    print("RUN_MATCHED_ROLLOUTS", len(cp_all), horizon, flush=True)
    traces, v0, vend, alive = jax.tree.map(np.asarray, jax.jit(rollout)())
    results = []
    for i in range(len(cp_all)):
        mode = modes[i % len(modes)]
        trace = {k: v[:, i] for k, v in traces.items()}
        valid = trace["valid"]
        n = int(valid.sum())
        assert np.array_equal(valid, np.arange(horizon) < n)
        np.testing.assert_allclose(
            trace["rewards"][valid].sum(axis=1), trace["total_reward"][valid], atol=1e-5
        )
        assert trace["target_reconstruction_error_rad"][valid].max() < 1e-6
        if bilateral:
            assert trace["physics_replay_error"][valid].max() < 1e-5
            assert trace["substep_actuator_error_nm"][valid].max() < 1e-5
            assert trace["substep_floor_vertical_error_n"][valid].max() < 1e-4
            assert np.all(
                trace["substep_constraint_error"][valid]
                <= 2e-5 * (1 + trace["substep_constraint_scale"][valid])
            )
        name = f"{labels[cp_all[i]]}_start{start_all[i]}_{mode}"
        np.savez_compressed(output / (name + ".npz"), **trace)
        discount = cfg.ppo.discounting ** np.arange(n)
        g = float(trace["total_reward"][valid].astype(float) @ discount)
        components = trace["rewards"][valid].astype(float).T @ discount
        during = valid & (np.arange(horizon) < treatment_steps + ramp_steps)
        result = {
            "name": name,
            "checkpoint": labels[cp_all[i]],
            "start_tick": int(start_all[i]),
            "mode": mode,
            "samples": n,
            "fell": bool(trace["fall_event"].any()),
            "nonfinite": bool(trace["nonfinite_event"].any()),
            "discounted_return": g,
            "bootstrapped_return": g
            + (cfg.ppo.discounting**n * float(vend[i]) if alive[i] else 0.0),
            "initial_critic_value": float(v0[i]),
            "discounted_components": dict(
                zip(reward_names, components.tolist(), strict=True)
            ),
            "net_forward_TB_m_s": float(
                trace["path_delta_m"][during, 0].mean() / dt / scale
            ),
            "forward_velocity_std_TB_m_s": float(
                trace["velocity_m_s"][during, 0].std() / scale
            ),
            "backward_velocity_fraction": float(
                (trace["velocity_m_s"][during, 0] < 0).mean()
            ),
            "lateral_velocity_std_TB_m_s": float(
                trace["velocity_m_s"][during, 1].std() / scale
            ),
        }
        if bilateral:
            steady_start = ramp_steps + 2 * round(nominal.gait_cycle_s / dt)
            steady = valid & (np.arange(horizon) >= steady_start) & (
                np.arange(horizon) < treatment_steps
            )
            result["steady"] = None
            if steady.any():
                result["steady"] = {
                    "samples": int(steady.sum()),
                    "complete_window": bool(steady.sum() == treatment_steps - steady_start and not trace["fall_event"][steady].any()),
                    "net_forward_TB_m_s": float(trace["path_delta_m"][steady, 0].mean() / dt / scale),
                    "forward_velocity_std_TB_m_s": float(trace["velocity_m_s"][steady, 0].std() / scale),
                    "lateral_velocity_std_TB_m_s": float(trace["velocity_m_s"][steady, 1].std() / scale),
                    "yaw_rate_rms_TB_rad_s": float(np.sqrt(np.mean(trace["yaw_rate_rad_s"][steady]**2)) * scale),
                    "reward_rate_per_TB_s": float(trace["total_reward"][steady].mean() * scale / dt),
                    "reward_rates_per_TB_s": dict(zip(
                        reward_names,
                        (trace["rewards"][steady].mean(axis=0) * scale / dt).tolist(),
                        strict=True,
                    )),
                    "target_range_deg": np.rad2deg(np.ptp(trace["delivered_rad"][steady], axis=0)).tolist(),
                    "target_mean_deg": np.rad2deg(trace["delivered_rad"][steady].mean(axis=0)).tolist(),
                    "actual_range_deg": np.rad2deg(np.ptp(trace["actual_rad"][steady], axis=0)).tolist(),
                    "target_error_rmse_deg": np.rad2deg(np.sqrt(np.mean(trace["substep_error_rad"][steady]**2, axis=(0, 1)))).tolist(),
                    "torque_clip_fraction": trace["substep_torque_clipped"][steady].mean(axis=(0, 1)).tolist(),
                    "peak_torque_nm": np.abs(trace["substep_torque_nm"][steady]).max(axis=(0, 1)).tolist(),
                    "mean_floor_yaw_moment_nm": trace["substep_floor_wrench"][steady, :, :, 5].mean(axis=(0, 1)).tolist(),
                    "placements": placement_summary(
                        trace, steady_start, treatment_steps,
                        env._air_time_min_stance_steps, dt, scale,
                    ),
                    "double_support_fraction": float(trace["contact"][steady].all(axis=1).mean()),
                    "flight_fraction": float((~trace["contact"][steady].any(axis=1)).mean()),
                    "foot_self_force_peak_n": float(trace["foot_self_force_n"][steady].max()),
                }
        results.append(result)
    for i in range(0, len(results), len(modes)):
        if bilateral:
            common = traces["valid"][:, i + 1] & traces["valid"][:, i + 2]
            np.testing.assert_array_equal(traces["action"][:, i + 1, :6], traces["action"][:, i + 2, :6])
            np.testing.assert_allclose(traces["delivered_rad"][common, i + 1, :6], traces["delivered_rad"][common, i + 2, :6], atol=1e-6)
        else:
            for offset in (2, 3):
                np.testing.assert_array_equal(
                    traces["action"][:, i + 1, solver.hip],
                    traces["action"][:, i + offset, solver.hip],
                )
                np.testing.assert_allclose(
                    traces["delivered_rad"][:, i + 1, solver.hip],
                    traces["delivered_rad"][:, i + offset, solver.hip],
                    atol=1e-6,
                )
        duplicate = i + len(modes) - 1
        np.testing.assert_array_equal(
            traces["contact"][:, i], traces["contact"][:, duplicate]
        )
        assert (
            abs(results[i]["discounted_return"] - results[duplicate]["discounted_return"])
            < 1e-4
        )
        for r in results[i : i + len(modes)]:
            for field in [
                "discounted_return",
                "bootstrapped_return",
                "net_forward_TB_m_s",
            ]:
                r["delta_" + field] = r[field] - results[i][field]
    report = {
        "normalization_scale": scale,
        "control_period_s": dt,
        "active_joint_names": names,
        "reward_names": reward_names,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            (run_dir / "training_config.yaml").read_bytes()
        ).hexdigest(),
        "runtime": {
            "backend": jax.default_backend(),
            "matmul_precision": "high",
            "packages": {
                n: metadata.version(n)
                for n in ["jax", "brax", "mujoco", "mujoco-mjx", "flax"]
            },
        },
        "scope": "nominal matched states; deterministic other action tapes held fixed; no training update",
        "request_field": "current_request_rad includes the diagnostic intervention; baseline gives the original policy request",
        "compensation": (
            "left_coordinated and bilateral_coordinated: identical left targets; preserve original requested height, lateral position and full sole orientation; project locally infeasible hip goals within unchanged actor/mechanical bounds; forward position free"
            if bilateral else
            "coordinated: preserve limit-bounded requested height/pitch with knee/ankle; full_pose: preserve height/full orientation with four remaining left joints, fixed hip roll, free horizontal position; same hip target"
        ),
        "normal_axis": solver.normal_axis,
        "normal_sign": solver.normal_sign,
        "hip_wave_peak_deg": peaks,
        "hip_constant_offset_forward_deg": None if bilateral else 3.0,
        "hip_reference_mean_deg": reference_centres,
        "hip_desired_mean_deg": target_centres,
        "reference_wave_rule": "zero mean over applied treatment window, unit maximum applied deviation; excludes startup-cycle normalization",
        "horizon": horizon,
        "treatment_steps": treatment_steps,
        "ramp_steps": ramp_steps,
        "gamma": cfg.ppo.discounting,
        "reference_lookup_command": grid[nearest].tolist(),
        "results": results,
    }
    if bilateral:
        report["physics_substep_s"] = env.robot.simulation_timestep_s
        report["steady_window_ticks"] = [steady_start, treatment_steps]
        report["substep_scope"] = "native controller replay from identical state; raw terminal physics retained; solver contact frames and subtree CoM remain cached consistently; friction projected from friction constraint rows"
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        "COMPLETE",
        json.dumps(
            {"branches": len(results), "falls": sum(r["fell"] for r in results)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bilateral", action="store_true", help="Compare identical left-only and bilateral reference-centered controls with substep force/torque recording")
    args = parser.parse_args()
    run(args.run_dir, args.output, args.bilateral)


if __name__ == "__main__":
    main()
