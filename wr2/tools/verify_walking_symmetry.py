"""Pre-training model, reference, actor and substep-impulse verification.

Sagittal reflection follows the polar/axial-vector rules used by WR1 and
Yu et al. (2018, arXiv:1801.08093). This tool measures symmetry; it does not
enable symmetry loss or alter the training environment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from wr2.tools.diagnose_walking import (
    foot_contact_samples,
    target_trace,
)


REFLECTION = np.diag([1.0, -1.0, 1.0])


def joint_reflection(model, names):
    """Derive coordinate signs from neutral world joint axes, not WR1 signs."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    indices = {name: index for index, name in enumerate(names)}
    source, signs = [], []
    for name in names:
        partner = (
            name.replace("left_", "right_", 1)
            if name.startswith("left_")
            else name.replace("right_", "left_", 1)
        )
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, partner)
        source.append(indices[partner])
        signs.append(
            float(np.sign(data.xaxis[other] @ (-REFLECTION @ data.xaxis[joint])))
        )
    return np.asarray(source), np.asarray(signs)


def mirror_actor(observation, source, signs, active, *, heading=True):
    """Reflect every newest-first frame of WR2's actual actor input layout."""
    frame_size = 55 + 2 * heading
    frames = observation.reshape((*observation.shape[:-1], -1, frame_size))
    mirrored = frames.at[..., :2].set(-frames[..., :2])  # half-cycle phase shift
    mirrored = mirrored.at[..., 2:5].set(frames[..., 2:5] * jp.asarray([1, -1, -1]))
    for start in (5, 22):
        mirrored = mirrored.at[..., start : start + 17].set(
            frames[..., start + source] * signs
        )
    active_lookup = (
        jp.full(17, -1, dtype=jp.int32).at[active].set(jp.arange(len(active)))
    )
    active_source = active_lookup[source[active]]
    mirrored = mirrored.at[..., 39:49].set(
        frames[..., 39 + active_source] * signs[active]
    )
    mirrored = mirrored.at[..., 49:52].set(frames[..., 49:52] * jp.asarray([-1, 1, -1]))
    mirrored = mirrored.at[..., 52:55].set(frames[..., 52:55] * jp.asarray([1, -1, 1]))
    if heading:
        mirrored = mirrored.at[..., 55:57].set(frames[..., 55:57] * jp.asarray([-1, 1]))
    return mirrored.reshape(observation.shape)


def summarize_error(values):
    values = np.asarray(values)
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "max_abs": float(np.max(np.abs(values))),
    }


def conditional_mean(values, mask):
    """An unsupported swing foot has no slip measurement, not zero or NaN."""
    return float(np.mean(values[mask])) if np.any(mask) else None


def full_mass_matrix(model, data):
    """Support both MuJoCo APIs allowed by WR2's training dependency."""
    matrix = np.empty((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, matrix)
    except TypeError:
        mujoco.mj_fullM(model, matrix, data.qM)
    return matrix


def verify_model_reference(env, model, source, signs):
    """Measure limits, world anchors/axes, leg geometry/inertia and full lookup."""
    names = env.robot.actuator_names
    joint_ids = np.asarray([model.joint(name).id for name in names])
    qids = model.jnt_qposadr[joint_ids]
    active = np.asarray(env._active_indices)
    home = np.asarray(env._home_ctrl)
    lower, upper = np.asarray(env._ctrl_lower), np.asarray(env._ctrl_upper)
    reflected_limits = np.sort(np.stack([lower[source], upper[source]]) * signs, axis=0)
    report = {
        "joint_signs": dict(zip(names, signs.tolist(), strict=True)),
        "limit_error_deg": summarize_error(
            np.rad2deg(reflected_limits - [lower, upper])
        ),
        "home_error_deg": summarize_error(np.rad2deg(home[source] * signs - home)),
        "pairs": {},
    }
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(env._default_qpos)
    data.qpos[qids] = 0
    mujoco.mj_forward(model, data)
    for i, name in enumerate(names):
        if not name.startswith("left_"):
            continue
        j, k = joint_ids[i], joint_ids[source[i]]
        body, other = model.jnt_bodyid[j], model.jnt_bodyid[k]
        rotation, other_rotation = (
            data.ximat[body].reshape(3, 3),
            data.ximat[other].reshape(3, 3),
        )
        inertia = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        other_inertia = (
            other_rotation @ np.diag(model.body_inertia[other]) @ other_rotation.T
        )
        report["pairs"][name] = {
            "anchor_error_mm": float(
                1000 * np.linalg.norm(data.xanchor[k] - REFLECTION @ data.xanchor[j])
            ),
            "axis_error_deg": float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            data.xaxis[k] @ (-REFLECTION @ data.xaxis[j]) * signs[i],
                            -1,
                            1,
                        )
                    )
                )
            ),
            "mass_difference_kg": float(model.body_mass[other] - model.body_mass[body]),
            "inertial_center_error_mm": float(
                1000 * np.linalg.norm(data.xipos[other] - REFLECTION @ data.xipos[body])
            ),
            "inertia_relative_error": float(
                np.linalg.norm(other_inertia - REFLECTION @ inertia @ REFLECTION)
                / np.linalg.norm(inertia)
            ),
        }

    sites = [model.site(f"{side}_foot_center").id for side in ("left", "right")]
    bodies = [model.body(f"{side}_foot").id for side in ("left", "right")]
    geoms = [model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
    neutral_rotations = data.xmat[bodies].reshape(2, 3, 3).copy()
    foot_frame_reflection = np.asarray(
        [
            neutral_rotations[i].T @ REFLECTION @ neutral_rotations[1 - i]
            for i in range(2)
        ]
    )
    # Include off-home poses; neutral-only checks cannot validate axis signs.
    rng = np.random.default_rng(20261007)
    fk_errors, gravity_errors, com_errors, mass_matrix_errors = [], [], [], []
    for _ in range(64):
        pose = home.copy()
        pose[active] = np.clip(
            pose[active] + rng.uniform(-0.3, 0.3, len(active)),
            lower[active],
            upper[active],
        )
        data.qpos[:] = np.asarray(env._default_qpos)
        data.qpos[qids] = pose
        mujoco.mj_forward(model, data)
        feet, rotations, com = (
            data.site_xpos[sites].copy(),
            data.xmat[bodies].reshape(2, 3, 3).copy(),
            data.subtree_com[1].copy(),
        )
        matrix = full_mass_matrix(model, data)
        dofs = model.jnt_dofadr[joint_ids][active]
        matrix = matrix[np.ix_(dofs, dofs)]
        data.qpos[qids] = pose[source] * signs
        mujoco.mj_forward(model, data)
        fk_errors.append(1000 * (data.site_xpos[sites] - feet[::-1] @ REFLECTION))
        # Gravity in the foot frame tolerates mirrored local CAD frame rotations.
        gravity = np.einsum("bij,j->bi", rotations.transpose(0, 2, 1), [0, 0, -1])
        mirrored_gravity = np.einsum(
            "bij,j->bi",
            data.xmat[bodies].reshape(2, 3, 3).transpose(0, 2, 1),
            [0, 0, -1],
        )
        gravity_errors.append(
            mirrored_gravity
            - np.einsum("bij,bj->bi", foot_frame_reflection, gravity[::-1])
        )
        com_errors.append(1000 * (data.subtree_com[1] - REFLECTION @ com))
        other_matrix = full_mass_matrix(model, data)
        other_matrix = other_matrix[np.ix_(dofs, dofs)]
        permutation = np.asarray([list(active).index(int(source[j])) for j in active])
        expected = (
            matrix[np.ix_(permutation, permutation)]
            * signs[active, None]
            * signs[None, active]
        )
        mass_matrix_errors.append(
            np.linalg.norm(other_matrix - expected) / np.linalg.norm(expected)
        )
    report["fk_position_error_mm"] = summarize_error(fk_errors)
    report["foot_local_gravity_error"] = summarize_error(gravity_errors)
    report["whole_robot_com_error_mm"] = summarize_error(com_errors)
    report["leg_mass_matrix_relative_error"] = summarize_error(mass_matrix_errors)
    report["collision_size_difference_mm"] = (
        1000 * (model.geom_size[geoms[0]] - model.geom_size[geoms[1]])
    ).tolist()
    report["collision_friction_difference"] = (
        model.geom_friction[geoms[0]] - model.geom_friction[geoms[1]]
    ).tolist()

    lookup = np.asarray(env._zmp_reference._joint_lookup)
    mirrored_lookup = np.roll(lookup, -lookup.shape[1] // 2, axis=1)
    error = mirrored_lookup - lookup[..., source] * signs
    report["reference_joint_error_deg"] = summarize_error(
        np.rad2deg(error[..., active])
    )
    report["reference_joint_error_deg_per_joint"] = {
        names[j]: summarize_error(np.rad2deg(error[..., j])) for j in active
    }
    reference_fk_error = []
    for poses in lookup:
        feet = []
        for pose in poses:
            data.qpos[:] = np.asarray(env._default_qpos)
            data.qpos[qids] = pose
            mujoco.mj_forward(model, data)
            feet.append(data.site_xpos[sites].copy())
        feet = np.asarray(feet)
        reference_fk_error.append(
            1000 * (np.roll(feet, -len(feet) // 2, axis=0) - feet[:, ::-1] @ REFLECTION)
        )
    report["reference_foot_error_mm"] = summarize_error(reference_fk_error)
    return report


def fresh_kinematics(model, pipeline):
    """Refresh post-integration kinematics without rerunning dynamics/solver."""
    from mujoco.mjx._src import smooth

    return smooth.com_vel(
        model, smooth.com_pos(model, smooth.kinematics(model, pipeline))
    )


def total_momentum(model, data):
    """Sum world linear momentum at each body's inertial center."""
    offset = data.xipos - data.subtree_com[model.body_rootid]
    velocity = data.cvel[:, 3:] + jp.cross(data.cvel[:, :3], offset)
    return jp.sum(model.body_mass[:, None] * velocity, axis=0)


def contact_impulses(force, dt):
    """Integrate all substeps, retaining propulsion/braking before cancellation."""
    return (
        jp.sum(force, axis=0) * dt,
        jp.sum(jp.maximum(force[..., 0], 0), axis=0) * dt,
        jp.sum(jp.maximum(-force[..., 0], 0), axis=0) * dt,
    )


def replay_control(env, before, action):
    """Replay the exact held target and integrate each substep's solved forces."""
    _, target, _, _, _ = target_trace(env, before, action)
    dt = env.robot.simulation_timestep_s
    momentum_before = total_momentum(
        env.sys, fresh_kinematics(env.sys, before.pipeline_state)
    )

    def substep(current, _):
        torque = env._controller_torque(current, target, before.info["actuator_noise"])
        after = env._pipeline.step(env.sys, current, torque, env._debug)
        forces, slip, counts = foot_contact_samples(env, after)
        from mujoco.mjx._src import support

        mass_matrix = support.full_m(env.sys, after)
        generalized_residual = (
            mass_matrix @ after.qacc - after.qfrc_smooth - after.qfrc_constraint
        )[:3]
        return after, (forces, slip, counts, generalized_residual)

    pipeline, (force, slip, counts, solver_residual) = jax.lax.scan(
        substep, before.pipeline_state, (), env._n_frames
    )
    impulse, positive, braking = contact_impulses(force, dt)
    fresh = fresh_kinematics(env.sys, pipeline)
    momentum_change = total_momentum(env.sys, fresh) - momentum_before
    residual = (
        momentum_change
        - jp.sum(impulse, axis=0)
        - jp.sum(env.sys.body_mass) * env.sys.opt.gravity * env.robot.control_period_s
    )
    support = counts > 0
    return pipeline, {
        "impulse_world_ns": impulse,
        "propulsive_impulse_ns": positive,
        "braking_impulse_ns": braking,
        "momentum_residual_ns": residual,
        "solver_residual_ns": jp.sum(solver_residual, axis=0) * dt,
        "support_fraction": jp.mean(support.astype(jp.float32), axis=0),
        "support_slip_m_s": jp.sum(jp.where(support, slip, 0), axis=0)
        / jp.maximum(jp.sum(support, axis=0), 1),
        "foot_position_m": fresh.site_xpos[env._foot_site_ids],
        "torso_position_m": fresh.xpos[env._root_link_index + 1],
    }


def summarize_rollout(trace, names, active, scale_deg):
    """Summarize only valid, settled samples; retain paired per-seed results."""
    mask = trace["valid"] & (np.arange(len(trace["valid"]))[:, None] >= 200)
    result = {
        "fall_rate": float(np.mean(np.any(trace["fall"] > 0, axis=0))),
        "nonfinite_rate": float(np.mean(np.any(trace["nonfinite"] > 0, axis=0))),
        "episode_length_per_run": np.sum(trace["valid"], axis=0).tolist(),
        "physics_replay_max_q_error": float(np.max(trace["replay_q_error"][mask])),
        "momentum_residual_ns": summarize_error(trace["momentum_residual_ns"][mask]),
        "solver_residual_ns": summarize_error(trace["solver_residual_ns"][mask]),
        "momentum_residual_ns_per_axis": [
            summarize_error(trace["momentum_residual_ns"][mask, axis])
            for axis in range(3)
        ],
        "solver_residual_ns_per_axis": [
            summarize_error(trace["solver_residual_ns"][mask, axis])
            for axis in range(3)
        ],
        "actor_mirror_error_deg": summarize_error(
            trace["mirror_action_error"][mask] * scale_deg
        ),
        "actor_mirror_error_deg_per_joint": {
            names[j]: summarize_error(trace["mirror_action_error"][mask, i] * scale_deg)
            for i, j in enumerate(active)
        },
        "command_groups": [],
    }
    for command in np.unique(trace["command_m_s"]):
        group = mask & np.isclose(trace["command_m_s"], command)
        speed_error = np.abs(trace["vx_m_s"] - command)
        entry = {
            "command_m_s": float(command),
            "speed_mae_m_s": float(np.mean(speed_error[group])),
            "speed_mae_per_run": [
                float(np.mean(speed_error[:, run][group[:, run]]))
                for run in np.flatnonzero(np.any(group, axis=0))
            ],
            "actor_mirror_error_deg": summarize_error(
                trace["mirror_action_error"][group] * scale_deg
            ),
            "quarters": [],
            "cycles": [],
        }
        runs = np.flatnonzero(np.any(group, axis=0))
        relative_feet = (
            trace["foot_position_m"] - trace["torso_position_m"][..., None, :]
        )
        cycle_ids = np.arange(len(mask)) // 36
        for run in runs:
            for cycle in np.unique(cycle_ids[group[:, run]]):
                samples = (cycle_ids == cycle) & group[:, run]
                if samples.sum() != 36:
                    continue
                foot = relative_feet[:, run][samples]
                shifted_foot = np.roll(foot, -18, axis=0)[:, ::-1] @ REFLECTION
                joint = trace["joint_position_deg"][:, run][samples]
                source = np.asarray(
                    [
                        names.index(
                            name.replace("left_", "right_", 1)
                            if name.startswith("left_")
                            else name.replace("right_", "left_", 1)
                        )
                        for name in names
                    ]
                )
                joint_error = (
                    np.roll(joint, -18, axis=0)[:, active] + joint[:, source[active]]
                )
                support = trace["support_fraction"][:, run][samples]
                entry["cycles"].append(
                    {
                        "run": int(run),
                        "cycle": int(cycle),
                        "mean_vx_m_s": float(np.mean(trace["vx_m_s"][:, run][samples])),
                        "joint_half_cycle_mismatch_deg": summarize_error(joint_error),
                        "relative_foot_half_cycle_mismatch_mm": summarize_error(
                            1000 * (foot - shifted_foot)
                        ),
                        "support_fraction_lr": np.mean(support, axis=0).tolist(),
                        "net_forward_impulse_ns_lr": np.sum(
                            trace["impulse_world_ns"][:, run][samples, :, 0], axis=0
                        ).tolist(),
                        "touchdown_phase_deg_lr": [
                            np.rad2deg(
                                trace["phase_rad"][:, run][samples][
                                    (support[:, side] > 0.5)
                                    & (np.roll(support[:, side], 1) <= 0.5)
                                ]
                            ).tolist()
                            for side in range(2)
                        ],
                    }
                )
        for quarter in range(4):
            samples = group & (np.floor(trace["phase_rad"] * 2 / np.pi) == quarter)
            entry["quarters"].append(
                {
                    "quarter": quarter,
                    "vx_m_s": float(np.mean(trace["vx_m_s"][samples])),
                    "mean_impulse_ns_per_control_lr_xyz": np.mean(
                        trace["impulse_world_ns"][samples], axis=0
                    ).tolist(),
                    "positive_forward_impulse_ns_lr": np.mean(
                        trace["propulsive_impulse_ns"][samples], axis=0
                    ).tolist(),
                    "braking_forward_impulse_ns_lr": np.mean(
                        trace["braking_impulse_ns"][samples], axis=0
                    ).tolist(),
                    "support_fraction_lr": np.mean(
                        trace["support_fraction"][samples], axis=0
                    ).tolist(),
                    "support_slip_mm_s_lr": [
                        conditional_mean(
                            1000 * trace["support_slip_m_s"][..., side],
                            samples & (trace["support_fraction"][..., side] > 0),
                        )
                        for side in range(2)
                    ],
                    "relative_foot_position_mm_lr": (
                        1000
                        * np.mean(
                            (
                                trace["foot_position_m"]
                                - trace["torso_position_m"][..., None, :]
                            )[samples],
                            axis=0,
                        )
                    ).tolist(),
                }
            )
        result["command_groups"].append(entry)
    return result


def run_verification(
    run_dir,
    output,
    seeds,
    *,
    solver_iterations=None,
    project_actor=False,
    summarize_only=False,
):
    from wr2.locomotion.configs import load_training_config
    from wr2.locomotion.rsl_rl_training import (
        _jax_policy_factory,
        load_rsl_actor_parameters,
    )
    from wr2.locomotion.walking_env import WR2WalkingEnv

    if output.exists() and not summarize_only:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if summarize_only:
        previous = json.loads((output / "summary.json").read_text())
        seeds = tuple(previous["seeds"])
        solver_iterations = previous.get("solver_iterations", 1)
        project_actor = previous.get("actor_projection", False)
    config = load_training_config(run_dir / "training_config.yaml")
    commands = (0.05, 0.075, 0.10)
    environment = replace(config.environment, zero_command_probability=0.0)
    env = WR2WalkingEnv(environment, add_observation_noise=False)
    if solver_iterations is not None:
        env.sys = env.sys.tree_replace({"opt.iterations": solver_iterations})
    model = mujoco.MjModel.from_xml_path(str(env.robot.directory / "scene_mjx.xml"))
    source, signs = joint_reflection(model, env.robot.actuator_names)
    active = np.asarray(env._active_indices)
    active_source = jp.asarray([list(active).index(int(source[j])) for j in active])
    source_jax, signs_jax = jp.asarray(source), jp.asarray(signs)
    report = {
        "run_dir": str(run_dir.resolve()),
        "training_config_sha256": hashlib.sha256(
            (run_dir / "training_config.yaml").read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(
            (env.robot.directory / "wr2_mjx.xml").read_bytes()
        ).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "seeds": seeds,
        "commands_m_s": commands,
        "mode": "nominal_model_configured_reset_variation_no_observation_noise",
        "solver_iterations": int(env.sys.opt.iterations),
        "actor_projection": project_actor,
        "model_reference": verify_model_reference(env, model, source, signs),
        "policies": {},
    }
    output.mkdir(parents=True, exist_ok=summarize_only)
    (output / "model_reference.json").write_text(
        json.dumps(report["model_reference"], indent=2) + "\n"
    )
    print("Model/reference verification complete", flush=True)
    keys = jp.stack(
        [jax.random.PRNGKey(seed) for command in commands for seed in seeds]
    )
    command_array = jp.asarray(np.repeat(commands, len(seeds)))
    factory = _jax_policy_factory(config.network.activation)

    def rollout(parameters):
        initial = jax.vmap(env.reset)(keys)
        command = jp.stack(
            [command_array, jp.zeros_like(command_array), jp.zeros_like(command_array)],
            axis=-1,
        )
        policy = factory(parameters)

        def step(carry, index):
            state, alive = carry
            # Force each command in current/newest observations and environment,
            # while letting actual histories develop normally after startup.
            info = {**state.info, "command": command, "reset_command": command}
            observation = {
                name: value.at[:, 2:5].set(command) for name, value in state.obs.items()
            }
            state = state.replace(info=info, obs=observation)
            action, _ = policy(state.obs, None)
            mirrored = mirror_actor(
                state.obs["state"],
                source_jax,
                signs_jax,
                active,
                heading=environment.heading_observation,
            )
            other_action, _ = policy({"state": mirrored}, None)
            expected = action[:, active_source] * signs_jax[active]
            mirror_error = other_action - expected
            if project_actor:
                # Diagnostic only: an exactly mirror-equivariant projection,
                # not a trained policy or a proposed deployment adapter.
                action = 0.5 * (
                    action + other_action[:, active_source] * signs_jax[active]
                )
            replayed, physical = jax.vmap(lambda s, a: replay_control(env, s, a))(
                state, action
            )
            after = jax.vmap(env.step)(state, action)
            data = {
                **physical,
                "valid": alive,
                "command_m_s": command_array,
                "phase_rad": jp.mod(
                    state.info["gait_phase"] + env._gait_phase_increment, 2 * np.pi
                ),
                # World/path-forward base velocity, not a stale body transform.
                # The intended path heading is zero in these fixed-forward runs.
                "vx_m_s": after.pipeline_state.qd[:, 0],
                "action": action,
                "mirror_action_error": mirror_error,
                "joint_position_deg": after.pipeline_state.q[:, env._joint_qpos_indices]
                * 180
                / np.pi,
                "replay_q_error": jp.max(
                    jp.abs(replayed.q - after.pipeline_state.q), axis=-1
                ),
                "fall": after.metrics["fall"],
                "nonfinite": after.metrics["nonfinite_state"],
            }
            alive = alive & (data["fall"] == 0) & (data["nonfinite"] == 0)
            return (after, alive), data

        return jax.lax.scan(
            step,
            (initial, jp.ones(len(keys), dtype=jp.bool_)),
            jp.arange(environment.episode_length),
        )[1]

    compiled = jax.jit(rollout)
    for name, filename in (("starting_best", "best_params.pt"), ("final", "params.pt")):
        checkpoint = run_dir / filename
        print(
            f"Replaying {name}: {len(keys)} episodes with all physics substeps",
            flush=True,
        )
        if summarize_only:
            with np.load(output / f"{name}.npz") as archive:
                trace = dict(archive)
        else:
            trace = jax.tree.map(
                np.asarray, compiled(load_rsl_actor_parameters(checkpoint))
            )
            np.savez_compressed(output / f"{name}.npz", **trace)
        report["policies"][name] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **summarize_rollout(
                trace,
                env.robot.actuator_names,
                active,
                np.rad2deg(environment.policy_action_scale_rad),
            ),
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Finished {name}: mirror error={report['policies'][name]['actor_mirror_error_deg']['rms']:.2f} deg",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute summaries from existing traces without rerunning physics",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        help="Read-only convergence probe; does not change training files",
    )
    parser.add_argument(
        "--project-actor",
        action="store_true",
        help="Diagnostic mirror projection, without training or checkpoint writes",
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be nonempty and unique")
    if args.solver_iterations is not None and args.solver_iterations < 1:
        parser.error("solver iterations must be positive")
    run_verification(
        args.run_dir,
        args.output,
        seeds,
        solver_iterations=args.solver_iterations,
        project_actor=args.project_actor,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    main()
