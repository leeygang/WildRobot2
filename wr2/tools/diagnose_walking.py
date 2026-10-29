"""Matched MJX heading, Paper stance, force/slip and clipping diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wr2.locomotion.train import (
    _configure_backend_logging,
    _filter_optional_backend_import_messages,
)


def contact_point_velocity(linear, angular, point, origin):
    """World velocity at a contact, including foot rotation about its origin."""
    import jax.numpy as jp

    return linear + jp.cross(angular, point - origin)


def foot_contact_samples(env, pipeline):
    """Sample forces ON each foot and support-point slip at the control boundary.

    MuJoCo's contact force acts on geom2; negate it when the foot is geom1.
    These are sampled forces, not integrated control-period impulses.
    """
    import jax.numpy as jp
    from mujoco.mjx._src import support

    contact = pipeline._impl.contact
    forces = jp.stack(
        [
            support.contact_force(env.sys, pipeline, i, True)[:3]
            for i in range(contact.dist.shape[0])
        ]
    )
    foot_forces, slip_speed, support_counts = [], [], []
    for geom, link in zip(env._foot_geom_ids, env._foot_link_indices, strict=True):
        on_second = (contact.geom2 == geom) & (contact.geom1 == env._floor_geom_id)
        on_first = (contact.geom1 == geom) & (contact.geom2 == env._floor_geom_id)
        sign = on_second.astype(jp.float32) - on_first.astype(jp.float32)
        on_foot = forces * sign[:, None]
        active = (contact.efc_address >= 0) & (sign != 0)
        supporting = active & (on_foot[:, 2] > env.config.contact_force_threshold_n)
        foot_forces.append(jp.sum(jp.where(active[:, None], on_foot, 0.0), axis=0))
        velocity = contact_point_velocity(
            pipeline.xd.vel[link],
            pipeline.xd.ang[link],
            contact.pos,
            pipeline.x.pos[link],
        )
        count = jp.sum(supporting)
        slip_speed.append(
            jp.sum(jp.where(supporting, jp.linalg.norm(velocity[:, :2], axis=1), 0.0))
            / jp.maximum(count, 1)
        )
        support_counts.append(count)
    return jp.stack(foot_forces), jp.stack(slip_speed), jp.stack(support_counts)


def rotate_heading(env, state, angle):
    """Rotate the physical robot, not its reference, then refresh the newest frame.

    Free-joint translation velocity is world-frame; angular velocity is local.
    Reinitialize all cases, including the zero-angle control, identically.
    Older observation frames and command/controller/IMU memory are retained.
    """
    import jax.numpy as jp
    from wr2.locomotion.walking_env import _quaternion_multiply, _rotate_vector

    yaw = jp.asarray([jp.cos(angle / 2), 0.0, 0.0, jp.sin(angle / 2)])
    q = state.pipeline_state.q.at[3:7].set(
        _quaternion_multiply(yaw, state.pipeline_state.q[3:7])
    )
    qd = state.pipeline_state.qd.at[:3].set(
        _rotate_vector(yaw, state.pipeline_state.qd[:3])
    )
    pipeline = env.pipeline_init(
        q, qd, act=state.pipeline_state.act, ctrl=state.pipeline_state.ctrl
    ).replace(time=state.pipeline_state.time)
    info = state.info
    contact = env._foot_contact(pipeline)
    from wr2.locomotion.walking_env import expected_foot_contacts

    desired = expected_foot_contacts(
        info["gait_phase"],
        env._is_walking(info["command"]),
        env.config.zmp_reference.single_double_support_ratio,
    )
    frames, _ = env._observation(
        pipeline,
        info["previous_action"],
        info["command"],
        info["gait_phase"],
        info["rng"],
        info["imu_state"],
        info["backlash"],
        contact,
        desired,
        reference_heading=info["reference_heading"],
    )
    observation = {
        name: state.obs[name].at[: frame.shape[0]].set(frame)
        for name, frame in frames.items()
    }
    return state.replace(pipeline_state=pipeline, obs=observation)


def target_trace(env, before, action):
    """Reconstruct the exact delayed, biased, bounded and slew-limited target."""
    import jax.numpy as jp

    applied = (
        before.info["previous_action"] if env.config.action_delay_steps else action
    )
    requested = env._policy_target(applied) + before.info["actuator_target_bias"]
    bounded = jp.clip(requested, env._ctrl_lower, env._ctrl_upper)
    rounded = env._quantize_target(bounded)
    quantized = jp.clip(rounded, env._ctrl_lower, env._ctrl_upper)
    delta = quantized - before.info["previous_target"]
    target = before.info["previous_target"] + jp.clip(
        delta, -env._max_command_step_rad, env._max_command_step_rad
    )
    lower = (requested < env._ctrl_lower - 1e-7) | (rounded < env._ctrl_lower - 1e-7)
    upper = (requested > env._ctrl_upper + 1e-7) | (rounded > env._ctrl_upper + 1e-7)
    return (
        requested,
        target,
        lower,
        upper,
        jp.abs(delta) > env._max_command_step_rad + 1e-7,
    )


def sample_step(env, before, after, action):
    """Trace the delayed request, quantization, limits and applied servo target."""
    import jax.numpy as jp
    from wr2.locomotion.walking_env import torso_heading

    requested, target, lower, upper, slew = target_trace(env, before, action)
    force, slip, counts = foot_contact_samples(env, after.pipeline_state)
    heading = before.info["reference_heading"]
    c, s = jp.cos(heading), jp.sin(heading)
    path_force = force.at[:, 0].set(c * force[:, 0] + s * force[:, 1])
    path_force = path_force.at[:, 1].set(-s * force[:, 0] + c * force[:, 1])
    velocity = env._kinematic_observation(after.pipeline_state)[2]
    error = torso_heading(after.pipeline_state.x.rot[env._root_link_index]) - heading
    metrics = after.metrics
    return {
        "position_m": after.pipeline_state.x.pos[env._root_link_index],
        "heading_error_deg": jp.arctan2(jp.sin(error), jp.cos(error)) * 180 / jp.pi,
        "velocity_torso_m_s": velocity,
        "command_m_s": before.info["command"][0],
        "phase_rad": jp.mod(
            before.info["gait_phase"] + env._gait_phase_increment, 2 * jp.pi
        ),
        "command_age_steps": before.info["command_age_steps"],
        "foot_force_path_n": path_force,
        "support_point_slip_m_s": slip,
        "support_point_count": counts,
        "requested_target_deg": requested * 180 / jp.pi,
        "applied_target_deg": target * 180 / jp.pi,
        "joint_position_deg": after.pipeline_state.q[env._joint_qpos_indices]
        * 180
        / jp.pi,
        "clip_lower": lower,
        "clip_upper": upper,
        "slew_limited": slew,
        "fall": metrics["fall"],
        "nonfinite": metrics["nonfinite_state"],
        "torque_rms_nm": metrics["actuator_torque_rms_nm_per_step"],
    }


def summarize(trace, names, dt, *, injection_step=150, active_indices=None):
    """Keep conditional denominators and per-run recovery/termination explicit."""
    valid = trace["valid"].astype(bool)
    walking = trace["command_m_s"] >= 0.05
    steps = np.arange(valid.shape[0])[:, None]
    steady = (
        valid
        & walking
        & (steps >= injection_step + 50)
        & (trace["command_age_steps"] >= 25)
    )

    def mean(value, mask):
        return float(np.mean(value[mask])) if np.any(mask) else None

    result = {
        "fall_rate": float(np.mean(np.any((trace["fall"] > 0) & valid, axis=0))),
        "nonfinite_rate": float(
            np.mean(np.any((trace["nonfinite"] > 0) & valid, axis=0))
        ),
        "mean_episode_length": float(np.mean(np.sum(valid, axis=0))),
        "steady_speed_mae_m_s": mean(
            np.abs(trace["velocity_torso_m_s"][:, :, 0] - trace["command_m_s"]), steady
        ),
        "steady_heading_mae_deg": mean(np.abs(trace["heading_error_deg"]), steady),
        "torque_rms_nm": mean(trace["torque_rms_nm"], valid),
    }
    # Recovery is held within 2 degrees for one full gait (~0.72s), not one sample.
    recovery, paths = [], []
    for run in range(valid.shape[1]):
        good = valid[:, run] & (np.abs(trace["heading_error_deg"][:, run]) <= 2)
        recovered = None
        for step in range(injection_step, valid.shape[0] - 35):
            if good[step : step + 36].all():
                recovered = (step - injection_step) * dt
                break
        recovery.append(recovered)
        last = np.flatnonzero(valid[:, run])[-1]
        paths.append(
            (
                trace["position_m"][last, run, :2]
                - trace["position_m"][injection_step - 1, run, :2]
            ).tolist()
            if last >= injection_step
            else None
        )
    result["heading_recovery_s_per_run"] = recovery
    result["final_3s_signed_heading_deg_per_run"] = [
        mean(
            trace["heading_error_deg"][:, run],
            valid[:, run] & (steps[:, 0] >= valid.shape[0] - 150),
        )
        for run in range(valid.shape[1])
    ]
    result["post_injection_path_xy_m_per_run"] = paths
    result["phase"] = []
    for quarter in range(4):
        mask = steady & (np.floor(trace["phase_rad"] * 2 / np.pi) == quarter)
        entry = {
            "quarter": quarter,
            "samples": int(mask.sum()),
            "vx_m_s": mean(trace["velocity_torso_m_s"][:, :, 0], mask),
        }
        for side, index in (("left", 0), ("right", 1)):
            support = mask & (trace["support_point_count"][:, :, index] > 0)
            entry[side] = {
                "support_samples": int(support.sum()),
                "mean_force_xyz_n": [
                    mean(trace["foot_force_path_n"][:, :, index, axis], mask)
                    for axis in range(3)
                ],
                "support_positive_forward_force_n": mean(
                    np.maximum(trace["foot_force_path_n"][:, :, index, 0], 0), support
                ),
                "support_braking_force_n": mean(
                    np.maximum(-trace["foot_force_path_n"][:, :, index, 0], 0), support
                ),
                "support_slip_m_s": mean(
                    trace["support_point_slip_m_s"][:, :, index], support
                ),
            }
        result["phase"].append(entry)
    result["clipping"] = {}
    categories = {
        "standing": valid & ~walking,
        "walking": valid & walking,
        "start_0p5s": valid & walking & (trace["command_age_steps"] < 25),
        "stop_0p5s": valid & ~walking & (trace["command_age_steps"] < 25),
    }
    for category, mask in categories.items():
        joints = {}
        for j, name in enumerate(names):
            clip = trace["clip_lower"][:, :, j] | trace["clip_upper"][:, :, j]
            events = np.argwhere(mask & clip)
            if not len(events):
                continue
            step, run = events[0]
            joints[name] = {
                "lower_fraction": mean(trace["clip_lower"][:, :, j], mask),
                "upper_fraction": mean(trace["clip_upper"][:, :, j], mask),
                "first_event": {
                    "time_s": float(step * dt),
                    "run": int(run),
                    "command_age_s": float(trace["command_age_steps"][step, run] * dt),
                    "requested_deg": float(trace["requested_target_deg"][step, run, j]),
                    "applied_deg": float(trace["applied_target_deg"][step, run, j]),
                    "actual_deg": float(trace["joint_position_deg"][step, run, j]),
                },
                "event_time_range_s": [
                    float(events[0, 0] * dt),
                    float(events[-1, 0] * dt),
                ],
                "requested_range_deg": [
                    float(np.min(trace["requested_target_deg"][:, :, j][mask & clip])),
                    float(np.max(trace["requested_target_deg"][:, :, j][mask & clip])),
                ],
            }
        indices = list(range(len(names))) if active_indices is None else active_indices
        clipped = (trace["clip_lower"] | trace["clip_upper"])[:, :, indices]
        result["clipping"][category] = {
            "samples": int(mask.sum()),
            "active_joint_fraction": mean(np.mean(clipped, axis=-1), mask),
            "joints": joints,
        }
    return result


def run_diagnostics(
    run_dir: Path, output: Path, seeds: tuple[int, ...], *, summarize_only=False,
    solver_iterations=None,
):
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        import jax.numpy as jp
        from wr2.locomotion.configs import load_training_config
        from wr2.locomotion.evaluate import transition_environment
        from wr2.locomotion.rsl_rl_training import (
            _jax_policy_factory,
            load_rsl_actor_parameters,
        )
        from wr2.locomotion.walking_env import WR2WalkingEnv

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if summarize_only:
        previous = json.loads((output / "summary.json").read_text())
        solver_iterations = previous.get("solver_iterations", 1)
    if not summarize_only and (output / "summary.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic run: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = load_training_config(run_dir / "training_config.yaml")
    if solver_iterations is not None:
        config = replace(
            config, environment=replace(config.environment, solver_iterations=solver_iterations)
        )
    environment_config = replace(
        config.environment,
        command_forward_range_m_s=(0.10, 0.10),
        zero_command_probability=0.0,
    )
    report = {
        "run": str(run_dir.resolve()),
        "seeds": seeds,
        "command_m_s": 0.10,
        "solver_iterations": config.environment.solver_iterations,
        "evaluation_mode": "nominal_dynamics_with_configured_reset_actuator_and_pose_variation",
        "observation_noise": False,
        "force_sampling": "last_physics_substep_not_impulse",
        "heading_injection_s": 3.0,
        "policies": {},
    }
    factory = _jax_policy_factory(config.network.activation)
    for mode in ("heading", "transitions"):
        env = (
            WR2WalkingEnv(
                environment_config, add_observation_noise=False,
                reward_discounting=config.ppo.discounting,
            )
            if mode == "heading"
            else transition_environment(
                environment_config, add_observation_noise=False,
                reward_discounting=config.ppo.discounting,
            )
        )
        report["joint_limits_deg"] = {
            name: np.rad2deg(
                [env.robot.lower_limit_rad[j], env.robot.upper_limit_rad[j]]
            ).tolist()
            for j, name in enumerate(env.robot.actuator_names)
        }
        angles = (0, -10, -5, 5, 10) if mode == "heading" else (0,)
        keys = jp.stack(
            [jax.random.PRNGKey(seed) for angle in angles for seed in seeds]
        )
        offsets = jp.asarray(np.repeat(np.deg2rad(angles), len(seeds)))

        def rollout(params):
            initial = jax.vmap(env.reset)(keys)
            make_policy = factory(params)

            def step(carry, index):
                state, alive = carry
                if mode == "heading":
                    state = jax.lax.cond(
                        index == 150,
                        lambda s: jax.vmap(lambda v, a: rotate_heading(env, v, a))(
                            s, offsets
                        ),
                        lambda s: s,
                        state,
                    )
                action, _ = make_policy(state.obs, None)
                after = jax.vmap(env.step)(state, action)
                trace = jax.vmap(lambda b, a, u: sample_step(env, b, a, u))(
                    state, after, action
                )
                trace["valid"] = alive
                alive = (
                    alive
                    & (after.metrics["fall"] == 0)
                    & (after.metrics["nonfinite_state"] == 0)
                )
                # Freeze terminated runs; never count autoreset episodes as recovery.
                state = jax.tree.map(
                    lambda old, new: jp.where(
                        alive.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
                    ),
                    state,
                    after,
                )
                return (state, alive), trace

            return jax.lax.scan(
                step, (initial, jp.ones(keys.shape[0], dtype=bool)), jp.arange(1000)
            )[1]

        compiled = jax.jit(rollout)
        for label, filename in (
            ("starting_best", "best_params.pt"),
            ("final", "params.pt"),
        ):
            print(
                f"Diagnostic {mode}: {label}, {len(keys)} matched episodes", flush=True
            )
            trace_path = output / f"{label}_{mode}.npz"
            if summarize_only:
                with np.load(trace_path) as saved:
                    trace = {name: saved[name] for name in saved.files}
            else:
                params = load_rsl_actor_parameters(run_dir / filename)
                trace = jax.tree.map(np.asarray, compiled(params))
                np.savez_compressed(trace_path, **trace)
            mode_results = {}
            for index, angle in enumerate(angles):
                subset = {
                    name: value[:, index * len(seeds) : (index + 1) * len(seeds)]
                    for name, value in trace.items()
                }
                mode_results[str(angle)] = summarize(
                    subset,
                    env.robot.actuator_names,
                    env.robot.control_period_s,
                    active_indices=np.asarray(env._active_indices),
                )
                print(
                    json.dumps(
                        {
                            "policy": label,
                            "mode": mode,
                            "angle_deg": angle,
                            **{
                                k: v
                                for k, v in mode_results[str(angle)].items()
                                if k not in ("phase", "clipping")
                            },
                        }
                    ),
                    flush=True,
                )
            if mode == "heading":
                zero = mode_results["0"]["final_3s_signed_heading_deg_per_run"]
                for metrics in mode_results.values():
                    tail = metrics["final_3s_signed_heading_deg_per_run"]
                    metrics["final_3s_heading_delta_vs_zero_deg_per_run"] = [
                        value - baseline
                        if value is not None and baseline is not None
                        else None
                        for value, baseline in zip(tail, zero, strict=True)
                    ]
            report["policies"].setdefault(label, {})[mode] = mode_results
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote traces and summary to {output}", flush=True)
    return report


def reference_action(env, action, reference, mask):
    """Replace selected action coordinates through the normal target pipeline."""
    import jax.numpy as jp

    active = env._active_indices
    requested = (
        reference[active] - env._policy_target(jp.zeros_like(action))[active]
    ) / env.config.policy_action_scale_rad
    return jp.where(mask, jp.clip(requested, -1.0, 1.0), action)


def signed_foot_distances(model, geom_ids, centers, rotations, valid):
    """Use the traced geometry frames, without re-forwarding integrated qpos.

    Solver/contact frames and integrated qpos need not represent the same
    physics instant. MuJoCo computes distance from these cached geom frames.
    """
    import mujoco

    data = mujoco.MjData(model)
    distances = np.zeros(len(valid))
    for index in np.flatnonzero(valid):
        data.geom_xpos[geom_ids] = centers[index]
        data.geom_xmat[geom_ids] = rotations[index].reshape(2, 9)
        distances[index] = mujoco.mj_geomDistance(model, data, *geom_ids, 1.0, None)
    return distances


def summarize_foot_planting(trace, names, scale, dt):
    """Describe sampled contacts after 1.5 TB seconds, not qualified strides.

    Exclude load intervals that began before warmup or end at the recording
    boundary. Placement compares the same foot's cached collider center at
    successive contact onsets, projected onto the nominal world command direction.
    """
    valid = trace["valid"].astype(bool)
    steady = valid & (np.arange(len(valid)) * dt / scale >= 1.5)
    duration = int(steady.sum()) * dt / scale
    result = {"warmup_TB_s": 1.5, "duration_TB_s": duration, "feet": {}}
    for foot, side in enumerate(("left", "right")):
        loaded = trace["contact"][:, foot].astype(bool) & valid
        onsets = np.flatnonzero(loaded & ~np.r_[False, loaded[:-1]])
        onsets = onsets[steady[onsets]]
        intervals = []
        for start in onsets:
            end = start
            while end < len(valid) and loaded[end]:
                end += 1
            if end < len(valid) and valid[end]:
                intervals.append((end - start) * dt / scale)
        progress = None
        if len(onsets) > 1:
            command = trace["command_m_s"][onsets[0], :2].astype(float)
            speed = np.linalg.norm(command)
            if speed > 1e-6:
                centers = trace["foot_geom_center_m"][onsets, foot, :2]
                advances = np.diff(centers.astype(float), axis=0) @ (command / speed)
                progress = float(np.median(advances) / scale**2 * 1000)
        result["feet"][side] = {
            "contact_fraction": float(loaded[steady].mean()) if duration else None,
            "contact_onsets": len(onsets),
            "contact_onset_rate_TB_hz": len(onsets) / duration if duration else None,
            "complete_loaded_intervals": len(intervals),
            "median_loaded_interval_TB_s": float(np.median(intervals))
            if intervals
            else None,
            "median_contact_onset_interval_TB_s": float(
                np.median(np.diff(onsets)) * dt / scale
            )
            if len(onsets) > 1
            else None,
            "median_commanded_foot_center_advance_TB_mm": progress,
        }
    result["ankle_pitch_requests"] = {}
    for index, name in enumerate(names):
        if not name.endswith("ankle_pitch") or steady.sum() < 20:
            continue
        signal = trace["actor_request_rad"][steady, index].astype(float)
        signal -= signal.mean()
        std = float(signal.std())
        frequency = None
        if std > 1e-6:
            power = np.abs(np.fft.rfft(signal)) ** 2
            power[0] = 0
            frequency = float(np.fft.rfftfreq(len(signal), dt)[power.argmax()])
        result["ankle_pitch_requests"][name] = {
            "std_rad": std,
            "dominant_frequency_physical_hz": frequency,
            "dominant_frequency_TB_hz": frequency * scale
            if frequency is not None
            else None,
        }
    return result

def summarize_stance(trace, names, scale, dt):
    """One censored episode; never pool its padding or count resets as survival."""
    valid = trace["valid"].astype(bool)
    count = int(valid.sum())

    def mean(value):
        return float(np.mean(value[valid])) if count else None

    def rms(value):
        return float(np.sqrt(np.mean(np.square(value[valid])))) if count else None

    requested_double = trace["reference_contact"].all(axis=1) & valid
    actual_double = trace["contact"].all(axis=1)
    error = np.linalg.norm(trace["velocity_m_s"] - trace["command_m_s"][:, :2], axis=1)
    result = {
        "sample_count": count,
        "duration_physical_s": count * dt,
        "duration_TB_equivalent_s": count * dt / scale,
        "fell": bool(trace["fall_event"].any()),
        "nonfinite": bool(trace["nonfinite_event"].any()),
        "velocity_TB_equivalent_m_s": [
            mean(trace["velocity_m_s"][:, j] / scale) for j in range(2)
        ],
        "velocity_std_TB_equivalent_m_s": [
            float(np.std(trace["velocity_m_s"][valid, j] / scale)) if count else None
            for j in range(2)
        ],
        "translation_TB_equivalent_m": (
            trace["path_delta_m"][valid].sum(axis=0) / scale**2
        ).tolist(),
        "net_velocity_TB_equivalent_m_s": [
            mean(trace["path_delta_m"][:, j] / (dt * scale)) for j in range(2)
        ],
        "xy_error_TB_equivalent_m_s": mean(error / scale),
        "single_support_fraction": mean(np.logical_xor(*trace["contact"].T)),
        "double_support_fraction": mean(actual_double),
        "no_support_fraction": mean(~trace["contact"].any(axis=1)),
        "reference_double_support_sample_count": int(requested_double.sum()),
        "reference_double_support_match_fraction": (
            float(actual_double[requested_double].mean())
            if requested_double.any()
            else None
        ),
        "foot_self_contact_fraction": mean(trace["foot_self_force_n"] > 0.1),
        "foot_gap_min_m": float(trace["foot_gap_m"][valid].min()) if count else None,
        "foot_gap_median_m": float(np.median(trace["foot_gap_m"][valid]))
        if count
        else None,
        "maximum_foot_self_force_n": float(trace["foot_self_force_n"][valid].max())
        if count
        else None,
        "target_reconstruction_max_error_rad": float(
            trace["target_reconstruction_error_rad"][valid].max()
        )
        if count
        else None,
        "air_time_credit_s": float(trace["air_time_credit_s"][valid].sum()),
        "leg_actual_reference_rmse_rad": rms(
            trace["actual_rad"] - trace["reference_rad"]
        ),
        "leg_actual_delivered_rmse_rad": rms(
            trace["actual_rad"] - trace["delivered_rad"]
        ),
        "feet_lateral_distance_m": mean(trace["feet_lateral_distance_m"]),
        "feet_distance_reward_per_step": mean(trace["feet_distance_reward_per_step"]),
        "foot_vertical_force_mean_n": [
            mean(trace["foot_vertical_force_n"][:, j]) for j in range(2)
        ],
        "planting": summarize_foot_planting(trace, names, scale, dt),
        "joints": {
            name: {
                "actor_request_reference_rmse_rad": rms(
                    trace["actor_request_rad"][:, j] - trace["reference_rad"][:, j]
                ),
                "current_request_reference_rmse_rad": rms(
                    trace["current_request_rad"][:, j] - trace["reference_rad"][:, j]
                ),
                "delivered_reference_rmse_rad": rms(
                    trace["delivered_rad"][:, j] - trace["reference_rad"][:, j]
                ),
                "actual_delivered_rmse_rad": rms(
                    trace["actual_rad"][:, j] - trace["delivered_rad"][:, j]
                ),
                "target_clip_fraction": mean(trace["target_clip"][:, j]),
                "target_slew_fraction": mean(trace["target_slew"][:, j]),
                "reference_outside_action_range_fraction": mean(
                    trace["reference_outside_action_range"][:, j]
                ),
            }
            for j, name in enumerate(names)
        },
    }
    reward_fields = {
        key.removeprefix("paper_reward_"): value
        for key, value in trace.items()
        if key.startswith("paper_reward_")
    }
    if reward_fields:
        result["reward_components"] = {
            name: float(value[valid].astype(float).sum())
            for name, value in reward_fields.items()
        }
        result["total_reward"] = float(trace["reward"][valid].astype(float).sum())
        component_sum = np.stack(list(reward_fields.values()), axis=1).sum(axis=1)
        result["reward_reconstruction_max_error"] = (
            float(np.abs(component_sum[valid] - trace["reward"][valid]).max())
            if count else None
        )
    for name in (
        "raw_air_time_credit_s", "candidate_air_time_s",
        "qualified_air_time_s", "rejected_air_time_s",
    ):
        if name in trace:
            result[name] = float(trace[name][valid].astype(float).sum())
    return result


def reference_geometry_audit(env, model):
    """Check every nominal training lookup pose, including reference time zero."""
    import mujoco

    reference = env._zmp_reference
    geom_ids = np.asarray(env._foot_geom_ids)
    body_ids = model.geom_bodyid[geom_ids]
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    data = mujoco.MjData(model)
    active = np.asarray(env._active_indices)
    entries = []
    for command, poses, stance in zip(
        reference._host["commands"],
        reference._host["qpos"],
        reference._host["stance"],
        strict=True,
    ):
        gaps, spacing = [], []
        double = stance.all(axis=1).copy()
        double[0] = True
        motors = poses[:, reference._joint_indices][:, active].copy()
        motors[0] = np.asarray(env._home_ctrl)[active]
        for frame, pose in enumerate(poses):
            data.qpos[:] = reference.standing_qpos if frame == 0 else pose
            mujoco.mj_kinematics(model, data)
            gaps.append(mujoco.mj_geomDistance(model, data, *geom_ids, 1.0, None))
            spacing.append(
                abs(
                    (
                        data.xmat[torso].reshape(3, 3).T
                        @ (data.xpos[body_ids[0]] - data.xpos[body_ids[1]])
                    )[1]
                )
            )
        gaps, spacing = np.asarray(gaps), np.asarray(spacing)
        outside = (
            np.abs(motors - np.asarray(env._policy_target(env.home_action))[active])
            > env.config.policy_action_scale_rad + 1e-7
        )
        entries.append(
            {
                "command_TB_equivalent": (
                    command
                    * [
                        1 / env.config.normalization_scale,
                        1 / env.config.normalization_scale,
                        env.config.normalization_scale,
                    ]
                ).tolist(),
                "samples": len(gaps),
                "minimum_foot_gap_m": float(gaps.min()),
                "overlap_fraction": float((gaps < -1e-5).mean()),
                "minimum_double_support_gap_m": float(gaps[double].min())
                if double.any()
                else None,
                "below_spacing_reward_minimum_fraction": float(
                    (spacing < env.config.min_feet_lateral_distance_m).mean()
                ),
                "outside_action_range_fraction_per_joint": outside.mean(
                    axis=0
                ).tolist(),
                "maximum_joint_limit_violation_rad": float(
                    np.maximum(
                        np.maximum(
                            np.asarray(env._ctrl_lower)[active] - motors,
                            motors - np.asarray(env._ctrl_upper)[active],
                        ),
                        0,
                    ).max()
                ),
            }
        )
    return {
        "scope": "all nominal training lookup commands, kinematics only",
        "command_count": len(entries),
        "sample_count": sum(entry["samples"] for entry in entries),
        "minimum_foot_gap_m": min(entry["minimum_foot_gap_m"] for entry in entries),
        "commands_with_overlap": sum(
            entry["overlap_fraction"] > 0 for entry in entries
        ),
        "maximum_joint_limit_violation_rad": max(
            entry["maximum_joint_limit_violation_rad"] for entry in entries
        ),
        "commands": entries,
    }


def run_paper_stance_diagnostics(
    run_dir, output, seeds, *, checkpoint_path=None, allow_cpu=False
):
    """Compare learned, reference hip-roll, and full leg-reference controls."""
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.io import model as brax_model
        from brax.training import types
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import checkpoint, networks
        from wr2.locomotion.configs import load_training_config
        from wr2.locomotion.paper_evaluation import CONDITIONS
        from wr2.locomotion.ppo import make_network_factory
        from wr2.locomotion.train import _training_provenance
        from wr2.locomotion.walking_env import WR2WalkingEnv

    if jax.default_backend() != "gpu" and not allow_cpu:
        raise RuntimeError(
            "Paper stance diagnostics require a GPU; --allow-cpu is for development"
        )
    if (output / "summary.json").exists():
        raise FileExistsError(f"Diagnostic already exists: {output}")
    config = load_training_config(run_dir / "training_config.yaml")
    if config.environment.recipe != "tb_paper":
        raise ValueError("--paper-stance requires a saved Paper training configuration")
    path = (checkpoint_path or run_dir / "best_params").resolve()
    params = (
        checkpoint.load(str(path)) if path.is_dir() else brax_model.load_params(path)
    )
    if any(
        not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(params)
    ):
        raise ValueError("Checkpoint has nonfinite parameters")
    scale = config.environment.normalization_scale
    length = round(1000 * scale)
    nominal = replace(
        config.environment,
        command_resample_steps=length + 1,
        randomization=replace(config.environment.randomization, enabled=False),
    )
    print(
        "Building the recorded Paper reference and checking the full command grid...",
        flush=True,
    )
    env = WR2WalkingEnv(
        nominal, add_observation_noise=False, reward_discounting=config.ppo.discounting
    )
    model = mujoco.MjModel.from_xml_path(str(env.robot.directory / "scene_mjx.xml"))
    geometry = reference_geometry_audit(env, model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_geometry.json").write_text(
        json.dumps(geometry, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in geometry.items() if key != "commands"}
        ),
        flush=True,
    )
    active = np.asarray(env._active_indices)
    names = [env.robot.actuator_names[j] for j in active]
    masks = {
        "learned": np.zeros(len(active), bool),
        "hip_roll_reference": np.asarray(
            [name.endswith("_hip_roll") for name in names]
        ),
        "leg_reference": np.ones(len(active), bool),
    }
    conditions = [
        item
        for item in CONDITIONS
        if item[0]
        in (
            "forward_010",
            "backward_010",
            "forward_005",
            "backward_005",
            "left_005",
            "right_005",
            "standing",
        )
    ]
    cases = [
        (mode, name, xy, seed)
        for mode in masks
        for name, xy, _ in conditions
        for seed in seeds
    ]
    commands = jp.asarray([[*np.asarray(xy) * scale, 0.0] for _, _, xy, _ in cases])
    keys = jp.stack(
        [
            jax.random.split(jax.random.split(jax.random.PRNGKey(seed))[1], 8)[0]
            for _, _, _, seed in cases
        ]
    )
    action_masks = jp.asarray(np.stack([masks[mode] for mode, *_ in cases]))
    preprocess = (
        running_statistics.normalize
        if config.ppo.normalize_observations
        else types.identity_observation_preprocessor
    )
    network = make_network_factory(config.network)(
        env.observation_size, env.action_size, preprocess_observations_fn=preprocess
    )
    policy = networks.make_inference_fn(network)(params[:2], deterministic=True)
    geom_ids = env._foot_geom_ids
    foot_sizes = jp.asarray(model.geom_size[np.asarray(geom_ids)])
    dt = env.robot.control_period_s

    def sample(before, after, actor_action, action, reference, reference_contact):
        requested, delivered, lower, upper, slew = target_trace(env, before, action)
        pipeline, metrics = after.pipeline_state, after.metrics
        forces = env._contact_forces(pipeline)[:, :3]
        pair = (
            (pipeline.contact.geom1 == geom_ids[0])
            & (pipeline.contact.geom2 == geom_ids[1])
        ) | (
            (pipeline.contact.geom1 == geom_ids[1])
            & (pipeline.contact.geom2 == geom_ids[0])
        )
        pair &= (pipeline.contact.dist < 0) & (pipeline._impl.contact.efc_address >= 0)
        delta = pipeline.q[:2] - before.pipeline_state.q[:2]
        heading = before.info["reference_heading"]
        c, s = jp.cos(heading), jp.sin(heading)
        return {
            "qpos": pipeline.q,
            "path_delta_m": jp.asarray(
                [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
            ),
            "velocity_m_s": jp.asarray(
                [
                    metrics["forward_velocity_m_s_per_step"],
                    metrics["lateral_velocity_m_s_per_step"],
                ]
            ),
            "command_m_s": before.info["command"],
            "actor_request_rad": env._policy_target(actor_action)[active],
            "current_request_rad": env._policy_target(action)[active],
            "delayed_request_rad": requested[active],
            "delivered_rad": delivered[active],
            "actual_rad": pipeline.q[env._joint_qpos_indices][active],
            "reference_rad": reference[active],
            "target_clip": (lower | upper)[active],
            "target_slew": slew[active],
            "reference_outside_action_range": jp.abs(
                reference[active] - env._policy_target(env.home_action)[active]
            )
            > env.config.policy_action_scale_rad + 1e-7,
            "target_reconstruction_error_rad": jp.where(
                after.done == 0,
                jp.max(jp.abs(delivered - after.info["previous_target"])),
                0.0,
            ),
            "contact": jp.asarray(
                [
                    metrics["left_foot_contact_per_step"],
                    metrics["right_foot_contact_per_step"],
                ]
            )
            > 0,
            "reference_contact": reference_contact,
            "foot_geom_center_m": pipeline.geom_xpos[geom_ids],
            "foot_geom_rotation": pipeline.geom_xmat[geom_ids],
            "sole_clearance_m": pipeline.geom_xpos[geom_ids, 2]
            - jp.sum(jp.abs(pipeline.geom_xmat[geom_ids, 2, :]) * foot_sizes, axis=1)
            - model.geom_pos[env._floor_geom_id, 2],
            "foot_vertical_force_n": jp.asarray(
                [
                    metrics["left_foot_vertical_force_n_per_step"],
                    metrics["right_foot_vertical_force_n_per_step"],
                ]
            ),
            "feet_lateral_distance_m": metrics["feet_lateral_distance_m_per_step"],
            "feet_distance_reward_per_step": metrics["paper_reward/feet_distance"],
            "torso_height_m": metrics["torso_height_m_per_step"],
            "foot_self_force_n": jp.max(
                jp.where(pair, jp.linalg.norm(forces, axis=1), 0.0)
            ),
            "air_time_credit_s": metrics["air_time_credit_s_per_step"],
            "reward": after.reward,
            **{
                name.removesuffix("_per_step"): metrics[name]
                for name in (
                    "raw_air_time_credit_s_per_step", "candidate_air_time_s_per_step",
                    "qualified_air_time_s_per_step", "rejected_air_time_s_per_step",
                    "left_foot_stance_duration_s_per_step", "right_foot_stance_duration_s_per_step",
                ) if name in metrics
            },
            **{
                "paper_reward_" + name.removeprefix("paper_reward/"): value
                for name, value in metrics.items()
                if name.startswith("paper_reward/")
            },
        }

    def rollout():
        initial = jax.vmap(env.reset)(keys, commands)

        def step(carry, _):
            before, alive = carry
            reference, contact, _ = jax.vmap(env._zmp_reference.get_reference)(
                before.info["step_count"] * dt, commands
            )
            actor_action, _ = policy(before.obs, keys[0])
            action = jax.vmap(lambda a, r, m: reference_action(env, a, r, m))(
                actor_action, reference, action_masks
            )
            after = jax.vmap(env.step)(before, action)
            values = jax.vmap(sample)(
                before, after, actor_action, action, reference, contact
            )
            finite = jp.isfinite(after.pipeline_state.q).all(axis=1) & jp.isfinite(
                after.pipeline_state.qd
            ).all(axis=1)
            for value in (*values.values(), *after.obs.values(), after.reward):
                finite &= jp.isfinite(value).reshape((len(cases), -1)).all(axis=1)
            values["valid"] = alive & finite
            values["fall_event"] = alive & finite & (after.metrics["fall"] > 0)
            values["nonfinite_event"] = alive & (
                ~finite | (after.metrics["nonfinite_state"] > 0)
            )
            following_alive = alive & finite & (after.done == 0)
            following = jax.tree.map(
                lambda old, new: jp.where(
                    following_alive.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
                ),
                before,
                after,
            )
            return (following, following_alive), values

        return jax.lax.scan(
            step, (initial, jp.ones(len(cases), dtype=bool)), jp.arange(length)
        )[1]

    print(f"Running {len(cases)} paired nominal stance episodes...", flush=True)
    traces = jax.tree.map(np.asarray, jax.jit(rollout)())
    results = []
    for index, (mode, name, xy, seed) in enumerate(cases):
        trace = {key: value[:, index] for key, value in traces.items()}
        trace["foot_gap_m"] = signed_foot_distances(
            model,
            np.asarray(geom_ids),
            trace["foot_geom_center_m"],
            trace["foot_geom_rotation"],
            trace["valid"],
        )
        np.savez_compressed(output / f"{mode}_{name}_seed{seed}.npz", **trace)
        summary = summarize_stance(trace, names, scale, dt)
        results.append(
            {
                "control": mode,
                "condition": name,
                "seed": seed,
                "command_TB_equivalent_m_s": list(xy),
                **summary,
            }
        )
        print(
            json.dumps(
                {
                    key: results[-1][key]
                    for key in (
                        "control",
                        "condition",
                        "sample_count",
                        "fell",
                        "net_velocity_TB_equivalent_m_s",
                        "xy_error_TB_equivalent_m_s",
                        "foot_self_contact_fraction",
                    )
                }
            ),
            flush=True,
        )
    report = {
        "schema_version": 4,
        "run": str(run_dir.resolve()),
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else None,
        "evaluation_source": _training_provenance(),
        "training_config_version": config.version,
        "normalization_scale": scale,
        "control_period_s": dt,
        "seeds": list(seeds),
        "scope": "nominal paired controls; prescribed references are diagnostics, not trained policies",
        "domain_randomization": False,
        "observation_noise": False,
        "physical_pushes": False,
        "action_intervention": "current reference mapped into bounded actions; native delay, quantization, limits and slew retained",
        "reference_error_timing": "current reference vs end-of-tick joint positions; delivered targets retain the recorded action delay",
        "geometry_sampling": "cached last-substep geometry/solver frames; not re-forwarded integrated qpos",
        "reward_sampling": "native weighted components after control-dt multiplication; valid terminal tick retained, padding excluded",
        "air_reward": {
            "minimum_stance_requested_s": nominal.air_time_min_stance_s,
            "minimum_stance_TB_s": nominal.air_time_min_stance_s / scale,
            "minimum_contact_samples": env._air_time_min_stance_steps,
            "payment_delay_ticks": max(env._air_time_min_stance_steps - 1, 0),
            "discounting": config.ppo.discounting,
            "payment_discount_compensation": env._air_time_discount_compensation,
            "credit_units": "air_time_credit_s is seconds-equivalent reward credit; qualified_air_time_s is uncompensated physical credit",
        },
        "termination": "finite terminal tick included; nonfinite tick excluded; no autoreset samples",
        "active_joint_names": names,
        "home_position_rad": np.asarray(env._home_ctrl)[active].tolist(),
        "action_center_rad": np.asarray(env._policy_target(env.home_action))[active].tolist(),
        "joint_limits_rad": np.stack(
            [np.asarray(env._ctrl_lower)[active], np.asarray(env._ctrl_upper)[active]],
            axis=1,
        ).tolist(),
        "policy_action_scale_rad": nominal.policy_action_scale_rad,
        "reference_geometry": geometry,
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote stance traces and summary to {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds")
    parser.add_argument(
        "--paper-stance",
        action="store_true",
        help="Paired learned/reference stance controls for a Paper checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Paper checkpoint; defaults to run-dir/best_params",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow development Paper diagnostics on CPU",
    )
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute summaries from existing traces without simulation",
    )
    args = parser.parse_args()
    if args.solver_iterations is not None and args.solver_iterations < 1:
        parser.error("solver iterations must be positive")
    from wr2.locomotion.evaluate import _parse_seeds

    seeds = _parse_seeds(
        args.seeds or ("22102" if args.paper_stance else "101,202,303,404")
    )
    if args.paper_stance:
        if args.summarize_only or args.solver_iterations is not None:
            parser.error(
                "--paper-stance uses the recorded solver and runs fresh diagnostics"
            )
        run_paper_stance_diagnostics(
            args.run_dir,
            args.output,
            seeds,
            checkpoint_path=args.checkpoint,
            allow_cpu=args.allow_cpu,
        )
        return
    run_diagnostics(
        args.run_dir,
        args.output,
        seeds,
        summarize_only=args.summarize_only,
        solver_iterations=args.solver_iterations,
    )


if __name__ == "__main__":
    main()
