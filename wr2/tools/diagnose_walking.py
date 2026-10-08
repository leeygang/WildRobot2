"""Matched MJX heading, stance-force/slip, and joint-clipping diagnostics."""

from __future__ import annotations

import argparse
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
    run_dir: Path, output: Path, seeds: tuple[int, ...], *, summarize_only=False
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
    if not summarize_only and (output / "summary.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic run: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = load_training_config(run_dir / "training_config.yaml")
    environment_config = replace(
        config.environment,
        command_forward_range_m_s=(0.10, 0.10),
        zero_command_probability=0.0,
    )
    report = {
        "run": str(run_dir.resolve()),
        "seeds": seeds,
        "command_m_s": 0.10,
        "evaluation_mode": "nominal_dynamics_with_configured_reset_actuator_and_pose_variation",
        "observation_noise": False,
        "force_sampling": "last_physics_substep_not_impulse",
        "heading_injection_s": 3.0,
        "policies": {},
    }
    factory = _jax_policy_factory(config.network.activation)
    for mode in ("heading", "transitions"):
        env = (
            WR2WalkingEnv(environment_config, add_observation_noise=False)
            if mode == "heading"
            else transition_environment(environment_config, add_observation_noise=False)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute summaries from existing traces without simulation",
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    run_diagnostics(
        args.run_dir, args.output, seeds, summarize_only=args.summarize_only
    )


if __name__ == "__main__":
    main()
