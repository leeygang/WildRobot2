"""Compare an isolated left hip change with height/pitch-preserving leg IK.

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


class SagittalCompensator:
    """Keep commanded site height and sole pitch while fixing the hip target.

    Knee and ankle pitch are the two unknowns. All remaining joint positions,
    including the hip, are fixed. Bounds include the actor's reachable range.
    Full foot roll/yaw are measured, rather than assumed constrained by two DOFs.
    """

    def __init__(self, model, names, home, action_scale, height_scale_m):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.actuator(name).trnid[0] for name in names]
        self.qpos_indices = model.jnt_qposadr[joints]
        self.hip = names.index("left_hip_pitch")
        self.ka = np.asarray(
            [names.index("left_knee_pitch"), names.index("left_ankle_pitch")]
        )
        bounds = model.jnt_range[joints]
        self.lower = np.maximum(bounds[:, 0], np.asarray(home) - action_scale)
        self.upper = np.minimum(bounds[:, 1], np.asarray(home) + action_scale)
        self.height_scale_m = height_scale_m
        self.site = model.site("left_foot_center").id
        self.geom = model.geom("left_foot_collision").id
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


def run(run_dir: Path, output: Path):
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
        model, names, home, action_scale, env.zmp_reference_diagnostics.com_height_m
    )
    starts = np.asarray([180, 190, 200])
    horizon, treatment_steps, ramp_steps = 99, 60, 6
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
    cycle = round(nominal.gait_cycle_s / dt)
    cycle_ref = np.asarray(
        jax.vmap(lambda t: env._zmp_reference.get_reference(t, command)[0])(
            (jp.arange(cycle) + cycle) * dt
        )
    )[:, active[solver.hip]]
    wave = (refs[:, :, active[solver.hip]] - cycle_ref.mean()) / np.abs(
        cycle_ref - cycle_ref.mean()
    ).max()
    k = np.arange(horizon)
    u = np.clip(
        np.minimum(k / ramp_steps, (treatment_steps + ramp_steps - k) / ramp_steps),
        0,
        1,
    )
    strength = u * u * (3 - 2 * u)
    requested = home + action_scale * original
    shifted = requested[:, :, solver.hip] - np.deg2rad(3) * strength[:, None]
    shaped = strength[:, None] * wave
    room = np.full_like(shaped, np.inf)
    positive, negative = shaped > 1e-8, shaped < -1e-8
    room[positive] = (
        (home[solver.hip] + action_scale - shifted) / np.where(positive, shaped, 1)
    )[positive]
    room[negative] = (
        (shifted - (home[solver.hip] - action_scale)) / np.where(negative, -shaped, 1)
    )[negative]
    peaks = np.minimum(np.deg2rad(5), room.min(axis=0) * 0.999)
    hip_only = original.copy()
    hip_only[:, :, solver.hip] = (
        shifted + shaped * peaks - home[solver.hip]
    ) / action_scale
    hip_only[strength == 0, :, solver.hip] = original[strength == 0, :, solver.hip]
    coordinated = hip_only.copy()
    errors = []
    print("SOLVE_COORDINATED_TARGETS", np.rad2deg(peaks).tolist(), flush=True)
    for i in range(len(cp)):
        for tick in range(horizon):
            if hip_only[tick, i, solver.hip] == original[tick, i, solver.hip]:
                continue
            hip_target = float(
                home[solver.hip] + action_scale * hip_only[tick, i, solver.hip]
            )
            targets, error = solver.solve(
                before_q[tick, i], requested[tick, i], hip_target
            )
            coordinated[tick, i, solver.ka] = (targets - home[solver.ka]) / action_scale
            errors.append(
                {
                    "checkpoint": labels[cp[i]],
                    "start": int(start[i]),
                    "tick": tick,
                    **error,
                }
            )
    (output / "ik_validation.json").write_text(json.dumps(errors, indent=2) + "\n")
    assert np.abs(hip_only).max() <= 1 and np.abs(coordinated).max() <= 1
    np.testing.assert_array_equal(
        hip_only[:, :, solver.hip], coordinated[:, :, solver.hip]
    )
    other = np.setdiff1d(np.arange(env.action_size), np.r_[solver.hip, solver.ka])
    np.testing.assert_array_equal(coordinated[:, :, other], original[:, :, other])
    modes = ["baseline", "hip_only", "coordinated", "baseline_duplicate"]
    tapes = jp.asarray(
        np.stack([original, hip_only, coordinated, original], axis=2).reshape(
            horizon, -1, env.action_size
        )
    )
    cp_all, start_all = np.repeat(cp, 4), np.repeat(start, 4)
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
                _, delivered, low, high, slew = target_trace(env, b, act)
                forces = env._contact_forces(p)
                pair = ((p.contact.geom1 == feet[0]) & (p.contact.geom2 == feet[1])) | (
                    (p.contact.geom1 == feet[1]) & (p.contact.geom2 == feet[0])
                )
                pair &= (p.contact.dist < 0) & (p._impl.contact.efc_address >= 0)
                return {
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
                    "action": act,
                    "actor_request_rad": home + action_scale * act,
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
        mode = modes[i % 4]
        trace = {k: v[:, i] for k, v in traces.items()}
        valid = trace["valid"]
        n = int(valid.sum())
        assert np.array_equal(valid, np.arange(horizon) < n)
        np.testing.assert_allclose(
            trace["rewards"][valid].sum(axis=1), trace["total_reward"][valid], atol=1e-5
        )
        assert trace["target_reconstruction_error_rad"][valid].max() < 1e-6
        name = f"{labels[cp_all[i]]}_start{start_all[i]}_{mode}"
        np.savez_compressed(output / (name + ".npz"), **trace)
        discount = cfg.ppo.discounting ** np.arange(n)
        g = float(trace["total_reward"][valid].astype(float) @ discount)
        components = trace["rewards"][valid].astype(float).T @ discount
        during = valid & (np.arange(horizon) < 66)
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
        results.append(result)
    for i in range(0, len(results), 4):
        np.testing.assert_array_equal(
            traces["action"][:, i + 1, solver.hip],
            traces["action"][:, i + 2, solver.hip],
        )
        np.testing.assert_allclose(
            traces["delivered_rad"][:, i + 1, solver.hip],
            traces["delivered_rad"][:, i + 2, solver.hip],
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            traces["contact"][:, i], traces["contact"][:, i + 3]
        )
        assert (
            abs(results[i]["discounted_return"] - results[i + 3]["discounted_return"])
            < 1e-4
        )
        for r in results[i : i + 4]:
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
        "compensation": "preserve original requested-pose foot site height and fore-aft sole angle; same hip target",
        "normal_axis": solver.normal_axis,
        "normal_sign": solver.normal_sign,
        "hip_wave_peak_deg": np.rad2deg(peaks).tolist(),
        "horizon": horizon,
        "treatment_steps": treatment_steps,
        "ramp_steps": ramp_steps,
        "gamma": cfg.ppo.discounting,
        "reference_lookup_command": grid[nearest].tolist(),
        "results": results,
    }
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
    args = parser.parse_args()
    run(args.run_dir, args.output)


if __name__ == "__main__":
    main()
