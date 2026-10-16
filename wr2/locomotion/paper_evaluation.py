"""Per-episode WR2 comparison using the completed TB control's 15 conditions.

Selection draws are fixed at seed 12102; confirmation uses new draws (22102).
Native rewards and mixed episode errors are not checkpoint selection metrics.
"""

import copy
from dataclasses import replace

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper

from wr2.locomotion.walking_env import _inverse_rotate_vector


CONDITIONS = (
    ("forward_010", (0.10, 0.0), False),
    ("backward_010", (-0.10, 0.0), False),
    ("backward_005", (-0.05, 0.0), False),
    ("forward_005", (0.05, 0.0), False),
    ("right_005", (0.0, -0.05), False),
    ("left_005", (0.0, 0.05), False),
    ("backward_right", (-0.07071067811865475, -0.035355339059327376), False),
    ("backward_left", (-0.07071067811865475, 0.035355339059327376), False),
    ("forward_right", (0.07071067811865475, -0.035355339059327376), False),
    ("forward_left", (0.07071067811865475, 0.035355339059327376), False),
    ("standing", (0.0, 0.0), False),
    ("forward_transitions", (0.10, 0.0), True),
    ("backward_transitions", (-0.10, 0.0), True),
    ("right_transitions", (0.0, -0.05), True),
    ("left_transitions", (0.0, 0.05), True),
)
GROUPS = ("all", "standing", "walking", "start", "stop")
MEASUREMENTS = (
    "velocity_x_m_s",
    "velocity_y_m_s",
    "xy_error_m_s",
    "xy_speed_m_s",
    "leg_motor_rmse_rad",
    "torque_rms_nm",
    "mean_step_peak_torque_nm",
    "target_clipping_fraction",
    "target_slew_fraction",
    "bounded_action_saturation_fraction",
    "left_contact_fraction",
    "right_contact_fraction",
    "double_support_fraction",
    "left_foot_xy_path_m",
    "right_foot_xy_path_m",
)


def command_and_groups(step, target, transitions, scale=1.0):
    """TB's normalized 3/6/3/5/3 second schedule and 0.5 second recovery."""
    a, b, c, d = (round(t * scale) for t in (150, 450, 600, 850))
    window = round(25 * scale)
    moving = jp.linalg.norm(target) > 1e-6
    scheduled = ((step >= a) & (step < b)) | ((step >= c) & (step < d))
    walking = moving & jp.where(transitions, scheduled, True)
    start = (
        transitions
        & walking
        & (((step >= a) & (step < a + window)) | ((step >= c) & (step < c + window)))
    )
    stop = (
        transitions
        & moving
        & ~walking
        & (((step >= b) & (step < b + window)) | ((step >= d) & (step < d + window)))
    )
    command = jp.concatenate([jp.where(walking, target, jp.zeros(2)), jp.zeros(1)])
    return command, jp.asarray([True, ~walking, walking, start, stop])


def set_command(state, command, scale):
    """Keep native noisy history and replace the current frame's command."""
    observed = command * jp.asarray([1 / scale, 1 / scale, scale])
    return state.replace(
        info={**state.info, "command": command},
        obs={
            name: value.at[..., 2:5].set(observed) for name, value in state.obs.items()
        },
    )


def summarize(sums, counts):
    """Average within episodes first; RMSE is sqrt of each episode's mean MSE."""
    groups, per_episode = {}, {}
    for group_index, name in enumerate(GROUPS):
        count = counts[:, group_index]
        eligible = count > 0
        mean = np.divide(
            sums[:, group_index],
            count[:, None],
            out=np.zeros_like(sums[:, group_index]),
            where=count[:, None] > 0,
        )
        mean[:, 4] = np.sqrt(np.maximum(mean[:, 4], 0))
        mean[:, 5] = np.sqrt(np.maximum(mean[:, 5], 0))
        mean[:, 13:15] = sums[:, group_index, 13:15]
        groups[name] = {
            "sample_count": int(count.sum()),
            "episode_count": int(eligible.sum()),
        }
        per_episode[name] = {"sample_count": count.tolist()}
        for index, measurement in enumerate(MEASUREMENTS):
            values = mean[eligible, index]
            groups[name][measurement] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std()) if values.size else None,
            }
            per_episode[name][measurement] = [
                float(v) if valid else None
                for v, valid in zip(mean[:, index], eligible, strict=True)
            ]
    return groups, per_episode


class CommandVmapWrapper(DomainRandomizationVmapWrapper):
    def reset(self, rng, command):
        def reset(sys, key, cmd):
            with self.v_env_fn(sys) as env:
                return env.reset(key, cmd)

        return jax.vmap(reset, in_axes=[self._in_axes, 0, 0])(self._sys_v, rng, command)


class PaperEvaluator:
    """Stop on each terminal/invalid transition; retain finite terminal samples."""

    def __init__(
        self,
        env,
        make_policy,
        num_envs,
        randomization_fn=None,
        seed=12102,
        *,
        engineering_horizon=False,
    ):
        env = copy.copy(env)
        self.scale = env.config.normalization_scale
        self.schedule_scale = 1.0 if engineering_horizon else self.scale
        self.episode_length = round(1000 * self.schedule_scale)
        self.num_envs, self.seed = num_envs, seed
        self.randomized = randomization_fn is not None
        self.noisy = env.add_observation_noise
        env.config = replace(
            env.config,
            command_resample_steps=self.episode_length + 1,
            randomization=replace(env.config.randomization, enabled=self.randomized),
        )
        self.make_policy = make_policy
        self.key = jax.random.PRNGKey(seed)
        model_key, reset_key = jax.random.split(self.key)
        self.reset_keys = jax.random.split(reset_key, num_envs)

        def randomize(sys):
            if randomization_fn is None:
                return sys, jax.tree.map(lambda _: None, sys)
            return randomization_fn(sys, rng=jax.random.split(model_key, num_envs))

        self.env = CommandVmapWrapper(env, randomize)
        self.root = env._root_link_index
        self.com_height = env.zmp_reference_diagnostics.com_height_m
        self._run = jax.jit(self._rollout)

    def _rollout(self, params, target, transitions):
        policy = self.make_policy(params[:2], deterministic=True)
        initial, _ = command_and_groups(0, target, transitions, self.schedule_scale)
        state = self.env.reset(
            self.reset_keys, jp.broadcast_to(initial, (self.num_envs, 3))
        )
        alive = jp.ones(self.num_envs, dtype=jp.bool_)
        lengths = jp.zeros(self.num_envs, dtype=jp.int32)
        falls, nonfinite = jp.zeros_like(alive), jp.zeros_like(alive)
        counts = jp.zeros((self.num_envs, len(GROUPS)), dtype=jp.int32)
        sums = jp.zeros((self.num_envs, len(GROUPS), len(MEASUREMENTS)))

        def step(carry, index):
            state, alive, lengths, falls, nonfinite, counts, sums = carry
            command, groups = command_and_groups(
                index, target, transitions, self.schedule_scale
            )
            commands = jp.broadcast_to(command, (self.num_envs, 3))
            state = set_command(state, commands, self.scale)
            action, _ = policy(state.obs, self.key)
            next_state = self.env.step(state, action)
            p, m = next_state.pipeline_state, next_state.metrics
            velocity = jax.vmap(_inverse_rotate_vector)(
                p.x.rot[:, self.root], p.xd.vel[:, self.root]
            )
            values = jp.stack(
                [
                    velocity[:, 0],
                    velocity[:, 1],
                    jp.linalg.norm(velocity[:, :2] - commands[:, :2], axis=1),
                    jp.linalg.norm(velocity[:, :2], axis=1),
                    m["leg_motor_mse_rad2_per_step"],
                    m["actuator_torque_rms_nm_per_step"] ** 2,
                    m["actuator_torque_peak_nm_per_step"],
                    m["target_clipping_fraction_per_step"],
                    m["target_slew_limiting_fraction_per_step"],
                    m["bounded_action_saturation_fraction_per_step"],
                    m["left_foot_contact_per_step"],
                    m["right_foot_contact_per_step"],
                    m["double_support_per_step"],
                    m["left_foot_displacement_m_per_step"],
                    m["right_foot_displacement_m_per_step"],
                ],
                axis=1,
            )
            finite = (
                jp.isfinite(p.q).all(axis=1)
                & jp.isfinite(p.qd).all(axis=1)
                & jp.isfinite(next_state.reward)
                & jp.isfinite(action).all(axis=1)
                & jp.isfinite(values).all(axis=1)
            )
            for obs in next_state.obs.values():
                finite &= jp.isfinite(obs).all(axis=1)
            active = alive & finite
            mask = active[:, None] & groups[None, :]
            counts += mask.astype(jp.int32)
            sums += jp.where(mask[:, :, None], values[:, None, :], 0.0)
            lengths += alive.astype(jp.int32)
            falls |= alive & finite & (m["fall"] > 0)
            nonfinite |= alive & ~finite
            next_state = jax.tree.map(
                lambda new, old: jp.where(
                    active.reshape((self.num_envs,) + (1,) * (new.ndim - 1)), new, old
                ),
                next_state,
                state,
            )
            alive &= finite & (next_state.done == 0)
            return (next_state, alive, lengths, falls, nonfinite, counts, sums), None

        result, _ = jax.lax.scan(
            step,
            (state, alive, lengths, falls, nonfinite, counts, sums),
            jp.arange(self.episode_length),
        )
        return result[2:]

    def evaluate(self, params):
        conditions = []
        for name, target, transitions in CONDITIONS:
            physical_target = np.asarray(target) * self.scale
            lengths, falls, nonfinite, counts, sums = jax.device_get(
                self._run(params, jp.asarray(physical_target), jp.asarray(transitions))
            )
            groups, per_episode = summarize(sums, counts)
            conditions.append(
                {
                    "name": name,
                    "command_xy_m_s": physical_target.tolist(),
                    "transitions": transitions,
                    "episodes": self.num_envs,
                    "mean_episode_length": float(lengths.mean()),
                    "fall_count": int(falls.sum()),
                    "nonfinite_count": int(nonfinite.sum()),
                    "groups": groups,
                    "per_episode": {
                        "length": lengths.tolist(),
                        "fell": falls.tolist(),
                        "nonfinite": nonfinite.tolist(),
                        **per_episode,
                    },
                }
            )
        return {
            "schema_version": 1,
            "seed": self.seed,
            "normalization_scale": self.scale,
            "com_height_m": self.com_height,
            "froude_velocity_m_s": float(np.sqrt(9.81 * self.com_height)),
            "episode_length": self.episode_length,
            "control_period_s": 0.02,
            "profile": "engineering_20s"
            if self.schedule_scale == 1.0
            else "TB_normalized",
            "deterministic_policy": True,
            "model_randomization": self.randomized,
            "reset_and_local_actuator_randomization": self.randomized,
            "push_interval_steps": self.env.config.push_interval_steps
            if self.randomized
            else 0,
            "observation_noise": self.noisy,
            "conditions": conditions,
        }


def selection_summary(report):
    """Finite, <=1% falls; minimize per-episode walking error within that gate.

    This selects a candidate for held-out confirmation, not a claim of TB parity.
    """
    conditions = report["conditions"]
    episodes = sum(c["episodes"] for c in conditions)
    falls = sum(c["fall_count"] for c in conditions)
    nonfinite = sum(c["nonfinite_count"] for c in conditions)
    errors = [
        v
        for c in conditions
        if not c["transitions"]
        for v in c["per_episode"]["walking"]["xy_error_m_s"]
        if v is not None
    ]
    error = float(np.mean(errors)) if errors else None
    complete = len(conditions) == len(CONDITIONS)
    eligible = (
        complete and nonfinite == 0 and falls / episodes <= 0.01 and error is not None
    )
    return {
        "episodes": episodes,
        "fall_count": falls,
        "fall_fraction": falls / episodes,
        "nonfinite_count": nonfinite,
        "walking_xy_error_m_s": error,
        "walking_xy_error_TB_equivalent_m_s": None
        if error is None
        else error / report["normalization_scale"],
        "walking_xy_error_dimensionless": None
        if error is None
        else error / report["froude_velocity_m_s"],
        "tracking_scope": "ten_fixed_nonzero_conditions_per_episode",
        "eligible": eligible,
    }
