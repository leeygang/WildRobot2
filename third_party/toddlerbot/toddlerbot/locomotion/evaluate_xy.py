"""Deterministic signed-command measurements for the published TB baseline."""

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper

from toddlerbot.utils.math_utils import quat_inv, rotate_vec


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
)


def command_and_groups(step, target, transitions, dt=0.02):
    """3 s stand, 6 s walk, 3 s stand, 5 s walk, 3 s stand; 0.5 s recovery."""
    start1, stop1, start2, stop2 = (int(round(t / dt)) for t in (3, 9, 12, 17))
    window = int(round(0.5 / dt))
    moving = jnp.linalg.norm(target) > 1e-6
    scheduled_walk = ((step >= start1) & (step < stop1)) | (
        (step >= start2) & (step < stop2)
    )
    walking = moving & jnp.where(transitions, scheduled_walk, True)
    start = (
        transitions
        & walking
        & (
            ((step >= start1) & (step < start1 + window))
            | ((step >= start2) & (step < start2 + window))
        )
    )
    stop = (
        transitions
        & moving
        & ~walking
        & (
            ((step >= stop1) & (step < stop1 + window))
            | ((step >= stop2) & (step < stop2 + window))
        )
    )
    command = jnp.zeros(8).at[5:7].set(jnp.where(walking, target, 0.0))
    return command, jnp.asarray([True, ~walking, walking, start, stop])


def set_command(state, command):
    """Update only the newest frame's command, preserving native noisy history."""
    info = {**state.info, "command": command, "command_obs": command[..., 5:8]}
    return state.replace(
        info=info,
        obs=state.obs.at[..., 2:5].set(command[..., 5:8]),
        privileged_obs=state.privileged_obs.at[..., 2:5].set(command[..., 5:8]),
    )


class CommandVmapWrapper(DomainRandomizationVmapWrapper):
    def reset(self, rng, command):
        def reset(sys, embedding, key, cmd):
            return self._env_fn(sys, embedding).reset(key, cmd)

        return jax.vmap(reset, in_axes=[self._in_axes, 0, 0, 0])(
            self._sys_v, self._embeddings, rng, command
        )


def summarize(sums, counts):
    """Normalize within each episode before comparing episodes of unequal length."""
    groups = {}
    per_episode = {}
    for group_index, name in enumerate(GROUPS):
        count = counts[:, group_index]
        eligible = count > 0
        mean = np.divide(
            sums[:, group_index],
            count[:, None],
            out=np.zeros_like(sums[:, group_index]),
            where=count[:, None] > 0,
        )
        mean[:, -1] = np.sqrt(np.maximum(mean[:, -1], 0.0))
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
                float(value) if valid else None
                for value, valid in zip(mean[:, index], eligible)
            ]
    return groups, per_episode


class XYEvaluator:
    def __init__(
        self,
        env,
        make_policy,
        num_envs,
        episode_length,
        randomization_fn=None,
        seed=12102,
    ):
        # Brax's wrapper assigns sys while tracing. Do not share that mutable
        # environment object with PPO's training or random-command evaluator.
        env = copy.copy(env)
        env.resample_steps = episode_length + 1
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.dt = float(env.dt)
        self.make_policy = make_policy
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        model_key, reset_key = jax.random.split(self.key)
        self.reset_keys = jax.random.split(reset_key, num_envs)
        if randomization_fn is None:

            def randomization_fn(sys, rng):
                return sys, jax.tree.map(lambda _: None, sys), jnp.zeros((num_envs, 0))

        self.env = CommandVmapWrapper(
            env,
            lambda sys: randomization_fn(
                sys, rng=jax.random.split(model_key, num_envs)
            ),
        )
        self.motor_indices = env.q_start_idx + env.leg_motor_indices
        self.ref_indices = env.ref_start_idx + env.leg_ref_indices
        self._run = jax.jit(self._rollout)

    def _rollout(self, params, target, transitions):
        policy = self.make_policy(params, deterministic=True)
        command, _ = command_and_groups(0, target, transitions, self.dt)
        command = jnp.broadcast_to(command, (self.num_envs, 8))
        state = self.env.reset(self.reset_keys, command)
        alive = jnp.ones(self.num_envs, dtype=bool)
        lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)
        falls = jnp.zeros(self.num_envs, dtype=bool)
        nonfinite = jnp.zeros(self.num_envs, dtype=bool)
        counts = jnp.zeros((self.num_envs, len(GROUPS)), dtype=jnp.int32)
        sums = jnp.zeros((self.num_envs, len(GROUPS), len(MEASUREMENTS)))

        def step(carry, index):
            state, alive, lengths, falls, nonfinite, counts, sums = carry
            command, groups = command_and_groups(index, target, transitions, self.dt)
            command = jnp.broadcast_to(command, (self.num_envs, 8))
            state = set_command(state, command)
            action, _ = policy(state.obs, self.key)
            next_state = self.env.step(state, action)
            p = next_state.pipeline_state
            velocity = jax.vmap(rotate_vec)(
                p.xd.vel[:, 0], jax.vmap(quat_inv)(p.x.rot[:, 0])
            )
            motor_error = (
                p.q[:, self.motor_indices]
                - next_state.info["state_ref"][:, self.ref_indices]
            )
            values = jnp.stack(
                [
                    velocity[:, 0],
                    velocity[:, 1],
                    jnp.linalg.norm(velocity[:, :2] - command[:, 5:7], axis=1),
                    jnp.linalg.norm(velocity[:, :2], axis=1),
                    jnp.mean(jnp.square(motor_error), axis=1),
                ],
                axis=1,
            )
            finite = (
                jnp.isfinite(p.q).all(axis=1)
                & jnp.isfinite(p.qd).all(axis=1)
                & jnp.isfinite(next_state.obs).all(axis=1)
                & jnp.isfinite(next_state.privileged_obs).all(axis=1)
                & jnp.isfinite(next_state.reward)
                & jnp.isfinite(action).all(axis=1)
                & jnp.isfinite(values).all(axis=1)
            )
            active = alive & finite
            mask = active[:, None] & groups[None, :]
            counts += mask.astype(jnp.int32)
            sums += jnp.where(mask[:, :, None], values[:, None, :], 0.0)
            lengths += alive.astype(jnp.int32)
            falls |= alive & finite & (next_state.done > 0)
            nonfinite |= alive & ~finite
            # Include the terminal finite sample, then stop that episode.
            next_state = jax.tree.map(
                lambda new, old: jnp.where(
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
            jnp.arange(self.episode_length),
        )
        return result[2:]

    def evaluate(self, params: Any):
        conditions = []
        for name, target, transitions in CONDITIONS:
            lengths, falls, nonfinite, counts, sums = jax.device_get(
                self._run(params, jnp.asarray(target), jnp.asarray(transitions))
            )
            groups, per_episode = summarize(sums, counts)
            conditions.append(
                {
                    "name": name,
                    "command_xy_m_s": list(target),
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
            "seed": self.seed,
            "episode_length": self.episode_length,
            "control_period_s": self.dt,
            "deterministic_policy": True,
            "conditions": conditions,
        }
