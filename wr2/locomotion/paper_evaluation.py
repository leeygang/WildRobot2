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

from wr2.locomotion.walking_env import _inverse_rotate_vector, torso_heading


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
    "single_support_fraction",
    "no_foot_support_fraction",
    "left_foot_vertical_force_n",
    "right_foot_vertical_force_n",
    "reference_double_support_fraction",
    "reference_double_support_match_fraction",
    "foot_self_contact_fraction",
    "air_time_credit_s",
    "contact_phase_match_fraction",
    "reference_velocity_x_m_s",
    "reference_velocity_y_m_s",
    "reference_xy_error_m_s",
    "heading_error_abs_rad",
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
    reference_index = MEASUREMENTS.index("reference_double_support_fraction")
    match_index = MEASUREMENTS.index("reference_double_support_match_fraction")
    air_index = MEASUREMENTS.index("air_time_credit_s")
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
        mean[:, air_index] = sums[:, group_index, air_index]
        reference_count = sums[:, group_index, reference_index]
        mean[:, match_index] = np.divide(
            sums[:, group_index, match_index], reference_count,
            out=np.zeros_like(reference_count), where=reference_count > 0,
        )
        groups[name] = {
            "sample_count": int(count.sum()),
            "episode_count": int(eligible.sum()),
        }
        per_episode[name] = {"sample_count": count.tolist()}
        for index, measurement in enumerate(MEASUREMENTS):
            measurement_eligible = eligible & (reference_count > 0) if index == match_index else eligible
            values = mean[measurement_eligible, index]
            groups[name][measurement] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std()) if values.size else None,
            }
            if index == match_index:
                groups[name][measurement]["episode_count"] = int(measurement_eligible.sum())
            per_episode[name][measurement] = [
                float(v) if valid else None
                for v, valid in zip(mean[:, index], measurement_eligible, strict=True)
            ]
    return groups, per_episode


class CommandVmapWrapper(DomainRandomizationVmapWrapper):
    def reset(self, rng, command):
        def reset(sys, key, cmd):
            with self.v_env_fn(sys) as env:
                return env.reset(key, cmd)

        return jax.vmap(reset, in_axes=[self._in_axes, 0, 0])(self._sys_v, rng, command)


class PaperEvaluator:
    """Stop on terminal/invalid transitions; retain finite terminal samples.

    ``pushes=None`` uses the recorded config; True uses TB-scaled physical
    bounds, and False zeros the amplitudes while preserving the key split.
    """

    def __init__(
        self,
        env,
        make_policy,
        num_envs,
        randomization_fn=None,
        seed=12102,
        *,
        engineering_horizon=False,
        pushes=None,
    ):
        env = copy.copy(env)
        self.scale = env.config.normalization_scale
        self.schedule_scale = 1.0 if engineering_horizon else self.scale
        self.episode_length = round(1000 * self.schedule_scale)
        self.num_envs, self.seed = num_envs, seed
        self.randomized = randomization_fn is not None
        self.noisy = env.add_observation_noise
        if pushes is True:
            if not self.randomized:
                raise ValueError("Physical-push benchmark requires randomized evaluation")
            # Frozen TB's configured bounds and 2-second interval, in WR2 units.
            env.config = replace(
                env.config,
                push_interval_steps=round(100 * self.scale),
                push_linear_velocity_m_s=0.5 * self.scale,
                push_angular_velocity_rad_s=1.0 / self.scale,
            )
        elif pushes is False:
            # Preserve the interval and random-key split for a paired control.
            env.config = replace(
                env.config,
                push_linear_velocity_m_s=0.0,
                push_angular_velocity_rad_s=0.0,
            )
        self.pushes_enabled = self.randomized and env.config.push_interval_steps > 0 and (
            env.config.push_linear_velocity_m_s > 0
            or env.config.push_angular_velocity_rad_s > 0
        )
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
        # Every condition has zero yaw command, including transitions.
        reference_heading = jax.vmap(torso_heading)(state.pipeline_state.x.rot[:, self.root])
        reference_quat = jp.stack([
            jp.cos(reference_heading / 2), jp.zeros(self.num_envs),
            jp.zeros(self.num_envs), jp.sin(reference_heading / 2),
        ], axis=1)
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
            reference_velocity = jax.vmap(_inverse_rotate_vector)(
                reference_quat, p.xd.vel[:, self.root],
            )
            heading_error = jax.vmap(torso_heading)(p.x.rot[:, self.root]) - reference_heading
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
                    m["single_support_per_step"],
                    m["no_foot_support_per_step"],
                    m["left_foot_vertical_force_n_per_step"],
                    m["right_foot_vertical_force_n_per_step"],
                    m["reference_double_support_per_step"],
                    m["matched_reference_double_support_per_step"],
                    m["foot_self_contact_per_step"],
                    m["air_time_credit_s_per_step"],
                    m["contact_phase_match_per_step"],
                    reference_velocity[:, 0],
                    reference_velocity[:, 1],
                    jp.linalg.norm(reference_velocity[:, :2] - commands[:, :2], axis=1),
                    jp.abs(jp.arctan2(jp.sin(heading_error), jp.cos(heading_error))),
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
            "schema_version": 4,
            "seed": self.seed,
            "normalization_scale": self.scale,
            "com_height_m": self.com_height,
            "froude_velocity_m_s": float(np.sqrt(9.81 * self.com_height)),
            "episode_length": self.episode_length,
            "control_period_s": 0.02,
            "measurement_sampling": "end_of_control_step",
            "reference_velocity_frame": "initial desired heading; all conditions command zero yaw; native cached torso velocity",
            "heading_error_rule": "mean absolute wrapped cached torso yaw minus initial desired heading, radians",
            "support_contact_rule": "summed upward floor/foot force",
            "support_contact_force_threshold_n": self.env.config.contact_force_threshold_n,
            "reference_double_support_match_rule": "per-episode matched/reference double-support ticks; null without reference double support",
            "air_time_credit_rule": (
                "sum of qualified supported-swing durations, including discount compensation; seconds-equivalent reward credit before reward weight and control-period multiplication"
                if self.env.config.air_time_min_stance_s > 0
                else "sum of command-eligible landing durations before reward weight and control-period multiplication"
            ),
            "profile": "engineering_20s"
            if self.schedule_scale == 1.0
            else "TB_normalized",
            "deterministic_policy": True,
            "model_randomization": self.randomized,
            "reset_and_local_actuator_randomization": self.randomized,
            "push_interval_steps": self.env.config.push_interval_steps
            if self.randomized
            else 0,
            "physical_pushes_enabled": self.pushes_enabled,
            "push_linear_velocity_m_s": self.env.config.push_linear_velocity_m_s
            if self.randomized
            else 0.0,
            "push_angular_velocity_rad_s": self.env.config.push_angular_velocity_rad_s
            if self.randomized
            else 0.0,
            "observation_noise": self.noisy,
            "conditions": conditions,
        }


def selection_summary(report, metric="paper_xy_error"):
    """Finite, <=1% falls; minimize per-episode walking error within that gate.

    This selects a candidate for held-out confirmation, not a claim of TB parity.
    """
    conditions = report["conditions"]
    episodes = sum(c["episodes"] for c in conditions)
    falls = sum(c["fall_count"] for c in conditions)
    nonfinite = sum(c["nonfinite_count"] for c in conditions)
    if metric not in {"paper_xy_error", "paper_reference_xy_error"}:
        raise ValueError(f"Unknown Paper selection metric: {metric}")

    def average(name):
        values = [v for c in conditions if not c["transitions"]
                  for v in c["per_episode"]["walking"].get(name, []) if v is not None]
        return float(np.mean(values)) if values else None

    error = average("xy_error_m_s")
    reference_error = average("reference_xy_error_m_s")
    selected_error = reference_error if metric == "paper_reference_xy_error" else error
    complete = len(conditions) == len(CONDITIONS)
    eligible = (
        complete and nonfinite == 0 and falls / episodes <= 0.01 and selected_error is not None
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
        "walking_reference_xy_error_m_s": reference_error,
        "walking_reference_xy_error_TB_equivalent_m_s": None
        if reference_error is None else reference_error / report["normalization_scale"],
        "walking_heading_error_abs_rad": average("heading_error_abs_rad"),
        "selection_metric": metric,
        "selection_error_m_s": selected_error,
        "tracking_scope": "ten_fixed_nonzero_conditions_per_episode",
        "eligible": eligible,
    }
