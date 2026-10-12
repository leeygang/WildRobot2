"""Model-derived sagittal reflection shared by training and diagnostics."""

from __future__ import annotations

import numpy as np

REFLECTION = np.diag([1.0, -1.0, 1.0])


def joint_reflection(model, names):
    """Derive coordinate signs from neutral world joint axes, not WR1 signs."""
    import mujoco

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
    """Reflect every newest-first frame of WR2's actual JAX actor input."""
    import jax.numpy as jp

    joint_count, action_count = len(source), len(active)
    previous_start = 5 + 2 * joint_count
    gyro_start = previous_start + action_count
    gravity_start = gyro_start + 3
    heading_start = gravity_start + 3
    frame_size = heading_start + 2 * heading
    frames = observation.reshape((*observation.shape[:-1], -1, frame_size))
    mirrored = frames.at[..., :2].set(-frames[..., :2])
    mirrored = mirrored.at[..., 2:5].set(frames[..., 2:5] * jp.asarray([1, -1, -1]))
    for start in (5, 5 + joint_count):
        mirrored = mirrored.at[..., start : start + joint_count].set(
            frames[..., start + source] * signs
        )
    active_lookup = (
        jp.full(joint_count, -1, dtype=jp.int32).at[active].set(jp.arange(action_count))
    )
    active_source = active_lookup[source[active]]
    mirrored = mirrored.at[..., previous_start:gyro_start].set(
        frames[..., previous_start + active_source] * signs[active]
    )
    mirrored = mirrored.at[..., gyro_start:gravity_start].set(
        frames[..., gyro_start:gravity_start] * jp.asarray([-1, 1, -1])
    )
    mirrored = mirrored.at[..., gravity_start:heading_start].set(
        frames[..., gravity_start:heading_start] * jp.asarray([1, -1, 1])
    )
    if heading:
        mirrored = mirrored.at[..., heading_start:].set(
            frames[..., heading_start:] * jp.asarray([-1, 1])
        )
    return mirrored.reshape(observation.shape)


class ActorMirror:
    """Torch reflection and RSL-RL's original-plus-mirror callback contract.

    Inputs are raw, explicitly scaled WR2 observations, not empirically
    normalized channels. No privileged critic features or sampled transitions
    are augmented: RSL-RL uses this callback only for its actor mirror MSE.
    """

    def __init__(self, source, signs, active, *, heading, device):
        import torch

        source, signs, active = (
            np.asarray(source),
            np.asarray(signs),
            np.asarray(active),
        )
        joint_count, action_count = len(source), len(active)
        if not np.array_equal(source[source], np.arange(joint_count)) or not np.all(
            np.abs(signs) == 1
        ):
            raise ValueError(
                "Joint reflection must be an involutive signed permutation"
            )
        if not np.all(signs * signs[source] == 1):
            raise ValueError("Joint reflection signs must be involutive")
        lookup = np.full(joint_count, -1)
        lookup[active] = np.arange(len(active))
        action_source = lookup[source[active]]
        if np.any(action_source < 0):
            raise ValueError("Active joints must include their reflected partners")
        previous_start = 5 + 2 * joint_count
        gyro_start = previous_start + action_count
        gravity_start, heading_start = gyro_start + 3, gyro_start + 6
        self.frame_size = heading_start + 2 * heading
        index, scale = np.arange(self.frame_size), np.ones(self.frame_size)
        scale[:2], scale[2:5] = -1, [1, -1, -1]
        for start in (5, 5 + joint_count):
            index[start : start + joint_count] = start + source
            scale[start : start + joint_count] = signs
        index[previous_start:gyro_start] = previous_start + action_source
        scale[previous_start:gyro_start] = signs[active]
        scale[gyro_start:gravity_start], scale[gravity_start:heading_start] = [-1, 1, -1], [1, -1, 1]
        if heading:
            scale[heading_start:] = [-1, 1]
        self.observation_source = torch.tensor(index, dtype=torch.long, device=device)
        self.observation_signs = torch.tensor(scale, dtype=torch.float32, device=device)
        self.action_source = torch.tensor(
            action_source, dtype=torch.long, device=device
        )
        self.action_signs = torch.tensor(
            signs[active], dtype=torch.float32, device=device
        )

    @classmethod
    def from_environment(cls, env, device):
        import mujoco

        if env.config.active_groups != ("leg",):
            raise ValueError("Actor mirror experiment supports leg-only policies")
        robot = env.robot
        model = mujoco.MjModel.from_xml_path(str(robot.directory / "scene_mjx.xml"))
        source, signs = joint_reflection(model, robot.actuator_names)
        limits = np.sort(
            np.stack([robot.lower_limit_rad[source], robot.upper_limit_rad[source]])
            * signs,
            axis=0,
        )
        if not np.allclose(
            limits, [robot.lower_limit_rad, robot.upper_limit_rad], atol=1e-6, rtol=0
        ):
            raise ValueError("Actor mirror requires reflected safe joint limits")
        if not np.allclose(
            robot.home_position_rad[source] * signs,
            robot.home_position_rad,
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                "Actor mirror requires a reflected walk_home action origin"
            )
        return cls(
            source,
            signs,
            env._active_indices,
            heading=env.config.heading_observation,
            device=device,
        )

    def observations(self, observation):
        if observation.shape[-1] != 15 * self.frame_size:
            raise ValueError("Actor mirror requires all 15 raw actor history frames")
        frames = observation.reshape((*observation.shape[:-1], 15, self.frame_size))
        return (
            frames.index_select(-1, self.observation_source) * self.observation_signs
        ).reshape(observation.shape)

    def actions(self, action):
        if action.shape[-1] != len(self.action_source):
            raise ValueError("Actor mirror action count differs from active leg joints")
        return action.index_select(-1, self.action_source) * self.action_signs

    def __call__(self, *, obs=None, actions=None, env=None, obs_type="policy"):
        import torch

        if obs_type != "policy":
            raise ValueError("Actor mirror does not augment critic observations")
        observations = (
            None if obs is None else torch.cat([obs, self.observations(obs)], dim=0)
        )
        action_values = (
            None
            if actions is None
            else torch.cat([actions, self.actions(actions)], dim=0)
        )
        return observations, action_values
