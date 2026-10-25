"""Shared pose-based heading measurements for WR2 and native TB Paper.

This evaluates original policies. Rewards and observations are unchanged.
"""

import numpy as np


def heading_rollout(env, policy, keys, command, steps, root, sample_rewards):
    """Stop each episode at its first fall/invalid transition; keep finite falls."""
    import jax
    import jax.numpy as jp
    from brax.math import inv_rotate

    state = env.reset(keys, command)
    initial = state.pipeline_state.q[:, :7]
    alive = jp.ones(len(keys), dtype=bool)

    def step(carry, _):
        before, alive = carry
        action, _ = policy(before.obs, keys[0])
        after = env.step(before, action)
        p = after.pipeline_state
        quat = p.x.rot[:, root]
        body_velocity = jax.vmap(inv_rotate)(p.xd.vel[:, root], quat)
        body_angular = jax.vmap(inv_rotate)(p.xd.ang[:, root], quat)
        rewards, fall, actual_command = sample_rewards(before, after)
        finite = (
            jp.isfinite(p.q).all(axis=1) & jp.isfinite(p.qd).all(axis=1)
            & jp.isfinite(after.reward) & jp.isfinite(action).all(axis=1)
            & jp.isfinite(body_velocity).all(axis=1)
            & jp.isfinite(body_angular).all(axis=1)
            & jp.isfinite(rewards).all(axis=1)
        )
        for observation in jax.tree.leaves(after.obs):
            finite &= jp.isfinite(observation).reshape(len(keys), -1).all(axis=1)
        if hasattr(after, "privileged_obs"):
            finite &= jp.isfinite(after.privileged_obs).all(axis=1)
        valid = alive & finite
        trace = {
            "root_pose": p.q[:, :7], "physics_quaternion": quat,
            "body_velocity_m_s": body_velocity,
            "body_angular_velocity_rad_s": body_angular,
            "root_delta_m": p.q[:, :2] - before.pipeline_state.q[:, :2],
            "command_physical": actual_command, "reward_terms": rewards,
            "total_reward": after.reward, "valid": valid,
            "fall": valid & fall, "nonfinite": alive & ~finite,
        }
        following_alive = valid & (after.done == 0)
        following = jax.tree.map(
            lambda old, new: jp.where(
                following_alive.reshape((len(keys),) + (1,) * (new.ndim - 1)),
                new, old,
            ), before, after,
        )
        return (following, following_alive), trace

    _, trace = jax.lax.scan(step, (state, alive), None, length=steps)
    return {"initial_root_pose": initial, **trace}


def quaternion_euler(quat):
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=float), -1, 0)
    return np.stack([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ], axis=-1)


def summarize_heading(trace, dt, scale, warmup_TB_s=1.5):
    """Normalize episodes individually; report full and post-warmup windows.

    Heading is integrated root yaw relative to its initial heading. Commands
    are held forward with zero yaw; initial random yaw defines the goal axes.
    TB length/time/velocity conversion is s^2/s/s, respectively.
    """
    valid = np.asarray(trace["valid"], bool)
    steps, episodes = valid.shape
    initial = np.asarray(trace["initial_root_pose"], float)
    pose = np.asarray(trace["root_pose"], float)
    safe_pose = np.where(valid[..., None], pose, initial[None])
    all_pose = np.concatenate([initial[None], safe_pose], axis=0)
    yaw = np.unwrap(quaternion_euler(all_pose[..., 3:7])[..., 2], axis=0)
    initial_heading = yaw[0]
    heading = yaw[1:] - initial_heading
    heading_rate = np.diff(yaw, axis=0) / dt
    delta = np.asarray(trace["root_delta_m"], float)
    c, s = np.cos(initial_heading), np.sin(initial_heading)
    reference_delta = np.stack([
        delta[..., 0] * c + delta[..., 1] * s,
        -delta[..., 0] * s + delta[..., 1] * c,
    ], axis=-1)
    body = np.asarray(trace["body_velocity_m_s"], float) / scale
    command = np.asarray(trace["command_physical"], float)
    np.testing.assert_allclose(command[..., 1:], 0, atol=1e-7)
    if valid.any():
        np.testing.assert_allclose(
            delta[valid], np.diff(all_pose[..., :2], axis=0)[valid], atol=1e-6,
        )
    physical_pitch = quaternion_euler(safe_pose[..., 3:7])[..., 1]
    cached_euler = quaternion_euler(trace["physics_quaternion"])
    omega = np.asarray(trace["body_angular_velocity_rad_s"], float) * scale
    true_rate = (
        omega[..., 1] * np.sin(cached_euler[..., 0])
        + omega[..., 2] * np.cos(cached_euler[..., 0])
    ) / np.cos(cached_euler[..., 1])
    rewards = np.asarray(trace["reward_terms"], float)
    full = np.arange(steps)[:, None] >= 0
    steady_start = int(np.ceil(warmup_TB_s * scale / dt))
    groups = {"full": full, "post_warmup": np.arange(steps)[:, None] >= steady_start}
    result = []
    for i in range(episodes):
        episode = {
            "episode": i, "fell": bool(np.asarray(trace["fall"])[:, i].any()),
            "nonfinite": bool(np.asarray(trace["nonfinite"])[:, i].any()),
            "valid_steps": int(valid[:, i].sum()),
            "completed": bool(valid[:, i].all())
            and not bool(np.asarray(trace["fall"])[:, i].any()),
        }
        for name, group in groups.items():
            indices = np.flatnonzero(valid[:, i] & group[:, 0])
            if not len(indices):
                episode[name] = None
                continue
            first, last = indices[0], indices[-1]
            elapsed = len(indices) * dt
            d = reference_delta[indices, i].sum(axis=0)
            native_total = np.asarray(trace["total_reward"])[indices, i].sum()
            shadow_yaw = 5 * np.exp(-25 * true_rate[indices, i]**2) * dt
            episode[name] = {
                "duration_physical_s": elapsed, "duration_TB_s": elapsed / scale,
                "body_forward_TB_m_s": float(body[indices, i, 0].mean()),
                "body_xy_error_TB_m_s": float(np.linalg.norm(
                    body[indices, i, :2] - command[indices, i, :2] / scale, axis=-1,
                ).mean()),
                "reference_forward_m": float(d[0]),
                "reference_lateral_m": float(d[1]),
                "reference_lateral_abs_m": float(abs(d[1])),
                "reference_forward_TB_m": float(d[0] / scale**2),
                "reference_lateral_TB_m": float(d[1] / scale**2),
                "reference_forward_TB_m_s": float(d[0] / elapsed / scale),
                "root_path_m": float(np.linalg.norm(delta[indices, i], axis=-1).sum()),
                "root_net_distance_m": float(np.linalg.norm(d)),
                "mean_abs_heading_error_deg": float(np.rad2deg(np.abs(heading[indices, i]).mean())),
                "final_heading_error_deg": float(np.rad2deg(heading[last, i])),
                "net_heading_change_deg": float(np.rad2deg(yaw[last + 1, i] - yaw[first, i])),
                "abs_net_heading_change_deg": float(abs(np.rad2deg(yaw[last + 1, i] - yaw[first, i]))),
                "mean_heading_rate_TB_rad_s": float(heading_rate[indices, i].mean() * scale),
                "instantaneous_heading_rate_std_TB_rad_s": float(true_rate[indices, i].std()),
                "mean_pitch_deg": float(np.rad2deg(physical_pitch[indices, i].mean())),
                "native_return_per_TB_s": float(native_total / (elapsed / scale)),
                "native_xy_reward_per_TB_s": float(rewards[indices, i, 0].sum() / (elapsed / scale)),
                "native_yaw_reward_per_TB_s": float(rewards[indices, i, 1].sum() / (elapsed / scale)),
                "native_orientation_reward_per_TB_s": float(rewards[indices, i, 2].sum() / (elapsed / scale)),
                "heading_rate_shadow_return_per_TB_s": float((
                    native_total - rewards[indices, i, 1].sum() + shadow_yaw.sum()
                ) / (elapsed / scale)),
            }
        result.append(episode)
    means = {}
    for name in groups:
        available = [ep[name] for ep in result if ep[name] is not None]
        means[name] = {
            key: {"mean": float(np.mean([ep[key] for ep in available])),
                  "std": float(np.std([ep[key] for ep in available]))}
            for key in (available[0] if available else {})
        }
        means[name]["episode_count"] = len(available)
    return {
        "episodes": episodes, "fall_count": sum(ep["fell"] for ep in result),
        "nonfinite_count": sum(ep["nonfinite"] for ep in result),
        "completed_count": sum(ep["completed"] for ep in result),
        "groups": means, "per_episode": result,
    }
