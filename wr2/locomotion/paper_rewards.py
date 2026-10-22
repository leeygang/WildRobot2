"""Table VII rewards from the frozen ToddlerBot paper walking control.

Velocities use TB-equivalent units (v/s, omega*s). At the shared 50Hz control
rate, landing air time stays in seconds and clearance normalizes height only.
Their cycle integrals then scale with duration, as continuous rewards do.
Foot distances use the measured stance-width ratio. Torque stays in Nm.
Reference: https://arxiv.org/html/2502.00893v2, Table VII.
"""

import jax.numpy as jp


def range_reward(value, minimum, maximum, slope=100.0):
    below = jp.minimum(value - minimum, 0.0)
    above = jp.maximum(value - maximum, 0.0)
    return (jp.exp(-slope * jp.abs(below)) + jp.exp(-slope * jp.abs(above))) / 2


def qualified_air_time_step(
    state, contact, active, *, dt, minimum_stance_steps, discount_compensation
):
    """Pay supported swing time once its landing sustains sampled contact.

    The caller uses gamma**(-(minimum_stance_steps - 1)) to preserve the
    discounted value at the original landing time. Short contacts and command
    stops cancel pending credit. Raw TB air/clearance counters are separate.
    """
    contact, active = jp.asarray(contact, dtype=bool), jp.asarray(active, dtype=bool)
    landed = contact & (state["stance_steps"] == 0)
    candidate = jp.where(landed & active, state["supported_air_time"], 0.0)
    rejected = jp.where(~contact | ~active, state["pending_air_time"], 0.0)
    pending = jp.where(landed, candidate, state["pending_air_time"])
    pending = jp.where(contact & active, pending, 0.0)
    steps = jp.where(contact, state["stance_steps"] + 1, 0)
    qualified = contact & (steps >= minimum_stance_steps)
    paid = jp.where(qualified, pending, 0.0)
    following = {
        "supported_air_time": jp.where(
            contact, 0.0, state["supported_air_time"] + dt * contact[::-1]
        ),
        "pending_air_time": jp.where(qualified, 0.0, pending),
        "stance_steps": steps,
    }
    metrics = {
        "air_time_credit_s_per_step": jp.sum(paid) * discount_compensation,
        "candidate_air_time_s_per_step": jp.sum(candidate),
        "qualified_air_time_s_per_step": jp.sum(paid),
        "rejected_air_time_s_per_step": jp.sum(rejected),
        "left_foot_stance_duration_s_per_step": jp.maximum(steps[0] - 1, 0) * dt,
        "right_foot_stance_duration_s_per_step": jp.maximum(steps[1] - 1, 0) * dt,
    }
    return following, metrics

def paper_reward_terms(
    *,
    motor_position,
    home_position,
    reference_position,
    leg_indices,
    alignment_indices,
    motor_velocity,
    torque,
    linear_velocity,
    angular_velocity,
    command,
    orientation_error,
    euler,
    action,
    last_action,
    second_last_action,
    contact,
    previous_contact,
    reference_contact,
    air_time,
    air_distance,
    foot_velocity_xy,
    foot_distance,
    foot_distance_scale,
    minimum_foot_distance,
    maximum_foot_distance,
    collision_count,
    terminal,
    scale,
    deadzone,
):
    """Return unweighted terms; air accumulators are read before their update.

    TB's slip broadcast bug is corrected: each foot masks its own XY speed.
    Clearance retains TB's sample sum of foot heights, rather than a dt integral.
    """
    linear = linear_velocity / scale
    angular = angular_velocity * scale
    normalized_command = command * jp.asarray([1 / scale, 1 / scale, scale])
    active = jp.linalg.norm(normalized_command) > deadzone
    filtered_contact = contact | previous_contact
    first_time = (air_time > 0) & filtered_contact
    first_distance = (air_distance > 0) & filtered_contact
    leg_error = motor_position[leg_indices] - reference_position[leg_indices]
    home_error = jp.sum(
        jp.abs(motor_position[leg_indices] - home_position[leg_indices])
    )
    # Indices are [hip_pitch, ankle_pitch, knee_pitch], one row per leg.
    pitch = jp.abs(motor_position[alignment_indices])
    zero = jp.zeros(())
    return {
        "torso_pos": zero,
        "torso_quat": jp.exp(-20 * orientation_error**2),
        "lin_vel_xy": jp.exp(-100 * jp.sum((linear[:2] - normalized_command[:2]) ** 2)),
        "lin_vel_z": jp.exp(-100 * linear[2] ** 2),
        "ang_vel_xy": jp.exp(-25 * jp.sum(angular[:2] ** 2)),
        "ang_vel_z": jp.exp(-25 * (angular[2] - normalized_command[2]) ** 2),
        "leg_motor_pos": -jp.mean(leg_error**2),
        "neck_motor_pos": zero,
        "arm_motor_pos": zero,
        "waist_motor_pos": zero,
        "feet_contact": jp.sum(contact == reference_contact).astype(jp.float32),
        "feet_air_time": jp.sum(air_time * first_time) * active,
        "feet_clearance": jp.sum(air_distance * first_distance) / scale**2 * active,
        "feet_distance": range_reward(
            foot_distance,
            minimum_foot_distance,
            maximum_foot_distance,
            100 / foot_distance_scale,
        ),
        "feet_slip": -jp.sum(contact * jp.sum((foot_velocity_xy / scale) ** 2, axis=1)),
        "align_ground": -jp.mean((pitch[:, 0] + pitch[:, 1] - pitch[:, 2]) ** 2),
        "stand_still": -(home_error**2)
        * (jp.linalg.norm(normalized_command) < deadzone),
        "torso_roll": range_reward(euler[0], -0.1, 0.1),
        "torso_pitch": range_reward(euler[1], -0.2, 0.2),
        "collision": -collision_count,
        "leg_action_rate": -jp.mean((action - last_action) ** 2),
        "leg_action_acc": -jp.mean(
            (action - 2 * last_action + second_last_action) ** 2
        ),
        "motor_torque": -jp.mean(torque**2),
        "energy": -jp.mean((torque * motor_velocity * scale) ** 2),
        "survival": -terminal.astype(jp.float32),
    }
