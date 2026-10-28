"""Check a model-home hip action center with the original WR2 physics/policy.

This is a diagnostic, not a training configuration or a policy conversion.
TB source: paper-walking-baseline/toddlerbot/locomotion/mjx_env.py.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from wr2.locomotion.heading_evaluation import quaternion_euler
from wr2.tools.visual_eval_log import checkpoint_hash, json_value


def hip_center(env, use_model_home):
    center = np.asarray(env._home_ctrl).copy()
    indices = np.array([env.robot.actuator_names.index(f"{side}_hip_pitch")
                        for side in ("left", "right")])
    if use_model_home:
        center[indices] = np.asarray(env.robot.home_position_rad)[indices]
    return center, indices


def coverage(env, center, hips):
    """Audit every native lookup frame; include quantization and bias corners."""
    scale, gain = env.config.normalization_scale, env.config.policy_action_scale_rad
    ref = env._zmp_reference._host
    resolution = env._command_resolution_rad
    bias = env.config.randomization.target_bias_rad
    quantize = lambda q: np.clip(np.round(q/resolution)*resolution,
                                 np.asarray(env._ctrl_lower), np.asarray(env._ctrl_upper))
    lower, upper = quantize(center-gain), quantize(center+gain)
    guaranteed_lower = quantize(center-gain+bias[1])
    guaranteed_upper = quantize(center+gain+bias[0])
    rows = []
    for i, command in enumerate(ref['commands']):
        for foot, (joint, sign) in enumerate(zip(hips, (-1, 1))):
            q = ref['motors'][i, :, joint]*sign
            lo, hi = sorted([lower[joint]*sign, upper[joint]*sign])
            blo, bhi = sorted([guaranteed_lower[joint]*sign, guaranteed_upper[joint]*sign])
            stance = ref['stance'][i, :, foot]
            rows.append({
                'command_TB': (command/[scale, scale, 1/scale]).tolist(),
                'joint': env.robot.actuator_names[joint],
                'forward_angle_window_deg': np.rad2deg([lo, hi]).tolist(),
                'posterior_stance_outside_fraction': float(np.mean(q[stance] < lo)),
                'anterior_stance_outside_fraction': float(np.mean(q[stance] > hi)),
                'stance_outside_with_worst_bias_fraction': float(np.mean((q[stance] < blo) | (q[stance] > bhi))),
                'all_reference_outside_fraction': float(np.mean((q < lo) | (q > hi))),
            })
    return rows


def summarize(trace, dt, scale):
    """Include finite terminal prefixes; never average repeated frozen padding."""
    valid = trace['valid'].astype(bool)
    start = int(np.ceil(1.5*scale/dt))
    results = []
    for i in range(valid.shape[1]):
        mask = valid[:, i] & (np.arange(len(valid)) >= start)
        full = valid[:, i]
        row = {'episode': i, 'valid_steps': int(full.sum()),
               'fell': bool(trace['fall'][:, i].any()),
               'nonfinite': bool(trace['nonfinite'][:, i].any()),
               'completed': bool(full.all() and not trace['fall'][:, i].any())}
        if mask.any():
            pose = trace['qpos'][mask, i]
            angles = np.rad2deg(quaternion_euler(pose[:, 3:7]))
            initial = trace['initial_qpos'][i]
            yaw0 = quaternion_euler(initial[3:7])[2]
            delta = trace['root_delta_m'][mask, i]
            forward = delta[:, 0]*np.cos(yaw0)+delta[:, 1]*np.sin(yaw0)
            row.update({
                'forward_speed_TB_m_s': float(forward.sum()/(mask.sum()*dt)/scale),
                'mean_pitch_deg': float(angles[:, 1].mean()),
                'roll90_span_deg': float(np.ptp(np.quantile(angles[:, 0], [.05, .95]))),
                'root_z90_span_TB_m': float(np.ptp(np.quantile(pose[:, 2], [.05, .95]))/scale**2),
                'both_feet_supported_fraction': float(np.mean(trace['contact'][mask, i].all(axis=-1))),
                'foot_self_contact_fraction': float(trace['self_contact'][mask, i].mean()),
                'mean_floor_force_LR_n': trace['floor_force_n'][mask, i].mean(axis=0).tolist(),
                'delivered_min_rad': trace['target_rad'][mask, i].min(axis=0).tolist(),
                'delivered_max_rad': trace['target_rad'][mask, i].max(axis=0).tolist(),
                'joint_tracking_RMS_rad': np.sqrt(np.mean((trace['target_rad'][mask, i]-trace['motor_rad'][mask, i])**2, axis=0)).tolist(),
            })
            moving = mask & (np.linalg.norm(trace['command_m_s'][:, i, :2], axis=-1) > 1e-6)
            row['walking_steps'] = int(moving.sum())
            if moving.any():
                delta = trace['root_delta_m'][moving, i]
                forward = delta[:, 0]*np.cos(yaw0)+delta[:, 1]*np.sin(yaw0)
                row['walking_forward_speed_TB_m_s'] = float(forward.sum()/(moving.sum()*dt)/scale)
                row['walking_mean_pitch_deg'] = float(np.rad2deg(quaternion_euler(trace['qpos'][moving, i, 3:7]))[:, 1].mean())
        if full.any():
            row.update({
                'maximum_final_substep_torque_nm': float(np.max(abs(trace['torque_nm'][full, i]))),
                'maximum_target_reconstruction_error_rad': float(trace['target_error_rad'][full, i].max()),
                'maximum_joint_limit_violation_rad': float(trace['joint_violation_rad'][full, i].max()),
            })
        results.append(row)
    return results


def rollout(env, wrapped, policy, keys, command, mode, length):
    import jax
    import jax.numpy as jp
    from wr2.tools.diagnose_walking import target_trace
    from wr2.locomotion.paper_evaluation import command_and_groups, set_command

    initial_command = jp.where(mode == 3, jp.zeros_like(command), command)
    state = wrapped.reset(keys, jp.tile(initial_command, (len(keys), 1)))
    initial = state.pipeline_state.q
    initial_obs = state.obs
    hold = (env._home_ctrl[env._active_indices]-env.action_center[env._active_indices])/env._policy_action_scale_rad
    initial_action = jp.tile(jp.where(mode == 2, hold, jp.zeros_like(hold)), (len(keys), 1))
    # Constant holding probes do not use the actor. Seed the complete delay
    # history with their holding action, retaining the original physical reset.
    state = state.replace(info={**state.info,
        'previous_action': initial_action, 'last_applied_action': initial_action,
        'second_last_applied_action': initial_action})
    alive = jp.ones(len(keys), dtype=bool)

    def sample(before, after, action):
        p, m = after.pipeline_state, after.metrics
        _, target, _, _, _ = target_trace(env, before, action)
        motor = p.q[env._joint_qpos_indices]
        violation = jp.maximum(env._ctrl_lower-motor, motor-env._ctrl_upper)
        return {
            'qpos': p.q, 'root_delta_m': p.q[:2]-before.pipeline_state.q[:2],
            'command_m_s': before.info['command'],
            'actor_action': action, 'target_rad': target, 'motor_rad': motor,
            'torque_nm': p.actuator_force,
            'contact': jp.asarray([m['left_foot_contact_per_step'], m['right_foot_contact_per_step']]) > 0,
            'floor_force_n': jp.asarray([m['left_foot_vertical_force_n_per_step'], m['right_foot_vertical_force_n_per_step']]),
            'self_contact': m['foot_self_contact_per_step'],
            'contact_phase_match': m['contact_phase_match_per_step'],
            'foot_geom_center_m': p.geom_xpos[env._foot_geom_ids],
            'foot_geom_rotation': p.geom_xmat[env._foot_geom_ids],
            'joint_violation_rad': jp.maximum(0., jp.max(violation)),
            'target_error_rad': jp.where(after.done == 0, jp.max(abs(target-after.info['previous_target'])), 0.),
        }

    def step(carry, index):
        before, alive = carry
        scheduled, _ = command_and_groups(index, command[:2], True, env.config.normalization_scale)
        before = jax.lax.cond(mode == 3,
            lambda s: set_command(s, jp.tile(scheduled, (len(keys), 1)), env.config.normalization_scale),
            lambda s: s, before)
        actor, _ = policy(before.obs, keys[0])
        action = jp.where((mode == 0) | (mode == 3), actor, jp.tile(jp.where(mode == 2, hold, jp.zeros_like(hold)), (len(keys), 1)))
        after = wrapped.step(before, action)
        values = jax.vmap(sample)(before, after, action)
        finite = jp.isfinite(after.reward)
        for value in jax.tree.leaves((values, after.obs)):
            finite &= jp.isfinite(value).reshape(len(keys), -1).all(axis=1)
        valid = alive & finite
        values.update({'valid': valid, 'fall': valid & (after.metrics['fall'] > 0), 'nonfinite': alive & ~finite})
        following_alive = valid & (after.done == 0)
        following = jax.tree.map(lambda old, new: jp.where(
            following_alive.reshape((len(keys),)+(1,)*(new.ndim-1)), new, old), before, after)
        return (following, following_alive), values

    _, trace = jax.lax.scan(step, (state, alive), jp.arange(length))
    return {'initial_qpos': initial, 'initial_obs': initial_obs, **trace}


def run(args):
    from wr2.locomotion.train import _configure_backend_logging, _filter_optional_backend_import_messages
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        import jax.numpy as jp
        from brax.training import types
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import checkpoint, networks
        from wr2.locomotion.configs import load_training_config
        from wr2.locomotion.domain_randomization import make_domain_randomizer
        from wr2.locomotion.paper_evaluation import PaperEvaluator
        from wr2.locomotion.ppo import make_network_factory
        from wr2.locomotion.train import _training_provenance
        from wr2.locomotion.walking_env import WR2WalkingEnv

    if jax.default_backend() != 'gpu' and not args.allow_cpu:
        raise RuntimeError('Use --allow-cpu for intentional Mac validation')
    jax.config.update('jax_default_matmul_precision', 'high')
    cfg = load_training_config(args.run_dir/'training_config.yaml')
    if cfg.environment.recipe != 'tb_paper':
        raise ValueError('Use a saved WR2 Paper run')
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.checkpoint or args.run_dir/'checkpoints/000200294400'
    params = checkpoint.load(str(path.resolve()))

    class ProbeEnv(WR2WalkingEnv):
        def __init__(self, use_model_home):
            super().__init__(cfg.environment, add_observation_noise=not args.nominal,
                             reward_discounting=cfg.ppo.discounting)
            center, self.hips = hip_center(self, use_model_home)
            self.action_center = jp.asarray(center)

        def _policy_target(self, action):
            center = getattr(self, 'action_center', self._home_ctrl)
            return self._home_ctrl.at[self._active_indices].set(
                center[self._active_indices]+self._policy_action_scale_rad*action)

    print('Building the full native reference grid...', flush=True)
    current, candidate = ProbeEnv(False), ProbeEnv(True)
    np.testing.assert_array_equal(current._default_qpos, candidate._default_qpos)
    np.testing.assert_array_equal(current._home_ctrl, candidate._home_ctrl)
    np.testing.assert_array_equal(current._zmp_reference._motors, candidate._zmp_reference._motors)
    hold = (np.asarray(candidate._home_ctrl)-np.asarray(candidate.action_center))/cfg.environment.policy_action_scale_rad
    assert abs(hold[np.asarray(candidate._active_indices)]).max() <= 1
    args.output.mkdir(parents=True)
    coverage_report = {name: coverage(env, np.asarray(env.action_center), env.hips)
                       for name, env in [('current', current), ('model_hip', candidate)]}
    (args.output/'coverage.json').write_text(json.dumps(coverage_report, indent=2)+'\n')
    report = {'complete': False, 'provenance': _training_provenance(), 'arguments': vars(args),
              'configuration': asdict(cfg), 'checkpoint_sha256': checkpoint_hash(path),
              'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'saved_config_sha256': hashlib.sha256((args.run_dir/'training_config.yaml').read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256((current.robot.directory/'wr2_mjx.xml').read_bytes()).hexdigest(),
              'initial_qpos': np.asarray(current._default_qpos),
              'observation_home_rad': np.asarray(current._home_ctrl),
              'action_centers_rad': {name: np.asarray(env.action_center) for name, env in [('current', current), ('model_hip', candidate)]},
              'active_joint_names': [current.robot.actuator_names[int(i)] for i in current._active_indices],
              'motor_names': list(current.robot.actuator_names),
              'motor_limits_rad': np.stack([np.asarray(current._ctrl_lower), np.asarray(current._ctrl_upper)], axis=1),
              'control_period_s': current.robot.control_period_s,
              'physics_timestep_s': current.robot.simulation_timestep_s,
              'foot_half_sizes_m': np.asarray(current.sys.mj_model.geom_size)[np.asarray(current._foot_geom_ids)],
              'geometry_sampling': 'cached last-substep geometry/contact; one physics timestep before integrated qpos',
              'holding_action': hold[np.asarray(candidate._active_indices)],
              'scope': 'Diagnostic changes only hip-pitch action centers. Grounded reset, position-observation/reward home and references are preserved. Constant probes do not use the actor. Frozen-policy probes retain raw policy weights/actions and original reset observations; they are not converted or retrained policies.',
              'torque_sampling': 'last physics substep, not per-control-step peak', 'results': []}

    def save():
        (args.output/'summary.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False)+'\n')

    preprocess = running_statistics.normalize if cfg.ppo.normalize_observations else types.identity_observation_preprocessor
    net = make_network_factory(cfg.network)(current.observation_size, current.action_size,
                                            preprocess_observations_fn=preprocess)
    policy = networks.make_inference_fn(net)(params[:2], deterministic=True)
    initial_by_case = {}
    hold_traces = {}
    cases = [('zero', 1, 0.), ('hold', 2, 0.)] + [('policy', 0, v) for v in (0., .10, .15, .20)] + [('transition', 3, .20)]
    for name, env in [('current', current), ('model_hip', candidate)]:
        randomize = None if args.nominal else make_domain_randomizer(cfg.environment.randomization)
        for seed in args.seeds:
            evaluator = PaperEvaluator(env, lambda _: policy, args.num_envs, randomize, seed=seed, pushes=False)
            length = evaluator.episode_length
            compiled = jax.jit(lambda cmd, mode: rollout(env, evaluator.env, policy,
                evaluator.reset_keys, cmd, mode, length))
            for mode_name, mode, speed in cases:
                label = f'{name}_{mode_name}_forward{round(speed*100):03d}_seed{seed}'
                print(f'Native {jax.default_backend()} validation: {label}', flush=True)
                raw = jax.device_get(compiled(jp.asarray([speed*env.config.normalization_scale, 0., 0.], dtype=jp.float32), jp.asarray(mode)))
                initial_obs = raw.pop('initial_obs')
                obs_hash = hashlib.sha256(b''.join(np.asarray(initial_obs[k]).tobytes() for k in sorted(initial_obs))).hexdigest()
                trace = {k: np.asarray(v) for k, v in raw.items()}
                key = (seed, mode_name, speed)
                if name == 'current':
                    initial_by_case[key] = (trace['initial_qpos'], obs_hash)
                else:
                    np.testing.assert_array_equal(trace['initial_qpos'], initial_by_case[key][0])
                    assert obs_hash == initial_by_case[key][1]
                np.savez_compressed(args.output/f'{label}.npz', **trace)
                episodes = summarize(trace, env.robot.control_period_s, env.config.normalization_scale)
                entry = {'origin': name, 'mode': mode_name, 'command_TB_m_s': speed, 'seed': seed,
                         'trace': f'{label}.npz', 'initial_observation_sha256': obs_hash, 'per_episode': episodes,
                         'completed': sum(r['completed'] for r in episodes), 'falls': sum(r['fell'] for r in episodes),
                         'nonfinite': sum(r['nonfinite'] for r in episodes)}
                report['results'].append(entry)
                if mode_name == 'hold':
                    if name == 'current':
                        hold_traces[seed] = trace
                    else:
                        other = hold_traces[seed]
                        mask = trace['valid'] & other['valid']
                        error = float(np.max(abs(trace['target_rad'][mask]-other['target_rad'][mask])))
                        np.testing.assert_allclose(trace['target_rad'][mask], other['target_rad'][mask], atol=1e-6)
                        entry['same_pose_control_target_max_error_rad'] = error
                save()
                print(f"  completed={entry['completed']}/{args.num_envs}, falls={entry['falls']}", flush=True)
    report['complete'] = True
    save()
    print(f'Wrote {args.output}/summary.json', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nominal', action='store_true', help='Disable model/reset/actuator variation and observation noise')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[12102, 22102])
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args(argv)
    if args.num_envs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('num-envs must be positive; seeds must be unique')
    return args


if __name__ == '__main__':
    run(parse_args())
