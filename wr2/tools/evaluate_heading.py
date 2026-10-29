"""Evaluate retained WR2/TB Paper policies for forward progress and heading."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from wr2.locomotion.heading_evaluation import heading_rollout, summarize_heading
from wr2.tools.visual_eval_log import checkpoint_hash, json_value


COMMANDS = (.10, .15, .20)
TERMS = ("lin_vel_xy", "ang_vel_z", "torso_quat")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)
    parser.add_argument("--num-envs", type=int, default=128,
                        help="Randomized episodes per seed; nominal always runs once")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12102, 22102])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--tb-root", type=Path,
                        help="Verified Paper checkout, using its native Python/Brax runtime")
    args = parser.parse_args(argv)
    if args.num_envs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("num-envs must be positive and seeds must be unique")
    if args.checkpoint_steps is None:
        args.checkpoint_steps = [200089600] if args.tb_root else [120053760, 200089600, 300134400]
    if any(step < 1 for step in args.checkpoint_steps) or len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        parser.error("checkpoint steps must be positive and unique")
    return args


def load_wr2(run_dir):
    import jax
    import jax.numpy as jp
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    from wr2.locomotion.configs import load_training_config
    from wr2.locomotion.domain_randomization import make_domain_randomizer
    from wr2.locomotion.paper_evaluation import PaperEvaluator
    from wr2.locomotion.ppo import make_network_factory
    import wr2.locomotion.walking_env as module
    from wr2.reference.paper_walk import paper_command_grid

    cfg = load_training_config(run_dir / "training_config.yaml")
    if cfg.environment.recipe != "tb_paper":
        raise ValueError("A saved WR2 Paper configuration is required")
    scale = cfg.environment.normalization_scale
    grid = paper_command_grid(scale)
    requested = jp.asarray([[v * scale, 0, 0] for v in COMMANDS])
    indices = np.asarray(jax.vmap(lambda c: jp.argmin(
        jp.linalg.norm(jp.asarray(grid, dtype=jp.float32) - c, axis=1)
    ))(requested))
    reference = partial(module.PaperWalkReference, command_vectors=grid[indices])
    print("Building the three exact native reference lookup rows...", flush=True)
    with patch.object(module, "PaperWalkReference", reference):
        env = module.WR2WalkingEnv(cfg.environment, add_observation_noise=True,
                                   reward_discounting=cfg.ppo.discounting)
    preprocess = running_statistics.normalize if cfg.ppo.normalize_observations else types.identity_observation_preprocessor
    net = make_network_factory(cfg.network)(env.observation_size, env.action_size,
                                            preprocess_observations_fn=preprocess)
    native_policy = networks.make_inference_fn(net)

    def evaluator(varied, seed, count):
        randomize = make_domain_randomizer(cfg.environment.randomization) if varied else None
        env.add_observation_noise = varied
        wrapped = PaperEvaluator(env, native_policy, count, randomize, seed=seed, pushes=False)

        def sample(before, after):
            return (jp.stack([after.metrics["paper_reward/" + name] for name in TERMS], axis=1),
                    after.metrics["fall"] > 0, before.info["command"])

        return wrapped.env, wrapped.reset_keys, lambda p: native_policy(p[:2], deterministic=True), sample

    source = json.loads((run_dir / "run_config.json").read_text())
    return {
        "scale": scale, "dt": env.robot.control_period_s, "root": env._root_link_index,
        "load": lambda step: checkpoint.load(str((run_dir / "checkpoints" / f"{step:012d}").resolve())),
        "evaluator": evaluator, "command": lambda v, n: jp.tile(jp.asarray([v*scale, 0, 0]), (n, 1)),
        "configuration": asdict(cfg), "training_provenance": source["provenance"],
        "active_joint_names": [env.robot.actuator_names[int(i)] for i in env._active_indices],
        "policy_action_home_rad": np.asarray(env._action_center[env._active_indices]).tolist(),
        "observation_home_rad": np.asarray(env._home_ctrl[env._active_indices]).tolist(),
        "model_home_rad": np.asarray(env.robot.home_position_rad)[np.asarray(env._active_indices)].tolist(),
        "joint_limits_rad": np.stack([np.asarray(env._ctrl_lower), np.asarray(env._ctrl_upper)], axis=1)[np.asarray(env._active_indices)].tolist(),
        "policy_action_scale_rad": cfg.environment.policy_action_scale_rad,
        "command_resolution_rad": env._command_resolution_rad,
        "reference_lookup_commands_physical": grid[indices].tolist(),
        "physics_substeps": env._n_frames, "physics_timestep_s": float(env.sys.opt.timestep),
    }


def load_tb(run_dir):
    import jax.numpy as jp
    from brax.io import model
    from brax.training.agents.ppo import networks
    from toddlerbot.locomotion.evaluate_xy import XYEvaluator
    from toddlerbot.locomotion.mjx_config import MJXConfig
    from toddlerbot.locomotion.train_mjx import domain_randomize, get_body_mass_attr_range
    from toddlerbot.locomotion.walk_env import WalkEnv
    from toddlerbot.sim.robot import Robot
    from wr2.tools.diagnose_ppo import verify_tb_source

    source = verify_tb_source(run_dir)
    saved = json.loads((run_dir / "env_config.json").read_text())
    train = json.loads((run_dir / "train_config.json").read_text())
    cfg = MJXConfig()
    for section, values in saved.items():
        for name, value in values.items():
            setattr(getattr(cfg, section), name, value)
    # Match WR2's controlled no-push evaluation, independent of the native push bug.
    cfg.domain_rand.push_lin_vel = cfg.domain_rand.push_ang_vel = 0.
    robot = Robot("toddlerbot")
    print("Loading verified TB Paper model and saved reference...", flush=True)
    env = WalkEnv("walk", robot, cfg, add_noise=True, add_domain_rand=True)
    net = networks.make_ppo_networks(
        env.obs_size * env.num_obs_history,
        env.privileged_obs_size * env.num_privileged_obs_history, env.action_size,
        policy_hidden_layer_sizes=train["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=train["value_hidden_layer_sizes"],
    )
    native_policy = networks.make_inference_fn(net)

    class NominalWalkEnv(WalkEnv):
        def reset(self, rng, command=None):
            state = super().reset(rng, command)
            info = dict(state.info)
            info["state_ref"] = info["state_ref"].at[3:7].set(jp.asarray([1., 0., 0., 0.]))
            qpos = self.motion_ref.get_qpos_ref(info["state_ref"], path_frame=False)
            pipeline = self.pipeline_init(qpos, jp.zeros(self.nv))
            info["feet_height_init"] = pipeline.x.pos[self.feet_link_ids, 2]
            obs, privileged = self._get_obs(
                pipeline, info, jp.zeros_like(state.obs), jp.zeros_like(state.privileged_obs),
            )
            return state.replace(pipeline_state=pipeline, info=info, obs=obs,
                                 privileged_obs=privileged)

    def evaluator(varied, seed, count):
        if varied:
            active_env = env
            mass = get_body_mass_attr_range(
                robot, cfg.domain_rand.body_mass_range, cfg.domain_rand.ee_mass_range,
                cfg.domain_rand.other_mass_range, count, rng=np.random.default_rng(seed),
            )
            randomize = partial(domain_randomize, friction_range=cfg.domain_rand.friction_range,
                                damping_range=cfg.domain_rand.damping_range,
                                armature_range=cfg.domain_rand.armature_range,
                                frictionloss_range=cfg.domain_rand.frictionloss_range,
                                body_mass_attr_range=mass)
        else:
            # Existing frozen diagnostic rebuilds native reference reset at zero yaw.
            nominal_cfg = MJXConfig()
            for section, values in saved.items():
                for name, value in values.items():
                    setattr(getattr(nominal_cfg, section), name, value)
            nominal_cfg.noise.action_noise = 0.
            nominal_cfg.domain_rand.push_lin_vel = nominal_cfg.domain_rand.push_ang_vel = 0.
            active_env = NominalWalkEnv("walk", robot, nominal_cfg, add_noise=False, add_domain_rand=False)
            randomize = None
        wrapped = XYEvaluator(active_env, native_policy, count, 1000, randomize, seed=seed)

        def sample(before, after):
            return (jp.stack([after.metrics[name] * env.dt for name in TERMS], axis=1),
                    after.done > 0, before.info["command"][:, 5:8])

        return wrapped.env, wrapped.reset_keys, lambda p: native_policy(p, deterministic=True), sample

    return {
        "scale": 1., "dt": float(env.dt), "root": 0,
        "load": lambda step: model.load_params(run_dir / str(step) / "policy"),
        "evaluator": evaluator,
        "command": lambda v, n: jp.zeros((n, 8)).at[:, 5].set(v),
        "configuration": saved, "training_provenance": source,
        "physics_substeps": env._n_frames, "physics_timestep_s": float(env.sys.opt.timestep),
        "nominal_reset": "Native reference pose at zero yaw, action reset bias disabled; same reset as frozen StanceEnv diagnostic",
    }


def run(args):
    sys.dont_write_bytecode = True
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    if args.tb_root:
        tb_root = args.tb_root.resolve()
        os.environ["USE_JAX"] = "true"
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")
        os.chdir(tb_root)
        sys.path[:0] = [str(tb_root), str(tb_root / "toddlerbot/brax")]
    from wr2.locomotion.train import _configure_backend_logging, _filter_optional_backend_import_messages
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        if jax.default_backend() != "gpu" and not args.allow_cpu:
            raise RuntimeError("Use --allow-cpu for an intentional Mac evaluation")
        jax.config.update("jax_default_matmul_precision", "high")
        context = load_tb(run_dir) if args.tb_root else load_wr2(run_dir)
    steps = round(1000 * context["scale"])
    params = {step: context["load"](step) for step in args.checkpoint_steps}
    if any(not np.isfinite(np.asarray(value)).all() for p in params.values() for value in jax.tree.leaves(p)):
        raise ValueError("Checkpoint contains nonfinite parameters")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 2, "complete": False, "robot": "toddlerbot" if args.tb_root else "wr2",
        "run_dir": str(run_dir), "arguments": vars(args) | {"run_dir": str(run_dir), "output": str(output), "tb_root": str(args.tb_root) if args.tb_root else None},
        "commands_TB": list(COMMANDS), "episode_steps": steps,
        "normalization_scale": context["scale"], "control_period_s": context["dt"],
        "physical_pushes": False, "deterministic_policy": True,
        "nominal_episode_count_per_command": 1, "randomized_episodes_per_seed": args.num_envs,
        "measurement_scope": "Integrated root pose for path/heading; cached native torso frame for reward signals. Finite terminal samples retained; stop after fall/nonfinite. Full and 1.5 TB-second warmup-excluded windows.",
        "heading_scope": "Unwrapped yaw relative to initial desired heading, fixed forward command with zero yaw",
        "distance_units": "physical robot metres; _TB_m divides by normalization_scale^2",
        "evaluation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "evaluation_source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "evaluation_source_sha256": sha(__file__),
        "measurement_source_sha256": sha(root / "wr2/locomotion/heading_evaluation.py"),
        "checkpoint_hashes": {
            str(step): checkpoint_hash(
                run_dir / str(step) / "policy" if args.tb_root
                else run_dir / "checkpoints" / f"{step:012d}"
            ) for step in args.checkpoint_steps
        },
        "runtime": {"backend": jax.default_backend(), "python": sys.version,
                    "devices": [str(d) for d in jax.devices()], "matmul_precision": str(jax.config.jax_default_matmul_precision),
                    "packages": {name: metadata.version(name) for name in ["jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "flax", "numpy"]}},
        "context": {key: value for key, value in context.items() if not callable(value)},
        "results": [],
    }
    for source in [Path(__file__), root / "wr2/locomotion/heading_evaluation.py"]:
        (output / source.name).write_bytes(source.read_bytes())
    for profile, varied, seeds, count in [
        ("nominal", False, args.seeds[:1], 1),
        ("randomized_no_push", True, args.seeds, args.num_envs),
    ]:
        for seed in seeds:
            print(f"Preparing {profile}, seed={seed}, episodes={count}...", flush=True)
            env, keys, make_policy, sample = context["evaluator"](varied, seed, count)
            evaluate = jax.jit(lambda p, command: heading_rollout(
                env, make_policy(p), keys, command, steps, context["root"], sample,
            ))
            for step, p in params.items():
                for command in COMMANDS:
                    started = time.monotonic()
                    raw = jax.device_get(evaluate(p, context["command"](command, count)))
                    trace = {name: np.asarray(value) for name, value in raw.items()}
                    trace_name = f"{step}_{profile}_seed{seed}_forward{round(command*100):03d}.npz"
                    np.savez_compressed(output / trace_name, **trace)
                    summary = summarize_heading(trace, context["dt"], context["scale"])
                    row = {"checkpoint_step": step, "profile": profile, "seed": seed,
                           "command_TB_m_s": command, "trace": trace_name, **summary}
                    report["results"].append(row)
                    (output / "summary.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
                    g = summary["groups"]["post_warmup"]
                    print(f"{step} {profile} cmd={command:.2f}: falls={summary['fall_count']}/{count}, "
                          f"body_v={g.get('body_forward_TB_m_s', {}).get('mean')}, "
                          f"heading_abs={g.get('abs_net_heading_change_deg', {}).get('mean')}; "
                          f"{time.monotonic()-started:.1f}s", flush=True)
    report["complete"] = True
    (output / "summary.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
    print(f"Saved heading evaluation to {output}", flush=True)


if __name__ == "__main__":
    run(parse_args())
