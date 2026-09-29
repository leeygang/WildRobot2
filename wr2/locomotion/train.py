"""Smoke-test or train WR2's initial Brax PPO walking environment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def smoke_test(steps: int) -> None:
    import jax
    import jax.numpy as jp

    from wr2.locomotion.walking_env import WR2WalkingEnv

    environment = WR2WalkingEnv(add_observation_noise=False)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    state = reset(jax.random.PRNGKey(0))
    jax.block_until_ready(state.obs)
    started = time.monotonic()
    for _ in range(steps):
        state = step(state, jp.zeros(environment.action_size))
    jax.block_until_ready(state.obs)
    elapsed = time.monotonic() - started
    print(
        f"MJX smoke passed: steps={steps}, obs={state.obs.shape}, "
        f"action={environment.action_size}, reward={float(state.reward):.6f}, "
        f"done={float(state.done):.0f}, root_z={float(state.pipeline_state.q[2]):.6f}, "
        f"elapsed={elapsed:.3f}s"
    )


def train(args: argparse.Namespace) -> None:
    import jax
    from brax.io import model
    from brax.training.agents.ppo import train as ppo

    from wr2.locomotion.config import WalkingEnvConfig
    from wr2.locomotion.domain_randomization import make_domain_randomizer
    from wr2.locomotion.walking_env import WR2WalkingEnv

    environment = WR2WalkingEnv(
        WalkingEnvConfig(episode_length=args.episode_length),
        add_observation_noise=True,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def progress(step: int, metrics) -> None:
        reward = metrics.get(
            "eval/episode_reward", metrics.get("eval/episode_reward_mean")
        )
        print(f"step={step} eval_reward={reward}")

    randomization_fn = (
        None
        if args.no_domain_randomization
        else make_domain_randomizer(environment.config.randomization)
    )
    make_inference_fn, params, _ = ppo.train(
        environment=environment,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        episode_length=environment.config.episode_length,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        randomization_fn=randomization_fn,
        seed=args.seed,
        progress_fn=progress,
    )
    del make_inference_fn
    jax.block_until_ready(params)
    model.save_params(output / "params", params)
    print(f"Saved parameters to {output / 'params'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run JIT reset/step only")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--num-timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/wr2_walking"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        smoke_test(args.smoke_steps)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
