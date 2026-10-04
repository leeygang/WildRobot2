"""ToddlerBot-aligned RSL-RL training adapter for WR2's MJX environment."""

from __future__ import annotations

import collections
import math
import re
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np


def learning_iterations(num_timesteps: int, num_envs: int, unroll_length: int) -> int:
    """Return the number of complete RSL-RL rollout/update iterations."""
    return max(1, math.ceil(num_timesteps / (num_envs * unroll_length)))


def evaluation_iterations(total_iterations: int, num_evals: int) -> frozenset[int]:
    """Place post-update evaluations evenly; evaluation zero happens separately."""
    if num_evals <= 1:
        return frozenset({total_iterations})
    return frozenset(
        math.ceil(index * total_iterations / (num_evals - 1))
        for index in range(1, num_evals)
    )


class RSLRLWrapper:
    """Adapt a vectorized Brax/MJX environment to RSL-RL's VecEnv contract."""

    def __init__(
        self,
        env,
        *,
        device,
        num_envs: int,
        episode_length: int,
        seed: int,
    ) -> None:
        import jax
        import torch
        from brax.io.torch import jax_to_torch, torch_to_jax

        self.env = env
        self.device = device
        self.num_envs = num_envs
        self.num_actions = env.action_size
        self.max_episode_length = episode_length
        self.episode_length_buf = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._jax_to_torch = jax_to_torch
        self._torch_to_jax = torch_to_jax
        self._keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._reset_fn = jax.jit(env.reset)
        self._step_fn = jax.jit(env.step)
        self.last_state = self.reset()

    def reset(self):
        """Reset every parallel environment."""
        self.last_state = self._reset_fn(self._keys)
        return self.last_state

    def get_observations(self):
        """Return actor and asymmetric-critic observations as Torch tensors."""
        actor = self._jax_to_torch(self.last_state.obs["state"], device=self.device)
        critic = self._jax_to_torch(
            self.last_state.obs["privileged_state"], device=self.device
        )
        return actor, {"observations": {"critic": critic}}

    def step(self, actions):
        """Step MJX and preserve Brax's timeout/episode metric semantics."""
        actions_jax = self._torch_to_jax(actions.contiguous())
        state = self._step_fn(self.last_state, actions_jax)
        values = self._jax_to_torch(
            {
                "obs": state.obs["state"],
                "critic": state.obs["privileged_state"],
                "reward": state.reward,
                "done": state.done,
                "episode": state.info["episode_metrics"],
                "episode_done": state.info["episode_done"],
                "truncation": state.info["truncation"],
            },
            device=self.device,
        )
        done = values["done"]
        self.episode_length_buf.add_(1)
        self.episode_length_buf[done.bool()] = 0
        self.last_state = state
        return (
            values["obs"],
            values["reward"],
            done,
            {
                "observations": {"critic": values["critic"]},
                "episode": values["episode"],
                "dones": values["episode_done"],
                "time_outs": values["truncation"],
            },
        )


class _EpisodeMetrics:
    """Accumulate accurate means from completed, variable-length episodes."""

    def __init__(self, max_episodes: int = 100) -> None:
        self.values = collections.defaultdict(
            lambda: collections.deque(maxlen=max_episodes)
        )

    def update(self, metrics: dict[str, Any], dones) -> None:
        mask = dones.bool()
        if not bool(mask.any()):
            return
        lengths = metrics["length"][mask].flatten().clamp_min(1.0)
        for name, value in metrics.items():
            completed = value[mask].flatten()
            if name.endswith("_per_step"):
                completed = completed / lengths
            self.values[name].extend(completed.detach().cpu().tolist())

    def means(self) -> dict[str, float]:
        return {
            name: float(np.mean(values))
            for name, values in self.values.items()
            if values
        }


def _actor_parameters_for_jax(policy) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Copy RSL-RL's deterministic actor MLP into a JAX-friendly pytree."""
    import torch

    if not any(isinstance(module, torch.nn.Linear) for module in policy.actor):
        raise ValueError("RSL-RL actor has no linear layers")
    return _actor_parameters_from_state_dict(policy.state_dict())


def _actor_parameters_from_state_dict(
    state_dict: dict[str, Any],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Extract ordered actor linear layers from an RSL-RL state dictionary."""
    layer_indices = sorted(
        int(match.group(1))
        for name in state_dict
        if (match := re.fullmatch(r"actor\.(\d+)\.weight", name))
    )
    if not layer_indices:
        raise ValueError("RSL-RL checkpoint has no actor linear layers")
    return tuple(
        (
            state_dict[f"actor.{index}.weight"].detach().cpu().numpy().copy(),
            state_dict[f"actor.{index}.bias"].detach().cpu().numpy().copy(),
        )
        for index in layer_indices
    )


def load_rsl_actor_parameters(
    checkpoint_path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Load only deterministic actor parameters from a WR2 RSL-RL checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("backend", "rsl_rl") != "rsl_rl":
        raise ValueError(f"Not an RSL-RL checkpoint: {checkpoint_path}")
    return _actor_parameters_from_state_dict(checkpoint["model_state_dict"])


def _jax_policy_factory(activation: str):
    """Create the deterministic evaluator used by Brax's metric pipeline."""
    import jax.nn as jnn
    import jax.numpy as jnp

    activations = {
        "elu": jnn.elu,
        "relu": jnn.relu,
        "silu": jnn.silu,
        "tanh": jnp.tanh,
    }
    activate = activations[activation]

    def make_policy(parameters):
        def policy(observation, _key):
            value = observation["state"]
            for index, (weight, bias) in enumerate(parameters):
                value = value @ weight.T + bias
                if index + 1 < len(parameters):
                    value = activate(value)
            return value, {}

        return policy

    return make_policy


class RSLPPOTrainer:
    """Small PPO-only runner matching ToddlerBot's active RSL-RL settings."""

    def __init__(
        self,
        env: RSLRLWrapper,
        training_config,
        *,
        output: Path,
        restore_checkpoint: Path | None,
        progress_fn: Callable[[int, dict[str, Any]], None],
        policy_params_fn: Callable[[int, Any, Any], None],
        evaluator,
    ) -> None:
        import torch
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import ActorCritic

        self.torch = torch
        self.env = env
        self.config = training_config
        self.output = output
        self.progress_fn = progress_fn
        self.policy_params_fn = policy_params_fn
        self.evaluator = evaluator
        ppo = training_config.ppo
        network = training_config.network
        actor_obs, extras = env.get_observations()
        critic_obs = extras["observations"]["critic"]
        self.policy = ActorCritic(
            actor_obs.shape[-1],
            critic_obs.shape[-1],
            env.num_actions,
            actor_hidden_dims=list(network.policy_hidden_layer_sizes),
            critic_hidden_dims=list(network.value_hidden_layer_sizes),
            activation=network.activation,
            init_noise_std=network.init_noise_std,
            noise_std_type=network.noise_std_type,
        ).to(env.device)
        self.algorithm = PPO(
            self.policy,
            num_learning_epochs=ppo.num_updates_per_batch,
            num_mini_batches=ppo.num_minibatches,
            clip_param=ppo.clipping_epsilon,
            gamma=ppo.discounting,
            lam=ppo.gae_lambda,
            value_loss_coef=ppo.value_loss_coef,
            entropy_coef=ppo.entropy_cost,
            learning_rate=ppo.learning_rate,
            max_grad_norm=ppo.max_grad_norm,
            use_clipped_value_loss=True,
            schedule=ppo.learning_rate_schedule,
            desired_kl=ppo.desired_kl,
            device=env.device,
            normalize_advantage_per_mini_batch=not ppo.normalize_advantage,
        )
        self.algorithm.init_storage(
            "rl",
            env.num_envs,
            ppo.unroll_length,
            [actor_obs.shape[-1]],
            [critic_obs.shape[-1]],
            [env.num_actions],
        )
        self.completed_iterations = 0
        self.total_steps = 0
        self._run_steps = 0
        self._started = time.monotonic()
        self._last_metrics_step = 0
        self._episode_metrics = _EpisodeMetrics()
        if restore_checkpoint is not None:
            self.load(restore_checkpoint)

    def checkpoint_state(self) -> dict[str, Any]:
        """Create a portable full-state checkpoint, following ToddlerBot."""
        state = {
            "backend": "rsl_rl",
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "iteration": self.completed_iterations,
            "total_steps": self.total_steps,
            "learning_rate": self.algorithm.learning_rate,
            "torch_rng_state": self.torch.get_rng_state(),
        }
        if self.torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = self.torch.cuda.get_rng_state_all()
        return state

    def save(self, path: Path) -> None:
        """Save policy, optimizer, counters, adaptive LR, and Torch RNG state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.checkpoint_state(), path)

    def load(self, path: Path) -> None:
        """Resume all state represented by ToddlerBot's RSL checkpoint contract."""
        checkpoint = self.torch.load(
            path, map_location=self.env.device, weights_only=False
        )
        if checkpoint.get("backend", "rsl_rl") != "rsl_rl":
            raise ValueError(f"Not an RSL-RL checkpoint: {path}")
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.algorithm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.completed_iterations = int(
            checkpoint.get("iteration", checkpoint.get("iter", 0))
        )
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.algorithm.learning_rate = float(
            checkpoint.get(
                "learning_rate",
                self.algorithm.optimizer.param_groups[0]["lr"],
            )
        )
        for group in self.algorithm.optimizer.param_groups:
            group["lr"] = self.algorithm.learning_rate
        if "torch_rng_state" in checkpoint:
            self.torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.torch.cuda.is_available() and "torch_cuda_rng_state_all" in checkpoint:
            self.torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_rng_state_all"]]
            )

    def _training_metrics(self, losses: dict[str, float]) -> dict[str, float]:
        elapsed = max(time.monotonic() - self._started, 1e-9)
        policy_loss = float(losses.get("surrogate", 0.0))
        value_loss = float(losses.get("value_function", 0.0))
        entropy = float(losses.get("entropy", 0.0))
        entropy_loss = -self.config.ppo.entropy_cost * entropy
        metrics = {
            f"episode/{name}": value
            for name, value in self._episode_metrics.means().items()
        }
        metrics.update(
            {
                "training/sps": self._run_steps / elapsed,
                "training/policy_loss": policy_loss,
                "training/v_loss": value_loss,
                "training/entropy_loss": entropy_loss,
                "training/total_loss": (
                    policy_loss
                    + self.config.ppo.value_loss_coef * value_loss
                    + entropy_loss
                ),
                "training/learning_rate": float(self.algorithm.learning_rate),
            }
        )
        return metrics

    def _evaluate(self, run_step: int, losses: dict[str, float]) -> None:
        snapshot = self.checkpoint_state()
        self.policy_params_fn(run_step, None, snapshot)
        metrics = self.evaluator.run_evaluation(
            _actor_parameters_for_jax(self.policy), self._training_metrics(losses)
        )
        self.progress_fn(run_step, metrics)

    def learn(self) -> Path:
        """Run the requested additional timesteps and return the final checkpoint."""
        ppo = self.config.ppo
        iterations = learning_iterations(
            ppo.num_timesteps, ppo.num_envs, ppo.unroll_length
        )
        eval_at = evaluation_iterations(iterations, ppo.num_evals)
        steps_per_iteration = ppo.num_envs * ppo.unroll_length
        checkpoints = self.output / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        losses: dict[str, float] = {}

        self._evaluate(0, losses)
        if self.config.checkpoints.save_every_evaluation:
            self.save(checkpoints / "000000000000.pt")
        actor_obs, extras = self.env.get_observations()
        critic_obs = extras["observations"]["critic"]
        self.policy.train()
        for run_iteration in range(1, iterations + 1):
            with self.torch.inference_mode():
                for _ in range(ppo.unroll_length):
                    actions = self.algorithm.act(actor_obs, critic_obs)
                    actor_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = infos["observations"]["critic"]
                    self.algorithm.process_env_step(rewards, dones, infos)
                    self._episode_metrics.update(infos["episode"], infos["dones"])
                self.algorithm.compute_returns(critic_obs)

            losses = self.algorithm.update()
            self.completed_iterations += 1
            self.total_steps += steps_per_iteration
            self._run_steps += steps_per_iteration
            if (
                self._run_steps - self._last_metrics_step
                >= ppo.training_metrics_steps
            ):
                self.progress_fn(self._run_steps, self._training_metrics(losses))
                self._last_metrics_step = self._run_steps

            if run_iteration in eval_at:
                self._evaluate(self._run_steps, losses)
                if self.config.checkpoints.save_every_evaluation:
                    self.save(checkpoints / f"{self._run_steps:012d}.pt")

        final_path = self.output / "params.pt"
        self.save(final_path)
        return final_path


def train_rsl_ppo(
    environment,
    evaluation_environment,
    training_config,
    *,
    output: Path,
    restore_checkpoint: Path | None,
    progress_fn: Callable[[int, dict[str, Any]], None],
    policy_params_fn: Callable[[int, Any, Any], None],
    randomization_fn=None,
    allow_cpu: bool = False,
) -> Path:
    """Build the MJX↔Torch bridge, evaluator, and RSL-RL PPO runner."""
    import jax
    import torch
    from brax.envs import training as env_training
    from brax.training import acting

    ppo = training_config.ppo
    if training_config.network.distribution_type != "normal":
        raise ValueError("RSL-RL backend requires network.distribution_type=normal")
    if ppo.normalize_observations:
        raise ValueError("RSL-RL observation normalization is not implemented")
    if ppo.num_envs != ppo.batch_size * ppo.num_minibatches:
        raise ValueError(
            "RSL-RL requires ppo.num_envs == ppo.batch_size * "
            "ppo.num_minibatches, matching ToddlerBot"
        )
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "RSL-RL requires a Torch CUDA device; pass --allow-cpu only for a "
            "deliberate CPU test"
        )
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    print(f"  Torch device: {device} ({device_name})")

    vector_randomizer = None
    if randomization_fn is not None:
        randomization_rng = jax.random.split(
            jax.random.PRNGKey(training_config.seed), ppo.num_envs
        )
        vector_randomizer = partial(randomization_fn, rng=randomization_rng)
    wrapped_environment = env_training.wrap(
        environment,
        episode_length=training_config.environment.episode_length,
        randomization_fn=vector_randomizer,
    )
    wrapped_evaluation = env_training.wrap(
        evaluation_environment,
        episode_length=training_config.environment.episode_length,
    )
    evaluator = acting.Evaluator(
        wrapped_evaluation,
        _jax_policy_factory(training_config.network.activation),
        num_eval_envs=ppo.num_eval_envs,
        episode_length=training_config.environment.episode_length,
        action_repeat=1,
        key=jax.random.PRNGKey(training_config.seed + 1),
    )
    rsl_environment = RSLRLWrapper(
        wrapped_environment,
        device=device,
        num_envs=ppo.num_envs,
        episode_length=training_config.environment.episode_length,
        seed=training_config.seed,
    )
    trainer = RSLPPOTrainer(
        rsl_environment,
        training_config,
        output=output,
        restore_checkpoint=restore_checkpoint,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        evaluator=evaluator,
    )
    print(
        "Starting RSL-RL PPO; the first MJX/Torch compile and evaluation may "
        "take several minutes."
    )
    final_path = trainer.learn()
    print(f"Saved full RSL-RL state to {final_path}")
    return final_path
