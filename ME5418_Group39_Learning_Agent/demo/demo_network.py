"""Demonstrate the 23-D mission network and one real CPU PPO update.

Run: python -m demo.demo_network --steps 128 --updates 1
This smoke demonstration is not a claim of a converged flight policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import AerialSprayingEnv
from pure_rl.agent import PurePPO, PPOConfig
from pure_rl.collector import BatchedCollector
from pure_rl.learning import update
from pure_rl.task import LearningTask, TaskOptions


def run(seed=7, steps=128, updates=1, output="outputs/network_demo.json"):
    if updates < 1:
        raise ValueError("At least one update is required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = PPOConfig(
        total_timesteps=steps,
        n_envs=1,
        rollout_steps=steps,
        batch_size=min(64, steps),
        epochs=1,
        eval_every=1,
        eval_episodes=1,
    )
    agent = PurePPO(config)
    model = agent.model
    env = LearningTask(AerialSprayingEnv(), TaskOptions(curriculum=False))
    try:
        observation, _ = env.reset(seed=seed)
        with torch.no_grad():
            mean, std, value = model(torch.tensor(observation))
            action = mean.tanh()
        summary = {
            "seed": seed,
            "device": "cpu",
            "network": "mission_actor_critic_23d",
            "observation_dimensions": int(observation.shape[0]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_observation": observation.tolist(),
            "latent_mean": mean.tolist(),
            "latent_std": std.tolist(),
            "deterministic_action": action.tolist(),
            "value": value.item(),
            "updates": [],
        }
        for index in range(updates):
            collector = BatchedCollector([env], model, seed + index)
            batch, episodes = collector.collect(steps, config.gamma, config.gae_lambda)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            metrics = update(agent, batch, 0.0)
            changes = {
                name: float((p.detach() - before[name]).abs().max())
                for name, p in model.named_parameters()
            }
            actor_gradients = [p.grad for p in list(model.actor.parameters()) + [model.log_std]]
            critic_gradients = [p.grad for p in model.critic.parameters()]
            assert all(
                g is not None and torch.isfinite(g).all()
                for g in actor_gradients + critic_gradients
            )
            assert max(changes.values()) > 0, "Optimizer failed to change parameters"
            metrics.update(
                {
                    "transitions": steps,
                    "episode_end_reasons": episodes,
                    "actor_gradient_norm": float(
                        torch.linalg.vector_norm(
                            torch.stack([g.detach().norm() for g in actor_gradients])
                        )
                    ),
                    "critic_gradient_norm": float(
                        torch.linalg.vector_norm(
                            torch.stack([g.detach().norm() for g in critic_gradients])
                        )
                    ),
                    "max_parameter_change": max(changes.values()),
                    "actor_changed": any(
                        v > 0 for k, v in changes.items() if k.startswith("actor.")
                    ),
                    "critic_changed": any(
                        v > 0 for k, v in changes.items() if k.startswith("critic.")
                    ),
                }
            )
            summary["updates"].append(metrics)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        torch.save(model.state_dict(), path.with_suffix(".pt"))
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--output", default="outputs/network_demo.json")
    args = parser.parse_args()
    run(args.seed, args.steps, args.updates, args.output)
