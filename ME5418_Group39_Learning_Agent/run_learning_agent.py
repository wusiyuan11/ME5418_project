"""Single configured entry point for the third-milestone PPO training framework.

This file selects settings only; it does not replace PPO or supply expert actions.
Run from the submission root. Use a NEW output directory for every experiment.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from pure_rl.agent import PPOConfig
from pure_rl.task import TaskOptions
from pure_rl.train import train


def load_configuration(path):
    """Reject unknown fields and inconsistent policy/task interfaces early."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(data) != {'description', 'ppo', 'task'}:
        raise ValueError('Configuration requires description, ppo and task only')
    policy = PPOConfig(**data['ppo'])
    task = TaskOptions(**data['task'])
    policy.validate()
    task.validate()
    if (policy.features, policy.action_repeat) != (task.features, task.action_repeat):
        raise ValueError('Observation features and action repeat must match')
    initial_training_seeds = set(range(policy.seed * 1000, policy.seed * 1000 + policy.n_envs))
    # These are explicit initial reset streams, not demonstrations. Validation
    # uses a separate seed range and always runs the unassisted full mission.
    validation_seeds = set(range(policy.eval_seed, policy.eval_seed + policy.eval_episodes))
    if initial_training_seeds & validation_seeds:
        raise ValueError('Training and validation reset seeds must be disjoint')
    return policy, task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'configs/mission_ppo.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm-start-pure', type=Path,
                        help='Optional inherited 15-input PPO checkpoint; no imitation weights')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--total-timesteps', type=int,
                        help='POLICY DECISIONS, not physics steps; divisible by eight by default')
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    args = parser.parse_args()
    try:
        policy, task = load_configuration(args.config)
        overrides = {name:getattr(args, name) for name in
                     ('seed', 'total_timesteps', 'eval_every', 'eval_episodes')
                     if getattr(args, name) is not None}
        policy = replace(policy, **overrides)
        policy.validate()
        training_seeds = set(range(policy.seed * 1000, policy.seed * 1000 + policy.n_envs))
        if training_seeds & set(range(policy.eval_seed, policy.eval_seed + policy.eval_episodes)):
            raise ValueError('Training and validation reset seeds must be disjoint')
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    result = train(policy, task, args.output, args.warm_start_pure)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
