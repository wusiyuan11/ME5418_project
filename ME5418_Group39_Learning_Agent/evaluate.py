"""Evaluate saved policies on held-out seeds, optionally beside the inherited PID."""
import argparse
from pathlib import Path
import numpy as np

from demo.demo_pid import FixedGainController
from environment import EnvironmentConfig
from rl.agent import PPOAgent, TrainingConfig, plain_dict
from rl.evaluation import evaluate_policy, summarize, write_json, write_rows


class PIDPolicy:
    """Wrap the unchanged stage-2 PID and reset its integral for each flight."""
    def __init__(self, dt):
        self.dt = dt

    def reset(self):
        self.controller = FixedGainController(self.dt)

    def __call__(self, observation, env):
        return self.controller.action(observation, env.reference_velocity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--controller', choices=['ppo', 'pid', 'untrained', 'all'], default='ppo')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=20000)
    parser.add_argument('--policy-seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('results/heldout'))
    parser.add_argument('--animation', action='store_true', help='First-seed replay GIF for each controller')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error('episodes must be positive and seed-start nonnegative')
    if args.controller in ('ppo', 'all') and args.checkpoint is None:
        parser.error('--checkpoint is required for ppo/all')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    checkpoint = None
    if args.checkpoint:
        agent, env_config, checkpoint = PPOAgent.load(args.checkpoint)
        seeds = set(range(args.seed_start, args.seed_start + args.episodes))
        cfg = agent.config
        if seeds & set(range(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes)):
            parser.error('Held-out seeds overlap model-selection validation seeds')
    else:
        env_config = EnvironmentConfig()
        agent = None
    # A freshly initialized agent is a useful learning sanity check, not a learned baseline.
    fresh_seed = checkpoint['training_config']['seed'] if checkpoint else args.policy_seed
    untrained = PPOAgent(TrainingConfig(seed=fresh_seed))
    policies = {'untrained': lambda obs, _: untrained.act(obs),
                'pid': PIDPolicy(env_config.dt)}
    if agent is not None:
        policies['ppo'] = lambda obs, _: agent.act(obs)
    controllers = ['untrained', 'ppo', 'pid'] if args.controller == 'all' else [args.controller]
    summaries, rows = {}, []
    for name in controllers:
        records = evaluate_policy(policies[name], env_config,
            range(args.seed_start, args.seed_start + args.episodes), name, args.output, args.animation)
        rows.extend(records)
        summaries[name] = summarize(records)
        item = summaries[name]
        print(f'{name}: return={item["mean"]["return"]:.3f} +/- {item["std"]["return"]:.3f}, '
              f'success={item["rates"]["success"]:.1%}, '
              f'timed-reference RMSE={item["mean"]["tracking_rmse_m"]:.3f}m', flush=True)
    write_rows(args.output / 'episodes.csv', rows)
    write_json(args.output / 'summary.json', {'controllers': summaries,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'checkpoint_timesteps': checkpoint['total_timesteps'] if checkpoint else None,
        'environment': plain_dict(env_config), 'deterministic_policy': True,
        'fairness_note': 'Paired environment seeds; inherited PID also receives reference velocity, '
                         'which PPO does not observe. This is not an information-matched robustness benchmark.',
        'metric_note': 'Tracking error is to the timed 3-D reference. Cross-track is nearest full '
                       'polyline distance and does not prove coverage. Action squared is not physical energy.'})
    from visualization.learning_curves import plot_evaluation
    plot_evaluation(summaries, args.output / 'comparison.png')


if __name__ == '__main__':
    main()
