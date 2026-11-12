"""Train the stage-2 neural controller through repeated on-policy interaction."""
import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import time

from environment import AerialSprayingEnv, EnvironmentConfig
from rl.agent import PPOAgent, TrainingConfig, plain_dict, runtime_versions
from rl.evaluation import evaluate_policy, summarize, write_json
from rl.rollout import RolloutCollector


def append_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def train(config, output, env_config=None, resume=None):
    """Run learning and validation; the returned counters are measured steps.

    Validation seeds select best.pt. Held-out assessment belongs in evaluate.py.
    A resumed checkpoint restores model and Adam; the simulator starts a new
    episode because a partial PyBullet flight is not in the checkpoint.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}. Choose a new run directory.')
    config.validate()
    if resume:
        agent, saved_env, checkpoint = PPOAgent.load(resume)
        old_config = agent.config
        # Changing algorithm hyperparameters during resume is intentionally explicit.
        for item in fields(TrainingConfig):
            if item.name not in ('total_timesteps',) and getattr(config, item.name) != getattr(old_config, item.name):
                raise ValueError(f'Resume requires the saved {item.name}; only total_timesteps may change')
        if env_config is not None and plain_dict(env_config) != plain_dict(saved_env):
            raise ValueError('Resume requires the saved environment configuration')
        env_config = saved_env
        agent.config = config
        total, update = checkpoint['total_timesteps'], checkpoint['update']
    else:
        agent = PPOAgent(config)
        env_config = env_config or EnvironmentConfig()
        total, update = 0, 0
    if config.total_timesteps - total < 2:
        raise ValueError('The target must exceed the checkpoint by at least two timesteps')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', {'training': plain_dict(config),
        'environment': plain_dict(env_config), 'runtime': runtime_versions(),
        'resume_from': str(resume) if resume else None, 'starting_timesteps': total,
        'baseline': '2026-10-30 Group39 stage-2 environment and ActorCritic',
        'resume_note': 'Fresh simulator episode; partial old episodes are not continued.'})
    agent.save(output / 'initial.pt', env_config, total, update)
    best_return, best_step = float('-inf'), total
    validations = []

    def validate(stage):
        nonlocal best_return, best_step
        records = evaluate_policy(lambda obs, _: agent.act(obs), env_config,
            range(config.eval_seed, config.eval_seed + config.eval_episodes))
        summary = summarize(records)
        row = {'total_timesteps': total, 'stage': stage,
               'mean_return': summary['mean']['return'], 'std_return': summary['std']['return'],
               'success_rate': summary['rates']['success'],
               'tracking_rmse_m': summary['mean']['tracking_rmse_m']}
        validations.append(row)
        append_rows(output / 'validation.csv', [row])
        write_json(output / 'validation' / f'step_{total:09d}.json',
                   {'summary': summary, 'episodes': records})
        if row['mean_return'] > best_return:
            best_return, best_step = row['mean_return'], total
            agent.save(output / 'best.pt', env_config, total, update)
        print(f'validation step={total} return={row["mean_return"]:.3f} '
              f'success={row["success_rate"]:.1%} rmse={row["tracking_rmse_m"]:.3f}m', flush=True)

    start = time.perf_counter()
    validate('start')
    env = AerialSprayingEnv(env_config)
    collector = RolloutCollector(env, agent.model, seed=config.seed + total,
                                 timestep_offset=total)
    all_episodes = []
    try:
        while total < config.total_timesteps:
            remaining = config.total_timesteps - total
            count = min(config.n_steps, remaining)
            # Avoid a single-transition normalized-advantage minibatch at the tail.
            if remaining - count == 1:
                count = remaining if count == 2 else count - 1
            batch, episodes = collector.collect(count, config.gamma, config.gae_lambda)
            diagnostics = agent.update(batch)
            total = collector.total_timesteps
            update += 1
            all_episodes.extend(episodes)
            append_rows(output / 'episodes.csv', episodes)
            row = {'update': update, 'total_timesteps': total, 'rollout_steps': count,
                   'rollout_mean_step_reward': float(batch['rewards'].mean()),
                   'completed_episodes': len(episodes), **diagnostics,
                   'elapsed_seconds': time.perf_counter() - start}
            append_rows(output / 'updates.csv', [row])
            print(f'update={update} steps={total}/{config.total_timesteps} '
                  f'episodes={len(all_episodes)} loss={row["loss"]:.3f} '
                  f'kl={row["approx_kl"]:.5f}', flush=True)
            if update % config.eval_every == 0 or total == config.total_timesteps:
                validate('during_training')
                agent.save(output / 'latest.pt', env_config, total, update)
    except KeyboardInterrupt:
        agent.save(output / 'interrupted.pt', env_config, total, update)
        print('Interrupted. Saved current parameters; unfinished rollout is not counted.', flush=True)
        raise
    finally:
        env.close()
    agent.save(output / 'final.pt', env_config, total, update)
    summary = {'total_timesteps': total, 'updates': update,
               'completed_training_episodes': len(all_episodes),
               'unfinished_episode_steps': collector.length,
               'best_validation_step': best_step, 'best_validation_return': best_return,
               'best_selection': 'mean return on fixed validation seeds; may favor short failures',
               'elapsed_seconds': time.perf_counter() - start,
               'last_validation': validations[-1],
               'convergence_claim': False,
               'training_episodes_note': 'episodes.csv includes completed episodes only; tail is unfinished'}
    write_json(output / 'training_summary.json', summary)
    from visualization.learning_curves import plot_training
    plot_training(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/ppo_seed7')
    parser.add_argument('--resume', type=Path)
    for item in fields(TrainingConfig):
        value = getattr(TrainingConfig(), item.name)
        parser.add_argument('--' + item.name.replace('_', '-'), type=type(value), default=None)
    parser.add_argument('--wind-mode', choices=['none', 'constant', 'random', 'gusts'])
    parser.add_argument('--max-wind-speed', type=float)
    parser.add_argument('--reference-speed', type=float)
    parser.add_argument('--episode-seconds', type=float)
    args = parser.parse_args()
    if args.resume:
        _, env_config, saved = PPOAgent.load(args.resume)
        values = saved['training_config']
    else:
        env_config, values = EnvironmentConfig(), plain_dict(TrainingConfig())
    for item in fields(TrainingConfig):
        if getattr(args, item.name) is not None:
            values[item.name] = getattr(args, item.name)
    for argument, setting in [('wind_mode', 'wind_mode'), ('max_wind_speed', 'max_wind_speed'),
                              ('reference_speed', 'reference_speed'), ('episode_seconds', 'max_episode_time')]:
        value = getattr(args, argument)
        if value is not None:
            setattr(env_config, setting, value)
    print(json.dumps(train(TrainingConfig(**values), args.output, env_config, args.resume), indent=2))


if __name__ == '__main__':
    main()
