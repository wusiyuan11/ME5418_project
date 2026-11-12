"""Independent seeded rollouts and explicit tracking/completion measurements."""
import csv
import json
from pathlib import Path
import numpy as np
from environment import AerialSprayingEnv


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    """Create a CSV containing plain scalar records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_track_distances(positions, waypoints):
    """3-D distance to the nearest segment of the complete static polyline.

    This proximity metric alone cannot certify ordered waypoint completion.
    Timed-reference error and strict success are reported separately.
    """
    positions, waypoints = np.asarray(positions), np.asarray(waypoints)
    segments = np.diff(waypoints, axis=0)
    offset = positions[:, None, :] - waypoints[None, :-1, :]
    projection = np.clip((offset * segments).sum(-1) / np.square(segments).sum(-1), 0, 1)
    residual = offset - projection[..., None] * segments
    return np.linalg.norm(residual, axis=-1).min(axis=1)


def evaluate_policy(action_function, config, seeds, controller='ppo', output=None,
                    animation=False):
    """Run complete episodes on separate envs; never update network weights.

    action_function(observation, env) permits the inherited PID's reference-
    velocity access. PPO/untrained closures deliberately ignore env/info.
    """
    results = []
    for episode_index, seed in enumerate(seeds):
        env = AerialSprayingEnv(config)
        try:
            observation, _ = env.reset(seed=int(seed))
            # PID has an integral state that must start fresh for each seed.
            if hasattr(action_function, 'reset'):
                action_function.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = action_function(observation, env)
                observation, _, terminated, truncated, info = env.step(action)
            history = env.history
            error = np.asarray(history['error'])
            cross_track = cross_track_distances(history['position'], env.trajectory.waypoints)
            results.append({'controller': controller, 'seed': int(seed),
                'return': float(env.episode_reward), 'steps': env.step_count,
                'duration_s': env.time, 'tracking_rmse_m': float(np.sqrt(np.mean(error ** 2))),
                'max_tracking_error_m': float(error.max()),
                'cross_track_rmse_m': float(np.sqrt(np.mean(cross_track ** 2))),
                'max_cross_track_error_m': float(cross_track.max()),
                'mean_action_squared': float(np.square(history['action']).sum(axis=1).mean()),
                'waypoints_achieved': info['waypoints_achieved'],
                'total_waypoints': len(env.trajectory.waypoints),
                'success': info['is_success'], 'crashed': info['termination_reason'] == 'crash',
                'boundary_violation': info['termination_reason'] == 'boundary_violation',
                'time_limit': truncated, 'reason': info['termination_reason']})
            if output is not None:
                output = Path(output)
                samples = []
                for i, time in enumerate(history['time']):
                    sample = {'time_s': time, 'reward': history['reward'][i],
                              'tracking_error_m': history['error'][i],
                              'waypoints_achieved': history['waypoints_achieved'][i],
                              'liquid_mass_kg': history['payload'][i]}
                    for name, key in [('position', 'position'), ('reference', 'reference'),
                                      ('velocity', 'velocity'), ('action', 'action'), ('wind', 'wind')]:
                        sample.update({f'{name}_{axis}': float(v) for axis, v in enumerate(history[key][i])})
                    samples.append(sample)
                write_rows(output / 'trajectories' / f'{controller}_seed_{seed}.csv', samples)
                if episode_index == 0:
                    from visualization.plot_trajectory import save_episode_plots
                    save_episode_plots(history, env.trajectory.waypoints,
                        output / f'{controller}_trajectory.png', output / f'{controller}_tracking.png',
                        f'{controller}: seed {seed}; {info["termination_reason"]}')
                    if animation:
                        from visualization.animate_episode import save_episode_animation
                        save_episode_animation(history, env.trajectory.waypoints,
                                               output / f'{controller}_replay.gif')
        finally:
            env.close()
    return results


def summarize(results):
    numeric = ('return', 'steps', 'tracking_rmse_m', 'max_tracking_error_m',
               'cross_track_rmse_m', 'max_cross_track_error_m', 'mean_action_squared',
               'waypoints_achieved')
    return {'episodes': len(results), 'seeds': [r['seed'] for r in results],
            'mean': {key: float(np.mean([r[key] for r in results])) for key in numeric},
            'std': {key: float(np.std([r[key] for r in results])) for key in numeric},
            'rates': {key: float(np.mean([r[key] for r in results]))
                      for key in ('success', 'crashed', 'boundary_violation', 'time_limit')}}
