"""Behavioral checks for continuous sampling, checkpointing and the whole workflow."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import pytest
import torch

from environment import EnvironmentConfig
from rl.agent import PPOAgent, TrainingConfig
from rl.evaluation import cross_track_distances
from rl.rollout import RolloutCollector
from train import train


class FiveStepEnvironment:
    """Known episode boundaries expose sampler resets independently of physics."""
    def __init__(self):
        self.config = EnvironmentConfig(dt=.1)
        self.resets = 0

    def reset(self, seed=None):
        self.resets += 1
        self.steps = 0
        return np.zeros(15, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (np.zeros(15, dtype=np.float32), 1., False, self.steps == 5,
                {'tracking_error_norm': 2., 'waypoints_achieved': 1,
                 'is_success': False, 'termination_reason': 'time_limit'})


def test_collector_keeps_partial_episode_and_returns_real_episode_sum():
    agent = PPOAgent(TrainingConfig())
    env = FiveStepEnvironment()
    collector = RolloutCollector(env, agent.model)
    _, completed = collector.collect(3)
    assert completed == [] and env.resets == 1
    batch, completed = collector.collect(3)
    assert env.resets == 2 and collector.length == 1
    assert completed[0]['return'] == 5 and completed[0]['length'] == 5
    assert completed[0]['total_timesteps'] == 5
    assert completed[0]['tracking_rmse_m'] == 2
    assert batch['episode_ends'].tolist() == [False, True, False]


def test_static_cross_track_does_not_imply_waypoint_completion():
    # A point on a later segment has zero proximity error regardless of visit order.
    path = np.array([[0., 0., 2.], [2., 0., 2.], [2., 2., 2.]])
    errors = cross_track_distances([[1, 0, 3], [2, 1, 2], [3, 0, 2]], path)
    np.testing.assert_allclose(errors, [1, 0, 1])


def test_training_checkpoint_resume_and_cli_evaluation(tmp_path):
    config = TrainingConfig(total_timesteps=65, n_steps=16, batch_size=8,
                            epochs=1, eval_every=3, eval_episodes=1)
    env_config = EnvironmentConfig(max_episode_time=.25, wind_mode='none')
    first = tmp_path / 'first'
    summary = train(config, first, env_config)
    assert summary['total_timesteps'] == 65
    assert summary['completed_training_episodes'] > 0
    agent, restored_env, checkpoint = PPOAgent.load(first / 'final.pt')
    assert checkpoint['total_timesteps'] == 65
    assert agent.optimizer.state_dict()['state']
    assert restored_env.max_episode_time == .25
    observation = np.arange(15, dtype=np.float32) / 10
    action = agent.act(observation)
    reloaded, _, _ = PPOAgent.load(first / 'final.pt')
    np.testing.assert_array_equal(action, reloaded.act(observation))
    assert np.all(np.abs(action) <= 1)
    resumed = tmp_path / 'resumed'
    summary = train(replace(config, total_timesteps=97), resumed, resume=first / 'final.pt')
    assert summary['total_timesteps'] == 97
    assert (resumed / 'learning_curves.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        train(config, first, env_config)
    root = Path(__file__).parents[1]
    output = tmp_path / 'heldout'
    subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(first / 'final.pt'),
        '--controller', 'all', '--episodes', '1', '--output', str(output)],
        cwd=root, check=True, timeout=60)
    results = json.loads((output / 'summary.json').read_text())
    assert set(results['controllers']) == {'ppo', 'pid', 'untrained'}
    assert results['checkpoint_timesteps'] == 65
    assert all(s['episodes'] == 1 for s in results['controllers'].values())
    assert len(list((output / 'trajectories').glob('*.csv'))) == 3
    assert (output / 'comparison.png').stat().st_size > 1000


def test_training_configuration_rejects_invalid_learning_settings():
    with pytest.raises(ValueError):
        TrainingConfig(n_steps=1).validate()
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=float('nan')).validate()
