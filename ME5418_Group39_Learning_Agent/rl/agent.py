"""PPO agent and portable CPU checkpoints using the stage-2 ActorCritic."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import platform
import importlib.metadata

import numpy as np
import torch

from environment import EnvironmentConfig
from environment.reward import RewardWeights
from .network import ActorCritic
from .ppo import ppo_update


@dataclass
class TrainingConfig:
    """Algorithm settings; the architecture and action semantics are unchanged."""
    total_timesteps: int = 100_000
    n_steps: int = 2048
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_range: float = .2
    seed: int = 7
    eval_every: int = 4
    eval_episodes: int = 3
    eval_seed: int = 10000
    threads: int = 1

    def validate(self):
        for key in ('total_timesteps', 'n_steps', 'batch_size', 'epochs',
                    'eval_every', 'eval_episodes', 'threads'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.n_steps < 2 or self.total_timesteps < 2 or self.batch_size < 2:
            raise ValueError('Training, rollout and minibatch sizes must be at least two')
        if not 0 < self.learning_rate or not np.isfinite(self.learning_rate):
            raise ValueError('learning_rate must be finite and positive')
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError('gamma and gae_lambda must be in [0, 1]')
        if not 0 < self.clip_range < 1:
            raise ValueError('clip_range must be in (0, 1)')


def plain_dict(config):
    """Convert dataclass NumPy scalars to safe JSON/checkpoint metadata."""
    return json.loads(json.dumps(asdict(config), default=lambda x: x.item()))


def environment_config(data):
    """Reconstruct the saved environment including the reward dataclass."""
    data = dict(data)
    data['reward_weights'] = RewardWeights(**data['reward_weights'])
    return EnvironmentConfig(**data)


def runtime_versions():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'numpy', 'gymnasium', 'pybullet', 'matplotlib', 'Pillow')}}


class PPOAgent:
    """Select commands from observations and improve the inherited policy."""
    def __init__(self, config):
        config.validate()
        self.config = config
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        self.model = ActorCritic().cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=config.learning_rate, eps=1e-5)

    def act(self, observation, deterministic=True):
        """Only the 15-D observation enters the network; info is never input."""
        with torch.no_grad():
            action = self.model.sample(torch.as_tensor(observation, dtype=torch.float32),
                                       deterministic=deterministic)[0]
        return action.numpy()

    def update(self, batch):
        return ppo_update(self.model, self.optimizer, batch,
                          epochs=self.config.epochs, minibatch_size=self.config.batch_size,
                          clip=self.config.clip_range)

    def save(self, path, env_config, total_timesteps, update):
        """Save policy, optimizer, configuration and counters atomically.

        Physics state is deliberately not serialized. Resume starts a new
        episode, so it is a warm restart rather than a bit-exact continuation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'format_version': 1, 'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'training_config': plain_dict(self.config),
                      'environment_config': plain_dict(env_config),
                      'total_timesteps': int(total_timesteps), 'update': int(update),
                      'torch_rng_state': torch.get_rng_state(),
                      'runtime': runtime_versions(),
                      'resume_semantics': 'warm restart with a fresh environment episode'}
        temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(checkpoint, temp)
        temp.replace(path)

    @classmethod
    def load(cls, path):
        """Load our tensor-and-primitive checkpoint without arbitrary objects."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Expected a stage-3 PPOAgent checkpoint, not network_demo.pt')
        allowed = {f.name for f in fields(TrainingConfig)}
        agent = cls(TrainingConfig(**{k: v for k, v in checkpoint['training_config'].items()
                                     if k in allowed}))
        agent.model.load_state_dict(checkpoint['model'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        return agent, environment_config(checkpoint['environment_config']), checkpoint
