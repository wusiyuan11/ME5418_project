"""Stateful on-policy sampling: episodes continue across optimizer rollouts."""
import numpy as np
import torch
from .ppo import generalized_advantage


class RolloutCollector:
    """Keep the live observation and episode accounting between collect calls.

    Stage 2 reset the environment at the beginning of every rollout. Keeping
    state here avoids silently discarding a partial flight at each update.
    """
    def __init__(self, env, model, seed=7, timestep_offset=0, episode_offset=0):
        self.env, self.model = env, model
        self.observation, _ = env.reset(seed=seed)
        self.total_timesteps, self.episodes = timestep_offset, episode_offset
        self._reset_accounting()

    def _reset_accounting(self):
        self.episode_return = 0.0
        self.length = 0
        self.error_square = 0.0
        self.max_error = 0.0
        self.effort = 0.0

    def collect(self, steps, gamma=.99, gae_lambda=.95):
        if steps < 2:
            raise ValueError('At least two transitions are needed')
        records = {k: [] for k in ('observations', 'latents', 'old_log_probs', 'values',
                                   'rewards', 'next_values', 'terminated', 'episode_ends')}
        completed = []
        for _ in range(steps):
            observation = torch.as_tensor(self.observation, dtype=torch.float32)
            with torch.no_grad():
                action, latent, log_prob, value = self.model.sample(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action.numpy())
            with torch.no_grad():
                next_value = self.model(torch.as_tensor(next_observation, dtype=torch.float32))[2]
            items = (observation, latent, log_prob, value,
                     torch.tensor(reward, dtype=torch.float32), next_value,
                     torch.tensor(terminated), torch.tensor(terminated or truncated))
            for key, item in zip(records, items):
                records[key].append(item)
            self.total_timesteps += 1
            self.length += 1
            self.episode_return += reward
            error = info['tracking_error_norm']
            self.error_square += error ** 2
            self.max_error = max(self.max_error, error)
            self.effort += float(np.square(action.numpy()).sum())
            if terminated or truncated:
                self.episodes += 1
                completed.append({'episode': self.episodes, 'total_timesteps': self.total_timesteps,
                    'return': self.episode_return, 'length': self.length,
                    'duration_s': self.length * self.env.config.dt,
                    'tracking_rmse_m': (self.error_square / self.length) ** .5,
                    'max_tracking_error_m': self.max_error,
                    'mean_action_squared': self.effort / self.length,
                    'waypoints_achieved': info['waypoints_achieved'],
                    'success': info['is_success'], 'reason': info['termination_reason']})
                self.observation, _ = self.env.reset()
                self._reset_accounting()
            else:
                self.observation = next_observation
        batch = {key: torch.stack(items).detach() for key, items in records.items()}
        batch['advantages'], batch['returns'] = generalized_advantage(
            batch['rewards'], batch['values'], batch['next_values'], batch['terminated'],
            batch['episode_ends'], gamma=gamma, gae_lambda=gae_lambda)
        if any(not torch.isfinite(tensor).all() for tensor in batch.values()):
            raise FloatingPointError('Non-finite rollout or advantage')
        return batch, completed
