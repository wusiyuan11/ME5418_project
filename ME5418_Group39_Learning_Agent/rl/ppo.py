"""Minimal on-policy PPO rollout, time-limit-aware GAE and clipped updates."""
import numpy as np
import torch
from torch.nn import functional as F


def generalized_advantage(rewards, values, next_values, terminated, episode_ends,
                          gamma=0.99, gae_lambda=0.95):
    """Bootstrap at time limits, never at true terminals or across reset episodes."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (~terminated[i]) - values[i]
        carry = delta + gamma * gae_lambda * (~episode_ends[i]) * carry
        advantages[i] = carry
    return advantages, advantages + values


def collect_rollout(env, model, steps=128, seed=7):
    """Collect CPU tensors; capture the final observation before any reset."""
    if steps < 2:
        raise ValueError('Rollout requires at least two transitions')
    observation, _ = env.reset(seed=seed)
    records = {k: [] for k in ('observations', 'latents', 'old_log_probs', 'values',
                               'rewards', 'next_values', 'terminated', 'episode_ends')}
    episodes = []
    for _ in range(steps):
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action, latent, log_prob, value = model.sample(tensor)
        next_observation, reward, terminated, truncated, info = env.step(action.numpy())
        with torch.no_grad():
            next_value = model(torch.as_tensor(next_observation))[2]
        for key, item in zip(records, (tensor, latent, log_prob, value,
                                      torch.tensor(reward, dtype=torch.float32), next_value,
                                      torch.tensor(terminated), torch.tensor(terminated or truncated))):
            records[key].append(item)
        if terminated or truncated:
            episodes.append(info['termination_reason'])
            observation, _ = env.reset()
        else:
            observation = next_observation
    batch = {key: torch.stack(items).detach() for key, items in records.items()}
    batch['advantages'], batch['returns'] = generalized_advantage(
        batch['rewards'], batch['values'], batch['next_values'],
        batch['terminated'], batch['episode_ends'])
    return batch, episodes


def ppo_update(model, optimizer, batch, epochs=4, minibatch_size=64, clip=0.2):
    """Optimize clipped policy surrogate plus 0.5 times value MSE.

    No entropy bonus is used. Learned exploration is bounded through log_std.
    Stored rollout quantities and advantage targets remain detached throughout.
    """
    if epochs < 1 or minibatch_size < 2 or not 0 < clip < 1:
        raise ValueError('Expected positive epochs, minibatch >= 2 and clip in (0,1)')
    advantages = batch['advantages'].detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics = []
    for _ in range(epochs):
        for indices in torch.randperm(len(advantages)).split(minibatch_size):
            log_prob, value = model.evaluate(batch['observations'][indices], batch['latents'][indices])
            ratio = (log_prob - batch['old_log_probs'][indices]).exp()
            surrogate = torch.minimum(ratio * advantages[indices],
                                      ratio.clamp(1-clip, 1+clip) * advantages[indices])
            policy_loss = -surrogate.mean()
            value_loss = F.mse_loss(value, batch['returns'][indices])
            loss = policy_loss + .5 * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('Non-finite PPO gradient')
            optimizer.step()
            # Diagnostics do not change the stage-2 PPO loss or update rule.
            log_ratio = log_prob.detach() - batch['old_log_probs'][indices]
            approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
            clip_fraction = ((ratio.detach() - 1).abs() > clip).float().mean()
            metrics.append([loss.item(), policy_loss.item(), value_loss.item(), grad_norm.item(),
                            approx_kl.item(), clip_fraction.item()])
    return dict(zip(('loss', 'policy_loss', 'value_loss', 'gradient_norm_before_clip'),
                    np.mean(metrics, axis=0).tolist())) | {
                        'approx_kl': float(np.mean(metrics, axis=0)[4]),
                        'clip_fraction': float(np.mean(metrics, axis=0)[5]),
                        'optimizer_steps': len(metrics),
                        'policy_std_mean': float(model.log_std.detach().clamp(-5, 2).exp().mean())}
