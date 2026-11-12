"""Create the network, show I/O and verify backpropagation on real transitions.

Run: python -m demo.demo_network --steps 128 --updates 1
This smoke demonstration is not a claim of a converged flight policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import AerialSprayingEnv
from rl import ActorCritic
from rl.ppo import collect_rollout, ppo_update


def run(seed=7, steps=128, updates=1, output='outputs/network_demo.json'):
    if updates < 1:
        raise ValueError('At least one update is required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ActorCritic().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    env = AerialSprayingEnv()
    try:
        observation, _ = env.reset(seed=seed)
        with torch.no_grad():
            mean, std, value = model(torch.tensor(observation))
            action = mean.tanh()
        summary = {'seed': seed, 'device': 'cpu', 'parameters': sum(p.numel() for p in model.parameters()),
                   'input_observation': observation.tolist(), 'latent_mean': mean.tolist(),
                   'latent_std': std.tolist(), 'deterministic_action': action.tolist(),
                   'value': value.item(), 'updates': []}
        for index in range(updates):
            batch, episodes = collect_rollout(env, model, steps, seed+index)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            metrics = ppo_update(model, optimizer, batch)
            changes = {name: float((p.detach()-before[name]).abs().max()) for name,p in model.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert max(changes.values()) > 0, 'Optimizer failed to change parameters'
            metrics.update({'transitions': steps, 'episode_end_reasons': episodes,
                            'max_parameter_change': max(changes.values()),
                            'actor_changed': any(v > 0 for k,v in changes.items() if k.startswith('actor.')),
                            'critic_changed': any(v > 0 for k,v in changes.items() if k.startswith('critic.'))})
            summary['updates'].append(metrics)
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2)+'\n')
        torch.save(model.state_dict(), path.with_suffix('.pt'))
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--output', default='outputs/network_demo.json')
    args = parser.parse_args()
    run(args.seed, args.steps, args.updates, args.output)
