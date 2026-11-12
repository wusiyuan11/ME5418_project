"""Verify policy density, boundary handling and a real optimizer update."""
import torch
from rl import ActorCritic
from rl.ppo import generalized_advantage
from demo.demo_network import run


def test_shapes_bounds_and_log_probability():
    torch.manual_seed(2)
    model = ActorCritic()
    observations = torch.randn(8, 15)
    actions, latents, old_log_prob, values = model.sample(observations)
    assert actions.shape == (8, 4) and values.shape == (8,)
    assert torch.all(actions.abs() <= 1)
    log_prob, _ = model.evaluate(observations, latents)
    assert torch.allclose(log_prob, old_log_prob)
    # Independently compare the manually stabilized Jacobian with PyTorch's
    # transformed-density implementation for non-saturated random samples.
    mean, std, _ = model(observations)
    reference = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, std), [torch.distributions.TanhTransform()])
    assert torch.allclose(log_prob, reference.log_prob(actions).sum(-1), atol=1e-5)
    assert torch.isfinite(model.evaluate(observations, torch.full((8,4), 100.))[0]).all()


def test_gae_terminal_truncation_and_reset():
    rewards = torch.tensor([1., 2., 3.])
    values = torch.zeros(3)
    next_values = torch.tensor([10., 20., 30.])
    # True terminal: no bootstrap; time limit: bootstrap but no next-episode carry.
    advantage, returns = generalized_advantage(rewards, values, next_values,
        torch.tensor([True, False, False]), torch.tensor([True, True, False]), gamma=.9)
    assert torch.allclose(returns, torch.tensor([1., 20., 30.]))
    assert torch.equal(advantage, returns)


def test_forward_backward_real_environment(tmp_path):
    stats = run(steps=32, output=str(tmp_path/'demo.json'))
    update = stats['updates'][0]
    assert update['actor_changed'] and update['critic_changed']
    assert update['max_parameter_change'] > 0
    assert (tmp_path/'demo.pt').stat().st_size > 1000
