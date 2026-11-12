"""A bounded Gaussian actor and separate state-value critic, implemented in PyTorch."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class ActorCritic(nn.Module):
    """Map the 15 observable values to four bounded commands and one value.

    Wind, payload and waypoint counters are deliberately excluded from inputs.
    Fixed SI scales avoid learning from differently scaled position/rate units.
    Separate actor/critic MLPs each have two 64-unit tanh hidden layers.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('observation_scale', torch.tensor(
            [12., 8., 2., 3., 3., 3., math.pi, math.pi, math.pi,
             3., 3., 3., 12., 8., 2.]))
        self.actor = nn.Sequential(nn.Linear(15, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
        self.critic = nn.Sequential(nn.Linear(15, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((4,), -0.7))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    def forward(self, observation):
        """Accept shape (...,15); return latent mean, std (...,4), value (...)."""
        if observation.shape[-1] != 15:
            raise ValueError('Expected 15 observation features')
        x = observation / self.observation_scale
        mean = self.actor(x)
        std = self.log_std.clamp(-5., 2.).exp().expand_as(mean)
        return mean, std, self.critic(x).squeeze(-1)

    @staticmethod
    def _log_probability(distribution, latent):
        # Stable log(1-tanh(u)^2) Jacobian; avoids inverse tanh at saturation.
        correction = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(dim=-1)

    def sample(self, observation, deterministic=False):
        """Return bounded action, stored latent, action log-density and V(s)."""
        mean, std, value = self(observation)
        distribution = Normal(mean, std)
        # Policy-gradient updates treat collected actions as fixed data.
        latent = mean if deterministic else distribution.sample()
        return latent.tanh(), latent, self._log_probability(distribution, latent), value

    def evaluate(self, observation, latent):
        """Re-evaluate stored actions under the current policy for PPO ratios."""
        mean, std, value = self(observation)
        return self._log_probability(Normal(mean, std), latent), value
