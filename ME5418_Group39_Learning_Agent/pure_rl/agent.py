"""PPO-only network and auditable checkpoints; teacher checkpoints are rejected."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from environment import EnvironmentConfig
from rl.agent import plain_dict, environment_config, runtime_versions
from rl.network import ActorCritic
from .task import TaskOptions


@dataclass
class PPOConfig:
    # Sizes/budget refer to policy decisions across clients, not physics steps.
    # Use the root JSON-configured entry point for the selected 10 Hz preset.
    total_timesteps: int = 600000
    seed: int = 7
    n_envs: int = 8
    rollout_steps: int = 256
    batch_size: int = 256
    epochs: int = 8
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    gamma: float = .999
    gae_lambda: float = .97
    clip_range: float = .2
    target_kl: float = .02
    entropy_coef: float = 0.
    eval_every: int = 40
    eval_episodes: int = 5
    eval_seed: int = 60000
    features: str = 'mission'
    action_repeat: int = 1

    def validate(self):
        sizes = (self.n_envs, self.rollout_steps, self.batch_size, self.epochs,
                 self.eval_every, self.eval_episodes)
        if min(sizes) < 1 or self.total_timesteps < 2 or self.batch_size < 2:
            raise ValueError('Invalid PPO sizes')
        if self.total_timesteps % self.n_envs or self.n_envs*self.rollout_steps < 2:
            raise ValueError('Budget must be divisible by n_envs and rollout batch >= 2')
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip_range < 1):
            raise ValueError('Invalid PPO discount/clip')
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.learning_rate, self.value_learning_rate, self.target_kl)):
            raise ValueError('PPO rates/KL must be finite and positive')
        if not np.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError('Entropy coefficient must be nonnegative')
        if self.features not in ('mission','goal') or self.action_repeat<1:
            raise ValueError('Invalid public features/control period')


class MissionActorCritic(ActorCritic):
    """Extend the stage-2 64-64 actor/critic input layer from 15 to 23 values."""
    def __init__(self,features='mission'):
        super().__init__()
        # Reuse the inherited hidden/output design; widen only the input layers.
        # Actor: 23-64-64-4, critic: 23-64-64-1. Parameters are independent.
        self.actor[0], self.critic[0] = nn.Linear(23, 64), nn.Linear(23, 64)
        self.observation_scale = torch.tensor(
            [12.,8.,2.,3.,3.,3.,.5,.5,math.pi,3.,3.,3.,2.,2.,2.,2.,2.,2.,6.,6.,2.,1.,1.])
        if features=='goal':
            # Fixed public physical units, not normalization fitted on test data.
            self.observation_scale[18:21] = torch.tensor([2.,2.,1.])
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)
        with torch.no_grad():
            self.log_std.copy_(torch.tensor([-1.,-1.,-1.,-1.8]))

    def forward(self, observation):
        if observation.shape[-1] != 23:
            raise ValueError('Mission policy requires 23 public observation features')
        normalized = observation/self.observation_scale
        # The same observation feeds two separate branches. Critic output is
        # NOT concatenated into the actor's inputs and does not choose actions.
        mean = self.actor(normalized)
        std = self.log_std.clamp(-4.5, .2).exp().expand_as(mean)
        return mean, std, self.critic(normalized).squeeze(-1)


class PurePPO:
    def __init__(self, config):
        config.validate()
        self.config = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.model = MissionActorCritic(config.features).cpu()
        # Only actor + exploration parameters enter the policy optimizer.
        # The critic fits interaction-derived returns with its own optimizer.
        self.policy_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters())+[self.model.log_std], lr=config.learning_rate, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.model.critic.parameters(),
                                               lr=config.value_learning_rate, eps=1e-5)
        self.origin = {'initialization': 'random', 'teacher_environment_steps': 0,
                       'expert_action_labels': 0, 'imitation_loss_weight': 0,
                       'previous_pure_ppo_timesteps': 0}

    def act(self, observation):
        # sample() also computes a value for convenience, but index [0] selects
        # the actor action only. Evaluation uses its deterministic tanh(mean).
        with torch.no_grad():
            return self.model.sample(torch.as_tensor(observation, dtype=torch.float32),
                                     deterministic=True)[0].numpy()

    def warm_start_pure_only(self, path):
        # Explicit PPO-to-PPO staging, not an automatic imitation initialization.
        # Accept the inherited checkpoint format and reject known teacher runs.
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        provenance = checkpoint.get('training_provenance', {})
        if (checkpoint.get('format_version') != 1 or checkpoint.get('optimization', {}).get('method') == 'bc_ppo'
            or provenance.get('teacher_environment_steps', 0) != 0):
            raise ValueError('Only original teacher-free PPO checkpoints are allowed')
        state = checkpoint['model']
        if state['actor.0.weight'].shape[1] != 15:
            raise ValueError('Expected the inherited 15-input PPO model')
        with torch.no_grad():
            for name in ('actor','critic'):
                module = getattr(self.model,name)
                module[0].weight.zero_()
                # Added mission features initially have zero influence. PPO
                # must learn their weights from new interaction rewards.
                # Preserve the old physical mapping despite new input units.
                module[0].weight[:,:15].copy_(state[f'{name}.0.weight'] *
                    (self.model.observation_scale[:15]/state['observation_scale']))
                module[0].bias.copy_(state[f'{name}.0.bias'])
                for index in (2,4):
                    module[index].weight.copy_(state[f'{name}.{index}.weight'])
                    module[index].bias.copy_(state[f'{name}.{index}.bias'])
            self.model.log_std.copy_(state['log_std'])
            self.model.log_std[-1].clamp_(max=-1.8)
        self.origin.update({'initialization':'previous pure PPO, no teacher',
            'source_checkpoint':str(path), 'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'previous_pure_ppo_timesteps':checkpoint['total_timesteps'],
            'optimizer_restart':'fresh Adam; deterministic old action mapping preserved'})

    def save(self, path, options, total, update, physics_steps=None):
        path = Path(path)
        payload = {'format_version':'pure_ppo_v1', 'model':self.model.state_dict(),
            'policy_optimizer':self.policy_optimizer.state_dict(),
            'value_optimizer':self.value_optimizer.state_dict(),
            'config':asdict(self.config), 'task_options':asdict(options),
            'evaluation_environment':plain_dict(EnvironmentConfig()),
            'total_timesteps':int(total), 'update':int(update), 'origin':self.origin,
            'physics_timesteps':int(total if physics_steps is None else physics_steps),
            'torch_rng_state':torch.get_rng_state(), 'runtime':runtime_versions()}
        temporary = path.with_suffix('.tmp')
        # Atomic replacement avoids leaving a partly written model file.
        # This payload is not a snapshot of the live PyBullet client states.
        torch.save(payload,temporary)
        temporary.replace(path)

    @classmethod
    def load(cls,path):
        payload = torch.load(path, weights_only=True, map_location='cpu')
        if payload.get('format_version') != 'pure_ppo_v1':
            raise ValueError('Expected a pure-PPO checkpoint')
        if (payload['origin']['expert_action_labels'] or payload['origin']['imitation_loss_weight']
            or payload['origin']['teacher_environment_steps']):
            raise ValueError('This checkpoint contains teacher supervision')
        agent = cls(PPOConfig(**payload['config']))
        agent.model.load_state_dict(payload['model'])
        agent.policy_optimizer.load_state_dict(payload['policy_optimizer'])
        agent.value_optimizer.load_state_dict(payload['value_optimizer'])
        agent.origin = payload['origin']
        torch.set_rng_state(payload['torch_rng_state'])
        return agent, TaskOptions(**payload['task_options']), payload
