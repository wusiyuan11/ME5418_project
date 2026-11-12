"""Batched neural inference across independent original PyBullet clients."""
import numpy as np
import torch
from rl.ppo import generalized_advantage


class BatchedCollector:
    def __init__(self,envs,model,seed):
        self.envs,self.model = envs,model
        self.observations = np.stack([env.reset(seed=seed*1000+i)[0] for i,env in enumerate(envs)])
        # Keep unfinished episodes across rollout boundaries. Resetting every
        # PPO batch would prevent the policy from experiencing long missions.
        self.total = 0
        self.physics_total = 0
        self.returns = np.zeros(len(envs))
        self.raw_returns = np.zeros(len(envs))
        self.lengths = np.zeros(len(envs),dtype=int)
        self.physics_lengths = np.zeros(len(envs),dtype=int)
        self.errors = np.zeros(len(envs))

    def collect(self,steps,gamma,gae_lambda):
        records = {key:[] for key in ('observations','latents','old_log_probs','values',
                                    'rewards','next_values','terminated','episode_ends')}
        completed = []
        for _ in range(steps):
            observation = torch.as_tensor(self.observations,dtype=torch.float32)
            # On-policy samples are fixed data for the later PPO update; no
            # autograd graph is retained through simulation or target creation.
            with torch.no_grad():
                actions,latents,log_probs,values = self.model.sample(observation)
            transitions = [env.step(action) for env,action in zip(self.envs,actions.numpy())]
            next_obs = np.stack([item[0] for item in transitions])
            rewards = np.array([item[1] for item in transitions],dtype=np.float32)
            terminal = np.array([item[2] for item in transitions])
            ends = np.array([item[2] or item[3] for item in transitions])
            # Capture the final state value BEFORE replacing ended states with
            # reset observations. GAE uses terminal/end flags to mask bootstrap.
            with torch.no_grad():
                next_values = self.model(torch.as_tensor(next_obs))[2]
            items = (observation,latents,log_probs,values,torch.as_tensor(rewards),
                     next_values,torch.as_tensor(terminal),torch.as_tensor(ends))
            for key,item in zip(records,items):
                records[key].append(item)
            self.total += len(self.envs)
            self.physics_total += sum(item[4]['physics_steps'] for item in transitions)
            self.returns += rewards
            self.lengths += 1
            for i,(env,transition) in enumerate(zip(self.envs,transitions)):
                info = transition[4]
                self.raw_returns[i] += info['original_reward']
                self.physics_lengths[i] += info['physics_steps']
                self.errors[i] += info['tracking_error_square_sum']
                if ends[i]:
                    full_start = env.start_reference_time == 0.
                    # Raw count may include a declared curriculum prefix.
                    # waypoints_earned subtracts that prefix; subtask completion
                    # is not full-mission evidence. Full-start training may
                    # still use weaker wind; validation is always the full task.
                    completed.append({'total_timesteps':self.total,'env_index':i,
                        'training_return':float(self.returns[i]),'raw_return':float(self.raw_returns[i]),
                        'length':int(self.lengths[i]),'physics_length':int(self.physics_lengths[i]),
                        'total_physics_timesteps':self.physics_total,
                        'tracking_rmse_m':float(np.sqrt(self.errors[i]/self.physics_lengths[i])),
                        'waypoints_achieved':info['waypoints_achieved'],
                        'waypoints_earned':info['waypoints_achieved']-env.start_waypoints,
                        'reset_reference_time':env.start_reference_time,'original_start':full_start,
                        'subtask_success':info['is_success'],
                        'full_task_success':bool(full_start and info['is_success']),
                        'wind_limit':env.active_wind_limit,'reason':info['termination_reason']})
                    next_obs[i] = env.reset()[0]
                    self.returns[i] = self.raw_returns[i] = self.errors[i] = 0.
                    self.lengths[i] = 0
                    self.physics_lengths[i] = 0
            self.observations = next_obs
        batch = {key:torch.stack(value).detach() for key,value in records.items()}
        # Shape is [time, client, ...] here. Flattening first would leak return
        # estimates between clients or break each client's temporal sequence.
        batch['advantages'],batch['returns'] = generalized_advantage(batch['rewards'],batch['values'],
            batch['next_values'],batch['terminated'],batch['episode_ends'],gamma,gae_lambda)
        # GAE runs down time independently for each env, then flatten for SGD.
        batch = {key:value.flatten(0,1) for key,value in batch.items()}
        if not all(torch.isfinite(value).all() for value in batch.values()):
            raise FloatingPointError('Non-finite pure-PPO rollout')
        return batch,completed
