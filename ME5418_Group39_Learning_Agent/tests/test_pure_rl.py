"""Pure-PPO contracts: no teacher, correct features/GAE, honest curriculum metrics."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from environment import AerialSprayingEnv,EnvironmentConfig
from rl.agent import PPOAgent,TrainingConfig
from rl.ppo import generalized_advantage
from pure_rl.agent import PurePPO,PPOConfig
from pure_rl.task import LearningTask,TaskOptions,mission_observation
from pure_rl.collector import BatchedCollector
from pure_rl.learning import update
from pure_rl.train import train,evaluate


def test_public_features_expose_mission_progress_but_not_hidden_disturbances():
    env = AerialSprayingEnv()
    try:
        raw,_ = env.reset(seed=2)
        feature = mission_observation(raw,env)
        assert feature.shape == (23,)
        np.testing.assert_array_equal(feature[:15],raw)
        np.testing.assert_allclose(feature[15:18],env.reference_velocity)
        env.payload.liquid_mass = .001
        env.wind.base_velocity[:] = 99
        np.testing.assert_array_equal(feature,mission_observation(raw,env))
        env.waypoints_achieved += 1
        assert not np.array_equal(feature,mission_observation(raw,env))
    finally:
        env.close()


def test_no_curriculum_reset_and_scaled_reward_preserve_original_physics():
    raw = AerialSprayingEnv()
    wrapped = LearningTask(AerialSprayingEnv(),TaskOptions(reward='original'))
    try:
        a,_ = raw.reset(seed=31)
        b,_ = wrapped.reset(seed=31)
        np.testing.assert_array_equal(a,b[:15])
        for _ in range(6):
            a,reward,done,cut,info = raw.step(np.zeros(4))
            b,scaled,done_b,_,info_b = wrapped.step(np.zeros(4))
            np.testing.assert_array_equal(a,b[:15])
            assert scaled == .1*reward
            assert done_b == (done or cut)
        assert wrapped.start_reference_time == 0
        assert wrapped.unwrapped.config.waypoint_tolerance == .5
    finally:
        raw.close()
        wrapped.close()


def test_curriculum_is_removed_for_final_quarter_and_eval_always_original():
    env = LearningTask(AerialSprayingEnv(),TaskOptions(curriculum=True))
    try:
        partial = []
        for seed in range(6):
            env.reset(seed=seed)
            partial.append(env.start_reference_time>0)
        assert any(partial)
        env.set_fraction(.8)
        for seed in range(6):
            env.reset(seed=seed)
            assert env.start_reference_time == 0
            assert env.start_waypoints == 1
            assert env.active_wind_limit == 3
    finally:
        env.close()


def test_finite_mission_deadline_is_terminal_and_remaining_time_observed():
    env = LearningTask(AerialSprayingEnv(EnvironmentConfig(max_episode_time=.1)),TaskOptions())
    try:
        obs,_ = env.reset(seed=2)
        assert obs[-1] == 1
        env.step(np.zeros(4))
        obs,_,terminal,truncated,info = env.step(np.zeros(4))
        assert terminal and not truncated and info['termination_reason']=='time_limit'
        assert obs[-1] == 0 and not info['is_success']
    finally:
        env.close()


def test_two_environment_gae_has_no_cross_environment_carry():
    rewards = torch.tensor([[1.,10.],[2.,20.]])
    flags = torch.tensor([[False,True],[True,True]])
    advantage,returns = generalized_advantage(rewards,torch.zeros_like(rewards),
        torch.zeros_like(rewards),flags,flags,gamma=.9,gae_lambda=1.)
    torch.testing.assert_close(returns,torch.tensor([[2.8,10.],[2.,20.]]))


def test_pure_warm_start_preserves_old_mean_actions_and_rejects_teacher(tmp_path):
    old = PPOAgent(TrainingConfig())
    source = tmp_path/'old.pt'
    old.save(source,EnvironmentConfig(),100,1)
    pure = PurePPO(PPOConfig(total_timesteps=16))
    pure.warm_start_pure_only(source)
    rng = np.random.default_rng(2)
    for _ in range(8):
        observation = rng.normal(size=15).astype(np.float32)
        augmented = np.r_[observation,rng.normal(size=8)].astype(np.float32)
        np.testing.assert_allclose(old.act(observation),pure.act(augmented),atol=1e-6)
    data = torch.load(source,weights_only=True)
    data['optimization'] = {'method':'bc_ppo'}
    torch.save(data,tmp_path/'teacher.pt')
    with pytest.raises(ValueError):
        pure.warm_start_pure_only(tmp_path/'teacher.pt')


def test_batched_ppo_updates_and_counts_physical_steps(tmp_path):
    cfg = PPOConfig(total_timesteps=16,n_envs=2,rollout_steps=4,batch_size=4,epochs=1)
    agent = PurePPO(cfg)
    envs = [LearningTask(AerialSprayingEnv(EnvironmentConfig(max_episode_time=.1)),TaskOptions())
            for _ in range(2)]
    try:
        batch,episodes = BatchedCollector(envs,agent.model,3).collect(4,cfg.gamma,cfg.gae_lambda)
        assert batch['observations'].shape == (8,23)
        assert episodes[-1]['total_timesteps'] == 8
        assert all(row['original_start'] for row in episodes)
        assert not any(row['full_task_success'] for row in episodes)
        before = agent.model.actor[-1].weight.detach().clone()
        result = update(agent,batch,0.)
        assert result['policy_optimizer_steps'] > 0 and result['value_optimizer_steps'] > 0
        assert not torch.equal(before,agent.model.actor[-1].weight)
        path = tmp_path/'pure.pt'
        agent.save(path,TaskOptions(),8,1)
        loaded,_,data = PurePPO.load(path)
        assert data['origin']['expert_action_labels'] == 0
        np.testing.assert_array_equal(agent.act(np.zeros(23)),loaded.act(np.zeros(23)))
    finally:
        for env in envs:
            env.close()


def test_pure_training_smoke_and_configuration_guards(tmp_path):
    cfg = PPOConfig(total_timesteps=16,n_envs=2,rollout_steps=4,batch_size=4,
                    epochs=1,eval_every=2,eval_episodes=1)
    output = tmp_path/'run'
    summary = train(cfg,TaskOptions(),output)
    assert summary['additional_ppo_environment_steps'] == 16
    assert summary['origin']['imitation_loss_weight'] == 0
    assert (output/'learning_curves.png').is_file()
    with pytest.raises(FileExistsError):
        train(cfg,TaskOptions(),output)
    with pytest.raises(ValueError):
        replace(cfg,total_timesteps=17).validate()


def test_action_hold_sums_actual_physics_and_stops_at_original_deadline():
    env = LearningTask(AerialSprayingEnv(EnvironmentConfig(max_episode_time=.25)),
                       TaskOptions(action_repeat=4,reward='original',features='goal'))
    try:
        observation,_ = env.reset(seed=8)
        base = env.unwrapped
        np.testing.assert_allclose(observation[15:18],base.reference_velocity-observation[3:6])
        _,reward,terminal,_,info = env.step(np.zeros(4))
        assert not terminal and info['physics_steps']==4
        assert np.isclose(reward,.1*sum(base.history['reward']))
        _,_,terminal,_,info = env.step(np.zeros(4))
        assert terminal and info['physics_steps']==1 and base.step_count==5
    finally:
        env.close()


def test_training_miss_abort_is_disabled_in_final_quarter():
    env = LearningTask(AerialSprayingEnv(),TaskOptions(early_miss_abort=True))
    try:
        for fraction,expected in [(0.,True),(.8,False)]:
            env.set_fraction(fraction)
            env.reset(seed=7)
            base = env.unwrapped
            base.time,base.step_count = 8.,160
            base.reference_position,base.reference_velocity,base.current_waypoint,base.progress = base.trajectory.reference(8.)
            _,_,terminal,_,info = env.step(np.zeros(4))
            assert terminal==expected
            assert info['is_success'] is False
            assert (info['termination_reason']=='training_waypoint_lag')==expected
    finally:
        env.close()
