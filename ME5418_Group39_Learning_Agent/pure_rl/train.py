"""Train PPO from rewards only, with optional pure-PPO warm start / curriculum."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from environment import AerialSprayingEnv,EnvironmentConfig
from rl.evaluation import evaluate_policy,summarize,write_json
from train import append_rows
from .agent import PurePPO,PPOConfig
from .collector import BatchedCollector
from .task import LearningTask,TaskOptions,mission_observation
from .learning import update
from .plots import plot_run


def evaluate(agent,seeds,output=None,animation=False,name='pure_ppo'):
    # Crucial train/test boundary: use AerialSprayingEnv directly, NOT
    # LearningTask. Same 23 public mission features and saved action rate, but
    # no assisted reset, weak-wind curriculum, miss cutoff or shaped reward.
    class HeldPolicy:
        def reset(self):
            self.counter = 0

        def __call__(self,obs,env):
            if self.counter%agent.config.action_repeat==0:
                self.action = agent.act(mission_observation(obs,env,agent.config.features))
            self.counter += 1
            return self.action
    return evaluate_policy(HeldPolicy(),EnvironmentConfig(),seeds,name,output,animation)


def selection_key(summary):
    # Ordered completion comes before tracking precision: a small RMSE alone
    # can hide a missed required waypoint. Do not select on test-set results.
    failure = summary['rates']['crashed']+summary['rates']['boundary_violation']
    return (summary['rates']['success'],summary['mean']['waypoints_achieved'],
            -failure,-summary['mean']['tracking_rmse_m'])


def train(config,options,output,warm_start=None):
    config.validate()
    options.validate()
    if options.features!=config.features or options.action_repeat!=config.action_repeat:
        raise ValueError('Task and policy observation/control-rate settings must agree')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    agent = PurePPO(config)
    if warm_start:
        # Optional two-stage RL. Without this flag the run starts at random.
        agent.warm_start_pure_only(warm_start)
    output.mkdir(parents=True)
    write_json(output/'config.json',{'ppo':asdict(config),'training_task':asdict(options),
        'evaluation_environment':asdict(EnvironmentConfig()),'origin':agent.origin,
        'control_rate_hz':20/config.action_repeat,'physics_rate_hz':20,
        'budget_unit':'policy decisions; additional physics steps logged separately',
        'notes':'Curriculum successes are not original-task successes.'})
    total,iteration,best_key,best_step = 0,0,None,0
    completed_count,physics_total = 0,0
    start = time.perf_counter()

    def validate(stage):
        nonlocal best_key,best_step
        records = evaluate(agent,range(config.eval_seed,config.eval_seed+config.eval_episodes))
        summary = summarize(records)
        key = selection_key(summary)
        row = {'total_timesteps':total,'stage':stage,'mean_return':summary['mean']['return'],
            'std_return':summary['std']['return'],'success_rate':summary['rates']['success'],
            'waypoints_achieved':summary['mean']['waypoints_achieved'],
            'tracking_rmse_m':summary['mean']['tracking_rmse_m'],
            'failure_rate':summary['rates']['crashed']+summary['rates']['boundary_violation']}
        append_rows(output/'validation.csv',[row])
        write_json(output/'validation'/f'step_{total:09d}.json',{'summary':summary,'episodes':records})
        if best_key is None or key>best_key:
            best_key,best_step = key,total
            agent.save(output/'best.pt',options,total,iteration,physics_total)
        print('validation',row,flush=True)

    agent.save(output/'initial.pt',options,0,0)
    # Include step zero: if training fails to beat initialization, the logs and
    # best checkpoint must show that rather than implying a learned gain.
    validate('initial')
    envs = [LearningTask(AerialSprayingEnv(),options) for _ in range(config.n_envs)]
    try:
        collector = BatchedCollector(envs,agent.model,config.seed)
        while total<config.total_timesteps:
            fraction = total/config.total_timesteps
            for env in envs:
                env.set_fraction(fraction)
            # The curriculum fraction affects future resets/cutoffs; existing
            # episodes continue rather than being silently restarted per batch.
            steps = min(config.rollout_steps,(config.total_timesteps-total)//config.n_envs)
            batch,episodes = collector.collect(steps,config.gamma,config.gae_lambda)
            diagnostics = update(agent,batch,fraction)
            total,iteration = collector.total,iteration+1
            physics_total = collector.physics_total
            completed_count += len(episodes)
            append_rows(output/'episodes.csv',episodes)
            # Episode CSV separates transformed training_return from raw_return.
            # Update losses, validation scores and held-out scores are different
            # quantities and must not be drawn as one "training reward" curve.
            append_rows(output/'updates.csv',[{'update':iteration,'total_timesteps':total,
                'physics_timesteps':physics_total,
                'elapsed_seconds':time.perf_counter()-start,**diagnostics}])
            if iteration%config.eval_every==0 or total==config.total_timesteps:
                validate('ppo')
                agent.save(output/'latest.pt',options,total,iteration,physics_total)
            if iteration%20==0:
                print(f'PPO steps={total}/{config.total_timesteps} kl={diagnostics["approx_kl"]:.4f} '
                      f'episodes={completed_count} std={diagnostics["latent_std_mean"]:.3f}',flush=True)
    except KeyboardInterrupt:
        agent.save(output/'interrupted.pt',options,total,iteration,physics_total)
        raise
    finally:
        for env in envs:
            env.close()
    agent.save(output/'final.pt',options,total,iteration,physics_total)
    summary = {'additional_ppo_environment_steps':total,
        'budget_unit':'policy decision transitions',
        'additional_physics_steps':physics_total,'control_rate_hz':20/config.action_repeat,
        'previous_pure_ppo_timesteps':agent.origin['previous_pure_ppo_timesteps'],
        'best_validation_step':best_step,'best_validation_key':list(best_key),
        'completed_training_episodes':completed_count,'origin':agent.origin,
        'evaluation':'Original full task, original start, 2 m/s, 0.5 m tolerance, 45 s, gusts 3 m/s',
        'curriculum_caveat':'Training late-route subtask successes are not full-task successes',
        'elapsed_seconds':time.perf_counter()-start}
    write_json(output/'training_summary.json',summary)
    plot_run(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm-start-pure',type=Path)
    parser.add_argument('--total-timesteps',type=int,default=600000)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--reward',choices=['original','aligned'],default='aligned')
    parser.add_argument('--curriculum',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--eval-every',type=int,default=40)
    parser.add_argument('--eval-episodes',type=int,default=5)
    parser.add_argument('--action-repeat',type=int,default=1)
    parser.add_argument('--features',choices=['mission','goal'],default='mission')
    parser.add_argument('--early-miss-abort',action='store_true')
    parser.add_argument('--gamma',type=float,default=.999)
    parser.add_argument('--waypoint-bonus',type=float,default=1.)
    parser.add_argument('--timeout-penalty',type=float,default=4.)
    parser.add_argument('--success-bonus',type=float,default=8.)
    parser.add_argument('--progress-weight',type=float,default=.5)
    parser.add_argument('--goal-distance-weight',type=float,default=0.)
    args = parser.parse_args()
    cfg = PPOConfig(total_timesteps=args.total_timesteps,seed=args.seed,
        learning_rate=args.learning_rate,eval_every=args.eval_every,eval_episodes=args.eval_episodes,
        action_repeat=args.action_repeat,features=args.features,gamma=args.gamma)
    options = TaskOptions(reward=args.reward,curriculum=args.curriculum,features=args.features,
        action_repeat=args.action_repeat,early_miss_abort=args.early_miss_abort,
        waypoint_bonus=args.waypoint_bonus,timeout_penalty=args.timeout_penalty,
        success_bonus=args.success_bonus,progress_weight=args.progress_weight,
        goal_distance_weight=args.goal_distance_weight)
    print(json.dumps(train(cfg,options,
                           args.output,args.warm_start_pure),indent=2))


if __name__=='__main__':
    main()
