"""Evaluate one saved pure-PPO model on the unchanged original full mission."""
import argparse
from pathlib import Path
from rl.evaluation import summarize,write_json,write_rows
from .agent import PurePPO
from .train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--seed-start',type=int,default=70000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--animation',action='store_true')
    args = parser.parse_args()
    if args.episodes<1 or args.seed_start<0 or args.output.exists():
        parser.error('Positive episode count, nonnegative seed and NEW output directory required')
    agent,options,payload = PurePPO.load(args.checkpoint)
    seeds = list(range(args.seed_start,args.seed_start+args.episodes))
    validation = set(range(agent.config.eval_seed,agent.config.eval_seed+agent.config.eval_episodes))
    # Reject overlap with saved training-validation seeds. Published seed sets
    # are reproducibility checks, not fresh test sets for future policy tuning.
    if validation & set(seeds):
        parser.error('Test seeds overlap training validation seeds')
    rows = evaluate(agent,seeds[:1],args.output,args.animation)
    # Save one real trajectory/replay, then collect every requested episode for
    # the aggregate. A visually good replay does not replace seed-level metrics.
    rows += evaluate(agent,seeds[1:])
    summary = summarize(rows)
    write_rows(args.output/'episodes.csv',rows)
    write_json(args.output/'summary.json',{'summary':summary,'checkpoint':str(args.checkpoint),
        'additional_policy_decisions':payload['total_timesteps'],
        'additional_physics_steps':payload.get('physics_timesteps',payload['total_timesteps']),
        'origin':payload['origin'],'task_options':payload['task_options'],
        'evaluation':'Original reset, 2 m/s, 0.5 m tolerance, full 10 waypoints, 45 s, gusts 3 m/s',
        'no_training_curriculum_in_test':True,'reward':'Original unscaled environment reward',
        'control_rate_hz':20/agent.config.action_repeat})
    print(summary)


if __name__=='__main__':
    main()
