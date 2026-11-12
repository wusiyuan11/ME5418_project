"""Check pinned versions, Gymnasium API and the inherited network interface."""
import argparse
from importlib.metadata import version
import platform
from pathlib import Path
from packaging.requirements import Requirement
import yaml
from gymnasium.utils.env_checker import check_env
from environment import AerialSprayingEnv
from rl.agent import PPOAgent, TrainingConfig
from pure_rl.agent import PurePPO, PPOConfig
from pure_rl.task import mission_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-versions', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    configuration = yaml.safe_load((root / 'environment.yml').read_text())
    pip_requirements = next(d['pip'] for d in configuration['dependencies'] if isinstance(d, dict))
    declared = [Requirement(item) for item in pip_requirements if not item.startswith('--')]
    mismatches = []
    for requirement in declared:
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = version(requirement.name)
        if installed not in requirement.specifier:
            mismatches.append(f'{requirement.name}: installed {installed}, expected {requirement.specifier}')
    python_pin = next(d.split('=', 1)[1] for d in configuration['dependencies']
                      if isinstance(d, str) and d.startswith('python='))
    if platform.python_version() != python_pin:
        mismatches.append(f'Python: installed {platform.python_version()}, tested {python_pin}')
    env = AerialSprayingEnv()
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=7)
        agent = PPOAgent(TrainingConfig())
        action = agent.act(observation)
        assert action.shape == (4,) and env.action_space.contains(action)
        assert sum(p.numel() for p in agent.model.parameters()) == 10697
        env.step(action)
        observation, _ = env.reset(seed=7)
        mission_agent = PurePPO(PPOConfig(features='goal', action_repeat=2))
        features = mission_observation(observation, env, 'goal')
        mission_action = mission_agent.act(features)
        assert features.shape == (23,)
        assert mission_action.shape == (4,) and env.action_space.contains(mission_action)
        assert sum(p.numel() for p in mission_agent.model.parameters()) == 11721
        env.step(mission_action)
    finally:
        env.close()
    print(f'Python {platform.python_version()}; checked {len(declared)} dependency entries')
    if mismatches:
        print('Version differences:\n' + '\n'.join(mismatches))
        if args.strict_versions:
            raise SystemExit(1)
    print('Gymnasium, 15/23-feature interfaces and both neural policies: PASS')
    if not mismatches:
        print('Pinned runtime package versions: PASS')


if __name__ == '__main__':
    main()
