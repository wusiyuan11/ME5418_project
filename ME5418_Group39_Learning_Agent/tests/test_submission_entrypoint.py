"""Checks for the compact submission's configured public training entry point."""
import json
from pathlib import Path
import pytest
from run_learning_agent import load_configuration


CONFIG = Path(__file__).resolve().parents[1] / 'configs/mission_ppo.json'


def test_submission_configuration_agrees_with_policy_and_task():
    policy, task = load_configuration(CONFIG)
    assert policy.features == task.features == 'goal'
    assert policy.action_repeat == task.action_repeat == 2
    assert policy.gamma == .9975
    assert policy.n_envs * policy.rollout_steps == 2048
    assert task.waypoint_bonus == 10 and task.timeout_penalty == 30


def test_submission_configuration_rejects_unknown_keys(tmp_path):
    data = json.loads(CONFIG.read_text())
    data['ppo']['misspelled_gamma'] = .99
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(data))
    with pytest.raises(TypeError):
        load_configuration(path)


def test_submission_configuration_rejects_mismatched_action_period(tmp_path):
    data = json.loads(CONFIG.read_text())
    data['task']['action_repeat'] = 4
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='must match'):
        load_configuration(path)


def test_submission_configuration_rejects_validation_seed_overlap(tmp_path):
    data = json.loads(CONFIG.read_text())
    data['ppo']['seed'] = 60
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='disjoint'):
        load_configuration(path)
