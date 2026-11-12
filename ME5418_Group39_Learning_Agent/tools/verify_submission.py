"""Read-only source, model, metric and manifest checks; no training is performed."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    inheritance = json.loads((ROOT / 'INHERITANCE.json').read_text())
    for item in inheritance['inherited_files']:
        if item['path'].startswith('environment/') or item['path'] == 'rl/network.py':
            assert item['unchanged'], item['path']
            assert digest(ROOT / item['path']) == item['stage2_sha256'], item['path']
    for name in ('agent', 'collector', 'learning', 'task', 'train'):
        tree = ast.parse((ROOT / f'pure_rl/{name}.py').read_text())
        for node in ast.walk(tree):
            imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            assert not any(item.startswith(('optimization', 'demo')) for item in imports)
    expected = {
        'foundation_ppo.pt':'7d34f02f51aad669c455835cf7c315c0b3ad353dd86bccac739a6726e4dfe883',
        'mission_ppo.pt':'d3d1cec463fbf179001ff290192e896f83f37e7e2187a7911e1933f3feeeb605',
    }
    for filename, checksum in expected.items():
        assert digest(ROOT / 'models' / filename) == checksum, filename
    model = torch.load(ROOT / 'models/mission_ppo.pt', weights_only=True, map_location='cpu')
    assert model['format_version'] == 'pure_ppo_v1'
    assert model['model']['actor.0.weight'].shape == (64, 23)
    assert model['total_timesteps'] == 245760 and model['physics_timesteps'] == 490722
    for key in ('teacher_environment_steps', 'expert_action_labels', 'imitation_loss_weight'):
        assert model['origin'][key] == 0
    assert model['origin']['source_sha256'] == expected['foundation_ppo.pt']
    for phase, expected_rows, expected_successes in (
        ('heldout_50', 400, {'pure_turn_repeat2_warm_seed7':50,
                          'pure_turn_repeat2_warm_seed8':36,
                          'pure_turn_repeat2_fresh_seed7':0, 'tuned_classical':50}),
        ('strong_wind_20', 60, {'pure_turn_repeat2_warm_seed7':17, 'tuned_classical':20}),
    ):
        with (ROOT / f'evidence/{phase}/episodes.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_rows
        summary = json.loads((ROOT / f'evidence/{phase}/summary.json').read_text())['controllers']
        for controller, successes in expected_successes.items():
            subset = [row for row in rows if row['controller'] == controller]
            assert sum(row['success'] == 'True' for row in subset) == successes
            assert summary[controller]['successes'] == successes
            for metric in ('tracking_rmse_m', 'return'):
                mean = sum(float(row[metric]) for row in subset) / len(subset)
                assert abs(mean - summary[controller]['mean'][metric]) < 1e-9
    manifest = ROOT / 'SUBMISSION_MANIFEST.json'
    if manifest.is_file():
        files = json.loads(manifest.read_text())['files']
        for name, checksum in files.items():
            assert digest(ROOT / name) == checksum, name
        print(f'Content manifest: {len(files)} file hashes PASS')
    print('Inherited physics/network, teacher-free checkpoint lineage and recorded metrics: PASS')


if __name__ == '__main__':
    verify()
