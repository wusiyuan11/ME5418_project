"""Rebuild clear tables/figures from recorded CSV/JSON, without retraining.

Run: python -m tools.summarize_results
The output is descriptive evidence, not a fresh held-out test or new selection.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    ('original_ppo_300k', '15D PPO, 300k run'),
    ('original_ppo_1m', '15D PPO, ~1M run'),
    ('pure_turn_repeat4_warm_seed7', '23D PPO, hold 4'),
    ('pure_turn_repeat2_warm_seed7', '23D PPO, seed 7'),
    ('pure_turn_repeat2_warm_seed8', '23D PPO, seed 8'),
    ('pure_turn_repeat2_fresh_seed7', '23D PPO, random start'),
    ('original_pid', 'Original PD/PID'),
    ('tuned_classical', 'Tuned PD/PID'),
]
MAIN = 'pure_turn_repeat2_warm_seed7'


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def mean_line(axis, x, values, label):
    """Keep actual episode points visible; smoothing is explicitly labelled."""
    axis.scatter(x, values, s=7, alpha=.22, color='#657c92', label='Recorded episode sum')
    window = min(20, len(values))
    if window:
        average = np.convolve(values, np.ones(window)/window, mode='valid')
        axis.plot(x[window-1:], average, color='#1f4f7b', lw=1.7,
                  label=f'Trailing {window}-episode mean')
    axis.set_ylabel(label)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, loc='best')


def main():
    destination = ROOT / 'results'
    destination.mkdir(exist_ok=True)
    record = json.loads((ROOT / 'evidence/heldout_50/summary.json').read_text())
    summaries = record['controllers']
    rows = []
    for key, label in ORDER:
        item = summaries[key]
        rows.append({'controller':key, 'label':label,
                     'episodes':item['episodes'], 'successes':item['successes'],
                     'success_percent':100*item['rates']['success'],
                     'mean_ordered_waypoints':item['mean']['waypoints_achieved'],
                     'timed_reference_3d_rmse_m':item['mean']['tracking_rmse_m'],
                     'rmse_std_m':item['std']['tracking_rmse_m'],
                     'original_evaluation_return':item['mean']['return'],
                     'return_std':item['std']['return']})
    with (destination / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Separate success from timed-reference accuracy. Neither alone describes
    # completion plus tracking quality. Keep failures and strong PID visible.
    selected = [row for row in rows if row['controller'] not in
                ('pure_turn_repeat4_warm_seed7', 'original_pid')]
    y = np.arange(len(selected))
    colors = ['#1f4f7b' if row['controller']==MAIN else '#a6b4c0' for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    axes[0].barh(y, [row['success_percent'] for row in selected], color=colors)
    axes[1].barh(y, [row['timed_reference_3d_rmse_m'] for row in selected], color=colors)
    axes[0].set_yticks(y, [row['label'] for row in selected])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 125), xlabel='Success (%) - higher is better', title='Full original-task completion')
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[1].set(xlim=(0, 3.1), xlabel='3-D RMSE (m) - lower is better', title='Timed-reference tracking error')
    for i, row in enumerate(selected):
        axes[0].text(row['success_percent']+2, i,
                     f"{row['successes']}/{row['episodes']}", va='center', fontsize=9)
        axes[1].text(row['timed_reference_3d_rmse_m']+.04, i,
                     f"{row['timed_reference_3d_rmse_m']:.3f}", va='center', fontsize=9)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis='x', alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Held-out original task: 50 paired seeds (70000-70049)', fontsize=12)
    fig.text(.02, .015,
             'Blue: selected teacher-free PPO. Continuations share one foundation; budgets/rates differ.\n'
             'Evaluation has no curriculum. These plots do not establish superiority over tuned PID.', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(destination / 'heldout_comparison.png', dpi=180)
    plt.close(fig)

    # Training attempts are stochastic and may have an early miss cutoff. Do
    # not relabel this curve as deterministic full-mission evaluation reward.
    attempts = [row for row in load_csv(ROOT / 'evidence/selected_run/episodes.csv')
                if row['original_start'].lower()=='true']
    x = np.array([int(row['total_timesteps']) for row in attempts])
    shaped = np.array([float(row['training_return']) for row in attempts])
    raw = np.array([float(row['raw_return']) for row in attempts])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
    mean_line(axes[0], x, shaped, 'Shaped training return')
    mean_line(axes[1], x, raw, 'Original raw return')
    axes[1].set_xlabel('Policy decisions across eight clients (not physics steps)')
    fig.suptitle('Selected run: stochastic training attempts from the original start', fontsize=11)
    fig.text(.02, .015, 'Partial-route starts excluded. Early full-start attempts may still use curriculum wind/cutoffs.\n'
             'Rewards have different definitions; use independent evaluation for performance comparison.', fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(destination / 'training_rewards.png', dpi=180)
    plt.close(fig)

    lines = [
        '# Results guide - waypoint-aware PPO framework',
        '',
        'These tables and plots use the recorded experiment data; this revision adds comments and clearer presentation, not new long training.',
        '',
        '## What changed since the Neural Network submission?',
        '',
        'The Oct-30 environment already counted ordered arrivals, but its policy did not observe that history. The third milestone exposes c/N (completed-waypoint fraction) and the error to waypoint[c] (next outstanding goal), plus velocity error and remaining time. It adds sustained PPO training and task-aligned feedback. This is not a new waypoint list or a relaxed success test.',
        '',
        'The previous submission only demonstrated 128 transitions and one update. It did not report a trained policy success rate. The 15D policies below were trained later as baselines; do not label their scores as the second submission\'s demonstrated performance.',
        '',
        '## 1. Held-out original task: completion and tracking are different metrics',
        '',
        '50 paired seeds 70000-70049; original rest start, full wind/payload variation, 2 m/s reference, 45 s limit, all 10 ordered waypoints within 0.5 m in 3-D. Start counts once without reward. No training reset assistance, early cutoff or shaped test reward.',
        '',
        '| Controller | Success | Mean ordered waypoints / 10 | Timed 3-D RMSE (m) | Original evaluation return |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} ({row['success_percent']:.0f}%) | "
                     f"{row['mean_ordered_waypoints']:.2f} | {row['timed_reference_3d_rmse_m']:.3f} | "
                     f"{row['original_evaluation_return']:.3f} |")
    lines += ['', '![Held-out completion and tracking](heldout_comparison.png)', '',
              '**Interpretation:** the selected two-stage PPO completes all 50 sampled missions, but tuned PID also completes 50/50 and tracks the timed reference more accurately. The two continuation seeds share a foundation, the random-start variant fails, and no causal ablation isolates the new inputs. Unequal budgets/control rates and different episode lengths limit ranking claims. 50/50 Wilson 95% interval is 92.9%-100%.',
              '', '## 2. Learning progress on fixed original-task validation', '',
              'These are five-seed deterministic full-mission validation scores, not stochastic training reward and not the held-out 50-seed test.', '',
              '| New policy decisions | Mean ordered waypoints / 10 | Success | Original validation return |',
              '|---:|---:|---:|---:|']
    for row in load_csv(ROOT / 'evidence/selected_run/validation.csv'):
        lines.append(f"| {int(row['total_timesteps'])} | {float(row['waypoints_achieved']):.2f} | "
                     f"{100*float(row['success_rate']):.0f}% | {float(row['mean_return']):.3f} |")
    lines += ['', '## 3. Actual training reward', '',
              '![Training reward definitions shown separately](training_rewards.png)', '',
              '`training_return` sums the shaped training feedback; `raw_return` sums the inherited unscaled reward. Only original-start attempts are plotted, and early attempts can terminate under the training cutoff. Partial curriculum starts are excluded. Changing episode durations/distributions prevent a simple early-vs-late mean from being a matched performance estimate.',
              '', '## 4. Strong-wind check after model selection', '',
              '20 new paired seeds 80000-80019, max base wind 4.5 m/s rather than training 3 m/s, retaining the existing 0.7 m/s gust term. Not used to select the model.', '',
              '| Controller | Success | Timed 3-D RMSE (m) | Original evaluation return |',
              '|---|---:|---:|---:|']
    stress = json.loads((ROOT / 'evidence/strong_wind_20/summary.json').read_text())['controllers']
    for key, label in ((MAIN, 'Selected two-stage PPO'), ('original_pid', 'Original PD/PID'), ('tuned_classical', 'Tuned PD/PID')):
        item = stress[key]
        lines.append(f"| {label} | {item['successes']}/{item['episodes']} | "
                     f"{item['mean']['tracking_rmse_m']:.3f} | {item['mean']['return']:.3f} |")
    lines += ['', '## Remaining limits', '',
              'Reference still advances by time even after a miss. The new observation exposes the uncompleted obligation; it does not remove reference-vs-arrival conflicts. No independent end-to-end multi-seed convergence, new-route generalization, real spray coverage, physical energy saving or real-flight safety claim is established.',
              '', 'Source data: `evidence/heldout_50/episodes.csv`, `summary.json`; `evidence/selected_run/validation.csv`, `episodes.csv`; `evidence/strong_wind_20/summary.json`. Full logs and negative trial histories are retained.']
    (destination / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Rebuilt {len(rows)} controller rows and figures from recorded data: {destination}')


if __name__ == '__main__':
    main()
