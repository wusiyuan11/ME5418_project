"""Plot measured episode returns separately from optimizer diagnostics."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def plot_training(run_directory):
    run = Path(run_directory)
    episodes, updates, evaluations = (read_rows(run / name) for name in
                                     ('episodes.csv', 'updates.csv', 'validation.csv'))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axis = axes[0, 0]
    if episodes:
        x = [int(row['total_timesteps']) for row in episodes]
        rewards = np.array([float(row['return']) for row in episodes])
        axis.scatter(x, rewards, s=9, alpha=.35, label='Stochastic training episodes')
        smoothed = [rewards[max(0, i-9):i+1].mean() for i in range(len(rewards))]
        axis.plot(x, smoothed, color='#1f77b4', label='Trailing 10-episode mean')
        axis.legend(fontsize=8)
    else:
        axis.text(.5, .5, 'No completed episodes', ha='center', transform=axis.transAxes)
    axis.set(title='Actual training episode return', ylabel='Sum of raw rewards')
    x_eval = [int(row['total_timesteps']) for row in evaluations]
    axes[0, 1].errorbar(x_eval, [float(row['mean_return']) for row in evaluations],
                      yerr=[float(row['std_return']) for row in evaluations], marker='o', capsize=3)
    axes[0, 1].set(title='Deterministic validation return', ylabel='Mean +/- population std')
    x_update = [int(row['total_timesteps']) for row in updates]
    axes[1, 0].plot(x_update, [float(row['policy_loss']) for row in updates], label='Policy surrogate')
    axes[1, 0].plot(x_update, [float(row['value_loss']) for row in updates], label='Value MSE')
    axes[1, 0].set_yscale('symlog', linthresh=.1)
    axes[1, 0].set(title='Optimizer losses (not reward)', ylabel='Loss, symlog scale')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(x_eval, [100 * float(row['success_rate']) for row in evaluations], 'o-')
    axes[1, 1].set(title='Strict ordered-waypoint validation success', ylabel='Success (%)', ylim=(-2, 102))
    for axis in axes.flat:
        axis.set_xlabel('Environment timesteps')
        axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(run / 'learning_curves.png', dpi=170)
    plt.close(figure)


def plot_evaluation(summaries, output):
    names = list(summaries)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, metric, label in zip(axes[:2], ('return', 'tracking_rmse_m'),
                                   ('Episode return', 'Timed-reference RMSE (m)')):
        axis.bar(names, [summaries[name]['mean'][metric] for name in names],
                 yerr=[summaries[name]['std'][metric] for name in names], capsize=4)
        axis.set_ylabel(label)
    axes[2].bar(names, [100 * summaries[name]['rates']['success'] for name in names])
    axes[2].set(ylabel='Strict mission success (%)', ylim=(0, 105))
    for axis in axes:
        axis.grid(axis='y', alpha=.25)
    figure.suptitle('Paired-seed exploratory comparison; PID additionally sees reference velocity', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
