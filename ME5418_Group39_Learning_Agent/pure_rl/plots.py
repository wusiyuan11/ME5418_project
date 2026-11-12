"""Full-task validation is kept separate from curriculum subtask outcomes."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def plot_run(directory):
    root = Path(directory)
    vals,episodes,updates = [read(root/name) for name in ('validation.csv','episodes.csv','updates.csv')]
    figure,axes = plt.subplots(2,2,figsize=(12,7))
    x = [int(row['total_timesteps']) for row in vals]
    axes[0,0].plot(x,[100*float(row['success_rate']) for row in vals],'o-',label='Strict full mission success')
    axes[0,0].plot(x,[10*float(row['waypoints_achieved']) for row in vals],'s--',label='Ordered waypoint fraction')
    axes[0,0].set(ylabel='Percent (%)',ylim=(-2,105),title='Original start / full wind / original success test')
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(x,[float(row['tracking_rmse_m']) for row in vals],'o-')
    axes[0,1].set(ylabel='Timed-reference RMSE (m)',title='Full-task deterministic validation')
    full = [row for row in episodes if row['original_start']=='True']
    if full:
        reward = np.array([float(row['raw_return']) for row in full])
        steps = [int(row['total_timesteps']) for row in full]
        axes[1,0].scatter(steps,reward,s=8,alpha=.3)
        smooth = [reward[max(0,i-19):i+1].mean() for i in range(len(reward))]
        axes[1,0].plot(steps,smooth)
    axes[1,0].set(ylabel='Original reward sum',title='Training: full-start attempts, may end early (20-episode mean)')
    axes[1,1].plot([int(row['total_timesteps']) for row in updates],
                   [float(row['value_loss']) for row in updates],label='Value MSE')
    axes[1,1].set(ylabel='Value loss (symlog)',yscale='symlog',title='PPO critic diagnostic, not task reward')
    for axis in axes.flat:
        axis.set_xlabel('Additional PPO policy decisions (physics steps logged separately)')
        axis.grid(alpha=.25)
    figure.suptitle('Pure PPO; zero expert action labels and zero imitation loss',fontsize=12)
    figure.tight_layout()
    figure.savefig(root/'learning_curves.png',dpi=160)
    plt.close(figure)
