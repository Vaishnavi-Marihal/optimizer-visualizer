import numpy as np
import matplotlib.pyplot as plt
from src.functions import quadratic, rosenbrock

def get_trajectory(optimizer, grad_fn, start, n_steps=500):
    """
    Run optimizer for n_steps and record position at each step.
    Returns array of shape (n_steps, 2) — the path through parameter space.
    """
    w = np.array(start, dtype=float)
    trajectory = [w.copy()]

    for _ in range(n_steps):
        grad = grad_fn(w[0], w[1])
        w = optimizer.update(w, grad)
        trajectory.append(w.copy())

    return np.array(trajectory)


def plot_trajectories(function, grad_fn, optimizers, labels, start,
                      xlim, ylim, title, filename, minimum, n_steps=500):
    """
    Plot optimization trajectories of multiple optimizers
    on a contour map of the given function.
    """
    x = np.linspace(xlim[0], xlim[1], 300)
    y = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for opt, label, color in zip(optimizers, labels, colors):
        traj = get_trajectory(opt, grad_fn, start, n_steps)
        ax.plot(traj[:, 0], traj[:, 1], color=color,
                label=label, linewidth=1.8, alpha=0.85)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=6)

    ax.plot(*minimum, '*', color='white', markersize=14,
            markeredgecolor='black', label='Minimum', zorder=5)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_loss_curves(function, grad_fn, optimizers, labels,
                     start, n_steps=500, title="Loss curves", filename="loss_curves.png"):
    """
    Plot loss value at each step for all optimizers.
    Shows convergence speed clearly.
    """
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(9, 5))

    for opt, label, color in zip(optimizers, labels, colors):
        traj = get_trajectory(opt, grad_fn, start, n_steps)
        losses = [function(w[0], w[1]) for w in traj]
        ax.plot(losses, color=color, label=label, linewidth=2)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_lr_sensitivity(grad_fn, function, OptimizerClass, label,
                        learning_rates, start, n_steps=300,
                        filename="lr_sensitivity.png"):
    """
    Plot how different learning rates affect convergence for one optimizer.
    This shows mathematical understanding of the lr hyperparameter.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(learning_rates)))

    for lr, color in zip(learning_rates, colors):
        opt = OptimizerClass(lr=lr)
        traj = get_trajectory(opt, grad_fn, start, n_steps)
        losses = [function(w[0], w[1]) for w in traj]
        losses = np.clip(losses, 1e-10, 1e10)
        ax.plot(losses, color=color, label=f'lr={lr}', linewidth=2)

    ax.set_title(f'Learning Rate Sensitivity — {label}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


def plot_mean_std(function, grad_fn, OptimizerClasses, labels,
                  start, seeds, lr, n_steps=500,
                  filename="mean_std.png"):
    """
    Run each optimizer with multiple random starting points (seeds).
    Plot mean loss with shaded std band.
    Shows statistical robustness of each optimizer.
    """
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, ax = plt.subplots(figsize=(9, 5))

    for OptClass, label, color in zip(OptimizerClasses, labels, colors):
        all_losses = []
        for seed in seeds:
            np.random.seed(seed)
            noise = np.random.randn(2) * 0.5
            noisy_start = np.array(start) + noise
            opt = OptClass(lr=lr)
            traj = get_trajectory(opt, grad_fn, noisy_start, n_steps)
            losses = np.array([function(w[0], w[1]) for w in traj])
            losses = np.clip(losses, 1e-10, 1e10)
            all_losses.append(losses)

        all_losses = np.array(all_losses)
        mean = np.mean(all_losses, axis=0)
        std = np.std(all_losses, axis=0)

        ax.plot(mean, color=color, label=label, linewidth=2)
        ax.fill_between(range(len(mean)),
                        np.clip(mean - std, 1e-10, None),
                        mean + std,
                        color=color, alpha=0.15)

    ax.set_title('Mean ± Std Loss across 3 Seeds', fontsize=13, fontweight='bold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")