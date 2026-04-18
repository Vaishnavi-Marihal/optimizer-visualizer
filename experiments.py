import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.optimizers import SGD, Momentum, RMSProp, Adam
from src.functions import (quadratic, quadratic_gradient,
                           rosenbrock, rosenbrock_gradient)
from src.visualizer import (plot_trajectories, plot_loss_curves,
                             plot_lr_sensitivity, plot_mean_std)

os.makedirs('plots', exist_ok=True)

OPTIMIZERS = [
    SGD(lr=0.01),
    Momentum(lr=0.01),
    RMSProp(lr=0.01),
    Adam(lr=0.01)
]

OPTIMIZER_CLASSES = [SGD, Momentum, RMSProp, Adam]
LABELS = ['SGD', 'Momentum', 'RMSProp', 'Adam']
SEEDS = [0, 1, 2]

print("=" * 55)
print("  OPTIMIZER VISUALIZATION EXPERIMENTS")
print("=" * 55)

# ── EXPERIMENT 1: Trajectories on Quadratic ──────────────
print("\n[1/6] Quadratic trajectories...")
plot_trajectories(
    function=quadratic,
    grad_fn=quadratic_gradient,
    optimizers=[SGD(lr=0.01), Momentum(lr=0.01),
                RMSProp(lr=0.01), Adam(lr=0.01)],
    labels=LABELS,
    start=[-1.5, 1.5],
    xlim=(-2, 2),
    ylim=(-2, 2),
    title='Optimizer Trajectories — Quadratic Function (x² + y²)',
    filename='quadratic_trajectories.png',
    minimum=(0, 0),
    n_steps=100
)

# ── EXPERIMENT 2: Trajectories on Rosenbrock ─────────────
print("[2/6] Rosenbrock trajectories...")
plot_trajectories(
    function=rosenbrock,
    grad_fn=rosenbrock_gradient,
    optimizers=[SGD(lr=0.0001), Momentum(lr=0.0001),
                RMSProp(lr=0.001), Adam(lr=0.001)],
    labels=LABELS,
    start=[-1.0, 1.0],
    xlim=(-2, 2),
    ylim=(-1, 3),
    title='Optimizer Trajectories — Rosenbrock Function (non-convex)',
    filename='rosenbrock_trajectories.png',
    minimum=(1, 1),
    n_steps=2000
)

# ── EXPERIMENT 3: Loss curves on Quadratic ───────────────
print("[3/6] Loss curves on quadratic...")
plot_loss_curves(
    function=quadratic,
    grad_fn=quadratic_gradient,
    optimizers=[SGD(lr=0.01), Momentum(lr=0.01),
                RMSProp(lr=0.01), Adam(lr=0.01)],
    labels=LABELS,
    start=[-1.5, 1.5],
    n_steps=300,
    title='Loss Curves — Quadratic Function',
    filename='quadratic_loss_curves.png'
)

# ── EXPERIMENT 4: Loss curves on Rosenbrock ──────────────
print("[4/6] Loss curves on Rosenbrock...")
plot_loss_curves(
    function=rosenbrock,
    grad_fn=rosenbrock_gradient,
    optimizers=[SGD(lr=0.0001), Momentum(lr=0.0001),
                RMSProp(lr=0.001), Adam(lr=0.001)],
    labels=LABELS,
    start=[-1.0, 1.0],
    n_steps=2000,
    title='Loss Curves — Rosenbrock Function',
    filename='rosenbrock_loss_curves.png'
)

# ── EXPERIMENT 5: Learning rate sensitivity ──────────────
print("[5/6] Learning rate sensitivity...")
plot_lr_sensitivity(
    grad_fn=quadratic_gradient,
    function=quadratic,
    OptimizerClass=Adam,
    label='Adam',
    learning_rates=[0.001, 0.01, 0.1, 0.5, 1.0],
    start=[-1.5, 1.5],
    n_steps=300,
    filename='lr_sensitivity.png'
)

# ── EXPERIMENT 6: Mean ± Std across 3 seeds ──────────────
print("[6/6] Mean ± std across seeds...")
plot_mean_std(
    function=quadratic,
    grad_fn=quadratic_gradient,
    OptimizerClasses=OPTIMIZER_CLASSES,
    labels=LABELS,
    start=[-1.5, 1.5],
    seeds=SEEDS,
    lr=0.01,
    n_steps=300,
    filename='mean_std.png'
)

print("\n" + "=" * 55)
print("  ALL DONE. Check your plots/ folder.")
print("  6 plots generated successfully.")
print("=" * 55)