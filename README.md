# Gradient Descent Optimizer Visualization
## From Mathematical Functions to Empirical Analysis

A from-scratch NumPy implementation and visual analysis of four
gradient-based optimization algorithms on convex and non-convex
loss surfaces.

---

## Optimizers Implemented

All optimizers are implemented from scratch using NumPy only.
No PyTorch, no TensorFlow, no autograd.

SGD
Update rule: w = w - lr * grad

Momentum
v  = beta * v - lr * grad
w  = w + v

RMSProp
s  = beta * s + (1 - beta) * grad^2
w  = w - lr * grad / (sqrt(s) + eps)

Adam
m  = beta1 * m + (1 - beta1) * grad
v  = beta2 * v + (1 - beta2) * grad^2
m^ = m / (1 - beta1^t)
v^ = v / (1 - beta2^t)
w  = w - lr * m^ / (sqrt(v^) + eps)

---

## Test Functions

### 1. Quadratic — f(x,y) = x² + y²
Convex. Global minimum at (0,0).
Gradient: [2x, 2y]
Used to verify basic convergence behavior.

### 2. Rosenbrock — f(x,y) = (1-x)² + 100(y-x²)²
Non-convex. Global minimum at (1,1) where f=0.
Gradient:
  df/dx = -2(1-x) - 400x(y-x²)
  df/dy =  200(y-x²)
Classic benchmark for optimizer comparison.
The narrow curved valley makes this hard for naive gradient descent.

---

## Experiments

### Experiment 1 and 2: Optimization Trajectories
Visualize the path each optimizer takes through parameter space
overlaid on a contour map of the loss surface.

### Experiment 3 and 4: Loss Curves
Track loss value at every step for all 4 optimizers.
Y-axis is log scale to show convergence differences clearly.

### Experiment 5: Learning Rate Sensitivity
Run Adam with learning rates [0.001, 0.01, 0.1, 0.5, 1.0].
Shows how lr controls convergence speed and stability.
Mathematical insight: if lr is too large, updates overshoot
the minimum and loss diverges.

### Experiment 6: Mean and Std across 3 Seeds
Each optimizer is run from 3 slightly different starting points.
Mean loss is plotted with shaded standard deviation band.
Shows robustness and stability of each optimizer.

---

## Key Observations

- Adam and RMSProp converge faster than SGD on both functions
- SGD with Momentum avoids oscillations better than plain SGD
- On Rosenbrock, Adam navigates the curved valley most efficiently
- Learning rate above 0.5 causes Adam to diverge on quadratic
- Std bands show Adam is most stable across different initializations

---

## Tech Stack

- Python 3.11
- NumPy (all math, no ML frameworks)
- Matplotlib (all plots)

---

## How to Run

git clone https://github.com/YOURUSERNAME/optimizer-visualizer
cd optimizer-visualizer
python -m venv venv
venv\Scripts\activate
pip install numpy matplotlib scipy
python experiments.py

Plots saved to plots/ folder.

---

## Project Structure

optimizer-visualizer/
├── src/
│   ├── functions.py       # quadratic and rosenbrock + gradients
│   ├── optimizers.py      # SGD, Momentum, RMSProp, Adam from scratch
│   └── visualizer.py      # all plotting functions
├── plots/                 # generated plots
├── experiments.py         # runs all 6 experiments
└── README.md