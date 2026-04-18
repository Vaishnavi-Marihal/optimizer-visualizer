# Mathematical Notes
## Gradient Descent Optimizer Analysis

---

## 1. Gradient Descent — Core Idea

Optimization goal: find w* such that f(w*) is minimized.

Gradient descent iteratively updates parameters in the direction
of steepest descent:

    w_{t+1} = w_t - lr * ∇f(w_t)

where ∇f(w_t) is the gradient of the loss with respect to w at step t.

The gradient points in the direction of steepest INCREASE.
Subtracting it moves us toward the minimum.

---

## 2. Test Functions and Their Gradients

### 2.1 Quadratic Function

    f(x, y) = x² + y²

This is a convex bowl. One global minimum at (0, 0).

Partial derivatives:
    ∂f/∂x = 2x
    ∂f/∂y = 2y

Gradient vector:
    ∇f(x, y) = [2x, 2y]

This is the simplest possible test case.
Gradient descent is guaranteed to converge here for small enough lr.

Convergence condition on quadratic:
    lr < 1/L
where L is the Lipschitz constant of the gradient.
For f(x,y) = x² + y², L = 2, so lr < 0.5 guarantees convergence.

### 2.2 Rosenbrock Function

    f(x, y) = (1 - x)² + 100(y - x²)²

Non-convex. Global minimum at (1, 1) where f(1,1) = 0.

This function has a narrow curved valley that leads to the minimum.
The challenge: gradients along the valley floor are tiny,
but gradients across the valley are large.
This causes oscillation for naive gradient descent.

Partial derivatives:
    ∂f/∂x = -2(1 - x) - 400x(y - x²)
    ∂f/∂y = 200(y - x²)

Derivation of ∂f/∂x step by step:

Let A = (1 - x)²  and  B = 100(y - x²)²

∂A/∂x = 2(1 - x) * (-1) = -2(1 - x)

∂B/∂x = 100 * 2(y - x²) * (-2x) = -400x(y - x²)

Therefore:
    ∂f/∂x = -2(1 - x) - 400x(y - x²)

Derivation of ∂f/∂y:

    ∂f/∂y = 100 * 2(y - x²) * 1 = 200(y - x²)

---

## 3. Optimizer Mathematics

### 3.1 SGD (Stochastic Gradient Descent)

    w_{t+1} = w_t - lr * g_t

where g_t = ∇f(w_t)

Simplest update. No memory of past gradients.
Problem: slow convergence, sensitive to lr, oscillates in narrow valleys.

### 3.2 Momentum

Introduces velocity vector v to accumulate gradient history:

    v_t     = beta * v_{t-1} - lr * g_t
    w_{t+1} = w_t + v_t

Typical beta = 0.9 means 90% of previous velocity is retained.

Physical analogy: a ball rolling down a hill builds up speed
(momentum) in consistent directions and dampens oscillations.

Effect on Rosenbrock:
Momentum smooths out the oscillations across the valley
and accelerates movement along the valley floor.

### 3.3 RMSProp

Maintains running average of squared gradients:

    s_t     = beta * s_{t-1} + (1 - beta) * g_t²
    w_{t+1} = w_t - lr * g_t / (sqrt(s_t) + eps)

Dividing by sqrt(s_t) performs per-parameter lr scaling.
Parameters with large gradients get smaller effective lr.
Parameters with small gradients get larger effective lr.

This is called adaptive learning rate.

eps (typically 1e-8) prevents division by zero.

### 3.4 Adam (Adaptive Moment Estimation)

Combines Momentum (1st moment) and RMSProp (2nd moment):

First moment (mean of gradients):
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t

Second moment (mean of squared gradients):
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t²

Bias correction (both m and v are initialized at 0,
so early estimates are biased toward zero):
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)

Parameter update:
    w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)

Typical values: beta1=0.9, beta2=0.999, eps=1e-8

Why bias correction matters:
At t=1 with beta1=0.9:
    m_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1
    m_hat = 0.1 * g_1 / (1 - 0.9) = g_1

Without correction m_1 = 0.1*g_1 severely underestimates
the true gradient. Correction scales it back to g_1.

---

## 4. Learning Rate Sensitivity

Learning rate (lr) is the most critical hyperparameter.

Case 1 — lr too small:
    Updates are tiny. Convergence is extremely slow.
    Risk of getting stuck in local minima.

Case 2 — lr optimal:
    Fast, stable convergence to minimum.

Case 3 — lr too large:
    Updates overshoot the minimum.
    Loss oscillates or diverges to infinity.

Mathematical condition for divergence on quadratic f(x)=x²:
    If lr >= 1/L = 0.5, updates will overshoot.

This is visible in the lr_sensitivity.png plot:
    lr=0.001 → very slow convergence
    lr=0.01  → stable, good convergence
    lr=0.1   → fast convergence
    lr=0.5   → begins to oscillate
    lr=1.0   → diverges

---

## 5. Why Adam Wins on Rosenbrock

The Rosenbrock valley has two properties that make it hard:

1. Gradients are much larger in the y direction than x direction
2. The valley curves, so the optimal direction changes every step

SGD treats all directions equally — it struggles with (1).
Momentum helps with (2) but not (1).
RMSProp handles (1) via adaptive scaling but not (2) as well.
Adam handles both simultaneously — adaptive scaling + momentum.

This is why Adam converges fastest on Rosenbrock in our experiments.

---

## 6. Mean and Std Across Seeds — Why It Matters

Running one experiment once proves nothing.
Different initializations lead to different trajectories.

By running 3 seeds and reporting mean ± std we show:
- Mean: expected performance of the optimizer
- Std: stability and robustness of the optimizer

Low std = optimizer behaves consistently regardless of initialization.
High std = optimizer is sensitive to starting point.

In our results Adam shows lowest std, confirming it is
the most robust optimizer in this comparison.

---

## References

- Ruder, S. (2016). An overview of gradient descent optimization algorithms.
- Kingma, D., Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
- Rosenbrock, H.H. (1960). An automatic method for finding the greatest or
  least value of a function. Computer Journal, 3(3), 175-184.