import numpy as np

class SGD:
    """
    Stochastic Gradient Descent.
    Update rule: w = w - lr * gradient
    Simplest optimizer. No memory of past gradients.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, w, grad):
        return w - self.lr * grad


class Momentum:
    """
    SGD with Momentum.
    Keeps a running average of past gradients (velocity).
    Update rule:
        v = beta * v - lr * gradient
        w = w + v
    beta=0.9 means 90% of previous velocity is kept.
    Helps escape flat regions and oscillations.
    """
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def update(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.beta * self.v - self.lr * grad
        return w + self.v


class RMSProp:
    """
    RMSProp: Root Mean Square Propagation.
    Keeps running average of squared gradients.
    Update rule:
        s = beta * s + (1 - beta) * gradient^2
        w = w - lr * gradient / (sqrt(s) + epsilon)
    Dividing by sqrt(s) scales down large gradients automatically.
    Good for non-convex problems.
    """
    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def update(self, w, grad):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s = self.beta * self.s + (1 - self.beta) * grad**2
        return w - self.lr * grad / (np.sqrt(self.s) + self.epsilon)


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    Combines Momentum (first moment) and RMSProp (second moment).
    Update rule:
        m = beta1 * m + (1 - beta1) * gradient          (first moment)
        v = beta2 * v + (1 - beta2) * gradient^2        (second moment)
        m_hat = m / (1 - beta1^t)                       (bias correction)
        v_hat = v / (1 - beta2^t)                       (bias correction)
        w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
    Bias correction fixes the fact that m and v start at zero.
    Most popular optimizer in modern deep learning.
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)