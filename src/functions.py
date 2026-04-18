import numpy as np

def quadratic(x, y):
    """
    Simple convex function: f(x,y) = x^2 + y^2
    Global minimum at (0, 0)
    Gradient: df/dx = 2x, df/dy = 2y
    """
    return x**2 + y**2

def quadratic_gradient(x, y):
    return np.array([2*x, 2*y])

def rosenbrock(x, y):
    """
    Classic non-convex test function.
    f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    Global minimum at (1, 1) where f = 0
    This is hard for optimizers because the valley is narrow and curved.
    """
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """
    df/dx = -2(1-x) - 400x(y - x^2)
    df/dy = 200(y - x^2)
    """
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])