"""Lorenz system data generator (RK4)."""
import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
# Maximum Lyapunov exponent for the standard parameters (literature value, not recomputed here).
LYAPUNOV_MAX = 0.9056


def lorenz_rhs(s, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def _rk4_step(s, dt):
    k1 = lorenz_rhs(s)
    k2 = lorenz_rhs(s + 0.5 * dt * k1)
    k3 = lorenz_rhs(s + 0.5 * dt * k2)
    k4 = lorenz_rhs(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def generate(n_steps, dt=0.02, x0=(1.0, 1.0, 1.0), transient=5000):
    """Return n_steps samples of the attractor, sampled every dt, after discarding a transient."""
    s = np.asarray(x0, dtype=float)
    for _ in range(transient):
        s = _rk4_step(s, dt)
    out = np.empty((n_steps, 3))
    for i in range(n_steps):
        s = _rk4_step(s, dt)
        out[i] = s
    return out
