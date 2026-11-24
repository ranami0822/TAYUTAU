"""Basic checks for the ESN and the Lorenz generator. Run: python -m pytest -q reservoir-py"""
import numpy as np

from esn import ESN
from lorenz import generate, lorenz_rhs


def test_spectral_radius_is_set():
    esn = ESN(n_in=1, n_res=200, spectral_radius=0.7, seed=1)
    rho = np.max(np.abs(np.linalg.eigvals(esn.A)))
    assert np.isclose(rho, 0.7, atol=1e-6)


def test_states_bounded_by_tanh_with_full_leak():
    esn = ESN(n_in=1, n_res=50, seed=2)
    S = esn.run(np.random.default_rng(0).normal(size=(500, 1)) * 10)
    assert np.all(np.abs(S) <= 1.0)


def test_closed_loop_reproduces_sine():
    t = np.arange(3000) * 0.1
    x = np.sin(t)[:, None]
    esn = ESN(n_in=1, n_res=100, spectral_radius=0.9, input_scale=0.5, seed=3)
    esn.fit(x[:-1], x[1:], washout=100, beta=1e-8)
    r = esn.run(x[-200:-1])[-1]  # drive with real samples up to index -2
    pred = esn.predict_closed(r, 100)[:, 0]
    # pred[k] predicts the sample at index -1 + k, i.e. sin(t[-1] + 0.1 k)
    truth = np.sin(t[-1] + 0.1 * np.arange(100))
    assert np.max(np.abs(pred - truth)) < 0.05


def test_lorenz_rhs_fixed_point():
    # (x, y, z) = (sqrt(beta*(rho-1)), same, rho-1) is a fixed point of the standard Lorenz system
    b, r = 8.0 / 3.0, 28.0
    p = np.array([np.sqrt(b * (r - 1)), np.sqrt(b * (r - 1)), r - 1])
    assert np.allclose(lorenz_rhs(p), 0.0, atol=1e-9)


def test_lorenz_stays_bounded():
    z = generate(2000, dt=0.02, transient=1000)
    assert np.all(np.isfinite(z)) and np.max(np.abs(z)) < 100
