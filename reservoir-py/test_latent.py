"""Tests for NVAR and the latent (PCA) pipeline. Run: python -m pytest -q reservoir-py"""
import numpy as np

from latent import PCA, field_error, run_case, valid_steps
from lorenz import LYAPUNOV_MAX, generate
from nvar import NVAR


def test_nvar_feature_count():
    m = NVAR(k=2, s=1)
    lin = np.arange(6.0)  # k*d = 6
    assert len(m.features(lin)) == 1 + 6 + 6 * 7 // 2


def test_nvar_predicts_lorenz_for_a_few_lyapunov_times():
    dt = 0.02
    Z = generate(4000, dt=dt)
    mu, sd = Z[:3000].mean(0), Z[:3000].std(0)
    Zn = (Z - mu) / sd
    m = NVAR(k=2, s=1).fit(Zn[:3000], beta=1e-6)
    P = m.predict_closed(Zn[2990:3000], 400)
    err = np.sqrt(np.mean((P - Zn[3000:3400]) ** 2, axis=1))
    lt = valid_steps(err) * dt * LYAPUNOV_MAX
    assert lt > 2.0  # hand-picked bar for a quadratic NVAR on dt = 0.02 data (not a paper value)


def test_pca_recovers_low_rank_data_exactly():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 50)) + 5.0
    p = PCA().fit(A, 3)
    assert np.allclose(p.decode(p.encode(A)), A, atol=1e-9)
    assert p.energy > 1 - 1e-12


def test_latent_esn_predicts_a_travelling_wave():
    # a smooth travelling wave on a 1D "field": low rank (2 modes), periodic
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    t = np.arange(3000) * 0.05
    F = np.sin(x[None, :] - t[:, None]) + 0.5 * np.sin(2 * (x[None, :] - 1.3 * t[:, None]))
    F = (F - F.mean()) / F.std()
    res = run_case(F, 2000, r=4, model="esn", horizon=200, warm=100, esn_kw=dict(n_res=200), beta=1e-8)
    assert not res["diverged"]
    assert min(res["valid"]) == 200  # stays under the error threshold over the whole horizon


def test_field_error_is_relative():
    a = np.ones((5, 10))
    assert np.allclose(field_error(a, a), 0)
    assert np.allclose(field_error(2 * a, a), 1)
