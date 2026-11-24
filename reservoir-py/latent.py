"""Stage 6: latent prediction of the convection reference data (scripts/gen-reference.ts).

Pipeline (docs/design-memo.md 2, research/08 C):
  1. fields (T, u, v on an n x n slice) -> standardise per channel (training statistics)
  2. PCA (SVD) on the training part -> r latent coefficients
  3. predict the latent time series in closed loop with an ESN or with NVAR
  4. decode, compare with the truth: valid time = first time the normalised field error > THRESHOLD

The "divergence map" (research/08 C, Vlachas et al. 2020 warning) runs this over latent dimension r and
reservoir size and records whether the closed loop leaves the training range.
"""
import json
import os

import warnings

import numpy as np

from esn import ESN
from nvar import NVAR

THRESHOLD = 0.4


def load(path_base):
    with open(path_base + ".json") as f:
        meta = json.load(f)
    n, T = meta["n"], meta["snapshots"]
    X = np.fromfile(path_base + ".f32", dtype="<f4").reshape(T, 3, n, n).astype(np.float64)
    return X, meta


class PCA:
    def fit(self, X, r):
        """X: (T, D). Keeps r modes."""
        self.mean = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.modes = Vt[:r]
        self.energy = (S[:r] ** 2).sum() / (S ** 2).sum()
        return self

    def encode(self, X):
        return (X - self.mean) @ self.modes.T

    def decode(self, Z):
        return Z @ self.modes + self.mean


def standardise(X, n_train):
    """per-channel mean/std over the training snapshots; X: (T, C, n, n) -> (T, C*n*n)"""
    mu = X[:n_train].mean(axis=(0, 2, 3), keepdims=True)
    sd = X[:n_train].std(axis=(0, 2, 3), keepdims=True) + 1e-12
    Xs = (X - mu) / sd
    return Xs.reshape(len(X), -1)


def field_error(pred, truth):
    """normalised RMS error per time step (fields are standardised, so the scale is ~1)"""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=1)) / np.sqrt(np.mean(truth ** 2, axis=1).mean())


def valid_steps(err):
    bad = np.nonzero(err > THRESHOLD)[0]
    return int(bad[0]) if bad.size else len(err)


def run_case(F, n_train, r, model, horizon, warm=200, seed=0, esn_kw=None, beta=1e-6, nvar_kw=None, starts=3):
    """Returns dict(valid=[steps per start], diverged=bool, energy=PCA energy fraction)."""
    pca = PCA().fit(F[:n_train], r)
    Z = pca.encode(F)
    zs = Z[:n_train].std(axis=0) + 1e-12
    Zn = Z / zs
    zmax = np.abs(Zn[:n_train]).max()
    valid, diverged = [], False
    test_starts = [n_train + warm + q * (horizon // 2) for q in range(starts)]
    test_starts = [t for t in test_starts if t + horizon <= len(F)]
    if model == "esn":
        kw = dict(n_res=300, spectral_radius=0.9, input_scale=0.5, sparsity=0.05, leak=1.0)
        kw.update(esn_kw or {})
        m = ESN(n_in=r, seed=seed, **kw).fit(Zn[: n_train - 1], Zn[1:n_train], washout=100, beta=beta)
    else:
        m = NVAR(**(nvar_kw or {})).fit(Zn[:n_train], beta=beta)
    for t0 in test_starts:
        if model == "esn":
            rstate = m.run(Zn[t0 - warm : t0])[-1]
            P = m.predict_closed(rstate, horizon)
        else:
            P = m.predict_closed(Zn[t0 - m.warmup - 1 : t0], horizon)
        if not np.all(np.isfinite(P)) or np.abs(P).max() > 10 * zmax:
            diverged = True
        P = np.nan_to_num(P, nan=1e3, posinf=1e3, neginf=-1e3)
        pred = pca.decode(P * zs)
        # skill of the latent predictor alone: compare with the PCA projection of the truth, so that the
        # truncation error of the compression (reported separately) does not count as a prediction error
        target = pca.decode(Z[t0 : t0 + horizon])
        err = field_error(pred, target)
        valid.append(valid_steps(err))
    trunc = float(np.mean(field_error(pca.decode(Z[n_train:]), F[n_train:])))
    return dict(valid=valid, diverged=diverged, energy=float(pca.energy), trunc=trunc)


def persistence_valid(F, n_train, horizon, warm=200, starts=3):
    """baseline: predict that the field stays as it is (full field, no compression)"""
    out = []
    for q in range(starts):
        t0 = n_train + warm + q * (horizon // 2)
        if t0 + horizon > len(F):
            break
        err = field_error(np.repeat(F[t0 - 1 : t0], horizon, axis=0), F[t0 : t0 + horizon])
        out.append(valid_steps(err))
    return out


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # diverging closed loops overflow by design
    base = os.environ.get("TAYUTAU_REF", os.path.join(os.path.dirname(__file__), "data", "convection2d_n32"))
    X, meta = load(base)
    T = len(X)
    n_train = int(0.7 * T)
    F = standardise(X, n_train)
    dt = meta["dtSnapshot"]
    horizon = 300
    print(f"data {base}: {T} snapshots, dt={dt:.4f} s, train {n_train}, horizon {horizon} ({horizon * dt:.1f} s), threshold {THRESHOLD}")
    pers = persistence_valid(F, n_train, horizon)
    print(f"persistence baseline (full field): valid {pers} snapshots")
    for r in (4, 8, 16, 32):
        pca = PCA().fit(F[:n_train], r)
        Z = pca.encode(F)
        print(f"persistence baseline (latent r={r}): valid {persistence_valid(pca.decode(Z), n_train, horizon)} snapshots")
    rows = []
    for r in (4, 8, 16, 32):
        for n_res in (100, 300, 1000):
            res = run_case(F, n_train, r, "esn", horizon, esn_kw=dict(n_res=n_res))
            rows.append(("ESN", r, n_res, res))
        res = run_case(F, n_train, r, "nvar", horizon, nvar_kw=dict(k=2, s=1), beta=1e-4)
        rows.append(("NVAR k=2", r, "-", res))
    print(f"{'model':<9} {'r':>3} {'D_r':>5} {'PCA energy':>10} {'trunc err':>9} {'valid (snapshots)':>22} {'mean [s]':>9} {'diverged':>9}")
    for name, r, n_res, res in rows:
        v = res["valid"]
        print(f"{name:<9} {r:>3} {str(n_res):>5} {res['energy']:>10.3f} {res['trunc']:>9.3f} {str(v):>22} {np.mean(v) * dt:>9.2f} {str(res['diverged']):>9}")


if __name__ == "__main__":
    main()
