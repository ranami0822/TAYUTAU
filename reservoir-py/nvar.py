"""Next generation reservoir computing (NVAR), Gauthier et al. 2021 (Nat. Commun. 12, 5564).

Feature vector (research/02-nvar.md):
    lin(t)   = [u(t), u(t-s), ..., u(t-(k-1)s)]            (k*d)
    nonlin(t)= unique quadratic monomials of lin(t)          (kd(kd+1)/2)
    feat(t)  = [1, lin(t), nonlin(t)]
Readout predicts the increment: u(t+1) = u(t) + W_out feat(t)  (ridge regression).

Hyperparameters used by the scripts here are chosen by hand, not taken from the paper.
"""
import numpy as np


class NVAR:
    def __init__(self, k=2, s=1):
        self.k = k
        self.s = s
        self.W = None

    @property
    def warmup(self):
        return (self.k - 1) * self.s

    def _lin(self, U, t):
        return np.concatenate([U[t - q * self.s] for q in range(self.k)])

    @staticmethod
    def _quad(x):
        i, j = np.triu_indices(len(x))
        return x[i] * x[j]

    def features(self, lin):
        return np.concatenate(([1.0], lin, self._quad(lin)))

    def fit(self, U, beta):
        """U: (T, d) training series. Learns u(t+1) - u(t) from feat(t)."""
        rows, targets = [], []
        for t in range(self.warmup, len(U) - 1):
            rows.append(self.features(self._lin(U, t)))
            targets.append(U[t + 1] - U[t])
        X = np.asarray(rows)
        Y = np.asarray(targets)
        n = X.shape[1]
        self.W = np.linalg.solve(X.T @ X + beta * np.eye(n), X.T @ Y).T
        return self

    def predict_closed(self, history, steps):
        """history: (>= warmup+1, d) last observed samples (most recent last). Returns (steps, d)."""
        assert self.W is not None, "call fit() first"
        buf = [np.asarray(h, dtype=float) for h in history[-(self.warmup + 1):]]
        out = np.empty((steps, len(buf[-1])))
        for q in range(steps):
            lin = np.concatenate([buf[-1 - m * self.s] for m in range(self.k)])
            nxt = buf[-1] + self.W @ self.features(lin)
            out[q] = nxt
            buf.append(nxt)
            if len(buf) > self.warmup + 1:
                buf.pop(0)
        return out
