"""Standard leaky echo state network (Jaeger 2001; Lukoševičius 2012).

Update:   r~(t+1) = tanh(A r(t) + W_in [bias; u(t+1)])
          r(t+1)  = (1 - leak) r(t) + leak r~(t+1)
Readout:  y(t+1)  = W_out [r; r^2]
Training: ridge regression on the readout only; A and W_in are fixed.
"""
import numpy as np


class ESN:
    def __init__(self, n_in, n_res=500, spectral_radius=0.9, input_scale=0.5,
                 sparsity=0.1, leak=1.0, bias=1.0, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.uniform(-1.0, 1.0, (n_res, n_res))
        A *= rng.random((n_res, n_res)) < sparsity
        rho_A = np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A * (spectral_radius / rho_A)
        self.W_in = rng.uniform(-input_scale, input_scale, (n_res, n_in + 1))
        self.n_res = n_res
        self.leak = leak
        self.bias = bias
        self.W_out = None

    def _step(self, r, u):
        pre = self.A @ r + self.W_in @ np.concatenate(([self.bias], u))
        return (1.0 - self.leak) * r + self.leak * np.tanh(pre)

    @staticmethod
    def _features(r):
        return np.concatenate((r, r ** 2))

    def run(self, U, r0=None):
        """Open-loop drive with input sequence U (T x n_in). Returns states (T x n_res)."""
        r = np.zeros(self.n_res) if r0 is None else r0
        S = np.empty((len(U), self.n_res))
        for t, u in enumerate(U):
            r = self._step(r, u)
            S[t] = r
        return S

    def fit(self, U, Y, washout, beta):
        """Ridge-regress Y (targets, T x n_out) on the features of states driven by U."""
        S = self.run(U)
        X = np.hstack([S, S ** 2])[washout:]
        T = Y[washout:]
        n = X.shape[1]
        # Ridge as augmented least squares: min ||X W - T||^2 + beta ||W||^2
        Xa = np.vstack([X, np.sqrt(beta) * np.eye(n)])
        Ta = np.vstack([T, np.zeros((n, T.shape[1]))])
        W = np.linalg.lstsq(Xa, Ta, rcond=None)[0]
        self.W_out = W.T
        return self

    def predict_closed(self, r, steps):
        """Free-running prediction: feed each output back as the next input.

        r: reservoir state after the last observed input (as returned by run()[-1]).
        Returns outputs (steps x n_out); outputs[k] predicts the sample k+1 steps after r.
        """
        assert self.W_out is not None, "call fit() first"
        outs = np.empty((steps, self.W_out.shape[0]))
        for k in range(steps):
            y = self.W_out @ self._features(r)
            outs[k] = y
            r = self._step(r, y)
        return outs
