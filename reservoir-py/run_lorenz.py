"""Minimal check: standard ESN on the Lorenz system, closed-loop valid prediction time.

Hyperparameters below are starting values chosen by hand (NOT from a paper).
Valid prediction time = first time the normalized error exceeds THRESHOLD,
reported in Lyapunov times (t * LYAPUNOV_MAX).
"""
import numpy as np

from esn import ESN
from lorenz import LYAPUNOV_MAX, generate

DT = 0.02
TRAIN = 5000          # 100 time units
WASHOUT = 200
TEST_WARM = 200       # open-loop warm-up on true data before free-running
HORIZON = 1500        # 30 time units ≈ 27 Lyapunov times
THRESHOLD = 0.4       # normalized RMS error threshold (Lu et al. 2018 style)
SEEDS = range(5)
ESN_KW = dict(n_res=500, spectral_radius=0.9, input_scale=0.5, sparsity=0.1, leak=1.0)
BETA = 1e-6


def valid_time(pred, truth, sigma):
    err = np.sqrt(np.mean(((pred - truth) / sigma) ** 2, axis=1))
    bad = np.nonzero(err > THRESHOLD)[0]
    k = bad[0] if bad.size else len(err)
    return k * DT, k * DT * LYAPUNOV_MAX


def main():
    raw = generate(TRAIN + HORIZON + TEST_WARM + 10, dt=DT)
    mu = raw[:TRAIN].mean(axis=0)
    sd = raw[:TRAIN].std(axis=0)
    Z = (raw - mu) / sd
    sigma_norm = np.ones(3)  # data already standardized by training stats

    U_train, Y_train = Z[:TRAIN - 1], Z[1:TRAIN]
    test = Z[TRAIN:]
    warm, truth = test[:TEST_WARM], test[TEST_WARM:TEST_WARM + HORIZON]

    print(f"dt={DT}  train={TRAIN} steps  washout={WASHOUT}  horizon={HORIZON} steps  "
          f"threshold={THRESHOLD}  beta={BETA}  {ESN_KW}")
    print(f"{'seed':>4} {'valid_time':>11} {'in_LT':>7}")
    results = []
    for seed in SEEDS:
        esn = ESN(n_in=3, seed=seed, **ESN_KW).fit(U_train, Y_train, WASHOUT, BETA)
        r = esn.run(warm)[-1]
        pred = esn.predict_closed(r, HORIZON)
        t_valid, lt = valid_time(pred, truth, sigma_norm)
        results.append(lt)
        print(f"{seed:>4} {t_valid:>10.2f}s {lt:>7.2f}")
    print(f"mean valid time: {np.mean(results):.2f} Lyapunov times "
          f"(min {np.min(results):.2f}, max {np.max(results):.2f}) over {len(results)} seeds")


if __name__ == "__main__":
    main()
