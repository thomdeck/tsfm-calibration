"""Quickstart on synthetic data (CPU, no model downloads, a few seconds).

The data are built so that the seasonal-naive baseline has the three kinds of
miscalibration the package separates, each from one mechanism:

- more volatile final weeks, so every interval is too narrow (pooled error, fixable),
- a daily cycle that grows over time, so peaks are under- and troughs over-forecast
  (horizon cancellation, Delta_H),
- a mean-reverting daily level that the baseline copies from the day before, so high
  forecasts tend to be too high and low ones too low (value cancellation, Delta_C).

    python examples/quickstart.py [--model chronos-bolt-small]
"""

import argparse

import numpy as np

import tsfm_calibration as tc


def synthetic_series(n_series: int = 100, days: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    """Hourly series: level + growing daily cycle + AR(1) daily level + noise, more volatile at the end."""
    rng = np.random.default_rng(seed)
    cycle = np.sin(2 * np.pi * (np.arange(24) - 6) / 24)
    hour_sd = 1 + 0.3 * (cycle > 0.5)                         # peak hours slightly noisier
    vol = np.where(np.arange(days) >= days - 12, 2.0, 1.0)    # last 12 days twice as volatile
    out = {}
    for i in range(n_series):
        level, amp = rng.uniform(95, 105), rng.uniform(5, 15)
        daily = np.zeros(days)
        for d in range(1, days):
            daily[d] = 0.8 * daily[d - 1] + rng.normal()
        x = (level + amp * (1 + 0.08 * np.arange(days))[:, None] * cycle + daily[:, None]
             + vol[:, None] * hour_sd * rng.normal(size=(days, 24)))
        out[f"s{i}"] = x.ravel()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="seasonal-naive")
    args = ap.parse_args()

    windows = tc.rolling_windows(synthetic_series(), prediction_length=24, num_windows=10)
    kwargs = {"season_length": 24} if args.model == "seasonal-naive" else {}
    f = tc.forecast(args.model, windows, **kwargs)

    ev = tc.evaluate(f, recalibrator="isotonic", tests=True, B=199)
    print(ev.summary().drop(columns=["n_windows"]).round(4).T)


if __name__ == "__main__":
    main()
