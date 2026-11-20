"""Seasonal-naive baseline with empirical residual quantiles (CPU, no extra dependencies)."""

from __future__ import annotations

import numpy as np

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """q_tau at step h = x_{T-m+1+(h-1) mod m} + sqrt(ceil(h/m)) * Quantile_tau(x_t - x_{t-m}).

    Falls back to m = 1 for contexts shorter than two seasons.
    """

    name = "seasonal-naive"

    def __init__(self, season_length: int = 1):
        self.season_length = season_length

    def predict_quantiles(self, contexts, prediction_length, levels):
        levels = np.asarray(levels, dtype=np.float64)
        out = np.full((len(contexts), prediction_length, len(levels)), np.nan)  # NaN = missing forecast
        steps = np.arange(prediction_length)
        for i, c in enumerate(contexts):
            x = np.asarray(c, dtype=np.float64)
            if not np.isfinite(x).any():
                continue
            x = np.where(np.isfinite(x), x, np.nanmean(x))
            m = self.season_length if len(x) >= 2 * self.season_length else 1
            e = x[m:] - x[:-m]
            q = np.quantile(e, levels) if len(e) >= 2 else np.zeros(len(levels))
            point = x[-m:][steps % m]
            scale = np.sqrt(np.ceil((steps + 1) / m))
            out[i] = point[:, None] + scale[:, None] * q[None, :]
        return out
