"""Post-hoc recalibration of quantile forecasts by a global relabeling map.

Each recalibrator is fitted pooled over calibration windows and horizons and
learns a non-decreasing map tau -> r(tau): the quantile value predicted at
nominal level tau is relabelled as the r(tau) quantile.  ``relabel`` returns
these labels (the protocol in the paper); ``transform`` re-interpolates the
values to quantiles at the requested levels for downstream use.  Finite-grid
scores of these two representations need not coincide.

IR and HB coincide at fitted levels aligned with histogram edges, up to
numerical interpolation.  KDE smooths and PS parametrically approximates
the same curve.  Fitting weights valid window-horizon pairs equally, whereas
metrics weight horizons equally; these differ with unequal valid counts.

A positive original horizon gap prevents any global map from eliminating
horizon-wise error (PCE_avg >= original Delta_H / 2).  Neither gap is invariant
under relabeling; check both after recalibrating.

References: Kuleshov et al. (2018) isotonic; Platt (1999) logistic;
Zadrozny & Elkan (2002) histogram binning; Dheur & Ben Taieb (2023) KDE map.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .metrics import _as_3d, _levels, hits_and_mask

__all__ = [
    "Recalibrator",
    "IsotonicRecalibrator",
    "PlattRecalibrator",
    "HistogramBinningRecalibrator",
    "KDERecalibrator",
    "RECALIBRATORS",
    "get_recalibrator",
    "pit_from_quantile_grid",
]


def pooled_coverage(y, qf) -> np.ndarray:
    """Coverage per level (Q,) pooled over all valid (window, horizon) pairs."""
    z, valid = hits_and_mask(*_as_3d(y, qf))
    if not valid.any():
        raise ValueError("recalibration requires valid target-forecast pairs")
    return z.sum(axis=(0, 1)) / valid.sum()


def _fit_data(y, qf, levels):
    y, qf = _as_3d(y, qf)
    levels = _levels(levels)
    if len(levels) != qf.shape[-1]:
        raise ValueError("levels must match the last quantile dimension")
    return y, qf, levels


def pit_from_quantile_grid(y, qf, levels) -> np.ndarray:
    """PIT values from a quantile grid by linear interpolation of the CDF.

    Values at or below the lowest (above the highest) quantile get 1e-6 (1 - 1e-6).
    Ties y == q count as hits, so pit <= tau_q exactly when y <= q_{tau_q}.
    Invalid targets are dropped; returns a flat array.
    """
    y, qf, levels = _fit_data(y, qf, levels)
    y, qf = y.ravel(), qf.reshape(-1, qf.shape[-1])
    keep = np.isfinite(y) & np.all(np.isfinite(qf), axis=1)
    y, qf = y[keep], qf[keep]
    if not len(y):
        raise ValueError("recalibration requires valid target-forecast pairs")
    Q = qf.shape[1]
    c = (qf < y[:, None]).sum(axis=1)
    pit = np.empty(len(y))
    pit[c == 0] = 1e-6
    pit[c == Q] = 1.0 - 1e-6
    mid = (c > 0) & (c < Q)
    k = c[mid]
    lo, hi = qf[mid, k - 1], qf[mid, k]
    pit[mid] = levels[k] - (hi - y[mid]) / (hi - lo) * (levels[k] - levels[k - 1])  # exactly tau_k at a tie
    return pit


class Recalibrator(ABC):
    """Global monotone level map tau -> r(tau), fitted on calibration forecasts."""

    @abstractmethod
    def fit(self, y, qf, levels) -> Recalibrator:
        """Fit on calibration targets y (N, H) and quantiles qf (N, H, Q) at ``levels`` (Q,)."""

    @abstractmethod
    def relabel(self, levels) -> np.ndarray:
        """New labels r(tau) (Q,) for the unchanged quantile values."""

    def transform(self, qf, levels, target_levels=None) -> np.ndarray:
        """Recalibrated quantiles at ``target_levels`` (default: ``levels``), shape (..., Q').

        Linear interpolation of the values over the relabelled grid r(tau);
        targets outside [r(tau_1), r(tau_Q)] are clamped to the outermost value.
        For tied labels the first quantile value is retained.
        """
        levels = _levels(levels)
        target = levels if target_levels is None else _levels(target_levels)
        qf = np.asarray(qf, dtype=np.float64)
        if qf.ndim < 1 or qf.shape[-1] != len(levels):
            raise ValueError("levels must match the last quantile dimension")
        return _interpolate_relabeled(qf, self.relabel(levels), target)


class IsotonicRecalibrator(Recalibrator):
    """Isotonic regression of pooled empirical coverage on nominal level (Kuleshov et al. 2018)."""

    def fit(self, y, qf, levels):
        from sklearn.isotonic import IsotonicRegression

        y, qf, levels = _fit_data(y, qf, levels)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(levels, dtype=np.float64), pooled_coverage(y, qf))
        self._grid = np.linspace(0.0, 1.0, 1000)
        self._map = iso.predict(self._grid)
        return self

    def relabel(self, levels):
        return np.clip(np.interp(np.asarray(levels, dtype=np.float64), self._grid, self._map), 0.0, 1.0)


class PlattRecalibrator(Recalibrator):
    """r(tau) = sigmoid(a * logit(tau) + b), least squares on pooled coverage."""

    def fit(self, y, qf, levels):
        y, qf, levels = _fit_data(y, qf, levels)
        lt = logit(np.clip(np.asarray(levels, dtype=np.float64), 1e-6, 1 - 1e-6))
        emp = pooled_coverage(y, qf)
        # Closed-form start: least squares in logit space, then refine on the coverage scale with a >= 0
        le = logit(np.clip(emp, 1e-4, 1 - 1e-4))
        if lt.size >= 2 and np.ptp(lt) > 0:
            a0, b0 = np.polyfit(lt, le, 1)
        else:
            a0, b0 = 1.0, float(np.mean(le - lt))
        res = minimize(lambda ab: np.sum((emp - expit(ab[0] * lt + ab[1])) ** 2), x0=[max(a0, 1e-3), b0],
                       method="L-BFGS-B", bounds=[(0.0, None), (None, None)],
                       options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000})
        if res.status == 1:  # iteration limit; line-search stops at machine precision are benign here
            warnings.warn(f"Platt recalibration did not fully converge: {res.message}", stacklevel=2)
        self.a, self.b = res.x
        return self

    def relabel(self, levels):
        lt = logit(np.clip(np.asarray(levels, dtype=np.float64), 1e-6, 1 - 1e-6))
        return np.clip(expit(self.a * lt + self.b), 0.0, 1.0)


class HistogramBinningRecalibrator(Recalibrator):
    """Piecewise-linear CDF of the PIT histogram with ``n_bins`` equal-width bins."""

    def __init__(self, n_bins: int = 20):
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        self.n_bins = n_bins

    def fit(self, y, qf, levels):
        pit = pit_from_quantile_grid(y, qf, levels)
        self._edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        # right-continuous PIT ECDF at the edges, so a tie pit == edge counts below the edge
        self._cdf = np.searchsorted(np.sort(pit), self._edges + 1e-12, side="right") / len(pit)
        return self

    def relabel(self, levels):
        return np.interp(np.asarray(levels, dtype=np.float64), self._edges, self._cdf)


class KDERecalibrator(Recalibrator):
    """Sigmoid-smoothed PIT ECDF r(tau) = mean_i sigmoid(bandwidth * (tau - u_i))."""

    def __init__(self, bandwidth: float = 100.0):
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive")
        self.bandwidth = bandwidth

    def fit(self, y, qf, levels):
        self._pit = np.sort(pit_from_quantile_grid(y, qf, levels))
        return self

    def relabel(self, levels):
        levels = np.asarray(levels, dtype=np.float64)
        return expit(self.bandwidth * (levels.ravel()[:, None] - self._pit[None, :])).mean(axis=1).reshape(levels.shape)


RECALIBRATORS = {
    "isotonic": IsotonicRecalibrator,
    "platt": PlattRecalibrator,
    "histogram_binning": HistogramBinningRecalibrator,
    "kde": KDERecalibrator,
}


def _interpolate_relabeled(qf, labels, target) -> np.ndarray:
    """Quantile values qf (..., Q) labelled r(tau) = ``labels``, interpolated to ``target`` levels."""
    r, keep = np.unique(np.asarray(labels, dtype=np.float64), return_index=True)
    qk = np.asarray(qf, dtype=np.float64)[..., keep]
    if len(r) == 1:
        return np.repeat(qk, len(target), axis=-1)
    t = np.clip(target, r[0], r[-1])
    k = np.clip(np.searchsorted(r, t, side="right"), 1, len(r) - 1)
    w = (t - r[k - 1]) / (r[k] - r[k - 1])
    return qk[..., k - 1] * (1 - w) + qk[..., k] * w


def get_recalibrator(name: str, **kwargs) -> Recalibrator:
    """Instantiate a recalibrator by name: ``isotonic``, ``platt``, ``histogram_binning`` or ``kde``."""
    if name not in RECALIBRATORS:
        raise KeyError(f"unknown recalibrator {name!r}; choose from {sorted(RECALIBRATORS)}")
    return RECALIBRATORS[name](**kwargs)
