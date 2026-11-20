"""Significance tests for the diagnostic gaps Delta_C and Delta_H.

Both tests resample whole series, so within-series and cross-horizon
dependence of the hits is preserved.  p = (1 + #{null >= obs}) / (B + 1).
The null-reference adjustment is ``statistic - null_mean``, not an unbiased
population estimate.  Inference needs independent series clusters (Delta_H)
or exchangeable trajectories within length groups (Delta_C).

Delta_C  H0: hits carry no association with the predicted values.  Whole
         series hit trajectories are permuted among series with the same
         number of windows, keeping predicted quantiles fixed (PCE unchanged).
Delta_H  H0: coverage is homogeneous across horizons.  Series-cluster
         bootstrap with each coverage curve recentred to the pooled coverage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .metrics import _as_3d, _build_specs, _conditional_fits, _coverage, _labels, _pce, _qce, hits_and_mask

__all__ = ["TestResult", "delta_c_test", "delta_h_test"]


@dataclass(frozen=True)
class TestResult:
    statistic: float   # observed gap
    null_mean: float
    corrected: float   # statistic - null_mean
    p_value: float
    null: np.ndarray   # (B,) null draws
    reason: str | None = None  # unavailable inference; descriptive statistic is retained

    @property
    def available(self) -> bool:
        return self.reason is None


def _unavailable(obs: float, reason: str) -> TestResult:
    warnings.warn(reason, stacklevel=3)
    return TestResult(float(obs), np.nan, np.nan, np.nan, np.empty(0), reason)


def _result(obs: float, null: np.ndarray) -> TestResult:
    if not np.isfinite(null).all():
        return _unavailable(obs, "gap test unavailable: resampling produced undefined statistics")
    null_mean = float(null.mean())
    p = float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))
    return TestResult(float(obs), null_mean, float(obs - null_mean), p, null)


def _check_resampling(series_ids, n: int, B: int) -> np.ndarray:
    series_ids = np.asarray(series_ids)
    if series_ids.shape != (n,):
        raise ValueError("series_ids must have shape (N,)")
    if not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError("B must be a positive integer")
    return series_ids


def series_groups(series_ids: np.ndarray, window_indices: np.ndarray | None = None) -> list[np.ndarray]:
    """Series trajectories (row positions in time order), grouped by length: list of (S_g, W) arrays."""
    series_ids = np.asarray(series_ids)
    if series_ids.ndim != 1 or not series_ids.size:
        raise ValueError("series_ids must be a nonempty one-dimensional array")
    window_indices = np.arange(len(series_ids)) if window_indices is None else np.asarray(window_indices)
    if window_indices.shape != series_ids.shape:
        raise ValueError("window_indices must have shape (N,)")
    if len(set(zip(series_ids.tolist(), window_indices.tolist()))) != len(series_ids):
        raise ValueError("(series_id, window_index) keys must be unique")
    by_len: dict[int, list[np.ndarray]] = {}
    for sid in np.unique(series_ids):
        idx = np.flatnonzero(series_ids == sid)
        idx = idx[np.argsort(window_indices[idx], kind="mergesort")]
        by_len.setdefault(len(idx), []).append(idx)
    return [np.stack(v) for v in by_len.values()]


def permute_series(groups: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Row source indices after permuting trajectories within each length group."""
    src = np.arange(n)
    for g in groups:
        if len(g) > 1:
            src[g.ravel()] = g[rng.permutation(len(g))].ravel()
    return src


def delta_c_test(y, qf, levels, series_ids, window_indices=None, B: int = 999, seed=0) -> TestResult:
    """Series-permutation test of Delta_C = QCE_avg - PCE_avg."""
    y, qf = _as_3d(y, qf)
    series_ids = _check_resampling(series_ids, len(y), B)
    labels = _labels(levels, *qf.shape[1:])
    if window_indices is not None and np.shape(window_indices) != (len(y),):
        raise ValueError("window_indices must have shape (N,)")
    fvalid = np.all(np.isfinite(qf), axis=2)
    rows = fvalid.any(axis=1)  # windows without any forecast carry no hits
    if not rows.any():
        raise ValueError("no valid target-forecast pairs")
    if not rows.all():
        y, qf, fvalid, series_ids = y[rows], qf[rows], fvalid[rows], series_ids[rows]
        window_indices = None if window_indices is None else np.asarray(window_indices)[rows]
    rng = np.random.default_rng(seed)
    z, valid = hits_and_mask(y, qf)
    specs = _build_specs(qf)
    pce_avg = _pce(_coverage(z, valid), labels)["pce_avg"]
    obs = _qce(_conditional_fits(z, valid, specs), labels)["qce_avg"] - pce_avg
    groups = series_groups(series_ids, window_indices)
    if not any(len(g) > 1 for g in groups):
        return _unavailable(obs, "Delta_C test unavailable: no exchangeable series group has at least two series")
    if np.unique(series_ids[valid.any(axis=1)]).size < 2:
        return _unavailable(obs, "Delta_C test unavailable: fewer than two series have valid target-forecast pairs")
    null = np.empty(B)  # a permuted hit only counts where the receiving window has a forecast
    for b in range(B):
        src = permute_series(groups, len(y), rng)
        v = valid[src] & fvalid
        zv = z[src] * v[:, :, None]
        null[b] = _qce(_conditional_fits(zv, v, specs), labels)["qce_avg"] - _pce(_coverage(zv, v), labels)["pce_avg"]
    return _result(obs, null)


def delta_h_test(y, qf, levels, series_ids, B: int = 999, seed=0) -> TestResult:
    """Series-cluster bootstrap test of Delta_H = PCE_avg - PCE_pool."""
    y, qf = _as_3d(y, qf)
    series_ids = _check_resampling(series_ids, len(y), B)
    labels = _labels(levels, *qf.shape[1:])
    rng = np.random.default_rng(seed)
    z, valid = hits_and_mask(y, qf)

    def gap(p):
        m = _pce(p, labels)
        return m["pce_avg"] - m["pce_pool"]

    p_hat = _coverage(z, valid)
    obs = gap(p_hat)
    if np.unique(series_ids[valid.any(axis=1)]).size < 2:
        return _unavailable(obs, "Delta_H test unavailable: fewer than two series have valid target-forecast pairs")
    keep = valid.any(axis=0)
    p_pool = p_hat[keep].mean(axis=0, keepdims=True)
    _, inverse = np.unique(series_ids, return_inverse=True)
    S = inverse.max() + 1
    zs = np.zeros((S,) + z.shape[1:])
    vs = np.zeros((S, z.shape[1]))
    np.add.at(zs, inverse, z)
    np.add.at(vs, inverse, valid)
    null = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_star = np.tensordot(w, zs, 1) / np.tensordot(w, vs, 1)[:, None]
        null[b] = gap(p_star - p_hat + p_pool) if np.isfinite(p_star[keep]).all() else np.nan
    return _result(obs, null)
