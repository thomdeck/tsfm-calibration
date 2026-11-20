"""Calibration metrics for multi-horizon quantile forecasts.

Shapes: targets ``y (N, H)``, predicted quantiles ``qf (N, H, Q)``, labels
``levels (Q,)`` or ``(H, Q)``.  Single-horizon input ``y (N,)``, ``qf (N, Q)``
is treated as H = 1.  Missing targets (NaN) are excluded, never counted as misses.

With hits z = 1{y <= q_hat} and per-horizon coverage p_h(tau) = mean_i z:

    PCE_pool = mean_q | mean_h p_h(tau_q) - tau_q |
    PCE_h    = mean_q | p_h(tau_q) - tau_q |            PCE_avg / PCE_max = mean / max_h
    QCE_h    = mean_q mean_i | c_hq(q_hat_i) - tau_q |   QCE_avg / QCE_max = mean / max_h
    Delta_H  = PCE_avg - PCE_pool >= 0                   (horizon cancellation)
    Delta_C  = QCE_avg - PCE_avg  >= 0                   (value cancellation)

c_hq is the isotonic (PAV) regression of the hits on the predicted quantile
value.  The estimators satisfy PCE_pool <= PCE_avg <= QCE_avg <= QCE_max on
every sample.  The population isotonic target lower-bounds QCE, but can miss
decreasing or non-monotone conditional coverage; finite-sample values need
not lower-bound population QCE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.isotonic import isotonic_regression

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0

__all__ = [
    "CalibrationReport",
    "calibration_report",
    "pce",
    "qce",
    "coverage",
    "conditional_coverage",
    "crps_quantile",
    "interval_coverage",
]


# ──────────────────────────────────────────────────────────────────────────────
# Hits and coverage
# ──────────────────────────────────────────────────────────────────────────────


def _as_3d(y, qf) -> tuple[np.ndarray, np.ndarray]:
    y, qf = np.asarray(y, dtype=np.float64), np.asarray(qf, dtype=np.float64)
    if qf.ndim == 2 and y.ndim == 1:
        y, qf = y.reshape(-1, 1), qf[:, None, :]
    if y.ndim != 2 or qf.ndim != 3 or y.shape != qf.shape[:2]:
        raise ValueError(f"y {y.shape} and qf {qf.shape} do not match (expected (N,H) and (N,H,Q))")
    if 0 in qf.shape:
        raise ValueError("forecasts must contain at least one window, horizon and level")
    if np.isinf(y).any() or np.isinf(qf).any():
        raise ValueError("targets and quantiles must be finite or NaN")
    if np.any(np.diff(qf, axis=-1) < 0):
        raise ValueError("quantile values must be non-decreasing along the last axis")
    return y, qf


def _levels(levels) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64)
    if (levels.ndim != 1 or not levels.size or not np.isfinite(levels).all()
            or np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0)):
        raise ValueError("input levels must be a finite, strictly increasing grid in (0, 1)")
    return levels


def _labels(levels, H: int, Q: int) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64)
    if (levels.shape not in ((Q,), (H, Q)) or not np.isfinite(levels).all()
            or np.any((levels < 0) | (levels > 1)) or np.any(np.diff(levels, axis=-1) < 0)):
        raise ValueError("labels must have shape (Q,) or (H,Q), be finite and non-decreasing in [0, 1]")
    return np.broadcast_to(levels, (H, Q))


def hits_and_mask(y: np.ndarray, qf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hit indicators (N, H, Q) as float (0 where invalid) and validity mask (N, H)."""
    valid = np.isfinite(y) & np.all(np.isfinite(qf), axis=2)
    with np.errstate(invalid="ignore"):
        z = (y[:, :, None] <= qf) & valid[:, :, None]
    return z.astype(np.float64), valid


def _coverage(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n_valid = valid.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return z.sum(axis=0) / n_valid[:, None]


def coverage(y, qf) -> np.ndarray:
    """Per-horizon empirical coverage p_h(tau) of shape (H, Q); NaN for empty horizons."""
    return _coverage(*hits_and_mask(*_as_3d(y, qf)))


# ──────────────────────────────────────────────────────────────────────────────
# PCE family
# ──────────────────────────────────────────────────────────────────────────────


def _pce(p: np.ndarray, labels: np.ndarray) -> dict:
    dev = p - labels
    pce_h = np.abs(dev).mean(axis=1)  # NaN for horizons without valid targets
    keep = np.isfinite(pce_h)
    if not keep.any():
        raise ValueError("no valid target-forecast pairs")
    return {
        "pce_pool": float(np.abs(dev[keep].mean(axis=0)).mean()),
        "pce_avg": float(pce_h[keep].mean()),
        "pce_max": float(pce_h[keep].max()),
        "pce_per_horizon": pce_h,
    }


def pce(y, qf, levels) -> dict:
    """Probabilistic calibration errors ``pce_pool``, ``pce_avg``, ``pce_max``, ``pce_per_horizon``."""
    y, qf = _as_3d(y, qf)
    return _pce(coverage(y, qf), _labels(levels, *qf.shape[1:]))


# ──────────────────────────────────────────────────────────────────────────────
# QCE family (PAV conditional coverage)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _TieSpec:
    order: np.ndarray   # argsort of predicted values
    starts: np.ndarray  # first position of each tie group in sorted order


def _build_specs(qf: np.ndarray) -> list[list[_TieSpec]]:
    """Sort predicted quantiles once per (h, q) and locate ties."""
    _, H, Q = qf.shape
    specs = []
    for h in range(H):
        row = []
        for q in range(Q):
            order = np.argsort(qf[:, h, q], kind="mergesort")
            x = qf[order, h, q]
            row.append(_TieSpec(order, np.r_[0, 1 + np.flatnonzero(x[1:] != x[:-1])]))
        specs.append(row)
    return specs


def _group_fit(z_hq: np.ndarray, w_hq: np.ndarray, spec: _TieSpec) -> tuple[np.ndarray, np.ndarray]:
    """Weighted PAV of hits on predicted value; fitted value and weight per tie group (NaN if empty)."""
    w = np.add.reduceat(w_hq[spec.order], spec.starts)
    zs = np.add.reduceat(z_hq[spec.order], spec.starts)
    fitted = np.full(len(w), np.nan)
    keep = w > 0
    if keep.any():
        fitted[keep] = isotonic_regression(zs[keep] / w[keep], sample_weight=w[keep], y_min=0.0, y_max=1.0)
    return fitted, w


def _conditional_fits(z: np.ndarray, valid: np.ndarray, specs) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    w = valid.astype(np.float64)
    return [[_group_fit(z[:, h, q], w[:, h], specs[h][q]) for q in range(z.shape[2])]
            for h in range(z.shape[1])]


def _qce(fits, labels: np.ndarray) -> dict:
    H, Q = labels.shape
    table = np.full((H, Q), np.nan)
    for h in range(H):
        for q in range(Q):
            fitted, w = fits[h][q]
            if w.sum() > 0:
                ok = w > 0
                table[h, q] = np.sum(w[ok] * np.abs(fitted[ok] - labels[h, q])) / w.sum()
    qce_h = table.mean(axis=1)
    keep = np.isfinite(qce_h)
    if not keep.any():
        raise ValueError("no valid target-forecast pairs")
    return {
        "qce_avg": float(qce_h[keep].mean()),
        "qce_max": float(qce_h[keep].max()),
        "qce_per_horizon": qce_h,
        "qce_per_level": table[keep].mean(axis=0),
    }


def qce(y, qf, levels) -> dict:
    """Quantile-conditioned calibration errors ``qce_avg``, ``qce_max``, per horizon and per level."""
    y, qf = _as_3d(y, qf)
    z, valid = hits_and_mask(y, qf)
    return _qce(_conditional_fits(z, valid, _build_specs(qf)), _labels(levels, *qf.shape[1:]))


def conditional_coverage(y, qf, h: int, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PAV estimate of P(Y_h <= q_hat | q_hat) at horizon h, level index q.

    Returns sorted distinct predicted values x, fitted coverage c(x) and weights
    (number of valid windows per value).
    """
    y, qf = _as_3d(y, qf)
    z, valid = hits_and_mask(y, qf)
    spec = _build_specs(qf[:, h : h + 1, q : q + 1])[0][0]
    fitted, w = _group_fit(z[:, h, q], valid[:, h].astype(np.float64), spec)
    x = qf[spec.order[spec.starts], h, q]
    ok = w > 0
    return x[ok], fitted[ok], w[ok]


# ──────────────────────────────────────────────────────────────────────────────
# Scoring rules
# ──────────────────────────────────────────────────────────────────────────────


def _mean_pinball(y: np.ndarray, qf: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean pinball loss per (h, q) over valid windows, and valid counts per horizon."""
    valid = np.isfinite(y) & np.all(np.isfinite(qf), axis=2)
    r = np.where(valid[:, :, None], y[:, :, None] - qf, 0.0)
    loss = np.where(r >= 0, labels * r, (labels - 1) * r)
    n = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return loss.sum(axis=0) / n[:, None], n


def crps_quantile(y, qf, levels) -> float:
    """CRPS approximated by 2 * integral over [0, 1] of the pinball loss (trapezoid, zero at 0 and 1)."""
    y, qf = _as_3d(y, qf)
    labels = _labels(levels, *qf.shape[1:])
    mp, n = _mean_pinball(y, qf, labels)
    total = 0.0
    for h in np.flatnonzero(n):
        tau, idx = np.unique(labels[h], return_index=True)
        total += 2 * _trapezoid(np.r_[0.0, mp[h, idx], 0.0], np.r_[0.0, tau, 1.0]) * n[h]
    return float(total / n.sum())


def interval_coverage(y, qf, levels, central: float = 0.8) -> float:
    """Empirical coverage of the central ``central`` interval [q_{(1-c)/2}, q_{(1+c)/2}]."""
    levels = np.asarray(levels, dtype=np.float64)
    lo = np.flatnonzero(np.isclose(levels, (1 - central) / 2))
    hi = np.flatnonzero(np.isclose(levels, (1 + central) / 2))
    if not (len(lo) and len(hi)):
        raise ValueError(f"levels do not contain the bounds of the {central:.0%} central interval")
    y, qf = _as_3d(y, qf)
    lower, upper = qf[..., lo[0]], qf[..., hi[0]]
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean((y[ok] >= lower[ok]) & (y[ok] <= upper[ok])))


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration hierarchy, diagnostic gaps and CRPS for one set of forecasts."""

    pce_pool: float
    pce_avg: float
    pce_max: float
    qce_avg: float
    qce_max: float
    delta_h: float
    delta_c: float
    crps: float
    coverage: np.ndarray         # (Q,) coverage averaged over horizons
    pce_per_horizon: np.ndarray  # (H,)
    qce_per_horizon: np.ndarray  # (H,)
    n_windows: int

    def to_dict(self, arrays: bool = False) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()
                if arrays or not isinstance(v, np.ndarray)}

    def __str__(self) -> str:
        return (f"PCE_pool={self.pce_pool:.4f} <= PCE_avg={self.pce_avg:.4f} <= "
                f"QCE_avg={self.qce_avg:.4f} <= QCE_max={self.qce_max:.4f} | "
                f"Delta_H={self.delta_h:.4f} Delta_C={self.delta_c:.4f} | CRPS={self.crps:.4g} "
                f"(N={self.n_windows})")


def _report(y, qf, labels, z, valid, fits) -> CalibrationReport:
    p = _coverage(z, valid)
    m = _pce(p, labels)
    c = _qce(fits, labels)
    keep = np.isfinite(p).all(axis=1)
    return CalibrationReport(
        pce_pool=m["pce_pool"], pce_avg=m["pce_avg"], pce_max=m["pce_max"],
        qce_avg=c["qce_avg"], qce_max=c["qce_max"],
        delta_h=max(0.0, m["pce_avg"] - m["pce_pool"]), delta_c=max(0.0, c["qce_avg"] - m["pce_avg"]),
        crps=crps_quantile(y, qf, labels),
        coverage=p[keep].mean(axis=0),
        pce_per_horizon=m["pce_per_horizon"], qce_per_horizon=c["qce_per_horizon"],
        n_windows=int(y.shape[0]),
    )


def calibration_report(y, qf, levels) -> CalibrationReport:
    """Full PCE/QCE hierarchy, gaps Delta_H and Delta_C, and CRPS.

    ``levels`` (Q,) or (H, Q) are the labels the quantile values are scored
    against, e.g. the output of ``Recalibrator.relabel``.
    """
    y, qf = _as_3d(y, qf)
    z, valid = hits_and_mask(y, qf)
    labels = _labels(levels, *qf.shape[1:])
    return _report(y, qf, labels, z, valid, _conditional_fits(z, valid, _build_specs(qf)))
