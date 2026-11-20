"""Container for quantile forecasts over non-overlapping rolling windows."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import CalibrationReport, _as_3d, _levels, calibration_report

__all__ = ["Forecasts", "align"]


@dataclass
class Forecasts:
    """Quantile forecasts for N windows.

    y : (N, H) targets (NaN = missing)
    qf : (N, H, Q) predicted quantiles at ``levels`` (Q,)
    series_ids, window_indices : (N,) series of each window and its time order
        within the series (needed for chronological splits and significance tests)
    """

    y: np.ndarray
    qf: np.ndarray
    levels: np.ndarray
    series_ids: np.ndarray | None = None
    window_indices: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64)
        self.qf = np.asarray(self.qf, dtype=np.float64)
        self.levels = _levels(self.levels)
        if (self.y.ndim != 2 or self.qf.ndim != 3 or self.qf.shape[:2] != self.y.shape
                or self.qf.shape[2] != len(self.levels)):
            raise ValueError(f"inconsistent shapes y={self.y.shape} qf={self.qf.shape} levels={self.levels.shape}")
        self.y, self.qf = _as_3d(self.y, self.qf)
        n = self.y.shape[0]
        self.series_ids = np.arange(n) if self.series_ids is None else np.asarray(self.series_ids)
        self.window_indices = np.zeros(n, dtype=np.int64) if self.window_indices is None \
            else np.asarray(self.window_indices)
        if self.series_ids.shape != (n,) or self.window_indices.shape != (n,):
            raise ValueError("series_ids and window_indices must have shape (N,)")
        if self.series_ids.dtype.kind == "O" and all(isinstance(s, str) for s in self.series_ids):
            self.series_ids = self.series_ids.astype(str)
        if self.series_ids.dtype.kind not in "iuUS":
            raise ValueError("series_ids must be integers or strings")
        if self.window_indices.dtype.kind not in "iu" or np.any(self.window_indices < 0):
            raise ValueError("window_indices must be non-negative integers")
        if len(set(zip(self.series_ids.tolist(), self.window_indices.tolist()))) != n:
            raise ValueError("(series_id, window_index) keys must be unique")

    def __len__(self) -> int:
        return self.y.shape[0]

    @property
    def horizon(self) -> int:
        return self.y.shape[1]

    def report(self, levels=None) -> CalibrationReport:
        """Calibration report against ``levels`` (default: the nominal levels)."""
        return calibration_report(self.y, self.qf, self.levels if levels is None else levels)

    def subset(self, idx) -> Forecasts:
        idx = np.asarray(idx)
        return Forecasts(self.y[idx], self.qf[idx], self.levels, self.series_ids[idx],
                         self.window_indices[idx], dict(self.metadata))

    # ── Splits ────────────────────────────────────────────────────────────

    def _check_split(self, cal_fraction: float):
        if not 0 < cal_fraction < 1:
            raise ValueError("cal_fraction must be strictly between 0 and 1")
        if len(self) < 2:
            raise ValueError("splitting requires at least two windows")

    def chronological_split(self, cal_fraction: float = 0.2, seed: int = 42) -> tuple[Forecasts, Forecasts]:
        """(calibration, test) with no window in both.

        Multi-window series: the first ceil(W * cal_fraction) windows calibrate,
        capped at W - 1 to retain a test window.
        Single-window series: a seeded random cal_fraction of these series calibrate.
        External forecasts must have non-overlapping targets and calibration
        outcomes available before the first test forecast origin.
        """
        self._check_split(cal_fraction)
        cal, test, w1 = [], [], []
        for sid in np.unique(self.series_ids):
            pos = np.flatnonzero(self.series_ids == sid)
            pos = pos[np.argsort(self.window_indices[pos], kind="mergesort")]
            if len(pos) == 1:
                w1.append(int(pos[0]))
            else:
                n_cal = min(len(pos) - 1, max(1, math.ceil(len(pos) * cal_fraction)))
                cal.extend(pos[:n_cal])
                test.extend(pos[n_cal:])
        if w1:
            w1 = np.asarray(w1)[np.random.default_rng(seed).permutation(len(w1))]
            n_cal = math.ceil(len(w1) * cal_fraction)
            if len(w1) > 1:
                n_cal = min(n_cal, len(w1) - 1)
            cal.extend(w1[:n_cal])
            test.extend(w1[n_cal:])
        return self.subset(np.array(cal, dtype=np.int64)), self.subset(np.array(test, dtype=np.int64))

    def random_split(self, cal_fraction: float = 0.2, seed: int = 42) -> tuple[Forecasts, Forecasts]:
        """(calibration, test) by a seeded random permutation of windows."""
        self._check_split(cal_fraction)
        idx = np.random.default_rng(seed).permutation(len(self))
        n_test = min(len(self) - 1, max(1, int(len(self) * (1 - cal_fraction))))
        return self.subset(idx[n_test:]), self.subset(idx[:n_test])

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, y=self.y, qf=self.qf, levels=self.levels, series_ids=self.series_ids,
                            window_indices=self.window_indices, metadata=np.array(json.dumps(self.metadata)))
        return path

    @classmethod
    def load(cls, path) -> Forecasts:
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as z:
            return cls(z["y"], z["qf"], z["levels"], z["series_ids"], z["window_indices"],
                       json.loads(str(z["metadata"])))


def align(forecasts: dict[str, Forecasts]) -> dict[str, Forecasts]:
    """Restrict all models to their common (series, window) keys, sorted, for paired comparison."""
    if not forecasts:
        raise ValueError("at least one forecast container is required")
    keys = None
    for f in forecasts.values():
        k = set(zip(f.series_ids.tolist(), f.window_indices.tolist()))
        keys = k if keys is None else keys & k
    if not keys:
        raise ValueError("forecasts have no common (series, window) keys")
    out = {}
    reference = next(iter(forecasts.values()))
    for name, f in forecasts.items():
        if f.horizon != reference.horizon or not np.array_equal(f.levels, reference.levels):
            raise ValueError(f"{name}: paired forecasts must have identical horizons and quantile grids")
        rows = np.array([i for i, k in enumerate(zip(f.series_ids.tolist(), f.window_indices.tolist()))
                         if k in keys], dtype=np.int64)
        rows = rows[np.lexsort((f.window_indices[rows], f.series_ids[rows]))]
        out[name] = f.subset(rows)
        if len(out) > 1 and not np.array_equal(out[name].y, next(iter(out.values())).y, equal_nan=True):
            raise ValueError(f"{name}: targets differ on aligned windows; use shared identifiers and targets")
        if len(rows) < len(f):
            warnings.warn(f"{name}: dropped {len(f) - len(rows)} windows outside the common keys", stacklevel=2)
    return out
