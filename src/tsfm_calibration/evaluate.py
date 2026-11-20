"""End-to-end pipeline: forecast windows -> audit -> recalibrate -> significance tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Window
from .forecasts import Forecasts, align
from .metrics import CalibrationReport, _build_specs, _conditional_fits, _labels, _levels, _report, hits_and_mask
from .models.base import Forecaster, get_model
from .recalibration import Recalibrator, get_recalibrator
from .significance import TestResult, delta_c_test, delta_h_test

__all__ = ["DEFAULT_LEVELS", "Evaluation", "forecast", "evaluate", "compare"]

DEFAULT_LEVELS = tuple(round(0.1 * k, 1) for k in range(1, 10))


def forecast(model: Forecaster | str, windows: list[Window], levels=DEFAULT_LEVELS, batch_size: int = 32,
             progress: bool = False, **model_kwargs) -> Forecasts:
    """Run ``model`` (instance or registered name) on all windows; quantiles are sorted per window."""
    levels = _levels(levels).tolist()
    if not windows:
        raise ValueError("at least one forecast window is required")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if any(np.ndim(w.context) != 1 or np.ndim(w.target) != 1
           or not len(w.context) or not len(w.target) for w in windows):
        raise ValueError("contexts and targets must be nonempty one-dimensional arrays")
    H = len(windows[0].target)
    if any(len(w.target) != H for w in windows):
        raise ValueError("all windows must have the same prediction length")
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)
    elif model_kwargs:
        raise TypeError(f"model keyword arguments {sorted(model_kwargs)} are only used with a registered model name; "
                        "configure the Forecaster instance directly instead")
    starts = range(0, len(windows), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            starts = tqdm(starts, desc=model.name)
        except ImportError:
            pass
    qf = []
    for s in starts:
        batch = windows[s : s + batch_size]
        out = np.asarray(model.predict_quantiles([w.context for w in batch], H, levels), dtype=np.float64)
        if out.shape != (len(batch), H, len(levels)):
            raise ValueError(f"{model.name} returned shape {out.shape}, expected {(len(batch), H, len(levels))}")
        qf.append(out)
    qf = np.concatenate(qf)
    incomplete = ~np.all(np.isfinite(qf), axis=-1)
    qf = np.sort(qf, axis=-1)
    qf[incomplete] = np.nan  # a partially missing quantile vector is treated as missing as a whole
    ids = [w.series_id for w in windows]
    if not all(isinstance(i, (int, np.integer)) for i in ids):
        ids = [str(i) for i in ids]
    return Forecasts(
        y=np.stack([w.target for w in windows]), qf=qf, levels=levels,
        series_ids=np.asarray(ids), window_indices=np.array([w.window_index for w in windows]),
        metadata={"model": model.name},
    )


@dataclass
class Evaluation:
    """Audit on all windows; before/after recalibration on the test split; optional gap tests."""

    audit: CalibrationReport
    before: CalibrationReport | None = None
    after: CalibrationReport | None = None
    labels: np.ndarray | None = None  # r(tau) learned on the calibration split
    test: Forecasts | None = None     # the test split of before / after
    delta_c: TestResult | None = None
    delta_h: TestResult | None = None

    def summary(self):
        """DataFrame with one row per report (audit / before / after) and the gap p-values."""
        import pandas as pd

        rows = {k: getattr(self, k).to_dict() for k in ("audit", "before", "after") if getattr(self, k)}
        df = pd.DataFrame(rows).T
        for name in ("delta_c", "delta_h"):
            t = getattr(self, name)
            if t is not None:
                df.loc["audit", f"{name}_corrected"] = t.corrected
                df.loc["audit", f"{name}_p"] = t.p_value
                if not t.available:
                    df.loc["audit", f"{name}_reason"] = t.reason
        return df

    def to_dict(self) -> dict:
        out = {k: getattr(self, k).to_dict(arrays=True) for k in ("audit", "before", "after") if getattr(self, k)}
        if self.labels is not None:
            out["labels"] = np.asarray(self.labels).tolist()
        for name in ("delta_c", "delta_h"):
            t = getattr(self, name)
            if t is not None:
                out[name] = {"statistic": t.statistic, "null_mean": t.null_mean if t.available else None,
                             "corrected": t.corrected if t.available else None,
                             "p_value": t.p_value if t.available else None,
                             "available": t.available, "reason": t.reason}
        return out


def evaluate(forecasts: Forecasts, recalibrator: Recalibrator | str | None = "isotonic",
             split: str = "chronological", cal_fraction: float = 0.2, seed: int = 42,
             tests: bool = False, B: int = 999) -> Evaluation:
    """Calibration audit, and optionally recalibration and the Delta_C / Delta_H tests.

    ``recalibrator`` is a name in ``recalibration.RECALIBRATORS``, an instance, or None
    (default ``isotonic``, as in the paper's figures).  It is fitted on the calibration
    split (``chronological``, the paper's primary protocol, or ``random``) and both
    reports are on the test split.
    """
    f = forecasts
    ev = Evaluation(audit=f.report())
    if recalibrator is not None:
        rec = get_recalibrator(recalibrator) if isinstance(recalibrator, str) else recalibrator
        if split not in ("chronological", "random"):
            raise ValueError("split must be 'chronological' or 'random'")
        splitter = {"chronological": f.chronological_split, "random": f.random_split}[split]
        cal, test = splitter(cal_fraction, seed)
        if len(cal) == 0 or len(test) == 0:
            raise ValueError("empty calibration or test split")
        rec.fit(cal.y, cal.qf, cal.levels)
        ev.labels, ev.test = rec.relabel(f.levels), test
        z, valid = hits_and_mask(test.y, test.qf)
        fits = _conditional_fits(z, valid, _build_specs(test.qf))  # shared by before/after
        H, Q = test.qf.shape[1:]
        ev.before = _report(test.y, test.qf, _labels(f.levels, H, Q), z, valid, fits)
        ev.after = _report(test.y, test.qf, _labels(ev.labels, H, Q), z, valid, fits)
    if tests:
        ev.delta_c = delta_c_test(f.y, f.qf, f.levels, f.series_ids, f.window_indices, B=B, seed=seed)
        ev.delta_h = delta_h_test(f.y, f.qf, f.levels, f.series_ids, B=B, seed=seed)
    return ev


def compare(forecasts: dict[str, Forecasts], **evaluate_kwargs):
    """Evaluate several models on their common windows; one row per model."""
    import pandas as pd

    rows = []
    for name, f in align(forecasts).items():
        ev = evaluate(f, **evaluate_kwargs)
        row = {"model": name, **ev.audit.to_dict()}
        if ev.after is not None:
            row.update({f"{k}_recal": v for k, v in ev.after.to_dict().items() if k != "n_windows"})
        for t in ("delta_c", "delta_h"):
            if getattr(ev, t) is not None:
                row[f"{t}_p"] = getattr(ev, t).p_value
                if not getattr(ev, t).available:
                    row[f"{t}_reason"] = getattr(ev, t).reason
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
