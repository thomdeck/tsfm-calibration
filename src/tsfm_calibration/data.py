"""Rolling-origin evaluation windows from arrays, DataFrames, CSV files or GIFT-Eval."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np

__all__ = ["Window", "rolling_windows", "from_dataframe", "from_csv", "gift_eval_windows"]


class Window(NamedTuple):
    series_id: object
    window_index: int  # 0 = earliest window of the series
    context: np.ndarray
    target: np.ndarray


def rolling_windows(
    series: Mapping[object, np.ndarray] | Iterable[np.ndarray],
    prediction_length: int,
    num_windows: int = 1,
    stride: int | None = None,
    max_context: int | None = None,
    min_context: int = 1,
) -> list[Window]:
    """Rolling-origin windows over the end of each series (GIFT-Eval convention).

    The last window ends at the last observation; earlier windows are shifted
    back by ``stride`` (at least ``prediction_length``, non-overlapping).
    Windows with fewer than ``min_context`` context points are skipped.
    """
    stride = prediction_length if stride is None else stride
    for name, value in (("prediction_length", prediction_length), ("num_windows", num_windows),
                        ("stride", stride), ("min_context", min_context), ("max_context", max_context)):
        if value is None and name == "max_context":
            continue
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if stride < prediction_length:
        raise ValueError("stride must be at least prediction_length to avoid overlapping calibration/test targets")
    if max_context is not None and max_context < min_context:
        raise ValueError("max_context must be at least min_context")
    items = series.items() if isinstance(series, Mapping) else enumerate(series)
    out = []
    for sid, x in items:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"series {sid!r} must be one-dimensional")
        for w in range(num_windows):
            end = len(x) - (num_windows - 1 - w) * stride
            start = end - prediction_length
            if start < min_context:
                continue
            ctx = x[:start] if max_context is None else x[max(0, start - max_context):start]
            out.append(Window(sid, w, ctx, x[start:end]))
    if not out:
        raise ValueError("no series has enough observations for the requested windows")
    return out


def from_dataframe(df, id_col: str = "item_id", time_col: str = "timestamp",
                   target_col: str = "target") -> dict[object, np.ndarray]:
    """Long-format DataFrame (one row per series and time step) -> {series_id: values sorted by time}."""
    df = df.sort_values([id_col, time_col], kind="mergesort")
    return {sid: g[target_col].to_numpy(dtype=np.float64) for sid, g in df.groupby(id_col, sort=False)}


def from_csv(path, id_col: str = "item_id", time_col: str = "timestamp",
             target_col: str = "target", **read_csv_kwargs) -> dict[object, np.ndarray]:
    """Long-format CSV -> {series_id: values sorted by time}."""
    import pandas as pd

    df = pd.read_csv(path, **read_csv_kwargs)
    if time_col in df:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except (ValueError, TypeError):
            pass  # keep the original ordering key (e.g. integer time index)
    return from_dataframe(df, id_col, time_col, target_col)


def gift_eval_windows(name: str, term: str = "short", storage_path=None) -> tuple[list[Window], dict]:
    """GIFT-Eval test windows of one configuration (requires the upstream source installation).

    Multivariate datasets are split channel-wise (series id = series * D + channel).
    Returns the windows and a dict with ``prediction_length``, ``freq`` and ``windows``.
    """
    if storage_path is not None:
        os.environ["GIFT_EVAL"] = str(storage_path)
    try:
        from gift_eval.data import Dataset, Term
    except ImportError as e:
        raise ImportError("GIFT-Eval loading needs the SalesforceAIResearch/gift-eval source installation; "
                          "see the README's GIFT-Eval setup instructions") from e

    ds = Dataset(name=name, term=Term(term), storage_env_var="GIFT_EVAL", to_univariate=False)
    W = ds.windows
    out = []
    for i, (inp, label) in enumerate(ds.test_data):
        sid, w = i // W, i % W
        ctx = np.asarray(inp["target"], dtype=np.float64)
        tgt = np.asarray(label["target"], dtype=np.float64)
        if ctx.ndim == 2:
            out.extend(Window(sid * ctx.shape[0] + c, w, ctx[c], tgt[c]) for c in range(ctx.shape[0]))
        else:
            out.append(Window(sid, w, ctx, tgt))
    return out, {"prediction_length": ds.prediction_length, "freq": ds.freq, "windows": W}
