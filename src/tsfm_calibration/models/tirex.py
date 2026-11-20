"""NX-AI TiRex (xLSTM, native quantiles 0.1, ..., 0.9).  ``pip install 'tsfm-calibration[tirex]'``."""

from __future__ import annotations

import numpy as np

from .base import Forecaster, interpolate_levels


class TiRex(Forecaster):
    """Direct quantile forecaster; levels off the native grid are interpolated (clamped outside 0.1-0.9)."""

    def __init__(self, name: str = "tirex", device: str | None = None, max_context: int = 2016):
        import torch
        from tirex import load_model

        self.name = name
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model("NX-AI/TiRex" if name == "tirex" else name, backend="torch", device=device)
        cfg = getattr(self.model, "config", None)
        self.native = np.asarray(getattr(cfg, "quantiles", np.linspace(0.1, 0.9, 9)), dtype=np.float64)
        # Respect the user's limit, but never exceed the context length supported by the checkpoint
        self.max_context = min(max_context, getattr(cfg, "max_context", max_context))

    def predict_quantiles(self, contexts, prediction_length, levels):
        import torch

        # Left-pad with NaN (treated as missing by TiRex) to a common length
        L = min(max(len(c) for c in contexts), self.max_context)
        batch = np.full((len(contexts), L), np.nan, dtype=np.float32)
        for i, c in enumerate(contexts):
            c = np.asarray(c, dtype=np.float32)[-L:]
            batch[i, L - len(c):] = c
        with torch.no_grad():
            q, _ = self.model.forecast(batch, prediction_length=prediction_length, output_type="numpy")
        qf = interpolate_levels(np.asarray(q, dtype=np.float64), self.native, levels)
        return np.maximum.accumulate(qf, axis=-1)
