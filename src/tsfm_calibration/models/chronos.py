"""Amazon Chronos (T5 sampling, Bolt / Chronos-2 direct quantiles).  ``pip install 'tsfm-calibration[chronos]'``."""

from __future__ import annotations

import numpy as np

from .base import Forecaster, samples_to_quantiles


class Chronos(Forecaster):
    """``name`` is ``chronos-t5-*``, ``chronos-bolt-*``, ``chronos-2`` or a HF repo id.

    Chronos-T5 forecasts ``num_samples`` paths (drawn in chunks to bound GPU
    memory) and returns their empirical quantiles; Bolt and Chronos-2 return
    native quantiles.
    """

    def __init__(self, name: str = "chronos-bolt-small", device: str | None = None,
                 torch_dtype: str = "float32", num_samples: int = 100, sample_chunk: int = 5):
        import torch
        from chronos import BaseChronosPipeline

        self.name = name.split("/")[-1]
        self.num_samples, self.sample_chunk = num_samples, sample_chunk
        self.kind = "t5" if "t5" in self.name else ("bolt" if "bolt" in self.name else "chronos2")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import transformers  # transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``
        new_api = tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (4, 56)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            name if "/" in name else f"amazon/{name}", device_map=device,
            **{"dtype" if new_api else "torch_dtype": getattr(torch, torch_dtype)})

    def predict_quantiles(self, contexts, prediction_length, levels):
        import torch

        levels = [float(t) for t in levels]
        with torch.no_grad():
            if self.kind == "t5":
                ctx = [torch.tensor(c, dtype=torch.float32) for c in contexts]
                chunks, left = [], self.num_samples
                while left > 0:
                    k = min(self.sample_chunk, left)
                    chunks.append(self.pipeline.predict(ctx, prediction_length=prediction_length,
                                                        num_samples=k).cpu().numpy())
                    left -= k
                qf = samples_to_quantiles(np.concatenate(chunks, axis=1), levels)
            elif self.kind == "bolt":
                ctx = [torch.tensor(c, dtype=torch.float32) for c in contexts]
                q, _ = self.pipeline.predict_quantiles(ctx, prediction_length=prediction_length,
                                                       quantile_levels=levels)
                qf = q.cpu().numpy()
            else:
                q, _ = self.pipeline.predict_quantiles(
                    [{"target": np.asarray(c, dtype=np.float32)} for c in contexts],
                    prediction_length=prediction_length, quantile_levels=levels)
                qf = np.stack([t.cpu().numpy().reshape(prediction_length, len(levels)) for t in q])
        return np.maximum.accumulate(qf.astype(np.float64), axis=-1)
