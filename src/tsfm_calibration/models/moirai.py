"""Salesforce Moirai 1.1 / Moirai-MoE via uni2ts (sampling).  ``pip install 'tsfm-calibration[moirai]'``."""

from __future__ import annotations

import numpy as np

from .base import Forecaster, samples_to_quantiles

_REPOS = {
    "moirai-small": "Salesforce/moirai-1.1-R-small",
    "moirai-base": "Salesforce/moirai-1.1-R-base",
    "moirai-large": "Salesforce/moirai-1.1-R-large",
    "moirai-moe-small": "Salesforce/moirai-moe-1.0-R-small",
    "moirai-moe-base": "Salesforce/moirai-moe-1.0-R-base",
}


class Moirai(Forecaster):
    """``name`` is a key of ``_REPOS`` or a HF repo id; quantiles from ``num_samples`` paths.

    ``freq`` is the pandas frequency of the series; with ``patch_size="auto"`` it selects the patch
    size, so set it to the data's frequency (``freq`` may be reassigned between datasets).
    Contexts are truncated to ``context_length``.
    """

    def __init__(self, name: str = "moirai-small", context_length: int = 512, num_samples: int = 100,
                 patch_size: int | str = "auto", batch_size: int = 32, freq: str = "h"):
        self.name = name.split("/")[-1]
        self.repo = _REPOS.get(name, name)
        self.moe = "moe" in self.repo.lower()
        self.context_length, self.num_samples = context_length, num_samples
        self.patch_size, self.batch_size, self.freq = patch_size, batch_size, freq
        self._module, self._predictors = None, {}

    def _predictor(self, prediction_length: int):
        key = (prediction_length, self.freq)
        if key not in self._predictors:
            if self.moe:
                from uni2ts.model.moirai_moe import MoiraiMoEForecast as Forecast
                from uni2ts.model.moirai_moe import MoiraiMoEModule as Module
            else:
                from uni2ts.model.moirai import MoiraiForecast as Forecast
                from uni2ts.model.moirai import MoiraiModule as Module
            self._module = self._module or Module.from_pretrained(self.repo)
            # MoE checkpoints do not support patch_size="auto"
            ps = self._module.patch_sizes[0] if self.moe and self.patch_size == "auto" else self.patch_size
            model = Forecast(module=self._module, prediction_length=prediction_length,
                             context_length=self.context_length, patch_size=ps, num_samples=self.num_samples,
                             target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self._predictors[key] = model.create_predictor(batch_size=self.batch_size)
        return self._predictors[key]

    def predict_quantiles(self, contexts, prediction_length, levels):
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        start = pd.Period("2000-01-01", freq=self.freq)
        ds = ListDataset([{"start": start, "target": np.asarray(c, dtype=np.float32)} for c in contexts],
                         freq=self.freq)
        samples = np.stack([fc.samples for fc in self._predictor(prediction_length).predict(ds)])
        return samples_to_quantiles(samples, levels)
