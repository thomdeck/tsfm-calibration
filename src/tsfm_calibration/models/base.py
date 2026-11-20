"""Forecaster interface and model registry.

A forecaster maps a batch of univariate contexts to quantiles::

    class MyModel(Forecaster):
        def predict_quantiles(self, contexts, prediction_length, levels):
            ...  # -> np.ndarray (B, H, Q)

    register("my-model")(MyModel)      # optional: makes get_model("my-model") work
"""

from __future__ import annotations

import importlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["Forecaster", "FunctionForecaster", "samples_to_quantiles", "register", "get_model", "list_models"]


class Forecaster(ABC):
    """Probabilistic forecaster returning quantiles for a batch of univariate series."""

    name: str = "forecaster"

    @abstractmethod
    def predict_quantiles(self, contexts: list[np.ndarray], prediction_length: int,
                          levels: Sequence[float]) -> np.ndarray:
        """Quantile forecasts (B, H, Q) for ``contexts`` (list of 1-D arrays, may contain NaN)."""


class FunctionForecaster(Forecaster):
    """Wrap any callable ``fn(contexts, prediction_length, levels) -> (B, H, Q)``."""

    def __init__(self, fn: Callable[..., np.ndarray], name: str = "custom"):
        self.fn, self.name = fn, name

    def predict_quantiles(self, contexts, prediction_length, levels):
        return np.asarray(self.fn(contexts, prediction_length, levels))


def samples_to_quantiles(samples: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Empirical quantiles (B, H, Q) from sample paths (B, S, H)."""
    return np.moveaxis(np.quantile(samples, levels, axis=1), 0, -1)


def interpolate_levels(qf: np.ndarray, native: Sequence[float], levels: Sequence[float]) -> np.ndarray:
    """Linear interpolation of quantiles (..., Q_native) to ``levels``; flat outside the native grid."""
    native, levels = np.asarray(native, dtype=np.float64), np.asarray(levels, dtype=np.float64)
    if native.shape == levels.shape and np.allclose(native, levels):
        return qf
    if levels.min() < native.min() - 1e-9 or levels.max() > native.max() + 1e-9:
        warnings.warn(f"levels outside the native grid [{native.min()}, {native.max()}] are clamped "
                      "to the outermost native quantile", stacklevel=3)
    k = np.clip(np.searchsorted(native, levels, side="right"), 1, len(native) - 1)
    w = np.clip((levels - native[k - 1]) / (native[k] - native[k - 1]), 0.0, 1.0)
    return qf[..., k - 1] * (1 - w) + qf[..., k] * w


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., Forecaster]] = {}

# Built-in wrappers are imported lazily so that missing optional dependencies
# only fail when the model is requested.
_CHRONOS, _MOIRAI = "tsfm_calibration.models.chronos:Chronos", "tsfm_calibration.models.moirai:Moirai"
_BUILTINS: dict[str, str] = {
    "seasonal-naive": "tsfm_calibration.models.baseline:SeasonalNaive",
    **{f"chronos-t5-{s}": _CHRONOS for s in ("tiny", "mini", "small", "base", "large")},
    **{f"chronos-bolt-{s}": _CHRONOS for s in ("tiny", "mini", "small", "base")},
    "chronos-2": _CHRONOS,
    **{f"moirai-{s}": _MOIRAI for s in ("small", "base", "large")},
    **{f"moirai-moe-{s}": _MOIRAI for s in ("small", "base")},
    "tirex": "tsfm_calibration.models.tirex:TiRex",
}


def register(name: str):
    """Class/factory decorator: ``get_model(name, **kwargs)`` then calls ``obj(**kwargs)``."""
    def deco(obj):
        _REGISTRY[name] = obj
        return obj
    return deco


def get_model(name: str, **kwargs) -> Forecaster:
    """Instantiate a registered or built-in model by name."""
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name in _BUILTINS:
        module, cls = _BUILTINS[name].split(":")
        obj = getattr(importlib.import_module(module), cls)
        return obj(**kwargs) if name == "seasonal-naive" else obj(name, **kwargs)
    raise KeyError(f"unknown model {name!r}; available: {list_models()}")


def list_models() -> list[str]:
    return sorted(set(_REGISTRY) | set(_BUILTINS))
