"""Calibration evaluation and recalibration for time series foundation models."""

__version__ = "0.1.0"

from .data import Window, from_csv, from_dataframe, gift_eval_windows, rolling_windows
from .evaluate import DEFAULT_LEVELS, Evaluation, compare, evaluate, forecast
from .forecasts import Forecasts, align
from .metrics import (
    CalibrationReport,
    calibration_report,
    conditional_coverage,
    coverage,
    crps_quantile,
    interval_coverage,
    pce,
    qce,
)
from .models import Forecaster, FunctionForecaster, get_model, list_models, register, samples_to_quantiles
from .recalibration import (
    HistogramBinningRecalibrator,
    IsotonicRecalibrator,
    KDERecalibrator,
    PlattRecalibrator,
    Recalibrator,
    get_recalibrator,
    pit_from_quantile_grid,
)
from .significance import TestResult, delta_c_test, delta_h_test

__all__ = [
    "Window", "from_csv", "from_dataframe", "gift_eval_windows", "rolling_windows",
    "DEFAULT_LEVELS", "Evaluation", "compare", "evaluate", "forecast",
    "Forecasts", "align",
    "CalibrationReport", "calibration_report", "conditional_coverage", "coverage", "crps_quantile",
    "interval_coverage", "pce", "qce",
    "Forecaster", "FunctionForecaster", "get_model", "list_models", "register", "samples_to_quantiles",
    "HistogramBinningRecalibrator", "IsotonicRecalibrator", "KDERecalibrator", "PlattRecalibrator",
    "Recalibrator", "get_recalibrator", "pit_from_quantile_grid",
    "TestResult", "delta_c_test", "delta_h_test",
]
