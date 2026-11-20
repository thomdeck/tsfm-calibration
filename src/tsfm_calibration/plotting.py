"""Diagnostic plots for the calibration hierarchy and its two gaps (requires matplotlib).

plot_hierarchy             PCE_pool <= PCE_avg <= QCE_avg (<= QCE_max), before / after recalibration
plot_horizon_error         PCE per horizon vs. pooled PCE (Delta_H)
plot_conditional_coverage  share of outcomes below a predicted quantile, by quintile of its value (Delta_C)
plot_diagnostics           plot_hierarchy and plot_horizon_error in one figure
"""

from __future__ import annotations

import warnings

import numpy as np

from .forecasts import Forecasts
from .metrics import hits_and_mask
from .recalibration import _interpolate_relabeled
from .significance import permute_series, series_groups

__all__ = ["plot_hierarchy", "plot_horizon_error", "plot_conditional_coverage", "plot_diagnostics"]

ORIGINAL, RECALIBRATED = "#1f77b4", "#2ca02c"  # blue / green, as in the paper
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ebeae5"


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.subplots(figsize=(4.6, 3.4))[1]
    return ax


def _style(ax, title: str | None = None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10)


def _level_index(levels, level: float) -> int:
    i = int(np.argmin(np.abs(np.asarray(levels) - level)))
    if not np.isclose(levels[i], level):
        raise ValueError(f"level {level} not in {list(levels)}")
    return i


def plot_hierarchy(ev, include_max: bool = False, ax=None):
    """Calibration errors of an ``Evaluation``: test split before / after recalibration (audit if none).

    Brackets mark Delta_H and Delta_C of the original forecasts, labels on the recalibrated bars
    the relative change.  The black tick on the recalibrated PCE_avg bar is Delta_H / 2, a floor
    for PCE_avg under any global relabeling map.  ``include_max`` adds QCE_max.
    """
    ax = _ax(ax)
    keys = ("pce_pool", "pce_avg", "qce_avg") + (("qce_max",) if include_max else ())
    names = ("PCE$_{pool}$", "PCE$_{avg}$", "QCE$_{avg}$", "QCE$_{max}$")[:len(keys)]
    reports = [("original", ev.audit, ORIGINAL)] if ev.after is None else \
        [("original", ev.before, ORIGINAL), ("recalibrated", ev.after, RECALIBRATED)]
    x, w = np.arange(len(keys)), 0.8 / len(reports)
    top = max(getattr(r, m) for _, r, _ in reports for m in keys)
    for k, (label, rep, color) in enumerate(reports):
        ax.bar(x + (k - (len(reports) - 1) / 2) * w, [getattr(rep, m) for m in keys], w * 0.92,
               color=color, label=label, zorder=3)
    o = reports[0][1]
    xo = x - (len(reports) - 1) / 2 * w  # centres of the original bars
    for i, (gap, sym) in enumerate(((o.delta_h, "Delta_H"), (o.delta_c, "Delta_C"))):
        lo, hi = getattr(o, keys[i]), getattr(o, keys[i + 1])
        ax.plot([xo[i] - 0.46 * w, xo[i + 1] + 0.46 * w], [lo, lo], color=INK2, lw=0.8, ls=":", zorder=4)
        ax.text(xo[i + 1], hi + 0.02 * top, f"+$\\{sym}$\n{gap:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK, zorder=5)
    if ev.after is not None:
        for i, m in enumerate(keys):
            before, after = getattr(ev.before, m), getattr(ev.after, m)
            if before > 0:
                ax.text(x[i] + w / 2, after + 0.02 * top, f"{100 * (after / before - 1):+.0f}%", ha="center",
                        va="bottom", fontsize=8, color=RECALIBRATED, zorder=5)
        xr = x[1] + w / 2
        ax.plot([xr - 0.46 * w, xr + 0.46 * w], [o.delta_h / 2] * 2, color=INK, lw=1.6, zorder=6,
                label="floor $\\Delta_H/2$")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.35 * top)
    ax.set_ylabel("calibration error")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    return ax


def _recalibrated(f: Forecasts, labels) -> np.ndarray | None:
    """Quantiles of ``f`` re-interpolated to its own levels after relabeling to ``labels`` (as ``transform``)."""
    return None if labels is None else _interpolate_relabeled(f.qf, labels, np.asarray(f.levels, dtype=np.float64))


def _versions(f: Forecasts, labels):
    out = [("original", f.qf, ORIGINAL)]
    if labels is not None:
        out.append(("recalibrated", _recalibrated(f, labels), RECALIBRATED))
    return out


def _target_line(ax, level: float):
    ax.axhline(100 * level, color=INK, lw=1.0, ls=":", zorder=4, label=f"target {100 * level:g}%")


def _ylabel(level: float) -> str:
    return f"% of outcomes below the\nforecast {100 * level:g}% quantile"


def plot_horizon_error(ev, ax=None):
    """PCE per horizon of an ``Evaluation`` (test split before / after recalibration; audit if none).

    Dashed lines are PCE_pool, the error of coverage averaged over horizons, which is what pooled
    benchmarks report.  Per-horizon errors above it are miscalibration that pooling cancels
    (Delta_H); a global map lowers the pooled error more than the per-horizon errors.
    Scored like ``plot_hierarchy``: recalibrated forecasts against their relabeled levels r(tau).
    """
    ax = _ax(ax)
    reports = [("original", ev.audit, ORIGINAL)] if ev.after is None else \
        [("original", ev.before, ORIGINAL), ("recalibrated", ev.after, RECALIBRATED)]
    for name, rep, color in reports:
        e = np.asarray(rep.pce_per_horizon)
        ax.plot(np.arange(1, len(e) + 1), e, color=color, lw=1.8, label=name, zorder=3)
        ax.axhline(rep.pce_pool, ls="--", lw=1.1, color=color, zorder=2)
    ax.plot([], [], ls="--", lw=1.1, color=INK2, label="pooled error (PCE$_{pool}$)")
    ax.set(xlabel="horizon h", ylabel="calibration error", xlim=(1, len(e)))
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style(ax)
    return ax


def _quantile_bins(x: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    """Within-horizon quantile bin (0 .. n_bins - 1) of x (N, H); -1 where invalid."""
    b = np.full(x.shape, -1)
    for h in range(x.shape[1]):
        idx = np.flatnonzero(valid[:, h])
        r = np.empty(len(idx))
        r[np.argsort(x[idx, h], kind="mergesort")] = np.arange(len(idx))
        b[idx, h] = np.minimum((n_bins * r / max(len(idx), 1)).astype(int), n_bins - 1)
    return b


def _binned_rate(z: np.ndarray, valid: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    m = valid & (bins >= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(bins[m], weights=z[m], minlength=n_bins) / np.bincount(bins[m], minlength=n_bins)


def _share_by_bin(y, qf, q: int, n_bins: int):
    """Share of outcomes below quantile q per within-horizon quantile bin of its value; also bins, hits, mask."""
    z, valid = hits_and_mask(y, qf)
    bins = _quantile_bins(qf[:, :, q], valid, n_bins)
    return _binned_rate(z[:, :, q], valid, bins, n_bins), bins, z[:, :, q], valid


def plot_conditional_coverage(f: Forecasts, level: float, labels=None, n_bins: int = 5, null_band: bool = True,
                              B: int = 200, seed: int = 0, ax=None):
    """Share of outcomes below the predicted level-quantile, by quintile of that predicted value.

    Quintiles are formed within each horizon and pooled; the target is ``level`` in every bin.
    With ``labels`` (Q,) the second bars use the recalibrated quantiles (``Recalibrator.transform``).
    The grey band is the pointwise 5-95% range of the original bars if coverage were unrelated to
    the predicted value (series permutation, as in ``delta_c_test``); bars outside it indicate
    value-dependent miscalibration (Delta_C), which a global map cannot remove.
    """
    ax = _ax(ax)
    q = _level_index(f.levels, level)
    x = np.arange(1, n_bins + 1)
    if labels is not None and not np.min(labels) <= level <= np.max(labels):
        warnings.warn(f"level {level:g} is outside the relabeled range [{np.min(labels):.3g}, {np.max(labels):.3g}]: "
                      "the recalibrated quantile is clamped to the outermost one", stacklevel=2)
    versions = _versions(f, labels)
    w = 0.36 if len(versions) > 1 else 0.0
    groups = series_groups(f.series_ids, f.window_indices) if null_band else []
    if null_band and not any(len(g) > 1 for g in groups):
        warnings.warn("null band unavailable: no exchangeable series group has at least two series", stacklevel=2)
        null_band = False
    for k, (name, qf, color) in enumerate(versions):
        share, bins, z, valid = _share_by_bin(f.y, qf, q, n_bins)
        xk = x + (k - (len(versions) - 1) / 2) * w
        if null_band and k == 0:
            if not isinstance(B, (int, np.integer)) or B < 1:
                raise ValueError("B must be a positive integer")
            rng = np.random.default_rng(seed)
            null = np.stack([_binned_rate(z[src], valid[src], bins, n_bins)
                             for src in (permute_series(groups, len(f), rng) for _ in range(B))])
            lo, hi = np.nanquantile(null, [0.05, 0.95], axis=0)
            ax.bar(xk, 100 * (hi - lo), w * 0.8, bottom=100 * lo, color=BAND, edgecolor=MUTED, lw=0.6,
                   zorder=2, label="expected without value dependence")
        ax.plot(xk, 100 * share, "o", ms=7, color=color, label=name, zorder=3)
    _target_line(ax, level)
    ax.set_xticks(x, ["lowest"] + [str(k) for k in x[1:-1]] + ["highest"])
    ax.set(xlabel=f"forecast {100 * level:g}% quantile ({'quintile' if n_bins == 5 else 'bin'})",
           ylabel=_ylabel(level))
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(0, hi_y + 0.45 * hi_y)
    ax.set_xlim(0.4, n_bins + 0.6)
    handles, names = ax.get_legend_handles_labels()
    order = sorted(range(len(names)), key=lambda i: names[i].startswith("expected"))  # band last
    ax.legend([handles[i] for i in order], [names[i] for i in order], frameon=False, fontsize=8,
              loc="upper left", ncol=2)
    _style(ax)
    ax.grid(axis="x", visible=False)
    return ax


def plot_diagnostics(f: Forecasts, ev=None):
    """(a) calibration hierarchy and (b) calibration error by horizon, as one figure.

    ``ev`` is the ``Evaluation`` of ``f``; with a recalibrator both panels use its test split and
    show the forecasts before (blue) and after (green) recalibration, scored against the
    relabeled levels r(tau) as in the paper.  For coverage by predicted value (Delta_C) see
    ``plot_conditional_coverage``.
    """
    import matplotlib.pyplot as plt

    from .evaluate import evaluate

    ev = evaluate(f, recalibrator=None) if ev is None else ev
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    plot_hierarchy(ev, ax=axes[0])
    plot_horizon_error(ev, ax=axes[1])
    for ax, t in zip(axes, ("(a) Calibration errors", "(b) Calibration error by horizon")):
        ax.set_title(t, loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig
