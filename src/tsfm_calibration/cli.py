"""Command line interface.

    tsfm-calibration models
    tsfm-calibration evaluate --model chronos-bolt-small --csv data.csv --horizon 24 --num-windows 10
    tsfm-calibration evaluate --forecasts out/forecasts.npz --tests
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .data import from_csv, rolling_windows
from .evaluate import DEFAULT_LEVELS, evaluate, forecast
from .forecasts import Forecasts
from .models.base import list_models


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tsfm-calibration", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list available models")

    ev = sub.add_parser("evaluate", help="forecast a CSV with a model (or load forecasts) and evaluate calibration")
    src = ev.add_argument_group("input (either --forecasts, or --model with --csv)")
    src.add_argument("--forecasts", help="saved Forecasts .npz")
    src.add_argument("--model", help="model name, see `tsfm-calibration models`")
    src.add_argument("--model-kwargs", default="{}", help='JSON, e.g. \'{"device": "cpu"}\'')
    src.add_argument("--csv", help="long-format CSV with one row per series and time step")
    src.add_argument("--id-col", default="item_id")
    src.add_argument("--time-col", default="timestamp")
    src.add_argument("--target-col", default="target")
    win = ev.add_argument_group("windows")
    win.add_argument("--horizon", type=int, help="prediction length H")
    win.add_argument("--num-windows", type=int, default=10, help="rolling windows per series")
    win.add_argument("--stride", type=int, help="shift between windows, at least H (default: H)")
    win.add_argument("--max-context", type=int)
    win.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    win.add_argument("--batch-size", type=int, default=32)
    cal = ev.add_argument_group("recalibration and tests")
    cal.add_argument("--recalibrator", default="isotonic",
                     help="isotonic (default) | platt | histogram_binning | kde | none")
    cal.add_argument("--split", default="chronological", choices=["chronological", "random"])
    cal.add_argument("--cal-fraction", type=float, default=0.2)
    cal.add_argument("--tests", action="store_true", help="run the Delta_C / Delta_H significance tests")
    cal.add_argument("-B", type=int, default=999, help="null draws per test")
    cal.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out", default="results", help="output directory")
    ev.add_argument("--no-plots", action="store_true")
    return ap


def _plots(f: Forecasts, ev, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return
    from .plotting import plot_diagnostics

    plot_diagnostics(f, ev).savefig(out / "diagnostics.png", dpi=150)


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.cmd == "models":
        print("\n".join(list_models()))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.forecasts:
        f = Forecasts.load(args.forecasts)
    elif args.model and args.csv and args.horizon:
        series = from_csv(args.csv, args.id_col, args.time_col, args.target_col)
        windows = rolling_windows(series, args.horizon, args.num_windows, args.stride, args.max_context)
        f = forecast(args.model, windows, args.levels, args.batch_size, progress=True,
                     **json.loads(args.model_kwargs))
        f.save(out / "forecasts.npz")
    else:
        raise SystemExit("provide --forecasts, or --model, --csv and --horizon")

    rec = None if args.recalibrator == "none" else args.recalibrator
    ev = evaluate(f, rec, args.split, args.cal_fraction, args.seed, tests=args.tests, B=args.B)
    (out / "report.json").write_text(json.dumps({"metadata": f.metadata, **ev.to_dict()}, indent=2))
    ev.summary().to_csv(out / "summary.csv")
    if not args.no_plots:
        _plots(f, ev, out)
    print(ev.summary().drop(columns=["n_windows"]).round(4).to_string())
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
