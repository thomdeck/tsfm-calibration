"""Regenerate docs/diagnostics.png, the figure in the README.

Chronos-2 on M_DENSE/H from the GIFT-Eval Hugging Face repository (30 hourly series,
about 17 MB), 20 rolling windows of 48 steps per series as in the GIFT-Eval short term,
isotonic recalibration on the chronological 20/80 split.  About 25 s on a GPU.

    pip install "tsfm-calibration[chronos,notebook]"
    python examples/readme_figure.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
from huggingface_hub import hf_hub_download

import tsfm_calibration as tc
from tsfm_calibration.plotting import plot_diagnostics

matplotlib.use("Agg")


def load_m_dense() -> pd.DataFrame:
    """M_DENSE/H as a long-format table (item_id, timestamp, target)."""
    path = hf_hub_download("Salesforce/GiftEval", "M_DENSE/H/data-00000-of-00001.arrow", repo_type="dataset")
    table = pa.ipc.open_stream(path).read_all().to_pandas()
    return pd.concat([pd.DataFrame({"item_id": r.item_id, "target": np.asarray(r.target, dtype=float),
                                    "timestamp": pd.date_range(r.start, periods=len(r.target), freq="h")})
                      for r in table.itertuples()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chronos-2")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "diagnostics.png"))
    args = ap.parse_args()

    series = tc.from_dataframe(load_m_dense())
    windows = tc.rolling_windows(series, prediction_length=48, num_windows=20)
    f = tc.forecast(args.model, windows, batch_size=64, progress=True)
    ev = tc.evaluate(f, recalibrator="isotonic", tests=True)
    print(ev.summary().drop(columns=["n_windows"]).apply(pd.to_numeric, errors="coerce").round(4).T)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_diagnostics(f, ev).savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
