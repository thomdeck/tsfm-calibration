"""Audit and recalibrate one model on GIFT-Eval with the paper's protocol.

Protocol: short term, levels 0.1..0.9, Delta_C / Delta_H tests with B null
draws, and the four pooled recalibration maps fitted on a chronological 20/80
split for configurations with >= 50 windows.  This runs the evaluation for your
model; it does not reproduce every table of the paper.

    pip install "tsfm-calibration[chronos]"  # install GIFT-Eval from source as described in the README
    python examples/gift_eval.py --model chronos-bolt-small --storage-path /data/gift-eval \
        --datasets m4_hourly electricity/H solar/10T --out results/gift_eval
"""

import argparse
from pathlib import Path

import pandas as pd

import tsfm_calibration as tc

MIN_WINDOWS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="GIFT-Eval configurations, e.g. electricity/H")
    ap.add_argument("--term", default="short")
    ap.add_argument("--storage-path", help="GIFT-Eval download (else $GIFT_EVAL)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("-B", type=int, default=999)
    ap.add_argument("--out", default="results/gift_eval")
    args = ap.parse_args()

    out = Path(args.out)
    model = tc.get_model(args.model)
    audit, recal = [], []
    for name in args.datasets:
        windows, info = tc.gift_eval_windows(name, args.term, args.storage_path)
        if hasattr(model, "freq"):  # e.g. Moirai derives its patch size from the series frequency
            model.freq = info["freq"]
        f = tc.forecast(model, windows, batch_size=args.batch_size, progress=True)
        f.metadata.update(dataset=name, term=args.term, **info)
        f.save(out / "forecasts" / args.model / name.replace("/", "_") / args.term)

        ev = tc.evaluate(f, recalibrator=None, tests=True, B=args.B)
        audit.append({"model": args.model, "dataset": name, **ev.audit.to_dict(),
                      "delta_c_corrected": ev.delta_c.corrected, "delta_c_p": ev.delta_c.p_value,
                      "delta_h_corrected": ev.delta_h.corrected, "delta_h_p": ev.delta_h.p_value,
                      "delta_c_reason": ev.delta_c.reason, "delta_h_reason": ev.delta_h.reason})

        if len(f) < MIN_WINDOWS:
            continue
        cal, test = f.chronological_split()
        for method in tc.recalibration.RECALIBRATORS:
            ev = tc.evaluate(f, recalibrator=method, split="chronological")
            row = {"model": args.model, "dataset": name, "method": method, "n_cal": len(cal), "n_test": len(test)}
            for k, v in ev.before.to_dict().items():
                if k != "n_windows":
                    row[f"{k}_before"], row[f"{k}_after"] = v, ev.after.to_dict()[k]
            recal.append(row)

    audit = pd.DataFrame(audit)
    audit.to_csv(out / "audit.csv", index=False)
    if recal:
        pd.DataFrame(recal).to_csv(out / "recalibration.csv", index=False)
    print(audit.round(4).to_string())
    for gap in ("delta_c", "delta_h"):
        p = audit[f"{gap}_p"].dropna()
        if p.empty:
            print(f"{gap}: inference unavailable for all {len(audit)} configurations")
        else:
            print(f"{gap}: {(p <= 0.05).sum()}/{len(p)} ({(p <= 0.05).mean():.0%}) significant at 0.05; "
                  f"{len(audit) - len(p)} unavailable")


if __name__ == "__main__":
    main()
