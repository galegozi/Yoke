import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Filenames like: validation_studyXXX_epochYYYY.csv
CSV_RE = re.compile(r"validation_study(?P<sid>\d{3})_epoch(?P<ep>\d{4})\.csv$", re.IGNORECASE)

def normalize_sid(k) -> str:
    """Accept 7, '7', '007' → '007'."""
    s = re.sub(r"\D", "", str(k))
    return f"{int(s):03d}"

def load_study_epoch_stats(study_dir: Path):
    """
    Load validation CSVs for a study.

    CSVs have no header and 3 columns with possible spaces after commas:
      epochID, batchID, loss

    For each epoch file, compute:
      - median loss
      - [2.5%, 97.5%] quantiles
    Returns:
      stats: dict[epoch -> {"median": float, "q025": float, "q975": float}]
      series_median: pd.Series indexed by epoch with median loss (for line plots)
    """
    stats = {}
    rows_for_series = []

    for f in sorted(study_dir.glob("validation_study*_epoch*.csv")):
        m = CSV_RE.search(f.name)
        if not m:
            continue
        epoch_from_name = int(m.group("ep"))
        df = pd.read_csv(
            f,
            header=None,
            names=["epochID", "batchID", "loss"],
            sep=",",
            engine="python",
            skipinitialspace=True,
        )

        if len(df) == 0:
            continue

        losses = df["loss"].to_numpy()
        median = float(np.median(losses))
        q025 = float(np.quantile(losses, 0.025))
        q975 = float(np.quantile(losses, 0.975))

        stats[epoch_from_name] = {"median": median, "q025": q025, "q975": q975}
        rows_for_series.append((epoch_from_name, median))

    series_median = (
        pd.Series({ep: med for ep, med in rows_for_series}).sort_index()
        if rows_for_series else pd.Series(dtype=float)
    )
    return stats, series_median

def plot_lines(series_by_sid, labels_by_sid, title, outpath, ylim):
    """Plot one line per study. Legend label taken from labels_by_sid[sid]."""
    plt.figure()
    for sid, s in sorted(series_by_sid.items(), key=lambda kv: kv[0]):
        if s.empty:
            continue
        plt.plot(s.index, s.values, label=labels_by_sid[sid])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (median)")
    if ylim[1] is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_per_epoch_scatter(epoch_k: int, study_epoch_stats, sid_to_maxoff, outdir: Path):
    """
    For a fixed epoch K, plot median validation loss vs max_offset,
    with error bars = [2.5%, 97.5%] quantiles across batches.
    """
    xs, ys, yerr_lower, yerr_upper = [], [], [], []
    for sid, epochs in study_epoch_stats.items():
        if epoch_k in epochs:
            mo = sid_to_maxoff[sid]
            med = epochs[epoch_k]["median"]
            q025 = epochs[epoch_k]["q025"]
            q975 = epochs[epoch_k]["q975"]
            xs.append(mo)
            ys.append(med)
            yerr_lower.append(med - q025)
            yerr_upper.append(q975 - med)

    if not xs:
        return None

    figdir = outdir / "per_epoch"
    figdir.mkdir(parents=True, exist_ok=True)
    # LINE SEPERATOR
    outpath = figdir / f"val_loss_vs_min_off__epoch_{epoch_k}.png"

    plt.figure()
    plt.errorbar(xs, ys, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=4)
    # LINE SEPERATOR
    plt.xlabel("Min offset")
    plt.ylabel("Validation Loss (median, 95% quantile CI)")
    # LINE SEPERATOR
    plt.title(f"Validation Loss vs Min offset @ Epoch {epoch_k}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return outpath

def parse_epochs_arg(arg_str):
    return [int(t.strip()) for t in arg_str.split(",") if t.strip() != ""]

def main():
    ap = argparse.ArgumentParser(
        description="Validation loss plots with per-epoch quantile-based CIs."
    )
    # ap.add_argument("--root", required=True, help="Root dir containing study_X folders.")
    # ap.add_argument("--map", required=True,
    #                 help='JSON dict mapping study index to max_offset (one-to-one).')
    ap.add_argument("--out", default="/usr/projects/artimis/mpmm/galgal/Yoke/applications/evaluation/fixed-max", help="Output dir.")
    ap.add_argument("--epochs", default=None,
                    help="Comma-separated list of epochs for per-epoch plots. "
                         "If omitted, use 1..min_len across studies.")
    ap.add_argument("--ylim", default=None, type=float)
    args = ap.parse_args()

    root = Path("/usr/projects/artimis/mpmm/galgal/Yoke/applications/harnesses/chicoma_lsc_loderunner-ch-subsampling/runs")
    outdir = Path(args.out)

    # Parse and normalize mapping (one-to-one study ↔ max_offset)
    raw_map = {
        600: 0,
        601: 1,
        602: 2,
        605: 5,
        610: 10,
        620: 20,
        625: 25,
        630: 30,
        640: 40,
        650: 50,
        660: 60,
        670: 70,
        680: 80,
        690: 90,
        700: 100
    }
    sid_to_maxoff = {normalize_sid(k): int(v) for k, v in raw_map.items()}

    # Load studies
    study_dirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"^study_\d+$", p.name)]
    series_by_sid, labels_by_sid, lengths_by_sid, study_epoch_stats = {}, {}, {}, {}

    for sd in study_dirs:
        sid_str = normalize_sid(re.sub(r"[^\d]", "", sd.name))
        if sid_str not in sid_to_maxoff:
            continue
        stats, s_median = load_study_epoch_stats(sd)
        if s_median.empty:
            continue
        study_epoch_stats[sid_str] = stats
        series_by_sid[sid_str] = s_median
        # LINE SEPERATOR
        labels_by_sid[sid_str] = f"min off {sid_to_maxoff[sid_str]}"
        lengths_by_sid[sid_str] = int(s_median.index.max())

    if not series_by_sid:
        print("[ERROR] No usable studies found.")
        return

    # --- Full-length plot
    fig1 = outdir / "val_loss__per_study__ALL.png"
    plot_lines(series_by_sid, labels_by_sid, "Validation Loss vs Epoch (All Data)", fig1, (0, args.ylim))
    print(f"[OK] Wrote: {fig1}")

    # --- Aligned-short plot
    min_len = min(lengths_by_sid.values())
    max_len = max(lengths_by_sid.values())
    short_series = {sid: s[s.index <= min_len] for sid, s in series_by_sid.items()}
    fig2 = outdir / f"val_loss__per_study__ALIGNED_to_{min_len}.png"
    plot_lines(short_series, labels_by_sid, f"Validation Loss vs Epoch (Aligned to {min_len})", fig2, (0, args.ylim))
    print(f"[OK] Wrote: {fig2}")

    # --- Console advisory
    limiting_sids = [sid for sid, L in lengths_by_sid.items() if L == min_len]
    next_target = min((L for L in lengths_by_sid.values() if L > min_len), default=None)
    print("\n=== Short-graph length analysis ===")
    print(f"Current shortest length: {min_len}")
    print(f"Studies at this length: {', '.join(limiting_sids)}")
    if next_target:
        delta = next_target - min_len
        print(f"Next candidate length: {next_target} (Δ={delta})")
        print(f"To extend, run {', '.join(limiting_sids)} for +{delta} epoch(s)")
    else:
        print("All studies end at the same epoch.")

    # --- Per-epoch scatter plots
    if args.epochs:
        epochs_to_plot = parse_epochs_arg(args.epochs)
    else:
        epochs_to_plot = list(range(5, max_len + 1, 5))

    for k in epochs_to_plot:
        outpath = plot_per_epoch_scatter(k, study_epoch_stats, sid_to_maxoff, outdir)
        # if outpath:
        #     print(f"[OK] Wrote per-epoch plot: {outpath}")
        # else:
        #     print(f"[INFO] Skipped epoch {k}: no data.")

if __name__ == "__main__":
    main()