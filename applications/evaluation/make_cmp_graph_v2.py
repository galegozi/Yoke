import argparse
import json
import re
from pathlib import Path
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt

# Filenames like: validation_studyXXX_epochYYYY.csv (X=3 digits, Y=4 digits)
CSV_RE = re.compile(r"validation_study(?P<sid>\d{3})_epoch(?P<ep>\d{4})\.csv$", re.IGNORECASE)

def normalize_sid(k) -> str:
    # Accept 7, "7", "007" → "007"
    s = re.sub(r"\D", "", str(k))
    return f"{int(s):03d}"

def load_study_epoch_stats(study_dir: Path):
    """
    Load all validation CSVs in a study directory.

    CSVs have no header and three columns with possible spaces after commas:
      epochID, batchID, loss

    For each epoch file, compute:
      - mean loss across rows (batches)
      - std dev across rows (sample std, ddof=1)
      - n (number of rows/batches)

    Returns:
      stats: dict[epoch -> {"mean": float, "std": float, "n": int}]
      series_mean: pd.Series indexed by epoch with the mean loss (for line plots)
    """
    stats = {}
    rows_for_series = []

    for f in sorted(study_dir.glob("validation_study*_epoch*.csv")):
        m = CSV_RE.search(f.name)
        if not m:
            continue
        epoch_from_name = int(m.group("ep"))  # trust filename epoch
        df = pd.read_csv(
            f,
            header=None,
            names=["epochID", "batchID", "loss"],
            sep=",",
            engine="python",
            skipinitialspace=True,
        )

        n = int(len(df))
        if n == 0:
            continue

        mean_loss = float(df["loss"].mean())
        # sample std (ddof=1); fallback to 0.0 when n==1
        std_loss = float(df["loss"].std(ddof=1)) if n > 1 else 0.0

        stats[epoch_from_name] = {"mean": mean_loss, "std": std_loss, "n": n}
        rows_for_series.append((epoch_from_name, mean_loss))

    series_mean = (
        pd.Series({ep: mean for ep, mean in rows_for_series}).sort_index()
        if rows_for_series else pd.Series(dtype=float)
    )
    return stats, series_mean

def plot_lines(series_by_sid, labels_by_sid, title, outpath):
    """
    Plot one line per study. Legend label taken from labels_by_sid[sid].
    """
    plt.figure()
    for sid, s in sorted(series_by_sid.items(), key=lambda kv: kv[0]):
        if s.empty:
            continue
        plt.plot(s.index, s.values, label=labels_by_sid[sid])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def ci95_from_std(std: float, n: int) -> float:
    """
    95% confidence half-width using normal approximation:
      1.96 * std / sqrt(n)
    If n <= 1, return 0.0 (no variability estimate).
    """
    if n is None or n <= 1:
        return 0.0
    return 1.96 * std / sqrt(n)

def plot_per_epoch_scatter(epoch_k: int, study_epoch_stats, sid_to_maxoff, outdir: Path):
    """
    For a fixed epoch K, plot validation loss (mean ± 95% CI) vs max_offset.
    Saves to: per_epoch/val_loss_vs_max_off__epoch_K.png

    - Points are NOT connected.
    - Error bars use 95% CI across batches in that epoch's CSV (per study).
    """
    xs, ys, yerrs = [], [], []
    for sid, epochs in study_epoch_stats.items():
        if epoch_k in epochs:
            mo = sid_to_maxoff[sid]
            mean_ = epochs[epoch_k]["mean"]
            std_ = epochs[epoch_k]["std"]
            n_ = epochs[epoch_k]["n"]
            err_ = ci95_from_std(std_, n_)
            xs.append(mo)
            ys.append(mean_)
            yerrs.append(err_)

    if not xs:
        return None  # nothing to plot

    figdir = outdir / "per_epoch"
    figdir.mkdir(parents=True, exist_ok=True)
    outpath = figdir / f"val_loss_vs_max_off__epoch_{epoch_k}.png"

    plt.figure()
    # fmt="o" for markers only; no connecting lines
    plt.errorbar(xs, ys, yerr=yerrs, fmt="o", capsize=4)
    plt.xlabel("max_offset")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss vs max_offset @ Epoch {epoch_k}")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return outpath

def parse_epochs_arg(arg_str):
    """
    Accept strings like '10,25,50' and returns [10, 25, 50].
    Ignores empty tokens; raises on non-integers.
    """
    toks = [t.strip() for t in arg_str.split(",") if t.strip() != ""]
    return [int(t) for t in toks]

def main():
    ap = argparse.ArgumentParser(
        description="Plot validation loss: per-study lines + per-epoch vs max_offset with 95% CI."
    )
    ap.add_argument("--out", default="/usr/projects/artimis/mpmm/galgal/Yoke/applications/evaluation/output", help="Output dir for figures.")
    ap.add_argument(
        "--epochs",
        default=None,
        help="Comma-separated list of specific epochs to plot vs max_offset (e.g. '10,25,50'). "
             "If omitted, will generate for all epochs up to the shortest run.",
    )
    args = ap.parse_args()

    root = Path("/usr/projects/artimis/mpmm/galgal/Yoke/applications/harnesses/chicoma_lsc_loderunner-ch-subsampling/runs")
    outdir = Path(args.out)

    # Parse and normalize mapping (one-to-one study ↔ max_offset)
    raw_map = {
        102: 2,
        200: 100,
        300: 0,
        201: 1,
        205: 5,
        210: 10,
        225: 25,
        250: 50,
        220: 20,
        230: 30,
        240: 40,
        260: 60,
        270: 70,
        280: 80,
        290: 90
    }
    sid_to_maxoff = {normalize_sid(k): int(v) for k, v in raw_map.items()}

    # Discover study folders named like study_<number>
    study_dirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"^study_\d+$", p.name)]
    if not study_dirs:
        print(f"[WARN] No study_X directories found in {root}")
        return

    # Load each mapped study
    series_by_sid = {}
    labels_by_sid = {}
    lengths_by_sid = {}
    study_epoch_stats = {}  # sid -> {epoch -> {"mean","std","n"}}

    for sd in study_dirs:
        sid_str = normalize_sid(re.sub(r"[^\d]", "", sd.name))
        if sid_str not in sid_to_maxoff:
            print(f"[INFO] Skipping {sd.name}: no max_offset provided for study {sid_str}")
            continue

        stats, s_mean = load_study_epoch_stats(sd)
        if s_mean.empty:
            print(f"[WARN] No matching validation CSVs in {sd}")
            continue

        study_epoch_stats[sid_str] = stats
        series_by_sid[sid_str] = s_mean
        labels_by_sid[sid_str] = f"max_off {sid_to_maxoff[sid_str]}"
        lengths_by_sid[sid_str] = int(s_mean.index.max())

    if not series_by_sid:
        print("[ERROR] No usable studies loaded.")
        return

    # -------------------------
    # Figure 1: full-length per study
    # -------------------------
    fig1 = outdir / "val_loss__per_study__ALL.png"
    plot_lines(series_by_sid, labels_by_sid, "Validation Loss vs Epoch (All Available Data)", fig1)
    print(f"[OK] Wrote: {fig1}")

    # -------------------------
    # Figure 2: aligned-short per study
    # -------------------------
    min_len = min(lengths_by_sid.values())
    series_short = {sid: s[s.index <= min_len] for sid, s in series_by_sid.items()}
    fig2 = outdir / f"val_loss__per_study__ALIGNED_to_{min_len}.png"
    plot_lines(series_short, labels_by_sid, f"Validation Loss vs Epoch (Aligned to {min_len})", fig2)
    print(f"[OK] Wrote: {fig2}")

    # -------------------------
    # Console advisory (which studies limit the short plot)
    # -------------------------
    sorted_lengths = sorted(lengths_by_sid.items(), key=lambda kv: kv[1])
    limiting_sids = [sid for sid, L in sorted_lengths if L == min_len]
    distinct_lengths = sorted(set(lengths_by_sid.values()))
    next_target = next((L for L in distinct_lengths if L > min_len), None)

    print("\n=== Short-graph length analysis ===")
    print(f"Current shortest length (min last epoch): {min_len}")
    print(f"Studies at this length: {', '.join(limiting_sids)}")

    if next_target is None:
        print("All studies end at the same epoch. To lengthen the aligned-short plot, extend every study beyond "
              f"{min_len} epochs.")
    else:
        delta = next_target - min_len
        print(f"Next longer candidate length among existing runs: {next_target} (Δ = +{delta} epochs)")
        print("To extend the aligned-short graph to this length, run the following studies for at least "
              f"{delta} more epoch(s): {', '.join(limiting_sids)}")
        for sd in study_dirs:
            sid_str = normalize_sid(re.sub(r"[^\d]", "", sd.name))
            if sid_str in limiting_sids:
                mo = sid_to_maxoff.get(sid_str, "?")
                print(f"  - study {sid_str} (folder: {sd.name}, max_off={mo}) needs +{delta} epoch(s)")

    # -------------------------
    # Per-epoch plots (loss vs max_offset with 95% CI) 
    # -------------------------
    if args.epochs:
        requested_epochs = parse_epochs_arg(args.epochs)
        epochs_to_plot = sorted(set(requested_epochs))
    else:
        # Default: plot for all epochs where EVERY study has data (1..min_len)
        epochs_to_plot = list(range(5, min_len + 1, 5))

    made_any = False
    for k in epochs_to_plot:
        outpath = plot_per_epoch_scatter(k, study_epoch_stats, sid_to_maxoff, outdir)
        if outpath is not None:
            made_any = True
            # print(f"[INFO] Skipping per-epoch plot for epoch {k}: no studies with data.")
        # else:
            # print(f"[OK] Wrote per-epoch plot: {outpath}")
            # made_any = True

    if not made_any:
        print("[WARN] No per-epoch plots were generated (no matching epochs found among loaded studies).")

if __name__ == "__main__":
    main()