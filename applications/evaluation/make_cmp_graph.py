import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Filenames like: validation_studyXXX_epochYYYY.csv (X=3 digits, Y=4 digits)
CSV_RE = re.compile(r"validation_study(?P<sid>\d{3})_epoch(?P<ep>\d{4})\.csv$", re.IGNORECASE)

def normalize_sid(k) -> str:
    # Accept 7, "7", "007" → "007"
    s = re.sub(r"\D", "", str(k))
    return f"{int(s):03d}"

def load_study_epoch_series(study_dir: Path) -> pd.Series:
    """
    For each epoch file in a study directory:
      - read CSV with no header: columns are epochID, batchID, loss
      - allow spaces after commas
      - compute mean loss across all rows in that file (epoch)
    Returns a Series indexed by epoch number (int) with mean loss (float).
    """
    rows = []
    for f in sorted(study_dir.glob("validation_study*_epoch*.csv")):
        m = CSV_RE.search(f.name)
        if not m:
            continue
        epoch_from_name = int(m.group("ep"))  # trust filename epoch
        # read CSV: 3 cols, no header, spaces allowed
        df = pd.read_csv(
            f, header=None, names=["epochID", "batchID", "loss"],
            sep=",", engine="python", skipinitialspace=True
        )
        # Use the loss column's mean for that epoch (robust to multi-row files)
        loss_mean = float(df["loss"].mean())
        rows.append((epoch_from_name, loss_mean))

    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series({ep: loss for ep, loss in rows}).sort_index()
    return s

def plot_lines(series_by_sid, labels_by_sid, title, outpath):
    """
    Plot one line per study. Legend label is taken from labels_by_sid[sid].
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

def main():
    ap = argparse.ArgumentParser(
        description="Plot validation loss by epoch (one line per study)."
        )
    ap.add_argument("--out", default="/usr/projects/artimis/mpmm/galgal/Yoke/applications/evaluation/output", help="Output dir for figures.")
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
    for sd in study_dirs:
        sid_str = normalize_sid(re.sub(r"[^\d]", "", sd.name))
        if sid_str not in sid_to_maxoff:
            print(f"[INFO] Skipping {sd.name}: no max_offset provided for study {sid_str}")
            continue

        s = load_study_epoch_series(sd)
        if s.empty:
            print(f"[WARN] No matching validation CSVs in {sd}")
            continue

        series_by_sid[sid_str] = s
        labels_by_sid[sid_str] = f"max_offset {sid_to_maxoff[sid_str]}"
        lengths_by_sid[sid_str] = int(s.index.max())

    if not series_by_sid:
        print("[ERROR] No usable studies loaded.")
        return

    # Figure 1: full-length (each study to its own last epoch)
    fig1 = outdir / "val_loss__per_study__ALL.png"
    plot_lines(series_by_sid, labels_by_sid, "Validation Loss vs Epoch (All Available Data)", fig1)
    print(f"[OK] Wrote: {fig1}")

    # Figure 2: aligned-short (truncate to min last epoch across studies)
    min_len = min(lengths_by_sid.values())
    series_short = {sid: s[s.index <= min_len] for sid, s in series_by_sid.items()}
    fig2 = outdir / f"val_loss__per_study__ALIGNED_to_{min_len}.png"
    plot_lines(series_short, labels_by_sid, f"Validation Loss vs Epoch (Aligned to {min_len})", fig2)
    print(f"[OK] Wrote: {fig2}")

    # Console advisory: who limits the short graph and by how much to reach next-longest
    # Sort studies by their last epoch
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
        # Nicety: echo max_off values and folder guesses
        for sd in study_dirs:
            sid_str = normalize_sid(re.sub(r"[^\d]", "", sd.name))
            if sid_str in limiting_sids:
                mo = sid_to_maxoff.get(sid_str, "?")
                print(f"  - study {sid_str} (folder: {sd.name}, max_off={mo}) needs +{delta} epoch(s)")

if __name__ == "__main__":
    main()