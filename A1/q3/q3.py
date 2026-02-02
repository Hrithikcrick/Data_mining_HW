#!/usr/bin/env python3
"""
q3_report_plots.py

Generate all 6 plots from COL761 A1 Q3 candidates.dat (q # ... / c # ... format).

Usage:
  python q3_report_plots.py
    - expects candidates.dat in the CURRENT working directory

  python q3_report_plots.py /path/to/candidates.dat
    - reads candidates.dat from the given path

Output:
  - Shows the 6 plots on screen
  - Also saves PNGs in the same directory as candidates.dat:
      fig1_raw.png, fig2_log.png, fig3_hist.png, fig4_cdf.png, fig5_top20.png, fig6_zero_ratio.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_candidate_sizes(path: Path) -> np.ndarray:
    """
    Parse candidates.dat:
      q # <qid>
      c # <id1> <id2> ...
    Returns: array where each entry = number of candidates for that query.
    """
    sizes = []
    count = 0
    started = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("q #"):
                if started:
                    sizes.append(count)
                started = True
                count = 0

            elif line.startswith("c #"):
                parts = line.split()
                # format: ["c", "#", id1, id2, ...]
                count += max(0, len(parts) - 2)

            else:
                # safety: if candidate ids continue on next line (rare)
                parts = line.split()
                count += sum(1 for p in parts if p.isdigit())

    if started:
        sizes.append(count)

    return np.array(sizes, dtype=int)


def main():
    # 1) Resolve candidates.dat path
    if len(sys.argv) >= 2:
        candidates_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        candidates_path = Path("candidates.dat").resolve()

    if not candidates_path.exists():
        print(f"\nERROR: candidates.dat not found at:\n  {candidates_path}\n")
        print("Fix:")
        print("  (A) cd into the folder containing candidates.dat, then run:")
        print("      python q3_report_plots.py")
        print("  OR")
        print("  (B) pass the file path explicitly:")
        print("      python q3_report_plots.py /full/path/to/candidates.dat\n")
        sys.exit(1)

    out_dir = candidates_path.parent

    # 2) Load sizes
    candidate_sizes = load_candidate_sizes(candidates_path)
    query_ids = np.arange(len(candidate_sizes))  # 0-based IDs like your report plots

    # 3) Quick stats
    print("\n✅ Loaded:", candidates_path)
    print("Total queries:", len(candidate_sizes))
    print("Min:", candidate_sizes.min())
    print("Max:", candidate_sizes.max())
    print("Mean:", candidate_sizes.mean())
    print("Median:", float(np.median(candidate_sizes)))
    print("Zero-candidate queries:", int(np.sum(candidate_sizes == 0)))

    # -----------------------------
    # Figure 1: Raw
    # -----------------------------
    plt.figure()
    plt.plot(query_ids, candidate_sizes)
    plt.title("Candidate Set Size per Query Graph (Raw)")
    plt.xlabel("Query Graph ID")
    plt.ylabel("Candidate Set Size")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_raw.png", dpi=200)
    plt.show()

    # -----------------------------
    # Figure 2: Log-scaled
    # -----------------------------
    plt.figure()
    plt.plot(query_ids, np.log(candidate_sizes))
    plt.title("Log-Scaled Candidate Set Size per Query Graph")
    plt.xlabel("Query Graph ID")
    plt.ylabel("log(Candidate Set Size)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_log.png", dpi=200)
    plt.show()

    # -----------------------------
    # Figure 3: Histogram (log x)
    # -----------------------------
    plt.figure()
    plt.hist(candidate_sizes, bins=30)
    plt.xscale("log")
    plt.title("Histogram of Candidate Set Sizes")
    plt.xlabel("Candidate Set Size (log scale)")
    plt.ylabel("Number of Queries")
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_hist.png", dpi=200)
    plt.show()

    # -----------------------------
    # Figure 4: CDF (log x)
    # -----------------------------
    sorted_sizes = np.sort(candidate_sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

    plt.figure()
    plt.plot(sorted_sizes, cdf)
    plt.xscale("log")
    plt.title("CDF of Candidate Set Sizes")
    plt.xlabel("Candidate Set Size (log scale)")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_cdf.png", dpi=200)
    plt.show()

    # -----------------------------
    # Figure 5: Top-20 hardest queries
    # -----------------------------
    k = min(20, len(candidate_sizes))
    top_idx = np.argsort(candidate_sizes)[-k:][::-1]

    plt.figure()
    plt.bar(query_ids[top_idx].astype(str), candidate_sizes[top_idx])
    plt.title("Top-20 Hardest Queries")
    plt.xlabel("Query ID")
    plt.ylabel("Candidate Set Size")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_top20.png", dpi=200)
    plt.show()

    # -----------------------------
    # Figure 6: Zero vs non-zero pie
    # -----------------------------
    zero_count = int(np.sum(candidate_sizes == 0))
    nonzero_count = int(np.sum(candidate_sizes > 0))

    plt.figure()
    plt.pie([zero_count, nonzero_count],
            labels=["Zero candidates", "Non-zero candidates"],
            autopct="%1.1f%%")
    plt.title("Zero-Candidate Query Ratio")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_zero_ratio.png", dpi=200)
    plt.show()

    print("\n✅ Saved PNGs to:", out_dir)
    print("   fig1_raw.png, fig2_log.png, fig3_hist.png, fig4_cdf.png, fig5_top20.png, fig6_zero_ratio.png\n")


if __name__ == "__main__":
    main()
