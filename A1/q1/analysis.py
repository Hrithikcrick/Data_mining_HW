"""
analysis.py

This script documents the analysis workflow used for comparing the
Apriori and FP-Growth algorithms in COL761 Assignment 1.

The experiments measure runtime performance at different minimum
support thresholds for:
  - Task 1: webdocs.dat dataset
  - Task 2: constructed synthetic dataset

NOTE:
- Runtime measurements were collected using `/usr/bin/time -v`
  on the Baadal server.
- Due to restricted environments on Baadal, plotting was done locally.
- Final plots and detailed analysis are included in q1.pdf.
"""

import matplotlib.pyplot as plt

# Minimum support thresholds used in the experiments
SUPPORTS = [5, 10, 25, 50, 90]

def plot_runtimes(apriori_times, fpgrowth_times, title, output_file):
    """
    Plot runtime vs minimum support for Apriori and FP-Growth.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(SUPPORTS, apriori_times, marker='o', label='Apriori')
    plt.plot(SUPPORTS, fpgrowth_times, marker='o', label='FP-Growth')

    plt.xlabel("Minimum Support (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":

    # Task 1: webdocs.dat
    # Apriori at 5% exceeded the 3600 second time limit
    apriori_task1 = [3600, 636, 38, 35, 33]
    fpgrowth_task1 = [176, 120, 36, 34, 33]

    plot_runtimes(
        apriori_task1,
        fpgrowth_task1,
        "Runtime Comparison on Webdocs Dataset",
        "runtime_webdocs.png"
    )

    # Task 2: constructed synthetic dataset
    apriori_task2 = [0.54, 0.54, 0.07, 0.04, 0.03]
    fpgrowth_task2 = [0.11, 0.07, 0.04, 0.03, 0.03]

    plot_runtimes(
        apriori_task2,
        fpgrowth_task2,
        "Runtime Comparison on Generated Dataset",
        "runtime_generated.png"
    )

