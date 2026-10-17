"""Summarize and plot the negativity-weighted graph.

Reads the trajectory records from scripts/ewg_run.py (large, not committed), saves the small
summary results/ewg_graph_L18_p0.17.npz (committed), and plots it to results/ewg_L18_p0.17.png.
Without a records file it plots the existing summary.

Colour: white at the bare measurement rate p, blue below, red above (as in arXiv:2602.04969, Fig. 6).
Each trajectory is also counted mirrored (A and B swapped), which flips the brickwork parity, so
circuits ending on even and odd layers enter half-half (see mipt_sim.ewg.mirror_sites).
Time runs upward, so the final state (where A and B are evaluated) is the top row. Only the last
`layers` time steps are shown: earlier ones sit at p within noise.
Usage: python scripts/ewg_plot.py [records.npz]
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from mipt_sim.ewg import summarize

RECORDS = "results/ewg_records_L18_p0.17.npz"
SUMMARY = "results/ewg_graph_L18_p0.17.npz"


def plot_ewg(s, layers=20, ax=None):
    p, L = float(s["p"]), int(s["L"])
    A, B = list(s["A"]), list(s["B"])
    g = s["graph"][-layers:]
    t0 = len(s["graph"]) - layers  # time steps shown: t0 + 1 ... depth
    shift = (L - 1 - A[0] - B[-1]) // 2  # roll the ring so A and B sit in the middle
    cols = (np.arange(L) - shift) % L
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=p, vmin=0.0, vmax=3 * p)  # values above 3p saturate
    im = ax.imshow(
        np.roll(g, shift, axis=1),
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        extent=(0.5, L + 0.5, t0 + 0.5, t0 + layers + 0.5),
    )
    ax.set_xticks(range(1, L + 1, 2), [str(cols[i] + 1) for i in range(0, L, 2)])
    ax.set_xlabel("site")
    ax.set_ylabel("time step (final state at the top)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    top = t0 + layers + 0.5
    for q, lab in [(A, "A"), (B, "B")]:
        q = [(x + shift) % L for x in q]
        ax.annotate(
            lab,
            ((q[0] + q[-1]) / 2 + 1, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            annotation_clip=False,
        )
        for x in q:
            ax.plot([x + 1], [top], marker="v", color="#33332f", ms=7, clip_on=False)
    ax.set_xlim(0.5, L + 0.5)
    ax.set_ylim(t0 + 0.5, top)
    ax.set_title(
        f"Negativity-weighted graph, L = {L}, p = {p:g}\n({int(s['n_entangled'])} of {int(s['n'])} trajectories entangle A and B)",
        pad=22,
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label=f"weighted measurement rate (white = p = {p:g})")
    return ax


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else RECORDS
    if os.path.exists(src):
        d = np.load(src)
        A, B = list(d["A"]), list(d["B"])
        np.savez_compressed(SUMMARY, **summarize(d["records"], d["negativity"], int(d["L"]), float(d["p"]), A, B))
        print("wrote", SUMMARY)
    ax = plot_ewg(np.load(SUMMARY))
    ax.figure.savefig("results/ewg_L18_p0.17.png", dpi=150)
    print("wrote results/ewg_L18_p0.17.png")
