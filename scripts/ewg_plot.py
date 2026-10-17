"""Plot the negativity-weighted graph from results/ewg_L18_p0.17.npz.

Colour: white at the bare measurement rate p, blue below, red above (as in arXiv:2602.04969, Fig. 6).
Time runs upward, so the final state (where A and B are evaluated) is the top row. Only the last
`layers` time steps are shown: earlier ones sit at p within noise.
Usage: python scripts/ewg_plot.py [in.npz] [out.png] [layers]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from mipt_sim.ewg import weighted_graph


def plot_ewg(d, layers=20, ax=None):
    rec, p, L = d["records"], float(d["p"]), int(d["L"])
    A, B = list(d["A"]), list(d["B"])
    g = weighted_graph(rec, d["negativity"])[-layers:]
    t0 = len(rec[0]) - layers  # time steps shown: t0 + 1 ... depth
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
    n_neg = int((d["negativity"] > 1e-10).sum())
    ax.set_title(
        f"Negativity-weighted graph, L = {L}, p = {p:g}\n({n_neg} of {len(rec)} trajectories entangle A and B)",
        pad=22,
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label=f"weighted measurement rate (white = p = {p:g})")
    return ax


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/ewg_L18_p0.17.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/ewg_L18_p0.17.png"
    layers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ax = plot_ewg(np.load(src), layers)
    ax.figure.savefig(out, dpi=150)
    print("wrote", out)
