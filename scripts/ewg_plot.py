"""Plot entanglement-weighted graphs from results/ewg_L18_p0.17.npz.

Colour: white at the bare measurement rate p, blue below, red above (as in arXiv:2602.04969, Fig. 6).
Time runs upward, so the final state (where A and B are evaluated) is the top row.
Usage: python scripts/ewg_plot.py [in.npz] [out.png]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from mipt_sim.ewg import weighted_graph

src = sys.argv[1] if len(sys.argv) > 1 else "results/ewg_L18_p0.17.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "results/ewg_L18_p0.17.png"
d = np.load(src)
rec, p, L = d["records"], float(d["p"]), int(d["L"])
A, B = list(d["A"]), list(d["B"])
neg, mi = d["negativity"], d["mutual_information"]
n_neg = int((neg > 1e-10).sum())

panels = [
    (f"negativity-weighted\n({n_neg} of {len(neg)} samples contribute)", weighted_graph(rec, neg)),
    ("mutual-information-weighted", weighted_graph(rec, mi)),
    ("unweighted (should be p everywhere)", rec.mean(0)),
]
norm = TwoSlopeNorm(vcenter=p, vmin=0.0, vmax=3 * p)  # values above 3p saturate

# roll the periodic chain so that A and B sit symmetrically about the middle of the plot
shift = (L - 1 - A[0] - B[-1]) // 2
cols = (np.arange(L) - shift) % L  # site shown in each column (0-based)

fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), constrained_layout=True)
for ax, (title, g) in zip(axes, panels):
    im = ax.imshow(
        np.roll(g, shift, axis=1),
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        extent=(0.5, L + 0.5, 0.5, len(g) + 0.5),
    )
    ax.set_title(title, fontsize=11, pad=22)
    ax.set_xlabel("site")
    ax.set_xticks(range(1, L + 1, 2), [str(cols[i] + 1) for i in range(0, L, 2)])
    for q, lab in [(A, "A"), (B, "B")]:
        q = [(x + shift) % L for x in q]
        ax.annotate(
            lab,
            ((q[0] + q[-1]) / 2 + 1, len(g) + 0.5),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            annotation_clip=False,
        )
        for x in q:
            ax.plot([x + 1], [len(g) + 0.5], marker="v", color="#33332f", ms=7, clip_on=False)
    ax.set_xlim(0.5, L + 0.5)
axes[0].set_ylabel("layer (time runs up)")
cb = fig.colorbar(im, ax=axes, shrink=0.85)
cb.set_label(f"weighted measurement rate (white = p = {p:g})")
fig.suptitle(
    f"Entanglement-weighted graphs, Haar brickwork, L = {L}, p = {p:g}, A = sites {A[0] + 1},{A[1] + 1}, B = sites {B[0] + 1},{B[1] + 1}",
    fontsize=12,
)
fig.savefig(out, dpi=150)
print("wrote", out)
