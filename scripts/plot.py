"""Plot results/subregion_L{L}.json: entropy of l consecutive qubits versus l, one curve per measurement rate.

Usage: python scripts/plot.py [L]
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def page(l, n):  # average entropy (bits) of l qubits in a uniformly random state of n qubits
    m, k = 2**l, 2 ** (n - l)
    return (sum(1 / j for j in range(k + 1, m * k + 1)) - (m - 1) / (2 * k)) / np.log(2)


L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
with open(f"results/subregion_L{L}.json") as f:
    d = json.load(f)
ell = d["ell"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
ax.plot(ell, [page(l, L) for l in ell], ls="--", color="#33332f", zorder=3, label="random state (Page)")
colors = plt.cm.Blues(np.linspace(0.35, 1, len(d["p"])))  # light = few measurements, dark = many
for p, m, e, color in zip(d["p"], d["mean"], d["sem"], colors):
    ax.errorbar(ell, m, e, color=color, lw=2, marker="o", ms=5, capsize=0, label=f"p = {p:g}")
ax.set(
    xlabel="block size ℓ (consecutive qubits)",
    ylabel="entanglement entropy S (bits)",
    title=f"Entropy of a block, L = {L}",
)
ax.set_xticks(ell)
ax.legend(frameon=False, fontsize=9)
fig.savefig(f"results/entropy_vs_ell_L{L}.png", dpi=150)
print(f"wrote results/entropy_vs_ell_L{L}.png")
