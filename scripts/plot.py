"""Plot results/subregion_L{L}.json: entropy of l consecutive qubits versus l, one curve per measurement rate.

Usage: python scripts/plot.py [L]
"""

import json
import sys

import matplotlib.pyplot as plt

# one-hue ordinal ramp (light = few measurements, dark = many)
BLUES = ["#9cc3f2", "#6aa5ea", "#3f86dd", "#2368c0", "#174f94", "#0d366b"]

L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
with open(f"results/subregion_L{L}.json") as f:
    d = json.load(f)
ell = d["ell"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
ax.plot(ell, ell, ls=":", lw=1.5, color="#8a8a86")
ax.text(ell[-1], ell[-1], "  maximum, S = ℓ", va="center", color="#55554f", fontsize=9)
for p, m, e, c in zip(d["p"], d["mean"], d["sem"], BLUES):
    ax.errorbar(ell, m, e, color=c, lw=2, marker="o", ms=5, capsize=0)
    ax.text(ell[-1] + 0.25, m[-1], f"p = {p:g}", va="center", color="#33332f", fontsize=9)
ax.set(
    xlabel="subregion size ℓ (consecutive qubits)",
    ylabel="entanglement entropy S (bits)",
    title=f"Entropy of a subregion, L = {L}",
    xlim=(0.5, ell[-1] + 1.6),
)
ax.set_xticks(ell)
fig.savefig(f"results/entropy_vs_ell_L{L}.png", dpi=150)
print(f"wrote results/entropy_vs_ell_L{L}.png")
