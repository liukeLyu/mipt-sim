"""Plot results/sweep.json: half-chain entropy vs system size, one curve per measurement rate."""

import json
from collections import defaultdict

import matplotlib.pyplot as plt

# one-hue ordinal ramp (light = few measurements, dark = many)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

with open("results/sweep.json") as f:
    rows = json.load(f)
by_p = defaultdict(list)
for r in rows:
    by_p[r["p"]].append(r)
ps = sorted(by_p)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
Ls = sorted({r["L"] for r in rows})
a.plot(Ls, [L / 2 for L in Ls], ls=":", lw=1.5, color="#8a8a86")
a.text(Ls[-1], Ls[-1] / 2, " max L/2", va="center", color="#55554f", fontsize=9)
for p, c in zip(ps, BLUES):
    d = sorted(by_p[p], key=lambda r: r["L"])
    x = [r["L"] for r in d]
    a.errorbar(
        x, [r["mean"] for r in d], [r["sem"] for r in d], color=c, lw=2, marker="o", ms=5, capsize=0, label=f"p = {p:g}"
    )
    a.text(x[-1] + 0.3, d[-1]["mean"], f"p = {p:g}", va="center", color="#33332f", fontsize=9)
a.set(
    xlabel="system size L (qubits)",
    ylabel="half-chain entropy S (bits)",
    title="Entropy vs size: linear growth (volume law)\nbelow p_c, saturation (area law) above",
)
a.set_xlim(Ls[0] - 0.5, Ls[-1] + 3.5)
a.grid(alpha=0.25)
a.legend(frameon=False, fontsize=9, loc="upper left")

for p, c in zip(ps, BLUES):
    d = sorted(by_p[p], key=lambda r: r["L"])
    b.plot([r["L"] for r in d], [r["mean"] / r["L"] for r in d], color=c, lw=2, marker="o", ms=5, label=f"p = {p:g}")
b.set(
    xlabel="system size L (qubits)",
    ylabel="S / L",
    title="Entropy density: flat for volume law,\nfalls like 1/L for area law",
)
b.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("results/entropy_vs_L.png", dpi=150)
print("wrote results/entropy_vs_L.png")
