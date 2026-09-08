import matplotlib.pyplot as plt
import numpy as np

# Data
lambdas = [0.1, 1, 5, 10, 20, 50, 100]

robust = [0.8757, 0.8186, 0.8123, 0.7750, 0.7415, 0.7007, 0.6728]

dl_passage = [0.9601, 0.9485, 0.9106, 0.8817, 0.8525, 0.8178, 0.7977]

dl_document = [0.8241, 0.8558, 0.8767, 0.8540, 0.8097, 0.7260, 0.6498]

dl_passage_v2 = [0.7081, 0.7279, 0.6802, 0.6460, 0.6148, 0.5812, 0.5628]

plt.style.use('seaborn-v0_8-whitegrid')

# Figure style
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # "legend.fontsize": 25,
    "legend.fontsize": 15,
    "font.family": "serif",
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42
})

# fig, ax = plt.subplots(figsize=(8, 5))

fig, ax = plt.subplots(figsize=(8, 6.5))
# Curves
ax.plot(
    lambdas, robust,
    marker='o', linewidth=3,
    markersize=8,
    label='Robust'
)

ax.plot(
    lambdas, dl_passage,
    marker='s', linestyle='--',
    linewidth=3, markersize=8,
    label='DL19–20 Passage'
)

ax.plot(
    lambdas, dl_document,
    marker='^', linestyle='-.',
    linewidth=3, markersize=8,
    label='DL19–20 Document'
)

ax.plot(
    lambdas,
    dl_passage_v2,
    marker='D',          # diamond marker
    linestyle=':',
    linewidth=3,
    markersize=8,
    label='DL21–22 Passage'
)

# Log scale
# ax.set_xscale('log')
ax.set_xscale('linear')

# Labels
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('MAP')

# Grid
ax.grid(True, linestyle='--', alpha=0.3)

# Legend
#ax.legend(frameon=True)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False
    # columnspacing=1.5,
    # handlelength=2.5
)


# Optional: remove top/right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save as vector graphics
plt.savefig("map_vs_lambda.pdf", bbox_inches="tight")
plt.savefig("map_vs_lambda.svg", bbox_inches="tight")
plt.savefig("map_vs_lambda.png", dpi=400, bbox_inches="tight")

plt.show()
