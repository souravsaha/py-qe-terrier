import matplotlib.pyplot as plt
import numpy as np

# Data
lambdas = [0.1, 1, 5, 10, 20, 50, 100]

robust = [99.3494, 131.5502, 164.7189, 175.8514, 179.5984, 168.5823, 157.6787]
dl_passage = [115.4113, 133.5258, 145.1856, 147.7216, 146.7938,142.1753, 138.8144]
dl_document = [89.3295, 100.9886, 124.2614, 137.9318, 150.1705, 156.3977, 148.3523]
dl_passage_v2 = [130.3333, 153.6744, 166.1085, 167.2248, 164.4341, 157.8837, 153.5194]


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

fig, ax = plt.subplots(figsize=(8, 7.5))
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
ax.set_ylabel('Avg. overlap')

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
plt.savefig("avg_overlap_vs_lambda.pdf", bbox_inches="tight")
plt.savefig("avg_overlap_vs_lambda.svg", bbox_inches="tight")
plt.savefig("avg_overlap_vs_lambda.png", dpi=400, bbox_inches="tight")

plt.show()