"""Run as python3 narrow_band_corr_max_ap.py  /path/to/data/file /path/to/ap/file dataset-name
Example : 
"""
import ast
import numpy as np
import sys
import matplotlib.pyplot as plt
import importlib
data = sys.argv[1]
ap_file = sys.argv[2]
dataset_name = sys.argv[3]

module = importlib.import_module(f"definitions.{dataset_name}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")
num_rel_non_rel = getattr(module, "num_rel_non_rel")


qid_to_ideal_ap = {}

with open(ap_file, "r") as f:
    for line in f:
        # Strip trailing newlines and ensure the line contains data
        clean_line = line.strip()
        if clean_line and '\t' in clean_line:
            _, qid, ideal_ap = clean_line.split('\t', 2)

            # Map the qid to the ideal_ap value
            qid_to_ideal_ap[qid] = float(ideal_ap)


def separation_score(lists):
    """
    lists: list of lists of numbers

    Returns:
        score = between-group variance / within-group variance
        between_var
        within_var
    """

    means = np.array([np.mean(lst) for lst in lists])

    # population variance within each list
    within_vars = np.array([
        np.var(lst, ddof=0) for lst in lists
    ])

    between_var = np.var(means, ddof=0)
    within_var = np.mean(within_vars)

    score = np.inf if within_var == 0 else between_var / within_var

    return score, between_var, within_var

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def partial_corr(a, b, c):
    """
    Correlation between a and b controlling for c.
    """
    rho_ab = corr(a, b)
    rho_ac = corr(a, c)
    rho_bc = corr(b, c)

    return (
        (rho_ab - rho_ac * rho_bc)
        / np.sqrt((1 - rho_ac**2) * (1 - rho_bc**2))
    )


avg_sim_vals = []
avg_ap_vals = []
# Added these two
max_sim_vals = []
max_ap_vals = []

numrel_vals = []

all_angles = []
delta_angles = []

all_aps = []
delta_aps = []

with open(data) as f:
    for line in f:
        qid, angles, sims, aps, numrel = line.rstrip("\n").split("\t")
        
        if qid_to_ideal_ap[qid] < 0.7:
            continue

        angles = ast.literal_eval(angles)
        sims = ast.literal_eval(sims)
        aps = ast.literal_eval(aps)

        avg_sim_vals.append(np.mean(sims))
        avg_ap_vals.append(np.mean(aps))
        # fetch the position (index) of maximum ap measure
        # get it's similarity score
        max_ap_idx = np.argmax(aps)
        # TODO: read idealap file; filter bad qids 
        max_ap_vals.append(aps[max_ap_idx])
        max_sim_vals.append(sims[max_ap_idx])

        numrel_vals.append(float(numrel))

        delta_angles.append(max(angles) - min(angles))
        all_angles.append(angles)

        delta_aps.append(max(aps) - min(aps))
        all_aps.append(aps)



avg_sim_vals = np.asarray(avg_sim_vals)
avg_ap_vals = np.asarray(avg_ap_vals)
max_sim_vals = np.asarray(max_sim_vals)     # ADDED
max_ap_vals = np.asarray(max_ap_vals)       # ADDED
numrel_vals = np.asarray(numrel_vals)

rho_ab = corr(avg_sim_vals, avg_ap_vals)
rho_ab_with_max = corr(max_sim_vals, max_ap_vals)

rho_ac = corr(avg_sim_vals, numrel_vals)
rho_bc = corr(avg_ap_vals, numrel_vals)

rho_ab_c = partial_corr(avg_sim_vals, avg_ap_vals, numrel_vals)

print(f"corr(avg_sim, avg_ap) = {rho_ab:.6f}")
print(f"corr(max_sim, max_ap) = {rho_ab_with_max:.6f}")
print(f"after filtering bad qids: = {len(max_sim_vals)}")

print(f"corr(avg_sim, numrel) = {rho_ac:.6f}")
print(f"corr(avg_ap, numrel) = {rho_bc:.6f}")
print(f"partial corr(avg_sim, avg_ap | numrel) = {rho_ab_c:.6f}")

print(f'correlation between #relDocs and #delta-angle: {corr(delta_angles, num_rel_non_rel["num_relevant"])}')
angle_separation = separation_score(all_angles)
ap_separation = separation_score(all_aps)

print("ANGLE separation:")
print("Between-group variance / within-group variance:", angle_separation[0])
print("Between-group variance:", angle_separation[1])
print("within-group variance:", angle_separation[2])
print("Within-group max-min:", np.mean(delta_angles) * 180 / np.pi)

print("AP separation:")
print("Between-group variance / within-group variance:", ap_separation[0])
print("Between-group variance:", ap_separation[1])
print("within-group variance:", ap_separation[2])
print("Within-group max-min (degrees):", np.mean(delta_aps))

plt.figure(figsize=(8, 6))
plt.scatter(avg_sim_vals, avg_ap_vals, alpha=0.7)

plt.xlabel("Average Similarity")
plt.ylabel("Average AP")
plt.title("Average Similarity vs Average AP")

# Optional: show correlation in the title
plt.suptitle(f"corr = {rho_ab:.4f}")

plt.grid(True)
plt.tight_layout()
plt.savefig(f"./qpp_{data.split('/')[-1]}.png")
