import ast
import numpy as np
import sys
import matplotlib.pyplot as plt

data = sys.argv[1]

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
numrel_vals = []

all_angles = []
delta_angles = []

all_aps = []
delta_aps = []

with open(data) as f:
    for line in f:
        qid, angles, sims, aps, numrel = line.rstrip("\n").split("\t")

        angles = ast.literal_eval(angles)
        sims = ast.literal_eval(sims)
        aps = ast.literal_eval(aps)

        avg_sim_vals.append(np.mean(sims))
        avg_ap_vals.append(np.mean(aps))
        numrel_vals.append(float(numrel))

        delta_angles.append(max(angles) - min(angles))
        all_angles.append(angles)

        delta_aps.append(max(aps) - min(aps))
        all_aps.append(aps)



avg_sim_vals = np.asarray(avg_sim_vals)
avg_ap_vals = np.asarray(avg_ap_vals)
numrel_vals = np.asarray(numrel_vals)

rho_ab = corr(avg_sim_vals, avg_ap_vals)
rho_ac = corr(avg_sim_vals, numrel_vals)
rho_bc = corr(avg_ap_vals, numrel_vals)

rho_ab_c = partial_corr(avg_sim_vals, avg_ap_vals, numrel_vals)

print(f"corr(avg_sim, avg_ap) = {rho_ab:.6f}")
print(f"corr(avg_sim, numrel) = {rho_ac:.6f}")
print(f"corr(avg_ap, numrel) = {rho_bc:.6f}")
print(f"partial corr(avg_sim, avg_ap | numrel) = {rho_ab_c:.6f}")

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
