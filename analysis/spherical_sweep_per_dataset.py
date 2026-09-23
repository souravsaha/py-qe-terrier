import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import importlib
import argparse
import matplotlib.ticker as ticker

from iqg_learn import IdealQueryGeneration
from io_utils import parse_queries


parser = argparse.ArgumentParser(description="Run different IQG methods with options")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--idealq_name", required=False, default=None, help="Existing ideal query name")
args = parser.parse_args()

sns.set_theme(style="whitegrid")

module = importlib.import_module(f"definitions.{args.dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")

def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tensor(x, dev):
    return torch.tensor(x, device=dev, dtype=torch.float32)


def cosine(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos


def average_precision(scores, labels):
    scores = scores.view(-1)
    labels = labels.view(-1)

    order = torch.argsort(scores, descending=True)
    labels = labels[order]

    cum_rel = torch.cumsum(labels, dim=0)
    ranks = torch.arange(1, len(labels) + 1, device=labels.device)
    precision = cum_rel / ranks

    if labels.sum() == 0:
        return torch.tensor(0.0, device=labels.device)

    return precision[labels == 1].mean()


def discriminative_power(X, y, w):
    proj = X @ w
    neg = proj[y == 0]
    pos = proj[y == 1]

    if len(neg) < 2 or len(pos) < 2:
        return torch.tensor(0.0, device=X.device)

    mean_diff = pos.mean() - neg.mean()
    pooled_var = 0.5 * (pos.var(unbiased=True) + neg.var(unbiased=True))

    return mean_diff / torch.sqrt(pooled_var + 1e-12)


def geodesic_theta(w0, direction):
    w0 = w0 / (w0.norm() + 1e-12)
    direction = direction / (direction.norm() + 1e-12)

    d = direction - torch.dot(direction, w0) * w0
    if d.norm() < 1e-12:
        return None

    d = d / d.norm()

    cos_t = torch.dot(direction, w0)
    sin_t = torch.dot(direction, d)

    theta = torch.atan2(sin_t, cos_t)
    return theta + 2 * np.pi if theta < 0 else theta


def spherical_sweep(X, y, w_start, directions, points=500):
    thetas = torch.linspace(0, 2 * np.pi, points, device=w_start.device)
    w0 = w_start / (w_start.norm() + 1e-12)

    D_all = []
    AP_all = []
    SIM_all = []

    for direction in tqdm(directions):
        d = direction - torch.dot(direction, w0) * w0
        if d.norm() < 1e-12:
            zeros = torch.zeros(points, device=w0.device)
            D_all.append(zeros)
            AP_all.append(zeros)
            SIM_all.append(zeros)
            continue

        d = d / d.norm()

        D_vals = []
        AP_vals = []
        SIM_vals = []

        for theta in thetas:
            w = torch.cos(theta) * w0 + torch.sin(theta) * d
            w = w / (w.norm() + 1e-12)

            D_vals.append(discriminative_power(X, y, w))
            AP_vals.append(average_precision(X @ w, y))
            SIM_vals.append(torch.dot(w, d))

        D_all.append(torch.stack(D_vals))
        AP_all.append(torch.stack(AP_vals))
        SIM_all.append(torch.stack(SIM_vals))

    return D_all, AP_all, SIM_all, thetas

if __name__ == "__main__":
    dev = device()
    collection = args.dataset
    ideal_name = args.idealq_name
    outdir = f"./spherical_sweep/{collection}/{ideal_name}"
    os.makedirs(outdir, exist_ok=True)

    expansion_methods = ["rm3", "bo1", "kl", "ceqe", "hyde", "hyderocchio"]
    num_exp_terms = [15, 25, 35, 45, 55]
    num_top_docs = {
            "rm3":  [10, 20, 30, 40],
            "bo1":  [10, 20, 30, 40],
            "kl":   [10, 20, 30, 40],
            "ceqe": [10, 20, 30, 40],
    }
    method_params = {"rm3": "0.6", "ceqe": "max"}

    expanded_query_path = f"./expanded-queries/{collection}/weights"

    iqg = IdealQueryGeneration(index, qrels_bin, collection)
    restricted_ap = {}
    theta_ap = {}

    qe_theta = defaultdict(list)
    qe_ap = defaultdict(list)
    for qid in qid_range:
        #if qid != "1063750":
        # if qid != "336":
        #     continue
        X_np, y_np, term_list = iqg.get_data(qid)

        X = tensor(X_np, dev)
        y = tensor(y_np, dev)

        ideal_query_path = f"./ideal_queries/{collection}/{ideal_name}/{qid}"
        ideal = iqg.qvec_to_array(parse_queries(ideal_query_path).get(qid), term_list)

        w_start = tensor(ideal, dev)

        directions_np = []
        direction_methods = []

        ap_by_method = defaultdict(list)
        sim_by_method = defaultdict(list)

        for method, color in zip(
            expansion_methods, sns.color_palette("tab10", len(expansion_methods))
        ):
            for nd, nt in product(num_top_docs.get(method, ["na"]), num_exp_terms):

                run_id = f"{method}-{nt}-{nd}-{method_params.get(method, 'na')}"
                path = os.path.join(expanded_query_path, run_id + ".term_weights")

                if not os.path.exists(path):
                    continue

                query = parse_queries(path).get(qid)
                if query is None:
                    continue

                vec = iqg.qvec_to_array(query, term_list)
                ap = average_precision(X @ tensor(vec, dev), y).item()

                sim = cosine(ideal, vec)

                ap_by_method[method].append(ap)
                sim_by_method[method].append(sim)

                directions_np.append(vec)
                direction_methods.append(method)

        if not directions_np:
            raise RuntimeError("No expansion directions found")

        directions = [tensor(d / (np.linalg.norm(d) + 1e-12), dev) for d in directions_np]

        D_all, AP_all, SIM_all, thetas = spherical_sweep(X, y, w_start, directions)

        colors = {
            "rm3": "tab:blue",
            "bo1": "tab:orange",
            "kl": "tab:green",
            "ceqe": "tab:red",
            "hyde": "tab:purple",
            "hyderocchio": "tab:gray"
        }

        plt.figure(figsize=(10, 6))
        theta_vals = thetas.cpu().numpy()

        for i, method in enumerate(direction_methods):
            ap_vals = AP_all[i].cpu().numpy()
            color = colors.get(method, "gray")

            degree_vals = np.degrees(theta_vals)
            if qid not in restricted_ap:
                restricted_ap[qid] = {}

            if qid not in theta_ap:
                theta_ap[qid] = {}

            restricted_ap[qid][method] = ap_vals
            theta_ap[qid][method] = degree_vals


            # plt.plot(np.degrees(degree_vals), ap_vals, alpha=0.45, color=color, label=method)

            theta_star = geodesic_theta(w_start, directions[i])
            if theta_star is None:
                continue

            # t = theta_star.item()
            # ap_star = np.interp(t, theta_vals, ap_vals)
            # plt.scatter(t, ap_star, color=color, s=35, zorder=4)
            t = np.degrees(theta_star.item())          # convert to degrees
            ap_star = np.interp(theta_star.item(), theta_vals, ap_vals)

            qe_theta[method].append(t)
            qe_ap[method].append(ap_star)

            # keep the query-wise plot if you want
            plt.scatter(t, ap_star, color=color, s=35, zorder=4)

    # plt.xlabel(r"$\theta$ (degrees)")
    # plt.ylabel("Restricted AP")
    # plt.title(r"Restricted AP along great circles from $Q_\text{IEQ}^\text{LS}$ $(\lambda=0.1)$")
    # plt.tight_layout()
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # #plt.savefig(f'{outdir}/{qid}.png')
    # plt.savefig(f"{outdir}/{qid}.pdf", format="pdf", bbox_inches="tight")

    # plt.figure(figsize=(8, 6))

        all_ap = []
        all_sim = []

        for method, color in colors.items():
            aps = np.array(ap_by_method.get(method, []))
            sims = np.array(sim_by_method.get(method, []))

            if len(aps) == 0:
                continue

            plt.scatter(aps, sims, color=color, alpha=0.75, s=45, label=method)
            all_ap.extend(aps)
            all_sim.extend(sims)

        plt.xlabel("average precision")
        plt.ylabel("cosine similarity to ideal query")
        plt.title("average precision vs similarity to ideal query")
        plt.legend()
        plt.tight_layout()
        plt.show()

        all_ap = np.array(all_ap)
        all_sim = np.array(all_sim)

        pearson_corr, _ = pearsonr(all_sim, all_ap)
        spearman_corr, _ = spearmanr(all_sim, all_ap)
        kendall_corr, _ = kendalltau(all_sim, all_ap)

        print("Overall correlations")
        print("Pearson:", pearson_corr)
        print("Spearman:", spearman_corr)
        print("Kendall:", kendall_corr)

        # with open(f"{outdir}/query_wise_correlation-l2.csv","a") as f:
        #     print(f"{qid}\t{pearson_corr}\t{kendall_corr}\t{spearman_corr}", file=f)

    # Average AP and theta across all qids for each method
    avg_ap = {}
    avg_theta = {}
    plt.figure(figsize=(7, 5))

    for method in direction_methods:
        all_ap = []
        all_theta = []

        for qid in restricted_ap:
            if method in restricted_ap[qid]:
                all_ap.append(np.array(restricted_ap[qid][method]))
                all_theta.append(np.array(theta_ap[qid][method]))

        # shape: (#qids, #angles)
        all_ap = np.vstack(all_ap)
        all_theta = np.vstack(all_theta)

        avg_ap[method] = np.mean(all_ap, axis=0)
        avg_theta[method] = np.mean(all_theta, axis=0)
        
        

    for i, method in enumerate(direction_methods):
        color = colors.get(method, "gray")

        plt.plot(
            avg_theta[method],
            avg_ap[method],
            label=method,
            linewidth=2,
            markersize=5,
            color=color
        )
        """
        if len(qe_theta[method]) > 0:
            plt.scatter(
                np.mean(qe_theta[method]),
                np.mean(qe_ap[method]),
                color=color,
                edgecolors="black",
                s=35,
                zorder=4,
            )
        """
    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel("Restricted AP")
    # plt.title(r"Restricted AP along great circles from $Q_\text{IEQ}^\text{LS}$ $(\lambda=0.1)$")
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
        #plt.savefig(f'{outdir}/{qid}.png')
    plt.savefig(f"{outdir}/{args.idealq_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
        # print("\nPer method correlations")

        # for method in ap_by_method:
        #     aps = np.array(ap_by_method[method])
        #     sims = np.array(sim_by_method[method])
        #
        #     if len(aps) < 3:
        #         continue
        #
        #     p, _ = pearsonr(sims, aps)
        #     s, _ = spearmanr(sims, aps)
        #     knd, _ = kendalltau(sims, aps)
        #
        #     print(f"\n{method}")
        #     print("Pearson:", p)
        #     print("Spearman:", s)
        #     print("Kendall:", knd)
