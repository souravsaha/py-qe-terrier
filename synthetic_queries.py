import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
import copy

from iqg_learn import IdealQueryGeneration
from io_utils import parse_queries

# from definitions.trec678rb import index, qrels_bin, qid_range
from definitions.msmarco_passage_v2 import index, qrels_bin, qid_range
def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tensor(x, dev):
    return torch.tensor(x, device=dev, dtype=torch.float32)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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


def spherical_sweep(w_start, directions, iqg, qid, term_list, rotate_upto=None, num_points=100, num_trim=200):
    if rotate_upto is None:
        rotate_upto = np.pi / 2
    thetas = torch.linspace(0, rotate_upto, num_points, device=w_start.device)
    w0 = w_start / (w_start.norm() + 1e-12)

    AP_all = []
    SIM_all = []

    for direction in directions:
        d = direction - torch.dot(direction, w0) * w0
        if d.norm() < 1e-12:
            zeros = torch.zeros(num_points, device=w0.device)
            AP_all.append(zeros)
            SIM_all.append(zeros)
            continue

        d = d / d.norm()

        AP_vals = []
        SIM_vals = []

        for theta in tqdm(thetas):
            w = torch.cos(theta) * w0 + torch.sin(theta) * d
            w = w / (w.norm() + 1e-12)

            SIM_vals.append(torch.dot(w, w0))

            query = iqg.array_to_qvec(w, term_list)
            query.remove_non_positive_weights()
            query.sort_by_weight()
            query.trim(num_trim)
            AP_vals.append(tensor(iqg.computeAP(qid, query), dev=w.device))

        AP_all.append(torch.stack(AP_vals))
        SIM_all.append(torch.stack(SIM_vals))

    return AP_all, SIM_all, thetas


if __name__ == "__main__":
    dev = device()
    # collection = "trec678rb"
    collection = "msmarcov2_passage"
    ideal_name = "leastsq_1"
    
    rotate_upto = np.pi / 2
    num_points = 100
    num_trim = 200

    base_dir = f"./corr-synthetic/{collection}/{ideal_name}_{num_trim}"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/correlation_scatter/", exist_ok=True)
    os.makedirs(f"{base_dir}/ap_vs_theta/", exist_ok=True)

    iqg = IdealQueryGeneration(index, qrels_bin, collection)

    for qid in qid_range:
        # if int(qid) <= 410:
        #     continue
        X, y, term_list = iqg.get_data(qid)

        ideal_query_path = f"./ideal_queries/{collection}/{ideal_name}/{qid}"
        ideal = iqg.qvec_to_array(parse_queries(ideal_query_path).get(qid), term_list)

        w_start = tensor(ideal, dev)

        np.random.seed(42) 
        random_dir_np = np.random.randn(len(term_list))
        random_dir = tensor(random_dir_np, dev)
        random_dir = random_dir / (random_dir.norm() + 1e-12)

        directions = [random_dir]

        AP_all, SIM_all, thetas = spherical_sweep(w_start, directions, iqg, qid, term_list, rotate_upto=rotate_upto, num_points=num_points, num_trim=num_trim)

        if not AP_all:
            raise RuntimeError("No directions found")

        ap_vals = AP_all[0].cpu().numpy()
        sim_vals = SIM_all[0].cpu().numpy()
        theta_vals = thetas.cpu().numpy()

        # Overall correlations along the random vector rotation
        pearson_corr, _ = pearsonr(sim_vals, ap_vals)
        spearman_corr, _ = spearmanr(sim_vals, ap_vals)
        kendall_corr, _ = kendalltau(sim_vals, ap_vals)

        #print("Overall correlations for the random direction")
        #print("Pearson:", pearson_corr)
        #print("Spearman:", spearman_corr)
        #print("Kendall:", kendall_corr)

        with open(f"{base_dir}/correlations.csv", 'a') as corr_file:
            print(f"{qid},{pearson_corr},{spearman_corr},{kendall_corr}", file=corr_file)

        # Plot AP vs Theta
        plt.figure(figsize=(10, 6))
        plt.plot(theta_vals, ap_vals, color="tab:blue")
        plt.xlabel("theta")
        plt.ylabel("average precision")
        plt.title("Average Precision along a random geodesic direction")
        plt.tight_layout()
        plt.savefig(f'{base_dir}/ap_vs_theta/{qid}.png')

        # Plot AP vs Cosine Similarity (w to ideal query)
        plt.figure(figsize=(8, 6))
        plt.scatter(ap_vals, sim_vals, color="tab:blue", alpha=0.75, s=45)
        plt.xlabel("average precision")
        plt.ylabel("cosine similarity to ideal query")
        plt.title("Average Precision vs Similarity to Ideal Query (Random Direction)")
        plt.tight_layout()
        plt.savefig(f'{base_dir}/correlation_scatter/{qid}.png')
