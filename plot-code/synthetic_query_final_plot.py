import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau

from iqg_learn import IdealQueryGeneration
from io_utils import parse_queries

from definitions.trec678rb import index, qrels_bin, qid_range


def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tensor(x, dev):
    return torch.tensor(x, device=dev, dtype=torch.float32)


def spherical_sweep(
    w_start,
    directions,
    iqg,
    qid,
    term_list,
    rotate_upto=None,
    num_points=100,
    num_trim=200,
):
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
    collection = "trec678rb"
    ideal_name = "leastsq_0.1"
    rotate_upto = np.pi / 2
    num_points = 100
    num_trim = 200
    num_directions = 30  # 30 random directions

    base_dir = f"./corr-synthetic/{collection}/{ideal_name}_{num_trim}"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/correlation_scatter/", exist_ok=True)
    os.makedirs(f"{base_dir}/ap_vs_theta/", exist_ok=True)

    iqg = IdealQueryGeneration(index, qrels_bin, collection)

    for qid in qid_range:
        if int(qid) != 336:
            continue
        X, y, term_list = iqg.get_data(qid)

        ideal_query_path = f"./ideal_queries/{collection}/{ideal_name}/{qid}"
        ideal = iqg.qvec_to_array(parse_queries(ideal_query_path).get(qid), term_list)

        w_start = tensor(ideal, dev)

        np.random.seed(42)
        
        # Generate 30 random directions
        directions = []
        for _ in range(num_directions):
            random_dir_np = np.random.randn(len(term_list))
            random_dir = tensor(random_dir_np, dev)
            random_dir = random_dir / (random_dir.norm() + 1e-12)
            directions.append(random_dir)

        AP_all, SIM_all, thetas = spherical_sweep(
            w_start,
            directions,
            iqg,
            qid,
            term_list,
            rotate_upto=rotate_upto,
            num_points=num_points,
            num_trim=num_trim,
        )

        if not AP_all:
            raise RuntimeError("No directions found")

        # Convert list of tensors to a 2D numpy array: Shape (30, num_points)
        ap_vals_matrix = torch.stack(AP_all).cpu().numpy() 
        theta_vals = thetas.cpu().numpy()
        theta_vals = np.degrees(theta_vals)

        # Calculate mean and standard deviation
        ap_mean = np.mean(ap_vals_matrix, axis=0)
        ap_std = np.std(ap_vals_matrix, axis=0)

        # Plot
        plt.figure(figsize=(10, 6))
        
        # Mean trend
        plt.plot(theta_vals, ap_mean, color="tab:blue", linewidth=2.5, label="Mean AP")
        
        # Variation band
        plt.fill_between(
            theta_vals, 
            np.clip(ap_mean - ap_std, 0, 1), 
            np.clip(ap_mean + ap_std, 0, 1), 
            color="tab:blue", 
            alpha=0.3, 
            label="±1 Std Dev"
        )

        plt.xlabel(r"$\theta$ (degrees)", fontsize=15)
        plt.ylabel("Average Precision", fontsize=15)
        plt.tick_params(axis="both", labelsize=16)
        # plt.title(f"AP Variation along {num_directions} Random Geodesic Directions (QID: {qid})", fontsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        mid_x = (theta_vals.min() + theta_vals.max()) / 2
        arrow_top = 0.95
        text_y = 0.65

        plt.annotate(
            "Ideal Queries",
            xy=(mid_x, arrow_top),
            xytext=(mid_x, text_y),
            fontsize=16,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5
            ),

            arrowprops=dict(
                arrowstyle="->",
                linewidth=3
            )
        )
        # SAVED AS PDF HERE
        plt.savefig(f"{base_dir}/ap_vs_theta/{qid}.pdf", format="pdf", bbox_inches="tight")
        # np.savez(
        #     f"{base_dir}/ap_vs_theta/{qid}.npz",
        #     theta_vals=theta_vals,
        #     ap_vals_matrix=ap_vals_matrix
        # )
        
        plt.close()
        
        np.savez(
            f"{base_dir}/ap_vs_theta/{qid}_{num_directions}.npz",
            theta_vals=theta_vals,
            ap_vals_matrix=ap_vals_matrix
        )
