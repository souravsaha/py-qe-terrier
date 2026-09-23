import pandas as pd
import pyterrier as pt
import os
from pathlib import Path

collection = "msmarco_passage"
iqg_name = "leastsq_idealq_heavyreg"
threshold = 0.8

# Filter out qids which don't have good correlation in QREL AP and REAL AP
qrel_ap_vs_ap_corr_file = (
    f"./correlation-computations/{collection}/ap-vs-ap/query-wise-correlation.csv"
)
qrel_ap_df = pd.read_csv(qrel_ap_vs_ap_corr_file, header=None)
qrel_ap_df.columns = ["qid", "pearson", "kendall", "spearman"]
print("Number of qids originally: ", len(qrel_ap_df.qid))

print("Filtering out qids where restricted_AP and real_AP correlation is <", threshold)
qid_range = list(qrel_ap_df[qrel_ap_df.pearson >= threshold].qid.unique())
print("Remaining number of qids:", len(qid_range))

initial_corr_file = Path(
    f"./correlation-computations/{collection}/{iqg_name}/query-wise-correlation-l2.csv"
)
print(f"Reading file {initial_corr_file}")
final_corr_file = initial_corr_file.with_name(
    f"{initial_corr_file.stem}-filtered{initial_corr_file.suffix}"
)
print(f"Writing to file {final_corr_file}")

corr_df = pd.read_csv(initial_corr_file)
corr_df.columns = ["qid", "pearson", "kendall", "spearman"]
corr_df = corr_df[corr_df.qid.isin(qid_range)]
corr_df.to_csv(final_corr_file, index=False, header=False)
