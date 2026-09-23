import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_list_column_distribution(
    input_file,
    column="angles",
    output_pdf=None,
    max_qids=200,
    figsize=(16, 24),
    jitter_std=0.03,
    point_size=25,
):
    """
    Plot a list-valued column for each qid.

    Parameters
    ----------
    input_file : str
        TSV file with no header.

    column : str
        Name of the column to visualize.
        Examples: "angles", "sims", "aps"

    output_pdf : str or None
        Output PDF filename.
        Defaults to '<column>_distribution.pdf'.

    max_qids : int
        Maximum number of qids shown.

    jitter_std : float
        Standard deviation of vertical jitter.

    point_size : float
        Scatter marker size.
    """

    if output_pdf is None:
        output_pdf = f"{column}_distribution.pdf"

    # ------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------
    df = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["qid", "angles", "sims", "aps", "num_rel"],
    )

    # Parse all list-valued columns
    for col in df.columns[:-1]:
        if col != "qid":
            df[col] = df[col].apply(ast.literal_eval)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------
    # Convert angles from radians to degrees
    # ------------------------------------------------------------
    if column == "angles":
        df["angles"] = df["angles"].apply(
            lambda x: np.degrees(x)
        )

    # ------------------------------------------------------------
    # Subsample qids uniformly
    # ------------------------------------------------------------
    n_qids = len(df)

    if n_qids > max_qids:
        idx = np.linspace(
            0,
            n_qids - 1,
            max_qids,
            dtype=int,
        )
        df = df.iloc[idx].reset_index(drop=True)

    # ------------------------------------------------------------
    # Global range
    # ------------------------------------------------------------
    values = df[column]

    global_min = min(min(v) for v in values)
    global_max = max(max(v) for v in values)

    padding = 0.02 * (global_max - global_min)

    xmin = global_min - padding
    xmax = global_max + padding

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(42)

    for y, vals in enumerate(values):

        # Horizontal reference line
        ax.hlines(
            y,
            xmin,
            xmax,
            color="black",
            linewidth=0.35,
            alpha=0.15,
            zorder=1,
        )

        # Vertical jitter
        yvals = y + rng.normal(
            0,
            jitter_std,
            len(vals),
        )

        ax.scatter(
            vals,
            yvals,
            s=point_size,
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )

    # ------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, len(df))

    if column == "angles":
        ax.set_xlabel(r"$\theta$ (degrees)", fontsize=16)
    else:
        ax.set_xlabel(column, fontsize=16)

    ax.set_ylabel("QID index", fontsize=16)

    # No title

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Y-axis ticks
    # ------------------------------------------------------------
    n = len(df)

    if n <= 20:
        yticks = np.arange(n)
    else:
        yticks = np.linspace(
            0,
            n - 1,
            min(20, n),
            dtype=int,
        )

    ax.set_yticks(yticks)

    plt.tight_layout()

    plt.savefig(
        output_pdf,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure to: {output_pdf}")


# ------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------

plot_list_column_distribution(
    sys.argv[1],
    column="angles",
)

plot_list_column_distribution(
    sys.argv[1],
    column="sims",
)

plot_list_column_distribution(
    sys.argv[1],
    column="aps",
)
