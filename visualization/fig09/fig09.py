#%% # import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%% # Automatically identify the project root directory

current_dir = Path(__file__).resolve().parent

PROJECT_ROOT = next(
    (
        parent for parent in [current_dir, *current_dir.parents]
        if all(
            (parent / folder).is_dir()
            for folder in ["raw_data", "model", "assessment", "visualization"]
        )
    ),
    None
)

if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root directory not found.")

path_data = PROJECT_ROOT / "visualization" / "fig09"
path_fig = path_data

init_months = [2, 5, 8, 11]

lead_times = [1, 2, 3]

season_labels = {
    2: "MAM",
    5: "JJA",
    8: "SON",
    11: "DJF"
}

metrics = ["r", "RMSE", "CRPS"]

metric_axis_labels = {
    "r": "Correlation",
    "RMSE": "RMSE (mm/month)",
    "CRPS": "CRPS (mm/month)",
}

# Load dataframes

dfs_season = {
    "dm": {},
    "ml": {},
    "qm": {},
    "glm": {}
}

dfs_crps = {
    "dm": {},
    "ml": {},
    "qm": {},
    "glm": {}
}

#%% # Load deterministic and probabilistic metrics for each initialization month and lead time

for im in init_months:

    for lt in lead_times:

        # Deterministic metrics
        dfs_season["dm"][(im, lt)] = pd.read_csv(
            path_data / "s9_ml_all_data_metrics" /
            f"s9_all_data_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_season["ml"][(im, lt)] = pd.read_csv(
            path_data / "s9_ml_all_data_metrics" /
            f"ml_all_data_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_season["qm"][(im, lt)] = pd.read_csv(
            path_data / "qm_glm_all_data_metrics" /
            f"qm_all_data_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_season["glm"][(im, lt)] = pd.read_csv(
            path_data / "qm_glm_all_data_metrics" /
            f"glm_all_data_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        # CRPS
        dfs_crps["dm"][(im, lt)] = pd.read_csv(
            path_data / "s9_ml_all_data_metrics" /
            f"s9_all_data_im{im}_lt{lt}_crps.csv"
        )

        dfs_crps["ml"][(im, lt)] = pd.read_csv(
            path_data / "s9_ml_all_data_metrics" /
            f"ml_all_data_im{im}_lt{lt}_crps.csv"
        )

        dfs_crps["qm"][(im, lt)] = pd.read_csv(
            path_data / "qm_glm_all_data_metrics" /
            f"qm_all_data_im{im}_lt{lt}_crps.csv"
        )

        dfs_crps["glm"][(im, lt)] = pd.read_csv(
            path_data / "qm_glm_all_data_metrics" /
            f"glm_all_data_im{im}_lt{lt}_crps.csv"
        )


#%% # filter outliers
# RMSE
for key, df in dfs_season["glm"].items():

    dfs_season["glm"][key] = (
        df[df["RMSE"] <= 150]
        .copy()
    )

# CRPS
for key, df in dfs_crps["glm"].items():

    dfs_crps["glm"][key] = (
        df[df["CRPS"] <= 150]
        .copy()
    )

#%% # configuration

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

# Plotting order
zorders = {
    "ml": 4,
    "dm": 3,
    "qm": 2,
    "glm": 1
}


color_dm = "#4A90C2"
color_ml = "#08519C"

color_qm  = "#B22222"
color_glm = "#B8860B"

model_info = [
    ("dm",  color_dm,  "S9"),
    ("ml",  color_ml,  "S9-ML"),
    ("qm",  color_qm,  "S9-QM"),
    ("glm", color_glm, "S9-GLM")
]


# ==========================================================
# FIGURE - PER SEASON MODEL
# Metrics: r, RMSE, CRPS
# ==========================================================

from scipy.interpolate import PchipInterpolator


def plot_ecdf(ax, values, color, label, zorder, logx=False):

    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if logx:
        values = values[values > 0]

    if len(values) < 2:
        return

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.4,
        label=label,
        zorder=zorder
    )


# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=len(init_months),
    figsize=(7.5, 6.0),
    sharey=True
)


# ----------------------------------------------------------
# Axis configuration
# ----------------------------------------------------------

xlims = {
    "r": (-0.5, 1.0),
    "RMSE": (0.5, 150),
    "CRPS": (0.2, 150)
}

xticks = {
    "r": [-0.5, 0, 0.5, 1],
    "RMSE": [1, 10, 100],
    "CRPS": [1, 10, 100]
}

for row, metric in enumerate(metrics):

    for col in range(len(init_months)):

        ax = axes[row, col]

        ax.set_xlim(xlims[metric])
        ax.set_ylim(0, 1)

        if metric in ["RMSE", "CRPS"]:

            ax.set_xscale("log")
            ax.set_xticks(xticks[metric])
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())

        else:

            ax.set_xticks(xticks[metric])

        ax.set_xlabel(metric_axis_labels[metric])

        ax.grid(
            True,
            linestyle="--",
            alpha=0.35,
            which="both"
        )

        ax.tick_params(
            axis="both",
            labelsize=8
        )

        if col != 0:

            ax.tick_params(
                axis="y",
                labelleft=False
            )

        if metric == "r":

            ax.set_xticklabels(
                ['-0.5', '0', '0.5', '1'],
                fontsize=8
            )


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

for row, metric in enumerate(metrics):

    for col, im in enumerate(init_months):

        ax = axes[row, col]

        for model_key, color, label in model_info:

            values = []

            for lt in lead_times:

                if metric == "CRPS":

                    values.extend(
                        dfs_crps[model_key][(im, lt)]["CRPS"]
                        .dropna()
                        .values
                    )

                else:

                    values.extend(
                        dfs_season[model_key][(im, lt)][metric]
                        .dropna()
                        .values
                    )

            plot_ecdf(
                ax,
                values,
                color,
                label,
                zorder=zorders[model_key],
                logx=(metric in ["RMSE", "CRPS"])
            )

        if row == 0:
            ax.set_title(season_labels[im])


# ----------------------------------------------------------
# Y labels
# ----------------------------------------------------------

for row, metric in enumerate(metrics):

    axes[row, 0].set_ylabel(
        f"ECDF"
    )

# ----------------------------------------------------------
# Y ticks
# ----------------------------------------------------------

for row in range(len(metrics)):

    axes[row, 0].set_yticks(
        [0, 0.25, 0.50, 0.75, 1.00]
    )

# ----------------------------------------------------------
# Legend
# ----------------------------------------------------------

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=4,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.0
)


# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------

plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.93,
    bottom=0.12,
    wspace=0.12,
    hspace=0.40
)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.savefig(
    path_fig / "fig09.pdf",
    dpi=900,
    bbox_inches="tight"
)
plt.show()

#%%
