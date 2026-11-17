#%% # Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

path_data_assessment = PROJECT_ROOT / "assessment" / "per_season" / "paper_results"
path_fig = PROJECT_ROOT / "visualization" / "fig02"

#%% # Define initialization months, lead times and metrics

init_months = [2, 5, 8, 11]
lead_times = [1, 2, 3]

season_labels = {
    2: "M A M",
    5: "J J A",
    8: "S O N",
    11: "D J F"
}

metrics = ["r", "RMSE", "CRPS"]


metric_axis_labels = {
    "r": "Correlation",
    "RMSE": "RMSE (mm/month)",
    "CRPS": "CRPS (mm/month)",
}

#%% # Load all DataFrames

dfs_dm_s9 = {}
dfs_ml_s9 = {}

dfs_crps_dm_s9 = {}
dfs_crps_ml_s9 = {}


for im in init_months:
    for lt in lead_times:

        # S9
        dfs_dm_s9[(im, lt)] = pd.read_csv(
            path_data_assessment / f"per_season_s9_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        # S9-ML
        dfs_ml_s9[(im, lt)] = pd.read_csv(
            path_data_assessment / f"per_season_ml_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        # S9 CRPS
        dfs_crps_dm_s9[(im, lt)] = pd.read_csv(
            path_data_assessment / f"per_season_s9_im{im}_lt{lt}_crps.csv"
        )

        # S9-ML CRPS
        dfs_crps_ml_s9[(im, lt)] = pd.read_csv(
            path_data_assessment / f"per_season_ml_im{im}_lt{lt}_crps.csv"
        )

#%% # Plot

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

color_S9 = '#4A90C2'
color_S9_ML = '#08519C'
color_median = 'black'


fig, axes = plt.subplots(len(metrics), 1, figsize=(3.5, 4.4), sharex=True)

flierprops = dict(
    marker='.',
    markersize=1.5,
    markerfacecolor='black',
    markeredgecolor='black',
    linestyle='none',
    alpha=0.10,
)

# Spacing between seasons
x = np.arange(len(init_months)) * 0.25

width = 0.07

offsets = [-0.045, 0.045]

margin = 0.05

for ax_idx, (ax, met) in enumerate(zip(axes, metrics)):

    plot_data = []
    pos_plot = []
    colors_plot = []
    combined_plot_values = []
    perc_signif_all = []

    if met == 'CRPS':

        models = [
            ("S9", dfs_crps_dm_s9, color_S9),
            ("S9-ML", dfs_crps_ml_s9, color_S9_ML),
        ]

    else:

        models = [
            ("S9", dfs_dm_s9, color_S9),
            ("S9-ML", dfs_ml_s9, color_S9_ML),
        ]


    for i, im in enumerate(init_months):

        for j, (name, dfs, color) in enumerate(models):

            lt123 = []

            for lt in [1, 2, 3]:
                metric_values = dfs[(im, lt)][met]
                metric_values = metric_values.replace([np.inf, -np.inf], np.nan).dropna()
                lt123.append(metric_values.values)

            lt123 = np.concatenate(lt123)

            if "pval" in dfs[(im, 1)].columns:

                p123 = []

                for lt in [1, 2, 3]:
                    p = dfs[(im, lt)]["pval"]
                    p = p.replace([np.inf, -np.inf], np.nan).dropna()
                    p123.append(p.values)

                p123 = np.concatenate(p123)
                perc123 = 100 * np.sum(p123 < 0.05) / len(p123)

            else:
                perc123 = np.nan

            xpos = x[i] + offsets[j]

            plot_data.append(lt123)
            pos_plot.append(xpos)
            colors_plot.append(color)
            perc_signif_all.append(perc123)
            combined_plot_values.append(lt123)

    bp = ax.boxplot(
        plot_data,
        positions=pos_plot,
        widths=width,
        patch_artist=True,
        flierprops=flierprops
    )

    for box, c in zip(bp["boxes"], colors_plot):
        box.set_facecolor(c)

    for median in bp["medians"]:
        median.set_color(color_median)
        median.set_linewidth(0.7)

    # Determine axis limits
    if len(combined_plot_values) > 0:
        combined_plot_values = np.concatenate(combined_plot_values)
        combined_plot_values = combined_plot_values[np.isfinite(combined_plot_values)]
        lim_min = combined_plot_values.min()
        lim_max = combined_plot_values.max()
    else:
        lim_min, lim_max = -1, 1

    if met == "r":
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.3)
        ax.set_ylim(-0.6, 1)
        ax.set_yticks([0, 1])

    elif met == "RMSE":

        ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])

    elif met == "CRPS":
        ax.set_ylim(0, 60)

    ax.set_ylabel(metric_axis_labels[met])
    ax.grid(True, linestyle='--', alpha=0.25)

    # Calculate median values
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in plot_data]

    for i_box, (xpos, med) in enumerate(zip(pos_plot, medians)):
        if np.isfinite(med):

            fmt = ".2f" if met == "r" else ".1f"

            ax.text(
                xpos,
                -0.05,
                format(med, fmt),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=7,
                color='black'
            )

    # Add percentage labels at the top (first subplot only)
    if ax_idx == 0:
        for xpos, perc, c in zip(pos_plot, perc_signif_all, colors_plot):
            if np.isfinite(perc):
                ax.text(
                    xpos,
                    1.02,
                    f"{int(round(perc))}",
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='black'  # Same color as the box for better readability
                )

fig.text(
    0.09,   # x-coordinate
    1.005,  # y-coordinate
    "% of stations\n with pval < 0.05",
    ha='center',
    va='top',
    fontsize=7,
    color='black',
)

# Remove default tick labels
axes[-1].set_xticks([])

letters = [
    ["M A M"],
    ["J J A"],
    ["S O N"],
    ["D J F"]
]

for i, season in enumerate(letters):
    xpos = [x[i]]

    for xp, letter in zip(xpos, season):
        axes[-1].text(
            xp,
            -0.16,
            letter,
            transform=axes[-1].get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=9
        )

for ax in axes:

    ax.set_xlim(
        x[0] + min(offsets) - margin,
        x[-1] + max(offsets) + margin
    )

    # ax.tick_params(axis='y', labelsize=7)

# Legend
legend_handles = [
    plt.Line2D([0], [0], color=color_S9, lw=8),
    plt.Line2D([0], [0], color=color_S9_ML, lw=8),
]

fig.legend(
    legend_handles,
    ["S9", "S9-ML"],
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.575, -0.03),
    frameon=False,
    handlelength=0.3,
    handletextpad=0.8,
    columnspacing=1.2,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    path_fig / "fig02.pdf",
    bbox_inches="tight",
    dpi=900
)
plt.show()

#%%