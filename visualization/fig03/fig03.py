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

path_climatology = PROJECT_ROOT / "visualization" / "fig03" / "climatology"
path_assessment = PROJECT_ROOT / "assessment" / "per_season" / "paper_results"
path_fig = PROJECT_ROOT / "visualization" / "fig03"

#%% ##### LOAD DATA #####

init_months = ["2", "5", "8", "11"]
lead_times = [1, 2, 3]


skill_dm = {}
skill_ml = {}

skill_dm = {}
skill_ml = {}

for init_month in init_months:

    for lt in lead_times:

        # Climatology
        df_clim = pd.read_csv(
            path_climatology /
            f"per_season_climatology_im{init_month}_lt{lt}.csv"
        )

        # Dynamic model (S9)
        df_dm = pd.read_csv(
            path_assessment /
            f"per_season_s9_im{init_month}_lt{lt}_deterministic_metrics.csv"
        )

        df_dm = df_dm.merge(
            df_clim[["station", "ubRMSE_clim"]],
            on="station",
            how="inner"
        )

        df_dm["ubRMSE_difference"] = (
            df_dm["ubRMSE_clim"] - df_dm["ub RMSE"]
        )

        # Machine-learning model (S9-ML)
        df_ml = pd.read_csv(
            path_assessment /
            f"per_season_ml_im{init_month}_lt{lt}_deterministic_metrics.csv"
        )

        df_ml = df_ml.merge(
            df_clim[["station", "ubRMSE_clim"]],
            on="station",
            how="inner"
        )

        df_ml["ubRMSE_difference"] = (
            df_ml["ubRMSE_clim"] - df_ml["ub RMSE"]
        )

        skill_dm[(int(init_month), lt)] = df_dm.copy()
        skill_ml[(int(init_month), lt)] = df_ml.copy()


#%% # RMSSS plot

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

color_S9 = '#4A90C2'
color_S9_ml = '#08519C'

color_median = 'black'


season_labels = {
    '2': "M A M",
    '5': "J J A",
    '8': "S O N",
    '11': "D J F"
}


fig, ax = plt.subplots(
    figsize=(3.5, 2.8)
)

plot_data = []
pos_plot = []
colors_plot = []

x = np.arange(len(init_months)) * 0.25

width = 0.07

offsets = [-0.045, 0.045]

margin = 0.05

models = [

    ("S9",
     skill_dm,
     color_S9),

    ("S9-ML",
     skill_ml,
     color_S9_ml),
]

for i, im in enumerate(init_months):

    for j, (name, dfs, color) in enumerate(models):

        rmsss = []

        for lt in [1, 2, 3]:

            metric_values = dfs[(int(im), lt)]["ubRMSE_difference"]

            metric_values = (
                metric_values
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            rmsss.append(
                metric_values.values
            )

        rmsss = np.concatenate(
            rmsss
        )

        plot_data.append(rmsss)

        pos_plot.append(
            x[i] + offsets[j]
        )

        colors_plot.append(
            color
        )

flierprops = dict(
    marker='.',
    markersize=3,
    markerfacecolor='black',
    markeredgecolor='black',
    linestyle='none',
    alpha=0.10,
)

bp = ax.boxplot(
    plot_data,
    positions=pos_plot,
    widths=width,
    patch_artist=True,
    flierprops=flierprops
)

for box, c in zip(
    bp["boxes"],
    colors_plot
):
    box.set_facecolor(c)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(0.7)

# --------------------------------------------------
# Reference line: skill = 0
# --------------------------------------------------

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.7,
    alpha=0.4
)

# --------------------------------------------------
# Median labels
# --------------------------------------------------

medians = [
    np.nanmedian(d)
    for d in plot_data
]

for xpos, med in zip(
    pos_plot,
    medians
):

    ax.text(
        xpos,
        -0.01,
        f"{med:.2f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7
    )

# --------------------------------------------------
# Y-axis
# --------------------------------------------------

ax.set_ylabel("∆ubRMSE (mm/month)")


ax.grid(
    True,
    linestyle="--",
    alpha=0.25
)

# Adjust after inspecting the data

ax.set_ylim(-8, 6.5)
ax.set_yticks([-7, -5, -3, -1, 1, 3, 5])

# --------------------------------------------------
# Seasons
# --------------------------------------------------

ax.set_xticks([])

for i, im in enumerate(init_months):

    ax.text(
        x[i],
        -0.07,
        season_labels[im],
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9
    )

# --------------------------------------------------
# Axis limits
# --------------------------------------------------

ax.set_xlim(
    x[0] + min(offsets) - margin,
    x[-1] + max(offsets) + margin
)

# --------------------------------------------------
# Legend
# --------------------------------------------------

legend_handles = [
    plt.Line2D([0], [0], color=color_S9, lw=8),
    plt.Line2D([0], [0], color=color_S9_ml, lw=8),
]

fig.legend(
    legend_handles,
    [
        "S9",
        "S9-ML",
    ],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.575, -0.05),
    frameon=False,
    borderpad=0.6,
    handlelength=0.3,
    handletextpad=0.8,
    columnspacing=1.2,
    fontsize=8

)

plt.tight_layout()


plt.savefig(
    path_fig / "fig03.pdf",
    bbox_inches="tight",
    dpi=900
)

plt.show()

#%%