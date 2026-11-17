#%% # import libraries
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

path_metrics = (
    PROJECT_ROOT / "visualization" / "fig06" / "validation_model_metrics"
)

path_fig = PROJECT_ROOT / "visualization" / "fig06"

#%% # configurations

init_months = [2, 5, 8, 11]
lead_times = [0, 1, 2, 3]

season_labels = {
    2: "FMAM",
    5: "MJJA",
    8: "ASON",
    11: "NDJF"
}

metricas = ["RMSE"]

metric_axis_labels = {
    "RMSE": "RMSE (mm/month)",
}

#%% # Load all DataFrames containing validation metrics
# Spatio-temporal validation: evaluate model performance
# on the training and test datasets

dfs_train_val = {}
dfs_test_val = {}
dfs_train_dm = {}
dfs_test_dm = {}

for im in init_months:
    for lt in lead_times:

        # S9-ML
        dfs_train_val[(im, lt)] = pd.read_csv(
            path_metrics / f"per_season_validation_ml_im{im}_lt{lt}_train_rmse.csv"
        )

        dfs_test_val[(im, lt)] = pd.read_csv(
            path_metrics / f"per_season_validation_ml_im{im}_lt{lt}_test_rmse.csv"
        )

        # S9
        dfs_train_dm[(im, lt)] = pd.read_csv(
            path_metrics / f"per_season_validation_s9_im{im}_lt{lt}_train_rmse.csv"
        )

        dfs_test_dm[(im, lt)] = pd.read_csv(
            path_metrics / f"per_season_validation_s9_im{im}_lt{lt}_test_rmse.csv"
        )

#%% # Plot

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

met = 'RMSE'

flierprops = dict(
    marker='.',
    markersize=3,
    markerfacecolor='black',
    markeredgecolor='black',
    linestyle='none',
    alpha=0.10,
)

color_median = 'black'

# Panel colors
# S9
color_S9_1 = '#8ECAE6'
color_S9_2 = '#219EBC'

# ML
color_ML_1 = '#126782'
color_ML_2 = '#023047'

x = np.arange(len(init_months)) * 0.4

width = 0.065

offsets = [-0.125, -0.045, 0.045, 0.125]

margin = 0.06

fig, ax = plt.subplots(
    figsize=(5.2, 2),
)

# =========================
# LOWER PANEL
# =========================
plot_data = []
pos_plot = []
colors_plot = []

for i, im in enumerate(init_months):

    ml_train_series = []
    ml_test_series = []

    for lt in [1, 2, 3]:

        ml_train_values = (
            dfs_train_val[(im, lt)][met]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        ml_test_values = (
            dfs_test_val[(im, lt)][met]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        ml_train_series.append(ml_train_values.values)
        ml_test_series.append(ml_test_values.values)

    ml_train_series = np.concatenate(ml_train_series)
    ml_test_series = np.concatenate(ml_test_series)

    s9_test_series = []
    ml_test_values_dm_rest = []

    for lt in [1, 2, 3]:

        ml_train_values_dm = (
            dfs_train_dm[(im, lt)][met]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        ml_test_values_dm = (
            dfs_test_dm[(im, lt)][met]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        s9_test_series.append(ml_train_values_dm.values)
        ml_test_values_dm_rest.append(ml_test_values_dm.values)

    s9_test_series = np.concatenate(s9_test_series)
    ml_test_values_dm_rest = np.concatenate(ml_test_values_dm_rest)


    season_data = [
        s9_test_series,
        ml_test_values_dm_rest,
        ml_train_series,
        ml_test_series,
    ]



    season_pos = [x[i] + off for off in offsets]

    season_colors = [
    color_S9_1,
    color_S9_2,
    color_ML_1,
    color_ML_2,
]

    plot_data.extend(season_data)
    pos_plot.extend(season_pos)
    colors_plot.extend(season_colors)

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

medians = [
    np.nanmedian(d) if len(d) > 0 else np.nan
    for d in plot_data
]

for xpos, med in zip(pos_plot, medians):
    if np.isfinite(med):
        ax.text(
            xpos,
            -0.05,
            f"{med:.1f}",
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=7,
            color='black'
        )

ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.grid(True, linestyle='--', alpha=0.25)

# =========================
# X-AXIS / LABELS
# =========================

letters = [
    "M A M",
    "J J A",
    "S O N",
    "D J F"
]

for i, letter in enumerate(letters):
    ax.text(
        x[i],
        -0.14,
        letter,
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='top',
        fontsize=9
    )

ax.set_xticks([])

ax.set_ylabel(metric_axis_labels[met])

ax.set_xlim(
    x[0] + min(offsets) - margin,
    x[-1] + max(offsets) + margin
)

# =========================
# LEGENDS
# =========================


handles_bottom = [
    plt.Line2D([0], [0], color=color_S9_1, lw=8),
    plt.Line2D([0], [0], color=color_S9_2, lw=8),
    plt.Line2D([0], [0], color=color_ML_1, lw=8),
    plt.Line2D([0], [0], color=color_ML_2, lw=8),
]

ax.legend(
    handles_bottom,
    [
        "S9 train-set",
        "S9 test-set",
        "S9-ML train-set",
        "S9-ML test-set",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=4,
    frameon=False,
    borderpad=0.2,

    handlelength=0.3,

    handletextpad=0.8,
    columnspacing=1.8,
    labelspacing=0.2,
    fontsize=8
)


plt.tight_layout()
fig.subplots_adjust(hspace=0.55, bottom=0.12)

plt.savefig(
    path_fig / "fig06.pdf",
    bbox_inches="tight",
    dpi=900
)

plt.show()

#%%