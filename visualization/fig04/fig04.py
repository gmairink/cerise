#%% # Import libraries
import numpy as np
import pandas as pd
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

path_data = PROJECT_ROOT / "assessment"
path_fig = PROJECT_ROOT / "visualization" / "fig04"

#%% # Import files and calculate skill for deterministic metrics

# Configuration
init_months = [2, 5, 8, 11]
lead_times = [0, 1, 2, 3]

metrics = ["r", "RMSE"]

# Load DataFrames
dfs_ml_ref = {}
dfs_ml_all_season = {}

# Reference model

for im in init_months:
    for lt in lead_times:

        # ML
        file_ml = (
            path_data / "per_season" / "paper_results" /
            f"per_season_ml_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_ml_ref[(im, lt)] = pd.read_csv(file_ml)

# All-season model

for im in init_months:
    for lt in lead_times:

        # ML
        file_ml = (
            path_data / "all_season" / "paper_results" /
            f"all_season_ml_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_ml_all_season[(im, lt)] = pd.read_csv(file_ml)


# Calculate skill scores and create the final DataFrames

# dfs_skill_dm = {}
dfs_skill_ml = {}

for key in dfs_ml_ref.keys():

    im, lt = key

    df_ref = dfs_ml_ref[key]
    df_all_season = dfs_ml_all_season[key]

    # Merge datasets by station
    df_merge = pd.merge(
        df_ref,
        df_all_season,
        on="station",
        suffixes=("_ref", "_all")
    )

    df_final = pd.DataFrame()

    df_final["station"] = df_merge["station"]
    df_final["n_valid"] = df_merge["n_valid_ref"]

    for metric in metrics:

        ref = df_merge[f"{metric}_ref"]
        all_season_data = df_merge[f"{metric}_all"]

        if metric == "r":

            num = np.abs(all_season_data) - 1
            den = np.abs(ref) - 1

            skill = np.where(
                den != 0,
                1 - (num / den),
                np.nan
            )

        else: # RMSE

            num = np.abs(all_season_data)
            den = np.abs(ref)

            skill = np.where(
                den != 0,
                1 - (num / den),
                np.nan
            )

        df_final[f"{metric}_skill"] = skill

    dfs_skill_ml[key] = df_final

print(dfs_skill_ml)

#%% # Import files and calculate the skill score for the CRPS metric (S9)

# Configuration
init_months = [2, 5, 8, 11]
lead_times = [0, 1, 2, 3]

metrics = ["CRPS"]

# Load DataFrames
dfs_ml_ref = {}
dfs_ml_all_season = {}

# Reference model

for im in init_months:
    for lt in lead_times:

        # ML
        file_ml = (
            path_data / "per_season" / "paper_results" /
            f"per_season_ml_im{im}_lt{lt}_crps.csv"
        )

        dfs_ml_ref[(im, lt)] = pd.read_csv(file_ml)

# All-season model

for im in init_months:
    for lt in lead_times:

        # ML
        file_ml = (
            path_data / "all_season" / "paper_results" /
            f"all_season_ml_im{im}_lt{lt}_crps.csv"
        )
        
        dfs_ml_all_season[(im, lt)] = pd.read_csv(file_ml)


# Calculate skill scores and create the final DataFrames

# dfs_skill_dm = {}
dfs_skill_CRPS_ml = {}

for key in dfs_ml_ref.keys():

    im, lt = key

    df_ref = dfs_ml_ref[key]
    df_all_season = dfs_ml_all_season[key]

    # Merge datasets by station
    df_merge = pd.merge(
        df_ref,
        df_all_season,
        on="station",
        suffixes=("_ref", "_all")
    )

    df_final = pd.DataFrame()

    df_final["station"] = df_merge["station"]
    df_final["n_valid"] = df_merge["n_valid_ref"]

    for metric in metrics:

        ref = df_merge[f"{metric}_ref"]
        all_season_data = df_merge[f"{metric}_all"]

        num = np.abs(all_season_data)
        den = np.abs(ref)

        skill = np.where(
            den != 0,
            1 - (num / den),
            np.nan
        )

        df_final[f"{metric}_skill"] = skill

    dfs_skill_CRPS_ml[key] = df_final

print(dfs_skill_CRPS_ml)

#%% # Exclude lead time 0; concatenate lead times 1, 2, and 3

init_months = [2, 5, 8, 11]
lead_times_keep = [1, 2, 3]   # Exclude lt=0

season_labels = {
    2: "MAM",
    5: "JJA",
    8: "SON",
    11: "DJF"
}

skill_metrics = ["r_skill", "RMSE_skill", "CRPS_skill"]

# Complete DataFrame
rows = []

dfs_skill_all = {}
for key in dfs_skill_ml.keys():

    df_det = dfs_skill_ml[key]

    df_crps = dfs_skill_CRPS_ml[key][
        ["station", "CRPS_skill"]
    ]

    df_merge = pd.merge(
        df_det,
        df_crps,
        on="station",
        how="left"
    )

    dfs_skill_all[key] = df_merge

for (im, lt), df in dfs_skill_all.items():


    if lt not in lead_times_keep:
        continue

    for met in skill_metrics:

        if met not in df.columns:
            continue

        tmp = df[["station", "n_valid"]].copy()

        tmp["init_month"] = im
        tmp["season"] = season_labels[im]
        tmp["lead_time"] = lt

        tmp["metric"] = met.replace("_skill", "")
        tmp["skill"] = df[met].values

        rows.append(tmp)

df_skill_long_s9 = pd.concat(rows, ignore_index=True)

#%% # Plot skill scores

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 3.7
})

metrics = ["r", "RMSE", "CRPS"]

season_order = ["MAM", "JJA", "SON", "DJF"]

colors = {
    "S9-ML": "#08519C"
}

season_labels = {
    '2': "M A M",
    '5': "J J A",
    '8': "S O N",
    '11': "D J F"
}

fig, axes = plt.subplots(
    len(metrics),
    1,
    figsize=(1.5, 2.),
    sharex=True
)

# Group positions
x = np.arange(len(season_order)) * 0.04

width = 0.02

# X-axis limits
margin = 0.02

flierprops = dict(
    marker='.',
    markersize=0.4,
    markerfacecolor='black',
    markeredgecolor='black',
    linestyle='none',
    alpha=0.10,
)

for ax, met in zip(axes, metrics):

    plot_data = []
    pos_plot = []
    colors_plot = []

    for i, season in enumerate(season_order):

        sub = df_skill_long_s9[
            (df_skill_long_s9["metric"] == met) &
            (df_skill_long_s9["season"] == season) &
            (df_skill_long_s9["lead_time"].isin([1, 2, 3]))
        ]

        vals = (
            sub["skill"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        plot_data.append(vals)
        pos_plot.append(x[i])
        colors_plot.append(colors["S9-ML"])

    bp = ax.boxplot(
        plot_data,
        positions=pos_plot,
        widths=width,
        patch_artist=True,
        flierprops=flierprops
    )


    # Boxes
    for box in bp["boxes"]:
        box.set_linewidth(0.5)

    # Whiskers
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.5)

    # Caps
    for cap in bp["caps"]:
        cap.set_linewidth(0.5)

    # Medians
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(0.4)


    # Colors
    for box, c in zip(bp["boxes"], colors_plot):
        box.set_facecolor(c)


    # Zero reference line
    ax.axhline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.4,
        alpha=0.4
    )

    # Axis limits
    if met == "r":
        ax.set_ylim(-1.22, 1)
        ax.set_yticks([-1, 0, 1])

    elif met == "RMSE":
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, 0, 1])

    elif met == "CRPS":
        ax.set_ylim(-1.1, 1)
        ax.set_yticks([-1, 0, 1])


    ax.grid(True, linestyle='--', alpha=0.25)

    # Display median values
    medians = [
        np.nanmedian(d) if len(d) > 0 else np.nan
        for d in plot_data
    ]

    for xpos, med in zip(pos_plot, medians):

        if np.isfinite(med):

            fmt = ".2f" if met == "r" else ".2f"

            ax.text(
                xpos,
                -0.08,
                format(med, fmt),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=3.7,
                color='black'
            )

    ylabel_dict = {
    "r": "Correlation skill",
    "RMSE": "RMSE skill",
    "CRPS": "CRPS skill"
}

    ax.set_ylabel(ylabel_dict[met], labelpad=1)


season_labels = {
    "MAM": "M A M",
    "JJA": "J J A",
    "SON": "S O N",
    "DJF": "D J F",
}

axes[-1].set_xticks(x)
axes[-1].set_xticklabels([season_labels[s] for s in season_order])
axes[-1].tick_params(axis='x', pad=6)   # Increase the distance from the axis


for ax in axes:

    ax.set_xlim(
        x[0] - margin,
        x[-1] + margin
    )

# Legend
for ax in axes:

    # Plot frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    # Tick settings
    ax.tick_params(
        axis='both',
        which='both',
        width=0.3,      # Line width
        length=2,       # Tick length
        direction='out'
    )


plt.tight_layout(rect=[0, 0.06, 1, 1])

fig.subplots_adjust(hspace=0.3)

plt.savefig(
    path_fig / "fig04.pdf",
    dpi=900,
    bbox_inches="tight"
)

plt.show()

#%%