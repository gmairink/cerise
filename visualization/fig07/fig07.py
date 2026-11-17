#%% # import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

path_assessment = PROJECT_ROOT / "assessment" / "per_season" / "paper_results"
path_metadata = PROJECT_ROOT / "visualization" / "fig01" / "stations_train_test_metadata_fig01.csv"
path_fig = PROJECT_ROOT / "visualization" / "fig07"

init_months = [2, 5, 8, 11]
lead_times = [1, 2, 3]

season_labels = {
    2: "FMAM",
    5: "MJJA",
    8: "ASON",
    11: "NDJF"
}

#%% # load all dataframes

dfs_dm = {}
dfs_ml = {}

for im in init_months:
    for lt in lead_times:

        dfs_dm[(im, lt)] = pd.read_csv(
            path_assessment /
            f"per_season_s9_im{im}_lt{lt}_deterministic_metrics.csv"
        )

        dfs_ml[(im, lt)] = pd.read_csv(
            path_assessment /
            f"per_season_ml_im{im}_lt{lt}_deterministic_metrics.csv"
        )

#%% # Load station drainage area information

df_stations = pd.read_csv(path_metadata)

# Select test stations and retain their drainage area
df_area = (
    df_stations.loc[
        df_stations["data_type"] == "test",
        ["station_id", "area"]
    ]
    .rename(columns={"station_id": "station"})
    .copy()
)

# Check station uniqueness
if df_area["station"].duplicated().any():
    raise ValueError("Duplicate station IDs found in the metadata.")

print(f"Stations loaded: {len(df_area)}")
print(f"Missing drainage areas: {df_area['area'].isna().sum()}")


for key in dfs_dm:
    dfs_dm[key] = dfs_dm[key].merge(
        df_area,
        on="station",
        how="left"
    )

for key in dfs_ml:
    dfs_ml[key] = dfs_ml[key].merge(
        df_area,
        on="station",
        how="left"
    )

#%% # calculate RMSE skill between the machine learning and dynamic models, grouped by drainage area

dfs_skill = {}

for key in dfs_ml:

    df_skill = dfs_ml[key].copy()

    rmse_dm = dfs_dm[key]["ub RMSE"].values
    rmse_ml = dfs_ml[key]["ub RMSE"].values

    df_skill["RMSE_skill"] = 1 - (rmse_ml / rmse_dm)

    dfs_skill[key] = df_skill

# ------------------------------------------------------------
# create drainage area classes
# ------------------------------------------------------------

def classify_area(area):

    if 1e3 <= area < 1e4:
        return r"$10^3$-$10^4$"

    elif 1e4 <= area < 1e5:
        return r"$10^4$-$10^5$"

    elif 1e5 <= area < 1e6:
        return r"$10^5$-$10^6$"

    elif 1e6 <= area < 1e7:
        return r"$10^6$-$10^7$"

    else:
        return np.nan

# ------------------------------------------------------------
# create consolidated dataframe
# ------------------------------------------------------------

all_data = []

for im in init_months:

    df_im = []

    # combine lead times 1–3
    for lt in [1, 2, 3]:

        df_tmp = dfs_skill[(im, lt)].copy()

        df_im.append(df_tmp)

    df_im = pd.concat(df_im, ignore_index=True)

    # assign drainage area classes
    df_im["area_class"] = df_im["area"].apply(classify_area)

    area_order = [
    r"$10^3$-$10^4$",
    r"$10^4$-$10^5$",
    r"$10^5$-$10^6$",
    r"$10^6$-$10^7$"
    ]

    # assign season
    df_im["Season"] = season_labels[im]

    all_data.append(df_im)

df_plot = pd.concat(all_data, ignore_index=True)

# remove NaN values
df_plot = df_plot.dropna(subset=["area_class", "RMSE_skill"])

# print the number of basins per class
print(
    df_plot
    .groupby("area_class")["station"]
    .nunique()
)

#%% # plot all seasons

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7
})

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(5, 3.5),
    sharey=True
)

custom_titles = {
    2: "M A M",
    5: "J J A",
    8: "S O N",
    11: "D J F"
}

axes = axes.flatten()

for ax, im in zip(axes, init_months):

    season = season_labels[im]

    df_season = df_plot[
        df_plot["Season"] == season
    ]

    sns.violinplot(
        data=df_season,
        x="area_class",
        y="RMSE_skill",
        order=area_order,
        ax=ax,
        inner=None,
        cut=0,
        linewidth=0.8,
        color='#08519C'
    )

    # calculate the median
    medians = (
        df_season
        .groupby("area_class")["RMSE_skill"]
        .median()
    )

    for i, area_class in enumerate(medians.index):

        ax.scatter(
            i,
            medians[area_class],
            color="white",
            s=12,
            zorder=3,
            linewidth=0.4
        )

    ax.set_title(season)

    ax.tick_params(axis='x', labelsize=6)

    ax.set_xlabel("")

    ax.set_ylim(-0.5, 1)

    ax.set_yticks([-0.5, 0, 0.5, 1])

    ax.set_yticklabels([
        "-0.5",
        "0",
        "0.5",
        "1"
    ])

    ax.set_title(
        custom_titles[im],
        x=0.5,
        y=1.08,
    )

    # add grid line
    ax.axhline(linestyle='--', linewidth=0.8, color='white', alpha=0.5)

    # add axis labels
    ax.set_xlabel("Drainage area (km²)", labelpad=2)


    n_basins = (
        df_season
        .groupby("area_class")["station"]
        .nunique()
        .reindex(area_order)
    )

    for i, area_class in enumerate(area_order):

        ax.text(
            i,
            1.15,   # height on the y-axis
            f"n={n_basins[area_class]}",
            ha='center',
            va='top',
            fontsize=6
        )



# set y-axis labels
axes[0].set_ylabel("RMSE skill", labelpad=0)
axes[2].set_ylabel("RMSE skill", labelpad=0)

axes[1].set_ylabel("")
axes[3].set_ylabel("")


# adjust figure layout
plt.subplots_adjust(
    hspace=0.7,
    wspace=0.10,
    bottom=0.16
)

# save figure
plt.savefig(
    path_fig / "fig07.pdf",
    dpi=900,
    bbox_inches="tight"
)

plt.show()

# %%