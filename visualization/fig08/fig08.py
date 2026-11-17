#%% # Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap

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

path_metadata = (
    PROJECT_ROOT / "visualization" / "fig01"
    / "stations_train_test_metadata_fig01.csv"
)
path_metrics = PROJECT_ROOT / "visualization" / "fig08" / "lt123_metrics"
path_fig = PROJECT_ROOT / "visualization" / "fig08"

init_months = ['2', '5', '8', '11']

#%%  # Station metadata: coordinates and watershed information

df_stations = pd.read_csv(path_metadata)

# Select test stations, since the assessment metrics are based on X_test
df_r2d2 = (
    df_stations.loc[
        df_stations["data_type"] == "test",
        ["station_id", "lat", "lon", "area", "station_name"]
    ]
    .rename(columns={
        "station_id": "station",
        "station_name": "name"
    })
    .copy()
)

# Check station uniqueness
if df_r2d2["station"].duplicated().any():
    raise ValueError("Duplicate station IDs found in the metadata.")

print(f"Stations loaded: {len(df_r2d2)}")
print(f"Missing latitude: {df_r2d2['lat'].isna().sum()}")
print(f"Missing longitude: {df_r2d2['lon'].isna().sum()}")


#%%

# ======================================================
# Seasons
# ======================================================

season_dict = {
    "2": "MAM",
    "5": "JJA",
    "8": "SON",
    "11": "DJF"
}

# ======================================================
# Final dictionaries
# ======================================================

dfs_metrics_dm = {}
dfs_metrics_ml = {}

# ======================================================
# Loop over seasons
# ======================================================

for init_month in init_months:

    season = season_dict[init_month]

    # ----------------------------
    # Load files
    # ----------------------------

    file_dm = (
        path_metrics / f"s9_im{init_month}_lt123_metrics.csv"
    )

    file_ml = (
        path_metrics / f"ml_im{init_month}_lt123_metrics.csv"
    )

    df_dm = pd.read_csv(file_dm)
    df_ml = pd.read_csv(file_ml)

    # ----------------------------
    # Add latitude and longitude
    # ----------------------------

    df_dm = pd.merge(
        df_dm,
        df_r2d2,
        on="station",
        how="left"
    )

    df_ml = pd.merge(
        df_ml,
        df_r2d2,
        on="station",
        how="left"
    )

    # ----------------------------
    # Store results in dictionaries
    # ----------------------------

    dfs_metrics_dm[season] = df_dm

    dfs_metrics_ml[season] = df_ml

#%% Calculate skill: DM vs ML (reference = ML)

dfs_skill_r = {}

for season in season_dict.values():

    df_dm = dfs_metrics_dm[season].copy()
    df_ml = dfs_metrics_ml[season].copy()

    # ======================================================
    # Merge by station
    # ======================================================

    df_merge = pd.merge(

        df_dm[
            ["station", "r", "pval"]
        ],

        df_ml[
            ["station", "r", "pval", "lat", "lon"]
        ],

        on="station",
        suffixes=("_dm", "_ml")
    )

    # ======================================================
    # Calculate skill
    # ======================================================

    num = df_merge["r_ml"] - 1

    den = df_merge["r_dm"] - 1

    skill = np.where(
        den != 0,
        1 - (num / den),
        np.nan
    )


    # ======================================================
    # Build the final DataFrame
    # ======================================================

    df_skill = pd.DataFrame({

        "station": df_merge["station"],

        "lat": df_merge["lat"],

        "lon": df_merge["lon"],

        "r_dm": df_merge["r_dm"],

        "r_ml": df_merge["r_ml"],

        "pval_dm": df_merge["pval_dm"],

        "pval_ml": df_merge["pval_ml"],

        "skill_r": skill
    })

    # Store results
    dfs_skill_r[season] = df_skill


#%% # Plot

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

season_order = ["MAM", "JJA", "SON", "DJF"]

def fmt_lon(x):
    return f"{abs(int(x))}° W" if x < 0 else f"{int(x)}° E" if x > 0 else "0°"

def fmt_lat(y):
    return f"{abs(int(y))}° S" if y < 0 else f"{int(y)}° N" if y > 0 else "0°"


def style_map(ax, extent=[-180, 180, -56, 75]):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.1, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.1, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.05, zorder=1)

    gl = ax.gridlines(
    draw_labels=False,
    linewidth=0.1,
    linestyle="--",
    alpha=0.6
    )

    xticks = [-180, -90, 0, 90, 180]
    yticks = [-45, 0, 45]

    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.set_xticklabels(
    [fmt_lon(x) for x in xticks]
    )

    ax.set_yticklabels(
        [fmt_lat(y) for y in yticks]
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.4,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        labelsize=7
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


fig = plt.figure(figsize=(9, 8))
gs = fig.add_gridspec(
    4, 2,
    width_ratios=[1, 1],
    wspace=0.10,
)

axes = np.empty((4, 2), dtype=object)

for i in range(4):
    axes[i, 0] = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
    axes[i, 1] = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())

# Correlation

colors_corr = [
    "#FA9857",
    "#ADDC6F",
    "#3BA858",
    "#006836"
]

cmap_r = ListedColormap(colors_corr)

# Color for values < 0
cmap_r.set_under("#AA0526")

bounds_r = [0, 0.25, 0.5, 0.75, 1]

norm_r = BoundaryNorm(
    bounds_r,
    cmap_r.N
)

# Skill
colors_skill = [
    "#EA5A39",
    "#FA9857",
    "#ADDC6F",
    "#3BA858",
]

cmap_skill = ListedColormap(colors_skill)

# Color for values < 1
cmap_skill.set_under("#AA0526")
# cmap_skill.set_over("#006836")


bounds_skill = [-1, -0.5, 0, 0.5, 1]

norm_skill = BoundaryNorm(
    bounds_skill,
    cmap_skill.N
)


for i, season in enumerate(season_order):

    df_dm = dfs_metrics_dm[season]
    df_skill = dfs_skill_r[season]

    # Plot correlation and CDF
    ax = axes[i, 0]
    style_map(ax)

    if i != 3: # Remove bottom labels
        ax.set_xticklabels([])

    # ======================================================
    # Correlation significance
    # ======================================================

    sig_dm = df_dm["pval"] < 0.05

    # Not statistically significant
    ax.scatter(
        df_dm.loc[~sig_dm, "lon"],
        df_dm.loc[~sig_dm, "lat"],

        c=df_dm.loc[~sig_dm, "r"],

        cmap=cmap_r,
        norm=norm_r,

        s=2,

        alpha=0.18,

        marker="o",

        edgecolors="none",

        transform=ccrs.PlateCarree(),

        zorder=2
    )

    # Statistically significant
    sc1 = ax.scatter(
        df_dm.loc[sig_dm, "lon"],
        df_dm.loc[sig_dm, "lat"],

        c=df_dm.loc[sig_dm, "r"],

        cmap=cmap_r,
        norm=norm_r,

        s=2,

        alpha=1.0,

        marker="o",

        edgecolors="none",

        transform=ccrs.PlateCarree(),

        zorder=3
    )


    # Plot correlation skill
    ax = axes[i, 1]
    style_map(ax)
    ax.set_yticklabels([])

    if i != 3: # Remove bottom labels
        ax.set_xticklabels([])

    # ======================================================
    # Skill significance
    # Significant in BOTH:
    # DM and ML
    # ======================================================

    sig_both = (
        # (df_skill["pval_dm"] < 0.05)
        # &
        (df_skill["pval_ml"] < 0.05)
    )

    # Not statistically significant
    ax.scatter(
        df_skill.loc[~sig_both, "lon"],
        df_skill.loc[~sig_both, "lat"],

        c=df_skill.loc[~sig_both, "skill_r"],

        cmap=cmap_skill,
        norm=norm_skill,

        s=2,

        alpha=0.2,

        marker="o",

        edgecolors="none",

        transform=ccrs.PlateCarree(),

        zorder=2
    )

    # Statistically significant
    sc2 = ax.scatter(
        df_skill.loc[sig_both, "lon"],
        df_skill.loc[sig_both, "lat"],

        c=df_skill.loc[sig_both, "skill_r"],

        cmap=cmap_skill,
        norm=norm_skill,

        s=2,

        alpha=1.0,

        marker="o",

        edgecolors="none",

        transform=ccrs.PlateCarree(),

        zorder=3
    )


# ======================================================
# Season labels
# ======================================================

for i, season in enumerate(season_order):

    axes[i, 0].text(
        -0.11,   # Further to the left
        0.5,

        season,

        transform=axes[i, 0].transAxes,

        rotation=90,

        ha='center',
        va='center')

# ======================================================
# Horizontal color bars
# ======================================================
# Correlation
cbar1 = fig.colorbar(
    sc1,
    ax=axes[:, 0],
    orientation='horizontal',
    fraction=0.025,
    aspect=30,
    pad=0.06,

    boundaries=bounds_r,

    ticks=[0, 0.25, 0.5, 0.75, 1],

    extend='min',
    extendfrac=0.15
)


cbar1.set_label("S9 Correlation")


cbar1.set_ticklabels([
    "0",
    "0.25",
    "0.5",
    "0.75",
    "1"
])

# Skill
cbar2 = fig.colorbar(
    sc2,
    ax=axes[:, 1],
    orientation='horizontal',
    fraction=0.025,
    aspect=30,
    pad=0.06,

    ticks=[-1, -0.5, 0, 0.5, 1],

    extend='min',
    extendfrac=0.15
)

cbar2.set_label("Correlation skill")

cbar2.set_ticklabels([
    "-1",
    "-0.5",
    "0",
    "0.5",
    "1"
])


plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    path_fig / "fig08.pdf",
    dpi=900,
    bbox_inches="tight"
)

plt.show()

#%%