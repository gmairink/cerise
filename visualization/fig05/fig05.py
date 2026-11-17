#%% # Import libraries
from pathlib import Path
import joblib  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

path_model = PROJECT_ROOT / "model"
path_fig = PROJECT_ROOT / "visualization" / "fig05"

path_fig.mkdir(parents=True, exist_ok=True)

#%% ###

init_months = [2, 5, 8, 11]  

results = []

# All-season model
all_season_model_path = (
    path_model / "all_season_model" / "paper_model"
    / "all_season_model_opt.dat"
)

all_season_model = joblib.load(all_season_model_path)

fi_all_season = all_season_model.feature_importances_

# If feature names are available

feature_names_all_season = ['init_month', 'lead_time', 'rivdis', 'lwee', 'lwepr',
        'lwesnw', 'rzwc', 'tas', 'PRE_annual', 'Evaporative_index', 
        'SWE_variation', 'TWS_variation', 'Area', 'Elev', 'Slope', 'dor', 
        'bare_soil','rock_urban','snow_ice','forest','crops','grasslands','wetlands']

feature_names_seasonal = ['lead_time', 'rivdis', 'lwee', 'lwepr',
        'lwesnw', 'rzwc', 'tas', 'PRE_annual', 'Evaporative_index', 
        'SWE_variation', 'TWS_variation', 'Area', 'Elev', 'Slope', 'dor', 
        'bare_soil','rock_urban','snow_ice','forest','crops','grasslands','wetlands']

df_all_season = pd.DataFrame({
    "feature": feature_names_all_season,
    "importance": fi_all_season,
    "model": "all_season"
})

results.append(df_all_season)


# Per-season models
for im in init_months:

    model_file = (
        path_model / "per_season_model" / "paper_model"
        / f"per_season_model_opt_im_{im}.dat"
    )

    model = joblib.load(model_file)

    fi = model.feature_importances_

    df = pd.DataFrame({
    "feature": feature_names_seasonal,
    "importance": fi,
    "model": f"im{im}"
    })

    results.append(df)

# --- Combine all results ---
df_final = pd.concat(results)

# Optional pivot (features vs. models)
df_table = df_final.pivot(index="feature", columns="model", values="importance")

# Dictionary of display names
feature_display_names = {
    "PRE_annual": "Pa",
    "Evaporative_index": "AET/Pa",
    "Area": "Area",
    "Elev": "Elevation",
    "Slope": "Slope",
    "dor": "DoR",
    "SWE_variation": "SWE seasonality",
    "TWS_variation": "TWS seasonality",
    'bare_soil': "BareSoil",
    'crops': "Crop",
    'forest': "Forest",
    'grasslands': "Grass",
    'rock_urban': "UrbainRock",
    'snow_ice': "SnowIce",
    'wetlands': "Wetland",

    "lwee": "ET",
    "lwepr": "Prec",
    "tas": "T",                              
    "lwesnw": "SWE",  
    "rivdis": "Streamflow",
    "rzwc": "Root SM",


    "lead_time": "Lead time",
    'init_month': 'Season ID',
}

df_table = df_table.reset_index()
df_table["feature"] = df_table["feature"].map(feature_display_names).fillna(df_table["feature"])

# Select only model columns (avoid additional metrics)
model_cols = [c for c in df_table.columns if c.startswith("im") or c == "all_season"]

# Calculate feature rankings (1 = most important)
for col in model_cols:
    df_table[f"{col}_rank"] = df_table[col].rank(
        ascending=False, method="min"
    )

print(df_table)

#%% # Plot: feature importance for the all-season and seasonal models

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11
})

model_order = ["all_season", "im2", "im5", "im8", "im11"]
y_labels = ["All-season", "MAM", "JJA", "SON", "DJF"]

df_plot = df_table.copy()

feature_order = [

    # Static variables
    'Pa',
    'AET/Pa',
    'Area',
    'Elevation',
    'Slope',
    'DoR',
    'SWE seasonality',
    'TWS seasonality',

    # Land cover variables
    'BareSoil',
    'Crop',
    'Forest',
    'Grass',
    'SnowIce',
    'UrbainRock',
    'Wetland',

    # Dynamic variables
    'ET',
    'Prec',
    'T',
    'SWE',
    'Streamflow',
    'Root SM',

    # Recurrent dynamic variables
    'Lead time',
    'Season ID'
]

df_plot = df_plot.set_index("feature")
df_plot = df_plot.reindex(feature_order)

# Matrices
importance = df_plot[model_order].values
ranks = df_plot[[f"{m}_rank" for m in model_order]].values

# Handle NaN values
importance = np.nan_to_num(importance, nan=0)
ranks = np.nan_to_num(ranks, nan=np.nanmax(ranks))

# Size
# Importance classes
bins = [0, 0.25, 0.5, 0.75, 1.0]

# Fixed size for each class
sizes_fixed = [40, 80, 120, 160]

# Normalize importance values between 0 and 1
importance_norm = importance / importance.max()

# Create the size array
size = np.zeros_like(importance_norm)

for i in range(len(sizes_fixed)):
    mask = (
        (importance_norm >= bins[i]) &
        (importance_norm < bins[i+1])
    )
    size[mask] = sizes_fixed[i]

# Include exactly 1.0 in the last class
size[importance_norm == 1.0] = sizes_fixed[-1]

# Ranking classes
n_classes = 4
rank_max = np.nanmax(ranks)

rank_classes = np.ceil((ranks / rank_max) * n_classes)
rank_classes = n_classes - rank_classes + 1

# Grid
x, y = np.meshgrid(np.arange(len(feature_order)), np.arange(len(model_order)))

# Plot
fig, ax = plt.subplots(figsize=(11, 3))

# Use ranking directly (continuous)
color_values = ranks

# Mask valid values
mask = ~np.isnan(df_plot[model_order].values)

# Flatten valid values only
x_plot = x.flatten()[mask.T.flatten()]
y_plot = y.flatten()[mask.T.flatten()]

size_plot = size.T.flatten()[mask.T.flatten()]
color_plot = color_values.T.flatten()[mask.T.flatten()]


# Set limits (1 = best)
vmin = 1
vmax = np.nanmax(ranks)

sc = ax.scatter(
    x_plot,
    y_plot,
    s=size_plot,
    c=color_plot,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    edgecolor="k",
    linewidth=0.7
)

ax.set_xticks(np.arange(len(feature_order)))
ax.set_xticklabels(feature_order, rotation=45, ha="right")

ax.set_yticks(np.arange(len(model_order)))
ax.set_yticklabels(y_labels)

ax.invert_yaxis()
ax.set_ylim(4.5, -0.5)


cbar = fig.colorbar(
    sc,
    ax=ax,
    shrink=1,   # Controls bar length
    aspect=20   # Height-to-width ratio
    )


cbar.set_label("Ranking")
cbar.set_ticks([1, 5, 10, 15, 20])

# ---- Size legend ----
sizes_legend = [0.01, 0.25, 0.5, 0.75, 1]

handles = [
    plt.scatter([], [], s=s, facecolors='none', edgecolors='k')
    for s in sizes_fixed
]

labels = ["0.25", "0.5", "0.75", "1"]

legend = ax.legend(
    handles,
    labels,
    title="Importance",
    scatterpoints=1,
    frameon=False,
    labelspacing=0.7,
    borderpad=1.5,
    handletextpad=0.2,
    loc="center left",
    bbox_to_anchor=(0.87, -0.35),   # x, y
    ncol=2,
    fontsize=10)

legend.get_title().set_verticalalignment('center')
legend.get_title().set_horizontalalignment('left')

ax.add_artist(legend)

plt.tight_layout()

# [left, bottom, width, height]
cbar.ax.set_position([0.81, 0.365, 0.2, 0.581])

plt.savefig(
    path_fig / "fig05.pdf",
    bbox_inches="tight",
    dpi=900
)

plt.show()

#%%