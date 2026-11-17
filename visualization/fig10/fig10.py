#%% imports
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from properscoring import crps_ensemble
import matplotlib.dates as mdates
from pathlib import Path

#%% # Automatically identify the project root directory

try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

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

path_model = PROJECT_ROOT / "model" / "per_season_model" / "paper_model"

path_data = (
    PROJECT_ROOT / "raw_data" / "per_season_model" / "stations_fig10_data"
)

path_fig = PROJECT_ROOT / "visualization" / "fig10"

path_fig.mkdir(parents=True, exist_ok=True)


init_months = ['2', '5', '8', '11']

season_dict = {
    '2': 'MAM',
    '5': 'JJA',
    '8': 'SON',
    '11': 'DJF'
}


stations_target = [
    2120,
    2153,
    46,
    189,
    568,
    2288
]
#%% target stations

# Final dictionary

station_series = {
    station: []
    for station in stations_target
}

# Loop over seasons

for init_month in init_months:

    # print("\n----------------------")

    season = season_dict[init_month]

    # ======================================================
    # Load model
    # ======================================================

    with open(
        path_model / f"per_season_model_opt_im_{init_month}.dat", "rb"
    ) as f:
        model = pickle.load(f)

   # Load test datasets
    with open(
        path_data / f"per_season_X_test_im_{init_month}_st.pkl", "rb"
    ) as f:
        X_test = pickle.load(f)

    with open(
        path_data / f"per_season_y_test_im_{init_month}_st.pkl", "rb"
    ) as f:
        y_test = pickle.load(f)

    # ======================================================
    # Filter selected stations only
    # ======================================================

    mask_station = X_test["station"].isin(
        stations_target
    )

    X_test = X_test.loc[
        mask_station
    ].reset_index(drop=True)

    y_test = pd.Series(y_test)[
        mask_station.values
    ].reset_index(drop=True)

    # ======================================================
    # Preserve time
    # ======================================================

    time_col = X_test["time"].reset_index(drop=True)



    # ======================================================
    # Model features
    # ======================================================

    cols_model = [
        'lead_time', 'rivdis', 'lwepr', 'lwee',
        'lwesnw', 'rzwc', 'tas', 'PRE_annual',
        'Evaporative_index', 'SWE_variation',
        'TWS_variation', 'Area', 'Elev',
        'Slope', 'dor', 'bare_soil',
        'rock_urban', 'snow_ice', 'forest',
        'crops', 'grasslands', 'wetlands'
    ]

    X_test_model = X_test[cols_model]

    X_test_aux = X_test.drop(
        columns=cols_model,
        errors='ignore'
    )

    # Preserve time
    time_col = X_test["time"].reset_index(drop=True)

    # ======================================================
    # Prediction
    # ======================================================

    X_input = X_test_model.values

    y_pred = model.predict(X_input)

    y_pred = pd.DataFrame(
        y_pred,
        columns=['Qpred']
    )

    # ======================================================
    # Organize dataframe
    # ======================================================

    y_test = pd.DataFrame(
        y_test,
        columns=['Qobs']
    )

    X_test_model = X_test_model.reset_index(drop=True)
    X_test_aux = X_test_aux.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)


    Eval_DF_ensemble = pd.concat(
        [
           y_test,
            y_pred,
            X_test_model,
            X_test_aux
        ],
        axis=1
    )

    # Remove lead time zero
    Eval_DF_ensemble = Eval_DF_ensemble[
        Eval_DF_ensemble["lead_time"] != 0
    ].copy()

    # Explicitly add time
    Eval_DF_ensemble["time"] = time_col

    Eval_DF_ensemble = Eval_DF_ensemble.sort_values("time")


    # ======================================================
    # Ensemble statistics
    # ======================================================

    Eval_DF_stats = (
        Eval_DF_ensemble
        .groupby(
            ['station', 'time', 'lead_time', 'cluster'],
            as_index=False
        )
        .agg({
            'Qpred': ['mean', 'min', 'max'],
            'rivdis': ['mean', 'min', 'max'],
            'Qobs': 'mean'
        })
    )

    Eval_DF_stats.columns = [
        'station',
        'time',
        'lead_time',
        'cluster',
        'Qpred',
        'Qpred_min',
        'Qpred_max',
        'rivdis',
        'rivdis_min',
        'rivdis_max',
        'Qobs'
    ]

    Eval_DF = Eval_DF_stats.copy()



    for station, df_station in Eval_DF.groupby("station"):

            df_station = (
                df_station
                .sort_values("time")
                .reset_index(drop=True)
            )

            df_station["season"] = season

            station_series[station].append(df_station)

for station in stations_target:

    station_series[station] = pd.concat(
        station_series[station],
        ignore_index=True
    )


df_station_series = pd.concat(
    [df.assign(station=station)
     for station, df in station_series.items()],
    ignore_index=True
)

df_station_series['time'] = pd.to_datetime(df_station_series['time'])

df_station_series = df_station_series.sort_values('time')

# df_station_series.to_csv('station_series.csv', index=False)



#%% plot

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9
})

fig, axes = plt.subplots(
    nrows=6,
    ncols=1,
    figsize=(10, 11),
    sharex=True
)

# Ensure axes is iterable
axes = np.atleast_1d(axes)


station_titles = {

    2120: (
        "Station: Prescott | "
        "River: Upper Mississippi | "
        "Area: 116,031 km² | "
        "Process-based: Vegetation-controlled"
    ),

    2153: (
        "Station: Stenay | "
        "River: Meuse | "
        "Area: 3,904 km² | "
        "Process-based: Small basins"
    ),

    46: (
        "Station: Rundu | "
        "River: Okavango | "
        "Area: 104,058 km² | "
        "Process-based: Water-limited"
    ),

    189: (
        "Station: Kyusyur | "
        "River: Lena | "
        "Area: 2,430,000 km² | "
        "Process-based: Snow-driven"
    ),

    568: (
        "Station: Nebraska City | "
        "River: Missouri | "
        "Area: 1,072,154 km² | "
        "Process-based: Higher potential regulation"
    ),

    2288: (
        "Station: Manacapuru | "
        "River: Solimões | "
        "Area: 2,147,740 km² | "
        "Process-based: Energy-limited"
    )
}

# Loop over stations

for i, station in enumerate(stations_target):

    ax = axes[i]

    df_plot = station_series[station].copy()

    if station == 293:
        mask = df_plot["Qobs"].notna() | (df_plot["time"].dt.year == 2002)
    else:
        mask = df_plot["Qobs"].notna()

    df_plot = df_plot.loc[mask]

    # Calculate ensemble metrics
    df_ensemble_station = Eval_DF_ensemble[
        Eval_DF_ensemble["station"] == station
    ].copy()

    # Reset index to create a continuous time series
    df_plot = (
        df_plot
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Continuous x-axis
    x = pd.to_datetime(df_plot["time"])

    # ======================================================
    # Observations
    # ======================================================

    ax.plot(
        x,
        df_plot['Qobs'],
        linewidth=1.5,
        color='black',
        label='Observed' if i == 0 else None
            )


    # ======================================================
    # S9 ensemble spread
    # ======================================================

    ax.fill_between(
        x,
        df_plot['rivdis_min'],
        df_plot['rivdis_max'],
        color='#4A90C2',
        alpha=0.20,
        linewidth=0,
        zorder=1
    )


    # ======================================================
    # S9 forecast
    # ======================================================

    ax.plot(
        x,
        df_plot['rivdis'],
        linewidth=1.0,
        color='#4A90C2',
        label='S9' if i == 0 else None
    )


    # ======================================================
    # S9-ML ensemble spread
    # ======================================================


    ax.fill_between(
        x,
        df_plot['Qpred_min'],
        df_plot['Qpred_max'],
        color='#08519C',
        alpha=0.70,
        linewidth=0,
        zorder=2
    )


    # ======================================================
    # S9-ML forecast
    # ======================================================

    ax.plot(
        x,
        df_plot['Qpred'],
        linewidth=1.5,
        color='#08519C',
        label='S9-ML' if i == 0 else None
    )


    # ======================================================
    # Metrics
    # ======================================================

    # Aligned dataframe
    df_metrics = df_plot[
        ["Qobs", "Qpred", "rivdis"]
    ].copy()

    # Remove NaN values jointly
    df_metrics = df_metrics.dropna()

    # Aligned arrays
    qobs = df_metrics["Qobs"].values

    qpred = df_metrics["Qpred"].values

    qrivdis = df_metrics["rivdis"].values

    # ======================================================
    # S9-ML metrics
    # ======================================================

    corr, pval = pearsonr(
        qobs,
        qpred
    )

    rmse = np.sqrt(
        np.mean(
            (qpred - qobs)**2
        )
    )


    # ======================================================
    # RivDis metrics
    # ======================================================

    corr_rivdis, pval_rivdis = pearsonr(
        qobs,
        qrivdis
    )

    rmse_rivdis = np.sqrt(
        np.mean(
            (qrivdis - qobs)**2
        )
    )


    # ======================================================
    # S9 ensemble CRPS
    # ======================================================

    crps_list = []

    for (time_i, lt_i), df_member in df_ensemble_station.groupby(
        ["time", "lead_time"]
    ):

        obs = df_member["Qobs"].iloc[0]

        # Skip if no observation is available
        if pd.isna(obs):
            continue

        ens = df_member["rivdis"].values

        crps_list.append(
            crps_ensemble(obs, ens)
        )

    crps_mean_s9 = np.nanmean(crps_list)
    # ======================================================
    # ML ensemble CRPS
    # ======================================================

    crps_list = []

    for (time_i, lt_i), df_member in df_ensemble_station.groupby(
        ["time", "lead_time"]
    ):

        obs = df_member["Qobs"].iloc[0]

        # Skip if no observation is available
        if pd.isna(obs):
            continue

        ens = df_member["Qpred"].values

        crps_list.append(
            crps_ensemble(obs, ens)
        )

    crps_mean_ml = np.nanmean(crps_list)



    # ======================================================
    # P-value text
    # ======================================================

    if pval < 0.001:
        pval_text = "p < 0.001"
    else:
        pval_text = f"p = {pval:.3f}"

    if pval_rivdis < 0.001:
        pval_text_rivdis = "p < 0.001"
    else:
        pval_text_rivdis = f"p = {pval_rivdis:.3f}"

    # ======================================================
    # Metrics box
    # ======================================================

    metrics_text = (
        r"$\mathrm{r}_{\mathrm{S9}}$"
        f" = {corr_rivdis:.2f} ({pval_text_rivdis})"
        r"$\;\;$"
        r"$\mathrm{RMSE}_{\mathrm{S9}}$"
        f" = {rmse_rivdis:.2f}"
        r"$\;\;$"
        r"$\mathrm{CRPS}_{\mathrm{S9}}$"
        f" = {crps_mean_s9:.2f}"
        "\n"
        r"$\mathrm{r}_{\mathrm{S9\!-\!ML}}$"
        f" = {corr:.2f} ({pval_text})"
        r"$\;\;$"
        r"$\mathrm{RMSE}_{\mathrm{S9\!-\!ML}}$"
        f" = {rmse:.2f}"
        r"$\;\;$"
        r"$\mathrm{CRPS}_{\mathrm{S9\!-\!ML}}$"
        f" = {crps_mean_ml:.2f}"
    ) 

    # Position
    if i in [3]:   # 4th station
        x_text = 0.68
        y_text = 0.96

    else:
        x_text = 0.0085
        y_text = 0.96



    ax.text(
        x_text,
        y_text,

        metrics_text,

        transform=ax.transAxes,

        fontsize=7,

        va='top',

        ha='left',

        bbox=dict(
            facecolor='white',
            edgecolor='white',
            alpha=0.9,
            linewidth=0.5
        ),

        zorder=0
    )

    # ======================================================
    # Station title
    # ======================================================

    ax.set_title(
        station_titles[station],
        loc='left',
        fontsize=9,
        fontweight='bold'
    )

    # ======================================================
    # Grid
    # ======================================================

    ax.grid(
        True,
        linestyle='--',
        alpha=0.3
    )

    # ======================================================
    # Legend
    # ======================================================

    if i == 0:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.89, 1.02),
            frameon=False
        )

    # ======================================================
    # Y-axis label
    # ======================================================

    ax.set_ylabel('Streamflow (mm/month)')




# X-axis configuration

# Define the plotting period
start_year = 1993
end_year = 2017

# Display annual ticks
axes[-1].xaxis.set_major_locator(
    mdates.YearLocator(1)
)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter('%Y')
)

# Set x-axis limits
axes[-1].set_xlim(
    pd.Timestamp(f'{start_year}-01-01'),
    pd.Timestamp(f'{end_year}-12-31')
)

axes[-1].tick_params(
    axis='x',
    labelsize=8,
    rotation=0
)


# Layout
plt.tight_layout()

# Save

plt.savefig(
    path_fig / "fig10.pdf",
    dpi=900,
    bbox_inches="tight"
)

plt.show()

#%%