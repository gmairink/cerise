#%% # Import libraries
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

#%% # Define initialization months and paths

init_months = ['2', '5', '8', '11']

# Automatically identify the project root directory
current_dir = Path(__file__).resolve().parent

PROJECT_ROOT = next(
    (
        parent for parent in [current_dir, *current_dir.parents]
        if all(
            (parent / folder).is_dir()
            for folder in ["raw_data", "model", "visualization"]
        )
    ),
    None
)

if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root directory not found.")

path_data = PROJECT_ROOT / "raw_data" / "per_season_model"
path_save = PROJECT_ROOT / "visualization" / "fig03" / "climatology"

path_save.mkdir(parents=True, exist_ok=True)


#%% # Loop over initialization months

for init_month in init_months:

    print(f"\nProcessing init_month = {init_month}")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    with open(path_data / f"per_season_X_test_im_{init_month}.pkl", "rb") as f:
        X_test = pickle.load(f)

    with open(path_data / f"per_season_y_test_im_{init_month}.pkl", "rb") as f:
        y_test = pickle.load(f)

    # Keep only variables required for climatology

    df = X_test[
        ['station', 'time', 'lead_time']
    ].copy().reset_index(drop=True)

    df['Qobs'] = np.asarray(y_test).reshape(-1)

    # --------------------------------------------------
    # Extract month and year from time
    # --------------------------------------------------

    df = df[
        df["lead_time"].isin([1, 2, 3])
    ].copy()

    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year


    # --------------------------------------------------
    # Calculate mean by station/time/lead_time
    # --------------------------------------------------

    df_station_time = (
        df
        .groupby(
            ["station", "time", "lead_time"],
            as_index=False
        )["Qobs"]
        .mean()
    )

    df_station_time["month"] = df_station_time["time"].dt.month
    df_station_time["year"] = df_station_time["time"].dt.year


    # --------------------------------------------------
    # Leave-one-year-out climatology
    # --------------------------------------------------

    clim_stats = (
        df_station_time
        .groupby(["station", "month"])["Qobs"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "sum_all", "count": "n_all"})
        .reset_index()
    )

    year_stats = (
        df_station_time
        .groupby(["station", "month", "year"])["Qobs"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "sum_year", "count": "n_year"})
        .reset_index()
    )

    year_stats = year_stats.merge(
        clim_stats,
        on=["station", "month"],
        how="left"
    )

    year_stats["Qclim"] = (
        (year_stats["sum_all"] - year_stats["sum_year"])
        /
        (year_stats["n_all"] - year_stats["n_year"])
    )

    df_station_time = df_station_time.merge(
        year_stats[
            ["station", "month", "year", "Qclim"]
        ],
        on=["station", "month", "year"],
        how="left"
    )


    # --------------------------------------------------
    # Calculate climatological ubRMSE per lead time
    # --------------------------------------------------

    unique_lead_times = sorted(
        df_station_time["lead_time"].dropna().unique()
    )

    for lt in unique_lead_times:

        print(f"\nProcessing lead_time = {lt}")

        df_lt = df_station_time[
            df_station_time["lead_time"] == lt
        ].copy()

        # Remove negative streamflow values

        n_before = len(df_lt)

        df_lt = df_lt[
            (df_lt["Qobs"] > 0) &
            (df_lt["Qclim"] > 0)
        ]

        print("Rows removed:", n_before - len(df_lt))

        # Calculate ubRMSE per station

        results = []
        removed_stations = 0

        for station, df_station in df_lt.groupby("station"):

            if len(df_station) < 15:
                removed_stations += 1
                continue

            errors = (
                df_station["Qobs"] - df_station["Qclim"]
            )

            bias = errors.mean()

            ubrmse_clim = np.sqrt(
                np.mean((errors - bias)**2)
            )

            results.append({
                "station": station,
                "ubRMSE_clim": ubrmse_clim
            })

        metrics_df_clim = pd.DataFrame(results)

        print(
            f"LT{lt} | Stations removed (<15 samples): "
            f"{removed_stations}"
        )

        print(
            f"LT{lt} | Stations retained: "
            f"{len(metrics_df_clim)}"
        )

        # Save results

        metrics_df_clim.to_csv(
            path_save / f"per_season_climatology_im{init_month}_lt{int(lt)}.csv",
            index=False
        )

#%%