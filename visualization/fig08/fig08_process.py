#%% # Import libraries
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
import evalhyd as eh
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

path_model = PROJECT_ROOT / "model" / "per_season_model" / "paper_model"
path_data = PROJECT_ROOT / "raw_data" / "per_season_model"
path_save = PROJECT_ROOT / "visualization" / "fig08" / "lt123_metrics"

path_save.mkdir(parents=True, exist_ok=True)

init_months = ['2', '5', '8', '11']

#%% # Function to calculate metrics per station

def calculate_station_metrics(df, sim_col):
    metrics_dict = {}
    removed_stations = 0

    for station, group in df.groupby('station'):
        obs = group['Qobs'].to_numpy(copy=True)
        sim = group[sim_col].to_numpy(copy=True)

        valid = np.isfinite(obs) & np.isfinite(sim)
        obs = obs[valid]
        sim = sim[valid]

        results = eh.evald(
            q_obs=obs,
            q_prd=sim,
            metrics=["RMSE", "KGE", "NSE"]
        )

        rmse_val = float(results[0][0][0][0])
        kge_val = float(results[1][0][0][0])
        nse_val = float(results[2][0][0][0])

        bias_val = np.mean(sim - obs)

        ub_rmse_squared = rmse_val**2 - bias_val**2
        ub_rmse = np.sqrt(ub_rmse_squared) if ub_rmse_squared >= 0 else np.nan

        sum_obs = np.sum(obs)
        pbias_val = 100 * np.sum(sim - obs) / sum_obs if sum_obs != 0 else np.nan

        bias_abs_val = np.mean(np.abs(sim - obs))

        corr_val, pval = pearsonr(obs, sim)

        metrics_dict[station] = {
            'station': station,
            'n_valid': len(obs),
            'RMSE': rmse_val,
            'Bias': bias_val,
            'ub RMSE': ub_rmse,
            'KGE': kge_val,
            'NSE': nse_val,
            'BIAS abs': bias_abs_val,
            'PBIAS': pbias_val,
            'r': corr_val,
            'pval': pval,
        }

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    metrics_df.reset_index(drop=True, inplace=True)

    return metrics_df, removed_stations

#%% # Generate predictions and calculate the mean for LT1, LT2, and LT3

for init_month in init_months:

    print("\n----------------------")
    print(f"\nProcessing init_month = {init_month}")

    # Load the model
    with open(
        path_model / f"per_season_model_opt_im_{init_month}.dat", "rb"
    ) as f:
        model = pickle.load(f)

    # Load the test datasets
    with open(
        path_data / f"per_season_X_test_im_{init_month}.pkl", "rb"
    ) as f:
        X_test = pickle.load(f)

    with open(
        path_data / f"per_season_y_test_im_{init_month}.pkl", "rb"
    ) as f:
        y_test = pickle.load(f)


    cols_model = ['lead_time', 'rivdis', 'lwepr', 'lwee',
            'lwesnw', 'rzwc', 'tas', 'PRE_annual', 'Evaporative_index', 
            'SWE_variation', 'TWS_variation', 'Area', 'Elev', 'Slope', 'dor', 
            'bare_soil','rock_urban','snow_ice','forest','crops','grasslands','wetlands']

    X_test_model = X_test[cols_model]
    X_test_aux = X_test.drop(columns=cols_model, errors='ignore')   # Auxiliary variables (station, cluster, etc.)

    # Generate predictions
    X_test = X_test_model.values

    # Apply the log transformation because the model was trained using log-transformed y_train (Qobs)
    y_pred_log = model.predict(X_test)

    # Ensure a 2D format
    y_pred = pd.DataFrame(y_pred_log, columns=['Qpred'])

    # Reverse the log transformation
    # y_pred = np.expm1(y_pred_log['Qpred_log']).to_frame(name='Qpred')

    # Ensure y_test is a DataFrame
    y_test = pd.DataFrame(y_test, columns=['Qobs'])

    # Align indices
    X_test_model = X_test_model.reset_index(drop=True)
    X_test_aux = X_test_aux.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    # Build the final DataFrame
    Eval_DF = pd.concat(
        [y_test, y_pred, X_test_model, X_test_aux],
        axis=1
    )

    # Calculate the mean by station, time, and lead time

    df_station_time = (
        Eval_DF
        .groupby(['station', 'time', 'lead_time'], as_index=False)[['Qpred', 'rivdis', 'Qobs']]
        .mean()
    )

    ## ======================================================
    # Combine lead times 1, 2, and 3
    # ======================================================

    # Select LT1, LT2, and LT3
    df_lt123 = df_station_time[
        df_station_time['lead_time'].isin([1, 2, 3])
    ].copy()

    # n_stations = df_lt123['station'].nunique()
    # print(n_stations)

    # ======================================================
    # Remove negative Qpred values
    # ======================================================

    n_before = len(df_lt123)

    df_lt123 = df_lt123[
        (df_lt123["Qpred"] > 0) 
    ].copy()

    n_after = len(df_lt123)

    print(
        f"\nRows removed due to negative Qpred values: "
        f"{n_before - n_after}"
    )

    # ======================================================
    # Remove stations with fewer than 15 years
    # containing complete LT1, LT2, and LT3 records
    # ======================================================

    valid_stations = []
    removed_stations_list = []

    # Create the year variable
    df_lt123["year"] = pd.to_datetime(
        df_lt123["time"]
    ).dt.year

    for station, group in df_lt123.groupby("station"):

        n_valid_years = 0

        for year, gyear in group.groupby("year"):

            lts = sorted(
                gyear["lead_time"].unique()
            )

            # Check whether the year contains LT1, LT2, and LT3
            if lts == [1, 2, 3]:

                counts = (
                    gyear["lead_time"]
                    .value_counts()
                    .sort_index()
                )

                # Exactly one value per lead time
                if np.all(counts.values == 1):

                    n_valid_years += 1

        # Retain only stations with at least 15 valid years
        if n_valid_years >= 15:

            valid_stations.append(station)

        else:

            removed_stations_list.append(station)

    # Filter the DataFrame
    df_lt123 = df_lt123[
        df_lt123["station"].isin(valid_stations)
    ].copy()

    print(
        f"Stations removed (<15 years with complete LT1, LT2, and LT3 records): "
        f"{len(removed_stations_list)}"
    )

    print(
        f"Stations retained: "
        f"{len(valid_stations)}"
    )


    # ------------------------------------------------------
    # Calculate the mean across LT1, LT2, and LT3
    # Only when LT1, LT2, and LT3 are all available
    # for the same station and year
    # ------------------------------------------------------

    dfs_mean = []

    n_total = 0
    n_removed_missing_lt = 0
    n_removed_duplicate_lt = 0

    # Create the year variable
    df_lt123["year"] = pd.to_datetime(
        df_lt123["time"]
    ).dt.year

    for (station, year), group in df_lt123.groupby(["station", "year"]):

        n_total += 1

        # Check the available lead times
        lts = sorted(group["lead_time"].unique())

        # Check whether any lead time is missing
        if lts != [1, 2, 3]:

            n_removed_missing_lt += 1
            continue

        # Count the number of samples per lead time
        counts = (
            group["lead_time"]
            .value_counts()
            .sort_index()
        )

        # Check whether any lead time has more than one value
        if not np.all(counts.values == 1):

            n_removed_duplicate_lt += 1
            continue

        # Calculate the LT1-LT3 mean
        df_mean = pd.DataFrame({

            "station": [station],

            "year": [year],

            "Qpred": [group["Qpred"].mean()],

            "rivdis": [group["rivdis"].mean()],

            "Qobs": [group["Qobs"].mean()]
        })

        dfs_mean.append(df_mean)


    # Create the final DataFrame
    df_mean_lt123 = pd.concat(
        dfs_mean,
        ignore_index=True
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print(
        f"Seasonal samples removed due to missing lead times: "
        f"{n_removed_missing_lt}"
    )

    print(
        f"Seasonal samples removed due to duplicated lead times: "
        f"{n_removed_duplicate_lt}"
    )

    print(
        f"Total stations: {df_mean_lt123['station'].nunique()}"
    )


    # ======================================================
    # DYNAMIC MODEL
    # ======================================================

    metrics_df_dm, removed_stations_dm = (
        calculate_station_metrics(
            df_mean_lt123.copy(),
            sim_col='rivdis'
        )
    )

    metrics_df_dm.to_csv(
        path_save / f"s9_im{init_month}_lt123_metrics.csv",
        index=False
    )

    # ======================================================
    # ML MODEL
    # ======================================================

    metrics_df_ml, removed_stations_ml = (
        calculate_station_metrics(
            df_mean_lt123.copy(),
            sim_col='Qpred'
        )
    )

    metrics_df_ml.to_csv(
        path_save / f"ml_im{init_month}_lt123_metrics.csv",
        index=False
    )

#%%