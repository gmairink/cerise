#%% # Import libraries
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from properscoring import crps_ensemble
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

path_model = PROJECT_ROOT / "model" / "all_season_model" / "paper_model"
path_data = PROJECT_ROOT / "raw_data" / "all_season_model"
path_results = PROJECT_ROOT / "assessment" / "all_season" / "paper_results"

path_results.mkdir(parents=True, exist_ok=True)


#%% # Load files

with open(path_model / "all_season_model_opt.dat", "rb") as f:
    model = pickle.load(f)

with open(path_data / "all_season_X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open(path_data / "all_season_y_test.pkl", "rb") as f:
    y_test = pickle.load(f)


# Initialization months and their corresponding forecast seasons:
# 2  = FMAM (February–May)
# 5  = MJJA (May–August)
# 8  = ASON (August–November)
# 11 = NDJF (November–February)
init_months = [2, 5, 8, 11]

#%% # Function to calculate deterministic metrics per station

def calculate_station_metrics(df, sim_col):
    metrics_dict = {}
    removed_stations = 0

    for station, group in df.groupby('station'):
        obs = group['Qobs'].to_numpy(copy=True)
        sim = group[sim_col].to_numpy(copy=True)

        valid = np.isfinite(obs) & np.isfinite(sim)
        obs = obs[valid]
        sim = sim[valid]

        if len(obs) < 15:
            removed_stations += 1
            continue

        rmse_val = np.sqrt(np.mean((sim - obs)**2))

        corr_val, pval = pearsonr(obs, sim)

        metrics_dict[station] = {
            'station': station,
            'n_valid': len(obs),
            'r': corr_val,
            'RMSE': rmse_val,
            'pval': pval
        }

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    metrics_df.reset_index(drop=True, inplace=True)

    return metrics_df, removed_stations


#%% # Function to calculate CRPS per station

def calculate_station_crps(
    df,
    ensemble_col='Qpred'
):

    metrics_dict = {}
    removed_stations = 0

    for station, df_station in df.groupby('station'):

        crps_list = []

        grouped = df_station.groupby(
            ['time', 'lead_time']
        )

        for (_, _), group in grouped:

            obs = group['Qobs'].iloc[0]

            ensemble = group[ensemble_col].to_numpy()

            valid = (
                np.isfinite(obs) and
                np.all(np.isfinite(ensemble))
            )

            if not valid:
                continue

            if len(ensemble) < 20:
                continue

            crps = crps_ensemble(
                observations=obs,
                forecasts=ensemble
            )

            crps_list.append(crps)

        if len(crps_list) < 15:
            removed_stations += 1
            continue

        metrics_dict[station] = {
            'station': station,
            'n_valid': len(crps_list),
            'CRPS': np.mean(crps_list)
        }

    metrics_df = pd.DataFrame.from_dict(
        metrics_dict,
        orient='index'
    )

    metrics_df.reset_index(drop=True, inplace=True)

    return metrics_df, removed_stations

#%% # Generate predictions

cols_model = ['init_month', 'lead_time', 'rivdis', 'lwee', 'lwepr',
        'lwesnw', 'rzwc', 'tas', 'PRE_annual', 'Evaporative_index', 
        'SWE_variation', 'TWS_variation', 'Area', 'Elev', 'Slope', 'dor', 
        'bare_soil','rock_urban','snow_ice','forest','crops','grasslands','wetlands']

X_test_model = X_test[cols_model]
X_test_aux = X_test.drop(columns=cols_model, errors='ignore')

# Generate predictions using the trained model.
X_test = X_test_model.values
y_pred_aux = model.predict(X_test)

# Store predictions in a DataFrame.
y_pred = pd.DataFrame(y_pred_aux, columns=['Qpred'])

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


# Count negative ensemble members by station, time, lead time, and season
negative_members_summary = (
            Eval_DF
            .groupby(['station', 'time', 'lead_time', 'init_month'])
            .agg(
                n_members=('Qpred', 'size'),
                n_negative=('Qpred', lambda x: (x < 0).sum())
            )
            .reset_index()
        )


# Identify ensembles to be discarded
ensembles_to_exclude = negative_members_summary.loc[
    negative_members_summary["n_negative"] > 10,
    ["station", "time", "lead_time", "init_month"]
]


# Remove these ensembles from Eval_DF
Eval_DF = (
    Eval_DF.merge(
        ensembles_to_exclude.assign(exclude_ensemble=True),
        on=["station", "time", "lead_time", "init_month"],
        how="left"
    )
)

Eval_DF = Eval_DF[
    Eval_DF["exclude_ensemble"] != True
].drop(columns="exclude_ensemble")


# Remove only the negative ensemble members
Eval_DF = Eval_DF[
    (Eval_DF["Qpred"] > 0)
].copy()


#%% # Deterministic metrics

df_station_time = (
        Eval_DF
        .groupby(['station', 'time', 'lead_time', 'init_month',], as_index=False)[['Qpred', 'rivdis', 'Qobs']]
        .mean()
    )

for init_month in init_months:
    print(f"\nProcessing init_month = {init_month}")

    df_lt_init_month = df_station_time[df_station_time['init_month'] == init_month].copy()

    unique_lead_times = sorted(df_station_time['lead_time'].dropna().unique())

    for lt in unique_lead_times:

        print(f"\nDeterministic metrics - processing lead_time = {lt}")


        # -----------------------------
        # Removal statistics
        # -----------------------------

        # Ensembles removed (>10 negative members)
        ensembles_to_exclude_lt = ensembles_to_exclude[
            (ensembles_to_exclude["lead_time"] == lt) &
            (ensembles_to_exclude["init_month"] == init_month)
        ]
        
        n_stations = ensembles_to_exclude_lt["station"].nunique()
        n_ensembles = len(ensembles_to_exclude_lt)


        print(f"\nStations removed (>10 negative members): {n_stations}")
        print(f"Ensembles removed: {n_ensembles}\n")



        df_init_month_lt = df_lt_init_month[df_lt_init_month['lead_time'] == lt].copy()

        # ==========================
        # DYNAMIC MODEL
        # ==========================

        metrics_df_dm, removed_stations_dm = calculate_station_metrics(
            df_init_month_lt.copy(),
            sim_col='rivdis'
        )

        print(f'DM - LT{lt} | Number of stations removed (< 15 samples): {removed_stations_dm}')
        print(f'DM - LT{lt} | Number of stations retained: {len(metrics_df_dm)}')

        metrics_df_dm.to_csv(
            path_results / f"all_season_s9_im{init_month}_lt{lt}_deterministic_metrics.csv",
            index=False
        )

        # ==========================
        # ML MODEL
        # ==========================

        metrics_df_ml, removed_stations_ml = calculate_station_metrics(
            df_init_month_lt.copy(),
            sim_col='Qpred'
        )

        print(f'ML - LT{lt} | Number of stations removed (< 15 samples): {removed_stations_ml}')
        print(f'ML - LT{lt} | Number of stations retained: {len(metrics_df_ml)}')

        metrics_df_ml.to_csv(
            path_results / f"all_season_ml_im{init_month}_lt{lt}_deterministic_metrics.csv",
            index=False
        )
        
#%% # Probabilistic metrics

for init_month in init_months:
    print(f"\nProcessing init_month = {init_month}")

    df_lt_init_month = Eval_DF[Eval_DF['init_month'] == init_month].copy()

    for lt in sorted(Eval_DF['lead_time'].unique()):

        print(f"\nCRPS - processing lead_time = {lt}")

        df_lt_crps = df_lt_init_month[
            df_lt_init_month['lead_time'] == lt
        ]

        # ==========================
        # DYNAMIC MODEL
        # ==========================

        crps_df_dm, removed_stations_dm_crps = calculate_station_crps(
            df_lt_crps,
            ensemble_col='rivdis'
        )

        print(f'DM - LT{lt} | Number of stations removed (< 15 samples): {removed_stations_dm_crps}')
        print(f'DM - LT{lt} | Number of stations retained: {len(crps_df_dm)}')


        crps_df_dm.to_csv(
            path_results / f"all_season_s9_im{init_month}_lt{lt}_crps.csv",
            index=False
        )

        # ==========================
        # ML MODEL
        # ==========================

        crps_df_ml, removed_stations_ml_crps = calculate_station_crps(
            df_lt_crps,
            ensemble_col='Qpred'
        )

        print(f'ML - LT{lt} | Number of stations removed (< 15 samples): {removed_stations_ml_crps}')
        print(f'ML - LT{lt} | Number of stations retained: {len(crps_df_ml)}')


        crps_df_ml.to_csv(
            path_results / f"all_season_ml_im{init_month}_lt{lt}_crps.csv",
            index=False
        )
        
#%%