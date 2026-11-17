### import libraries
import numpy as np
import pickle
import joblib
import optuna
from pickle import dump
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
import time
from pathlib import Path

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))



# Automatically identify the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

path_data = PROJECT_ROOT / "raw_data" / "per_season_model"
path_model = PROJECT_ROOT / "model" / "per_season_model"

path_model.mkdir(parents=True, exist_ok=True)



init_months = ['2', '5', '8', '11'] 

# Initialization months and their corresponding forecast seasons:
# 2  = FMAM (February–May)
# 5  = MJJA (May–August)
# 8  = ASON (August–November)
# 11 = NDJF (November–February)


### Bayesian optimization
def XGBoost_OptunaOptimization(init_month, X_train, y_train, groups_train, n_trials):
    
    path_all_study = path_model / f"per_season_model_im_{init_month}_optuna_study.pkl"
    path_best_params = path_model / f"per_season_model_im_{init_month}_best_params.pkl"
    

    gkf = GroupKFold(n_splits=3)

    def objective(trial):

        # The goal is to define a search space that avoids extreme combinations;
        # for example, a low learning rate requires more estimators (trees),
        # whereas a high learning rate requires fewer estimators.

        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2100, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'huber_slope': trial.suggest_float('huber_slope', 0.1 * np.std(y_train), 5 * np.std(y_train), log=True)
        }


        fold_mae = []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]


            model = XGBRegressor(
                **params,
                tree_method="hist",
                objective='reg:pseudohubererror',
                eval_metric='mae',
                early_stopping_rounds=50,
                random_state=1
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            fold_mae.append(mae)

        return np.mean(fold_mae)

    # Create an Optuna study to minimize the validation error.
    # The MedianPruner stops trials that perform poorly compared with previous trials.
    study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)

    # Track the optimization time
    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    print(f'\nTotal optimization time: {format_time(elapsed)}')

    print('\nBest mean validation score (MAE):')
    print(study.best_value)

    print('\nBest hyperparameters:')
    print(study.best_params)

    joblib.dump(study, path_all_study)
    joblib.dump(study.best_params, path_best_params)

    return study

for init_month in init_months:

    print(f"Initialization month: {init_month}")

    # Load training data for the current initialization month
    with open(path_data / f"per_season_X_train_im_{init_month}.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(path_data / f"per_season_y_train_im_{init_month}.pkl", "rb") as f:
        y_train = pickle.load(f)

    with open(path_data / f"per_season_groups_train_im_{init_month}.pkl", "rb") as f:
        groups_train = pickle.load(f)

    # Bayesian optimization
    study = XGBoost_OptunaOptimization(
        init_month=init_month,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        n_trials=30
    )

    # Train final model
    best_params = joblib.load(
        path_model / f"per_season_model_im_{init_month}_best_params.pkl"
    )

    model = XGBRegressor(
        objective="reg:pseudohubererror",
        tree_method="hist",
        random_state=1,
        **best_params
    )

    print("\nFitting the final model")
    model.fit(X_train, y_train)

    # Save final model
    with open(path_model / f"per_season_model_opt_im_{init_month}.dat", "wb") as f:
        dump(model, f)

    print("\nFinal model saved successfully")
    print("Best parameters used for the final model:")
    print(best_params)