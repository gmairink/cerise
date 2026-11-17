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

path_data = PROJECT_ROOT / "raw_data" / "all_season_model"
path_model = PROJECT_ROOT / "model" / "all_season_model"

path_model.mkdir(parents=True, exist_ok=True)


# data
with open(path_data / "all_season_X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open(path_data / "all_season_y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open(path_data / "all_season_groups_train.pkl", "rb") as f:
    groups_train = pickle.load(f)


### bayesian optimization
def XGBoost_OptunaOptimization(X_train, y_train, groups_train, n_trials):

    # Save the study and the best hyperparameters
    path_all_study = path_model / "all_season_model_optuna_study.pkl"
    path_best_params = path_model / "all_season_model_best_params.pkl"

    gkf = GroupKFold(n_splits=3)    # 3 splits in cross-validation

    def objective(trial):

        # The idea is to define a search space that avoids extreme combinations;
        # for example: a low learning rate requires a higher number of estimators (tress)
        # and a high learning rate requires fewer estimators;

        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),              # Maximum tree depth
            'n_estimators': trial.suggest_int('n_estimators', 100, 2100, step=100),  # Number of trees
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Learning rate (shrinkage)
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),        # Fraction of samples used for each tree
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Fraction of features used for each tree
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),  # L1 regularization (feature selection)
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True), # L2 regularization (reduces overfitting)
            'huber_slope': trial.suggest_float('huber_slope', 0.1 * np.std(y_train), 5 * np.std(y_train), log=True)  # Transition between squared and absolute errors in the Pseudo-Huber loss
}

        fold_mae = []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):

            # Split the data into training and validation sets while respecting the groups
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            # Create the XGBoost model using the hyperparameters suggested by Optuna
            model = XGBRegressor(
                **params,
                objective='reg:pseudohubererror',
                eval_metric='mae',
                early_stopping_rounds=50,  # Stop training if the MAE does not improve for 50 consecutive trees
                random_state=1
            )

            # Train the model and monitor its performance on the validation set
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Generate predictions for the validation set
            y_pred = model.predict(X_val)

            # Calculate the mean absolute error (MAE) for the fold
            mae = mean_absolute_error(y_val, y_pred)

            # Store the result for calculating the mean across folds
            fold_mae.append(mae)

        return np.mean(fold_mae)


    # Create an Optuna study to minimize the validation error
    # The MedianPruner stops trials that perform clearly worse than previous trials
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner()
    )

    # Start measuring the optimization time
    start = time.time()

    # Run the hyperparameter search
    study.optimize(objective, n_trials=n_trials)

    # Calculate the total optimization time
    elapsed = time.time() - start

    print(f'\nTotal optimization time: {format_time(elapsed)}')

    # Best mean MAE score obtained during cross-validation
    print('\nBest mean validation score (MAE):')
    print(study.best_value)

    # Best hyperparameters found by Optuna
    print('\nBest hyperparameters:')
    print(study.best_params)

    # Save the complete study
    joblib.dump(study, path_all_study)

    # Save only the best hyperparameters
    joblib.dump(study.best_params, path_best_params)

    # Return the study object
    return study

# Run Bayesian hyperparameter optimization
study = XGBoost_OptunaOptimization(
    X_train=X_train,
    y_train=y_train,
    groups_train=groups_train,
    n_trials=30,
)


# Final training
best_params = joblib.load(
    path_model / "all_season_model_best_params.pkl"
)

model = XGBRegressor(
    objective='reg:pseudohubererror',
    random_state=1,
    **best_params
)

print('\nFitting the final model')
model.fit(X_train, y_train)

with open(path_model / "all_season_model_opt.dat", "wb") as f:
    dump(model, f)

print('\nModel saved successfully')
print('Best parameters used for the final model:')
print(best_params)