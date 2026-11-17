# Machine-learning post-processing of global seasonal streamflow forecasts from Météo-France System 9

This repository contains the code used to train, evaluate, and visualize the XGBoost-based post-processing models developed for the study *"Machine-learning post-processing of global seasonal streamflow forecasts from Météo-France System 9"*.

The workflow is divided into three main stages:

1. **Model training:** train the all-season and per-season XGBoost models.
2. **Model assessment:** evaluate the trained models using deterministic and probabilistic metrics.
3. **Visualization:** generate the figures and supporting analyses presented in the study.

## 1. Installation and data setup

### Python dependencies

The `requirements.txt` file lists the Python packages required to run the framework. From the repository root, install them with:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended to keep the project dependencies separate from other Python environments.

### Input data and trained models

> **ATTENTION:** The input datasets and trained model files are not included in this GitHub repository. They are provided separately through the project's Zenodo archive.

**Zenodo archive:** [https://doi.org/10.5281/zenodo.24202901](https://doi.org/10.5281/zenodo.24202901)

The Zenodo record contains two downloadable archives:

- `cerise_input_data.zip`: contains the input datasets organized in the `raw_data/` directory.
- `cerise_trained_models.zip`: contains the trained models, Optuna studies, and selected hyperparameters organized in the `model/` directory.

Download both archives and extract their contents into the root directory of the cloned GitHub repository. Preserve the original directory structure and filenames, as the scripts rely on these paths to locate the corresponding files.

After extraction, the project directory should contain both `raw_data/` and `model/`.

### Input data structure

The `raw_data/` directory contains two modelling configurations:

- `all_season_model/`: data used by the all-season model.
- `per_season_model/`: data separated by initialization month for the per-season models.

For both configurations, the main files are:

| File | Description |
|---|---|
| `*_X_train*.pkl` | Predictor variables used for model training. |
| `*_y_train*.pkl` | Observed streamflow targets corresponding to the training data. |
| `*_X_test*.pkl` | Predictor variables used for model assessment. |
| `*_y_test*.pkl` | Observed streamflow targets corresponding to the test data. |
| `*_groups_train*.pkl` | Training station groups used in the river-based data split. |
| `*_groups_test*.pkl` | Test station groups used in the river-based data split. |

The `X` files contain the predictors, whereas the corresponding `y` files contain the observed streamflow targets. The train/test group files describe the station grouping used to define the river-based split.

### Per-season datasets

The per-season datasets are stored separately for each initialization month. The month identifies the forecast period as follows:

| Initialization month | Period |
|---:|---|
| 2 | February–May |
| 5 | May–August |
| 8 | August–November |
| 11 | November–February |

For example, `per_season_X_train_im_2.pkl` and `per_season_y_train_im_2.pkl` contain the predictors and observed targets used to train the February-initialized model. The same naming convention applies to the other initialization months and to the test datasets and station-group files.

The `per_season_model/stations_fig10_data/` directory contains the data prepared for the selected-station case studies presented in Figure 10.

### Trained model structure

The `model/` directory contains the trained model artifacts used in the study:

```text
model/
├── all_season_model/
│   └── paper_model/
└── per_season_model/
    └── paper_model/
```

The `paper_model/` directories contain the trained XGBoost models, Optuna studies, and selected best hyperparameters for the corresponding modelling configurations.

## 2. Model training

The model-training scripts are located in `model_train_scripts/`:

- `all_season_model.py`: trains the all-season XGBoost model.
- `per_season_model.py`: trains separate XGBoost models for the four initialization months (2, 5, 8, and 11).

The scripts load the corresponding training datasets from `raw_data/`. They perform hyperparameter optimization and save the Optuna study, the selected best parameters, and the trained model in the corresponding subdirectories of `model/`.

The trained model archive provided through Zenodo can be used directly to reproduce the assessment and visualization stages without repeating the training process. Alternatively, the models can be retrained using the provided scripts and input datasets.

## 3. Model assessment

The `assessment/` directory contains the scripts and results used to evaluate the trained models:

```text
assessment/
├── scripts/
│   ├── all_season_assessment.py
│   └── per_season_assessment.py
├── all_season/
│   └── paper_results/
└── per_season/
    └── paper_results/
```

The assessment scripts load the trained models and the corresponding test datasets, generate streamflow predictions, and compare them with the observations (`y_test`).

They calculate deterministic metrics, including Pearson correlation (`r`), its p-value, and root mean square error (RMSE), as well as the probabilistic continuous ranked probability score (CRPS). The centred RMSE (`ubRMSE`) is also retained in the assessment outputs where required by the subsequent analyses.

The resulting CSV files are saved in the corresponding `paper_results/` directory under `assessment/all_season/` or `assessment/per_season/`.

## 4. Visualization

The `visualization/` directory contains the scripts used to generate the paper figures, organized into subdirectories `fig01/` through `fig10/`.

Each figure directory contains its respective plotting code and, where needed, the intermediate data or supporting files required to reproduce the figure. The Figure 10 case-study data are stored separately in `raw_data/per_season_model/stations_fig10_data/`.

## 5. Repository structure

The GitHub repository contains the code, scripts, assessment results, and visualization files. The `raw_data/` and `model/` directories are distributed separately through Zenodo and are therefore not included in the GitHub repository.

```text
cerise/
├── assessment/
│   ├── all_season/
│   │   └── paper_results/
│   ├── per_season/
│   │   └── paper_results/
│   └── scripts/
│       ├── all_season_assessment.py
│       └── per_season_assessment.py
├── model_train_scripts/
│   ├── all_season_model.py
│   └── per_season_model.py
├── visualization/
│   ├── fig01/
│   ├── fig02/
│   ├── fig03/
│   ├── fig04/
│   ├── fig05/
│   ├── fig06/
│   ├── fig07/
│   ├── fig08/
│   ├── fig09/
│   └── fig10/
├── .gitignore
├── requirements.txt
└── README.md
```

After downloading and extracting the Zenodo archives, the complete project structure also includes:

```text
cerise/
├── raw_data/
│   ├── all_season_model/
│   └── per_season_model/
└── model/
    ├── all_season_model/
    └── per_season_model/
```

## 6. Notes

- Keep the filenames and directory structure of the downloaded Zenodo archives unchanged, as the scripts rely on this structure to locate the input data and trained models.
- The scripts automatically identify the repository root and construct the corresponding paths. No manual path configuration is required.
- The input data and trained models are distributed through the project's Zenodo archive, while the source code and assessment results are maintained on GitHub.