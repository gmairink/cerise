# Machine-learning post-processing of global seasonal streamflow forecasts from Météo-France System 9

This repository contains the code and data used to train, evaluate, and visualize the XGBoost-based post-processing models developed for the study *"Machine-learning post-processing of global seasonal streamflow forecasts from Météo-France System 9"*.

The workflow is divided into three main stages:

1. **Model training**: train the all-season and per-season XGBoost models.
2. **Model assessment**: evaluate the trained models using deterministic and probabilistic metrics.
3. **Visualization**: generate the figures and supporting analyses presented in the study.

## 1. Installation and data setup

### Python dependencies

The `requirements.txt` file lists the Python packages required to run the framework. From the repository root, install them with:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended to keep the project dependencies separate from other Python environments.

### Input data

> **ATTENTION:** The input datasets are not included in this repository. You must download them from the project's Zenodo archive before running the scripts and place them in the corresponding directories within `raw_data/`.

**The input datasets are available from the project's Zenodo archive: [Zenodo repository link]**.

Download the data archive and place its contents in the repository's `raw_data/` directory. Preserve the directory structure and filenames, since the training and assessment scripts use these paths to locate the input files.

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

The per-season datasets are stored separately for each initialization month. The month identifies the forecast season as follows:

| Initialization month | Period            |
| -------------------- | ----------------- |
| 2                    | February–May      |
| 5                    | May–August        |
| 8                    | August–November   |
| 11                   | November–February |

For example, `per_season_X_train_im_2.pkl` and `per_season_y_train_im_2.pkl` contain the predictors and observed targets used to train the February-initialized model. The same naming convention applies to the other initialization months and to the test datasets and station-group files.

The `per_season_model/stations_fig10_data/` directory contains the data prepared for the selected-station case studies presented in Figure 10.


## 2. Model training

The model-training scripts are located in `model_train_scripts/`:

- `all_season_model.py`: trains the all-season XGBoost model.
- `per_season_model.py`: trains separate XGBoost models for the four initialization months (2, 5, 8, and 11).

The scripts load the corresponding training datasets from `raw_data/`. They perform hyperparameter optimization and save the Optuna study, the selected best parameters, and the trained model.

Model artifacts are organized by modelling configuration:

```text
model/
├── all_season_model/
│   └── paper_model/
└── per_season_model/
    └── paper_model/
```

The `paper_model/` directories contain the model files used for the analyses reported in the study.

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

The assessment scripts load the trained models and the corresponding test datasets, generate streamflow predictions, and compare them with the observations (`y_test`). They calculate deterministic metrics, including Pearson correlation (`r`), its p-value, and root mean square error (RMSE), as well as the probabilistic continuous ranked probability score (CRPS). The centred RMSE (`ub RMSE`) is also retained in the assessment outputs where required by the subsequent analyses.

The resulting CSV files are saved in the corresponding `paper_results/` directory under `assessment/all_season/` or `assessment/per_season/`.

## 4. Visualization

The `visualization/` directory contains the scripts used to generate the paper figures, organized into subdirectories `fig01/` through `fig10/`.

Each figure directory contains its respective plotting code and, where needed, the intermediate data or supporting files required to reproduce the figure. The Figure 10 case-study data are stored separately in `raw_data/per_season_model/stations_fig10_data/`.

## 5. Repository structure

The main project structure is summarized below. Individual files inside the figure directories are omitted for clarity.

```text
cerise_github/
├── assessment/
│   ├── all_season/
│   │   └── paper_results/
│   ├── per_season/
│   │   └── paper_results/
│   └── scripts/
│       ├── all_season_assessment.py
│       └── per_season_assessment.py
├── model/
│   ├── all_season_model/
│   │   └── paper_model/
│   └── per_season_model/
│       └── paper_model/
├── model_train_scripts/
│   ├── all_season_model.py
│   └── per_season_model.py
├── raw_data/
│   ├── all_season_model/
│   │   ├── all_season_groups_train.pkl
│   │   ├── all_season_groups_test.pkl
│   │   ├── all_season_X_train.pkl
│   │   ├── all_season_X_test.pkl
│   │   ├── all_season_y_train.pkl
│   │   └── all_season_y_test.pkl
│   └── per_season_model/
│       ├── stations_fig10_data/
│       ├── per_season_groups_train_im_*.pkl
│       ├── per_season_groups_test_im_*.pkl
│       ├── per_season_X_train_im_*.pkl
│       ├── per_season_X_test_im_*.pkl
│       ├── per_season_y_train_im_*.pkl
│       └── per_season_y_test_im_*.pkl
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
├── requirements.txt
└── README.md
```

## 6. Notes

- Keep the filenames and directory structure of the downloaded input data unchanged, as the scripts rely on this structure to locate the datasets.
- The scripts automatically identify the repository root and construct the corresponding paths. No manual path configuration is required.
