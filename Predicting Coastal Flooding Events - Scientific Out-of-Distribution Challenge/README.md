# Predicting Coastal Flooding Events - Scientific Out-of-Distribution Challenge

This repository contains sample training code and submissions for the Coastal Flooding Events challenge. It is designed to give participants a reference for working on the challenge, including generating training data, training the model, and running local evaluations.

## Repository Structure
```
Predicting Coastal Flooding Events - Scientific Out-of-Distribution Challenge/
│  ├── submissions/
│  │   └── xgboost_regression/
│  │       ├── best_offset.txt
│  │       ├── model.py
│  │       ├── xgb_reg_model.json
│  │       └── requirements.txt
│  └── training/
│      └── xgboost_regression/
│           ├── convert_to_parquet.py
│           ├── train_xgb_regression.py
│           ├── NEUSTG_19502020_12stations.mat
│           ├── Seed_Coastal_Stations_Thresholds.mat
│           ├── Seed_Coastal_Stations.txt
│           ├── Seed_Historical_Time_Intervals.txt
│           └── requirements.txt
└── README.md
└── LICENSE
```
## Installation & Running (for Training)
### Installation
To run this code, first create a fresh environment, then install the requirements file:
```bash
pip install -r training/xgboost_regression/requirements.txt
```
(Note: The environment requires packages such as `numpy`, `pandas`, `xgboost`, `scipy`, and `scikit-learn`.)

### Step 1: Generating Training Data
The raw historical data is provided in `.mat` and `.txt` formats. Before training, you need to convert these into a standardized `hourly_data.parquet` file.

Navigate to the training directory and run the conversion script:
```bash
cd training/xgboost_regression/
python convert_to_parquet.py
```
This script will parse `NEUSTG_19502020_12stations.mat`, `Seed_Coastal_Stations.txt`, and `Seed_Coastal_Stations_Thresholds.mat` to generate the consolidated Parquet file required for training.

### Step 2: Training
An example training run can be executed by running the following:
```bash
python train_xgb_regression.py
```
This script reads `hourly_data.parquet`, standardizes the sea level and threshold data, and trains an `XGBRegressor` using a 168-hour input window.

Upon completion, it will output two crucial files needed for your submission:
1. `xgb_reg_model.json`: The trained model weights.
2. `best_offset.txt`: The optimal margin offset calculated to maximize the MCC metric.


### Step 3: Evaluation
After training, you can locally evaluate your model by running the following:

First, ensure `xgb_reg_model.json` and `best_offset.txt` are copied into the `submissions/xgboost_regression/` directory. Then, execute the inference script:
```bash
cd ../../submissions/xgboost_regression/
python model.py --test_hourly <path_to_test_hourly.csv> --test_index <path_to_test_index.csv> --predictions_out submission.csv
```
The `model.py` script will load the XGBoost model, apply the `best_offset.txt` to the predicted margins, and convert them into final flood probabilities.
