# Coastal Flood Prediction - XGBoost Regression Approach

## 🌊 Overview
This submission adopts a **Regression** approach to solve the coastal flood prediction problem. Unlike traditional binary classification models (0 or 1), this XGBoost model predicts the specific continuous "margin" (how much the maximum sea level will exceed the threshold) over the next 14 days. By using an optimal offset discovered during validation, this margin is converted into a final flood probability. This method better preserves the continuous physical characteristics of water levels.

## 🛠️ Feature Engineering
The input uses a time series window of 168 hours, extracting 14 core features for each time step:

1. **Base Normalized Features**: `sea_level` and `threshold`.
2. **Physical/Cyclical Features (Crucial)**:
   - **Solar Day (24h)**: `sin`, `cos`
   - **Lunar Day / Tide (12.42h)**: `sin`, `cos` (Effectively helps the model understand tidal rise and fall patterns)
   - **Seasonality (366d)**: `sin`, `cos`
3. **Statistical Features**:
   - `diff`: First-order difference of the sequence
   - `rolling_mean`: Moving average
   - `rolling_std`: Moving standard deviation
   - `dist`: Distance between current sea level and the threshold
4. **Lag Features**: 
   - Sea level 24 hours ago (`lag_24`) and 25 hours ago (`lag_25`).

## 🧠 Model Mechanism
1. **Model**: `XGBRegressor` (trained using the `reg:squarederror` objective function).
2. **Inference**: The model outputs a continuous `Margin` value (Predicted Sea Level - Threshold).
3. **Probability Mapping**: It reads the pre-calculated optimal offset. If the predicted margin is greater than the offset, it indicates a high risk of flooding. A custom Sigmoid function `1 / (1 + exp(-k * (margin - offset)))` is applied to smoothly map this value to a 0~1 probability range.

## 📂 File Structure
* `model.py`: The main inference script for the test set, containing feature extraction, model loading, and batch inference logic.
* `xgb_reg_model.json`: The trained XGBoost regression model weights.
* `best_offset.txt`: A text file storing the optimal decision offset (e.g., `-0.10000000000000009`). It must be placed in the same directory as the script during inference.
* `requirements.txt`: The list of Python dependencies required to run the environment.

## 🚀 How to Run

Please ensure all dependencies are installed:
```bash
pip install -r requirements.txt