"""
Coastal Flood Prediction - XGBoost Regression Submission
"""
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import os

TRAIN_MEAN = 2.2691
TRAIN_STD = 1.3504

def get_flattened_window(df, start_time, input_hours=168):
    start_ts = pd.to_datetime(start_time)
    times = df['time'].values
    idx = np.searchsorted(times, start_ts.to_datetime64())
    
    if idx + input_hours > len(df) or times[idx] != start_ts.to_datetime64():
        return None
        
    sub_df = df.iloc[idx : idx + input_hours].copy()
    
    # 預處理
    sub_df['sea_level'] = sub_df['sea_level'].ffill().fillna(0).replace([np.inf, -np.inf], 0)
    if 'threshold' not in sub_df.columns:
         st_mean = sub_df['sea_level'].mean()
         st_std = sub_df['sea_level'].std()
         sub_df['threshold'] = st_mean + 1.5 * st_std
    else:
         sub_df['threshold'] = sub_df['threshold'].ffill().fillna(0)
    
    sl = sub_df['sea_level'].values.astype(np.float32)
    thresh = sub_df['threshold'].values.astype(np.float32)
    
    feat_sl = (sl - TRAIN_MEAN) / TRAIN_STD
    feat_thresh = (thresh - TRAIN_MEAN) / TRAIN_STD
    
    dt = sub_df['time'].dt
    # 1. 24h
    hour_rad = 2 * np.pi * dt.hour / 24.0
    feat_sin_h = np.sin(hour_rad).values.astype(np.float32)
    feat_cos_h = np.cos(hour_rad).values.astype(np.float32)
    
    # 2. 12.42h (Tide)
    tide_rad = 2 * np.pi * (dt.hour + dt.minute / 60.0) / 12.42
    feat_sin_t = np.sin(tide_rad).values.astype(np.float32)
    feat_cos_t = np.cos(tide_rad).values.astype(np.float32)
    
    # 3. 366d
    day_rad = 2 * np.pi * dt.dayofyear / 366.0
    feat_sin_d = np.sin(day_rad).values.astype(np.float32)
    feat_cos_d = np.cos(day_rad).values.astype(np.float32)
    
    diff = np.diff(feat_sl, prepend=feat_sl[0])
    kernel = np.ones(6) / 6.0
    rolling_mean = np.convolve(feat_sl, kernel, mode='same')
    sl_sq = feat_sl ** 2
    rolling_sq_mean = np.convolve(sl_sq, kernel, mode='same')
    rolling_var = rolling_sq_mean - (rolling_mean ** 2)
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))
    dist = feat_sl - feat_thresh
    lag_24 = np.roll(feat_sl, 24); lag_24[:24] = 0
    lag_25 = np.roll(feat_sl, 25); lag_25[:25] = 0
    
    features = np.stack([
        feat_sl, feat_thresh, feat_sin_h, feat_cos_h, feat_sin_d, feat_cos_d,
        feat_sin_t, feat_cos_t, # New Tide Features
        diff, rolling_mean, dist, rolling_std, lag_24, lag_25
    ], axis=1)
    
    return features.reshape(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_hourly", required=True)
    parser.add_argument("--test_index", required=True)
    parser.add_argument("--predictions_out", required=True)
    parser.add_argument("--train_hourly", help="ignored")
    args = parser.parse_args()

    print("Loading XGBoost Regressor...")
    model = xgb.Booster()
    model.load_model("xgb_reg_model.json")
    
    # 讀取 Offset
    try:
        with open("best_offset.txt", "r") as f:
            OFFSET = float(f.read().strip())
        print(f"Loaded Best Offset: {OFFSET}")
    except:
        OFFSET = 0.0
        print("Warning: Offset not found, using 0.0")

    test_df = pd.read_csv(args.test_hourly)
    index_df = pd.read_csv(args.test_index)
    test_df['time'] = pd.to_datetime(test_df['time'])
    station_dfs = {name: group.sort_values('time') for name, group in test_df.groupby('station_name')}
    
    X_batch = []
    valid_indices = []
    
    for i, row in index_df.iterrows():
        st_name = row['station_name']
        start_time = row['hist_start']
        if st_name in station_dfs:
            feats = get_flattened_window(station_dfs[st_name], start_time)
            if feats is not None:
                X_batch.append(feats)
                valid_indices.append(i)

    final_probs = np.zeros(len(index_df))
    
    if len(X_batch) > 0:
        X_np = np.array(X_batch)
        dtest = xgb.DMatrix(X_np)
        
        # 預測 Margin (Predicted Sea Level - Threshold)
        y_margin = model.predict(dtest)
        
        # 轉換為分類機率 (Sigmoid)
        # 邏輯：如果 (margin - offset) > 0，則機率 > 0.5
        # 這裡我們用一個係數 k 來控制 sigmoid 的陡峭程度
        # y_margin - OFFSET 是我們判斷是否淹水的 "信心分數"
        k = 10.0 # 係數越大，機率越兩極化
        y_probs = 1 / (1 + np.exp(-k * (y_margin - OFFSET)))
        
        final_probs[valid_indices] = y_probs
        
    out_df = pd.DataFrame({
        "id": index_df["id"],
        "y_prob": final_probs
    })
    out_df.to_csv(args.predictions_out, index=False)
    print("Done.")

if __name__ == "__main__":
    main()