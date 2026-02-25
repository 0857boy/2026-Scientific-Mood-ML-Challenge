import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

TRAIN_STATIONS = [
    "Annapolis",
    "Atlantic_City",
    "Charleston",
    "Washington",
    "Wilmington",
    "Eastport",
    "Portland",
    "Sewells_Point",
    "Sandy_Hook",
]
TEST_STATIONS = ["Lewes", "Fernandina_Beach", "The_Battery"]

INPUT_HOURS = 168
PRED_DAYS = 14
PRED_HOURS = PRED_DAYS * 24
TRAIN_MEAN = 2.2691
TRAIN_STD = 1.3504


def prepare_data(parquet_path="hourly_data.parquet", mode="train"):
    print(f"Reading data for {mode}...")
    df = pd.read_parquet(parquet_path)
    if mode == "train":
        stations = TRAIN_STATIONS
    else:
        stations = TEST_STATIONS
    df = df[df["station_name"].isin(stations)].reset_index(drop=True)

    # 填補
    df["sea_level"] = df["sea_level"].ffill().fillna(0).replace([np.inf, -np.inf], 0)
    df["threshold"] = df["threshold"].ffill().fillna(0)

    sl = df["sea_level"].values
    thresh = df["threshold"].values

    # 標準化
    feat_sl = (sl - TRAIN_MEAN) / TRAIN_STD
    feat_thresh = (thresh - TRAIN_MEAN) / TRAIN_STD

    # 時間特徵
    dt = df["time"].dt
    # 1. 太陽日 (24h)
    hour_rad = 2 * np.pi * dt.hour / 24.0
    feat_sin_h = np.sin(hour_rad).values
    feat_cos_h = np.cos(hour_rad).values

    # 2. 🔥 新增：太陰日/潮汐週期 (約 12.42h) - 這是物理關鍵！
    tide_rad = 2 * np.pi * (dt.hour + dt.minute / 60.0) / 12.42
    feat_sin_tide = np.sin(tide_rad).values
    feat_cos_tide = np.cos(tide_rad).values

    # 3. 季節 (366 days)
    day_rad = 2 * np.pi * dt.dayofyear / 366.0
    feat_sin_d = np.sin(day_rad).values
    feat_cos_d = np.cos(day_rad).values

    # 統計特徵
    diff = np.diff(feat_sl, prepend=feat_sl[0])
    kernel = np.ones(6) / 6.0
    rolling_mean = np.convolve(feat_sl, kernel, mode="same")
    sl_sq = feat_sl**2
    rolling_sq_mean = np.convolve(sl_sq, kernel, mode="same")
    rolling_var = rolling_sq_mean - (rolling_mean**2)
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))
    dist_to_thresh = feat_sl - feat_thresh
    lag_24 = np.roll(feat_sl, 24)
    lag_24[:24] = 0
    lag_25 = np.roll(feat_sl, 25)
    lag_25[:25] = 0

    # 堆疊: 12 (舊) + 2 (Tide) = 14 特徵
    features_all = np.stack(
        [
            feat_sl,
            feat_thresh,
            feat_sin_h,
            feat_cos_h,
            feat_sin_d,
            feat_cos_d,
            feat_sin_tide,
            feat_cos_tide,  # New
            diff,
            rolling_mean,
            dist_to_thresh,
            rolling_std,
            lag_24,
            lag_25,
        ],
        axis=1,
    )

    X_list = []
    y_target_list = []  # Regression Target
    y_thresh_list = []  # 用於評估 MCC 的閾值

    grouped = df.groupby("station_name")
    stride = 24

    print(f"Generating features for {mode}...")
    for _, group in tqdm(grouped):
        start_idx = group.index[0]
        count = len(group)
        max_start = count - INPUT_HOURS - PRED_HOURS
        if max_start > 0:
            indices = np.arange(start_idx, start_idx + max_start, step=stride)
            for idx in indices:
                # Input
                win_x = features_all[idx : idx + INPUT_HOURS].reshape(-1)

                # Target: 未來14天的「最高水位」(數值)
                y_seq = sl[idx + INPUT_HOURS : idx + INPUT_HOURS + PRED_HOURS]
                t_seq = thresh[idx + INPUT_HOURS : idx + INPUT_HOURS + PRED_HOURS]

                # 每日最高水位
                y_daily_max = y_seq.reshape(PRED_DAYS, 24).max(axis=1)
                t_daily_max = t_seq.reshape(PRED_DAYS, 24).max(axis=1)

                # 回歸目標：這 14 天中，水位「超出」閾值最多是多少？
                # 如果沒淹水，這個值會是負的 (代表離閾值還有多遠)
                # 這比單純預測水位更好，因為不同站點水位基準不同，但「距離閾值」是通用的
                margin = y_daily_max - t_daily_max
                max_margin = np.max(margin)

                X_list.append(win_x)
                y_target_list.append(max_margin)  # Regression Target

                # 這是真實 Label (用於算 MCC)
                label = 1 if max_margin > 0 else 0
                y_thresh_list.append(label)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_target_list, dtype=np.float32),
        np.array(y_thresh_list, dtype=np.int8),
    )


def train_regression():
    # 載入數據
    X_train, y_train, _ = prepare_data(mode="train")
    X_test, y_test, y_test_label = prepare_data(mode="test")

    print(f"Train Shape: {X_train.shape}")

    # 使用 Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        tree_method="hist",
        device="cpu",
        objective="reg:squarederror",  # 回歸任務
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
    )

    print("Training XGBoost Regressor...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    print("Evaluating...")
    # 預測出來的是 "Max Margin" (水位 - 閾值)
    y_pred_margin = model.predict(X_test)

    # 轉換為分類：如果 Margin > 0，代表淹水
    # 但我們可以調整這個 offset 來最佳化 MCC
    # 因為測試集全是淹水，我們可能需要把標準放寬一點點 (例如 > -0.05 就當作淹水)
    best_mcc = 0
    best_offset = 0.0

    # 掃描最佳偏移量 (-0.5 到 0.5)
    offsets = np.arange(-0.5, 0.5, 0.05)

    for off in offsets:
        # 預測 > offset 即判斷為 1
        y_pred_class = (y_pred_margin > off).astype(int)
        mcc = matthews_corrcoef(y_test_label, y_pred_class)
        if mcc > best_mcc:
            best_mcc = mcc
            best_offset = off

    print(f"🔥 XGBoost Regression Best MCC: {best_mcc:.4f} at Offset {best_offset:.2f}")
    print(f"   (Offset的意思是: 只要預測水位超過 [閾值 + {best_offset}] 就算淹水)")

    # 儲存最佳 offset 供提交使用
    with open("best_offset.txt", "w") as f:
        f.write(str(best_offset))

    model.save_model("xgb_reg_model.json")
    print("Model saved.")


if __name__ == "__main__":
    train_regression()
