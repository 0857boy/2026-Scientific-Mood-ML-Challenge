import pandas as pd
import scipy.io as sio


def matlab_to_datetime(matlab_datenum):
    """
    將 MATLAB 的 datenum (從 0000-01-00 算起的天數)
    轉換為 Python 的 datetime (UNIX epoch 1970-01-01)
    """
    # MATLAB 的 719529 對應到 1970-01-01
    return pd.to_datetime(matlab_datenum.flatten() - 719529, unit="D")


def main():
    print("1. 讀取測站名稱...")
    stations_df = pd.read_csv("Seed_Coastal_Stations.txt")
    station_names = stations_df["station_name"].tolist()
    print(f"找到 {len(station_names)} 個測站。")

    print("\n2. 讀取測站閾值...")
    thresh_mat = sio.loadmat("Seed_Coastal_Stations_Thresholds.mat")
    # 根據您的輸出，閾值的變數名稱是 'thminor_stnd'
    thresholds = thresh_mat["thminor_stnd"].flatten()

    # 建立測站名稱與閾值的對應字典
    thresh_dict = dict(zip(station_names, thresholds))

    print("\n3. 讀取歷史水位資料 (這可能需要幾秒鐘)...")
    data_mat = sio.loadmat("NEUSTG_19502020_12stations.mat")

    # 根據您的輸出，明確指定時間和水位的變數名稱
    time_key = "t"
    sl_key = "sltg"

    raw_time = data_mat[time_key]
    raw_sea_level = data_mat[sl_key]  # 預期形狀: (時間長度, 12 個測站)

    print("轉換 MATLAB 時間格式...")
    times = matlab_to_datetime(raw_time)

    print("\n4. 組合資料並轉為 DataFrame...")
    all_records = []

    for i, station in enumerate(station_names):
        print(f"  處理測站: {station}")
        # 提取該測站的水位 (假設矩陣的欄位順序與 txt 檔一致)
        sl_values = raw_sea_level[:, i]

        df_st = pd.DataFrame(
            {
                "time": times,
                "station_name": station,
                "sea_level": sl_values,
                "threshold": thresh_dict[station],
            }
        )
        all_records.append(df_st)

    final_df = pd.concat(all_records, ignore_index=True)

    print("\n5. 儲存為 hourly_data.parquet...")
    final_df.to_parquet("hourly_data.parquet", engine="pyarrow", index=False)
    print("轉換大功告成！🎉 您現在可以使用 hourly_data.parquet 來訓練模型了！")


if __name__ == "__main__":
    main()
