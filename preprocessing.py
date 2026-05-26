"""
preprocessing.py
Modul pembersihan dan normalisasi data kualitas udara untuk SiKuDara.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Fitur input dan target output
FEATURE_COLS = ["TEMP", "PRES", "DEWP", "WSPM", "SO2", "NO2", "PM10"]
TARGET_COL   = "PM2.5"

DATA_PATH    = os.path.join("data", "air_quality.csv")
SCALER_PATH  = os.path.join("model", "scaler.pkl")


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load dataset CSV. Jika tidak ada, buat data sintetis secara otomatis.
    """
    if not os.path.exists(path):
        print(f"[!] Dataset tidak ditemukan di '{path}'.")
        print("    Membuat dataset sintetis secara otomatis...")
        from data.generate_sample_data import generate_sample_data
        os.makedirs("data", exist_ok=True)
        df = generate_sample_data(n_samples=5000)
        df.to_csv(path, index=False)
        print(f"[OK] Dataset sintetis disimpan ke '{path}'")
    else:
        df = pd.read_csv(path)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data:
    - Pilih kolom yang relevan
    - Hapus baris dengan nilai NaN
    - Hapus outlier ekstrem (nilai negatif pada polutan)
    """
    required_cols = FEATURE_COLS + [TARGET_COL]
    available = [c for c in required_cols if c in df.columns]
    df = df[available].copy()

    # Drop NaN
    df.dropna(inplace=True)

    # Drop baris dengan polutan negatif
    for col in ["SO2", "NO2", "PM10", "PM2.5"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Clip PM2.5 maksimum 500 (outlier ekstrem)
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] <= 500]

    df.reset_index(drop=True, inplace=True)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Pisahkan fitur (X) dan target (y), lalu normalisasi.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Normalisasi fitur (MinMax)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Simpan scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_scaler():
    """Load scaler yang sudah disimpan."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler tidak ditemukan di '{SCALER_PATH}'. "
            "Jalankan train_model.py terlebih dahulu."
        )
    return joblib.load(SCALER_PATH)


def get_prepared_data():
    """
    Pipeline lengkap: load → clean → prepare.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, df_clean
    """
    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    print(f"[DATA] Dataset: {len(df_clean)} baris setelah pembersihan")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_features(df_clean)
    print(f"       Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, df_clean


if __name__ == "__main__":
    get_prepared_data()
    print("✅ Preprocessing selesai.")
