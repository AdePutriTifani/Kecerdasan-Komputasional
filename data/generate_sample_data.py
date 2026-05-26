"""
generate_sample_data.py
Membuat dataset sintetis untuk SiKuDara jika dataset asli belum tersedia.
Dataset asli: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
"""

import numpy as np
import pandas as pd
import os

def generate_sample_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Membuat data kualitas udara sintetis dengan distribusi realistis.
    Fitur: TEMP, PRES, DEWP, WSPM, SO2, NO2, PM10, PM2.5
    """
    rng = np.random.default_rng(seed)

    # Suhu: -10 s/d 40 °C
    TEMP = rng.uniform(-10, 40, n_samples)
    # Tekanan: 990–1040 hPa
    PRES = rng.uniform(990, 1040, n_samples)
    # Titik embun: selalu < suhu
    DEWP = TEMP - rng.uniform(2, 20, n_samples)
    # Kecepatan angin: 0–10 m/s
    WSPM = rng.exponential(scale=2.0, size=n_samples).clip(0, 10)

    # SO2 dan NO2 (ppb)
    SO2 = rng.exponential(scale=15, size=n_samples).clip(0, 200)
    NO2 = rng.exponential(scale=30, size=n_samples).clip(0, 250)

    # PM10 berkorelasi dengan PM2.5
    PM10 = rng.exponential(scale=60, size=n_samples).clip(0, 500)

    # PM2.5 dipengaruhi secara realistis oleh variabel lain
    PM25 = (
        0.5 * PM10
        + 0.3 * SO2
        + 0.2 * NO2
        - 0.5 * WSPM
        - 0.1 * TEMP
        + rng.normal(0, 10, n_samples)
    ).clip(0, 400)

    df = pd.DataFrame({
        "TEMP": TEMP.round(1),
        "PRES": PRES.round(1),
        "DEWP": DEWP.round(1),
        "WSPM": WSPM.round(2),
        "SO2":  SO2.round(1),
        "NO2":  NO2.round(1),
        "PM10": PM10.round(1),
        "PM2.5": PM25.round(1),
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sample_data(n_samples=5000)
    output_path = os.path.join("data", "air_quality.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset sintetis berhasil dibuat: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   PM2.5 min={df['PM2.5'].min():.1f}, max={df['PM2.5'].max():.1f}, "
          f"mean={df['PM2.5'].mean():.1f}")
