"""
train_model.py
Training ANN menggunakan PSO sebagai optimizer bobot.

Arsitektur ANN: Input(7) -> Hidden1(16) -> Hidden2(8) -> Output(1)
Optimizer     : PSO (Particle Swarm Optimization)
Loss          : Mean Squared Error (MSE)

Cara penggunaan:
    python train_model.py
"""

import os
import sys
import time
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import get_prepared_data
from pso_optimizer import PSOOptimizer, count_ann_weights

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# Konfigurasi Arsitektur ANN
# ─────────────────────────────────────────────
LAYER_SIZES  = [7, 16, 8, 1]   # Input -> Hidden1 -> Hidden2 -> Output
MODEL_PATH   = os.path.join("model", "ann_model.pkl")
METRICS_PATH = os.path.join("model", "metrics.pkl")

# Konfigurasi PSO
PSO_CONFIG = {
    "n_particles": 30,
    "max_iter":    100,
    "w":           0.7,
    "c1":          1.5,
    "c2":          1.5,
    "bounds":      (-5.0, 5.0),
}


# ─────────────────────────────────────────────
# ANN Forward Pass
# ─────────────────────────────────────────────

def _relu(x):
    return np.maximum(0, x)


def decode_weights(flat_weights, layer_sizes):
    """Dekode vektor bobot flat menjadi list (W, b) per layer."""
    layers = []
    idx = 0
    for i in range(len(layer_sizes) - 1):
        rows = layer_sizes[i]
        cols = layer_sizes[i + 1]
        n_w  = rows * cols
        W = flat_weights[idx : idx + n_w].reshape(rows, cols)
        idx += n_w
        b = flat_weights[idx : idx + cols]
        idx += cols
        layers.append((W, b))
    return layers


def ann_predict(X, flat_weights, layer_sizes):
    """Forward pass ANN dengan aktivasi ReLU (hidden) dan linear (output)."""
    layers = decode_weights(flat_weights, layer_sizes)
    out = X
    for i, (W, b) in enumerate(layers):
        out = out @ W + b
        if i < len(layers) - 1:
            out = _relu(out)
    return out.flatten()


def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# ─────────────────────────────────────────────
# Fungsi Fitness untuk PSO
# ─────────────────────────────────────────────

def make_fitness_fn(X_val, y_val, layer_sizes):
    """Buat fungsi fitness (closure) atas data validasi."""
    def fitness(flat_weights):
        y_pred = ann_predict(X_val, flat_weights, layer_sizes)
        return mse_loss(y_val, y_pred)
    return fitness


# ─────────────────────────────────────────────
# Training Utama
# ─────────────────────────────────────────────

class ANNModel:
    """Wrapper ANN sederhana agar mudah di-save/load dengan joblib."""

    def __init__(self, flat_weights, layer_sizes, scaler):
        self.flat_weights = flat_weights
        self.layer_sizes  = layer_sizes
        self.scaler       = scaler

    def predict(self, X_raw):
        """Prediksi dari data mentah (belum dinormalisasi)."""
        X_scaled = self.scaler.transform(X_raw)
        return ann_predict(X_scaled, self.flat_weights, self.layer_sizes)

    def predict_single(self, feature_dict):
        """
        Prediksi satu sampel dari dictionary fitur.
        Keys: TEMP, PRES, DEWP, WSPM, SO2, NO2, PM10
        """
        from preprocessing import FEATURE_COLS
        X = np.array([[feature_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
        return float(self.predict(X)[0])


def train():
    print("=" * 60)
    print("  SiKuDara - Training ANN dengan PSO")
    print("=" * 60)

    # 1. Persiapkan data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, _ = get_prepared_data()

    # 2. Hitung dimensi bobot
    n_dims = count_ann_weights(LAYER_SIZES)
    arch = " -> ".join(str(s) for s in LAYER_SIZES)
    print(f"\n[ANN] Arsitektur     : {arch}")
    print(f"      Total parameter: {n_dims}")
    print(f"      Train samples  : {len(X_train)}")
    print(f"      Val samples    : {len(X_val)}")

    # 3. Buat fungsi fitness
    fitness_fn = make_fitness_fn(X_val, y_val, LAYER_SIZES)

    # 4. Jalankan PSO
    print(f"\n[PSO] Memulai PSO ({PSO_CONFIG['n_particles']} partikel, "
          f"{PSO_CONFIG['max_iter']} iterasi)...\n")

    t0  = time.time()
    pso = PSOOptimizer(
        n_particles = PSO_CONFIG["n_particles"],
        n_dims      = n_dims,
        fitness_fn  = fitness_fn,
        w           = PSO_CONFIG["w"],
        c1          = PSO_CONFIG["c1"],
        c2          = PSO_CONFIG["c2"],
        max_iter    = PSO_CONFIG["max_iter"],
        bounds      = PSO_CONFIG["bounds"],
        verbose     = True,
    )
    best_weights, best_val_mse = pso.optimize()
    elapsed = time.time() - t0

    print(f"\n[INFO] Waktu training: {elapsed:.1f} detik")
    print(f"       Best Val MSE  : {best_val_mse:.4f}")

    # 5. Evaluasi di test set
    y_pred_test = ann_predict(X_test, best_weights, LAYER_SIZES)
    y_pred_test = np.clip(y_pred_test, 0, None)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2   = r2_score(y_test, y_pred_test)

    print("\n[EVAL] Evaluasi pada Test Set:")
    print(f"       MAE  : {mae:.4f} ug/m3")
    print(f"       RMSE : {rmse:.4f} ug/m3")
    print(f"       R2   : {r2:.4f}")

    # 6. Simpan model
    os.makedirs("model", exist_ok=True)
    model = ANNModel(
        flat_weights = best_weights,
        layer_sizes  = LAYER_SIZES,
        scaler       = scaler,
    )
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "val_mse":     best_val_mse,
        "pso_history": pso.history,
        "elapsed_sec": elapsed,
        "layer_sizes": LAYER_SIZES,
        "n_dims":      n_dims,
        "y_test":      y_test,
        "y_pred_test": y_pred_test,
    }
    joblib.dump(metrics, METRICS_PATH)

    print(f"\n[OK] Model disimpan  : {MODEL_PATH}")
    print(f"     Metrik disimpan : {METRICS_PATH}")
    print("=" * 60)

    return model, metrics


def load_model():
    """Load model yang sudah dilatih."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{MODEL_PATH}'. "
            "Jalankan train_model.py terlebih dahulu."
        )
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load metrik evaluasi."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrik tidak ditemukan di '{METRICS_PATH}'. "
            "Jalankan train_model.py terlebih dahulu."
        )
    return joblib.load(METRICS_PATH)


if __name__ == "__main__":
    train()
