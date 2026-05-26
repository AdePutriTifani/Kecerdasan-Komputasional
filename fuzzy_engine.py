"""
fuzzy_engine.py
Implementasi Fuzzy Logic untuk menginterpretasi output numerik ANN (nilai PM2.5)
menjadi kategori kualitas udara yang mudah dipahami.

Kategori berdasarkan standar ISPU Indonesia:
  BAIK        : 0   – 50   µg/m³
  SEDANG      : 51  – 100  µg/m³
  TIDAK SEHAT : 101 – 150  µg/m³
  BERBAHAYA   : > 150      µg/m³

Referensi:
    Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338–353.
"""

import numpy as np


# ─────────────────────────────────────────────
# Membership Functions (Trapezoidal & Triangular)
# ─────────────────────────────────────────────

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    Naik dari a ke b, plateau dari b ke c, turun dari c ke d.
    """
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # c < x < d
        return (d - x) / (d - c)


def _triangle(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    Naik dari a ke b (puncak), turun dari b ke c.
    """
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


# ─────────────────────────────────────────────
# Derajat Keanggotaan per Kategori
# ─────────────────────────────────────────────

def membership_baik(pm25: float) -> float:
    """BAIK: 0–50 µg/m³ (trapesium kiri)."""
    return _trapezoid(pm25, -1, 0, 40, 55)


def membership_sedang(pm25: float) -> float:
    """SEDANG: 51–100 µg/m³ (segitiga)."""
    return _triangle(pm25, 40, 75, 110)


def membership_tidak_sehat(pm25: float) -> float:
    """TIDAK SEHAT: 101–150 µg/m³ (segitiga)."""
    return _triangle(pm25, 90, 125, 165)


def membership_berbahaya(pm25: float) -> float:
    """BERBAHAYA: >150 µg/m³ (trapesium kanan)."""
    return _trapezoid(pm25, 140, 155, 9999, 10000)


# ─────────────────────────────────────────────
# Konfigurasi Kategori
# ─────────────────────────────────────────────

CATEGORY_CONFIG = {
    "BAIK": {
        "fn":    membership_baik,
        "color": "#2ecc71",       # hijau
        "emoji": "🟢",
        "saran": "Aman untuk semua aktivitas luar ruangan. Nikmati udara segar!",
        "badge": "success",
    },
    "SEDANG": {
        "fn":    membership_sedang,
        "color": "#f1c40f",       # kuning
        "emoji": "🟡",
        "saran": (
            "Kelompok sensitif (anak-anak, lansia, penderita asma) sebaiknya "
            "membatasi aktivitas luar ruangan yang berat."
        ),
        "badge": "warning",
    },
    "TIDAK SEHAT": {
        "fn":    membership_tidak_sehat,
        "color": "#e67e22",       # oranye
        "emoji": "🟠",
        "saran": (
            "Kurangi aktivitas luar ruangan. Gunakan masker N95 jika harus keluar. "
            "Tutup jendela dan nyalakan penyaring udara."
        ),
        "badge": "danger",
    },
    "BERBAHAYA": {
        "fn":    membership_berbahaya,
        "color": "#e74c3c",       # merah
        "emoji": "🔴",
        "saran": (
            "HINDARI semua aktivitas luar ruangan! Tetap di dalam ruangan, "
            "gunakan pemurni udara, dan pantau kondisi kesehatan."
        ),
        "badge": "critical",
    },
}


# ─────────────────────────────────────────────
# Inferensi Fuzzy (Mamdani sederhana)
# ─────────────────────────────────────────────

def infer_category(pm25: float) -> dict:
    """
    Inferensi fuzzy: hitung derajat keanggotaan setiap kategori,
    lalu tentukan kategori dengan derajat tertinggi.

    Parameters
    ----------
    pm25 : float
        Nilai PM2.5 hasil prediksi ANN (µg/m³).

    Returns
    -------
    dict dengan keys:
        - category   : str  — label kategori terpilih
        - color      : str  — hex color
        - emoji      : str
        - saran      : str  — saran aktivitas
        - memberships: dict — derajat keanggotaan semua kategori
        - pm25       : float
    """
    pm25 = max(0.0, float(pm25))

    memberships = {
        cat: cfg["fn"](pm25) for cat, cfg in CATEGORY_CONFIG.items()
    }

    # Defuzzifikasi: pilih kategori dengan derajat keanggotaan tertinggi
    best_cat = max(memberships, key=memberships.get)

    # Jika semua nol (tidak ada yang cocok), fallback ke threshold sederhana
    if memberships[best_cat] == 0.0:
        if pm25 <= 50:
            best_cat = "BAIK"
        elif pm25 <= 100:
            best_cat = "SEDANG"
        elif pm25 <= 150:
            best_cat = "TIDAK SEHAT"
        else:
            best_cat = "BERBAHAYA"

    cfg = CATEGORY_CONFIG[best_cat]
    return {
        "pm25":        pm25,
        "category":    best_cat,
        "color":       cfg["color"],
        "emoji":       cfg["emoji"],
        "saran":       cfg["saran"],
        "badge":       cfg["badge"],
        "memberships": memberships,
    }


def get_all_memberships_curve(pm25_range=None) -> dict:
    """
    Hitung kurva membership function untuk semua kategori.
    Berguna untuk visualisasi.

    Returns
    -------
    dict: {"x": array, "BAIK": array, "SEDANG": array, ...}
    """
    if pm25_range is None:
        pm25_range = np.linspace(0, 250, 500)

    result = {"x": pm25_range}
    for cat, cfg in CATEGORY_CONFIG.items():
        result[cat] = np.array([cfg["fn"](v) for v in pm25_range])
    return result


if __name__ == "__main__":
    test_values = [10, 60, 120, 200]
    print("Demo Fuzzy Engine — SiKuDara\n")
    for val in test_values:
        res = infer_category(val)
        print(
            f"  PM2.5 = {val:>5.1f} µg/m³  →  {res['emoji']} {res['category']}"
        )
        for cat, deg in res["memberships"].items():
            print(f"      {cat:<14}: {deg:.3f}")
        print()
