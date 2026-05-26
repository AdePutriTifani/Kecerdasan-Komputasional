"""
pso_optimizer.py
Implementasi Particle Swarm Optimization (PSO) dari scratch menggunakan NumPy.
Digunakan untuk mengoptimalkan bobot ANN (Artificial Neural Network).

Referensi:
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of IEEE International Conference on Neural Networks.
"""

import numpy as np
from typing import Callable, Tuple


class PSOOptimizer:
    """
    Particle Swarm Optimization untuk optimasi bobot ANN.

    Parameters
    ----------
    n_particles : int
        Jumlah partikel dalam swarm.
    n_dims : int
        Dimensi ruang pencarian (= jumlah total bobot ANN).
    fitness_fn : Callable
        Fungsi fitness f(weights) → float (MSE). Akan diminimalkan.
    w : float
        Inertia weight — mengontrol momentum partikel.
    c1 : float
        Cognitive coefficient — tarikan ke posisi terbaik partikel sendiri.
    c2 : float
        Social coefficient — tarikan ke posisi terbaik global.
    max_iter : int
        Jumlah iterasi maksimum.
    bounds : Tuple[float, float]
        Batas bawah dan atas nilai bobot.
    verbose : bool
        Jika True, cetak progress setiap 10 iterasi.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_dims: int = 100,
        fitness_fn: Callable = None,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iter: int = 100,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        verbose: bool = True,
    ):
        self.n_particles = n_particles
        self.n_dims      = n_dims
        self.fitness_fn  = fitness_fn
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.max_iter    = max_iter
        self.bounds      = bounds
        self.verbose     = verbose

        # State partikel
        self._init_swarm()

        # Riwayat fitness per iterasi (untuk visualisasi)
        self.history: list[float] = []

    def _init_swarm(self):
        """Inisialisasi posisi dan kecepatan partikel secara acak."""
        low, high = self.bounds
        self.positions  = np.random.uniform(low, high, (self.n_particles, self.n_dims))
        self.velocities = np.random.uniform(
            -(high - low) * 0.1,
             (high - low) * 0.1,
            (self.n_particles, self.n_dims),
        )

        # Personal best
        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.full(self.n_particles, np.inf)

        # Global best
        self.gbest_pos = np.zeros(self.n_dims)
        self.gbest_fit = np.inf

    def _evaluate_all(self) -> np.ndarray:
        """Evaluasi fungsi fitness untuk semua partikel."""
        fitness = np.array([self.fitness_fn(p) for p in self.positions])
        return fitness

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Jalankan loop PSO.

        Returns
        -------
        gbest_pos : np.ndarray
            Bobot terbaik yang ditemukan.
        gbest_fit : float
            Nilai fitness terbaik (MSE terendah).
        """
        if self.fitness_fn is None:
            raise ValueError("fitness_fn belum diset.")

        for iteration in range(self.max_iter):
            fitness = self._evaluate_all()

            # Update personal best
            improved = fitness < self.pbest_fit
            self.pbest_fit = np.where(improved, fitness, self.pbest_fit)
            self.pbest_pos[improved] = self.positions[improved]

            # Update global best
            best_idx = np.argmin(self.pbest_fit)
            if self.pbest_fit[best_idx] < self.gbest_fit:
                self.gbest_fit = self.pbest_fit[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            self.history.append(self.gbest_fit)

            # Update kecepatan dan posisi
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            cognitive = self.c1 * r1 * (self.pbest_pos - self.positions)
            social    = self.c2 * r2 * (self.gbest_pos - self.positions)

            self.velocities = self.w * self.velocities + cognitive + social

            # Clip kecepatan
            max_v = (self.bounds[1] - self.bounds[0]) * 0.2
            self.velocities = np.clip(self.velocities, -max_v, max_v)

            self.positions += self.velocities

            # Clip posisi ke dalam batas
            self.positions = np.clip(self.positions, *self.bounds)

            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"  Iterasi {iteration + 1:>4}/{self.max_iter} | "
                    f"Best MSE: {self.gbest_fit:.4f}"
                )

        return self.gbest_pos, self.gbest_fit


def count_ann_weights(layer_sizes: list[int]) -> int:
    """
    Hitung total jumlah bobot + bias ANN.

    Parameters
    ----------
    layer_sizes : list[int]
        Ukuran setiap layer, contoh: [7, 16, 8, 1]
        = 7 input → 16 hidden1 → 8 hidden2 → 1 output

    Returns
    -------
    int : Total parameter
    """
    total = 0
    for i in range(len(layer_sizes) - 1):
        # Bobot + bias
        total += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
    return total


if __name__ == "__main__":
    # Demo singkat: minimasi fungsi Sphere f(x) = sum(x^2)
    print("Demo PSO — Minimasi Fungsi Sphere")

    def sphere(x):
        return float(np.sum(x ** 2))

    pso = PSOOptimizer(
        n_particles=20,
        n_dims=5,
        fitness_fn=sphere,
        max_iter=50,
        bounds=(-5.0, 5.0),
    )
    best_pos, best_fit = pso.optimize()
    print(f"\n✅ Best fitness: {best_fit:.6f}")
    print(f"   Best position: {best_pos.round(4)}")
