"""Test 1D del criterio de Cronos–Jeans (sesión de verificación del
autor, 21-sep-2026; reproducido en el repositorio).

Medio homogéneo colisionless de láminas en caja periódica con la
fuerza LOCAL +c²∂_x ε_c(ρ), ε_c = A·ρ^{3/2}, SIN gravedad. Unidades:
σ = 1, ρ₀ = 1, L = 1. Parámetro q ≡ (3/2)·c²·ε_c(ρ₀)/σ². Predicción de
la derivación (dynamics.cronos_jeans): q < 1 estable (el ruido de
Poisson no crece), q > 1 inestable a TODAS las k, con tasa ∝ k
(catástrofe ultravioleta).

Aquí se reproduce el umbral; los ajustes de tasa por modo son ruidosos
(fase lineal cortísima, modos sembrados por Poisson) y NO se citan como
medida de la ley ∝ k — esa es la puerta del test preinscrito del
frente 5 refundado (predicción convergida), no de este módulo.
"""

from __future__ import annotations

import numpy as np


def run_sheets(q: float, N: int = 400_000, ng: int = 256, T: float = 0.5, dt: float = 2e-3,
               seed: int = 1, nmodes: int = 8, sample_every: int = 25) -> dict:
    """Integra el sistema de láminas (leapfrog, densidad CIC) y devuelve
    muestras de rms(δ) y de |δ_k| para los primeros modos."""
    rng = np.random.default_rng(seed)
    L = 1.0
    x = rng.random(N) * L                      # ruido de Poisson como semilla
    v = rng.standard_normal(N)                 # σ = 1
    m = L / N                                  # ρ₀ = 1
    c2A = q * 2.0 / 3.0                        # (3/2)·c²A·ρ₀^{3/2} = q
    dx = L / ng
    k = 2.0 * np.pi * np.arange(1, nmodes + 1) / L

    def accel(x):
        xi = x / dx
        i0 = np.floor(xi).astype(int) % ng
        w1 = xi - np.floor(xi)
        rho = (np.bincount(i0, weights=(1.0 - w1), minlength=ng)
               + np.bincount((i0 + 1) % ng, weights=w1, minlength=ng)) * m / dx
        eps = c2A * rho ** 1.5                 # c²ε_c
        g = (np.roll(eps, -1) - np.roll(eps, 1)) / (2.0 * dx)
        return g[i0] * (1.0 - w1) + g[(i0 + 1) % ng] * w1, rho

    a, rho = accel(x)
    samples = [(0.0, float(rho.std()), _modes(k, x, N))]
    t = 0.0
    for s in range(int(round(T / dt))):
        v += 0.5 * dt * a
        x = (x + dt * v) % L
        a, rho = accel(x)
        v += 0.5 * dt * a
        t += dt
        if (s + 1) % sample_every == 0:
            samples.append((t, float(rho.std()), _modes(k, x, N)))
    poisson = float(np.sqrt(N / ng) / (N / ng))  # rms de Poisson por celda ≈ 1/√(N/ng)
    return {"q": q, "N": N, "ng": ng, "T": T, "dt": dt, "seed": seed, "k": k.tolist(),
            "poisson_rms_per_cell": poisson,
            "samples": [{"t": t_, "rms_delta": r_, "delta_k": dk_} for t_, r_, dk_ in samples]}


def _modes(k: np.ndarray, x: np.ndarray, N: int) -> list:
    ph = np.exp(-1j * np.outer(k, x))
    return (np.abs(ph.sum(axis=1)) / N).tolist()


def rms_at(res: dict, t: float) -> float:
    ts = np.array([s["t"] for s in res["samples"]])
    rs = np.array([s["rms_delta"] for s in res["samples"]])
    return float(np.interp(t, ts, rs))


def threshold_scan(qs=(0.8, 1.2, 2.0, 4.0), times=(0.05, 0.1, 0.2, 0.5), **kw) -> dict:
    """rms(δ) en los instantes pedidos para cada q; crecimiento relativo
    al ruido de Poisson inicial."""
    out = []
    for q in qs:
        res = run_sheets(q, **kw)
        r0 = res["samples"][0]["rms_delta"]
        out.append({"q": q, "rms_initial": r0, "rms_at": {str(t): rms_at(res, t) for t in times},
                    "growth_factor_final": rms_at(res, times[-1]) / r0,
                    "fluid_rate_over_k_sigma": float(np.sqrt(max(q - 1.0, 0.0)))})
    return {"scan": out, "reading": "q < 1: rms se queda en el ruido de Poisson; q > 1: crece (y satura pronto "
                                    "para q ≳ 2); el umbral está donde la derivación lo pone"}
