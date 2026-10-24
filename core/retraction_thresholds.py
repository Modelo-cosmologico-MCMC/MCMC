"""Umbrales de pérdida de dimensión del Maestro (§4.4 y Ap. G, ecs. 699–711 y Cuadro 19; Ap. L, Cuadro 32): las tres
incoherencias que aparecen al aplicarlos tal como están escritos. Registro del 23-oct-2026 como HUECO DECLARADO, con
las cifras del guion p1_comprobaciones.py del autor; sin corridas del interior del objeto crítico hasta sus decisiones
3–5 (orden y valores de los umbrales, variable del interior, qué ocurre al perder el tiempo).

1. Orden. La ec. (708) exige tensiones críticas decrecientes, T_crit(3+1D) > T_crit(3D) > T_crit(2D) > T_crit(1D),
   porque T_eff decrece (711) y el tiempo debe perderse primero. El Cuadro 19 las da crecientes (≈10⁻³, ≈0.002, ≈0.9,
   ≈1): leídas al pie de la letra, se perderían primero la polaridad y el giro, y el tiempo el último.
2. Valores. La ec. (709), T_crit,n = (ṽ_n − ṽ_{n+1})/(ṽ_n + ṽ_{n+1}) con ṽ_n = v_{N+1−n}, con los VEV que el propio
   Maestro deriva en el Cuadro 32 (v_n² ∝ √(j(j+1))) da entre 0 y 0.38 en los tres primeros tramos con cualquiera de las
   asignaciones de índices (y también si se lee con v_n en lugar de v_n²), donde el Cuadro 19 pone 10⁻³, 0.002 y 0.9.
   El Cuadro 19 cita los VEV del «Cuadro 2 del Tratado Técnico», no localizado en el Proyecto.
3. Época. La ec. (699), T_eff = (ρ_m − ρ_E)/(ρ_m + ρ_E), con las densidades observadas (Ω_m = 0.315) da T_eff = −0.37
   hoy: el umbral 10⁻³ de S_mx (ec. 700) se habría cruzado en z = 0.296, hace 3.5 Gyr, contra la ec. (556), que sitúa
   hoy S_ext ≃ 90 < S_mx.

Lectura del lapso (N² = 1 − r_s/r), sin letra: el cruce N² = T_crit cae en r = r_s/(1 − T_crit); con el Cuadro 19
literal el giro se perdería en r = 10 r_s, fuera de la ISCO (3 r_s); con el orden de (708) toda la secuencia cabe dentro
de r = 1.001 r_s. Estatuto: hueco declarado (comprobación interna de coherencia del texto, E8; sin datos).
"""

from __future__ import annotations

import math
from itertools import pairwise

from scipy.integrate import quad
from scipy.optimize import brentq

# Cuadro 19 (tiempo, gravedad 3D, giro, polaridad) y umbrales de retracción de la ec. (710), en ese orden
TABLE_19 = (1e-3, 2e-3, 0.9, 1.0)
TABLE_19_LABELS = ("tiempo", "gravedad 3D", "giro", "polaridad")
S_RETRACTION_710 = (1.001, 0.999, 0.099, 0.009)

# Asignaciones de espín a los índices n = 1..4 y al VEV v_0 (None ⟹ v_0 = 0), como en el guion del autor
SPIN_ASSIGNMENTS = {
    "Cuadro 32 (n = 1..4 ↔ j = ½, ½, 3/2, 5/2; v_0 = 0)": ({1: 0.5, 2: 0.5, 3: 1.5, 4: 2.5}, None),
    "Ap. G (V1D..V3+1D ↔ j = ½..7/2; v_0 = V0D, j = ½)": ({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5}, 0.5),
    "Ap. G (V1D..V3+1D ↔ j = ½..7/2; v_0 = 0)": ({1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5}, None),
}

H0_KMS_MPC = 67.4
MPC_KM = 3.0857e19
GYR_S = 3.156e16


def is_strictly_decreasing(seq) -> bool:
    return all(a > b for a, b in pairwise(seq))


def vev_squared(j: float) -> float:
    """v² ∝ √(j(j+1)) (Cuadro 32)."""
    return math.sqrt(j * (j + 1.0))


def t_crit_709(values: dict[int, float], N: int = 4) -> list[float]:
    """Ec. (709) con ṽ_n = v_{N+1−n}: T_crit,n = (ṽ_n − ṽ_{n+1})/(ṽ_n + ṽ_{n+1}), n = 1..N (values indexado 0..N)."""
    vt = {n: values[N + 1 - n] for n in range(1, N + 2)}
    return [(vt[n] - vt[n + 1]) / (vt[n] + vt[n + 1]) for n in range(1, N + 1)]


def t_crit_709_assignment(spins: dict[int, float], v0_spin: float | None, power: float = 1.0) -> list[float]:
    """(709) para una asignación de espines; power = 1 usa v_n² ∝ √(j(j+1)) (lectura del guion del autor), power = ½
    usa v_n."""
    vals = {n: vev_squared(j) ** power for n, j in spins.items()}
    vals[0] = 0.0 if v0_spin is None else vev_squared(v0_spin) ** power
    return t_crit_709(vals, N=len(spins))


def t_eff_699(z: float, omega_m: float = 0.315) -> float:
    """Ec. (699) con las densidades de fondo de un ΛCDM plano: (Ω_m(1+z)³ − Ω_Λ)/(Ω_m(1+z)³ + Ω_Λ)."""
    m, lam = omega_m * (1.0 + z) ** 3, 1.0 - omega_m
    return (m - lam) / (m + lam)


def z_at_t_eff(target: float, omega_m: float = 0.315) -> float:
    return brentq(lambda z: t_eff_699(z, omega_m) - target, 0.0, 3.0, xtol=1e-12)


def lookback_time_gyr(z: float, omega_m: float = 0.315, H0: float = H0_KMS_MPC) -> float:
    """Tiempo de retroceso en ΛCDM plano (Gyr)."""
    integral = quad(lambda x: 1.0 / ((1.0 + x) * math.sqrt(omega_m * (1.0 + x) ** 3 + 1.0 - omega_m)), 0.0, z)[0]
    return integral / (H0 / MPC_KM) / GYR_S


def lapse_crossing_radius(t_crit: float) -> float:
    """Lectura del lapso (sin letra): N² = 1 − r_s/r = T_crit en r/r_s = 1/(1 − T_crit)."""
    return 1.0 / (1.0 - t_crit)


def incoherences() -> dict:
    """Las tres incoherencias con sus cifras (las del guion del autor del 23-oct)."""
    z1 = z_at_t_eff(1e-3)
    return {
        "order": {"table_19": list(TABLE_19), "table_19_decreasing": is_strictly_decreasing(TABLE_19),
                  "required_by_708": "decreciente"},
        "values": {lab: {"v2": t_crit_709_assignment(s, v0), "v": t_crit_709_assignment(s, v0, power=0.5)}
                   for lab, (s, v0) in SPIN_ASSIGNMENTS.items()},
        "epoch": {"omega_m": 0.315, "t_eff_today": t_eff_699(0.0), "z_t_eff_zero": z_at_t_eff(0.0), "z_t_eff_1e-3": z1,
                  "lookback_gyr": lookback_time_gyr(z1)},
        "lapse_reading": {lab: lapse_crossing_radius(min(t, 0.999)) for lab, t in zip(TABLE_19_LABELS, TABLE_19)},
    }
