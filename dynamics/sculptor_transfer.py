"""La transferencia congelada del Frente 5E: A_req de Sculptor.

PROTOCOLO (preinscripción, 16-ago-2026): la amplitud de Cronos que el
análisis SPARC recibe queda CONGELADA aquí, calculada por la misma
maquinaria del medio paso 2 (problema inverso de Sculptor) ANTES de
ingerir o inspeccionar ningún dato SPARC. SPARC es validación fuera
de dominio, no entrenamiento: el análisis 5E no ajusta A por galaxia,
no la optimiza globalmente y no modifica la ley para mejorar el
resultado — el candado lo vigila un test que escanea el módulo de
análisis (tests/test_front5e_lock.py).

La pregunta del frente:
    ¿La predicción que Sculptor ya fijó sobrevive cuando SPARC no
    puede modificarla?

Convenciones congeladas (las del corpus del repositorio, no nuevas):
- Υ⋆(Sculptor) = 2.0 (fiducial del barrido [1, 3] declarado en
  dynamics/dsph_data.py); sensibilidad con 1.0 y 3.0.
- Lectura de la ec. (11.5): la transferencia es INDEPENDIENTE de la
  lectura (L/P) — A_req es el resultado del problema inverso, no de
  la cota; ambas lecturas quedan declaradas en dynamics/weak_field.py.
- β = 0 (montaje del medio paso 2; el promedio proyectado total es
  β-invariante — teorema virial proyectado, test permanente).
- R_half = 260 pc, σ_obs = 9.2 km/s (Walker et al. 2009; procedencia
  en dynamics/dsph_data.py).
- Unidades de A: (M⊙/pc³)^(−3/2) — ε_c = A·ρ^(3/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsph_data import SCULPTOR, plummer_scale_from_Rhalf, stellar_mass
from .jeans import sigma_los_sq_lum_avg
from .weak_field import g_eff_plummer, plummer_density, plummer_g_newton

A_UNIT = 1e-13
UPSILON_SCULPTOR_FIDUCIAL = 2.0


def sculptor_A_req(upsilon: float) -> float:
    """A_req del problema inverso de Sculptor (medio paso 2) — LA
    función canónica: el 5C estructural y el 5E la importan de aquí.
    σ_obs² = σ_N² + (A/A_unit)·dσ² ⟹ A_req (linealidad verificada
    por test)."""
    d = SCULPTOR
    a0 = plummer_scale_from_Rhalf(d["R_half_pc"])
    M = stellar_mass(d["L_V_Lsun"], upsilon)
    r = np.geomspace(0.05, 120.0 * a0, 800)
    nu = plummer_density(r, M, a0)
    kw = {"R_max": 8.0 * a0, "u_max": 120.0 * a0}
    s2_N = sigma_los_sq_lum_avg(r, nu, plummer_g_newton(r, M, a0), **kw)
    dS2 = sigma_los_sq_lum_avg(
        r, nu, g_eff_plummer(r, M, a0, A_UNIT), **kw) - s2_N
    return float((d["sigma_los_kms"] ** 2 - s2_N) / dS2 * A_UNIT)


@dataclass(frozen=True)
class CrossFalsificationConfig:
    """Configuración INMUTABLE del 5E: el análisis SPARC la recibe
    como entrada y no puede estimar ni modificar ninguno de sus
    campos (dataclass frozen; test de inmutabilidad)."""
    A_sculptor: float                 # (M⊙/pc³)^(-3/2), Υ⋆ fiducial
    A_sculptor_sensitivity: tuple     # (Υ⋆=1, Υ⋆=3) — declarados
    upsilon_sculptor: float
    beta_sculptor: float
    R_half_pc: float
    sigma_obs_kms: float
    # Convenciones bariónicas SPARC, congeladas ANTES de la ingesta
    # (estándar de la literatura SPARC a 3.6 μm — Lelli et al. 2016):
    upsilon_disk_sparc: float
    upsilon_bulge_sparc: float
    upsilon_disk_sensitivity: tuple
    zeta_disc: float                  # h/R_d (fiducial del 5C)
    zeta_sensitivity: tuple
    bootstrap_seed: int
    bootstrap_n: int


def frozen_config() -> CrossFalsificationConfig:
    """Construye la configuración congelada — A calculada AQUÍ por la
    función canónica, nunca copiada de un informe ni derivada de
    SPARC."""
    return CrossFalsificationConfig(
        A_sculptor=sculptor_A_req(UPSILON_SCULPTOR_FIDUCIAL),
        A_sculptor_sensitivity=(sculptor_A_req(1.0), sculptor_A_req(3.0)),
        upsilon_sculptor=UPSILON_SCULPTOR_FIDUCIAL,
        beta_sculptor=0.0,
        R_half_pc=SCULPTOR["R_half_pc"],
        sigma_obs_kms=SCULPTOR["sigma_los_kms"],
        upsilon_disk_sparc=0.5,
        upsilon_bulge_sparc=0.7,
        upsilon_disk_sensitivity=(0.3, 0.7),
        zeta_disc=0.15,
        zeta_sensitivity=(0.1, 0.2),
        bootstrap_seed=42,
        bootstrap_n=10000,
    )
