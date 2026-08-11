"""La primera muestra dSph del frente 5: Sculptor.

PROCEDENCIA Y ESTATUTO DE LOS DATOS (contrato de honestidad). Valores
compilados de la literatura pública y verificados el 10-ago-2026
contra los resúmenes de búsqueda de Walker et al. 2009 (AJ 137, 3100 —
survey Magellan/MMFS, 1497 gigantes rojas; y ApJ 704, 1274, «A
Universal Mass Profile for Dwarf Spheroidal Galaxies?»):

- σ_los ≈ 9–10 km/s (global; el valor puntual usual de la tabla de
  Walker et al. 2009 es 9.2 km/s) — verificado en el rango;
- R_half = 260 ± 39 pc (radio de media luz proyectado, perfil de
  Plummer; Walker et al. 2009) — verificado;
- M_V ≈ −11.2 — verificado; L_V se DERIVA aquí con M_V_sol = 4.83:
  L_V = 10^(−0.4·(M_V − 4.83)) ≈ 2.6e6 L_sol.

PENDIENTE DECLARADO: la ingesta de la tabla original legible por
máquina (VizieR J/ApJ/704/1274) y del perfil binado σ_los(R) — los
archivos astronómicos están bloqueados por el proxy de red de la
sesión de trabajo; hasta esa ingesta, este módulo usa los valores
globales anteriores y el análisis se limita al contraste GLOBAL
(un número por sistema), no al perfil radial.

Υ⋆ (M/L estelar en banda V) NO es un dato: es una hipótesis de
población estelar vieja; se declara el rango [1, 3] M_sol/L_sol y
todos los resultados se publican como barrido sobre él.
"""

from __future__ import annotations

SCULPTOR = {
    "name": "Sculptor dSph",
    # Cinemática global (Walker et al. 2009; rango verificado 9-10):
    "sigma_los_kms": 9.2,
    "sigma_los_scan_kms": (9.0, 9.2, 10.0),
    # Estructura (Walker et al. 2009, perfil de Plummer):
    "R_half_pc": 260.0,
    "R_half_err_pc": 39.0,
    # Fotometría (M_V verificado; L_V derivada con M_V_sol = 4.83):
    "M_V": -11.2,
    "L_V_Lsun": 2.6e6,
    # Hipótesis declarada (población vieja, banda V):
    "upsilon_scan": (1.0, 2.0, 3.0),
    "provenance": (
        "Walker et al. 2009 (AJ 137, 3100; ApJ 704, 1274); valores "
        "globales verificados vía resúmenes de búsqueda el 10-ago-2026; "
        "tabla original y perfil binado σ_los(R): ingesta pendiente "
        "(archivos bloqueados por el proxy de la sesión)"
    ),
}


def plummer_scale_from_Rhalf(R_half_pc: float) -> float:
    """Para el perfil de Plummer, el radio de media luz PROYECTADO es
    exactamente el parámetro de escala: a = R_half."""
    return float(R_half_pc)


def stellar_mass(L_V_Lsun: float, upsilon: float) -> float:
    """M⋆ = Υ⋆ · L_V  [M_sol]."""
    return float(upsilon) * float(L_V_Lsun)
