"""El objetivo ρ_id de Sculptor como curva de degeneración medible.

El medio paso 2 dejó la carga explicativa de los dSph en el sector
ρ_id con un único número: M_1/2 ≈ 2×10⁷ M⊙ dentro del radio de media
luz 3D r_1/2 ≈ (4/3)·R_half (estimador de Wolf declarado). El perfil
cored del corpus (Ap. A / Tratado Unificado v32, presentación):

    ρ_id(r) = ρ0 / (1 + (r/r_c)²)
    M_id(<r) = 4π·ρ0·r_c³ · [ r/r_c − arctan(r/r_c) ]   (forma cerrada)

Una sola restricción (M_id(<r_1/2) = M_1/2 − M⋆(<r_1/2)) no fija dos
parámetros: lo que Sculptor MIDE es la CURVA de degeneración ρ0(r_c).
El tratado no deriva (ρ0, r_c) a escala dSph — hueco declarado; esta
curva es el objetivo cuantitativo que cualquier derivación futura de
ρ_id debe atravesar, y el punto donde SPARC (que fija su propia
relación rcore(M) — B.5, calibrada, no derivada) puede cruzarse con
los dSph (5E).
"""

from __future__ import annotations

import numpy as np

from .weak_field import plummer_mass


def mass_cored(r, rho0: float, r_c: float):
    """M_id(<r) = 4π·ρ0·r_c³·[r/r_c − arctan(r/r_c)]  [M_sol]."""
    r = np.asarray(r, dtype=float)
    x = r / r_c
    return 4.0 * np.pi * rho0 * r_c ** 3 * (x - np.arctan(x))


def rho0_required(r_c, M_target: float, r_enclose: float):
    """La curva de degeneración: el ρ0 que hace
    M_id(<r_enclose) = M_target para cada r_c."""
    r_c = np.asarray(r_c, dtype=float)
    x = r_enclose / r_c
    return M_target / (4.0 * np.pi * r_c ** 3 * (x - np.arctan(x)))


def sculptor_rho_id_curve(r_c_grid, M_half: float, M_star: float,
                          a_plummer: float, r_half_3d: float):
    """ρ0(r_c) para Sculptor: M_id debe aportar M_1/2 − M⋆(<r_1/2)
    dentro de r_1/2 (Plummer para la parte estelar). Devuelve
    (M_id_target, ρ0(r_c))."""
    m_star_inside = float(plummer_mass(r_half_3d, M_star, a_plummer))
    m_id_target = M_half - m_star_inside
    if m_id_target <= 0.0:
        raise ValueError(
            "M⋆(<r_1/2) ya cubre M_1/2: no queda objetivo para ρ_id "
            f"(M_1/2 = {M_half:.3e}, M⋆(<r_1/2) = {m_star_inside:.3e})")
    return m_id_target, rho0_required(r_c_grid, m_id_target, r_half_3d)
