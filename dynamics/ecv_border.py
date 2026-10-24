"""Radio de aceleración nula de la ECV: el borde estático entre la burbuja de MCV de un halo y el océano de ECV (documento
vivo del 24-oct-2026, «borde de ECV, intercambio de masa por halo y f_h(a)»). Sin dinámica, sin datos y sin letra.

DE DÓNDE SALE. Con la gravedad del bloque P0 del 23-oct, ∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E), la ECV (w = −1) repele: dentro de
una esfera de ECV uniforme la aceleración es +(8πG/3)·ρ_E·r hacia fuera. Para una masa M rodeada de ECV la aceleración
neta es

  a(r) = −GM/r² + (8πG/3)·ρ_E·r,   nula en   R₀ = (3M/8πρ_E)^{1/3}.

Tres identidades exactas, sin ninguna elección:
  1. R₀ es la cima de la barrera entre el halo y el océano: el potencial Φ(r) = −GM/r − (4πG/3)ρ_E r² tiene allí su
     máximo, y más allá la ECV empuja hacia fuera. Con la masa M200 (200ρ_c), R₀/r200 = (100·E²(z)/Ω_Λ)^{1/3}, igual
     para todas las masas.
  2. La ECV dentro de la burbuja es la mitad de su masa: (4π/3)R₀³ρ_E = M/2. Si el halo crece, el borde barre dM/2.
  3. La densidad media de una burbuja es 2ρ_E, la misma condición que anula la deceleración del fondo: si toda la
     materia estuviera en burbujas, su volumen igualaría al del universo en z_acc = (2Ω_Λ/Ω_m)^{1/3} − 1.

USO COMO BORDE: CONDICIONAL. Que el borde de ECV del autor («el punto de intercambio entre la burbuja de MCV y la ECV
fuera de la influencia del punto gravitacional y térmico», 24-oct) esté en R₀ es la decisión 1 del documento. Dos
variantes: R₀ con la masa M200 y R₀ autoconsistente, con la masa del NFW prolongado hasta R₀ (x³ = (3/2)(ρ_s/ρ_E)·m(x)
en unidades de r_s). La masa total dentro de R₀ incluye todo lo que gravita (MCV, bariones y lo ya atrapado).

Unidades: kpc, M☉, km/s, como dynamics/mcv_equilibrium.py. Cosmología de referencia declarada: Planck 2018 VI (h = 0.674,
Ω_m = 0.315, Ω_Λ = 0.685, plana, sin radiación), la del documento del 24-oct; ρ_E = Ω_Λ·ρ_c0 es constante.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from dynamics.mcv_equilibrium import G_KPC

PLANCK18 = {"h": 0.674, "Omega_m": 0.315, "Omega_L": 0.685}


def E_of_z(z: float, cosmo: dict = PLANCK18) -> float:
    """H(z)/H0 en un fondo plano de materia y ECV."""
    return math.sqrt(cosmo["Omega_m"] * (1.0 + z) ** 3 + cosmo["Omega_L"])


def rho_crit(z: float = 0.0, cosmo: dict = PLANCK18) -> float:
    """ρ_c(z) = 3H(z)²/(8πG) en M☉/kpc³."""
    H0 = 100.0 * cosmo["h"] / 1000.0                        # km/s/kpc
    return 3.0 * H0 ** 2 * E_of_z(z, cosmo) ** 2 / (8.0 * math.pi * G_KPC)


def rho_ecv(cosmo: dict = PLANCK18) -> float:
    """ρ_E = Ω_Λ·ρ_c0 en M☉/kpc³ (constante)."""
    return cosmo["Omega_L"] * rho_crit(0.0, cosmo)


def zero_acceleration_radius(M: float, rho_E: float | None = None) -> float:
    """R₀ = (3M/8πρ_E)^{1/3} en kpc, con M en M☉."""
    rho_E = rho_ecv() if rho_E is None else rho_E
    return (3.0 * M / (8.0 * math.pi * rho_E)) ** (1.0 / 3.0)


def net_acceleration(r, M: float, rho_E: float | None = None):
    """a(r) = −GM/r² + (8πG/3)ρ_E r en (km/s)²/kpc: negativa (hacia dentro) antes de R₀ y positiva después."""
    rho_E = rho_ecv() if rho_E is None else rho_E
    r = np.asarray(r, dtype=float)
    return -G_KPC * M / r ** 2 + (8.0 * math.pi * G_KPC / 3.0) * rho_E * r


def potential(r, M: float, rho_E: float | None = None):
    """Φ(r) = −GM/r − (4πG/3)ρ_E r² en (km/s)², con máximo en R₀."""
    rho_E = rho_ecv() if rho_E is None else rho_E
    r = np.asarray(r, dtype=float)
    return -G_KPC * M / r - (4.0 * math.pi * G_KPC / 3.0) * rho_E * r ** 2


def ecv_mass_inside(R: float, rho_E: float | None = None) -> float:
    """(4π/3)R³ρ_E: en R = R₀ vale exactamente M/2."""
    rho_E = rho_ecv() if rho_E is None else rho_E
    return 4.0 * math.pi / 3.0 * R ** 3 * rho_E


def r0_over_r200(z: float = 0.0, cosmo: dict = PLANCK18) -> float:
    """R₀/r200 con la masa M200 (200ρ_c(z)): (100·E²(z)/Ω_Λ)^{1/3}, igual para todas las masas."""
    return (100.0 * E_of_z(z, cosmo) ** 2 / cosmo["Omega_L"]) ** (1.0 / 3.0)


def z_acceleration(cosmo: dict = PLANCK18) -> float:
    """z_acc = (2Ω_Λ/Ω_m)^{1/3} − 1: ρ_m = 2ρ_E, el fin de la deceleración y el llenado del espacio por las burbujas."""
    return (2.0 * cosmo["Omega_L"] / cosmo["Omega_m"]) ** (1.0 / 3.0) - 1.0


def _m(x: float) -> float:
    return math.log1p(x) - x / (1.0 + x)


def rho_s_over_rho_ecv(c: float, z: float = 0.0, cosmo: dict = PLANCK18) -> float:
    """ρ_s/ρ_E de un NFW con concentración c (M200 a 200ρ_c(z)): (200/3)·c³/m(c)·E²(z)/Ω_Λ."""
    return (200.0 / 3.0) * c ** 3 / _m(c) * E_of_z(z, cosmo) ** 2 / cosmo["Omega_L"]


def r0_selfconsistent(c: float, z: float = 0.0, cosmo: dict = PLANCK18) -> float:
    """x_R = R₀/r_s con la masa del NFW prolongado hasta R₀: la raíz de x³ = (3/2)(ρ_s/ρ_E)·m(x) más allá de r200."""
    a = 1.5 * rho_s_over_rho_ecv(c, z, cosmo)
    return brentq(lambda x: x ** 3 - a * _m(x), c, 1e4 * c, xtol=1e-14, rtol=1e-14)


def border_radius(M200: float, c: float, z: float = 0.0, cosmo: dict = PLANCK18) -> dict:
    """El borde estático de un halo NFW (M200 en M☉, a 200ρ_c(z)) en sus dos variantes, en kpc y en r_s."""
    rc = rho_crit(z, cosmo)
    r200 = (3.0 * M200 / (4.0 * math.pi * 200.0 * rc)) ** (1.0 / 3.0)
    r_s = r200 / c
    x_m200 = c * r0_over_r200(z, cosmo)
    x_self = r0_selfconsistent(c, z, cosmo)
    return {"M200": M200, "c": c, "z": z, "r200": r200, "r_s": r_s, "rho_s": rho_s_over_rho_ecv(c, z, cosmo) * rho_ecv(cosmo),
            "x_R0_M200": x_m200, "R0_M200": x_m200 * r_s, "x_R0_self": x_self, "R0_self": x_self * r_s,
            "M_R0_over_M200": _m(x_self) / _m(c)}
