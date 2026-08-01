"""Likelihoods ampliados: CMB comprimido y crecimiento fσ8 (ronda 4, opción B).

Los dos bloques que sondean la transición y de los que procedía la
ventaja reportada por el corpus (A.6): el fondo cósmico (geometría
comprimida) y el crecimiento de estructura.

CMB COMPRIMIDO (R, l_A, ω_b):
    R = √(Ω_m)·H0·D_M(z*)/c,   l_A = π·D_M(z*)/r_s(z*)
    con z* del ajuste de Hu & Sugiyama (1996) y el horizonte sonoro
    r_s(z*) = ∫_{z*}^∞ c_s dz/H,  c_s = c/√(3(1+R_b)),
    R_b = (3ω_b/4ω_γ)/(1+z),  ω_γ = 2.469e-5 (T_CMB = 2.7255 K).
    Valores observados: Planck 2018 TT,TE,EE+lowE comprimido según
    Chen, Huang & Wang 2019 (JCAP 02, 028) — los mismos de
    data/planck2018/compressed_geometry.txt.
    SIMPLIFICACIÓN DECLARADA: likelihood gaussiano DIAGONAL (las
    correlaciones entre R, l_A y ω_b del paper no se incluyen en esta
    versión mínima; su matriz completa está en la referencia).

CRECIMIENTO fσ8(z): factor de crecimiento D(a) integrado exactamente
sobre el H(z) del modelo (no la aproximación γ de Linder):
    D'' + (2 + dlnH/dlna)·D' − (3/2)·Ω_m(a)·D = 0
    fσ8(z) = σ8·f(a)·D(a)/D(1),  f = dlnD/dlna
contra la compilación RSD embebida en download_data (referencias por
punto). σ8 entra como parámetro nuevo del ajuste.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .background import H_of_z

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
C_KMS = 299792.458
OMEGA_GAMMA_H2 = 2.469e-5   # ω_γ para T_CMB = 2.7255 K
N_EFF = 3.046

# Planck 2018 comprimido (Chen, Huang & Wang 2019, JCAP 02, 028):
CMB_OBS = {"R": (1.7502, 0.0046), "l_A": (301.471, 0.090),
           "omega_b": (0.02236, 0.00015)}


def z_star_hu_sugiyama(omega_b: float, omega_m: float) -> float:
    """Redshift de recombinación z* (Hu & Sugiyama 1996, fórmula de ajuste)."""
    g1 = 0.0783 * omega_b ** -0.238 / (1.0 + 39.5 * omega_b ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b ** -0.738) \
        * (1.0 + g1 * omega_m ** g2)


def _omega_r0(h: float) -> float:
    """Ω_r0 = ω_γ(1 + 0.2271·N_eff)/h² — fotones + neutrinos relativistas."""
    return OMEGA_GAMMA_H2 * (1.0 + 0.2271 * N_EFF) / h ** 2


def sound_horizon(theta: tuple, omega_b: float, n_grid: int = 3000) -> float:
    """r_s(z*) [Mpc] integrando c_s/H de z* a z_alto (trapecios en ln z)."""
    H0, Om, eps, z_trans = theta
    h = H0 / 100.0
    z_st = z_star_hu_sugiyama(omega_b, Om * h ** 2)
    lnz = np.linspace(np.log(z_st), np.log(1e8), n_grid)
    z = np.exp(lnz)
    H = H_of_z(z, H0=H0, Omega_m=Om, Omega_r=_omega_r0(h),
               eps=eps, z_trans=z_trans)
    R_b = (3.0 * omega_b / (4.0 * OMEGA_GAMMA_H2)) / (1.0 + z)
    cs = C_KMS / np.sqrt(3.0 * (1.0 + R_b))
    # dz = z·dlnz
    return float(np.trapezoid(cs / H * z, lnz))


def comoving_to_zstar(theta: tuple, omega_b: float,
                      n_grid: int = 3000) -> float:
    """D_M(z*) [Mpc] (plano), con radiación físicamente consistente."""
    H0, Om, eps, z_trans = theta
    h = H0 / 100.0
    z_st = z_star_hu_sugiyama(omega_b, Om * h ** 2)
    z = np.linspace(0.0, z_st, n_grid)
    H = H_of_z(z, H0=H0, Omega_m=Om, Omega_r=_omega_r0(h),
               eps=eps, z_trans=z_trans)
    return float(np.trapezoid(C_KMS / H, z))


def cmb_compressed_loglike(theta: tuple, omega_b: float) -> float:
    """Likelihood gaussiano diagonal sobre (R, l_A, ω_b) — simplificación
    declarada (sin correlaciones)."""
    H0, Om, eps, z_trans = theta
    D_M = comoving_to_zstar(theta, omega_b)
    r_s = sound_horizon(theta, omega_b)
    R_pred = np.sqrt(Om) * H0 * D_M / C_KMS
    lA_pred = np.pi * D_M / r_s
    chi2 = ((R_pred - CMB_OBS["R"][0]) / CMB_OBS["R"][1]) ** 2 \
        + ((lA_pred - CMB_OBS["l_A"][0]) / CMB_OBS["l_A"][1]) ** 2 \
        + ((omega_b - CMB_OBS["omega_b"][0]) / CMB_OBS["omega_b"][1]) ** 2
    return -0.5 * float(chi2)


# ---------------------------------------------------------------------
# Crecimiento
# ---------------------------------------------------------------------

def growth_D_f(z_eval: np.ndarray, theta: tuple,
               n_grid: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """D(z)/D(0) y f(z) = dlnD/dlna integrando la ODE exacta sobre el
    H(z) del modelo (RK4 en ln a, malla fija — determinista y rápido)."""
    H0, Om, eps, z_trans = theta
    lna = np.linspace(np.log(1e-3), 0.0, n_grid)
    a = np.exp(lna)
    z = 1.0 / a - 1.0
    H = np.asarray(H_of_z(z, H0=H0, Omega_m=Om, eps=eps, z_trans=z_trans))
    dlnH = np.gradient(np.log(H), lna)
    Om_a = Om * a ** -3 * (H0 / H) ** 2

    h_step = lna[1] - lna[0]
    y = np.array([a[0], a[0]])   # D ≈ a, D' ≈ a en materia dominante

    def rhs(i_frac: float, y_vec: np.ndarray) -> np.ndarray:
        i = min(int(round(i_frac)), n_grid - 1)
        return np.array([y_vec[1],
                         -(2.0 + dlnH[i]) * y_vec[1]
                         + 1.5 * Om_a[i] * y_vec[0]])

    D_hist = np.empty(n_grid)
    f_hist = np.empty(n_grid)
    D_hist[0], f_hist[0] = y[0], y[1] / y[0]
    for i in range(n_grid - 1):
        k1 = rhs(i, y)
        k2 = rhs(i + 0.5, y + 0.5 * h_step * k1)
        k3 = rhs(i + 0.5, y + 0.5 * h_step * k2)
        k4 = rhs(i + 1, y + h_step * k3)
        y = y + (h_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        D_hist[i + 1] = y[0]
        f_hist[i + 1] = y[1] / y[0]
    D_norm = D_hist / D_hist[-1]
    z_grid = z[::-1]
    D_out = np.interp(z_eval, z_grid, D_norm[::-1])
    f_out = np.interp(z_eval, z_grid, f_hist[::-1])
    return D_out, f_out


def fsigma8_model(z_eval: np.ndarray, theta: tuple,
                  sigma8: float) -> np.ndarray:
    """fσ8(z) = σ8·f(z)·D(z)/D(0)."""
    D, f = growth_D_f(np.asarray(z_eval, dtype=float), theta)
    return sigma8 * f * D


def load_fsigma8_data(path: Path | str | None = None):
    """Carga la compilación RSD de data/boss_eboss/fsigma8.txt."""
    p = Path(path) if path else _DATA_DIR / "boss_eboss" / "fsigma8.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Genera las tablas con "
            "`python scripts/download_data.py boss_eboss`.")
    arr = np.loadtxt(p)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def rsd_loglike(theta: tuple, sigma8: float, z: np.ndarray,
                fs8: np.ndarray, sig: np.ndarray) -> float:
    """χ² gaussiano sobre la compilación fσ8."""
    pred = fsigma8_model(z, theta, sigma8)
    return float(-0.5 * np.sum(((fs8 - pred) / sig) ** 2))
