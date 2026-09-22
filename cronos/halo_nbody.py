"""Nivel A del frente 5 — halo aislado con y sin Cronos v3 (17-sep-2026).

Instrumento: N-cuerpos 3D con gravedad de árbol Barnes–Hut (pytreegrav,
octree + suavizado spline cúbico) y pasos temporales INDIVIDUALES por
niveles (potencias de 2), con el campo de Cronos evaluado como CAMPO
MEDIO ESFÉRICO sobre el perfil radial instantáneo del halo:

    ε_c(r) = A·ρ(r)^{3/2}          (A ≡ α₀⁻¹/ρ_c^{3/2}; ρ en M☉/pc³)
    F_extra = +c²·∇ε_c              (Cor. 11.3c; radial, hacia dentro)
    Γ = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇)          (Cor. 11.3a; ρ̇ lagrangiana =
                                     ∂_tρ + v_r ∂_rρ sobre el perfil)
    N = 1 + Φ_N/c² − ε_c            (Def. 11.1; kick y drift llevan N)

Es la aproximación de campo medio esférico para el campo de Cronos:
adecuada para un halo aislado en equilibrio (la pregunta del Nivel A);
la versión con densidad SPH por partícula queda para el Nivel B. La
gravedad newtoniana sí es 3D completa.

ESQUEMA TEMPORAL («drift-all, kick en el punto medio»): todas las
partículas derivan cada tick Δt_tick = Δt_min/2 con su velocidad (y su
lapse) actuales; cada partícula recibe un kick completo en el punto
medio de su paso individual s (en ticks, potencia de 2), con
Δt_i = min( sqrt(2η_acc ε_soft/|a_i|), η_dyn/sqrt(Gρ(r_i)) ) redondeado
hacia abajo a la potencia de 2. La gravedad de los activos se evalúa
por fuerza bruta (pocos activos, niveles finos del centro) o por árbol
(muchos activos), siempre sobre TODAS las fuentes en sus posiciones
actuales. Segundo orden; simpléctico a paso fijo.

Unidades: kpc, M☉, km/s; 1 kpc/(km/s) = 0.977792 Gyr; G = 4.30091e-6
kpc (km/s)²/M☉. El suavizado se da como equivalente Plummer ε_soft y se
pasa a pytreegrav como soporte spline h = 2.8·ε_soft (convención Gadget).

Condiciones iniciales: NFW truncado exponencialmente (Kazantzidis et al.
2004) con distribución isótropa de Eddington; pares A/B = misma semilla.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d

from dynamics.cronos_amplitude_validity import A_SCULPTOR, rho_crit_today

G_KPC = 4.30091e-6            # kpc (km/s)² / M☉
C_KMS = 299792.458            # km/s
GYR_PER_TIME_UNIT = 0.977792  # 1 kpc/(km/s) en Gyr
KPC3_TO_PC3 = 1e-9            # M☉/kpc³ → M☉/pc³
SPLINE_OVER_PLUMMER = 2.8     # h_spline = 2.8 ε_Plummer (Gadget)

__all__ = ["A_SCULPTOR", "C_KMS", "G_KPC", "GYR_PER_TIME_UNIT", "HaloRun",
           "SphericalCronosField", "nfw_structural", "sample_equilibrium_nfw",
           "truncated_nfw_density"]


# ---------------------------------------------------------------------
# NFW truncado y condiciones iniciales de Eddington
# ---------------------------------------------------------------------

def nfw_structural(M200: float, c: float, H0: float) -> dict:
    """ρ_s [M☉/kpc³], r_s, r200 [kpc] y M200 de un NFW (200·ρ_crit)."""
    rho_crit = rho_crit_today(H0) / KPC3_TO_PC3          # M☉/kpc³
    r200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    rs = r200 / c
    fc = np.log(1.0 + c) - c / (1.0 + c)
    rho_s = M200 / (4.0 * np.pi * rs ** 3 * fc)
    return {"M200": M200, "c": c, "H0": H0, "rho_s": rho_s, "r_s": rs, "r200": r200,
            "rho_crit": rho_crit}


def truncated_nfw_density(r, p: dict, r_decay_factor: float = 0.3):
    """NFW para r ≤ r200 y cola exponencial con pendiente logarítmica
    continua para r > r200 (Kazantzidis et al. 2004): ρ_t (r/r_t)^κ
    exp(−(r − r_t)/r_d), κ = −(1 + 3c)/(1 + c) + r_t/r_d."""
    r = np.asarray(r, float)
    rs, rt = p["r_s"], p["r200"]
    rd = r_decay_factor * rt
    x = r / rs
    rho_in = p["rho_s"] / (x * (1.0 + x) ** 2)
    c = rt / rs
    kappa = -(1.0 + 3.0 * c) / (1.0 + c) + rt / rd
    rho_t = p["rho_s"] / (c * (1.0 + c) ** 2)
    rho_out = rho_t * (r / rt) ** kappa * np.exp(-(r - rt) / rd)
    return np.where(r <= rt, rho_in, rho_out)


def _eddington_tables(p: dict, r_decay_factor: float, n_r: int = 3000,
                      n_E: int = 500, r_min_factor: float = 1e-4, n_u: int = 200, edge_dense: bool = False) -> dict:
    # r_min_factor: borde interior de las tablas en unidades de r_s. El valor por defecto (1e-4) es el del
    # N-cuerpos (frente 5, rondas 1 y 5b; ICs reproducibles). El código de capas (halo_shells) lo baja porque
    # la inversión de Eddington tiene un pico numérico de f(E) en el último nodo (E ≈ Ψ(r_min)) que enfría
    # ~25 % las capas muestreadas cerca de ese borde (piloto declarado en la ronda 2).
    rs, rt = p["r_s"], p["r200"]
    rd = r_decay_factor * rt
    r_min, r_max = r_min_factor * rs, rt + 12.0 * rd
    r = np.geomspace(r_min, r_max, n_r)
    rho = truncated_nfw_density(r, p, r_decay_factor)
    # masa: analítica dentro de r_min (NFW: ≈ 2πρ_s r_s x² para x ≪ 1)
    x0 = r_min / rs
    M0 = 4.0 * np.pi * p["rho_s"] * rs ** 3 * (np.log(1.0 + x0) - x0 / (1.0 + x0))
    M = M0 + cumulative_trapezoid(4.0 * np.pi * rho * r ** 2, r, initial=0.0)
    # potencial relativo Ψ = −Φ: GM/r + 4πG ∫_r^{rmax} ρ r' dr'
    outer = cumulative_trapezoid((rho * r)[::-1], r[::-1], initial=0.0)[::-1] * -1.0
    Psi = G_KPC * M / r + 4.0 * np.pi * G_KPC * outer
    # ρ(Ψ) y derivadas (Ψ decrece con r): derivadas por diferencias
    drho_dr = np.gradient(rho, r)
    dPsi_dr = np.gradient(Psi, r)
    d1 = drho_dr / dPsi_dr                    # dρ/dΨ
    d2 = np.gradient(d1, r) / dPsi_dr         # d²ρ/dΨ²
    # tablas crecientes en Ψ
    order = np.argsort(Psi)
    Psi_s, d1_s, d2_s = Psi[order], d1[order], d2[order]
    d2_of_Psi = interp1d(Psi_s, d2_s, bounds_error=False, fill_value=(d2_s[0], d2_s[-1]))
    d1_at_0 = float(d1_s[0])
    if edge_dense:
        # malla de E densa hacia Ψ_max (la cúspide NFW tiene Ψ(0) finito y f(E) diverge al acercarse E a Ψ(0):
        # las capas con r ≲ 0.1 kpc viven a (Ψ_max − E)/Ψ_max ≲ 1e-2, donde la malla geométrica no resuelve f)
        y = np.geomspace(1e-7, 1.0 - Psi_s[0] * 1.001 / Psi_s[-1], n_E)[::-1]
        E_grid = Psi_s[-1] * (1.0 - y)
    else:
        E_grid = np.geomspace(Psi_s[0] * 1.001, Psi_s[-1] * 0.999, n_E)
    f = np.empty_like(E_grid)
    u = (np.arange(n_u) + 0.5) / n_u                      # Ψ = E − (u√E)² ⇒ dΨ = −2u E du
    for i, E in enumerate(E_grid):
        Psi_q = E * (1.0 - u ** 2)
        f[i] = np.sum(2.0 * np.sqrt(E) * d2_of_Psi(Psi_q) / n_u) + d1_at_0 / np.sqrt(E)
    f /= np.sqrt(8.0) * np.pi ** 2
    neg = float(np.mean(f < 0.0))
    f = np.clip(f, 0.0, None)
    return {"r": r, "rho": rho, "M": M, "Psi": Psi, "E": E_grid, "f": f,
            "f_negative_fraction": neg, "r_max": r_max, "M_tot": float(M[-1])}


def sample_equilibrium_nfw(M200: float, c: float, H0: float, N: int, seed: int,
                           r_decay_factor: float = 0.3, n_v: int = 96) -> dict:
    """Muestra N partículas de la distribución isótropa de Eddington del
    NFW truncado. Devuelve pos, vel [(N,3)], mass [(N,)], parámetros y la
    fracción de f(E) negativa (recortada a 0) como diagnóstico."""
    p = nfw_structural(M200, c, H0)
    T = _eddington_tables(p, r_decay_factor)
    rng = np.random.default_rng(seed)
    # radios por CDF inversa de M(r)
    cdf = (T["M"] - T["M"][0]) / (T["M"][-1] - T["M"][0])
    r_part = np.interp(rng.random(N), cdf, T["r"])
    Psi_part = np.interp(r_part, T["r"], T["Psi"])
    logf = interp1d(np.log(T["E"]), np.log(np.clip(T["f"], 1e-300, None)),
                    bounds_error=False, fill_value=-700.0)
    v = np.empty(N)
    chunk = 20000
    for i0 in range(0, N, chunk):
        Ps = Psi_part[i0:i0 + chunk][:, None]
        vgrid = np.sqrt(2.0 * Ps) * (np.arange(1, n_v + 1) / (n_v + 1.0))[None, :]
        E = Ps - 0.5 * vgrid ** 2
        pdf = vgrid ** 2 * np.exp(logf(np.log(np.clip(E, 1e-300, None))))
        cdfv = np.cumsum(pdf, axis=1)
        cdfv /= cdfv[:, -1:]
        uu = rng.random(len(Ps))
        idx = np.array([np.searchsorted(cdfv[j], uu[j]) for j in range(len(Ps))])
        idx = np.clip(idx, 1, n_v - 1)
        # interpolación lineal dentro de la celda
        c0 = cdfv[np.arange(len(Ps)), idx - 1]
        c1 = cdfv[np.arange(len(Ps)), idx]
        w = np.where(c1 > c0, (uu - c0) / np.maximum(c1 - c0, 1e-300), 0.0)
        v[i0:i0 + chunk] = vgrid[np.arange(len(Ps)), idx - 1] * (1 - w) + vgrid[np.arange(len(Ps)), idx] * w

    def isotropic(n):
        z = rng.uniform(-1.0, 1.0, n)
        ph = rng.uniform(0.0, 2.0 * np.pi, n)
        s = np.sqrt(1.0 - z ** 2)
        return np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=1)

    pos = r_part[:, None] * isotropic(N)
    vel = v[:, None] * isotropic(N)
    mass = np.full(N, T["M_tot"] / N)
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return {"pos": pos, "vel": vel, "mass": mass, "params": p, "M_tot": T["M_tot"],
            "r_max": T["r_max"], "f_negative_fraction": T["f_negative_fraction"],
            "seed": seed, "r_decay_factor": r_decay_factor}


# ---------------------------------------------------------------------
# Campo de Cronos de campo medio esférico
# ---------------------------------------------------------------------

class SphericalCronosField:
    """Perfil radial instantáneo del halo desde la MASA ACUMULADA:
    ln M(<r) muestreada en una malla logarítmica desde el radio de la
    partícula k_inner-ésima hasta r_max y ajustada con un spline
    suavizante (pesos de Poisson, factor de suavizado `smooth`);
    ρ = (dlnM/dlnr)·M/(4πr³) y dlnρ/dlnr salen de las derivadas del
    spline (nada de diferencias finitas sobre conteos bajos). Por debajo
    del radio interior ρ se toma constante (subestima la cúspide ⟹
    subestima ε_c: conservador, declarado). Provee M(<r), Φ_N(r) y las
    funciones de Cronos ε_c = A ρ^{3/2}, dε_c/dr y ∂_tρ (diferencia entre
    actualizaciones)."""

    def __init__(self, A: float, r_min: float = 0.01, r_max: float = 400.0,
                 n_grid: int = 180, k_inner: int = 16, smooth: float = 1.0,
                 r_soft: float = 0.0, tau_avg: float = 0.0, follow_particles: bool = False):
        self.A = float(A)
        self.r_floor, self.r_max, self.n_grid = float(r_min), float(r_max), int(n_grid)
        self.k_inner, self.smooth = int(k_inner), float(smooth)
        self.r_soft = float(r_soft)          # radio suavizado r_s = sqrt(r² + r_soft²) para el campo
        self.tau_avg = float(tau_avg)        # media móvil exponencial de ln M(ln r) (unidades internas)
        # campo INSTANTÁNEO que sigue a las partículas (frente 5b): la malla
        # logarítmica se reconstruye en cada actualización desde el radio de la
        # partícula k_inner-ésima ACTUAL (sin malla fija ni media móvil); el
        # Nivel A usaba malla fija tras la primera llamada + media móvil τ_avg
        self.follow_particles = bool(follow_particles)
        if self.follow_particles and self.tau_avg > 0.0:
            raise ValueError("follow_particles exige tau_avg = 0 (campo instantáneo)")
        self.frozen = False                  # campo congelado (control de conservación)
        self._lrho = None
        self._lr_prev = None
        self._lrho_prev = None
        self._t = None
        self._t_prev = None
        self._lnM_avg = None
        self._lr_avg = None

    def _rs(self, r):
        r = np.asarray(r, float)
        return np.sqrt(r ** 2 + self.r_soft ** 2) if self.r_soft > 0.0 else r

    def update(self, r: np.ndarray, mass: np.ndarray, t: float) -> None:
        if self.frozen and self._lrho is not None:
            return
        order = np.argsort(r)
        r_s = np.clip(r[order], self.r_floor, None)
        M_cum = np.cumsum(mass[order])
        r_in = max(r_s[min(self.k_inner, len(r_s) - 1)], self.r_floor)
        r_out = min(r_s[-1], self.r_max)
        if self._lr_avg is None or self.follow_particles:
            lr = np.linspace(np.log(r_in), np.log(r_out), self.n_grid)
        else:
            lr = self._lr_avg                # malla fija tras la primera llamada (media móvil)
        lnM_raw = np.interp(lr, np.log(r_s), np.log(M_cum))
        if self.tau_avg > 0.0 and self._lnM_avg is not None and self._t is not None:
            alpha = min(1.0, (float(t) - self._t) / self.tau_avg)
            lnM_raw = (1.0 - alpha) * self._lnM_avg + alpha * lnM_raw
        self._lnM_avg, self._lr_avg = lnM_raw, lr
        # spline suavizante de ln M(ln r) con pesos de Poisson (σ_lnM ≈ N^{-1/2});
        # ρ y dlnρ/dlnr salen de sus derivadas analíticas (sin ruido de
        # diferencias finitas sobre conteos bajos)
        N_cum = np.interp(lr, np.log(r_s), np.arange(1, len(r_s) + 1).astype(float))
        w = np.sqrt(np.clip(N_cum, 1.0, None))
        spl = UnivariateSpline(lr, lnM_raw, w=w, k=4, s=self.smooth * self.n_grid)
        lnM = spl(lr)
        slope = np.clip(spl.derivative(1)(lr), 1e-3, 3.0)
        curv = spl.derivative(2)(lr)
        M_g = np.exp(lnM)
        rho = np.clip(slope * M_g / (4.0 * np.pi * np.exp(lr) ** 3), 1e-300, None)
        self._lr_prev, self._lrho_prev, self._t_prev = self.lr_mid if self._lrho is not None else None, self._lrho, self._t
        self.lr_mid, self.r_mid = lr, np.exp(lr)
        self._lrho, self._t = np.log(rho), float(t)
        # dlnρ/dlnr = d ln(slope)/dlnr + slope − 3, con d ln(slope)/dlnr = curv/slope
        self._dlrho_dlr = np.clip(curv / slope + slope - 3.0, -4.0, 1.0)
        self._M_grid = M_g
        self.M_tot = float(M_cum[-1])
        # Φ_N(r) = −G M(<r)/r − 4πG ∫_r^{r_out} ρ r' dr' − G (M_tot − M(r_out))/r_out
        dr = np.gradient(self.r_mid)
        outer = np.cumsum((rho * self.r_mid * dr)[::-1])[::-1]
        self._Phi = (-G_KPC * M_g / self.r_mid - 4.0 * np.pi * G_KPC * outer
                     - G_KPC * (self.M_tot - M_g[-1]) / self.r_mid[-1])

    def _interp(self, r, y):
        lr = np.log(np.clip(r, self.r_mid[0], self.r_mid[-1]))
        return np.interp(lr, self.lr_mid, y)

    def rho(self, r):
        return np.exp(self._interp(r, self._lrho))

    def dlnrho_dlnr(self, r):
        return self._interp(r, self._dlrho_dlr)

    def drho_dr(self, r):
        r = np.asarray(r, float)
        return self.rho(r) * self.dlnrho_dlnr(r) / np.clip(r, self.r_mid[0], None)

    def drho_dt(self, r):
        if self._lrho_prev is None or self._t == self._t_prev:
            return np.zeros_like(np.asarray(r, float))
        lr = np.log(np.clip(r, self.r_mid[0], self.r_mid[-1]))
        lr_p = np.clip(lr, self._lr_prev[0], self._lr_prev[-1])
        d = (np.exp(np.interp(lr, self.lr_mid, self._lrho))
             - np.exp(np.interp(lr_p, self._lr_prev, self._lrho_prev)))
        return d / (self._t - self._t_prev)

    def eps_c(self, r):
        """ε_c = A ρ(r_s)^{3/2} evaluada en el radio suavizado r_s =
        sqrt(r² + r_soft²): el campo medio no resuelve escalas menores
        que el suavizado gravitatorio (declarado)."""
        return self.A * (self.rho(self._rs(r)) * KPC3_TO_PC3) ** 1.5

    def deps_c_dr(self, r):
        """dε_c/dr = (dε_c/dr_s)·(r/r_s): finita y → 0 en el centro."""
        r = np.asarray(r, float)
        rs = self._rs(r)
        rho_pc = self.rho(rs) * KPC3_TO_PC3
        d_rs = 1.5 * self.A * np.sqrt(rho_pc) * self.drho_dr(rs) * KPC3_TO_PC3
        return d_rs * (r / rs if self.r_soft > 0.0 else 1.0)

    def U_ext(self, r, mass):
        """Energía potencial del pozo de Cronos Σ m (−c² ε_c): con el campo
        congelado, K + W + U_ext se conserva."""
        return -C_KMS ** 2 * float(np.sum(mass * self.eps_c(r)))

    def U_self(self, r, mass):
        """Energía AUTOCONSISTENTE del campo de Cronos: la fuerza +c²∇ε_c con
        ε_c = A·ρ^{3/2} y ρ la densidad de las propias partículas deriva del
        funcional U[ρ] = −(2/5)·c²·A·∫ρ^{5/2} dV (δU/δρ = −c²ε_c), es decir
        U_self = (2/5)·Σ m(−c²ε_c): con el campo dinámico lo que se conserva
        es K + W + U_self + W_fric, no K + W + U_ext (frente 5b)."""
        return 0.4 * self.U_ext(r, mass)

    def Phi(self, r):
        r = np.asarray(r, float)
        inside = r < self.r_mid[0]
        out = self._interp(r, self._Phi)
        return np.where(inside, self._Phi[0], out)

    def M_enclosed(self, r):
        return np.exp(self._interp(r, np.log(self._M_grid)))

    def dominance(self, r):
        """D_Φ(r) = c²ε_c/|Φ_N| (condición literal de Cor. 11.3c)."""
        return C_KMS ** 2 * self.eps_c(r) / np.abs(self.Phi(r))

    def force_dominance(self, r):
        """D_F(r) = |c² dε_c/dr| / (G M(<r)/r²): el cociente de fuerzas."""
        r = np.asarray(r, float)
        g_N = G_KPC * self.M_enclosed(r) / np.clip(r, self.r_mid[0], None) ** 2
        return C_KMS ** 2 * np.abs(self.deps_c_dr(r)) / np.clip(g_N, 1e-300, None)


# ---------------------------------------------------------------------
# Integrador
# ---------------------------------------------------------------------

class HaloRun:
    """Halo aislado con pasos individuales; `cronos=False` es el brazo
    newtoniano de referencia. Los tres términos de Cronos v3 se activan
    juntos (es la ley) salvo que se apaguen explícitamente para
    diagnóstico."""

    def __init__(self, pos, vel, mass, *, A: float = A_SCULPTOR, cronos: bool = True,
                 use_force: bool = True, use_friction: bool = True, use_lapse: bool = True,
                 soft_plummer_kpc: float = 0.1, theta: float = 0.7,
                 eta_acc: float = 0.025, eta_dyn: float = 0.02, eta_cross: float = 0.1,
                 r_cross_over_soft: float = 10.0,
                 dt_max_gyr: float = 0.0512, n_levels: int = 12,
                 brute_max: int = 500, profile_every: int = 8,
                 stop_on_weak_violation: bool = False, weak_max: float = 1e-3,
                 stop_well_speed_kms: float | None = None, max_wall_s: float | None = None,
                 field_static: bool = False, tau_avg_myr: float = 50.0,
                 field_kwargs: dict | None = None):
        self.pos = np.array(pos, float)
        self.vel = np.array(vel, float)
        self.mass = np.array(mass, float)
        self.N = len(self.mass)
        self.A = float(A)
        self.cronos = bool(cronos)
        self.use_force, self.use_friction, self.use_lapse = use_force, use_friction, use_lapse
        self.eps_soft = float(soft_plummer_kpc)
        self.h = np.full(self.N, SPLINE_OVER_PLUMMER * self.eps_soft)
        self.theta, self.eta_acc, self.eta_dyn, self.eta_cross = theta, eta_acc, eta_dyn, eta_cross
        self.r_cross = r_cross_over_soft * self.eps_soft
        self.n_levels = int(n_levels)
        self.dt_max = dt_max_gyr / GYR_PER_TIME_UNIT             # unidades internas
        self.dt_min = self.dt_max / 2 ** self.n_levels
        self.dt_tick = self.dt_min / 2.0
        self.brute_max, self.profile_every = brute_max, profile_every
        self.stop_on_weak_violation, self.weak_max = stop_on_weak_violation, weak_max
        self.stop_well_speed_kms, self.max_wall_s = stop_well_speed_kms, max_wall_s
        fk = {"r_soft": self.eps_soft, "tau_avg": tau_avg_myr * 1e-3 / GYR_PER_TIME_UNIT, "k_inner": 64}
        fk.update(field_kwargs or {})
        self.field = SphericalCronosField(self.A, **fk)
        self.field_static = bool(field_static)
        self.t = 0.0
        self.tick = 0
        self.lapse = np.ones(self.N)
        self.gamma_last = np.zeros(self.N)
        self.s_old = np.full(self.N, 2, dtype=np.int64)
        self.next_kick = np.zeros(self.N, dtype=np.int64)
        self.events: list[dict] = []
        self.n_force_calls = {"brute": 0, "tree": 0}
        self.wall = 0.0
        self._runaway = False
        self.W_fric = 0.0                    # trabajo acumulado de la fricción (energía cinética drenada)
        self._init_levels()

    # -- utilidades -----------------------------------------------------
    def radii(self):
        return np.linalg.norm(self.pos, axis=1)

    def _gravity(self, idx):
        from pytreegrav import AccelTarget
        method = "bruteforce" if len(idx) <= self.brute_max else "tree"
        self.n_force_calls["brute" if method == "bruteforce" else "tree"] += 1
        return AccelTarget(self.pos[idx], self.pos, self.mass, softening_target=self.h[idx],
                           softening_source=self.h, G=G_KPC, theta=self.theta,
                           parallel=True, method=method)

    def _cronos_terms(self, idx):
        """(a_extra, Γ, N) para las partículas idx desde el campo esférico."""
        n = len(idx)
        if not self.cronos:
            return np.zeros((n, 3)), np.zeros(n), np.ones(n)
        r = np.linalg.norm(self.pos[idx], axis=1)
        rhat = self.pos[idx] / np.clip(r, 1e-12, None)[:, None]
        eps = self.field.eps_c(r)
        a_extra = np.zeros((n, 3))
        if self.use_force:
            a_extra = (C_KMS ** 2 * self.field.deps_c_dr(r))[:, None] * rhat
        gamma = np.zeros(n)
        if self.use_friction:
            v_r = np.sum(self.vel[idx] * rhat, axis=1)
            rho = self.field.rho(r)
            rho_dot = self.field.drho_dt(r) + v_r * self.field.drho_dr(r)
            gamma = np.where(rho_dot > 0.0, 1.5 * rho_dot / rho * eps, 0.0)
        lapse = np.ones(n)
        if self.use_lapse:
            lapse = 1.0 + self.field.Phi(r) / C_KMS ** 2 - eps
        return a_extra, gamma, lapse

    def _level_steps(self, idx, acc):
        """Paso individual en ticks (potencia de 2 en [2, 2^(n_levels+1)]):
        Δt_i = min( sqrt(2η_acc ε_soft/|a_i|), η_dyn/sqrt(Gρ(r_i)),
        η_cross ε_soft/|v_i| si r_i < r_cross ). El tercer criterio resuelve
        el paso de las partículas rápidas por el pozo estrecho de Cronos
        (escala ε_soft) y solo se aplica dentro de r_cross = 10·ε_soft."""
        a_mag = np.linalg.norm(acc, axis=1)
        v_mag = np.linalg.norm(self.vel[idx], axis=1)
        r = np.linalg.norm(self.pos[idx], axis=1)
        rho = self.field.rho(r)
        dt_acc = np.sqrt(2.0 * self.eta_acc * self.eps_soft / np.clip(a_mag, 1e-30, None))
        dt_dyn = self.eta_dyn / np.sqrt(G_KPC * np.clip(rho, 1e-30, None))
        # criterio de cruce solo donde el pozo de Cronos es relevante (r < r_cross):
        # fuera, la aceleración y la densidad bastan y el coste de los pasos
        # individuales se preserva
        dt_cross = np.where(r < self.r_cross,
                            self.eta_cross * self.eps_soft / np.clip(v_mag, 1e-30, None), np.inf)
        dt = np.minimum(np.minimum(dt_acc, dt_dyn), dt_cross)
        s = 2 ** np.floor(np.log2(np.clip(dt / self.dt_tick, 2.0, 2.0 ** (self.n_levels + 1))))
        return s.astype(np.int64)

    def _init_levels(self):
        self.field.update(self.radii(), self.mass, self.t)
        if self.field_static:
            self.field.frozen = True
        idx = np.arange(self.N)
        acc = self._gravity(idx)
        a_extra, gamma, lapse = self._cronos_terms(idx)
        self.lapse = lapse
        s = self._level_steps(idx, acc + a_extra)
        self.s_old = s.copy()
        self.next_kick = s // 2                    # primer kick en el punto medio
        self._check_regime(record=True)

    def _check_regime(self, record: bool = False) -> bool:
        """Puerta de régimen: ε_c < weak_max en r ≥ ε_soft; N > 0. Registra
        además la «velocidad del pozo» v_well = sqrt(2 c² ε_c(ε_soft)) — la
        velocidad que adquiere una partícula al caer al pozo de Cronos —
        para la regla de parada por colapso progresivo (runaway)."""
        r_probe = np.geomspace(self.eps_soft, self.field.r_mid[-1], 200)
        eps = self.field.eps_c(r_probe)
        D = self.field.dominance(r_probe)
        DF = self.field.force_dominance(r_probe)
        lapse_min = float(np.min(1.0 + self.field.Phi(r_probe) / C_KMS ** 2 - eps))
        weak_ok = bool(eps.max() < self.weak_max)
        v_well = float(np.sqrt(2.0 * C_KMS ** 2 * eps[0])) if self.cronos else 0.0
        runaway = bool(self.stop_well_speed_kms is not None and v_well > self.stop_well_speed_kms)
        if record or not weak_ok or lapse_min <= 0.0 or runaway:
            i1 = np.where(DF >= 1.0)[0]
            self.events.append({"t_gyr": self.t * GYR_PER_TIME_UNIT, "eps_c_max": float(eps.max()),
                                "D_phi_max": float(D.max()), "D_phi_at_soft": float(D[0]),
                                "D_F_at_soft": float(DF[0]), "D_F_max": float(DF.max()),
                                "r_DF_equals_one_kpc": float(r_probe[i1[-1]]) if i1.size else None,
                                "v_well_kms": v_well, "runaway_stop": runaway,
                                "r_inner_field_kpc": float(self.field.r_mid[0]),
                                "lapse_min": lapse_min, "weak_regime_ok": weak_ok})
        self._runaway = runaway
        return weak_ok and lapse_min > 0.0

    # -- bucle ----------------------------------------------------------
    def snapshot(self, edges_kpc=None, with_energy: bool = True) -> dict:
        from pytreegrav import Potential
        edges = np.geomspace(0.05, 100.0, 31) if edges_kpc is None else np.asarray(edges_kpc)
        r = self.radii()
        m_shell, _ = np.histogram(r, bins=edges, weights=self.mass)
        n_shell, _ = np.histogram(r, bins=edges)
        vol = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
        r_mid = np.sqrt(edges[1:] * edges[:-1])
        snap = {"t_gyr": self.t * GYR_PER_TIME_UNIT, "tick": int(self.tick),
                "r_mid_kpc": r_mid.tolist(), "rho": (m_shell / vol).tolist(),
                "count": n_shell.tolist(),
                "M_within": {f"{rk:g}": float(self.mass[r < rk].sum()) for rk in (0.2, 0.4, 1.0, 2.3, 5.0, 10.0)},
                "N_within_0p4": int(np.sum(r < 0.4)), "N_within_2p3": int(np.sum(r < 2.3)),
                "r_half_mass_kpc": float(np.quantile(r, 0.5)),
                "field": {"r_kpc": self.field.r_mid[::6].tolist(),
                          "rho_msun_kpc3": self.field.rho(self.field.r_mid[::6]).tolist(),
                          "eps_c": self.field.eps_c(self.field.r_mid[::6]).tolist(),
                          "D_phi": self.field.dominance(self.field.r_mid[::6]).tolist(),
                          "D_F": self.field.force_dominance(self.field.r_mid[::6]).tolist()},
                "gamma_active_fraction": float(np.mean(self.gamma_last > 0.0)),
                "gamma_mean_inner_per_gyr": float(np.mean(self.gamma_last[r < 2.3]) / GYR_PER_TIME_UNIT) if np.any(r < 2.3) else 0.0,
                "lapse_min": float(self.lapse.min()), "wall_s": round(self.wall, 1),
                "force_calls": dict(self.n_force_calls)}
        if with_energy:
            phi = Potential(self.pos, self.mass, softening=self.h, G=G_KPC, theta=self.theta, parallel=True)
            K = 0.5 * float(np.sum(self.mass * np.sum(self.vel ** 2, axis=1)))
            W = 0.5 * float(np.sum(self.mass * phi))
            U = self.field.U_ext(r, self.mass) if self.cronos else 0.0
            snap["energy"] = {"K": K, "W": W, "U_cronos": U, "E": K + W + U,
                              "E_grav_only": K + W, "virial_2K_over_W": -2.0 * K / W,
                              # balance autoconsistente (frente 5b): K + W + (2/5)U_C + W_fric
                              "U_self_2_5": 0.4 * U, "W_fric": self.W_fric,
                              "E_self": K + W + 0.4 * U + self.W_fric}
        return snap

    def run(self, t_end_gyr: float, snapshot_gyr, log=None, snapshot_kwargs=None) -> dict:
        """Integra hasta t_end y toma instantáneas en snapshot_gyr (Gyr)."""
        t_end = t_end_gyr / GYR_PER_TIME_UNIT
        snaps_due = sorted(float(s) / GYR_PER_TIME_UNIT for s in snapshot_gyr)
        out = {"snapshots": [], "stopped_early": False, "stop_reason": None}
        T0 = time.time()
        if snaps_due and snaps_due[0] <= 1e-12:
            out["snapshots"].append(self.snapshot(**(snapshot_kwargs or {})))
            snaps_due.pop(0)
        if self.stop_on_weak_violation and not self.events[-1]["weak_regime_ok"]:
            out.update(stopped_early=True, stop_reason="régimen débil violado en t = 0 (ε_c ≥ weak_max en r ≥ ε_soft)")
            self.wall = time.time() - T0
            if not out["snapshots"]:
                out["snapshots"].append(self.snapshot(**(snapshot_kwargs or {})))
            out["snapshots"][-1]["at_stop"] = True
            out["events"] = self.events
            out["wall_s"] = round(self.wall, 1)
            out["t_final_gyr"] = 0.0
            return out
        tick_end = int(np.ceil(t_end / self.dt_tick - 1e-9))
        next_profile_tick = self.tick + self.profile_every
        while self.tick < tick_end:
            # salto exacto al siguiente evento (kick, perfil, instantánea o fin):
            # entre kicks las velocidades son constantes, así que el drift-all
            # por varios ticks es exacto y el coste no depende de Δt_min
            target = int(min(self.next_kick.min(), next_profile_tick, tick_end))
            if snaps_due:
                target = min(target, int(np.ceil(snaps_due[0] / self.dt_tick - 1e-9)))
            n_jump = max(1, target - self.tick)
            self.pos += self.vel * (self.lapse * (n_jump * self.dt_tick))[:, None]
            self.tick += n_jump
            self.t = self.tick * self.dt_tick
            if self.tick >= next_profile_tick:
                next_profile_tick = self.tick + self.profile_every
                self.field.update(self.radii(), self.mass, self.t)
                if self.stop_on_weak_violation or self.stop_well_speed_kms is not None:
                    ok = self._check_regime()
                    if self.stop_on_weak_violation and not ok:
                        out.update(stopped_early=True, stop_reason=f"régimen violado en t = {self.t*GYR_PER_TIME_UNIT:.4f} Gyr")
                        break
                    if self._runaway:
                        out.update(stopped_early=True, stop_reason=f"runaway: v_well > {self.stop_well_speed_kms:g} km/s en t = {self.t*GYR_PER_TIME_UNIT:.4f} Gyr")
                        break
                if self.max_wall_s is not None and time.time() - T0 > self.max_wall_s:
                    out.update(stopped_early=True, stop_reason=f"tope de pared {self.max_wall_s:g} s en t = {self.t*GYR_PER_TIME_UNIT:.4f} Gyr")
                    break
            idx = np.nonzero(self.next_kick == self.tick)[0]
            if idx.size:
                acc = self._gravity(idx)
                a_extra, gamma, lapse = self._cronos_terms(idx)
                s_new = self._level_steps(idx, acc + a_extra)
                dt_kick = 0.5 * (self.s_old[idx] + s_new) * self.dt_tick
                if self.use_friction and self.cronos:
                    v2_before = np.sum(self.vel[idx] ** 2, axis=1)
                    self.vel[idx] *= np.exp(-gamma * dt_kick)[:, None]
                    self.W_fric += 0.5 * float(np.sum(self.mass[idx] * (v2_before - np.sum(self.vel[idx] ** 2, axis=1))))
                self.vel[idx] += (acc + a_extra) * (lapse * dt_kick)[:, None]
                self.lapse[idx] = lapse
                self.gamma_last[idx] = gamma
                self.next_kick[idx] = self.tick + s_new
                self.s_old[idx] = s_new
            while snaps_due and self.t >= snaps_due[0] - 1e-9:
                self.wall = time.time() - T0
                self.field.update(self.radii(), self.mass, self.t)
                self._check_regime(record=True)
                out["snapshots"].append(self.snapshot(**(snapshot_kwargs or {})))
                snaps_due.pop(0)
                if log is not None:
                    s_ = out["snapshots"][-1]
                    log(f"    t = {s_['t_gyr']:.2f} Gyr  M(<0.4) = {s_['M_within']['0.4']:.3e}  "
                        f"M(<2.3) = {s_['M_within']['2.3']:.3e}  E = {s_.get('energy', {}).get('E', float('nan')):.4e}  "
                        f"wall {s_['wall_s']:.0f}s  calls {s_['force_calls']}")
        self.wall = time.time() - T0
        if out["stopped_early"]:
            # instantánea final en el instante de parada (estado que motivó la parada)
            self.field.update(self.radii(), self.mass, self.t)
            snap = self.snapshot(**(snapshot_kwargs or {}))
            snap["at_stop"] = True
            out["snapshots"].append(snap)
        out["events"] = self.events
        out["wall_s"] = round(self.wall, 1)
        out["t_final_gyr"] = self.t * GYR_PER_TIME_UNIT
        return out
