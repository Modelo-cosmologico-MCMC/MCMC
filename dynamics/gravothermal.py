"""Código gravotérmico 1D de la MCV (c): fluido conductivo con autointercambio y borde de ECV (bloque P1 del autor del
23-oct-2026; definición del borde del 24-oct). Comprobación interna (E8), sin datos ni letra.

QUÉ RESUELVE. La MCV de un halo aislado como fluido conductor en equilibrio hidrostático (Lynden-Bell y Eggleton 1980;
Balberg, Shapiro e Inagaki 2002; Pollack, Spergel y Steinhardt 2015; Essig et al. 2019):

  ∂M/∂r = 4πr²ρ,   ∂p/∂r = −GMρ/r²,   p = ρν²,   u = (3/2)ν²,
  ∂L/∂M = −ν²·D/Dt ln(ν³/ρ),   L/(4πr²) = −κ·∂ν²/∂r,   1/κ = 1/κ_smfp + 1/κ_lmfp,
  κ_smfp = (3/2)·b·ν/(σ_x/m),   κ_lmfp = (3/2)·a·C·(σ_x/m)·ρν³/(4πG),

con a = √(16/π), b = 25√π/32 (Chapman–Enskog de esferas duras) y C la calibración del régimen de recorrido libre largo
(LMFP), la β de la ec. 6 del bloque. El autointercambio (MCV con MCV) entra por σ_x/m constante. Unidades: r_s y ρ_s del
NFW inicial, v₀ = √(4πGρ_s)·r_s y t en r_s/v₀; el único parámetro físico es σ̂ = (σ_x/m)·ρ_s·r_s. En el régimen LMFP la
evolución solo depende de C·σ̂·t.

ESQUEMA. Capas lagrangianas de masa fija (la central es la esfera de radio r_min) y dos pasos por iteración:
  1. conducción implícita (Euler hacia atrás con κ congelada al inicio del paso): conserva la energía interna total salvo
     el flujo del borde, y no tiene límite de estabilidad;
  2. relajación adiabática al equilibrio hidrostático (Newton tridiagonal con K = p/ρ^{5/3} fijo por capa).
El paso se adapta para que ninguna capa cambie u en más de eps; si cambia más de 2·eps, el paso se repite con la mitad.

EL BORDE DE ECV. Definición del autor (24-oct): «el borde de ECV corresponde al punto de intercambio entre la burbuja de
MCV y la ECV fuera de la influencia del punto gravitacional y térmico». En el código es un radio fijo r_b, la pared donde
la burbuja toca la ECV, con dos brazos:
  - «cerrado»: sin intercambio, L(r_b) = 0;
  - «intercambio»: contacto térmico con una ECV que conserva la temperatura que tiene en el punto de intercambio (la
    inicial de la capa del borde).
Que el borde esté fuera de la influencia se COMPRUEBA en cada corrida, no se supone:
  - térmica: el flujo conductivo que llega a la capa del borde, frente al máximo del halo, queda por debajo de THERMAL_TOL;
  - gravitatoria y mecánica: duplicar r_b mueve t_c menos de BORDER_TOL (`border_shift`);
  - al pasar a unidades físicas, r_b no puede caer dentro de r200 (`collapse_time_gyr`).
El intercambio de MASA con la ECV en el borde (atrapamiento y escape de la ec. 7 del bloque) no está implementado: su ley
por halo no está dada y f_h(a) no está definida.

CONDICIÓN INICIAL. NFW truncado en r_b con la presión del NFW completo, p(r) = ∫_r^∞ ρ·GM/r'² dr', relajado al equilibrio
discreto con la pared en r_b.

COLAPSO. t_c es el instante en que la capa central alcanza rho_stop (10⁸ ρ_s por defecto). En el régimen LMFP la última
fase es tan rápida que t_c cambia menos de 10⁻⁵ entre umbrales de 10⁶ y 10⁸ ρ_s (σ̂ = 0.01); hacia el SMFP se alarga
(σ̂ = 0.3: un 0.45 %).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.integrate import quad
from scipy.linalg.lapack import dgtsv

from dynamics.mcv_equilibrium import (
    CM2_PER_G_TO_KPC2_PER_MSUN,
    G_KPC,
    GYR_PER_KPC_KMS,
    H0_DEFAULT,
    gravothermal_collapse_time,
    nfw_halo,
)

A_COEF = math.sqrt(16.0 / math.pi)
B_COEF = 25.0 * math.sqrt(math.pi) / 32.0
C_DEFAULT = 0.75
BORDERS = ("cerrado", "intercambio")
THERMAL_TOL = 1e-6        # flujo que llega a la capa del borde / máximo del halo
BORDER_TOL = 1e-3         # cambio relativo de t_c al duplicar r_b


@dataclass(frozen=True)
class GravothermalConfig:
    sigma_hat: float                  # σ̂ = (σ_x/m)·ρ_s·r_s
    C: float = C_DEFAULT
    n_shells: int = 150
    r_min: float = 1e-3               # radio de la capa central, en r_s
    r_border: float = 100.0           # borde de ECV, en r_s
    border: str = "cerrado"
    eps: float = 1e-3                 # cambio relativo máximo de u por paso
    rho_stop: float = 1e8             # ρ_c/ρ_s que marca el colapso
    max_steps: int = 1_000_000


def _tridiagonal(lower, diag, upper, rhs):
    """Sistema tridiagonal (LAPACK dgtsv): lower y upper son la sub- y la superdiagonal."""
    *_, x, info = dgtsv(lower, diag, upper, rhs)
    if info != 0 or not np.all(np.isfinite(x)):
        raise RuntimeError(f"sistema tridiagonal singular (info = {info})")
    return x


def nfw_mass(r):
    """M(<r) del NFW en unidades de 4πρ_s r_s³."""
    r = np.asarray(r, dtype=float)
    return np.log1p(r) - r / (1.0 + r)


def nfw_density(r):
    r = np.asarray(r, dtype=float)
    return 1.0 / (r * (1.0 + r) ** 2)


class GravothermalHalo:
    """Estado lagrangiano: bordes r_0 = 0 < r_1 < … < r_N = r_b, masas dM de las N capas y energía específica u."""

    def __init__(self, cfg: GravothermalConfig):
        if cfg.border not in BORDERS:
            raise ValueError(f"borde {cfg.border!r} fuera de {BORDERS}")
        if not (cfg.sigma_hat > 0.0 and cfg.C > 0.0 and 0.0 < cfg.r_min < cfg.r_border and cfg.n_shells >= 8):
            raise ValueError("configuración fuera de dominio")
        self.cfg = cfg
        self.r = np.concatenate([[0.0], np.geomspace(cfg.r_min, cfg.r_border, cfg.n_shells)])
        M = nfw_mass(self.r)
        self.dM = np.diff(M)
        self.M = M[1:]                                      # masa dentro del borde exterior de cada capa
        rc = self.r_centre
        p = np.array([quad(lambda x: nfw_density(x) * nfw_mass(x) / x ** 2, x0, np.inf, limit=200)[0] for x0 in rc])
        self.K = p / nfw_density(rc) * self.rho ** (-2.0 / 3.0)
        self.t = 0.0
        self.steps = 0
        self._dt = None
        self.relax()
        self.u_ecv = float(self.u[-1])                     # la ECV en el punto de intercambio
        self.u_init = self.u.copy()

    @property
    def rho(self):
        return self.dM / ((self.r[1:] ** 3 - self.r[:-1] ** 3) / 3.0)

    @property
    def r_centre(self):
        """Radio que divide cada capa en dos volúmenes iguales."""
        return ((self.r[1:] ** 3 + self.r[:-1] ** 3) / 2.0) ** (1.0 / 3.0)

    def hydrostatic_residual(self) -> float:
        """max |dp/dM + M/r⁴| / (M/r⁴) en los bordes interiores."""
        n = len(self.dM)
        p = self.K * self.rho ** (5.0 / 3.0)
        g = self.M[:n - 1] / self.r[1:n] ** 4
        return float(np.max(np.abs((p[1:] - p[:-1]) / (0.5 * (self.dM[:-1] + self.dM[1:])) + g) / g))

    def relax(self, tol: float = 1e-12, itmax: int = 50) -> int:
        """Equilibrio hidrostático adiabático (K fijo por capa); r_0 = 0 y r_N = r_b fijos."""
        n = len(self.dM)
        dMb = 0.5 * (self.dM[:-1] + self.dM[1:])          # masa entre los centros de dos capas vecinas
        Mi = self.M[:n - 1]
        for it in range(itmax):
            r = self.r
            vol = (r[1:] ** 3 - r[:-1] ** 3) / 3.0
            p = self.K * (self.dM / vol) ** (5.0 / 3.0)
            ri = r[1:n]
            R = (p[1:] - p[:-1]) / dMb + Mi / ri ** 4
            dp_out = (5.0 / 3.0) * p * (-(r[1:] ** 2) / vol)  # ∂p_k/∂(borde exterior de k)
            dp_in = (5.0 / 3.0) * p * (r[:-1] ** 2 / vol)     # ∂p_k/∂(borde interior de k)
            dx = _tridiagonal(-dp_in[1:n - 1] / dMb[1:], (dp_in[1:] - dp_out[:-1]) / dMb - 4.0 * Mi / ri ** 5,
                              dp_out[1:n - 1] / dMb[:-1], -R)
            lam = 1.0
            while True:
                new = r.copy()
                new[1:n] = ri + lam * dx
                if np.all(np.diff(new) > 0.0):
                    break
                lam *= 0.5
            self.r = new
            if np.max(np.abs(dx / ri)) < tol:
                break
        else:
            raise RuntimeError("la relajación hidrostática no converge")
        self.u = 1.5 * self.K * self.rho ** (2.0 / 3.0)
        return it

    def _conductances(self):
        """g_i con L_i = −g_i·(u_i − u_{i−1}) en los bordes interiores, y g_N entre la capa del borde y la ECV."""
        rho = self.rho
        nu = np.sqrt(self.u / 1.5)
        rc = self.r_centre
        k_s = 1.5 * B_COEF * nu / self.cfg.sigma_hat
        k_l = 1.5 * A_COEF * self.cfg.C * self.cfg.sigma_hat * rho * nu ** 3
        kap = 1.0 / (1.0 / k_s + 1.0 / k_l)
        k_edge = 2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])          # media armónica
        g = (2.0 / 3.0) * self.r[1:-1] ** 2 * k_edge / (rc[1:] - rc[:-1])
        g_ecv = (2.0 / 3.0) * self.r[-1] ** 2 * kap[-1] / (self.r[-1] - rc[-1])
        return g, g_ecv

    def luminosity(self):
        """L en los N + 1 bordes (unidades de 4πρ_s r_s² v₀³): L_0 = 0; L_N = 0 cerrado o el intercambio con la ECV."""
        g, g_ecv = self._conductances()
        L = np.zeros(len(self.dM) + 1)
        L[1:-1] = -g * (self.u[1:] - self.u[:-1])
        if self.cfg.border == "intercambio":
            L[-1] = -g_ecv * (self.u_ecv - self.u[-1])
        return L

    def _conduct(self, dt: float):
        g, g_ecv = self._conductances()
        c = dt / self.dM
        g_in = np.concatenate([[0.0], g])
        g_out = np.concatenate([g, [g_ecv if self.cfg.border == "intercambio" else 0.0]])
        rhs = self.u.copy()
        if self.cfg.border == "intercambio":
            rhs[-1] += c[-1] * g_ecv * self.u_ecv
        return _tridiagonal(-c[1:] * g_in[1:], 1.0 + c * (g_in + g_out), -c[:-1] * g_out[:-1], rhs)

    def step(self) -> float:
        eps = self.cfg.eps
        if self._dt is None:
            L = self.luminosity()
            dudt = -(L[1:] - L[:-1]) / self.dM
            self._dt = eps * float(np.min(self.u / np.maximum(np.abs(dudt), 1e-300)))
        while True:
            u_new = self._conduct(self._dt)
            change = float(np.max(np.abs(u_new / self.u - 1.0)))
            if change <= 2.0 * eps:
                break
            self._dt *= 0.5
        self.K = (u_new / 1.5) * self.rho ** (-2.0 / 3.0)
        self.relax()
        dt = self._dt
        self.t += dt
        self.steps += 1
        self._dt *= min(2.0, max(0.5, eps / max(change, 1e-300)))
        return dt

    def energy(self) -> float:
        """E = Σ u·dM − Σ G·M·dM/r (con la masa y el radio de cada capa en su centro)."""
        return float(np.sum(self.u * self.dM) - np.sum((self.M - 0.5 * self.dM) * self.dM / self.r_centre))


def self_similar_slope(halo: GravothermalHalo, window: tuple[float, float] = (1e-6, 1e-3)) -> dict:
    """Pendiente de ln ρ frente a ln r (mínimos cuadrados) en las capas con ρ/ρ_c dentro de la ventana, lejos del núcleo,
    donde la solución autosemejante del régimen LMFP es una ley de potencias (−2.19 en Balberg et al. 2002). Junto al
    núcleo (ρ/ρ_c ≳ 10⁻³) la pendiente pasa de −2.4, como en la esfera isoterma."""
    rho = halo.rho
    m = (rho > window[0] * rho[0]) & (rho < window[1] * rho[0])
    if m.sum() < 5:
        return {"slope": None, "n_shells": int(m.sum()), "window": list(window)}
    s = np.polyfit(np.log(halo.r_centre[m]), np.log(rho[m]), 1)[0]
    return {"slope": float(s), "n_shells": int(m.sum()), "window": list(window)}


def run_collapse(cfg: GravothermalConfig, snapshot_rho: tuple[float, ...] = ()) -> dict:
    """Evoluciona el halo hasta ρ_c = rho_stop (o max_steps) y devuelve t_c, la fase de núcleo y los diagnósticos del
    borde. snapshot_rho: densidades centrales, en la fase de colapso, a las que se mide la pendiente autosemejante."""
    halo = GravothermalHalo(cfg)
    e0 = halo.energy()
    rho_c0 = float(halo.rho[0])
    rho_min, t_min = math.inf, 0.0
    l_max = l_edge = l_exchange = 0.0
    pending = sorted(snapshot_rho)
    snapshots = []
    while True:
        halo.step()
        rc = float(halo.rho[0])
        if rc < rho_min:
            rho_min, t_min = rc, halo.t
        aL = np.abs(halo.luminosity())
        l_max, l_edge, l_exchange = max(l_max, aL.max()), max(l_edge, aL[-2]), max(l_exchange, aL[-1])
        while pending and rc >= pending[0] and rho_min < pending[0]:
            snapshots.append({"rho_c": rc, "t": halo.t, **self_similar_slope(halo)})
            pending.pop(0)
        if rc >= cfg.rho_stop or halo.steps >= cfg.max_steps:
            break
    collapsed = rc >= cfg.rho_stop
    edge_rel = l_edge / l_max
    return {"config": asdict(cfg), "collapsed": collapsed, "steps": halo.steps, "t_c": halo.t if collapsed else None,
            "C_sigma_t_c": cfg.C * cfg.sigma_hat * halo.t if collapsed else None, "rho_c_initial": rho_c0, "rho_c_end": rc,
            "rho_c_min": rho_min, "t_core_max": t_min, "energy_rel_change": (halo.energy() - e0) / abs(e0),
            "border_flux_rel": edge_rel, "exchange_flux_rel": l_exchange / l_max,
            "border_u_rel_change": float(abs(halo.u[-1] / halo.u_init[-1] - 1.0)),
            "outside_thermal_influence": bool(edge_rel < THERMAL_TOL), "snapshots": snapshots}


def with_border(cfg: GravothermalConfig, r_border: float) -> GravothermalConfig:
    """La misma configuración con el borde en r_border y el mismo espaciado logarítmico de las capas."""
    n = 1 + round((cfg.n_shells - 1) * math.log(r_border / cfg.r_min) / math.log(cfg.r_border / cfg.r_min))
    return replace(cfg, r_border=r_border, n_shells=n)


def border_shift(cfg: GravothermalConfig, factor: float = 2.0) -> dict:
    """t_c con el borde en r_b y en factor·r_b (mismo espaciado): el borde está fuera de la influencia gravitatoria y
    mecánica si el cambio relativo queda por debajo de BORDER_TOL, y fuera de la térmica si ninguna de las dos corridas
    lleva flujo al borde."""
    near = run_collapse(cfg)
    far = run_collapse(with_border(cfg, factor * cfg.r_border))
    if not (near["collapsed"] and far["collapsed"]):
        return {"near": near, "far": far, "rel": None, "outside_influence": False}
    rel = far["t_c"] / near["t_c"] - 1.0
    return {"near": near, "far": far, "rel": rel,
            "outside_influence": bool(abs(rel) < BORDER_TOL and near["outside_thermal_influence"] and far["outside_thermal_influence"])}


# ------------------------------------------------------------------ paso a unidades físicas (las de mcv_equilibrium)
def time_unit_gyr(rho_s: float) -> float:
    """r_s/v₀ = 1/√(4πGρ_s) en Gyr, con ρ_s en M☉/kpc³."""
    return GYR_PER_KPC_KMS / math.sqrt(4.0 * math.pi * G_KPC * rho_s)


def sigma_hat(sigma_over_m: float, rho_s: float, r_s: float) -> float:
    """σ̂ = (σ_x/m)·ρ_s·r_s, con σ_x/m en cm²/g, ρ_s en M☉/kpc³ y r_s en kpc."""
    return sigma_over_m * CM2_PER_G_TO_KPC2_PER_MSUN * rho_s * r_s


def collapse_time_gyr(M200: float, c: float, sigma_over_m: float, cfg: GravothermalConfig | None = None,
                      H0: float = H0_DEFAULT) -> dict:
    """t_c del código para una envoltura NFW (M200 a 200·ρ_crit) en Gyr, junto al t_c de Essig et al. (2019) con β = C.
    Falla cerrado si el borde de ECV cae dentro de r200 (dentro de la influencia gravitatoria del halo)."""
    halo = nfw_halo(M200, c, H0)
    sh = sigma_hat(sigma_over_m, halo["rho_s"], halo["r_s"])
    cfg = replace(cfg or GravothermalConfig(sigma_hat=sh), sigma_hat=sh)
    if cfg.r_border < c:
        raise ValueError(f"el borde de ECV (r_b = {cfg.r_border} r_s) cae dentro de r200 = {c} r_s")
    res = run_collapse(cfg)
    tu = time_unit_gyr(halo["rho_s"])
    return {**res, "M200": M200, "c": c, "sigma_over_m": sigma_over_m, "sigma_hat": sh, "time_unit_gyr": tu,
            "t_c_gyr": res["t_c"] * tu if res["collapsed"] else None, "r_border_over_r200": cfg.r_border / c,
            "t_c_essig_gyr": gravothermal_collapse_time(sigma_over_m, halo["rho_s"], halo["r_s"], beta=cfg.C)}
