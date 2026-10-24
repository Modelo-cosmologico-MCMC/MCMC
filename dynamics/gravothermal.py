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
MCV y la ECV fuera de la influencia del punto gravitacional y térmico». En el código es una pared en r_b donde la burbuja
toca la ECV, con dos brazos térmicos:
  - «cerrado»: sin intercambio, L(r_b) = 0;
  - «intercambio»: contacto térmico con una ECV que conserva la temperatura que tiene en el punto de intercambio (la
    inicial de la capa del borde).
La pared puede ser:
  - «fija» en r_b (100 r_s por defecto; sin la fuerza de la ECV queda como control numérico);
  - «R0»: la pared sigue al radio de aceleración nula R₀ de la masa del propio código (dynamics/ecv_border.py), con la
    fuerza de la ECV en el equilibrio. Sin intercambio de masa no se mueve. Se construye con `ecv_config`.
Que el borde esté fuera de la influencia se COMPRUEBA en cada corrida, no se supone:
  - térmica: el flujo conductivo que llega a la capa del borde, frente al máximo del halo, queda por debajo de THERMAL_TOL;
  - sin la fuerza de la ECV y con la pared fija: duplicar r_b mueve t_c menos de BORDER_TOL (`border_shift`);
  - con la fuerza de la ECV: duplicar el número de capas mueve t_c menos de RESOLUTION_TOL (`resolution_shift`), y la
    fuerza neta en la pared es nula dentro de la discretización. Duplicar el borde deja de tener sentido: más allá de R₀
    la ECV empuja hacia fuera, y una pared más lejana sujetaría gas que se escaparía. Por eso falla cerrado una pared más
    allá de R₀;
  - al pasar a unidades físicas, r_b no puede caer dentro de r200 (`collapse_time_gyr`).

EL INTERCAMBIO DE MASA CON LA ECV (documento vivo del 24-oct, PR-64). Con la pared «R0», la capa del borde gana o pierde
masa con la ley de borde de dynamics/ecv_exchange.py, con su ρ_b y su ν_b:
  - barrido (ξ_s): autoalimentado, porque el borde avanza con la masa que atrapa;
  - condensación (ξ·ρ_E) y evaporación (ρ_b), en forma de Hertz–Knudsen o en su límite lineal de Onsager.
El brazo A (solo atrapamiento) es barrido y condensación; los de dos sentidos, cualquier combinación con evaporación.
Lo que entra lleva la energía de efusión de la ECV (2ν_ecv² por unidad de masa), lo que sale la de la capa del borde
(2ν_b²), lo barrido entra con u_ecv y un borde que retrocede suelta MCV con u_b. La pared sigue al R₀ de la nueva masa, y
la capa del borde se parte o se funde con su vecina para mantener la resolución. Masa y energía cierran: el cambio de la
energía es lo que trae el intercambio menos el trabajo sobre la pared, salvo la deriva del propio código. Las leyes
repartidas por el pozo o fuera del halo (reparto del fondo, volumen de la burbuja, dilatación temporal, escape constante,
reversión en el océano, objeto crítico) no actúan en el borde de un halo aislado: van en la suma sobre la función de masa
(PR-65).

LA FUERZA DE LA ECV (documento vivo del 24-oct, PR-63). Con la gravedad del bloque, ∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E), el
equilibrio hidrostático pasa a dp/dr = −ρ·[GM/r² − (8πG/3)ρ_E·r]: en unidades del código, la masa efectiva es
m − (2/3)·q_E·r³, con q_E = ρ_E/ρ_s constante (ρ_E del fondo ΛCDM; su variación por Q es de segundo orden), y la energía
suma el potencial −(q_E/3)·r² de la ECV. Con q_E = 0 el código es el de #61, cifra a cifra.

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
from scipy.optimize import brentq

from dynamics.ecv_exchange import CONTACT_FORMS, HK
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
WALLS = ("fija", "R0")
THERMAL_TOL = 1e-6        # flujo que llega a la capa del borde / máximo del halo
BORDER_TOL = 1e-3         # cambio relativo de t_c al duplicar r_b (pared fija sin la fuerza de la ECV)
RESOLUTION_TOL = 1e-3     # cambio relativo de t_c al duplicar las capas (con la fuerza de la ECV)


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
    q_E: float = 0.0                  # ρ_E/ρ_s: la repulsión de la ECV en el equilibrio (0: sin ella, como #61)
    wall: str = "fija"                # «fija» en r_border, o «R0» (sigue a R₀; r_border es el R₀ inicial)
    xi: float = 0.0                   # contacto: condensación ξ·ρ_E de la ECV en la capa del borde (0: ninguna)
    xi_s: float = 0.0                 # barrido: fracción de la ECV barrida por el borde que queda atrapada
    evaporation: bool = False         # contacto: evaporación de la MCV del borde hacia la ECV (E2)
    contact_form: str = "hertz-knudsen"   # o «onsager-lineal» (su límite lineal; necesita ξ > 0 y evaporación)

    @property
    def exchange(self) -> bool:
        return self.xi != 0.0 or self.xi_s != 0.0 or self.evaporation


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


def zero_acceleration_radius_code(M_tot: float, q_E: float) -> float:
    """R₀ en r_s para una masa M_tot (unidades de 4πρ_s r_s³) rodeada de ECV con q_E = ρ_E/ρ_s: R₀³ = (3/2)·M_tot/q_E."""
    return (1.5 * M_tot / q_E) ** (1.0 / 3.0)


def r0_selfconsistent_code(q_E: float) -> float:
    """R₀ autoconsistente del NFW prolongado, en r_s: la raíz de x³ = (3/2)·m(x)/q_E (la de dynamics/ecv_border.py)."""
    if not 0.0 < q_E < 0.1:
        raise ValueError(f"q_E = {q_E} fuera de dominio")
    return brentq(lambda x: x ** 3 - 1.5 * float(nfw_mass(x)) / q_E, 1.0, 1e8, xtol=1e-14, rtol=1e-14)


class GravothermalHalo:
    """Estado lagrangiano: bordes r_0 = 0 < r_1 < … < r_N = r_b, masas dM de las N capas y energía específica u."""

    def __init__(self, cfg: GravothermalConfig):
        if cfg.border not in BORDERS:
            raise ValueError(f"borde {cfg.border!r} fuera de {BORDERS}")
        if cfg.wall not in WALLS:
            raise ValueError(f"pared {cfg.wall!r} fuera de {WALLS}")
        if not (cfg.sigma_hat > 0.0 and cfg.C > 0.0 and 0.0 < cfg.r_min < cfg.r_border and cfg.n_shells >= 8
                and cfg.q_E >= 0.0):
            raise ValueError("configuración fuera de dominio")
        if cfg.wall == "R0" and (cfg.q_E <= 0.0 or abs(cfg.r_border / r0_selfconsistent_code(cfg.q_E) - 1.0) > 1e-9):
            raise ValueError("la pared «R0» necesita q_E > 0 y r_border = R₀ autoconsistente (usa ecv_config)")
        if cfg.exchange:
            if cfg.wall != "R0":
                raise ValueError("el intercambio de masa ocurre en el punto de intercambio: necesita la pared «R0»")
            if not 0.0 <= cfg.xi_s < 2.0 or cfg.contact_form not in CONTACT_FORMS:
                raise ValueError("intercambio fuera de dominio (0 ≤ ξ_s < 2; forma de Hertz–Knudsen u Onsager lineal)")
            if cfg.contact_form == "onsager-lineal" and not (cfg.xi > 0.0 and cfg.evaporation):
                raise ValueError("la forma de Onsager lineal es el contacto completo: necesita ξ > 0 y evaporación")
        self.cfg = cfg
        self.r = np.concatenate([[0.0], np.geomspace(cfg.r_min, cfg.r_border, cfg.n_shells)])
        M = nfw_mass(self.r)
        self.dM = np.diff(M)
        self.M = M[1:]                                      # masa dentro del borde exterior de cada capa
        if cfg.q_E > 0.0 and cfg.r_border > zero_acceleration_radius_code(self.M[-1], cfg.q_E) * (1.0 + 1e-9):
            raise ValueError(f"con la fuerza de la ECV, la pared (r_b = {cfg.r_border:g} r_s) queda más allá de R₀ = "
                             f"{zero_acceleration_radius_code(self.M[-1], cfg.q_E):.4g} r_s: sujetaría gas que se escaparía")
        rc = self.r_centre
        p = np.array([quad(lambda x: nfw_density(x) * nfw_mass(x) / x ** 2, x0, np.inf, limit=200)[0] for x0 in rc])
        self.K = p / nfw_density(rc) * self.rho ** (-2.0 / 3.0)
        self.t = 0.0
        self.steps = 0
        self._dt = None
        self.relax()
        self.u_ecv = float(self.u[-1])                     # la ECV en el punto de intercambio
        self.u_init = self.u.copy()
        self.M_init = float(self.M[-1])
        self.exchange_log = {"condensed": 0.0, "evaporated": 0.0, "swept": 0.0, "energy_in": 0.0, "energy_out": 0.0,
                             "energy_brought": 0.0, "wall_work": 0.0, "splits": 0, "merges": 0}

    @property
    def rho(self):
        return self.dM / ((self.r[1:] ** 3 - self.r[:-1] ** 3) / 3.0)

    @property
    def r_centre(self):
        """Radio que divide cada capa en dos volúmenes iguales."""
        return ((self.r[1:] ** 3 + self.r[:-1] ** 3) / 2.0) ** (1.0 / 3.0)

    def hydrostatic_residual(self) -> float:
        """max |dp/dM + m_ef/r⁴| / (M/r⁴) en los bordes interiores, con m_ef = M − (2/3)·q_E·r³."""
        n = len(self.dM)
        p = self.K * self.rho ** (5.0 / 3.0)
        ri, Mi = self.r[1:n], self.M[:n - 1]
        g = Mi / ri ** 4
        g_eff = (Mi - (2.0 / 3.0) * self.cfg.q_E * ri ** 3) / ri ** 4
        return float(np.max(np.abs((p[1:] - p[:-1]) / (0.5 * (self.dM[:-1] + self.dM[1:])) + g_eff) / g))

    def wall_net_force(self) -> float:
        """Fuerza neta en la pared en unidades de la atracción allí: (M − (2/3)·q_E·r_b³)/M; nula en R₀."""
        return float((self.M[-1] - (2.0 / 3.0) * self.cfg.q_E * self.r[-1] ** 3) / self.M[-1])

    def relax(self, tol: float = 1e-12, itmax: int = 50) -> int:
        """Equilibrio hidrostático adiabático (K fijo por capa), con la fuerza de la ECV; r_0 = 0 y r_N fijos."""
        n = len(self.dM)
        dMb = 0.5 * (self.dM[:-1] + self.dM[1:])          # masa entre los centros de dos capas vecinas
        Mi = self.M[:n - 1]
        q = self.cfg.q_E
        for it in range(itmax):
            r = self.r
            vol = (r[1:] ** 3 - r[:-1] ** 3) / 3.0
            p = self.K * (self.dM / vol) ** (5.0 / 3.0)
            ri = r[1:n]
            R = (p[1:] - p[:-1]) / dMb + (Mi - (2.0 / 3.0) * q * ri ** 3) / ri ** 4
            dp_out = (5.0 / 3.0) * p * (-(r[1:] ** 2) / vol)  # ∂p_k/∂(borde exterior de k)
            dp_in = (5.0 / 3.0) * p * (r[:-1] ** 2 / vol)     # ∂p_k/∂(borde interior de k)
            dx = _tridiagonal(-dp_in[1:n - 1] / dMb[1:],
                              (dp_in[1:] - dp_out[:-1]) / dMb - 4.0 * Mi / ri ** 5 + (2.0 / 3.0) * q / ri ** 2,
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

    def exchange_rates(self) -> dict:
        """El intercambio de masa en la capa del borde (dynamics/ecv_exchange.py), en unidades del código: masa en
        4πρ_s r_s³ por r_s/v₀, con R₀ = r_N, ρ_b y ν_b de la capa del borde. El flujo entrante lleva la energía de efusión
        de la ECV, 2ν_ecv² por unidad de masa, y el saliente la de la capa del borde, 2ν_b²; así el equilibrio del contacto
        (ρ_b = ξρ_E con T_b = T_ecv) lo es en masa y en energía. La ECV barrida entra con u_ecv, y un borde que retrocede
        suelta MCV con u_b. El barrido se alimenta del propio intercambio (halo aislado, sin acreción)."""
        cfg = self.cfg
        area = float(self.r[-1]) ** 2
        rho_b, u_b = float(self.rho[-1]), float(self.u[-1])
        nu_b, nu2_e = math.sqrt(u_b / 1.5), self.u_ecv / 1.5
        if cfg.contact_form == "onsager-lineal":
            contact = area * HK * rho_b * nu_b * math.log(cfg.xi * cfg.q_E / rho_b)
            cond, evap = max(contact, 0.0), max(-contact, 0.0)
        else:
            cond = area * HK * nu_b * cfg.xi * cfg.q_E
            evap = area * HK * nu_b * rho_b if cfg.evaporation else 0.0
            contact = cond - evap
        total = contact / (1.0 - 0.5 * cfg.xi_s)
        sweep = total - contact
        energy_in = 2.0 * nu2_e * max(cond, 0.0) + (self.u_ecv * sweep if sweep > 0.0 else 0.0)
        energy_out = 2.0 * nu_b ** 2 * (evap + max(-cond, 0.0)) + (-u_b * sweep if sweep < 0.0 else 0.0)
        return {"condensation": cond, "evaporation": evap, "contact": contact, "sweep": sweep, "total": total,
                "energy_in": energy_in, "energy_out": energy_out, "rho_b": rho_b, "nu_b": nu_b}

    def _border_potential_energy(self) -> float:
        """El término de la capa del borde en el potencial de energy(), con la geometría actual."""
        rc = float(self.r_centre[-1])
        return float(-(self.M[-1] - 0.5 * self.dM[-1]) * self.dM[-1] / rc - (self.cfg.q_E / 3.0) * rc ** 2 * self.dM[-1])

    def _apply_exchange(self, rates: dict, dt: float, u_new) -> None:
        """Suma a la capa del borde la masa y la energía interna del intercambio durante dt (explícito), y registra la
        energía que trae, interna y potencial, con la geometría del paso."""
        dm = rates["total"] * dt
        dU = (rates["energy_in"] - rates["energy_out"]) * dt
        U = u_new[-1] * self.dM[-1] + dU
        pot_before = self._border_potential_energy()
        self.dM[-1] += dm
        self.M[-1] += dm
        u_new[-1] = U / self.dM[-1]
        log = self.exchange_log
        log["energy_brought"] += dU + self._border_potential_energy() - pot_before
        log["condensed"] += max(rates["condensation"], 0.0) * dt
        log["evaporated"] += (rates["evaporation"] + max(-rates["condensation"], 0.0)) * dt
        log["swept"] += rates["sweep"] * dt
        log["energy_in"] += rates["energy_in"] * dt
        log["energy_out"] += rates["energy_out"] * dt

    def _move_wall(self) -> None:
        """La pared «R0» sigue al radio de aceleración nula de la masa del código; registra el trabajo p·dV."""
        r_old, r_new = float(self.r[-1]), zero_acceleration_radius_code(float(self.M[-1]), self.cfg.q_E)
        p_b = float(self.K[-1] * self.rho[-1] ** (5.0 / 3.0))
        self.exchange_log["wall_work"] += p_b * (r_new ** 3 - r_old ** 3) / 3.0
        self.r[-1] = r_new

    def _regrid_border(self) -> None:
        """Mantiene la resolución en el borde: parte la capa del borde en dos de igual masa y densidad si pesa más del
        doble que su vecina, y la funde con ella si pesa menos de un cuarto (sin cambiar masa ni energía interna)."""
        if self.dM[-1] > 2.0 * self.dM[-2]:
            r_in, r_out = float(self.r[-2]), float(self.r[-1])
            r_mid = ((r_in ** 3 + r_out ** 3) / 2.0) ** (1.0 / 3.0)
            half = 0.5 * self.dM[-1]
            self.r = np.concatenate([self.r[:-1], [r_mid, r_out]])
            self.dM = np.concatenate([self.dM[:-1], [half, half]])
            self.K = np.concatenate([self.K, [self.K[-1]]])
            self.u = np.concatenate([self.u, [self.u[-1]]])
            self.u_init = np.concatenate([self.u_init, [self.u_init[-1]]])
            self.exchange_log["splits"] += 1
        elif self.dM[-1] < 0.25 * self.dM[-2]:
            m = self.dM[-2] + self.dM[-1]
            U = self.u[-2] * self.dM[-2] + self.u[-1] * self.dM[-1]
            self.r = np.concatenate([self.r[:-2], [self.r[-1]]])
            self.dM = np.concatenate([self.dM[:-2], [m]])
            self.u = np.concatenate([self.u[:-2], [U / m]])
            self.K = np.concatenate([self.K[:-2], [(self.u[-1] / 1.5) * self.rho[-1] ** (-2.0 / 3.0)]])
            self.u_init = self.u_init[:-1]
            self.exchange_log["merges"] += 1
        else:
            return
        self.M = np.cumsum(self.dM)

    def step(self) -> float:
        eps = self.cfg.eps
        if self._dt is None:
            L = self.luminosity()
            dudt = -(L[1:] - L[:-1]) / self.dM
            self._dt = eps * float(np.min(self.u / np.maximum(np.abs(dudt), 1e-300)))
        rates = self.exchange_rates() if self.cfg.exchange else None
        if rates is not None:                               # el intercambio no cambia la capa del borde más de eps por paso
            self._dt = min(self._dt, eps * self.dM[-1] / max(abs(rates["total"]), 1e-300),
                           eps * self.u[-1] * self.dM[-1] / max(abs(rates["energy_in"] - rates["energy_out"]), 1e-300))
        while True:
            u_new = self._conduct(self._dt)
            change = float(np.max(np.abs(u_new / self.u - 1.0)))
            if change <= 2.0 * eps:
                break
            self._dt *= 0.5
        if rates is not None:
            self._apply_exchange(rates, self._dt, u_new)
        self.K = (u_new / 1.5) * self.rho ** (-2.0 / 3.0)
        if rates is not None:
            self._move_wall()
            self.u = u_new
            self._regrid_border()
        self.relax()
        dt = self._dt
        self.t += dt
        self.steps += 1
        self._dt *= min(2.0, max(0.5, eps / max(change, 1e-300)))
        return dt

    def energy(self) -> float:
        """E = Σ u·dM − Σ G·M·dM/r − Σ (q_E/3)·r²·dM (con la masa y el radio de cada capa en su centro; el último término
        es el potencial de la ECV)."""
        rc = self.r_centre
        return float(np.sum(self.u * self.dM) - np.sum((self.M - 0.5 * self.dM) * self.dM / rc)
                     - (self.cfg.q_E / 3.0) * np.sum(rc ** 2 * self.dM))


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
    wall_force0 = halo.wall_net_force()
    r_wall0 = float(halo.r[-1])
    rates0 = halo.exchange_rates() if cfg.exchange else None
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
            "outside_thermal_influence": bool(edge_rel < THERMAL_TOL), "wall_net_force": wall_force0,
            "wall_net_force_end": halo.wall_net_force(), "snapshots": snapshots,
            "exchange": None if rates0 is None else {
                "initial_rates": rates0, "initial_rate_per_mass": rates0["total"] / halo.M_init,
                "mass_ratio_end": float(halo.M[-1]) / halo.M_init, "wall_ratio_end": float(halo.r[-1]) / r_wall0,
                "n_shells_end": len(halo.dM),
                # E_fin − E_ini − (energía traída por el intercambio) + (trabajo sobre la pared), frente a |E_ini|
                "energy_budget_residual": (halo.energy() - e0 - halo.exchange_log["energy_brought"]
                                           + halo.exchange_log["wall_work"]) / abs(e0),
                **halo.exchange_log}}


def with_border(cfg: GravothermalConfig, r_border: float) -> GravothermalConfig:
    """La misma configuración con el borde en r_border y el mismo espaciado logarítmico de las capas."""
    n = 1 + round((cfg.n_shells - 1) * math.log(r_border / cfg.r_min) / math.log(cfg.r_border / cfg.r_min))
    return replace(cfg, r_border=r_border, n_shells=n)


def ecv_config(sigma_hat: float, c: float, z: float = 0.0, base: GravothermalConfig | None = None) -> GravothermalConfig:
    """Halo con la fuerza de la ECV y la pared «R0»: q_E = ρ_E/ρ_s del NFW con concentración c (M200 a 200ρ_c(z),
    Planck 2018) y el borde en su R₀ autoconsistente, con el espaciado de las capas de `base`."""
    from dynamics.ecv_border import rho_s_over_rho_ecv

    base = base or GravothermalConfig(sigma_hat=sigma_hat)
    q_E = 1.0 / rho_s_over_rho_ecv(c, z)
    return replace(with_border(base, r0_selfconsistent_code(q_E)), sigma_hat=sigma_hat, q_E=q_E, wall="R0")


def code_outskirts(c: float, z: float = 0.0, base: GravothermalConfig | None = None) -> dict:
    """Las afueras de la decisión 6 tal como las da el propio código: ρ_b y ν_b de la capa del borde del halo relajado con
    la fuerza de la ECV y la pared en R₀ (en el formato de dynamics/ecv_exchange.nfw_outskirts; 600 capas hasta 100 r_s
    por defecto). Sin la fuerza de la ECV, la capa del borde converge al NFW prolongado; con ella, el gas se acumula
    contra la pared y el borde es más denso y más caliente."""
    from dynamics.ecv_exchange import nfw_outskirts

    o = nfw_outskirts(c, z)
    h = GravothermalHalo(ecv_config(0.01, c, z, base=base or GravothermalConfig(sigma_hat=0.01, n_shells=600)))
    return {**o, "rho_b": float(h.rho[-1]), "nu_b": math.sqrt(float(h.u[-1]) / 1.5), "n_shells": len(h.dM)}


def resolution_shift(cfg: GravothermalConfig) -> dict:
    """t_c con n y con 2n − 1 capas en el mismo intervalo: con la fuerza de la ECV, la comprobación de que el borde no
    depende de la discretización (cambio relativo por debajo de RESOLUTION_TOL), junto a la térmica de las dos corridas."""
    coarse = run_collapse(cfg)
    fine = run_collapse(replace(cfg, n_shells=2 * cfg.n_shells - 1))
    if not (coarse["collapsed"] and fine["collapsed"]):
        return {"coarse": coarse, "fine": fine, "rel": None, "converged": False}
    rel = fine["t_c"] / coarse["t_c"] - 1.0
    return {"coarse": coarse, "fine": fine, "rel": rel,
            "converged": bool(abs(rel) < RESOLUTION_TOL and coarse["outside_thermal_influence"]
                              and fine["outside_thermal_influence"])}


def border_shift(cfg: GravothermalConfig, factor: float = 2.0) -> dict:
    """t_c con el borde en r_b y en factor·r_b (mismo espaciado): el borde está fuera de la influencia gravitatoria y
    mecánica si el cambio relativo queda por debajo de BORDER_TOL, y fuera de la térmica si ninguna de las dos corridas
    lleva flujo al borde. Solo para la pared fija sin la fuerza de la ECV (control): con ella, alejar la pared la deja más
    allá de R₀, y la comprobación es `resolution_shift`."""
    if cfg.q_E > 0.0 or cfg.wall != "fija":
        raise ValueError("duplicar el borde solo vale para la pared fija sin la fuerza de la ECV; usa resolution_shift")
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
                      H0: float = H0_DEFAULT, ecv: bool = False) -> dict:
    """t_c del código para una envoltura NFW (M200 a 200·ρ_crit) en Gyr, junto al t_c de Essig et al. (2019) con β = C.
    Con ecv = True, la fuerza de la ECV y la pared en el R₀ autoconsistente (`ecv_config`, con el espaciado de cfg).
    Falla cerrado si el borde de ECV cae dentro de r200 (dentro de la influencia gravitatoria del halo)."""
    halo = nfw_halo(M200, c, H0)
    sh = sigma_hat(sigma_over_m, halo["rho_s"], halo["r_s"])
    cfg = replace(cfg or GravothermalConfig(sigma_hat=sh), sigma_hat=sh)
    if ecv:
        cfg = ecv_config(sh, c, base=cfg)
    if cfg.r_border < c:
        raise ValueError(f"el borde de ECV (r_b = {cfg.r_border} r_s) cae dentro de r200 = {c} r_s")
    res = run_collapse(cfg)
    tu = time_unit_gyr(halo["rho_s"])
    return {**res, "M200": M200, "c": c, "sigma_over_m": sigma_over_m, "sigma_hat": sh, "time_unit_gyr": tu,
            "t_c_gyr": res["t_c"] * tu if res["collapsed"] else None, "r_border_over_r200": cfg.r_border / c,
            "t_c_essig_gyr": gravothermal_collapse_time(sigma_over_m, halo["rho_s"], halo["r_s"], beta=cfg.C)}
