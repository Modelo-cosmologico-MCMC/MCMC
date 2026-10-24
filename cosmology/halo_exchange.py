"""Q(z) del intercambio MCV–ECV sumado sobre los halos, y todas las posibilidades del brazo de dos sentidos con sus
consecuencias para la ontología (documento vivo del 24-oct-2026, PR-65). Sin datos y sin letra.

DECISIÓN DEL AUTOR (24-oct). «Respecto al brazo en dos sentidos, prefiero probar todas las posibilidades con las
consecuencias a la ontología y a partir de ahí hacer las pruebas para ver qué resultado nos dan.» Este módulo no elige
ley. Suma cada ley por halo sobre la función de masa, la combina con cada escape y devuelve sus consecuencias y su
respuesta en el fondo. La letra queda para la preinscripción del ajuste, con todos los brazos congelados antes de los datos.

EL FONDO (ec. 7 del bloque). ρ̇_M + 3Hρ_M = Q y ρ̇_E = −Q, con Q = q·H·ρ_E. Según la recomendación del documento, f_h no
se define aparte: Q es la suma de la ley por halo sobre la función de masa de halos anfitriones, solo donde la burbuja toca
océano libre,

  Q(z) = a⁻³·P_océano(z)·∫(dn_c/dM)·Ṁ_x(M, z)·dM,   P_océano = e^{−N_b},   N_b = k·f_h·ρ̄_m/(2ρ_E),

con k la masa dentro de R₀ en unidades de la del halo (1, o 2 de control) y el cierre lineal 1 − N_b de control. Los
subhalos quedan dentro de la burbuja de su anfitrión. El balance de hoy es Δ = (ρ̄_M(0) − ρ̄_m,CMB(0))/ρ̄_M(0),
integrando el fondo con H de ΛCDM (Q pequeño) desde z_ini. Con H fijo, el fondo es lineal en q.

LAS LEYES (tabla del documento; el encaje con la ontología es el que da el documento):
  Atrapamiento:
  - T1, reparto del fondo: la ec. 7 tal como está, repartida en proporción a la masa: q = ξ_t·f_h. Encaje bajo.
  - T2, volumen de la burbuja: la ECV dentro de R₀ se convierte a ritmo ξ_t·H: q = ξ_t·(1 − P_océano). Encaje medio.
  - T3, barrido del borde (ξ_s): el borde engulle la ECV al avanzar. Se alimenta de la acreción (ρ̄_m·ḟ_h) y de lo que
    las otras leyes por halo añaden o quitan, con el factor (ξ_s/2)/(1 − ξ_s/2). Encaje alto.
  - T4, peso de la dilatación temporal: atrapamiento ∝ ρ_E·(−Φ)/c² en todo el pozo (Ṁ ∝ M^{5/3}). Como T2, actúa en el
    interior de las burbujas, con su corrección de solape (1 − P_océano)/N_b. La normalización es libre; aquí, declarada,
    la que iguala T4 y T2 hoy con la misma constante, de modo que T4 solo difiere en el peso de las masas y en la
    época. Encaje medio.
  - Tc, condensación en el borde: mitad del contacto, ξ·ρ_E·ν_b/√(2π) por unidad de área. Encaje el más alto.
  Escape:
  - E1, escape constante: la ec. 7 tal como está, q = ξ_e. Encaje bajo.
  - E2, evaporación en el borde: ρ_b·ν_b/√(2π) por unidad de área, sin constante libre. Encaje alto.
  - E3, reversión en el océano: la MCV fuera de toda burbuja vuelve a ECV a ritmo ξ_e·H (Maestro §3.9.5).
    q = ξ_e·(1 − k·f_h)·(ρ̄_m/ρ_E)·P_océano. Encaje medio.
  - E4, objeto crítico: conversión Mp → Ep en el centro (Maestro §3.10.3). Con los agujeros negros en 10⁻⁵ de la materia
    (Graham y Driver 2007) y un ritmo ξ_4·H, es una cota: q ≤ ξ_4·f_BH·ρ̄_m/ρ_E. Encaje alto, pero no sirve al fondo.
  El contacto (C) es Tc + E2 con una sola ξ. Su equilibrio es ξ = ρ_b/ρ_E.

LAS AFUERAS (decisión 6): el NFW prolongado (las cifras del documento), la capa del borde del código gravotérmico con la
fuerza de la ECV (más densa y más caliente: PR-64) o la media del entorno. La concentración es c(M, z) de Dutton y Macciò
(2014), extrapolada por debajo de su rango simulado, o c = 10 para todos los halos (las cifras del documento).

Cosmología: Planck 2018; espectro de Eisenstein y Hu (1998) sin oscilaciones; crecimiento lineal de ΛCDM; Sheth y Tormen
(1999), con Press y Schechter (1974) de control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import erfc

from dynamics.ecv_border import E_of_z, rho_crit
from dynamics.ecv_exchange import border_rates, environment_outskirts, nfw_outskirts

COSMO = {"h": 0.674, "Omega_m": 0.315, "Omega_L": 0.685, "Omega_b": 0.0493, "n_s": 0.965, "sigma8": 0.811, "T_cmb": 2.7255}
DELTA_C = 1.686
F_BH = 1.0e-5                         # ρ_BH/ρ̄_m de hoy (Graham y Driver 2007), cota para E4
TRAPPING = ("T1", "T2", "T3", "T4", "Tc")
ESCAPE = ("E1", "E2", "E3", "E4")
PER_HALO = ("T1", "T2", "T4", "Tc", "E2", "E4")     # cambian la masa de cada halo: alimentan el barrido
ST_PARAMS = (0.3222, 0.707, 0.3)

LAW_ONTOLOGY = {
    "T1": {"nombre": "reparto del fondo", "donde": "en ningún sitio concreto", "mecanismo": False, "constante": "ξ_t",
           "se_apaga_con_las_burbujas": False, "encaje": "bajo",
           "consecuencia": "atrapamiento sin lugar ni mecanismo: la ec. 7 tal como está, igual en todo halo y a todo z, "
                           "también antes de que exista el océano; no usa el borde del autor"},
    "T2": {"nombre": "volumen de la burbuja", "donde": "todo el interior de la burbuja", "mecanismo": True, "constante": "ξ_t",
           "se_apaga_con_las_burbujas": False, "encaje": "medio",
           "consecuencia": "atrapamiento gravitatorio, pero no en el punto de intercambio: la ECV dentro de R₀ se convierte "
                           "aunque no toque el borde; se satura (no se apaga) cuando las burbujas llenan el espacio"},
    "T3": {"nombre": "barrido del borde", "donde": "borde", "mecanismo": True, "constante": "ξ_s",
           "se_apaga_con_las_burbujas": True, "encaje": "alto",
           "consecuencia": "el borde engulle la ECV al avanzar: atrapa «al formarse estructuras»; nulo en un halo aislado "
                           "que no cambia de masa; se alimenta de todo lo que cambia la masa del halo"},
    "T4": {"nombre": "peso de la dilatación temporal", "donde": "interior del pozo", "mecanismo": True, "constante": "ξ_4",
           "se_apaga_con_las_burbujas": True, "encaje": "medio",
           "consecuencia": "«la dilatación temporal es la dilatación de esa misma energía»: atrapa en todo el pozo, lejos "
                           "del borde, con normalización libre; los cúmulos dominan (Ṁ ∝ M^{5/3})"},
    "Tc": {"nombre": "condensación en el borde", "donde": "borde", "mecanismo": True, "constante": "ξ",
           "se_apaga_con_las_burbujas": True, "encaje": "el más alto",
           "consecuencia": "la ECV se condensa en MCV en la cima de la barrera, a la temperatura del punto de intercambio: "
                           "la mitad del contacto, que con E2 cierra el intercambio reversible con una sola constante"},
    "E1": {"nombre": "escape constante", "donde": "en ningún sitio concreto", "mecanismo": False, "constante": "ξ_e",
           "se_apaga_con_las_burbujas": False, "encaje": "bajo",
           "consecuencia": "escape sin lugar: la ec. 7 tal como está; no depende de la MCV que haya ni de dónde esté, y no "
                           "se apaga nunca"},
    "E2": {"nombre": "evaporación en el borde", "donde": "borde", "mecanismo": True, "constante": None,
           "se_apaga_con_las_burbujas": True, "encaje": "alto",
           "consecuencia": "la MCV del borde se evapora hacia la ECV en la cima de la barrera, sin constante libre: la "
                           "mitad del contacto; fija el sentido neto junto con la condensación"},
    "E3": {"nombre": "reversión en el océano", "donde": "océano, sobre todo vacíos", "mecanismo": True, "constante": "ξ_e",
           "se_apaga_con_las_burbujas": True, "encaje": "medio",
           "consecuencia": "la MCV fuera de toda burbuja vuelve a ECV: los vacíos son la zona de drenaje Mp → Ep del "
                           "Maestro (§3.9.5); nada por halo. Antes de los halos todo es océano: la fórmula literal "
                           "vaciaría la materia de la edad oscura (desde z = 1100 con ξ_e = 0.3 quedaría (21/1101)^0.3 ≈ "
                           "30 % en z = 20), así que solo cabe si actúa en vacíos de verdad, tras formarse las "
                           "estructuras, o con ξ_e ≪ 0.1"},
    "E4": {"nombre": "objeto crítico", "donde": "centro", "mecanismo": True, "constante": "ξ_4e",
           "se_apaga_con_las_burbujas": False, "encaje": "alto, pero no sirve al fondo",
           "consecuencia": "la conversión Mp → Ep del objeto crítico (§3.10.3): con los agujeros negros en 10⁻⁵ de la "
                           "materia, su efecto en el fondo es una cota despreciable; necesita una ley Γ_crit"},
}


@dataclass(frozen=True)
class Setup:
    mass_function: str = "ST"          # «ST» o «PS» (control)
    M_min: float = 1e8                 # M☉: el halo de MCV más pequeño (10⁶–10¹⁰ de sensibilidad)
    k: float = 1.0                     # masa dentro de R₀ / masa del halo (2 de control)
    closure: str = "porosidad"         # «porosidad» (e^{−N_b}) o «lineal» (1 − N_b, control)
    outskirts: str = "nfw"             # «nfw», «codigo» (capa del borde con la fuerza de la ECV) o «entorno»
    concentration: str = "dm14"        # «dm14» (Dutton y Macciò 2014) o «c10» (c = 10 para todos, el documento)
    rate_epoch: str = "z"              # «z»: tasas de cada época; «z0»: las de hoy, fijas en tiempo físico (el documento)
    z_ini: float = 6.0
    n_z: int = 241
    n_mass: int = 60


DOC_SETUP = Setup(concentration="c10", rate_epoch="z0")      # el modelo de las cifras del documento


# ------------------------------------------------------------------ espectro lineal, crecimiento y función de masa
def _transfer_eh_nowiggle(k):                  # k en Mpc⁻¹
    h, om, ob = COSMO["h"], COSMO["Omega_m"], COSMO["Omega_b"]
    omh2, obh2, fb = om * h * h, ob * h * h, ob / om
    s = 44.5 * math.log(9.83 / omh2) / math.sqrt(1 + 10 * obh2 ** 0.75)
    alpha = 1 - 0.328 * math.log(431 * omh2) * fb + 0.38 * math.log(22.3 * omh2) * fb ** 2
    gamma = om * h * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
    q = (k / h) * (COSMO["T_cmb"] / 2.7) ** 2 / gamma
    L0 = np.log(2 * math.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


_LNK = np.linspace(math.log(1e-5), math.log(1e4), 6000)
_KK = np.exp(_LNK)
_PK = _KK ** COSMO["n_s"] * _transfer_eh_nowiggle(_KK) ** 2
_RHO_M0 = COSMO["Omega_m"] * rho_crit(0.0, {"h": COSMO["h"], "Omega_m": COSMO["Omega_m"], "Omega_L": COSMO["Omega_L"]}) * 1e9


def _sigma_R_raw(R: float) -> float:
    x = _KK * R
    W = np.where(x < 1e-3, 1 - x * x / 10, 3 * (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-300) ** 3)
    return math.sqrt(np.trapezoid(_PK * W * W * _KK ** 3 / (2 * math.pi ** 2), _LNK))


_NORM = COSMO["sigma8"] / _sigma_R_raw(8.0 / COSMO["h"])


@lru_cache(maxsize=4096)
def sigma_M(M: float) -> float:
    """σ(M) lineal hoy, con M en M☉ (radio lagrangiano en Mpc comóviles)."""
    return _NORM * _sigma_R_raw((3.0 * M / (4.0 * math.pi * _RHO_M0)) ** (1.0 / 3.0))


@lru_cache(maxsize=4096)
def growth(z: float) -> float:
    """D(z)/D(0) de ΛCDM."""
    om, ol = COSMO["Omega_m"], COSMO["Omega_L"]

    def D(a):
        return 2.5 * om * math.sqrt(om / a ** 3 + ol) * quad(lambda x: 1.0 / (x * math.sqrt(om / x ** 3 + ol)) ** 3,
                                                              1e-8, a, limit=200)[0]
    return D(1.0 / (1.0 + z)) / D(1.0)


def _f_st(nu):
    A, a, p = ST_PARAMS
    nu = np.asarray(nu, dtype=float)
    return A * np.sqrt(2 * a / math.pi) * (1 + (a * nu * nu) ** (-p)) * np.exp(-a * nu * nu / 2)


def _f_ps(nu):
    nu = np.asarray(nu, dtype=float)
    return np.sqrt(2.0 / math.pi) * np.exp(-nu * nu / 2.0)


def f_h(z: float, M_min: float = 1e8, kind: str = "ST") -> float:
    """Fracción de la masa en halos por encima de M_min."""
    nu_min = DELTA_C / (sigma_M(M_min) * growth(z))
    if kind == "PS":
        return float(erfc(nu_min / math.sqrt(2.0)))
    return quad(lambda n: float(_f_st(n)), nu_min, np.inf, limit=200)[0]


def mass_weights(z: float, setup: Setup) -> tuple[np.ndarray, np.ndarray]:
    """Masas y pesos (fracción de la masa en halos por intervalo de ln M) entre M_min y 10¹⁶ M☉; suman f_h."""
    lnM = np.linspace(math.log(setup.M_min), math.log(1e16), setup.n_mass + 1)
    nu = DELTA_C / (np.array([sigma_M(float(math.exp(x))) for x in lnM]) * growth(z))
    f = _f_ps(nu) if setup.mass_function == "PS" else _f_st(nu)
    w = 0.5 * (f[1:] + f[:-1]) * np.abs(np.diff(nu))               # ∫ f dν por intervalo
    return np.exp(0.5 * (lnM[1:] + lnM[:-1])), w


# ------------------------------------------------------------------ concentración, afueras y tasas por halo
def c200(M: float, z: float, setup: Setup) -> float:
    """c200 de Dutton y Macciò (2014) con su dependencia en z, o c = 10 (el documento)."""
    if setup.concentration == "c10":
        return 10.0
    a = 0.520 + (0.905 - 0.520) * math.exp(-0.617 * z ** 1.21)
    b = -0.101 + 0.026 * z
    return 10.0 ** (a + b * math.log10(M * COSMO["h"] / 1e12))


def _outskirts(c: float, z: float, setup: Setup, code_table=None) -> dict:
    if setup.outskirts == "entorno":
        return environment_outskirts(c, z)
    o = nfw_outskirts(c, z)
    if setup.outskirts == "codigo":
        if code_table is None:
            raise ValueError("las afueras «codigo» necesitan la tabla de la capa del borde (code_border_table)")
        rr, rn = code_table(c, z)
        return {**o, "rho_b": o["rho_b"] * rr, "nu_b": o["nu_b"] * rn}
    return o


def _potential_weight(M: float, c: float, z: float) -> float:
    """Profundidad media del pozo dentro de R₀ (volumen), −Φ̄/c², salvo constantes: v₀²·(3/x³)∫₀ˣ y·ln(1+y) dy."""
    o = nfw_outskirts(c, z)
    x = o["x_R0"]
    g = 3.0 / x ** 3 * quad(lambda y: y * math.log1p(y), 0.0, x, limit=200)[0]
    rc = rho_crit(z, {"h": COSMO["h"], "Omega_m": COSMO["Omega_m"], "Omega_L": COSMO["Omega_L"]})
    r200 = (3.0 * M / (4.0 * math.pi * 200.0 * rc)) ** (1.0 / 3.0)
    rho_s = (200.0 / 3.0) * c ** 3 / (math.log1p(c) - c / (1.0 + c)) * rc
    return rho_s * (r200 / c) ** 2 * g


_T4_REF = None


def _t4_ref() -> float:
    global _T4_REF
    if _T4_REF is None:
        _T4_REF = _potential_weight(1e12, 10.0, 0.0)
    return _T4_REF


def halo_rates(M: float, z: float, setup: Setup, code_table=None) -> dict:
    """Tasas por unidad de masa de un halo, en unidades de H(z), de las leyes por halo con constante unidad."""
    c = c200(M, z, setup)
    zz = 0.0 if setup.rate_epoch == "z0" else z
    o = _outskirts(c, zz, setup, code_table)
    br = border_rates(o, xi=1.0)
    scale = E_of_z(0.0) / E_of_z(z) if setup.rate_epoch == "z0" else 1.0      # tasa física fija: H0/H(z)
    return {"Tc": br["condensation"] * scale, "E2": br["evaporation"] * scale, "xi_eq": br["xi_eq"],
            "T4": 0.5 * _potential_weight(M, c, z) / _t4_ref(), "c": c}


# ------------------------------------------------------------------ q(z) de cada ley y el fondo
def ocean(z: float, fh: float, setup: Setup) -> float:
    """P_océano: e^{−N_b} (porosidad) o 1 − N_b (lineal), con N_b = min(1, k·f_h)·ρ̄_m/(2ρ_E)."""
    n_b = min(1.0, setup.k * fh) * COSMO["Omega_m"] * (1 + z) ** 3 / (2 * COSMO["Omega_L"])
    return math.exp(-n_b) if setup.closure == "porosidad" else max(0.0, 1.0 - n_b)


def law_profiles(setup: Setup, code_table=None) -> dict:
    """q(z) = Q/(Hρ_E) de cada ley con constante unidad, en la rejilla de z; el barrido, por separado, como su fuente: la
    acreción q_acc = k·(ρ̄_m/ρ_E)·(ḟ_h/H)·P_océano."""
    zg = np.linspace(0.0, setup.z_ini, setup.n_z)
    fh = np.array([f_h(float(z), setup.M_min, setup.mass_function) for z in zg])
    dfdz = np.gradient(fh, zg)
    out = {n: np.zeros_like(zg) for n in ("T1", "T2", "T4", "Tc", "E1", "E2", "E3", "E4", "acc", "P", "f_h", "xi_eq")}
    for i, z in enumerate(zg):
        P = ocean(float(z), float(fh[i]), setup)
        rm = COSMO["Omega_m"] * (1 + z) ** 3 / COSMO["Omega_L"]
        Ms, w = mass_weights(float(z), setup)
        rates = [halo_rates(float(M), float(z), setup, code_table) for M in Ms]
        wsum = float(np.sum(w))
        norm = fh[i] / wsum if wsum > 0 else 0.0                       # los pesos discretos, normalizados a f_h
        for key in ("Tc", "E2"):
            out[key][i] = setup.k * rm * P * norm * float(np.dot(w, [r[key] for r in rates]))
        n_b = min(1.0, setup.k * fh[i]) * rm / 2.0                     # T4, como T2: interior, con el solape
        overlap = (1.0 - P) / n_b if n_b > 0 else 1.0
        out["T4"][i] = setup.k * rm * overlap * norm * float(np.dot(w, [r["T4"] for r in rates]))
        out["xi_eq"][i] = float(np.dot(w, [r["xi_eq"] for r in rates]) / wsum) if wsum > 0 else 0.0
        out["T1"][i] = fh[i]
        out["T2"][i] = 1.0 - P
        out["E1"][i] = 1.0
        out["E3"][i] = (1.0 - min(1.0, setup.k * fh[i])) * rm * P
        out["E4"][i] = F_BH * rm
        out["acc"][i] = setup.k * rm * (-(1 + z) * dfdz[i]) * P
        out["P"][i], out["f_h"][i] = P, fh[i]
    out["T4_raw_today"] = float(out["T4"][0])
    out["T4"] = out["T4"] * (out["T2"][0] / out["T4"][0])             # normalización declarada: T4(hoy) = T2(hoy)
    out["z"] = zg
    return out


def combine(profiles: dict, constants: dict, sweep_feeds: bool = True) -> np.ndarray:
    """q_neto(z) de una combinación: constants = {ley: constante} (las de escape se restan; el contacto es «Tc» y «E2»
    con la misma ξ). Con el barrido (T3, constante ξ_s) presente, se alimenta de la acreción y, si sweep_feeds, de las leyes
    por halo: q = s·q_acc + (1 + s)·q_halo + q_fuera, con s = (ξ_s/2)/(1 − ξ_s/2)."""
    sign = {"T1": 1, "T2": 1, "T4": 1, "Tc": 1, "E1": -1, "E2": -1, "E3": -1, "E4": -1}
    halo = sum(sign[k] * v * profiles[k] for k, v in constants.items() if k in PER_HALO)
    outside = sum(sign[k] * v * profiles[k] for k, v in constants.items() if k in ("E1", "E3"))
    halo = halo if not isinstance(halo, int) else np.zeros_like(profiles["z"])
    outside = outside if not isinstance(outside, int) else np.zeros_like(profiles["z"])
    if "T3" in constants:
        s = 0.5 * constants["T3"] / (1.0 - 0.5 * constants["T3"])
        return s * profiles["acc"] + (1.0 + s if sweep_feeds else 1.0) * halo + outside
    return halo + outside


def delta_today(q: np.ndarray, z: np.ndarray) -> float:
    """Δ = (ρ̄_M(0) − ρ̄_m,CMB(0))/ρ̄_M(0): dy/d ln a = −3y + q·Ω_Λ (y = ρ_M/ρ_c0, H de ΛCDM) desde z_ini."""
    om, ol = COSMO["Omega_m"], COSMO["Omega_L"]
    zmax = float(z[-1])

    def rhs(lna, y):
        return [-3.0 * y[0] + float(np.interp(math.exp(-lna) - 1.0, z, q)) * ol]
    sol = solve_ivp(rhs, (-math.log1p(zmax), 0.0), [om * (1 + zmax) ** 3], rtol=1e-10, atol=1e-13)
    y0 = sol.y[0, -1]
    return (y0 - om) / y0


def timing(q: np.ndarray, z: np.ndarray) -> dict:
    """Cuándo actúa una ley: z del máximo de |q| y fracción de ∫|q|·Ω_Λ d ln a que cae en z > 1 y en z > 3."""
    lna = -np.log1p(z)
    weight = np.abs(q)
    total = float(np.trapezoid(weight[::-1], lna[::-1])) or 1.0
    frac = {}
    for zc in (1.0, 3.0):
        m = z >= zc
        frac[f"z>{zc:g}"] = float(np.trapezoid(weight[m][::-1], lna[m][::-1]) / total) if m.sum() > 1 else 0.0
    return {"z_peak": float(z[int(np.argmax(weight))]), **frac}


def two_way_catalogue() -> list[dict]:
    """Todas las combinaciones de dos sentidos: cada atrapamiento (y el barrido con condensación) con cada escape. El
    contacto es la combinación Tc + E2; con el barrido, la ley de borde completa del documento."""
    trap = [("T1",), ("T2",), ("T3",), ("T4",), ("Tc",), ("T3", "Tc")]
    combos = []
    for t in trap:
        for e in ESCAPE:
            name = "+".join(t + (e,))
            special = {("Tc", "E2"): "contacto", ("T3", "Tc", "E2"): "barrido y contacto (ley de borde completa)"}.get(t + (e,))
            combos.append({"name": name, "laws": t + (e,), "label": special})
    return combos


def code_border_table(cs=(3.0, 5.0, 10.0, 20.0, 40.0), zs=(0.0, 0.5, 1.0, 2.0, 4.0, 6.0), n_shells: int = 600):
    """Cocientes (ρ_b, ν_b) de la capa del borde del código gravotérmico con la fuerza de la ECV frente al NFW prolongado,
    tabulados en (ln c, z) e interpolados: la afuera «codigo» de la decisión 6."""
    from dynamics.gravothermal import GravothermalConfig, code_outskirts

    base = GravothermalConfig(sigma_hat=0.01, n_shells=n_shells)
    rr = np.zeros((len(cs), len(zs)))
    rn = np.zeros_like(rr)
    for i, c in enumerate(cs):
        for j, z in enumerate(zs):
            o, oc = nfw_outskirts(c, z), code_outskirts(c, z, base=base)
            rr[i, j], rn[i, j] = oc["rho_b"] / o["rho_b"], oc["nu_b"] / o["nu_b"]
    lc = np.log(np.asarray(cs))
    za = np.asarray(zs)

    def interp(c: float, z: float):
        x = float(np.clip(math.log(c), lc[0], lc[-1]))
        y = float(np.clip(z, za[0], za[-1]))
        i = int(np.clip(np.searchsorted(lc, x) - 1, 0, len(lc) - 2))
        j = int(np.clip(np.searchsorted(za, y) - 1, 0, len(za) - 2))
        tx, ty = (x - lc[i]) / (lc[i + 1] - lc[i]), (y - za[j]) / (za[j + 1] - za[j])

        def bil(a):
            return (a[i, j] * (1 - tx) * (1 - ty) + a[i + 1, j] * tx * (1 - ty) + a[i, j + 1] * (1 - tx) * ty
                    + a[i + 1, j + 1] * tx * ty)
        return bil(rr), bil(rn)
    interp.table = {"c": list(cs), "z": list(zs), "rho_ratio": rr.tolist(), "nu_ratio": rn.tolist()}
    return interp


__all__ = ["COSMO", "DOC_SETUP", "ESCAPE", "LAW_ONTOLOGY", "Setup", "TRAPPING", "c200", "code_border_table", "combine",
           "delta_today", "f_h", "growth", "halo_rates", "law_profiles", "mass_weights", "ocean", "sigma_M",
           "timing", "two_way_catalogue"]
