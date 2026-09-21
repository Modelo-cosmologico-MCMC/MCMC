"""El reloj S — orquestador de la trayectoria S₀ → S_{1,001} como
SIMULADOR DE CONSISTENCIA (v35: C1; caps. 3–8; §5.3–5.4; §6.1–6.3).

LO QUE ES. Un solo bucle en σ que integra a la vez:

- el Flujo del Camino (ec. 4.2, `core.path_flow`) sobre el Plano Dual con
  el Potencial Basal COMPLETO (con la inclinación −η·χ), arrancando en el
  PUNTO DE ESCAPE de Coleman: el punto más allá de la barrera donde
  V = V(falso vacío) — el campo emerge sin velocidad y con la tensión del
  falso vacío (Prop. 3.5). Con la inclinación el falso vacío no está en
  el origen sino en ρ_fv ≃ η/(√2·M0²) sobre el eje θ = 0 (el residuo
  ⟨χ⟩ ≃ (ē/m̄²)·δ0 de la Prop. 3.4), y la barrera solo existe por debajo
  de un δ0 máximo que el módulo calcula (`delta0_metastability_max`):
  la metastabilidad de la Obs. 8.6 (D(S0) > 0) es necesaria, no
  suficiente, cuando la inclinación es O(δ0^{7/2}) frente a T₀ = O(δ0³);
- el contador de descarga f = (V_fv − V(Φ))/T₀ con T₀ = V_fv − V_tv (la
  Tensión Primordial MEDIDA en el paisaje completo; se publica frente a
  la ley c̄·δ0³ de la Prop. 3.4, cuya diferencia es la corrección de la
  inclinación) y el índice entrópico S = ∫Σ̇ dσ / T₀ con Σ̇ = (∇V)ᵀG⁻¹∇V
  (Teo. 4.5). NORMALIZACIÓN DECLARADA: S = 1 ⟺ descarga completa (el
  número del cuanto ΔS es convencional, F.4). Que S ≡ f a lo largo del
  descenso es una IDENTIDAD del flujo de gradiente (Σ̇ = −dV/dσ) que el
  bucle comprueba numéricamente, no una hipótesis;
- los acoplos λ = (M0², B, C0) por las β de Fokker–Planck (Def. 4.4,
  `core.fokker_planck_beta`, cierre canónico, diccionario τ declarado)
  — OPCIONAL: en el modo con estatuto los acoplos están congelados;
- el discriminante D(S) = B² − 4C0M0² (Def. 8.4) y su cruce D = 0 (el
  Cruce de Victoria, Obs. 8.6) como EVENTO de colapso d → d + 1 con el
  álgebra C(d+1, 0) de la cadena (`core.florencia.chain_generators`);
- las leyes de sellado por congelación de c_eff (§5.3, Cota de
  Delivery, β_c(S_0,099) = 0) y de m_eff (§5.4, dm_eff/dS|_{0,999} = 0)
  — con FORMA PARAMÉTRICA DECLARADA (`DECLARED_FORMS`): el tratado fija
  la propiedad de sellado, la ecuación (5.3) no está transcrita en el
  repositorio y la derivación de β_c desde el sustrato es la fila
  c-lieb-robinson-identificacion (claim-no-derivado);
- la fase θ y su cruce de la diagonal dual θ = π/4 (Def. 2.2 / Prop.
  3.5): seguida desde el flujo Y como perfil impuesto;
- en S = 1,001 la Rotación de Florencia (ec. 6.2) como cambio de régimen
  (C(4,0) → C(3,1), firma −+++, RP en la loncha con el control negativo
  J < 0, identidad m_H = √(2β₃)v₃ con β₃ del empalme) que ENTREGA el
  estado inicial cosmológico — y declara que la cosmología no puede
  leerlo (fila diccionario-primordial-cosmologico).

LO QUE NO ES: emergente. En el modo 'imposed' (el único con estatuto)
los colapsos se disparan en los umbrales de la Ley de la Década (Prop.
8.1, CALIBRADOS: λ = 10 es el frente 2) leídos sobre el reloj f; en el
modo 'emergent' (DIAGNÓSTICO, sin estatuto) los colapsos se disparan
donde D(S) = 0 con el diccionario τ declarado — publica dónde caerían,
no decide λ. En ese modo (v1, 21-sep) el cruce M0² = 0 se publica como
evento `inestabilidad_masa` (S_flip medido frente a su forma cerrada de
primer orden) y el bucle sigue hasta C0 ≤ 0; la lectura «Década ⟺ τ_k ≈
S_flip·δ₀/S_k» es diagnóstico E13 (reformulación del diccionario τ, no
derivación). La nucleación (Γ₀(δ₀), Prop. 3.5) es opcional
(`nucleation='bounce'`, core.nucleation); por defecto el reloj arranca
en el punto de escape con σ = 0 declarado. El transporte
azimutal hasta la diagonal (Prop. 3.5) no lo produce el Basal solo: se
publica el θ del flujo y se impone el perfil declarado para la entrega.
Los cuantos de confirmación (V3D en 1,000 y Florencia en 1,001 tras el
residuo de descarga ε_res, F.3) se declaran, no se producen.

Estatuto: comprobación interna (E8) — cada comprobación lleva su
etiqueta «derivado / impuesto / declarado / publicado». El interruptor
'emergent' solo tendrá estatuto cuando el frente 2 entregue τ(S).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import brentq

from mcmc_ontology import constants as C
from mcmc_ontology.clifford_algebra import regime

from .basal import (
    B_BAR,
    C0_DEFAULT,
    E_BAR,
    M_BAR,
    V0,
    T0_analytic,
    T0_numeric,
    c_bar,
    chi_residue,
    kappa_plus,
    quasi_cancellation_ok,
    scaled_params,
)
from .decade import discriminant_basal, metastability_bound
from .dual_plane import THETA_DIAGONAL, to_dual
from .florencia import (
    all_anticommute,
    chain_generators,
    euclidean_symbol,
    florencia_rotation,
    signature,
)
from .fokker_planck_beta import FPClosure, beta_functions
from .gea import atlas_healthy, newton_seal_ratio
from .reflection_positivity import measure_normalizable, rp_min_eigenvalue

STATUS = ("simulador de consistencia (E8): umbrales de la Década impuestos "
          "(Prop. 8.1, λ = 10 calibrado — frente 2), nucleación declarada en el "
          "punto de escape, sellados con forma paramétrica declarada, diagonal "
          "impuesta para la entrega; el estado entregado en S = 1,001 no es legible "
          "por la cosmología (diccionario ausente)")

# Formas paramétricas DECLARADAS de las leyes de sellado por congelación.
# El tratado fija la PROPIEDAD (logística; β_c(S_c) = 0; dm_eff/dS = 0 en
# S_c²); la forma concreta de la tasa es una elección del simulador y se
# publica como tal. Sustituirla por la ec. (5.3) transcrita es un cambio
# local que no toca el bucle.
DECLARED_FORMS = {
    "c_eff": "du/dS = β_c(S)·u·(1 − u), β_c(S) = β_c0·max(0, 1 − S/S_c), "
             "S_c = S_0,099; c_eff ≡ max|du/dS| alcanzado antes del sello; "
             "u(0) = u0. FORMA DECLARADA, no la ec. (5.3) del tratado.",
    "m_eff": "dm/dS = −γ_m·(m − m_min)·max(0, 1 − S/S_c²), S_c² = S_0,999; "
             "dm/dS|_{S_c²} = 0 fija m_min (§5.4). FORMA DECLARADA.",
    "theta_imposed": "θ_imp(S) = (π/4)·min(1, S/S_diag), S_diag = S_V3D = 1,000: "
                     "cruce de la diagonal en S ≃ 1 POR CONSTRUCCIÓN (Prop. 3.5 "
                     "declarada, no producida).",
    "S_normalization": "S ≡ ∫Σ̇ dσ / T₀ con T₀ = V_fv − V_tv medida en el paisaje "
                       "completo — descarga completa ⟺ S = 1 (el número del cuanto "
                       "ΔS es convencional, F.4).",
    "nucleation": "punto de escape de Coleman: ρ_esc > ρ_barrera con V(ρ_esc) = V_fv "
                  "(emerge sin velocidad con la tensión del falso vacío) en la "
                  "dirección θ_nuc declarada; Γ₀(δ₀) no calculada.",
    "confirmation_quanta": "tras alcanzar f = 1 − ε_res (residuo de descarga, "
                           "F.3) el reloj avanza ΔS por cuanto declarado: V3D en "
                           "1,000 y Florencia en 1,001 (Prop. 8.1).",
    "Phi_ten_at_florencia": "Φ_ten(S_1,001) = 0 por normalización declarada "
                            "(N = 1): no derivado.",
    "S_post_unit": "tras Florencia (S ≥ 1,001) el índice S deja de medir la fracción "
                   "descargada de T₀ (S = 1 ⟺ descarga completa, normalización "
                   "pre-geométrica de este reloj): los sellos posteriores de la Tabla "
                   "F.1 y el cuanto ΔS = 1e-3 (Calibre de Cronos, F.4) se leen en una "
                   "unidad POST-geométrica que el repositorio NO deriva de T₀ ni de "
                   "Σ̇ — la relación entre ambas unidades es un diccionario ausente "
                   "(fila diccionario-unidad-S-post-florencia). DECLARADO, hueco.",
    "mass_instability": "en modo emergente el reloj sigue integrando cuando M0² cruza "
                        "cero (el origen deja de ser mínimo; V sigue acotado por C0 > 0) "
                        "y publica el evento `inestabilidad_masa` con el S del cruce, "
                        "S_flip, junto a su forma cerrada de primer orden "
                        "S_flip ≃ δ₀·[b̄ − √(b̄² − 6C0m̄²)]/(24·a·τ·C0) (β canónicas, "
                        "M0⁴ despreciado); se detiene solo cuando C0 ≤ 0. DIAGNÓSTICO: "
                        "«Década ⟺ τ_k ≈ S_flip·δ₀/S_k» es una reformulación del "
                        "diccionario τ, no una derivación (E13: el número no es señal).",
    "J_circulation": "Camino de dos niveles (propuesta v36): dΦ/dσ = −G⁻¹∇V + J∇C con J = "
                     "|J|·ε antisimétrica (ε = [[0, −1], [1, 0]]) y |J| DECLARADO; C = V por "
                     "defecto (J∇V ⊥ ∇V: la Monotonía 4.5 y la identidad S ≡ f se conservan "
                     "EXACTAMENTE; Σ̇ cuenta solo el nivel disipativo) o C = ρ²/2 (rotación "
                     "rígida; trabaja contra la inclinación: W_J = ∫∇V·J∇C dσ ≠ 0, "
                     "publicado). Orientación de J fijada para que θ crezca hacia la diagonal "
                     "durante el descenso. El reloj publica el |J| que sitúa el cruce de la "
                     "diagonal en S_target (calibración con significado, no derivación: J "
                     "desde la ontología es el frente 2).",
    "tau_per_dim": "HIPÓTESIS DECLARADA (opcional, modo emergente): τ toma un valor "
                   "por dimensión, τ_d, y en cada inestabilidad de masa el colapso "
                   "d → d+1 se dispara y M0² se repone a su valor inicial (el "
                   "escalón del potencial); publica dónde caerían los tres colapsos "
                   "bajo esa hipótesis frente a la Década. Sin estatuto (E13).",
}


@dataclass(frozen=True)
class ClockConfig:
    """Entradas del reloj — todas declaradas; ninguna del tratado salvo
    las constantes de forma del Basal (F.2) y los umbrales (Prop. 8.1)."""
    delta0: float = 0.01                 # input de ciclo (F.2): SIN valor asignado
    m_bar: float = M_BAR                 # en el tratado; 0.01 es valor de prueba
    b_bar: float = B_BAR                 # (bajo el δ0 máximo de metastabilidad)
    e_bar: float = E_BAR
    C0: float = C0_DEFAULT
    G: float = 1.0                       # rigidez escalar (G ≻ 0)
    thresholds: str = "imposed"          # 'imposed' (estatuto) | 'emergent' (diagnóstico)
    couplings_flow: bool = False         # integrar dλ/dS = τ·β(λ) (Def. 4.4)
    fp: FPClosure = field(default_factory=FPClosure)
    theta_nuc: float = 0.0               # dirección de emergencia (polo de masa, Prop. 3.5)
    eps_res: float = C.DELTA_S           # residuo de descarga que termina el descenso
    d_sigma_hat: float = 5e-4            # paso en σ̂ = σ·δ0² (invariante de escala)
    max_steps: int = 400000
    # sellados con forma declarada
    beta_c0: float = 40.0
    u0: float = 0.02
    gamma_m: float = 8.0
    m_eff0: float = 1.0
    m_min: float = 0.5
    # modo emergente: fin del diagnóstico (S de parada del recorrido)
    S_end_emergent: float = 1.001
    # nucleación: 'escape_point' (V = V_fv, convención) | 'bounce' (centro del
    # instantón O(d) de core.nucleation: el estado que la nucleación entrega)
    nucleation: str = "escape_point"
    bounce_d: int = 4
    # modo emergente: hipótesis declarada τ por dimensión (None = τ único de fp)
    tau_per_dim: tuple | None = None
    # circulación J∇C del Camino de dos niveles (propuesta v36): |J| DECLARADO
    # (0 = flujo de gradiente puro) y generador C declarado: 'V' (el mismo
    # generador: J∇V ⊥ ∇V conserva la Monotonía y S ≡ f exactamente) |
    # 'rho2' (C = ρ²/2, rotación rígida: trabaja contra la inclinación)
    J_circ: float = 0.0
    circulation_C: str = "V"


def sextic_coupling_dimension(d: int) -> float:
    """[C0] = d − 3(d − 2) = 6 − 2d: el séxtico es marginal exactamente
    en d = 3 (Prop. H.1) — la cascada dimensional termina donde el
    potencial del Camino deja de ser irrelevante."""
    return float(6 - 2 * d)


def radial_landscape(delta0: float, theta: float = 0.0, m_bar: float = M_BAR,
                     b_bar: float = B_BAR, e_bar: float = E_BAR,
                     C0: float = C0_DEFAULT) -> dict:
    """Puntos estacionarios de V0(ρ; θ fijo) CON la inclinación −η·χ,
    χ = ρ(cos θ − sin θ)/√2.

    dV/dρ = M0²ρ − Bρ³ + C0ρ⁵ − η(cos θ − sin θ)/√2. Tres raíces positivas
    ⟹ falso vacío, barrera y vacío verdadero (metastable); una sola ⟹ la
    inclinación ha borrado la barrera (sin falso vacío: no hay S0 que
    nuclear). Devuelve ρ y V de cada punto, o None si no existe.
    """
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    tilt = p["eta"] * (np.cos(theta) - np.sin(theta)) / np.sqrt(2.0)

    def dV(r):
        return p["M0_sq"] * r - p["B"] * r ** 3 + C0 * r ** 5 - tilt

    def V(r):
        return (0.5 * p["M0_sq"] * r ** 2 - 0.25 * p["B"] * r ** 4 + (C0 / 6.0) * r ** 6
                - tilt * r)
    D = p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]
    r_max = 1.6 * np.sqrt((p["B"] + np.sqrt(max(D, 0.0))) / (2.0 * C0)) if D >= 0 else 2.0 * np.sqrt(delta0)
    grid = np.linspace(1e-9, r_max, 4001)
    vals = dV(grid)
    roots = []
    # la raíz del origen: dV(0) = −tilt. Sin inclinación el origen ES el falso
    # vacío (raíz exacta que la malla, que arranca en 1e-9, no ve — defecto v0
    # corregido en v1); con inclinación pequeña la raíz cae entre 0 y la malla.
    if tilt == 0.0:
        roots.append(0.0)
    elif -tilt * vals[0] < 0.0:
        roots.append(float(brentq(dV, 0.0, grid[0], xtol=1e-16)))
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            fa, fb = dV(float(grid[i])), dV(float(grid[i + 1]))
            if fa * fb < 0.0:
                roots.append(float(brentq(dV, grid[i], grid[i + 1], xtol=1e-14)))
            else:
                # tangencia numérica (barrera doble en δ₀ ≃ δ₀_max): la evaluación
                # vectorizada y la escalar discrepan en el redondeo; raíz en el centro
                roots.append(float(0.5 * (grid[i] + grid[i + 1])))
    roots = sorted(roots)
    out = {"roots": roots, "metastable": len(roots) >= 3, "tilt": float(tilt)}
    if len(roots) >= 3:
        fv, bar, tv = roots[0], roots[1], roots[-1]
        out.update({"rho_fv": fv, "V_fv": float(V(fv)), "rho_barrier": bar, "V_barrier": float(V(bar)),
                    "rho_tv": tv, "V_tv": float(V(tv)), "T0_full": float(V(fv) - V(tv)),
                    "barrier_height": float(V(bar) - V(fv))})
        # punto de escape: V(ρ_esc) = V_fv con ρ ∈ (ρ_barrera, ρ_tv)
        g = lambda r: V(r) - V(fv)  # noqa: E731
        out["rho_esc"] = float(brentq(g, bar, tv, xtol=1e-14)) if g(bar) > 0 > g(tv) else None
    elif roots:
        tv = roots[-1]
        out.update({"rho_fv": None, "V_fv": None, "rho_barrier": None, "V_barrier": None,
                    "rho_tv": tv, "V_tv": float(V(tv)), "T0_full": None, "barrier_height": 0.0,
                    "rho_esc": None})
    return out


def delta0_metastability_max(m_bar: float = M_BAR, b_bar: float = B_BAR,
                             e_bar: float = E_BAR, C0: float = C0_DEFAULT,
                             theta: float = 0.0, lo: float = 1e-5, hi: float = 2.0) -> float:
    """El δ0 máximo con falso vacío metastable en el paisaje COMPLETO:
    la inclinación η = ē·δ0³ crece frente a la barrera (∝ δ0³ con altura
    relativa fija) como δ0^{1/2}, y por encima de δ0_max la borra.
    Bisección sobre la existencia de las tres raíces (monótona en δ0
    para las formas O(1) del Basal)."""
    if not radial_landscape(lo, theta, m_bar, b_bar, e_bar, C0)["metastable"]:
        return 0.0
    if radial_landscape(hi, theta, m_bar, b_bar, e_bar, C0)["metastable"]:
        return float("inf")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if radial_landscape(mid, theta, m_bar, b_bar, e_bar, C0)["metastable"]:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _g_max(m_bar: float, b_bar: float, C0: float) -> float:
    """Máximo local de g(s) = m̄²s − b̄s³ + C0s⁵ (dV₀/dρ en la variable
    escalada ρ = √δ₀·s: dV₀/dρ = δ₀^{5/2}·g(s)); existe si 9b̄² > 20C0m̄²
    (más débil que la casi-cancelación (3.3))."""
    disc = 9.0 * b_bar ** 2 - 20.0 * C0 * m_bar ** 2
    if disc <= 0.0:
        return 0.0
    s2 = (3.0 * b_bar - np.sqrt(disc)) / (10.0 * C0)
    s = np.sqrt(s2)
    return float(m_bar ** 2 * s - b_bar * s ** 3 + C0 * s ** 5)


def delta0_metastability_max_analytic(m_bar: float = M_BAR, b_bar: float = B_BAR,
                                      e_bar: float = E_BAR, C0: float = C0_DEFAULT,
                                      theta: float = 0.0) -> float:
    """Forma cerrada de δ₀_max: la barrera existe mientras la inclinación
    ē·δ₀³·(cos θ − sin θ)/√2 no supere el máximo local de dV₀/dρ =
    δ₀^{5/2}·g_max, es decir

        δ₀_max = 2·g_max(m̄, b̄, C0)² / (ē·(cos θ − sin θ))²   ∝ ē⁻²

    (0.1028·ē⁻² con las formas por defecto; ∞ sin inclinación o sobre la
    diagonal θ = π/4, donde el origen es el falso vacío exacto). Se
    comprueba contra la bisección de `delta0_metastability_max`."""
    fac = e_bar * (np.cos(theta) - np.sin(theta))
    if fac <= 0.0:
        return float("inf")
    return float(2.0 * _g_max(m_bar, b_bar, C0) ** 2 / fac ** 2)


def kappa1_tilt(m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                C0: float = C0_DEFAULT, theta: float = 0.0) -> float:
    """Coeficiente de primer orden de la corrección de la inclinación a T₀:
    T₀_full/T₀_ley − 1 ≃ κ₁·√δ₀ con κ₁ = ē·(cos θ − sin θ)·√κ₊/(√2·c̄)
    (= 1.361 con las formas por defecto en θ = 0): la inclinación baja el
    vacío verdadero en η·χ_tv = ē·δ₀³·ρ₊·(cos θ − sin θ)/√2 y ρ₊ = √(κ₊δ₀)."""
    return float(e_bar * (np.cos(theta) - np.sin(theta)) * np.sqrt(kappa_plus(m_bar, b_bar, C0))
                 / (np.sqrt(2.0) * c_bar(m_bar, b_bar, C0)))


def T0_tilt_first_order(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                        e_bar: float = E_BAR, C0: float = C0_DEFAULT, theta: float = 0.0) -> float:
    """T₀ de la ley 3.4 corregida a primer orden en la inclinación:
    c̄·δ₀³·(1 + κ₁·√δ₀)."""
    return T0_analytic(delta0, m_bar, b_bar, C0) * (1.0 + kappa1_tilt(m_bar, b_bar, e_bar, C0, theta) * np.sqrt(delta0))


def S_flip_first_order(delta0: float, tau: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                       C0: float = C0_DEFAULT, a: float = 0.5) -> float | None:
    """S del cruce M0² = 0 bajo las β canónicas con M0⁴ despreciado:
    M0²(S) = m̄²δ₀² − 8aτ·b̄δ₀·S + 96a²τ²C0·S² (B también fluye,
    β_B ≃ −24aC0) ⟹ S_flip = δ₀·[b̄ − √(b̄² − 6C0m̄²)]/(24·a·τ·C0)
    (0.1057·δ₀/τ con las formas por defecto). None si b̄² < 6C0m̄² (M0²
    no llega a cero en esta aproximación) o τ = 0."""
    if tau <= 0.0:
        return None
    disc = b_bar ** 2 - 6.0 * C0 * m_bar ** 2
    if disc < 0.0:
        return None
    return float(delta0 * (b_bar - np.sqrt(disc)) / (24.0 * a * tau * C0))


def tau_decade_table(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                     C0: float = C0_DEFAULT, a: float = 0.5) -> dict:
    """Reformulación E13: el τ_k que haría caer la inestabilidad de masa
    en cada umbral S_k de la Década, τ_k = S_flip(τ = 1)·δ₀/S_k
    (≈ 11.8·δ₀·10⁻ᵏ con las formas por defecto). Es el diccionario τ
    leído al revés desde los umbrales calibrados, no una derivación: el
    número no es señal."""
    s1 = S_flip_first_order(delta0, 1.0, m_bar, b_bar, C0, a)
    th = C.decade_thresholds()[:3]
    return {"coefficient_S_flip_tau_over_delta0": None if s1 is None else s1 / delta0,
            "tau_k": None if s1 is None else [s1 / S_k for S_k in th], "thresholds": th,
            "status": "diagnóstico E13: reformulación del diccionario τ, no derivación"}


def naturalness_sweep(n: int = 4000, seed: int = 20260921, prior: str = "uniform",
                      delta0_probe: tuple = (0.012, 0.0581), n_check: int = 24,
                      **prior_kwargs) -> dict:
    """δ₀_max sobre los paisajes viables de `landscape_priors`: fracción
    de paisajes con falso vacío metastable en cada δ₀ sondeado (0.012 del
    ε_Λ heredado, 0.0581 del empalme), percentiles de δ₀_max y la
    comprobación forma cerrada frente a bisección en una submuestra."""
    from .landscape_priors import sample_shapes_prior
    sh = sample_shapes_prior(n, seed, prior, **prior_kwargs)
    d0max = np.array([delta0_metastability_max_analytic(m, b, e, c0)
                      for m, b, e, c0 in zip(sh["m_bar"], sh["b_bar"], sh["e_bar"], sh["C0"])])
    idx = np.linspace(0, len(d0max) - 1, min(n_check, len(d0max))).astype(int)
    bis = np.array([delta0_metastability_max(sh["m_bar"][i], sh["b_bar"][i], sh["e_bar"][i], sh["C0"][i], hi=5.0)
                    for i in idx])
    rel = np.abs(bis - d0max[idx]) / d0max[idx]
    return {"prior": prior, "n": int(len(d0max)), "seed": seed,
            "fraction_metastable_at": {str(d): float(np.mean(d0max > d)) for d in delta0_probe},
            "delta0_max_percentiles": {p: float(np.percentile(d0max, p)) for p in (5, 25, 50, 75, 95)},
            "closed_form_vs_bisection_max_rel_err": float(rel.max()),
            "scaling": "δ₀_max = 2·g_max(m̄, b̄, C0)²/ē² (∝ ē⁻²)",
            "status": "publicado (cartografía sobre priors declarados; sin veredicto)"}


def _logistic_rate(S: float, beta_c0: float, S_c: float) -> float:
    return beta_c0 * max(0.0, 1.0 - S / S_c)


def _m_rate(S: float, gamma_m: float, S_c2: float) -> float:
    return gamma_m * max(0.0, 1.0 - S / S_c2)


def to_jsonable(obj):
    """numpy → tipos nativos (para json.dumps)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class SClock:
    """El bucle único. `run()` devuelve el diccionario del recorrido:
    trayectoria, eventos, comprobaciones por estación con etiqueta de
    estatuto y el estado entregado en Florencia."""

    def __init__(self, cfg: ClockConfig | None = None):
        self.cfg = cfg or ClockConfig()
        c = self.cfg
        if not quasi_cancellation_ok(c.m_bar, c.b_bar, c.C0):
            raise ValueError("El Basal no cumple la casi-cancelación (3.3)")
        p = scaled_params(c.delta0, c.m_bar, c.b_bar, c.e_bar)
        self.lam0 = np.array([p["M0_sq"], p["B"], c.C0])
        self.eta = p["eta"]
        self.T0_law = T0_analytic(c.delta0, c.m_bar, c.b_bar, c.C0)
        self.land = radial_landscape(c.delta0, c.theta_nuc, c.m_bar, c.b_bar, c.e_bar, c.C0)
        if not self.land["metastable"] or self.land["rho_esc"] is None:
            raise ValueError(
                f"Sin falso vacío metastable en δ0 = {c.delta0:g}: la inclinación borra la "
                f"barrera (δ0_max = {delta0_metastability_max(c.m_bar, c.b_bar, c.e_bar, c.C0, c.theta_nuc):.4g}); "
                "no hay S0 que nuclear")
        self.T0 = self.land["T0_full"]
        self.V_fv = self.land["V_fv"]
        self.x_plus0 = self.land["rho_tv"] ** 2
        self.bounce = None
        if c.nucleation == "bounce":
            from .nucleation import bounce as _bounce
            self.bounce = _bounce(c.delta0, d=c.bounce_d, theta=c.theta_nuc, m_bar=c.m_bar, b_bar=c.b_bar,
                                  e_bar=c.e_bar, C0=c.C0)
            self.x_esc = self.bounce.phi0 ** 2          # arranca en el centro del bounce
        elif c.nucleation == "escape_point":
            self.x_esc = self.land["rho_esc"] ** 2
        else:
            raise ValueError("nucleation: 'escape_point' | 'bounce'")
        self.thresholds = C.decade_thresholds()          # [0.009, 0.099, 0.999, 1.001]
        self.S_c, self.S_c2 = C.S_SEALS["C2"], C.S_SEALS["C3"]
        self.S_V3D, self.S_flor = C.S_SEALS["V3D"], C.S_SEALS["C4"]

    # ---------------------------------------------------------- potencial
    def _V(self, phi: np.ndarray, lam: np.ndarray) -> float:
        M0_sq, B, C0 = lam
        x = float(phi[0] ** 2 + phi[1] ** 2)
        chi = (phi[0] - phi[1]) / np.sqrt(2.0)
        return float(0.5 * M0_sq * x - 0.25 * B * x ** 2 + (C0 / 6.0) * x ** 3 - self.eta * chi)

    def _grad(self, phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
        M0_sq, B, C0 = lam
        x = float(phi[0] ** 2 + phi[1] ** 2)
        radial = M0_sq - B * x + C0 * x ** 2
        return np.array([radial * phi[0] - self.eta / np.sqrt(2.0),
                         radial * phi[1] + self.eta / np.sqrt(2.0)])

    def _circulation(self, phi: np.ndarray, grad: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """J∇C con J = |J|·ε y orientación declarada: θ crece hacia la
        diagonal durante el descenso. C = V: ε∇V = radial·εΦ + ε·t con
        radial < 0 entre la barrera y el vacío verdadero ⟹ J = −|J|ε.
        C = ρ²/2: ∇C = Φ, εΦ = (−φ_E, φ_M) rota θ positivamente ⟹ J = +|J|ε."""
        c = self.cfg
        if c.circulation_C == "V":
            gC, sgn = grad, -1.0
        else:
            # C = δ0²·ρ²/2: el factor δ0² deja |J| adimensional en σ̂ = σ·δ0² (dθ/dσ̂ = |J|),
            # comparable con C = V, donde |∇V|/ρ ~ δ0²
            gC, sgn = c.delta0 ** 2 * phi, +1.0
        return sgn * c.J_circ * np.array([-gC[1], gC[0]])

    def _f(self, phi: np.ndarray, lam: np.ndarray) -> float:
        """f = (V_fv − V(Φ))/T₀: fracción de la Tensión Primordial ya
        descargada (paisaje inicial como referencia)."""
        return float((self.V_fv - self._V(phi, lam)) / self.T0)

    # --------------------------------------------------------------- bucle
    def run(self) -> dict:
        c = self.cfg
        rho_esc = np.sqrt(self.x_esc)
        phi = np.array([rho_esc * np.cos(c.theta_nuc), rho_esc * np.sin(c.theta_nuc)])
        lam = self.lam0.copy()
        G_inv = 1.0 / c.G
        d_sigma = c.d_sigma_hat / c.delta0 ** 2
        sigma, S_int, u, m = 0.0, 0.0, c.u0, c.m_eff0
        d_dim = 0
        pending = list(self.thresholds[:3])   # colapsos 1D, 2D, 3D
        keys = ("sigma", "S", "f", "clock", "V", "theta", "chi", "rho", "D", "Sprod",
                "u", "c_eff", "m_eff", "M0_sq", "B", "C0", "W_J")
        rec = {k: [] for k in keys}
        events = []
        c_eff_max, c_eff_max_S = 0.0, 0.0
        u_frozen_at, m_frozen_at = None, None
        lam_out_of_domain = None
        emergent = c.thresholds == "emergent"
        mass_events = []                      # cruces M0² = 0 (modo emergente)
        W_J = 0.0                             # trabajo de la circulación ∫∇V·J∇C dσ (0 si C = V en el interior)
        if c.circulation_C not in ("V", "rho2"):
            raise ValueError("circulation_C: 'V' | 'rho2'")
        tau_seq = list(c.tau_per_dim) if c.tau_per_dim else None
        tau_now = float(tau_seq[0]) if tau_seq else float(c.fp.tau)
        if tau_seq and not emergent:
            raise ValueError("tau_per_dim es una hipótesis del modo 'emergent'")

        def clock_of(f_val, S_val):
            return S_val if emergent else f_val

        def record(f_val, g, clk):
            dual = to_dual(phi[0], phi[1])
            D = float(lam[1] ** 2 - 4.0 * lam[2] * lam[0])
            vals = (sigma, S_int, f_val, clk, self._V(phi, lam), float(dual["theta"]), float(dual["chi"]),
                    float(dual["rho"]), D, float(g @ g) * G_inv, u, c_eff_max, m,
                    float(lam[0]), float(lam[1]), float(lam[2]), W_J)
            for k, v in zip(keys, vals):
                rec[k].append(v)

        def collapse_event(S_ev, sigma_ev, trigger):
            nonlocal d_dim
            d_dim += 1
            gens = chain_generators(d_dim)
            imposed_here = trigger.startswith("umbral")
            ev = {"kind": f"colapso_{d_dim}D", "S": float(S_ev), "S_integrated": float(S_int), "sigma": float(sigma_ev),
                  # las dos lecturas del reloj, siempre juntas: dónde se IMPUSO el
                  # colapso (Prop. 8.1) y dónde lo habría disparado D = 0 (None si
                  # D no cruza cero en el recorrido: el emergente no existe aquí)
                  "S_imposed": float(self.thresholds[d_dim - 1]) if imposed_here else None,
                  "S_emergent": None if imposed_here else float(S_ev),
                  "trigger": trigger, "d_after": d_dim, "algebra": f"C({d_dim + 1},0)",
                  "dim_rep": int(gens[0].shape[0]), "anticommute": bool(all_anticommute(gens)),
                  "signature": signature(gens),
                  "elliptic_symbol_positive": bool(euclidean_symbol(np.array([1.0]), np.ones(d_dim)) > 0),
                  "D_at_event": float(lam[1] ** 2 - 4.0 * lam[2] * lam[0]),
                  "x_over_x_plus": float((phi[0] ** 2 + phi[1] ** 2) / self.x_plus0),
                  "theta": float(to_dual(phi[0], phi[1])["theta"]),
                  "sextic_dimension": sextic_coupling_dimension(d_dim),
                  "status": ("impuesto (Prop. 8.1)" if trigger.startswith("umbral")
                             else "diagnóstico (D = 0 con τ declarado)")}
            events.append(ev)

        g = self._grad(phi, lam)
        f_val = self._f(phi, lam)
        self.f0 = float(f_val)         # entropía de nucleación: 0 en el punto de escape, > 0 en el bounce
        clk = clock_of(f_val, S_int)
        record(f_val, g, clk)
        # umbrales ya superados al emerger (el bounce entrega el campo con f₀ > umbral):
        # el colapso cae DENTRO de la nucleación; se dispara en σ = 0 con esa nota
        if not emergent:
            while pending and clk >= pending[0]:
                S_th = pending.pop(0)
                collapse_event(S_th, 0.0, "umbral de la Década (ya superado al emerger: f₀ ≥ umbral, colapso dentro de la nucleación)")
        f_stop = 1.0 - c.eps_res
        prev_D = float(lam[1] ** 2 - 4.0 * lam[2] * lam[0])
        n = 0
        finished, stop_reason = False, None
        while n < c.max_steps:
            n += 1

            def rhs(ph):
                # flujo de gradiente PROYECTADO sobre el dominio físico (Cap. 2):
                # en la frontera φ_i = 0 con velocidad saliente, la componente se
                # anula y la ligadura no trabaja — Σ̇ = −∇V·(dΦ/dσ) cuenta solo
                # las componentes activas (así S ≡ f exactamente, Teo. 4.5)
                gg = self._grad(ph, lam)
                v = -G_inv * gg
                if c.J_circ != 0.0:
                    # nivel conservativo J∇C (declarado). La ligadura (normal a la
                    # frontera) actúa sobre la velocidad TOTAL: en el interior
                    # ∇V·J∇V = 0 exactamente, pero sobre la frontera la componente
                    # tangencial de la circulación sobrevive y TRABAJA (W_J ≠ 0):
                    # se publica, y la identidad pasa a ser S = f − f₀ + W_J/T₀
                    vc = self._circulation(ph, gg, lam)
                    active = ~((ph <= 0.0) & (v + vc < 0.0))
                    v, vc = np.where(active, v, 0.0), np.where(active, vc, 0.0)
                    return v + vc, float(-(gg @ v)) / self.T0, float(gg @ vc)
                v = np.where((ph <= 0.0) & (v < 0.0), 0.0, v)
                return v, float(-(gg @ v)) / self.T0, 0.0
            k1, s1, w1 = rhs(phi)
            k2, s2, w2 = rhs(phi + 0.5 * d_sigma * k1)
            k3, s3, w3 = rhs(phi + 0.5 * d_sigma * k2)
            k4, s4, w4 = rhs(phi + d_sigma * k3)
            phi_new = np.clip(phi + (d_sigma / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, None)
            dS = (d_sigma / 6.0) * (s1 + 2 * s2 + 2 * s3 + s4)
            W_J += (d_sigma / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
            S_new = S_int + dS
            f_new = self._f(phi_new, lam)
            clk_new = clock_of(f_new, S_new)
            dclk = clk_new - clk
            # --- acoplos (Def. 4.4) en el reloj, si se pide ---------------
            if c.couplings_flow and dclk > 0.0:
                beta = beta_functions(lam[0], lam[1], lam[2], c.fp.a, c.fp.b)
                lam_new = lam + c.fp.flow_sign * tau_now * beta * dclk
                if lam[0] > 0.0 >= lam_new[0]:
                    # inestabilidad de masa: el origen deja de ser mínimo. Se
                    # publica y se SIGUE integrando (V acotado por C0 > 0)
                    w = float(lam[0] / (lam[0] - lam_new[0]))
                    S_flip = float(clk + w * dclk)
                    ev = {"kind": "inestabilidad_masa", "S": S_flip, "S_integrated": float(S_new),
                          "sigma": float(sigma + w * d_sigma), "tau": tau_now, "n_flip": len(mass_events) + 1,
                          "S_flip_first_order": S_flip_first_order(c.delta0, tau_now, c.m_bar, c.b_bar, c.C0, c.fp.a),
                          "lam_at_flip": lam_new.tolist(), "D_at_event": float(lam_new[1] ** 2 - 4.0 * lam_new[2] * lam_new[0]),
                          "d_after": d_dim, "form": DECLARED_FORMS["mass_instability"],
                          "status": "diagnóstico (M0² = 0 con τ declarado; E13: el número no es señal)"}
                    mass_events.append(ev)
                    events.append(ev)
                    if tau_seq is not None and d_dim < 3:
                        # hipótesis τ_d declarada: colapso d → d+1 y reposición de M0²
                        collapse_event(S_flip, sigma + w * d_sigma,
                                       "inestabilidad de masa M0² = 0 (hipótesis τ_d declarada, E13)")
                        lam_new[0] = self.lam0[0]
                        tau_now = float(tau_seq[min(d_dim, len(tau_seq) - 1)])
                        ev["reset"] = {"M0_sq_to": float(self.lam0[0]), "tau_next": tau_now, "d_after": d_dim}
                        # el salto de D que produce la reposición no es un cruce de la espinodal
                        prev_D = float(lam_new[1] ** 2 - 4.0 * lam_new[2] * lam_new[0])
                lam = lam_new
                if lam[2] <= 0.0:
                    lam_out_of_domain = {"S": float(clk_new), "lam": lam.tolist()}
            # --- sellados con forma declarada -----------------------------
            if dclk > 0.0:
                du = _logistic_rate(clk, c.beta_c0, self.S_c) * u * (1.0 - u) * dclk
                if du / dclk > c_eff_max:
                    c_eff_max, c_eff_max_S = float(du / dclk), float(clk)
                u += du
                m += -_m_rate(clk, c.gamma_m, self.S_c2) * (m - c.m_min) * dclk
                if u_frozen_at is None and clk_new >= self.S_c:
                    u_frozen_at = float(u)
                if m_frozen_at is None and clk_new >= self.S_c2:
                    m_frozen_at = float(m)
            # --- eventos de colapso ----------------------------------------
            if not emergent:
                while pending and clk_new >= pending[0]:
                    S_th = pending.pop(0)
                    w = (S_th - clk) / dclk if dclk > 0.0 else 1.0
                    collapse_event(S_th, sigma + min(max(w, 0.0), 1.0) * d_sigma, "umbral de la Década")
            else:
                D_new = float(lam[1] ** 2 - 4.0 * lam[2] * lam[0])
                if prev_D > 0.0 >= D_new and d_dim < 3:
                    collapse_event(clk_new, sigma + d_sigma, "cruce de la espinodal D = 0")
                prev_D = D_new
            phi, S_int, sigma, f_val, clk = phi_new, S_new, sigma + d_sigma, f_new, clk_new
            g = self._grad(phi, lam)
            done_descent = f_val >= f_stop
            # con acoplos en movimiento f no es un reloj fiable: el descenso se da
            # por completado cuando |∇V| cae bajo 1e-6 de la escala T₀/ρ₊
            gnorm_small = float(np.sqrt(g @ g)) < 1e-6 * self.T0 / np.sqrt(self.x_plus0)
            if not emergent:
                finished = done_descent
                stop_reason = "residuo de descarga alcanzado" if finished else None
            else:
                finished = (clk >= c.S_end_emergent or gnorm_small or done_descent
                            or lam_out_of_domain is not None)
                stop_reason = (None if not finished else
                               "S ≥ S_end (fin del recorrido)" if clk >= c.S_end_emergent else
                               "acoplos fuera del dominio (C0 ≤ 0)" if lam_out_of_domain else
                               "descenso completado (residuo de descarga alcanzado)" if done_descent else
                               "descenso completado (∇V ≈ 0)")
            if n % 5 == 0 or finished:
                record(f_val, g, clk)
            if finished:
                break
        if not finished:
            stop_reason = "max_steps agotado"
        rec = {k: np.asarray(v) for k, v in rec.items()}
        checks = {"S0": self._s0_checks(), "descent": self._descent_checks(rec, finished, n, stop_reason)}
        # --- fase de sellado: cuantos de confirmación declarados -----------
        S_end = float(clk)
        if not emergent:
            while pending:
                S_th = pending.pop(0)
                collapse_event(S_th, sigma, "umbral de la Década (declarado tras el residuo)")
            if m_frozen_at is None and S_end >= self.S_c2 - 1e-9:
                m_frozen_at = float(m)
            events.append({"kind": "V3D", "S": self.S_V3D, "sigma": float(sigma), "d_after": d_dim,
                           "sextic_dimension": sextic_coupling_dimension(3),
                           "marginal_in_d3": sextic_coupling_dimension(3) == 0.0,
                           "status": "declarado (cuanto de anticipación/confirmación, Prop. 8.1)"})
            flor = self._florencia()
            flor.update({"kind": "Florencia", "S": self.S_flor, "sigma": float(sigma), "d_after": d_dim})
            events.append(flor)
        checks["seals"] = {
            "c_eff": {"form": DECLARED_FORMS["c_eff"], "u_frozen": u_frozen_at, "c_eff_max": c_eff_max,
                      "S_of_max": c_eff_max_S, "rate_after_seal": _logistic_rate(max(S_end, self.S_c), c.beta_c0, self.S_c),
                      "sealed": u_frozen_at is not None, "status": "declarado (forma) + derivado (congelación)"},
            "m_eff": {"form": DECLARED_FORMS["m_eff"], "m_frozen": m_frozen_at,
                      "rate_after_seal": _m_rate(max(S_end, self.S_c2), c.gamma_m, self.S_c2),
                      "sealed": m_frozen_at is not None, "status": "declarado (forma) + derivado (punto fijo)"}}
        checks["diagonal"] = self._diagonal_checks(rec)
        checks["diagonal"]["circulation"] = {
            "J_circ": c.J_circ, "C": c.circulation_C, "form": DECLARED_FORMS["J_circulation"],
            "W_J_over_T0": float(W_J / self.T0),
            "monotonicity_preserved_exactly": c.circulation_C == "V",
            "grad_norm_integral": float(np.trapezoid(np.sqrt(rec["Sprod"] * c.G), rec["sigma"])) if len(rec["sigma"]) > 1 else 0.0,
            "status": ("declarado (|J| = 0: flujo de gradiente puro)" if c.J_circ == 0.0 else
                       "declarado (|J| y C) + derivado (cruce con Monotonía exacta)" if c.circulation_C == "V" else
                       "declarado (|J| y C) + publicado (W_J ≠ 0: la rotación rígida trabaja contra la inclinación)")}
        checks["handshake_cosmology"] = self._handshake_checks()
        delivered = self._deliver(phi, lam, S_end, u_frozen_at, m_frozen_at, rec) if not emergent else None
        # en modo impuesto, D no cruza cero (acoplos congelados): se publica junto a cada evento
        if not emergent:
            D_traj = rec["D"]
            for ev in events:
                if ev["kind"].startswith("colapso"):
                    ev["S_emergent_note"] = ("D > 0 en todo el recorrido (acoplos congelados): sin cruce emergente"
                                             if np.all(D_traj > 0) else "D cruza cero en el recorrido")
        ledger = self._ledger(checks, events)
        return to_jsonable({"config": {**asdict(c), "fp": asdict(c.fp)}, "status": STATUS,
                            "ledger": ledger,
                            "declared_forms": DECLARED_FORMS, "mode": c.thresholds,
                            "T0": self.T0, "T0_law_3_4": self.T0_law, "x_esc": self.x_esc, "x_plus": self.x_plus0,
                            "landscape": self.land, "trajectory": {k: v.tolist() for k, v in rec.items()},
                            "events": events, "checks": checks, "delivered_state": delivered,
                            "steps": n, "finished": finished, "stop_reason": stop_reason,
                            "couplings_out_of_domain": lam_out_of_domain,
                            "mass_instability_events": mass_events,
                            "tau_per_dim": list(c.tau_per_dim) if c.tau_per_dim else None})

    # ------------------------------------------------------ comprobaciones
    @staticmethod
    def _ledger(checks: dict, events: list) -> dict:
        """El LEDGER obligatorio: cada comprobación y cada evento con su
        etiqueta de estatuto, y el recuento por etiqueta. Es lo que el
        lector debe ver antes que ningún número."""
        rows = []
        for sec, d in checks.items():
            for name, v in d.items():
                if isinstance(v, dict) and "status" in v:
                    rows.append({"section": sec, "item": name, "status": v["status"],
                                 "pass": v.get("pass"), "sealed": v.get("sealed")})
        for ev in events:
            rows.append({"section": "events", "item": ev["kind"], "status": ev.get("status", "—"),
                         "pass": None, "S": ev.get("S")})
        counts = {}
        for r in rows:
            key = r["status"].split(" ")[0].rstrip(":,;(")
            counts[key] = counts.get(key, 0) + 1
        derived_fail = [r for r in rows if r["status"].startswith("derivado") and r["pass"] is False]
        return {"rows": rows, "counts_by_status_word": counts,
                "derived_checks_failed": [f"{r['section']}/{r['item']}" for r in derived_fail],
                "reading": "ningún desenlace: comprobación interna (E8); «impuesto/declarado» marca lo que el "
                           "tratado o el simulador fijan sin derivar; «publicado» marca hallazgos sin veredicto"}

    def _s0_checks(self) -> dict:
        c = self.cfg
        t0n = T0_numeric(c.delta0, c.m_bar, c.b_bar, c.C0)
        d0max = delta0_metastability_max(c.m_bar, c.b_bar, c.e_bar, c.C0, c.theta_nuc)
        d0max_an = delta0_metastability_max_analytic(c.m_bar, c.b_bar, c.e_bar, c.C0, c.theta_nuc)
        k1 = kappa1_tilt(c.m_bar, c.b_bar, c.e_bar, c.C0, c.theta_nuc)
        corr = self.T0 / self.T0_law - 1.0
        first = k1 * np.sqrt(c.delta0)
        ld = self.land
        return {"quasi_cancellation_3_3": {"pass": quasi_cancellation_ok(c.m_bar, c.b_bar, c.C0), "status": "derivado"},
                "T0_tilt_first_order": {"kappa1": k1, "kappa1_form": "ē·(cos θ − sin θ)·√κ₊/(√2·c̄)",
                                        "predicted_correction": first, "measured_correction": corr,
                                        "ratio_measured_over_first_order": corr / first if first else None,
                                        "residual_over_delta0": (corr - first) / c.delta0,
                                        "T0_first_order": T0_tilt_first_order(c.delta0, c.m_bar, c.b_bar, c.e_bar, c.C0, c.theta_nuc),
                                        # el residuo es el orden siguiente, O(δ₀) relativo: se exige que su
                                        # coeficiente quede por debajo de κ₁/2 (declarado)
                                        "pass": bool(abs(corr - first) < 0.5 * k1 * c.delta0),
                                        "note": "T₀_full/T₀_ley − 1 ≃ κ₁·√δ₀ + O(δ₀); κ₁ = 1.361 con las formas por defecto",
                                        "status": "derivado"},
                "delta0_max_closed_form": {"analytic": d0max_an, "bisection": d0max,
                                           "rel_err": abs(d0max_an - d0max) / d0max if np.isfinite(d0max) and d0max > 0 else 0.0,
                                           "form": "2·g_max(m̄, b̄, C0)²/(ē·(cos θ − sin θ))² ∝ ē⁻²",
                                           "pass": bool((not np.isfinite(d0max) and not np.isfinite(d0max_an))
                                                        or abs(d0max_an - d0max) < 1e-6 * d0max),
                                           "status": "derivado"},
                "metastability_8_6_D_only": {"pass": metastability_bound(c.delta0, c.m_bar, c.b_bar, c.C0),
                                             "D_S0": discriminant_basal(c.delta0, c.m_bar, c.b_bar, c.C0), "status": "derivado"},
                "metastability_with_tilt": {"pass": ld["metastable"], "delta0": c.delta0, "delta0_max": d0max,
                                            "barrier_height_over_T0": ld["barrier_height"] / ld["T0_full"],
                                            "rho_false_vacuum": ld["rho_fv"], "rho_barrier": ld["rho_barrier"],
                                            "rho_true_vacuum": ld["rho_tv"], "rho_esc": ld["rho_esc"],
                                            "finding": "la Obs. 8.6 (D > 0) es necesaria, no suficiente: con la inclinación "
                                                       "η = ē·δ0³ el falso vacío existe solo para δ0 < δ0_max(m̄, b̄, ē, C0)",
                                            "status": "derivado (hallazgo del simulador)"},
                "T0_scaling_3_4": {"law_c_bar_delta0_cubed": self.T0_law, "numeric_no_tilt": t0n,
                                   "rel_err_law_vs_numeric": abs(t0n - self.T0_law) / self.T0_law,
                                   "T0_full_with_tilt": self.T0, "tilt_correction": self.T0 / self.T0_law - 1.0,
                                   "c_bar": c_bar(c.m_bar, c.b_bar, c.C0),
                                   "pass": abs(t0n - self.T0_law) / self.T0_law < 1e-9,
                                   "note": "la corrección de la inclinación es O(δ0^{1/2}) relativa (Prop. 3.4: η·ρ₊ = O(δ0^{7/2}))",
                                   "status": "derivado"},
                "measure_normalizable_3_1": {"pass": measure_normalizable(c.delta0, c.C0), "status": "derivado"},
                "chi_residue_3_4": {"analytic": chi_residue(c.delta0, c.m_bar, c.e_bar),
                                    "false_vacuum_chi": ld["rho_fv"] * (np.cos(c.theta_nuc) - np.sin(c.theta_nuc)) / np.sqrt(2.0),
                                    "status": "derivado"},
                "nucleation": {"mode": c.nucleation, "x_esc": self.x_esc, "x_esc_over_x_plus": self.x_esc / self.x_plus0,
                               "bounce": None if self.bounce is None else
                               {k: getattr(self.bounce, k) for k in ("d", "phi0", "B", "Gamma_over_A", "f0",
                                                                     "barrier_over_T0", "converged")},
                               "V_esc_minus_V_fv_over_T0": (self._V(np.array([np.sqrt(self.x_esc) * np.cos(c.theta_nuc),
                                                                              np.sqrt(self.x_esc) * np.sin(c.theta_nuc)]), self.lam0)
                                                            - self.V_fv) / self.T0,
                               "Gamma0": None if self.bounce is None else self.bounce.Gamma_over_A,
                               "status": ("declarado (Γ₀ no calculada; punto de escape V = V_fv)" if self.bounce is None else
                                          f"derivado con convenciones declaradas (bounce O({c.bounce_d}), prefactor A no fijado)")}}

    def _descent_checks(self, rec: dict, finished: bool, n: int, reason) -> dict:
        V, S, f, Sp = rec["V"], rec["S"], rec["f"], rec["Sprod"]
        dV = np.diff(V)
        x = rec["rho"] ** 2
        emergent = self.cfg.thresholds == "emergent"
        return {"monotonia_4_5": {"max_dV_over_T0": float(dV.max() / self.T0) if dV.size else 0.0,
                                  "pass": bool(dV.size == 0 or dV.max() <= 1e-12 * self.T0) if not emergent else None,
                                  "note": None if not emergent else "con acoplos en movimiento V no es función de Lyapunov",
                                  "status": "derivado"},
                "produccion_entropica_4_5": {"min": float(Sp.min()), "pass": bool(Sp.min() >= 0.0), "status": "derivado"},
                "exclusion_4_7": {"pass": bool(np.all(np.diff(x) >= -1e-12 * self.x_plus0)) if not emergent else None,
                                  "status": "derivado"},
                "S_equals_f_identity": {"max_abs_diff": float(np.max(np.abs(S - (f - self.f0) - rec["W_J"] / self.T0))),
                                        "f0_nucleation": float(self.f0),
                                        "W_J_over_T0_final": float(rec["W_J"][-1] / self.T0),
                                        "pass": bool(np.max(np.abs(S - (f - self.f0) - rec["W_J"] / self.T0)) < 1e-6) if not emergent else None,
                                        "note": "Σ̇ = −dV/dσ ⟹ ∫Σ̇dσ = V(σ=0) − V = T₀·(f − f₀) exactamente en el flujo de "
                                                "gradiente (Teo. 4.5) con la normalización declarada; f₀ = 0 en el punto de "
                                                "escape y > 0 si la nucleación entrega el campo ya parcialmente descargado "
                                                "(bounce); con circulación J∇C la identidad es S = f − f₀ + W_J/T₀ (W_J = 0 en "
                                                "el interior para C = V; ≠ 0 sobre la frontera o con C rígida); en modo "
                                                "emergente V cambia con λ",
                                        "status": "derivado"},
                "exit_to_mass_pole_3_5": {"theta_final": float(rec["theta"][-1]), "pass": bool(rec["theta"][-1] < 0.35),
                                          "status": "derivado"},
                "descent_finished": {"pass": finished, "steps": n, "reason": reason, "f_final": float(f[-1]),
                                     "S_final": float(S[-1]), "sigma_final": float(rec["sigma"][-1]),
                                     "sigma_hat_final": float(rec["sigma"][-1] * self.cfg.delta0 ** 2), "status": "derivado"}}

    def _diagonal_checks(self, rec: dict) -> dict:
        th, clk = rec["theta"], rec["clock"]
        crossed = bool(np.any(th >= THETA_DIAGONAL))
        S_cross_flow = float(clk[np.argmax(th >= THETA_DIAGONAL)]) if crossed else None
        return {"flow": {"crossed": crossed, "S_at_crossing": S_cross_flow, "theta_final": float(th[-1]),
                         "theta_max": float(th.max()),
                         "finding": "el Basal solo lleva θ al polo de masa y lo mantiene: el transporte azimutal "
                                    "hasta la diagonal (Prop. 3.5) no está implementado", "status": "publicado"},
                "imposed": {"S_at_crossing": self.S_V3D, "form": DECLARED_FORMS["theta_imposed"], "status": "impuesto"}}

    def _florencia(self) -> dict:
        from mass_program.B7_empalme import beta3_derived, m_H_predicted
        c = self.cfg
        gens = chain_generators(3)
        rot = florencia_rotation(gens, 1)
        rot2 = florencia_rotation(gens, 2)
        # RP en la loncha: corte 1D de V0 a lo largo de ρ (η apagada), medida e^{−V}
        rho_grid = np.linspace(0.0, 2.5 * np.sqrt(self.x_plus0), 41)
        Vg = np.array([V0(r, 0.0, c.delta0, c.m_bar, c.b_bar, 0.0, c.C0) for r in rho_grid]) / self.T0
        rp_pos = rp_min_eigenvalue(rho_grid / rho_grid.max(), Vg, J=+1.0)
        rp_neg = rp_min_eigenvalue(rho_grid / rho_grid.max(), Vg, J=-1.0)
        beta3 = beta3_derived(c.delta0, "12.1", c.m_bar, c.b_bar, c.C0)
        mH = m_H_predicted(c.delta0, c.m_bar, c.b_bar, c.C0)
        return {"algebra_before": "C(4,0)", "algebra_after": "C(3,1)",
                "signature_after": signature(rot), "signature_pass": signature(rot) == [-1, 1, 1, 1],
                "anticommute_after": bool(all_anticommute(rot)),
                "negative_control_two_rotations": signature(rot2),
                "negative_control_pass": signature(rot2).count(-1) == 2,
                "regime_after": regime(self.S_flor),
                "rp_slice_min_eig_J_plus": rp_pos, "rp_slice_pass": rp_pos >= -1e-10,
                "rp_negative_control_J_minus": rp_neg, "rp_negative_control_pass": rp_neg < 0.0,
                "beta3_from_empalme": beta3, "m_H_GeV": mH,
                "m_H_identity_pass": abs(mH - np.sqrt(2.0 * beta3) * C.V3_GEV) < 1e-9,
                "m_H_PDG_GeV": C.M_HIGGS_PDG,
                "m_H_note": "identidad m_H = √(2β₃)v₃ comprobada; el VALOR depende de δ₀ (input) y β₃ es "
                            "condicional (frente 7): no es predicción",
                "status": "derivado (giro, firma, RP juguete, identidad) — evento declarado en S = 1,001"}

    def _handshake_checks(self) -> dict:
        from cosmology.background import OMEGA_M0, OMEGA_R0, H_of_z
        z = np.array([0.0, 0.5, 1.0, 3.0, 10.0])
        H_mcmc0 = H_of_z(z, eps=0.0)
        H_lcdm = C.H0_MCMC * np.sqrt(OMEGA_M0 * (1 + z) ** 3 + OMEGA_R0 * (1 + z) ** 4 + (1 - OMEGA_M0 - OMEGA_R0))
        dev = float(np.max(np.abs(H_mcmc0 / H_lcdm - 1.0)))
        return {"newton_seal_9_3": {"ratio_at_seal": newton_seal_ratio(1.0, 1.0), "pass": newton_seal_ratio(1.0, 1.0) == 1.0,
                                    "status": "derivado"},
                "atlas_healthy_9_4": {"lambda_K": 1.0 + C.EPSILON_K, "pass": atlas_healthy(1.0 + C.EPSILON_K), "status": "derivado"},
                "lcdm_recovery_A_1": {"max_rel_dev_eps0": dev, "pass": dev < 1e-12, "status": "derivado"},
                "dictionary": {"exists": False, "pass": None,
                               "note": "no hay fórmula (δ₀, m̄, b̄, C₀, ē, f, Φ_ten) → (Ω_id,0, ε_Λ, z_trans, α_n, κ_lat, η_lat): "
                                       "fila diccionario-primordial-cosmologico", "status": "hueco declarado"}}

    def _deliver(self, phi, lam, S_end, u_frozen, m_frozen, rec) -> dict:
        from mass_program.B7_empalme import beta3_derived, sealed_curvature_lambda
        c = self.cfg
        dual = to_dual(phi[0], phi[1])
        return {"S": self.S_flor, "regime": regime(self.S_flor), "algebra": "C(3,1)", "signature": [-1, 1, 1, 1],
                "d": 3, "S_end_of_descent": S_end, "f_end_of_descent": float(rec["f"][-1]),
                "T_residual_over_T0": float(1.0 - rec["f"][-1]),
                "couplings_sealed": {"M0_sq": float(lam[0]), "B": float(lam[1]), "C0": float(lam[2]),
                                     "D": float(lam[1] ** 2 - 4.0 * lam[2] * lam[0])},
                "lambda_Ad_sealed": sealed_curvature_lambda(c.delta0, c.m_bar, c.b_bar, c.C0),
                "beta3": beta3_derived(c.delta0, "12.1", c.m_bar, c.b_bar, c.C0),
                "c_eff_sealed_u": u_frozen, "m_eff_sealed": m_frozen,
                "theta_flow": float(dual["theta"]), "chi_flow": float(dual["chi"]), "rho_flow": float(dual["rho"]),
                "theta_imposed": THETA_DIAGONAL, "chi_residue_analytic_3_4": chi_residue(c.delta0, c.m_bar, c.e_bar),
                "delta0_input": c.delta0,
                "delta0_next_cycle": None,
                "delta0_next_cycle_note": "F_Vic(δ₀) exige γ_R (frente 4): no calculado (core.victoria.STATUS_LYDIA)",
                "Phi_ten": 0.0, "Phi_ten_note": DECLARED_FORMS["Phi_ten_at_florencia"],
                "channels_initial": None,
                "readable_by_cosmology": False,
                "readable_note": "el diccionario primordial → cosmológico no existe: cosmology/ no puede consumir "
                                 "este estado; se publica tal cual (fila diccionario-primordial-cosmologico)"}


def diagonal_crossing_S(delta0: float, J: float, circulation_C: str = "V", **kwargs) -> dict:
    """S (reloj f) en que el flujo con circulación |J| cruza θ = π/4, o None."""
    out = SClock(ClockConfig(delta0=delta0, J_circ=J, circulation_C=circulation_C, **kwargs)).run()
    fl = out["checks"]["diagonal"]["flow"]
    return {"J": J, "crossed": fl["crossed"], "S_cross": fl["S_at_crossing"], "theta_final": fl["theta_final"],
            "theta_max": fl["theta_max"], "W_J_over_T0": out["checks"]["diagonal"]["circulation"]["W_J_over_T0"],
            "S_equals_f_max_diff": out["checks"]["descent"]["S_equals_f_identity"]["max_abs_diff"],
            "monotonia_pass": out["checks"]["descent"]["monotonia_4_5"]["pass"],
            "grad_norm_integral": out["checks"]["diagonal"]["circulation"]["grad_norm_integral"],
            "finished": out["finished"]}


def J_min_threshold(delta0: float, circulation_C: str = "V", J_lo: float = 0.05, J_hi: float = 50.0,
                    rtol: float = 2e-3, **kwargs) -> dict:
    """El |J| mínimo con el que el flujo ALCANZA la diagonal (bisección
    sobre el booleano «cruza»), y el S del cruce justo por encima del
    umbral, S_tope: el S más alto en que la diagonal puede cruzarse con
    este C. Con C = V, J∇V se anula en los puntos críticos y el vacío
    verdadero 2D está sobre el polo de masa (θ = 0): θ sube, alcanza su
    máximo y vuelve a 0 — por eso el cruce tiene un S máximo y no basta
    subir |J| para llevarlo a S = 1 (subir |J| lo ADELANTA). Publica
    también la expectativa E13 |J|_est = (π/4)/∫|∇C|dσ (J = 0)."""
    base = diagonal_crossing_S(delta0, 0.0, circulation_C, **kwargs)
    J_est = (np.pi / 4.0) / base["grad_norm_integral"] if base["grad_norm_integral"] > 0 else None
    lo, hi = J_lo, J_hi
    r_lo, r_hi = diagonal_crossing_S(delta0, lo, circulation_C, **kwargs), diagonal_crossing_S(delta0, hi, circulation_C, **kwargs)
    if r_lo["crossed"] or not r_hi["crossed"]:
        return {"delta0": delta0, "C": circulation_C, "J_min": None, "bracket_ok": False,
                "crossed_at_J_lo": r_lo["crossed"], "crossed_at_J_hi": r_hi["crossed"],
                "J_estimate_pi4_over_int_gradC": J_est, "status": "sin umbral en el intervalo declarado"}
    n = 0
    while hi / lo > 1.0 + rtol and n < 60:
        n += 1
        mid = float(np.sqrt(lo * hi))
        if diagonal_crossing_S(delta0, mid, circulation_C, **kwargs)["crossed"]:
            hi = mid
        else:
            lo = mid
    at = diagonal_crossing_S(delta0, hi, circulation_C, **kwargs)
    return {"delta0": delta0, "C": circulation_C, "J_min": hi, "bracket_ok": True, "iterations": n,
            "S_cross_threshold": at["S_cross"], "theta_final_at_J_min": at["theta_final"], "theta_max_at_J_min": at["theta_max"],
            "W_J_over_T0": at["W_J_over_T0"], "S_equals_f_max_diff": at["S_equals_f_max_diff"],
            "monotonia_pass": at["monotonia_pass"], "finished": at["finished"],
            # frontera: el flujo vuelve a φ_E = 0 y la componente tangencial de la circulación
            # (que la ligadura normal no cancela) desplaza el equilibrio: el descenso no completa
            "returns_to_boundary": bool(at["theta_final"] <= 1e-9),
            "J_estimate_pi4_over_int_gradC": J_est,
            "ratio_J_min_over_estimate": (hi / J_est) if J_est else None,
            "status": "publicado: |J|_min y S_tope del cruce con C declarado (calibración, no derivación)"}


def J_required(delta0: float, S_target: float, circulation_C: str = "V", J_lo: float = 0.05, J_hi: float = 50.0,
               xtol: float = 1e-3, **kwargs) -> dict:
    """El |J| que sitúa el cruce de la diagonal en S = S_target (bisección en
    log|J| sobre S_cross(|J|), decreciente en |J|). Publica también la
    expectativa del autor |J|_est = (π/4)/∫|∇C| dσ sobre el descenso con
    J = 0 (E13: número esperado, no derivado) y el cociente."""
    base = diagonal_crossing_S(delta0, 0.0, circulation_C, **kwargs)
    J_est = (np.pi / 4.0) / base["grad_norm_integral"] if base["grad_norm_integral"] > 0 else None

    def S_of(logJ):
        r = diagonal_crossing_S(delta0, float(np.exp(logJ)), circulation_C, **kwargs)
        # sin cruce: el reloj "termina" en S_final > S_target (cruce virtual más allá)
        return (r["S_cross"] if r["crossed"] else 2.0) - S_target
    lo, hi = np.log(J_lo), np.log(J_hi)
    f_lo, f_hi = S_of(lo), S_of(hi)
    if not (f_lo > 0.0 > f_hi):
        return {"delta0": delta0, "S_target": S_target, "C": circulation_C, "J_required": None,
                "bracket_ok": False, "S_cross_at_J_lo": f_lo + S_target, "S_cross_at_J_hi": f_hi + S_target,
                "J_estimate_pi4_over_int_gradC": J_est, "status": "sin |J| en el intervalo declarado que cruce en S_target"}
    logJ = brentq(S_of, lo, hi, xtol=xtol)
    at = diagonal_crossing_S(delta0, float(np.exp(logJ)), circulation_C, **kwargs)
    return {"delta0": delta0, "S_target": S_target, "C": circulation_C, "J_required": float(np.exp(logJ)),
            "bracket_ok": True, "S_cross": at["S_cross"], "W_J_over_T0": at["W_J_over_T0"],
            "S_equals_f_max_diff": at["S_equals_f_max_diff"], "monotonia_pass": at["monotonia_pass"],
            "J_estimate_pi4_over_int_gradC": J_est,
            "ratio_required_over_estimate": (float(np.exp(logJ)) / J_est) if J_est else None,
            "status": "publicado: |J| es una calibración declarada con significado, no una derivación (frente 2)"}


def run_clock(delta0: float = 0.01, **kwargs) -> dict:
    """Atajo: un recorrido completo con la configuración por defecto
    (modo 'imposed', acoplos congelados) y δ₀ dado."""
    return SClock(ClockConfig(delta0=delta0, **kwargs)).run()


def emergent_diagnostic(delta0: float, tau: float = 1.0, **kwargs) -> dict:
    """Modo DIAGNÓSTICO: acoplos por las β de Fokker–Planck (cierre
    canónico) con diccionario τ declarado; el reloj es S = ∫Σ̇dσ/T₀ y los
    colapsos se disparan donde D(S) cruza cero. Publica dónde caerían
    frente a los umbrales de la Década; NO decide λ (frente 2)."""
    fp = FPClosure(tau=tau)
    out = SClock(ClockConfig(delta0=delta0, thresholds="emergent", couplings_flow=True,
                             fp=fp, **kwargs)).run()
    crossings = [e["S"] for e in out["events"] if e["kind"].startswith("colapso")]
    D = np.asarray(out["trajectory"]["D"])
    flips = out["mass_instability_events"]
    first = flips[0] if flips else None
    return {"delta0": delta0, "tau": tau, "S_crossings": crossings,
            "decade_thresholds": C.decade_thresholds()[:3],
            "D_initial": float(D[0]), "D_final": float(D[-1]), "D_min": float(D.min()),
            "D_sinks": bool(D[-1] < D[0]), "S_final": float(out["trajectory"]["S"][-1]),
            "stop_reason": out["stop_reason"], "couplings_out_of_domain": out["couplings_out_of_domain"],
            "mass_instability": None if first is None else {
                "S_flip": first["S"], "S_flip_first_order": first["S_flip_first_order"],
                "ratio": (first["S"] / first["S_flip_first_order"]) if first["S_flip_first_order"] else None,
                "n_flips": len(flips), "status": first["status"]},
            "tau_decade_table": tau_decade_table(delta0, kwargs.get("m_bar", M_BAR), kwargs.get("b_bar", B_BAR),
                                                 kwargs.get("C0", C0_DEFAULT), fp.a),
            "tau_per_dim": out["tau_per_dim"],
            "collapse_triggers": [e["trigger"] for e in out["events"] if e["kind"].startswith("colapso")],
            "status": "diagnóstico sin estatuto: el diccionario τ no es derivable (frente 2)"}
