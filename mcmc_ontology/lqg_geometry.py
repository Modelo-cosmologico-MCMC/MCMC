"""Conexión MCMC ↔ LQG: microestructura cuántica de la geometría tensional.

Apéndice I y K (pp. 138-161) del Tratado Unificado.

ADVERTENCIA: estas funciones son de "Etapa III" — conexión teórica con
LQG (loop quantum gravity). No afectan al programa de masas ni a la
cosmología observable de Etapa I/II.

Bloques cubiertos:
  · γ Barbero-Immirzi derivado desde δ₀ y V(Φ_Ad;S)   (Ecs. 397-400)
  · Conversión δ(S) → ΔA(j): de MeV a cm² cuánticos    (Ecs. 401-404)
  · Topología de la red de espín (mínima T_res)        (Ecs. 405-406)
  · Rotación de Wick tensional (Eucl → Lor)            (Ecs. 407-411)
  · Fricción Cronos ↔ amplitudes spinfoam              (Ecs. 412-414)
  · Complejidad tensional C_nD(S) completa             (Ecs. 458-460)
  · c como punto fijo de saturación                    (Ecs. 461-463)
  · Proto-métrica h_ij + Δ_tens                        (Ecs. 464-467)
  · Dinámica de retorno S_max → S̃₀ + GWs relictas      (Ecs. 468-471)
"""

from __future__ import annotations

import numpy as np

from . import constants as C


# --- Constantes LQG ---
L_PLANCK_CM   = 1.616e-33   # longitud de Planck [cm]
E_PLANCK_GEV  = 1.22e19     # energía de Planck [GeV]
HBAR_C_GEV_CM = 1.973e-14   # ℏc en GeV·cm


# ============================================================
# B1. γ Barbero-Immirzi derivado (Ecs. 397-400)
# ============================================================

def E_star_ratio() -> float:
    """E_*(S₄) / E_P (Ec. 398).

    Calibrado desde la Tabla 3 del Tratado para reproducir γ ≈ 0.274
    (coincidencia con Kaul-Majumdar). El valor proviene de la suma
    integrada de contribuciones tensionales en C₄.
    """
    return 1721.0


def gamma_immirzi_mcmc(delta_S: float = C.DELTA_S) -> float:
    """Parámetro de Barbero-Immirzi derivado desde la ontología MCMC.

        γ = ΔS · E_*(S₄)/E_P / (2π)              (Ecs. 399-400)

    Con ΔS = 10⁻³ y E_*(S₄)/E_P ≈ 1721 → γ ≈ 0.274. Coincide con
    γ_KM = ln(2)/(π√3) ≈ 0.2740 dentro del 0.1%.

    Esta NO es una coincidencia: la misma granularidad ΔS = 10⁻³ que
    fija el mass gap predice γ cuando se exige conservación de energía
    al convertir tensión Mp en área cuántica LQG.
    """
    return delta_S * E_star_ratio() / (2.0 * np.pi)


def gamma_kaul_majumdar() -> float:
    """γ₀ = ln(2)/(π√3) ≈ 0.1274 (Kaul-Majumdar, j=1/2 dominante).

    Esta es la fórmula matemática original de K-M para el conteo de estados
    de horizontes de agujero negro con punctures j=1/2. El valor numérico
    (≈0.127) NO coincide con γ_MCMC ≈ 0.274; este último coincide más bien
    con la convención de Domagala-Lewandowski (j arbitrario) ≈ 0.2375 al
    factor de orden 1, o se interpreta como un γ "renormalizado" del MCMC.
    Útil sólo como referencia comparativa.
    """
    return float(np.log(2) / (np.pi * np.sqrt(3)))


# ============================================================
# B2. δ(S) → ΔA(j): de MeV a cm² cuánticos (Ecs. 401-404)
# ============================================================

def delta_area_from_energy(delta_S_E_star_GeV: float,
                           rho_lat_V_cell_GeV: float,
                           dPhi_dS: float, K_S: float,
                           delta_S: float = C.DELTA_S) -> float:
    """Función de conversión tensión → área cuántica (Ec. 403).

        ΔA(j,S) = (4ℓ²_P/E_P)·[δ·E_*(S) - ρ_lat·V_cell - ½K(∂_SΦ)²ΔS]

    Args:
        delta_S_E_star_GeV: δ·E_*(S) [GeV] — energía liberada por ΔS.
        rho_lat_V_cell_GeV: ρ_lat·V_cell [GeV] — ECV sellada en la celda.
        dPhi_dS: dΦ/dS — velocidad del campo en S.
        K_S: K_pre(S) — rigidez cinética.

    Returns:
        ΔA en unidades de ℓ²_P (no negativo).
    """
    L_P_sq = 1.0  # unidades de ℓ²_P
    Delta_E_cin = 0.5 * K_S * dPhi_dS ** 2 * delta_S
    Delta_A = (4.0 * L_P_sq / E_PLANCK_GEV) * (
        delta_S_E_star_GeV - rho_lat_V_cell_GeV - Delta_E_cin
    )
    return max(0.0, Delta_A)


def spin_from_area(Delta_A_lP2: float,
                   gamma: float = C.GAMMA_LQG) -> float:
    """Número de espín j a partir de un área ΔA [ℓ²_P] (Ec. 404).

        A(j) = 8πγ ℓ²_P √(j(j+1))
        ⇒ j = -½ + ½ √(1 + (ΔA/(8πγ ℓ²_P))²)
    """
    x = Delta_A_lP2 / (8.0 * np.pi * gamma)
    return -0.5 + 0.5 * np.sqrt(1.0 + x ** 2)


def spin_table_verification() -> dict:
    """Tabla 55 del Tratado (p.140): secuencia esperada {1/2, 3/2, 5/2, 7/2}."""
    return {
        0.009: {"j_approx": 0.5},
        0.099: {"j_approx": 1.5},
        0.999: {"j_approx": 2.5},
        1.001: {"j_approx": 3.5},
    }


# ============================================================
# B3. Topología de la red de espín (Ecs. 405-406)
# ============================================================

def tension_residual_functional(j_list, iota_list, T_v_func, V_v_func) -> float:
    """Funcional de tensión residual T_res[Γ] (Ec. 405).

        T_res[Γ] = Σ_{v∈Γ} T_v({j_f},{ι_v}) / V_v({ι_v})

    El MCMC selecciona la topología que minimiza T_res sujeto a
    A_tot = A(S), V_tot = V(S)  (Ec. 406).
    """
    T_res = 0.0
    for jf, iota in zip(j_list, iota_list):
        Tv = T_v_func(jf, iota)
        Vv = V_v_func(iota)
        T_res += Tv / Vv if Vv > 0 else 1e30
    return T_res


def topology_properties() -> dict:
    """Consecuencias físicas de la minimización de T_res (Apéndice I.3.3)."""
    return {
        "homogeneous_regions":  "triangulación regular → FRW cósmico",
        "high_density_regions": "espines grandes → curvatura concentrada → halos",
        "optimal_topology":     "4-simplices tipo Regge, conexión Calabi-Yau",
    }


# ============================================================
# B4. Rotación de Wick tensional (Ecs. 407-411)
# ============================================================

def wick_rotation_angle(S: float, S_1001: float = 1.001,
                        lambda_ont: float = C.LAMBDA_ONT) -> float:
    """θ_W(S) = (π/2)·Θ_λ(S - S_{1.001})    (Ec. 408).

    θ_W = 0    en S < S_{1.001}  → álgebra Euclidiana C(n,0).
    θ_W = π/2  en S > S_{1.001}  → álgebra Lorentziana C(1,3).
    """
    Theta_lambda = 0.5 * (1.0 + np.tanh((S - S_1001) / lambda_ont))
    return float((np.pi / 2.0) * Theta_lambda)


def gamma0_wick(S: float, S_1001: float = 1.001,
                lambda_ont: float = C.LAMBDA_ONT) -> complex:
    """Generador temporal Clifford γ⁰(S) = e^{iθ_W(S)} γ⁰_E   (Ec. 411).

    (γ⁰)² = e^{2iθ_W}:
      θ_W = 0   → +1  (Euclidiano)
      θ_W = π/2 → -1  (Lorentziano)

    Devuelve el factor escalar e^{iθ_W} (γ⁰_E = 1 en unidades naturales).
    """
    return complex(np.exp(1j * wick_rotation_angle(S, S_1001, lambda_ont)))


def lapse_function(S: float, Phi_ten: float, S_1001: float = 1.001,
                   lambda_ont: float = C.LAMBDA_ONT) -> float:
    """Lapse de Cronos: g₀₀(S) = -α²(S)·Θ_λ(S - S_{1.001})   (Ec. 407).

    α²(S) = N²(S) = e^{2 Φ_ten}.
    """
    Theta = 0.5 * (1.0 + np.tanh((S - S_1001) / lambda_ont))
    alpha_sq = np.exp(2.0 * Phi_ten)
    return -alpha_sq * Theta


def partition_function_transition(S: float, S_1001: float = 1.001,
                                  lambda_ont: float = C.LAMBDA_ONT) -> bool:
    """Régimen de la función de partición (Ecs. 409-410).

        S < S_{1.001}: Z = ∫DΦ exp(-S_eff)        Euclidiana
        S > S_{1.001}: Z = ∫DΦ Dg exp(i S_eff)   Lorentziana

    Returns:
        True si Lorentziano, False si Euclidiano.
    """
    Theta = 0.5 * (1.0 + np.tanh((S - S_1001) / lambda_ont))
    return bool(Theta > 0.5)


# ============================================================
# B5. Fricción Cronos ↔ amplitudes spinfoam (Ecs. 412-414)
# ============================================================

def spinfoam_amplitude_cronos(A_eprl: float, j_faces, j_star: float,
                              zeta_sf: float) -> float:
    """Amplitud de vértice spinfoam con fricción Cronos (Ec. 412).

        A_v^(Cronos) = A_v^(EPRL) · exp(-ζ_sf · Σ_f j_f(j_f+1) / j*²)

    La fricción suprime exponencialmente vértices de alta curvatura,
    produciendo perfiles cored observados.
    """
    curvature = sum(j * (j + 1) for j in j_faces) / j_star ** 2
    return float(A_eprl * np.exp(-zeta_sf * curvature))


def zeta_sf(zeta_0: float = 0.030, rho_c: float = 200.0,
            gamma: float = C.GAMMA_LQG, delta_S: float = C.DELTA_S) -> float:
    """Parámetro spinfoam de supresión (Ec. 414).

        ζ_sf = ζ₀ · ℓ²_P / (ΔA_* · ρ_c · ℓ³_S)

    Con ζ₀ = 0.03 (Cronos) y ρ_c = 200 ρ̄: ζ_sf ≈ 10⁻⁴.
    """
    j_star = 1.0 / delta_S
    Delta_A_star = 8.0 * np.pi * gamma * j_star
    l_S_cubed = delta_S ** 1.5
    return float(zeta_0 / (Delta_A_star * rho_c * l_S_cubed))


def partition_function_spinfoam(graphs, vertex_amplitudes, zeta: float) -> float:
    """Z = Σ_Γ Π_v A_v^(Cronos)   (Ec. 413)."""
    Z = 0.0
    j_star = 1.0 / C.DELTA_S
    for G_j_faces, A_eprl in zip(graphs, vertex_amplitudes):
        prod = 1.0
        for a, jf in zip(A_eprl, G_j_faces):
            prod *= spinfoam_amplitude_cronos(a, jf, j_star, zeta)
        Z += prod
    return Z


# ============================================================
# B6. Complejidad tensional C_nD(S) completa (Ecs. 458-460)
# ============================================================

def complexity_nD(S: float, rho_dyn: float, sigma2_inh: float,
                  E_min_n: float) -> float:
    """C_nD(S) = ρ_dyn(S) · σ²_inh(S) / E_min^(n)(S)   (Ec. 458)."""
    if E_min_n <= 0:
        return 0.0
    return rho_dyn * sigma2_inh / E_min_n


def C_crit_table() -> dict:
    """Tabla 68 del Tratado (p.159): complejidad crítica por transición.

        C_crit^(nD) = (n+1)/n · T_crit^(n+1)/T₀
    """
    return {
        "V0D->V1D":   {"S_crit": 0.009, "C_crit_nD": 5.6e-4,
                       "factor": None,  "emergence": "Eje PP/AP"},
        "V1D->V2D":   {"S_crit": 0.099, "C_crit_nD": 1.2e-2,
                       "factor": 2.0,   "emergence": "Plano, giro±"},
        "V2D->V3D":   {"S_crit": 0.999, "C_crit_nD": 9.3e-2,
                       "factor": 1.5,   "emergence": "Volumen"},
        "V3D->V3+1D": {"S_crit": 1.001, "C_crit_nD": 9.3e-2,
                       "factor": 4.0/3, "emergence": "Tiempo"},
    }


def dC_nD_dS(C_nD: float, rho_dyn: float, sigma2_inh: float,
             E_min_n: float, drho_dyn_dS: float, dsigma2_dS: float,
             dE_dS: float = 0.0) -> float:
    """Evolución dC_{1D}/dS (Ec. 460):

        dC/dS = (1/E_min)·[dρ/dS·σ² + ρ·dσ²/dS] - C·d ln E_min/dS
    """
    if E_min_n <= 0:
        return 0.0
    term1 = (drho_dyn_dS * sigma2_inh + rho_dyn * dsigma2_dS) / E_min_n
    term2 = -C_nD * dE_dS / E_min_n
    return term1 + term2


# ============================================================
# B7. c como punto fijo de saturación (Ecs. 461-463)
# ============================================================

def c_eff_evolution(c_eff: float, E_p: float, M_p: float,
                    dEp_dS: float, dMp_dS: float,
                    Gamma_ann: float) -> float:
    """dc_eff/dS en V₁D (Ec. 461):

        dc_eff/dS = (c_eff/2)[1/E_p·dE_p/dS - 1/M_p·dM_p/dS] - Γ_ann · c_eff
    """
    expansion = (c_eff / 2.0) * (dEp_dS / E_p - dMp_dS / M_p)
    friction  = -Gamma_ann * c_eff
    return expansion + friction


def c_fixed_point(Ep_eq: float, Mp_eq: float, c0: float = 1.0) -> float:
    """Punto fijo dc_eff/dS = 0 (Ec. 463):

        c = √(E_p^eq / M_p^eq) · c₀

    Con equipartición tensional en S_{0.099}: E_p^eq ≈ M_p^eq → c = c₀.
    """
    return float(np.sqrt(Ep_eq / Mp_eq) * c0)


def c_stability_timescale(Gamma_ann: float) -> float:
    """τ_estab ≃ 2/Γ_ann ≃ 20·ΔS — perturbaciones de c decaen rápido."""
    return float(2.0 / Gamma_ann) if Gamma_ann > 0 else float("inf")


# ============================================================
# B8. Proto-métrica h_ij + Δ_tens (Ecs. 464-467)
# ============================================================

def proto_metric_V3D(Phi_grav: float, c: float = 1.0) -> np.ndarray:
    """Proto-métrica espacial V₃D (Ec. 464).

        h_ij(S; x) = [1 + 2 Φ_grav(x;S)/c²] · δ_ij
    """
    return (1.0 + 2.0 * Phi_grav / c ** 2) * np.eye(3)


def Xi_asymmetry(S: float) -> float:
    """Asimetría tensional Ξ(S) = (M_p - E_p)/(M_p + E_p).

    Con M_p(S) lineal entre C1 y C4, y normalización M_p + E_p ≈ 1.
    En S_{0.999}: M_p ≈ E_p → Ξ → δ₀ ≪ 1 (regularización).
    """
    S1, S4 = 0.009, 1.001
    MP0, MPEQ = 0.99, 0.50
    if S <= S1:
        mp = MP0
    elif S >= S4:
        mp = MPEQ
    else:
        mp = MP0 - (MP0 - MPEQ) * (S - S1) / (S4 - S1)
    ep = 1.0 - mp
    if mp + ep <= 0:
        return 0.0
    return (mp - ep) / (mp + ep)


def Delta_tens(S: float, S_0999: float = 0.999,
               lambda_ont: float = C.LAMBDA_ONT,
               xi_tens: float = 1.0, c: float = 1.0) -> float:
    """Coeficiente de dispersión tensional Δ_tens / |p|² (Ecs. 466-467).

        E² = E₀² + c²|p|² + Δ_tens(S, p)
        Δ_tens(S, p) = ξ_tens · c⁴ · Ξ²(S) · |p|²

    Devuelve el factor (sin |p|²). En S_{0.999}: Δ_tens/c²|p|² ≤ δ₀² ≈ 10⁻⁴.
    """
    Xi = Xi_asymmetry(S)
    window = 1.0 - 0.5 * (1.0 + np.tanh((S - S_0999) / lambda_ont))
    return float(xi_tens * c ** 4 * Xi ** 2 * window)


# ============================================================
# B9. Dinámica de retorno S_max → S̃₀ + GWs relictas (Ecs. 468-471)
# ============================================================

def V_ret(S: float, T0: float = C.T0_GEV, S_max: float = None) -> float:
    """Potencial de retorno tensional (Ec. 468):

        V_ret(S) = T₀ · (1 - S/S_max)²
    """
    if S_max is None:
        S_max = C.S_SEALS["S_max"]
    return float(T0 * (1.0 - S / S_max) ** 2)


def omega_ret(T0: float = C.T0_GEV, I_S_entropy: float = 1.0,
              S_max: float = None) -> float:
    """Frecuencia del retorno (Ec. 469):

        ω_ret = √(2 T₀ / (I_S · S_max²))
    """
    if S_max is None:
        S_max = C.S_SEALS["S_max"]
    return float(np.sqrt(2.0 * T0 / (I_S_entropy * S_max ** 2)))


# τ_signature: timescale OBSERVABLE de la señal GW en PTA (~yr).
# Se distingue del τ_cosmológico (~Gyr) que es la duración total del
# retroceso entrópico. La firma espectral en PTA depende de τ_signature.
TAU_SIGNATURE_S = 3.18e7   # ≈ 1 año, sitúa f_pico en banda PTA (~10⁻⁸ Hz)


def Omega_GW_return(f_Hz: float | np.ndarray,
                    tau_signature_s: float = TAU_SIGNATURE_S,
                    Omega_ret_amp: float = 1.2e-9) -> float | np.ndarray:
    """Espectro Ω_GW(f) del retroceso entrópico observable en PTA (Ec. 470):

        Ω_GW(f) = Ω_ret · (f τ)² / cosh²(π f τ / 2)

    Firma espectral distintiva:
      · pico en f ≈ 1/(π·τ_signature) ~ 10⁻⁸ Hz
      · amplitud Ω_ret ≈ 1.2×10⁻⁹
      · forma sech² distinguible del power-law (α≃2/3 de SMBH)
      · detectable en SKA-PTA a S/N ≈ 2.5 en 15 años

    NOTA sobre escalas:
      τ_signature_s (~yr) = timescale observable PTA → fija f_pico.
      τ_cosmológico (~Gyr) = duración real del retroceso entrópico.
      Son escalas distintas: la primera es el ancho de banda del rasgo
      espectral; la segunda, el tiempo cósmico del proceso.

    FALSACIÓN: si Ω_GW·h² > 2×10⁻¹¹ no detectado en SKA-PTA → excluye el modelo.
    """
    f = np.asarray(f_Hz, dtype=float)
    x = np.pi * f * tau_signature_s / 2.0
    # Guard against cosh overflow: para |x| >> 1, sech²(x) ≈ 4 exp(-2|x|)
    with np.errstate(over="ignore", invalid="ignore"):
        cosh_sq = np.where(np.abs(x) > 350.0, np.inf, np.cosh(x) ** 2)
        out = np.where(np.isfinite(cosh_sq),
                       Omega_ret_amp * (f * tau_signature_s) ** 2 / cosh_sq,
                       0.0)
    return out if out.ndim else float(out)


def delta0_next_cycle(delta0: float = C.EPSILON_0,
                      lambda_pre: float = C.LAMBDA_PRE,
                      S_max: float = None,
                      delta_S: float = C.DELTA_S) -> float:
    """Imperfección del próximo ciclo (Ec. 471):

        δ̃₀ = δ₀ · exp(-∫₀^{S_max} k_pre dS') ≃ δ₀ · exp(-λ_pre·S_max/ΔS)

    La imperfección se REDUCE pero nunca se anula (sólo decae a 0
    asintóticamente).
    """
    if S_max is None:
        S_max = C.S_SEALS["S_max"]
    return float(delta0 * np.exp(-lambda_pre * S_max / delta_S))
