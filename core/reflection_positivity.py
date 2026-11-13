"""Positividad por reflexión en modelo de juguete (v35, Cap. 7).

LO QUE ESTE MÓDULO ES — Y LO QUE NO. El Teorema de Positividad de
Florencia (7.1) establece ⟨ϑ(F̄)·F⟩ ≥ 0 bajo hipótesis (H1)-(H4) para la
teoría euclidiana en la loncha de Florencia, y con ello la
reconstrucción de Osterwalder-Schrader (7.3): un espacio de Hilbert de
norma positiva, Ĥ ≥ 0 y evolución unitaria — el tiempo nace unitario.
Este módulo verifica la CONCLUSIÓN en un retículo escalar 1D pequeño
con la medida e^{−S_E}: es una comprobación en modelo de juguete, NO la
demostración del teorema. El sector espinorial de Wilson (Teo. 7.4)
queda como lo que el tratado declara: CONDICIONAL (§13.4).

La verificación operativa: para un retículo con reflexión ϑ respecto de
una loncha, ⟨ϑ(F̄)·F⟩ = fᵀ·K·f con K el núcleo de transferencia

    K(φ, φ') = exp(−(φ−φ')²/2 − V(φ)/2 − V(φ')/2)
             = exp(−φ²/2 − φ'²/2 − V/2 − V'/2) · exp(J·φ·φ'),  J = +1

RP ⟺ K semidefinido positivo. El acoplo J = +1 procede del término
cinético (H1) — y el CONTROL NEGATIVO es exactamente ese: un acoplo
temporal invertido (J < 0) rompe la positividad (autovalores negativos),
que es lo que distingue una auditoría de una lista de éxitos.

El papel del término séxtico (§3.1): C0 > 0 hace la medida
normalizable (V → +∞); con C0 < 0 el paisaje se desfonda y la medida
e^{−V} no existe — segunda condición comprobada.

LA POSITIVIDAD DEL PROPIO CAMINO CON DIFUSIÓN (revisión del 10-nov-2026,
hallazgo 3; comprobación interna, E8). El núcleo de arriba es el de una
acción de SEGUNDO orden en S con V0 como potencial de sitio (ec. 7.1). El
Flujo del Camino es de PRIMER orden con difusión (Def. 4.4): dΦ = −G⁻¹∇V dσ
+ √(2Θ/G) dW. El puente entre los dos es exacto: la medida de trayectorias
del Camino es exp(−A) con la acción de Onsager–Machlup,

    A = ∫ dσ [ (G/4Θ)·Φ̇² + U_FP(Φ) ],   U_FP = |∇V|²/(4ΘG) − ∇²V/(2G),

y su generador de Fokker–Planck, simetrizado con e^{−V/2Θ}, es

    H_FP = −(Θ/G)·∇² + U_FP = (Θ/G)·a†a,   a = ∇ + ∇V/(2Θ),

que es ≥ 0 por construcción y tiene el modo cero e^{−V/2Θ}. El núcleo de
transferencia exp(−a·H_FP) es simétrico y semidefinido positivo: la
positividad por reflexión sale de la reversibilidad (balance detallado) del
propio Camino, sin el término cinético de (7.1). Las funciones `fp_*` y
`camino_fp_report` lo comprueban en el corte del polo de masa, en una
dimensión. Tres cosas cambian respecto del núcleo de arriba, y se publican:
el potencial de sitio no es V0 sino U_FP (vale −V″/2G en cada punto crítico
de V0, también en la cima de la barrera); la brecha del espectro es una
TASA (la de Kramers, ∝ δ0²) y no una frecuencia (∝ δ0); y entra una
constante, Θ, que es la difusión D_ent que la nucleación ya declara y que
nada fija. CONTROL NEGATIVO: con el balance detallado roto (una fuerza no
conservativa sobre un corte periódico) la simetrización falla. No es el
Teorema 7.1 ni lo sustituye: sobre qué acción se enuncia es la decisión 3
del autor.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal, expm

from .basal import B_BAR, C0_DEFAULT, E_BAR, M_BAR, scaled_params

STATUS_WILSON = ("condicional (§13.4): la RP del sector espinorial de "
                 "Wilson (Prop. 7.4 — reclasificada de Teorema en "
                 "FE-08-2026, E3) no se toca aquí — frente abierto nº 1 "
                 "para acoplos no estacionarios")


def transfer_kernel(phi_grid: np.ndarray, V_values: np.ndarray,
                    J: float = 1.0) -> np.ndarray:
    """Núcleo de transferencia del retículo escalar 1D.

    K(φ,φ') = exp(−φ²/2 − φ'²/2 + J·φφ' − V(φ)/2 − V(φ')/2).
    Con J = +1 equivale a exp(−(φ−φ')²/2)·e^{−V/2}·e^{−V'/2} (el término
    cinético físico). J < 0 es el control negativo (acoplo invertido).
    """
    phi = np.asarray(phi_grid, dtype=float)
    V = np.asarray(V_values, dtype=float)
    diag = np.exp(-0.5 * phi ** 2 - 0.5 * V)
    cross = np.exp(J * np.outer(phi, phi))
    return diag[:, None] * cross * diag[None, :]


def rp_min_eigenvalue(phi_grid: np.ndarray, V_values: np.ndarray,
                      J: float = 1.0) -> float:
    """Autovalor mínimo (normalizado) del núcleo simétrico.

    RP ⟺ ⟨ϑ(F̄)F⟩ = fᵀKf ≥ 0 para todo funcional f soportado en un
    lado de la loncha ⟺ min eig(K) ≥ 0.
    """
    K = transfer_kernel(phi_grid, V_values, J=J)
    eigs = np.linalg.eigvalsh(0.5 * (K + K.T))
    return float(eigs.min() / max(abs(eigs).max(), 1e-300))


def rp_holds(phi_grid: np.ndarray, V_values: np.ndarray,
             J: float = 1.0, tol: float = 1e-10) -> bool:
    """⟨ϑ(F̄)·F⟩ ≥ 0 sobre toda la familia de funcionales del grid."""
    return rp_min_eigenvalue(phi_grid, V_values, J=J) >= -tol


def measure_normalizable(delta0: float, C0: float,
                         rho_max: float = 50.0) -> bool:
    """¿Existe la medida e^{−V0}? Requiere V0 → +∞ (C0 > 0, §3.1).

    Comprobación operativa: V0 creciente y positivo en el borde lejano.
    """
    from .basal import V0
    v_far = float(V0(rho_max, 0.0, delta0, C0=C0))
    v_farther = float(V0(2.0 * rho_max, 0.0, delta0, C0=C0))
    return v_far > 0.0 and v_farther > v_far


# ---------------------------------------------------------------------------
# La positividad del propio Camino con difusión (Onsager–Machlup; revisión del 10-nov-2026)
# ---------------------------------------------------------------------------

STATUS_CAMINO_FP = ("comprobación interna (E8) en el corte del polo de masa, una dimensión: el generador de Fokker–Planck "
                    "del Camino con difusión, simetrizado, es ≥ 0 con modo cero e^{−V/2Θ} y su núcleo de transferencia es "
                    "semidefinido positivo; Θ DECLARADA (la D_ent de la nucleación, que nada fija); no es el Teorema 7.1 ni "
                    "lo sustituye — sobre qué acción se enuncia es la decisión 3 del autor")


def pole_cut(delta0: float, n: int = 2000, rho_max_over_rho_tv: float = 1.2, m_bar: float = M_BAR,
             b_bar: float = B_BAR, e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> dict:
    """El corte del polo de masa (θ = 0, φ_E = 0) del Potencial Basal COMPLETO en una malla uniforme de n puntos entre
    ρ = 0 (la frontera del dominio físico) y `rho_max_over_rho_tv`·ρ_vv:

        V(ρ) = ½M0²ρ² − (B/4)ρ⁴ + (C0/6)ρ⁶ − η·ρ/√2,

    con V′ y V″ analíticas y los puntos críticos de `core.s_clock.radial_landscape`."""
    from .s_clock import radial_landscape
    ld = radial_landscape(delta0, 0.0, m_bar, b_bar, e_bar, C0)
    if not ld["metastable"]:
        raise ValueError("Sin falso vacío metastable en el corte del polo: no hay barrera que cruzar")
    if n < 8 or rho_max_over_rho_tv <= 1.0:
        raise ValueError("n ≥ 8 y la malla debe llegar más allá del vacío verdadero")
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    tilt = p["eta"] / np.sqrt(2.0)
    rho = np.linspace(0.0, rho_max_over_rho_tv * ld["rho_tv"], n)
    return {"rho": rho, "h": float(rho[1] - rho[0]),
            "V": 0.5 * p["M0_sq"] * rho ** 2 - 0.25 * p["B"] * rho ** 4 + (C0 / 6.0) * rho ** 6 - tilt * rho,
            "dV": p["M0_sq"] * rho - p["B"] * rho ** 3 + C0 * rho ** 5 - tilt,
            "d2V": p["M0_sq"] - 3.0 * p["B"] * rho ** 2 + 5.0 * C0 * rho ** 4,
            "d2V_of": lambda r: p["M0_sq"] - 3.0 * p["B"] * r ** 2 + 5.0 * C0 * r ** 4,
            "V_of": lambda r: 0.5 * p["M0_sq"] * r ** 2 - 0.25 * p["B"] * r ** 4 + (C0 / 6.0) * r ** 6 - tilt * r,
            "landscape": ld, "delta0": delta0}


def fp_factor(V_values: np.ndarray, h: float, Theta: float, G: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """El factor bidiagonal B del generador simetrizado, H_FP = BᵀB (la versión en malla de H_FP = (Θ/G)·a†a con a = ∂ +
    V′/2Θ). Fila i de B: √c·(e^{−Δ_i/2}, −e^{+Δ_i/2}) sobre los sitios (i, i+1), con c = Θ/(G·h²) y Δ_i = (V_{i+1} −
    V_i)/(2Θ). B tiene n − 1 filas y n columnas: H_FP ≥ 0 y tiene un autovalor nulo POR CONSTRUCCIÓN, con B·e^{−V/2Θ} = 0.
    Devuelve (diagonal, superdiagonal) de B."""
    V = np.asarray(V_values, dtype=float)
    if Theta <= 0.0 or G <= 0.0 or h <= 0.0:
        raise ValueError("Θ > 0, G > 0 y h > 0")
    sc = np.sqrt(Theta / (G * h * h))
    d = (V[1:] - V[:-1]) / (2.0 * Theta)
    return sc * np.exp(-0.5 * d), -sc * np.exp(0.5 * d)


def fp_symmetrized_generator(V_values: np.ndarray, h: float, Theta: float, G: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """El generador de Fokker–Planck del Camino con difusión, simetrizado: H_FP = −e^{V/2Θ}·L·e^{−V/2Θ}, con L la
    ecuación maestra de tasas W(i → i±1) = c·exp(−(V_{i±1} − V_i)/2Θ), c = Θ/(G·h²), que cumple el balance detallado
    con e^{−V/Θ} exactamente y tiene fronteras reflectantes (flujo nulo). Es tridiagonal y simétrico: diagonal
    c·(e^{−Δ_i} + e^{+Δ_{i−1}}) y fuera de la diagonal −c. En el límite continuo, −(Θ/G)∂² + U_FP. Devuelve (diagonal,
    fuera de la diagonal)."""
    b_diag, b_up = fp_factor(V_values, h, Theta, G)
    diag = np.zeros(len(b_diag) + 1)
    diag[:-1] += b_diag ** 2
    diag[1:] += b_up ** 2
    return diag, b_diag * b_up


def fp_zero_mode(V_values: np.ndarray, Theta: float) -> np.ndarray:
    """El modo cero de H_FP, e^{−V/2Θ}, normalizado: la raíz de la medida de equilibrio del Camino, e^{−V/Θ}."""
    V = np.asarray(V_values, dtype=float)
    psi = np.exp(-(V - V.min()) / (2.0 * Theta))
    return psi / np.linalg.norm(psi)


def fp_site_potential(dV: np.ndarray, d2V: np.ndarray, Theta: float, G: float = 1.0) -> np.ndarray:
    """El potencial de sitio de la acción de Onsager–Machlup, U_FP = V′²/(4ΘG) − V″/(2G). En cada punto crítico de V
    vale −V″/(2G): tiene un pozo también en la cima de la barrera (si Θ es pequeña frente a ella)."""
    return np.asarray(dV, dtype=float) ** 2 / (4.0 * Theta * G) - np.asarray(d2V, dtype=float) / (2.0 * G)


def fp_rates(V_values: np.ndarray, h: float, Theta: float, G: float = 1.0, force: float = 0.0,
             periodic: bool = False) -> np.ndarray:
    """La ecuación maestra dP/dσ = L·P en forma densa, con una fuerza constante `force` añadida a −V′ y, si
    `periodic`, el enlace que cierra el anillo. Con fronteras reflectantes toda fuerza es un gradiente y el balance
    detallado se mantiene; sobre un anillo, `force ≠ 0` es NO conservativa (hay corriente estacionaria) y lo rompe:
    es el control negativo, no un corte del Plano Dual."""
    V = np.asarray(V_values, dtype=float)
    n = len(V)
    c = Theta / (G * h * h)
    L = np.zeros((n, n))
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic else [])
    for i, j in pairs:
        up = c * np.exp(-(V[j] - V[i] - force * h) / (2.0 * Theta))      # tasa i → j (a favor de la fuerza)
        dn = c * np.exp(-(V[i] - V[j] + force * h) / (2.0 * Theta))      # tasa j → i
        L[j, i] += up
        L[i, i] -= up
        L[i, j] += dn
        L[j, j] -= dn
    return L


def fp_symmetrize(L: np.ndarray, V_values: np.ndarray, Theta: float) -> np.ndarray:
    """−e^{V/2Θ}·L·e^{−V/2Θ}: simétrico si y solo si L cumple el balance detallado con e^{−V/Θ}."""
    w = np.exp((np.asarray(V_values, dtype=float) - np.min(V_values)) / (2.0 * Theta))
    return -(w[:, None] * np.asarray(L, dtype=float)) / w[None, :]


def detailed_balance_control(n: int = 240, amplitude_over_Theta: float = 2.0, affinity: float = 1.0) -> dict:
    """CONTROL NEGATIVO de la simetrización. Un corte periódico (anillo de longitud 1) con V = V_a·cos(2πx), Θ =
    V_a/`amplitude_over_Theta` y una fuerza constante f con afinidad de ciclo f·L/Θ = `affinity`. Sin fuerza el
    generador simetrizado es simétrico, con espectro real ≥ 0 y modo cero e^{−V/2Θ}; con fuerza no conservativa el
    criterio de Kolmogórov falla (el producto de tasas a lo largo del anillo no es el del sentido contrario), la
    matriz deja de ser simétrica y su espectro se hace complejo: sin balance detallado no hay H_FP simétrico ni
    núcleo de transferencia simétrico."""
    x = np.arange(n) / n
    h = 1.0 / n
    V = np.cos(2.0 * np.pi * x)
    Theta = 1.0 / amplitude_over_Theta
    out = {"n": n, "Theta": Theta, "affinity": affinity}
    for key, force in (("reversible", 0.0), ("broken", affinity * Theta)):
        L = fp_rates(V, h, Theta, force=force, periodic=True)
        H = fp_symmetrize(L, V, Theta)
        eig = np.linalg.eigvals(H)
        psi = fp_zero_mode(V, Theta)
        fwd = np.array([L[(i + 1) % n, i] for i in range(n)])
        bwd = np.array([L[i, (i + 1) % n] for i in range(n)])
        gap = float(np.sort(eig.real)[1])
        K = expm(-H / gap)                                               # núcleo a una distancia 1/brecha
        off_norm = np.linalg.norm(H - np.diag(np.diag(H)))               # la asimetría vive fuera de la diagonal
        out[key] = {"force": force, "asymmetry": float(np.linalg.norm(H - H.T) / off_norm),
                    "kernel_asymmetry": float(np.linalg.norm(K - K.T) / np.linalg.norm(K)),
                    "max_abs_imag_eigenvalue": float(np.max(np.abs(eig.imag))), "min_real_eigenvalue": float(eig.real.min()),
                    "gap": gap,
                    "zero_mode_residual": float(np.linalg.norm(H @ psi) / np.linalg.norm(H, ord=np.inf)),
                    "kolmogorov_cycle_affinity": float(np.sum(np.log(fwd / bwd))),
                    "symmetric": bool(np.linalg.norm(H - H.T) <= 1e-12 * off_norm)}
    return out


def kramers_rate_integral(cut: dict, Theta: float, G: float = 1.0) -> float:
    """La tasa de escape de Kramers sobreamortiguada en su forma integral, sin la aproximación gaussiana:

        Γ = (Θ/G) / [ ∫₀^{ρ_b} e^{−V/Θ} dρ · ∫_{ρ_fv}^{ρ_vv} e^{+V/Θ} dρ ],

    que cuenta la pared reflectante de ρ = 0 junto al falso vacío. Con el pozo y la barrera gaussianos se reduce a
    √(V″_fv·|V″_b|)/(2πG)·e^{−ΔV_b/Θ} (`core.nucleation.kramers_escape`)."""
    ld, V = cut["landscape"], cut["V_of"]
    well = quad(lambda r: np.exp(-(V(r) - ld["V_fv"]) / Theta), 0.0, ld["rho_barrier"], epsabs=0.0, epsrel=1e-11)[0]
    top = quad(lambda r: np.exp((V(r) - ld["V_barrier"]) / Theta), ld["rho_fv"], ld["rho_tv"], epsabs=0.0, epsrel=1e-11,
               points=[ld["rho_barrier"]])[0]
    return float(Theta / (G * well * top) * np.exp(-ld["barrier_height"] / Theta))


def camino_transfer_kernel(V_values: np.ndarray, h: float, Theta: float, a: float, G: float = 1.0) -> np.ndarray:
    """El núcleo de transferencia del Camino con difusión entre dos lonchas separadas a en σ: K = exp(−a·H_FP),
    calculado por la exponencial de matrices (no por los autovalores de H_FP)."""
    diag, off = fp_symmetrized_generator(V_values, h, Theta, G)
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return expm(-a * H)


def camino_fp_report(delta0: float, barrier_over_Theta: float = 6.0, G: float = 1.0, n: int = 2000, n_kernel: int = 300,
                     m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> dict:
    """La comprobación completa en el corte del polo, con Θ = ΔV_b/`barrier_over_Theta` (Θ DECLARADA: la D_ent de la
    nucleación). Publica, en unidades de δ0²/G (σ̂ = δ0²·σ): el autovalor mínimo de H_FP y el residuo relativo del modo
    cero e^{−V/2Θ}; la brecha frente a la tasa de Kramers (gaussiana, `core.nucleation.kramers_escape`, e integral) y el
    nivel siguiente frente a V″ del falso vacío; U_FP en los tres puntos críticos y sus pozos; y el autovalor mínimo del
    núcleo de transferencia exp(−a·H_FP) con a = G/V″_fv (una relajación del pozo del falso vacío)."""
    from .nucleation import kramers_escape
    cut = pole_cut(delta0, n, 1.2, m_bar, b_bar, e_bar, C0)
    ld = cut["landscape"]
    Theta = ld["barrier_height"] / barrier_over_Theta
    unit = delta0 ** 2 / G
    diag, off = fp_symmetrized_generator(cut["V"], cut["h"], Theta, G)
    lam = eigh_tridiagonal(diag, off, select="i", select_range=(0, 3), eigvals_only=True)
    psi0 = fp_zero_mode(cut["V"], Theta)
    res = diag * psi0
    res[:-1] += off * psi0[1:]
    res[1:] += off * psi0[:-1]
    norm_H = float(np.max(np.abs(diag) + 2.0 * np.max(np.abs(off))))
    b_diag, b_up = fp_factor(cut["V"], cut["h"], Theta, G)
    kr = kramers_escape(delta0, Theta, 0.0, G, m_bar, b_bar, e_bar, C0)
    gamma_int = kramers_rate_integral(cut, Theta, G)
    crit = {"false_vacuum": ld["rho_fv"], "barrier": ld["rho_barrier"], "true_vacuum": ld["rho_tv"]}
    U = fp_site_potential(cut["dV"], cut["d2V"], Theta, G)
    wells = cut["rho"][np.where((U[1:-1] < U[:-2]) & (U[1:-1] < U[2:]))[0] + 1]
    # el generador en malla frente a su límite continuo −(Θ/G)∂² + U_FP, entre la frontera y el vacío verdadero
    inner = (cut["rho"] > 0.0) & (cut["rho"] <= ld["rho_tv"])
    U_discrete = diag - 2.0 * Theta / (G * cut["h"] ** 2)
    # núcleo de transferencia en una malla más gruesa (la exponencial de matrices es densa)
    small = pole_cut(delta0, n_kernel, 1.2, m_bar, b_bar, e_bar, C0)
    a = G / kr["V_pp_fv"]
    K = camino_transfer_kernel(small["V"], small["h"], Theta, a, G)
    k_eigs = np.linalg.eigvalsh(0.5 * (K + K.T))
    return {"delta0": delta0, "Theta": float(Theta), "barrier_over_Theta": barrier_over_Theta, "G": G, "n": n,
            "unit": "δ0²/G", "eigenvalues_scaled": (lam / unit).tolist(), "lambda_min_scaled": float(lam[0] / unit),
            "zero_mode_residual_relative": float(np.linalg.norm(res) / norm_H),
            "factor_zero_mode_residual_relative": float(np.linalg.norm(b_diag * psi0[:-1] + b_up * psi0[1:])
                                                        / np.max(np.abs(b_up))),
            "gap_scaled": float(lam[1] / unit), "kramers_gaussian_scaled": float(kr["Gamma_K"] / unit),
            "kramers_integral_scaled": float(gamma_int / unit), "gap_over_kramers_gaussian": float(lam[1] / kr["Gamma_K"]),
            "gap_over_kramers_integral": float(lam[1] / gamma_int), "next_level_scaled": float(lam[2] / unit),
            "V_pp_false_vacuum_scaled": float(kr["V_pp_fv"] / delta0 ** 2),
            "false_vacuum_over_thermal_width": float(ld["rho_fv"] / np.sqrt(Theta / kr["V_pp_fv"])),
            "U_FP_at_critical_points_scaled": {k: float(-cut["d2V_of"](r) / (2.0 * G) / unit) for k, r in crit.items()},
            "U_FP_wells_rho": wells.tolist(), "critical_points_rho": crit,
            "U_FP_grid_vs_continuum_relative": float(np.max(np.abs(U_discrete[inner] - U[inner])) / np.max(np.abs(U[inner]))),
            "kernel": {"a": float(a), "n": n_kernel, "min_eigenvalue_over_max": float(k_eigs.min() / k_eigs.max()),
                       "asymmetry": float(np.linalg.norm(K - K.T) / np.linalg.norm(K)),
                       "positive_semidefinite": bool(k_eigs.min() >= -1e-12 * k_eigs.max())},
            "status": STATUS_CAMINO_FP}
