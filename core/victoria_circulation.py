"""La Circulación de Victoria: obstrucción de gradiente y circulación mínima (frente 2).

La mejora conceptual propuesta para la v36 (dos niveles del Camino)
convierte la tensión entre el Axioma 4 y el frente 2 en una estructura
medible. El Camino fundamental es descenso puro (dΦ/dσ = −G⁻¹ δV/δΦ,
monótono); el Camino EFECTIVO de los acoplos, tras coarse-graining, se
estudia en la clase declarada

    β(λ) = (−G_λ⁻¹ + J_λ)·∇C(λ),   G_λ ≻ 0,  J_λᵀ = −J_λ,

que conserva la monotonía (dC/dlnS = −∇Cᵀ G⁻¹ ∇C ≤ 0, porque
xᵀJx = 0) y sin embargo permite espectro complejo en el jacobiano
M = (−G⁻¹ + J)·H con H = Hess(C). El descenso no exige línea recta.

LO QUE ESTE MÓDULO HACE (subfrentes F2A–F2D):

F2A. TEOREMA DE OBSTRUCCIÓN (control negativo del frente): si J = 0,
     M = −G⁻¹H con G ≻ 0 y H simétrica es similar a −G^{-1/2}HG^{-1/2}
     (simétrica) ⟹ espectro REAL ⟹ s0 = 0: un flujo de gradiente puro
     no puede producir la DSI. Dentro de la clase declarada,
     s0 ≠ 0 ⟹ J ≠ 0.

F2B. CIRCULACIÓN MÍNIMA: en la familia M(α) = (−G⁻¹ + α·J₀)H se miden
     el umbral de complejificación α_c y α_Victoria ≡ min{α : s0(α) =
     π/ln 10}. En 2D hay forma cerrada — α_c = |h1−h2|/(2√(h1h2)),
     α_V = √((4s0² + (h1−h2)²)/(4h1h2)) — y la medición numérica debe
     clavarla (comprobación interna).

F2C. GEOMETRÍA DE LA ESPINODAL: n = ∇D/|∇D| en el cuello, P_T = I−nnᵀ,
     M_T = P_T·M·P_T — separa la circulación tangencial al cuello de la
     que cruza el discriminante.

F2D. TERCER ESTIMADOR INDEPENDIENTE: s0 medido sobre la SEÑAL de la
     trayectoria (cruces por cero interpolados de la proyección
     renormalizada), sin consultar el espectro de M. Tercera ruta junto
     a la espectral y la dinámica de victoria_exponent.

LO QUE NO HACE: derivar J_λ. Las β reales (jerarquía de Fokker-Planck,
Def. 4.4) siguen siendo el hueco del frente 2; α_Victoria es un MAPA
sobre (G, H, J₀) declarados, no un número de la naturaleza. Y el
alcance del teorema es la clase declarada: fuentes de no-gradiente
fuera de ella (G no simétrica o dependiente de λ) no quedan cubiertas.
"""

from __future__ import annotations

import numpy as np

from .victoria_exponent import S0_TARGET

# Cuello espinodal de referencia: D = B² − 4·C0·M0² = 4 − 4 = 0 con
# acoplos positivos — el punto donde la ec. 14.2 pide evaluar M.
SPINODAL_POINT = np.array([1.0, 2.0, 1.0])  # (M0², B, C0)

STATUS_CIRCULATION = (
    "condicional (frente 2, propuesta v36 «dos niveles del Camino»): "
    "el teorema de obstrucción es exacto DENTRO de la clase declarada "
    "β = (−G⁻¹+J)∇C con G ≻ 0 — ahí, s0 ≠ 0 exige J ≠ 0 —; α_Victoria "
    "es un mapa sobre (G, H, J₀) declarados, no una constante: las β "
    "reales (Fokker-Planck, Def. 4.4) siguen pendientes y este módulo "
    "no demuestra que la naturaleza los elija"
)


# ---------- F2A: obstrucción de gradiente ----------

def gradient_flow_matrix(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    """M = −G⁻¹H — el jacobiano del gradiente puro (J = 0)."""
    return -np.linalg.solve(np.asarray(G, float), np.asarray(H, float))


def circulation_flow_matrix(G: np.ndarray, H: np.ndarray, J0: np.ndarray,
                            alpha: float) -> np.ndarray:
    """M(α) = (−G⁻¹ + α·J₀)·H — gradiente más circulación declarada."""
    G = np.asarray(G, float); H = np.asarray(H, float)
    return (-np.linalg.inv(G) + alpha * np.asarray(J0, float)) @ H


def max_im_spectrum(M: np.ndarray) -> float:
    """s0 del espectro: max |Im μ(M)| (0.0 si es real)."""
    return float(np.abs(np.linalg.eigvals(np.asarray(M, float)).imag).max())


def obstruction_holds(G: np.ndarray, H: np.ndarray,
                      tol: float = 1e-9) -> bool:
    """F2A: el espectro de −G⁻¹H es real (G ≻ 0, H = Hᵀ)."""
    return max_im_spectrum(gradient_flow_matrix(G, H)) < tol


def J_plane(dim: int, i: int = 0, j: int = 1) -> np.ndarray:
    """Generador antisimétrico unitario del plano (i, j)."""
    J = np.zeros((dim, dim))
    J[i, j], J[j, i] = -1.0, 1.0
    return J


# ---------- F2B: circulación mínima ----------

def alpha_c_closed_form_2d(h1: float, h2: float) -> float:
    """Umbral de complejificación en 2D (G = I, J₀ = plano):
    α_c = |h1 − h2| / (2·√(h1·h2))."""
    return abs(h1 - h2) / (2.0 * np.sqrt(h1 * h2))


def alpha_victoria_closed_form_2d(h1: float, h2: float,
                                  s0: float = S0_TARGET) -> float:
    """α_V en 2D: s0(α)² = α²h1h2 − (h1−h2)²/4 ⟹
    α_V = √((4s0² + (h1−h2)²)/(4h1h2))."""
    return float(np.sqrt((4.0 * s0 ** 2 + (h1 - h2) ** 2)
                         / (4.0 * h1 * h2)))


def _s0_of_alpha(G, H, J0, alpha: float) -> float:
    return max_im_spectrum(circulation_flow_matrix(G, H, J0, alpha))


def alpha_threshold(G: np.ndarray, H: np.ndarray, J0: np.ndarray,
                    alpha_max: float = 50.0, tol: float = 1e-10) -> float:
    """α_c: la circulación mínima que complejifica el espectro
    (bisección sobre el primer cruce s0 > 0)."""
    lo, hi = 0.0, float(alpha_max)
    if _s0_of_alpha(G, H, J0, hi) <= tol:
        raise ValueError("Sin complejificación hasta alpha_max")
    while hi - lo > 1e-12 * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if _s0_of_alpha(G, H, J0, mid) > tol:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def alpha_victoria(G: np.ndarray, H: np.ndarray, J0: np.ndarray,
                   s0_target: float = S0_TARGET,
                   alpha_max: float = 50.0) -> float:
    """α_Victoria: min{α : s0(α) = s0_target} — cuánta circulación hace
    falta, en la familia declarada, para el exponente de la Década."""
    a_c = alpha_threshold(G, H, J0, alpha_max)
    lo, hi = a_c, float(alpha_max)
    if _s0_of_alpha(G, H, J0, hi) < s0_target:
        raise ValueError("s0_target inalcanzable hasta alpha_max")
    while hi - lo > 1e-12 * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if _s0_of_alpha(G, H, J0, mid) >= s0_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------- F2C: geometría de la espinodal ----------

def spinodal_normal(lam: np.ndarray = SPINODAL_POINT) -> np.ndarray:
    """n = ∇D/|∇D| en el cuello, con D = B² − 4·C0·M0² y
    λ = (M0², B, C0): ∇D = (−4C0, 2B, −4M0²)."""
    m0_sq, b, c0 = np.asarray(lam, float)
    grad = np.array([-4.0 * c0, 2.0 * b, -4.0 * m0_sq])
    return grad / np.linalg.norm(grad)


def tangential_projector(n: np.ndarray) -> np.ndarray:
    """P_T = I − n·nᵀ (proyector sobre el tangente a la espinodal)."""
    n = np.asarray(n, float)
    return np.eye(n.size) - np.outer(n, n)


def tangential_part(M: np.ndarray,
                    n: np.ndarray | None = None) -> np.ndarray:
    """M_T = P_T·M·P_T — la dinámica tangencial al cuello: separa
    evolucionar A LO LARGO de la espinodal de cruzarla."""
    if n is None:
        n = spinodal_normal()
    P = tangential_projector(n)
    return P @ np.asarray(M, float) @ P


# ---------- F2D: tercer estimador (señal, sin espectro de M) ----------

def s0_signal(M: np.ndarray, t_max: float = 500.0, dt: float = 5e-3,
              seed: int = 0, transient: float = 0.2) -> float:
    """s0 medido sobre la SEÑAL: se integra dδ/dt = M·δ (RK4), se
    renormaliza δ (quita la envolvente), se proyecta sobre una
    dirección fija y se mide el semiperiodo medio entre cruces por
    cero interpolados: s0 = π/⟨semiperiodo⟩. No consulta el espectro
    de M. Mide el par DOMINANTE; si la señal no oscila (gradiente
    puro), devuelve 0.0."""
    M = np.asarray(M, float)
    rng = np.random.default_rng(seed)
    delta = rng.standard_normal(M.shape[0])
    delta /= np.linalg.norm(delta)
    c = rng.standard_normal(M.shape[0])
    c /= np.linalg.norm(c)
    n_steps = int(t_max / dt)
    xs = np.empty(n_steps + 1)
    xs[0] = c @ delta
    for k in range(n_steps):
        k1 = M @ delta
        k2 = M @ (delta + 0.5 * dt * k1)
        k3 = M @ (delta + 0.5 * dt * k2)
        k4 = M @ (delta + dt * k3)
        delta = delta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        delta = delta / np.linalg.norm(delta)      # sin envolvente
        xs[k + 1] = c @ delta
    i0 = int(transient * n_steps)
    x = xs[i0:] - xs[i0:].mean()
    s = np.sign(x)
    idx = np.nonzero(s[:-1] * s[1:] < 0.0)[0]
    if idx.size < 2:
        return 0.0
    t = np.arange(x.size) * dt
    crossings = t[idx] - x[idx] * dt / (x[idx + 1] - x[idx])
    half = float(np.diff(crossings).mean())
    return float(np.pi / half)
