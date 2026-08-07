"""RP con acoplos no estacionarios — el juguete del frente nº 1 (v35, §13.4).

La Prop. 7.4 (reclasificada de Teorema en v35.1, E3) deja CONDICIONAL
la positividad por reflexión del sector
espinorial de Wilson con acoplos no estacionarios. Este módulo NO toca
ese sector: construye el análogo escalar exacto de la pregunta — una
cadena 1D cuyos acoplos (masa por sitio, enlace J) DEPENDEN de la
posición, como los acoplos del tramo euclidiano dependen de S — y mide
la positividad del producto reflejado ⟨ϑ(F̄)·G⟩ para funcionales de la
loncha ±1, contrayendo la cadena entera (matrices de transferencia).

LO QUE EL JUGUETE ESTABLECE:

1. Perfil ESPECULAR respecto de la loncha ⟹ RP exacta, aunque los
   acoplos varíen sitio a sitio: M = diag(v)·LᵀW₀L·diag(v) es
   semidefinida positiva por construcción (forma XᵀX). La
   no-estacionariedad NO rompe la RP por sí sola.

2. RUNNING MONÓTONO (m²(t) = 1 + g·t o J(t) creciente — el caso físico
   de acoplos que corren con S) ⟹ la reflexión ingenua PIERDE la
   positividad: el autovalor mínimo se hace negativo y la violación
   crece con el gradiente g (medida, sin ley simple reclamada).

Lectura honesta para el frente 1: el juguete muestra que la
estacionariedad no es la condición operativa — la SIMETRÍA ESPECULAR
del perfil es SUFICIENTE para la RP (K(S) = K(ϑS) ⟹ RP; la necesidad
NO está demostrada: puede haber perfiles no especulares accidentalmente
positivos). Con acoplos corriendo en S, la pregunta real del frente es
si el sector de Wilson admite una reflexión modificada (que refleje
también el perfil) — aquí la versión escalar de esa reflexión
modificada restaura la positividad exactamente (punto 1). El sector
espinorial queda donde el tratado lo deja: abierto.
"""

from __future__ import annotations

import numpy as np

STATUS_NONSTATIONARY = (
    "condicional (§13.4, frente abierto nº 1): en el juguete escalar la "
    "simetría especular del perfil es condición SUFICIENTE para la RP "
    "(la necesidad no está demostrada: puede haber perfiles no "
    "especulares accidentalmente positivos); los perfiles monótonos "
    "estudiados rompen la reflexión ingenua; la RP del sector "
    "espinorial de Wilson no se toca aquí"
)

# Reformulación del Lema de Precedencia (PROPUESTA PARA LA v36, nacida
# de este juguete; el enunciado del tratado no se toca): el presente no
# exige que las reglas estén congeladas — exige que el antes y el
# después sean compatibles bajo la reflexión que define la loncha. La
# condición pasa de la estacionariedad ∂_S K = 0 a la simetría
# K(S) = K(ϑS). Los cuatro casos, medidos aquí: perfil constante
# (especular trivial) → RP; variable pero especular → RP; monótono con
# reflexión ingenua → RP rota; reflejado junto con el campo → RP
# restaurada. Alcance: juguete escalar hasta resolver Wilson.
PRECEDENCE_REFORMULATED = (
    "propuesta v36 (desde el juguete escalar): la condición operativa "
    "del Lema de Precedencia pasa de ∂_S K = 0 (estacionariedad) a "
    "K(S) = K(ϑS) (simetría especular respecto de la loncha), como "
    "condición SUFICIENTE — el sellado exacto es suficiente pero no "
    "necesario, y la necesidad de la simetría no está demostrada"
)


def is_mirror_symmetric(m2_profile, J_profile, tol: float = 1e-12) -> bool:
    """La condición K(S) = K(ϑS), ejecutable: ¿es el perfil de acoplos
    especular respecto de la loncha (sitio 0)? Es condición SUFICIENTE
    para la RP del juguete (la necesidad no está demostrada)."""
    m2 = np.asarray(list(m2_profile), dtype=float)
    J = np.asarray(list(J_profile), dtype=float)
    if len(J) != len(m2) - 1 or len(m2) % 2 == 0:
        raise ValueError("Perfil mal formado (ver chain_rp_matrix)")
    return bool(np.all(np.abs(m2 - m2[::-1]) <= tol)
                and np.all(np.abs(J - J[::-1]) <= tol))


def V_site(phi: np.ndarray, m2: float, lam4: float = 0.05) -> np.ndarray:
    """Potencial de sitio con masa corriente: V = ½m²φ² + λ₄φ⁴.

    λ₄ (lam4) es el REGULARIZADOR CUÁRTICO del juguete — mantiene la
    medida normalizable para m² de cualquier signo; su valor 0.05 es de
    demostración. NOTA C6 (v35.1, E9 extendido): λ₄ no guarda relación
    alguna con λ (la razón de la Década) ni con λ_H = 0.130 (el
    cuártico del Higgs) — colisión de notación evitada por declaración."""
    return 0.5 * m2 * phi ** 2 + lam4 * phi ** 4


def chain_rp_matrix(phi_grid: np.ndarray, m2_profile, J_profile,
                    lam4: float = 0.05) -> np.ndarray:
    """M(a,b) = ⟨ϑ(F̄)·G⟩ para F = f(φ₋₁), G = g(φ₊₁), contrayendo la
    cadena de sitios −n..n con reflexión respecto del sitio 0.

    m2_profile: 2n+1 masas por sitio; J_profile: 2n acoplos por enlace.
    RP ⟺ la forma cuadrática fᵀMf es ≥ 0 ⟺ min eig((M+Mᵀ)/2) ≥ 0.
    """
    m2 = list(m2_profile)
    Js = list(J_profile)
    if len(Js) != len(m2) - 1:
        raise ValueError("J_profile debe tener len(m2_profile) − 1 enlaces")
    n_side = (len(m2) - 1) // 2
    if 2 * n_side + 1 != len(m2) or n_side < 1:
        raise ValueError("m2_profile debe tener longitud impar ≥ 3")
    idx0 = n_side
    phi = np.asarray(phi_grid, dtype=float)
    W = [np.exp(-V_site(phi, m, lam4)) for m in m2]
    L = [np.exp(J * np.outer(phi, phi)) for J in Js]

    tau_m = np.ones_like(phi)                    # cola −n .. −2
    for t in range(0, idx0 - 1):
        tau_m = L[t].T @ (tau_m * W[t])
    tau_p = np.ones_like(phi)                    # cola +n .. +2
    for t in range(len(m2) - 1, idx0 + 1, -1):
        tau_p = L[t - 1] @ (tau_p * W[t])
    # centro: C(a,b) = Σ_φ0 L[−1,0](a,φ0)·W0(φ0)·L[0,+1](φ0,b)
    C = L[idx0 - 1] @ np.diag(W[idx0]) @ L[idx0]
    vL = tau_m * W[idx0 - 1]
    vR = tau_p * W[idx0 + 1]
    return vL[:, None] * C * vR[None, :]


def rp_min_eig_nonstationary(phi_grid: np.ndarray, m2_profile, J_profile,
                             lam4: float = 0.05) -> float:
    """Autovalor mínimo normalizado de la parte simétrica de M."""
    M = chain_rp_matrix(phi_grid, m2_profile, J_profile, lam4)
    eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(eigs.min() / max(abs(eigs).max(), 1e-300))


def mirrored_profile(half_m2, m2_center: float, half_J) -> tuple:
    """Perfil especular respecto de la loncha: la REFLEXIÓN MODIFICADA
    que refleja también los acoplos (half_* son los perfiles del lado
    positivo, del sitio +1 hacia fuera)."""
    half_m2 = list(half_m2)
    half_J = list(half_J)
    m2 = half_m2[::-1] + [m2_center] + half_m2
    J = half_J[::-1] + half_J
    return m2, J


def running_profile(n_side: int, g_m2: float = 0.0, g_J: float = 0.0,
                    base_m2: float = 1.0, base_J: float = 1.0) -> tuple:
    """Acoplos corriendo monótonamente con la posición (el caso físico:
    los acoplos corren con S): m²(t) = base + g_m2·t por sitio,
    J = base + g_J·(t+½) por enlace."""
    m2 = [base_m2 + g_m2 * t for t in range(-n_side, n_side + 1)]
    J = [base_J + g_J * (t + 0.5) for t in range(-n_side, n_side)]
    return m2, J


def violation_curve(gradients, n_side: int = 3, which: str = "m2",
                    phi_grid: np.ndarray | None = None) -> dict:
    """Autovalor mínimo vs gradiente del running (which: 'm2' o 'J').
    Curva medida — no se reclama ley simple."""
    phi = np.linspace(-4.0, 4.0, 41) if phi_grid is None else phi_grid
    eigs = []
    for g in gradients:
        if which == "m2":
            m2, J = running_profile(n_side, g_m2=g)
        elif which == "J":
            m2, J = running_profile(n_side, g_J=g)
        else:
            raise ValueError("which: 'm2' o 'J'")
        eigs.append(rp_min_eig_nonstationary(phi, m2, J))
    return {"gradients": np.asarray(list(gradients), dtype=float),
            "min_eigs": np.array(eigs), "which": which}
