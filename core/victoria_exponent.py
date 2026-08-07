"""El frente 2 instrumentado: la matriz de estabilidad de la ec. 14.2.

El tratado (§14.2) define el objeto del frente: la matriz de estabilidad
del flujo de los acoplos del Basal λi = (M0², B, C0) en el cuello
espinodal,

    dλi/d ln S = βi(λ),   M_ij = ∂βi/∂λj |_{D=0}     (ec. 14.2)

y el exponente de Victoria es la parte imaginaria de sus autovalores
cuando se complejifican: s0 = |Im μ(M)|, de donde λ = e^{π/s0}
(Prop. 8.5 / Def. 8.3).

LO QUE ESTE MÓDULO HACE:

1. Dado M (el objeto de 14.2), extrae s0 por DOS RUTAS independientes
   y comprueba que coinciden: la espectral (|Im μ|) y la dinámica
   (integrar el flujo linealizado y medir el periodo de rotación del
   walking por unwrap del ángulo). La maquinaria de extracción queda
   validada con error ≲ 1e-4.

2. Barrido con ANSATZ O(1) declarado (entradas i.i.d. U(−a, a); las β
   reales no existen todavía): mide qué fracción de matrices produce
   cascada DSI (par complejo — genérica, ~2/3, robusta frente a la
   escala a) y qué fracción cae en la banda ±10% de s0 = π/ln 10 =
   1.3644 (pocas unidades por ciento, y DEPENDE de la escala a del
   ansatz porque s0 escala linealmente con ella).

LO QUE NO HACE — el hueco declarado del frente 2: derivar βM², βB, βC
del Flujo del Camino (la jerarquía de Fokker-Planck de la Def. 4.4).
Este módulo está listo para consumirlas cuando existan; hasta entonces,
la conclusión medible es que λ = 10 es una SELECCIÓN dentro del ansatz
O(1), no una consecuencia genérica — exactamente la disyuntiva que
§14.2 declara («si resulta λ ≠ 10, el diez era convención»).
"""

from __future__ import annotations

import numpy as np

from .decade import lambda_from_s0

STATUS_FRENTE2 = (
    "condicional (§14.2, frente abierto nº 2): la maquinaria espectro ⟺ "
    "walking está validada; las β-funciones reales (jerarquía de "
    "Fokker-Planck, Def. 4.4) son el hueco declarado — con ansatz O(1), "
    "la cascada DSI es genérica pero λ = 10 es una selección medible, "
    "no una consecuencia"
)

S0_TARGET = float(np.pi / np.log(10.0))   # π/ln 10 ≈ 1.3644 (ec. 8.2)


def s0_spectral(M: np.ndarray) -> float:
    """Ruta 1 (ec. 14.2): s0 = |Im μ(M)| del par complejo.

    Lanza ValueError si el espectro es enteramente real (sin cascada
    DSI: los exponentes no se complejifican)."""
    eigs = np.linalg.eigvals(np.asarray(M, dtype=float))
    im = np.abs(eigs.imag)
    if im.max() < 1e-12:
        raise ValueError("Espectro real: sin par complejo (sin DSI)")
    return float(im.max())


def has_complex_pair(M: np.ndarray) -> bool:
    """¿Se complejifican los exponentes? (la condición de cascada DSI)."""
    eigs = np.linalg.eigvals(np.asarray(M, dtype=float))
    return bool(np.abs(eigs.imag).max() >= 1e-12)


def s0_dynamic(M: np.ndarray, t_max: float = 40.0, dt: float = 1e-3,
               seed: int = 0) -> float:
    """Ruta 2: integra dδ/dt = M·δ (RK4) y mide la frecuencia de
    rotación del walking por unwrap del ángulo en el plano invariante
    del par complejo (el modo real, si lo hay, se proyecta fuera con
    su autovector izquierdo; la MEDIDA del periodo es dinámica)."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    eigs, vecs_r = np.linalg.eig(M)
    # proyector fuera de los modos reales (autovectores izquierdos):
    real_idx = [i for i in range(n) if abs(eigs[i].imag) < 1e-12]
    cplx_idx = [i for i in range(n) if abs(eigs[i].imag) >= 1e-12]
    if not cplx_idx:
        raise ValueError("Espectro real: sin walking que medir")
    # base del plano invariante del par complejo:
    v = vecs_r[:, cplx_idx[0]]
    e1, e2 = np.real(v), np.imag(v)
    rng = np.random.default_rng(seed)
    delta = e1 + 0.1 * rng.standard_normal(n)
    # quitar componentes de los modos reales via autovectores izquierdos
    if real_idx:
        eigs_l, vecs_l = np.linalg.eig(M.T)
        for i in range(n):
            if abs(eigs_l[i].imag) < 1e-12:
                w = np.real(vecs_l[:, i])
                # v_r derecho correspondiente: el de igual autovalor
                j = int(np.argmin(np.abs(eigs - eigs_l[i])))
                vr = np.real(vecs_r[:, j])
                denom = float(w @ vr)
                if abs(denom) > 1e-12:
                    delta = delta - vr * (float(w @ delta) / denom)

    # coordenadas en el plano (e1, e2) por mínimos cuadrados:
    basis = np.stack([e1, e2], axis=1)
    proj = np.linalg.pinv(basis)

    n_steps = int(t_max / dt)
    angles = np.empty(n_steps + 1)
    a, b = proj @ delta
    angles[0] = np.arctan2(b, a)
    for k in range(n_steps):
        k1 = M @ delta
        k2 = M @ (delta + 0.5 * dt * k1)
        k3 = M @ (delta + 0.5 * dt * k2)
        k4 = M @ (delta + dt * k3)
        delta = delta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        nrm = np.linalg.norm(delta)
        if nrm > 1e100 or nrm < 1e-100:
            delta = delta / nrm          # renormaliza (flujo lineal)
        a, b = proj @ delta
        angles[k + 1] = np.arctan2(b, a)
    total = np.unwrap(angles)
    return float(abs(total[-1] - total[0]) / (n_steps * dt))


def routes_coincide(M: np.ndarray, rtol: float = 1e-3) -> dict:
    """Las dos rutas al exponente de Victoria, comparadas."""
    spec = s0_spectral(M)
    dyn = s0_dynamic(M)
    return {"s0_spectral": spec, "s0_dynamic": dyn,
            "rel_error": abs(dyn / spec - 1.0),
            "coincide": abs(dyn / spec - 1.0) < rtol,
            "lambda_spectral": lambda_from_s0(spec)}


def o1_ansatz_scan(n: int = 20000, seed: int = 20260805,
                   scale: float = 1.5, dim: int = 3,
                   band: float = 0.10) -> dict:
    """Barrido del ansatz O(1): matrices de estabilidad con entradas
    i.i.d. U(−scale, scale) (el ansatz se DECLARA; las β reales son el
    hueco del frente 2).

    Devuelve la fracción con cascada DSI (par complejo), la
    distribución de s0 condicionada a DSI, y la fracción dentro de la
    banda ±band de s0 = π/ln10. ADVERTENCIA DECLARADA: s0 escala
    linealmente con `scale`, así que la fracción en banda depende de
    la escala del ansatz; la genericidad de la DSI, no."""
    rng = np.random.default_rng(seed)
    mats = rng.uniform(-scale, scale, size=(n, dim, dim))
    eigs = np.linalg.eigvals(mats)
    im_max = np.abs(eigs.imag).max(axis=1)
    dsi = im_max >= 1e-12
    s0s = im_max[dsi]
    in_band = np.abs(s0s - S0_TARGET) <= band * S0_TARGET
    return {
        "n": n, "scale": scale, "dim": dim,
        "frac_dsi": float(dsi.mean()),
        "s0_values": s0s,
        "s0_median": float(np.median(s0s)) if s0s.size else float("nan"),
        "frac_band": float(in_band.mean()) if s0s.size else float("nan"),
        "band": band,
    }
