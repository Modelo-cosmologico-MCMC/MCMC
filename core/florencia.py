"""La Cadena de Álgebras y la Rotación de Florencia (v35, caps. 5-6).

Realiza los axiomas 5-6 (tramo euclidiano) y 7 (nacimiento del tiempo).

Cadena (§5.1, con la corrección declarada frente al corpus): el álgebra
de trabajo del tramo d-dimensional es C(d+1, 0) — d generadores
espaciales MÁS el generador entrópico γS, todos de norma +1:

    C(2,0) --S_0,100--> C(3,0) --S_1,000--> C(4,0) --S_1,001--> C(3,1)

Hasta el último umbral toda la cadena es euclidiana (elipticidad,
Lema 5.2: el símbolo D_E² = ℏ²(∂S² + c²∇²) es definido). Los dos sellos
de c viven en este tramo: la Cota de Delivery (§5.3, sello de c en
S_0,099) y el gap euclidiano (§5.4, sello de c² en S_0,999).

Rotación de Florencia (§6.1, ec. 6.2): en el Umbral de Florencia UN SOLO
generador cambia de norma,

    γ⁰ ≡ i·γS  ⟹  (γ⁰)² = −𝟙,  {γ⁰, γᵏ} = 0

y nace C(3,1) con firma (−,+,+,+): la dirección de estructuración se
convierte en dirección temporal. El resto queda intacto (§13.5: la
verificación de signo es que el giro es de UN generador).
"""

from __future__ import annotations

import numpy as np

_s0 = np.eye(2, dtype=complex)
_s1 = np.array([[0, 1], [1, 0]], dtype=complex)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_s3 = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def chain_generators(d: int) -> list[np.ndarray]:
    """Generadores euclidianos de C(d+1, 0) para el tramo d-dimensional.

    Devuelve [γ¹..γᵈ, γS]: d espaciales + el generador entrópico γS,
    todos con norma +1 ({γᵃ, γᵇ} = 2δᵃᵇ). d ∈ {1, 2, 3}.
    """
    if d == 1:      # C(2,0): 2x2
        return [_s1.copy(), _s3.copy()]
    if d == 2:      # C(3,0): 2x2 (σ1, σ2 espaciales; σ3 entrópico)
        return [_s1.copy(), _s2.copy(), _s3.copy()]
    if d == 3:      # C(4,0): 4x4 (σ1⊗σk espaciales; σ3⊗𝟙 entrópico)
        return [_kron(_s1, _s1), _kron(_s1, _s2), _kron(_s1, _s3),
                _kron(_s3, _s0)]
    raise ValueError(f"Tramo dimensional no soportado: d={d}")


def florencia_rotation(generators: list[np.ndarray],
                       n_rotations: int = 1) -> list[np.ndarray]:
    """La Rotación de Florencia: γ⁰ ≡ i·γS (ec. 6.2).

    El último generador de la lista (γS) se multiplica por i y pasa a
    ser γ⁰ (primero de la lista devuelta). n_rotations > 1 es el CONTROL
    NEGATIVO del §13.5: girar más de un generador no produce un cono de
    luz sino una firma con más de una dirección temporal.
    """
    gens = [g.copy() for g in generators]
    if not 1 <= n_rotations <= len(gens):
        raise ValueError("n_rotations fuera de rango")
    rotated = [1j * gens[-(k + 1)] for k in range(n_rotations)][::-1]
    untouched = gens[: len(gens) - n_rotations]
    return rotated[:1] + untouched + rotated[1:]


def signature(generators: list[np.ndarray]) -> list[int]:
    """Firma η_μμ extraída de {γμ, γν} = 2·η_μν·𝟙 (diagonal)."""
    dim = generators[0].shape[0]
    eye = np.eye(dim, dtype=complex)
    out = []
    for g in generators:
        sq = g @ g
        eta = complex(np.trace(sq) / dim)
        if not np.allclose(sq, eta * eye, atol=1e-12):
            raise ValueError("El cuadrado del generador no es proporcional a 𝟙")
        out.append(int(round(eta.real)))
    return out


def all_anticommute(generators: list[np.ndarray]) -> bool:
    """{γμ, γν} = 0 para μ ≠ ν (la relación de Clifford fuera de la
    diagonal, antes y después del giro)."""
    for i, gi in enumerate(generators):
        for gj in generators[i + 1:]:
            if not np.allclose(gi @ gj + gj @ gi, 0.0, atol=1e-12):
                return False
    return True


def euclidean_symbol(k_S: np.ndarray | float, k_vec: np.ndarray | float,
                     c: float = 1.0, hbar: float = 1.0) -> np.ndarray | float:
    """Símbolo del operador euclidiano D_E² = ℏ²(∂S² + c²∇²) (Lema 5.2).

    σ(k_S, k) = ℏ²·(k_S² + c²|k|²) > 0 para (k_S, k) ≠ 0: elíptico.
    (El símbolo lorentziano k_S² − c²|k|² NO es definido — control
    negativo de la elipticidad.)
    """
    k_S = np.asarray(k_S, dtype=float)
    k2 = np.asarray(k_vec, dtype=float) ** 2
    k2 = k2.sum(axis=-1) if k2.ndim > 0 and np.ndim(k_vec) > 0 else k2
    out = hbar ** 2 * (k_S ** 2 + c ** 2 * k2)
    return float(out) if np.ndim(out) == 0 else out
