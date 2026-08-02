"""Tests de la positividad por reflexión en modelo de juguete (cap. 7)."""

import numpy as np

from core.basal import V0
from core.reflection_positivity import (
    transfer_kernel, rp_min_eigenvalue, rp_holds, measure_normalizable,
    STATUS_WILSON,
)


def _grid_and_potential(delta0: float = 0.05, n: int = 41):
    phi = np.linspace(-2.0, 2.0, n)
    V = np.array([V0(abs(p), 0.0, delta0) for p in phi])
    return phi, V


def test_rp_holds_with_physical_kinetic():
    """Teo. 7.1 (conclusión, en juguete): ⟨ϑ(F̄)F⟩ ≥ 0 con el término
    cinético físico (J=+1) y la medida del Potencial Basal."""
    phi, V = _grid_and_potential()
    assert rp_holds(phi, V, J=1.0)
    assert rp_min_eigenvalue(phi, V, J=1.0) >= -1e-10


def test_rp_holds_for_free_field_too():
    """La positividad no depende del detalle de V (V ≥ acotado): campo
    libre (V=0) también la cumple — es el término cinético quien la da."""
    phi = np.linspace(-2.0, 2.0, 31)
    assert rp_holds(phi, np.zeros_like(phi), J=1.0)


def test_negative_control_inverted_coupling():
    """CONTROL NEGATIVO: el acoplo temporal invertido (J<0) rompe la
    positividad — autovalores negativos de tamaño O(1). Esto es lo que
    distingue la auditoría de una lista de éxitos."""
    phi, V = _grid_and_potential()
    min_eig = rp_min_eigenvalue(phi, V, J=-1.0)
    assert min_eig < -1e-6


def test_kernel_symmetric():
    """El núcleo es simétrico (hipótesis de la verificación espectral)."""
    phi, V = _grid_and_potential(n=21)
    K = transfer_kernel(phi, V)
    assert np.allclose(K, K.T, atol=1e-14)


def test_sextic_term_normalizability():
    """El papel de C0 > 0 (§3.1): la medida e^{−V0} existe; con C0 < 0
    el paisaje se desfonda y no hay medida (control negativo)."""
    assert measure_normalizable(0.05, C0=+1.0)
    assert not measure_normalizable(0.05, C0=-1.0)


def test_wilson_sector_declared_conditional():
    """El sector espinorial (Teo. 7.4) queda declarado condicional
    (§13.4) — el módulo lo dice, no lo resuelve."""
    assert "condicional" in STATUS_WILSON
