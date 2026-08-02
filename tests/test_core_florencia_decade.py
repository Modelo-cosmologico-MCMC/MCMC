"""Tests de la Cadena de Álgebras, la Rotación de Florencia (caps. 5-6)
y el Mecanismo del Discriminante (cap. 8)."""

import numpy as np
import pytest

from core.decade import (
    discriminant_basal,
    fixed_points,
    lambda_from_s0,
    metastability_bound,
    radial_flow_rhs,
    s0_from_lambda,
    spinodal_crossing,
    walk_period,
)
from core.florencia import (
    all_anticommute,
    chain_generators,
    euclidean_symbol,
    florencia_rotation,
    signature,
)
from mcmc_ontology import constants as C

# --------------------------- caps. 5-6 -----------------------------------

def test_chain_euclidean_signatures():
    """§5.1: C(d+1,0) euclidiana en cada tramo — todos los generadores
    de norma +1 y anticomutantes."""
    for d in (1, 2, 3):
        gens = chain_generators(d)
        assert len(gens) == d + 1                 # d espaciales + γS
        assert signature(gens) == [1] * (d + 1)   # todos +1
        assert all_anticommute(gens)


def test_florencia_single_rotation():
    """Ec. 6.2 / §13.5: γ⁰ = i·γS → (γ⁰)² = −𝟙, UN solo generador
    girado, y la firma resultante es exactamente (−,+,+,+)."""
    gens = chain_generators(3)          # C(4,0)
    rotated = florencia_rotation(gens)  # C(3,1)
    sig = signature(rotated)
    assert sig == [-1, 1, 1, 1]
    assert all_anticommute(rotated)
    g0 = rotated[0]
    assert np.allclose(g0 @ g0, -np.eye(4), atol=1e-12)
    # un solo generador cambió de norma:
    assert sum(1 for s in sig if s == -1) == 1


def test_negative_control_two_rotations():
    """Control negativo (§13.5): girar DOS generadores no da un cono de
    luz — la firma tiene dos direcciones temporales."""
    gens = chain_generators(3)
    bad = florencia_rotation(gens, n_rotations=2)
    sig = signature(bad)
    assert sum(1 for s in sig if s == -1) == 2   # no es (−,+,+,+)
    assert sig != [-1, 1, 1, 1]


def test_ellipticity_lemma_5_2():
    """Lema 5.2: el símbolo euclidiano es definido positivo; el
    lorentziano (control) no lo es."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        kS = rng.normal()
        k = rng.normal(size=3)
        if abs(kS) + np.linalg.norm(k) < 1e-12:
            continue
        assert euclidean_symbol(kS, k) > 0.0
    # control negativo: kS² − c²|k|² cambia de signo
    def lorentz(kS, k):
        return kS ** 2 - float(np.sum(np.asarray(k) ** 2))
    assert lorentz(1.0, [0.1, 0, 0]) > 0 > lorentz(0.1, [1.0, 0, 0])


# --------------------------- cap. 8 --------------------------------------

def test_metastability_implies_D_positive():
    """Obs. 8.6: la casi-cancelación (3.3) implica D(S0) > (4/3)C0M0²."""
    assert metastability_bound(0.01)
    assert discriminant_basal(0.01) > 0.0


def test_fixed_points_merge_at_spinodal():
    """Prop. 8.5: x± se fusionan en D = 0 (x+ = x− = xR) y se
    complejifican para D < 0."""
    C0 = 1.0
    B = 2.0
    # D = B² − 4·C0·M0² = 0 → M0² = 1
    kind, xp, xm = fixed_points(B, 1.0, C0)
    assert kind == "real" and abs(xp - xm) < 1e-12
    kind, xR, omega = fixed_points(B, 1.5, C0)   # D < 0
    assert kind == "complex" and omega > 0.0 and abs(xR - 1.0) < 1e-12


def test_radial_flow_signs():
    """Ec. 8.3: el flujo empuja x hacia x+ (entre raíces sube, fuera baja)."""
    B, M0_sq, C0 = 3.0, 1.0, 1.0
    _, xp, xm = fixed_points(B, M0_sq, C0)
    x_mid = 0.5 * (xp + xm)
    assert radial_flow_rhs(x_mid, B, M0_sq, C0) > 0.0    # sube hacia x+
    assert radial_flow_rhs(1.5 * xp, B, M0_sq, C0) < 0.0  # baja hacia x+
    assert radial_flow_rhs(0.0, B, M0_sq, C0) == 0.0      # origen fijo


def test_walk_period_diverges_at_spinodal():
    """Ec. 8.4: Δσ_walk ∝ 1/Ω diverge al acercarse a la espinodal."""
    B, C0 = 2.0, 1.0
    t_deep = walk_period(B, M0_sq=2.0, C0=C0)    # D = −4
    t_near = walk_period(B, M0_sq=1.01, C0=C0)   # D = −0.04
    assert t_near > t_deep
    with pytest.raises(ValueError):
        walk_period(B, M0_sq=0.5, C0=C0)          # D > 0: sin walking


def test_lambda_s0_relation():
    """Def. 8.3 / ec. 8.2: λ = e^{π/s0}; para λ=10, s0 = π/ln10 —
    consistente con S0_VICTORIA de constants (calibrado, frente 2)."""
    s0 = s0_from_lambda(10.0)
    assert abs(s0 - C.S0_VICTORIA) < 1e-12
    assert abs(lambda_from_s0(s0) - 10.0) < 1e-12
    assert abs(s0 - 1.3644) < 1e-4


def test_spinodal_crossing_bisection():
    """El Cruce de Victoria localizado sobre un perfil D(S) que se hunde
    (Obs. 8.6): D(S) = D0 − k·S cruza cero en S* = D0/k."""
    def D_of_S(S):
        return 0.5 - 2.0 * S
    S_star = spinodal_crossing(D_of_S, 0.0, 1.0)
    assert abs(S_star - 0.25) < 1e-10
    with pytest.raises(ValueError):
        spinodal_crossing(lambda S: 1.0 + S, 0.0, 1.0)  # sin cruce


def test_negative_control_C0_negative():
    """Control negativo (§3.1/8.3): con C0 < 0 el paisaje se desfonda —
    no hay disparo espinodal bien definido (V(ρ→∞) → −∞)."""
    from core.basal import V0
    v_far = V0(10.0, 0.0, delta0=0.01, C0=-1.0)
    assert v_far < 0.0 and not np.isclose(v_far, 0.0)
    v_far_ok = V0(10.0, 0.0, delta0=0.01, C0=+1.0)
    assert v_far_ok > 0.0  # con C0 > 0 el paisaje está acotado por abajo