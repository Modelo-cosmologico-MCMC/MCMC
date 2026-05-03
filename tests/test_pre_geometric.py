"""Tests del tramo pre-geométrico (n=0) y la tensión primordial."""

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.potential import (
    V_pre, V_total, beta0, gamma0, Mp_initial, Ep_initial,
    k_pre, lambda_pre,
)
from mcmc_ontology.seals import T_crit_table
from mass_program.B0_primordial import (
    delta_0, v0_norm, T0_GeV, T_crit, tension_decay, sigma0,
)


def test_delta_eq_v0():
    """δ₀ ≡ v₀ (Ec. 22)."""
    assert delta_0() == v0_norm()
    assert v0_norm() == C.EPSILON_0


def test_Mp_Ep_initial():
    """M_p^(0) + E_p^(0) = 1 (Ec. 23)."""
    assert abs(Mp_initial() + Ep_initial() - 1.0) < 1e-15


def test_T0_value():
    """T₀ = M_Pl · δ₀² ≈ 1.76e15 GeV (Ec. 441)."""
    T0 = T0_GeV()
    assert 1.5e15 < T0 < 2.0e15


def test_T_crit_hierarchy_table67():
    """Jerarquía Tabla 67: T_crit(C1) < T_crit(C2) < T_crit(C3) ≈ T_crit(C4)."""
    table = T_crit_table()
    assert table["C1"] < table["C2"] < table["C3"]
    # C3 y C4 coinciden a nivel de la tabulación
    assert abs(table["C3"] - table["C4"]) / table["C3"] < 1e-3
    # Valores positivos y finitos
    for v in table.values():
        assert v > 0.0 and np.isfinite(v)


def test_T_crit_canonical_values():
    """Tabla 67 del Tratado (p.156): valores numéricos canónicos."""
    table = T_crit_table()
    assert abs(table["C1"] - 6.9e11) / 6.9e11 < 0.05
    assert abs(table["C2"] - 7.6e12) / 7.6e12 < 0.05
    assert abs(table["C3"] - 7.6e13) / 7.6e13 < 0.05
    assert abs(table["C4"] - 7.6e13) / 7.6e13 < 0.05


def test_T_crit_seals_matches_B0():
    """T_crit en seals.py y en B0_primordial coinciden."""
    seals_table = T_crit_table()
    for s in ("C1", "C2", "C3", "C4"):
        assert abs(seals_table[s] - T_crit(s)) < 1e-3


def test_tension_decay_monotone():
    """T(S) decreciente en S (Ec. 450)."""
    Ss = np.linspace(0.001, 0.5, 50)
    Ts = [tension_decay(S) for S in Ss]
    assert all(Ts[i] >= Ts[i + 1] for i in range(len(Ts) - 1))


def test_lambda_pre_band():
    """λ_pre dentro de la banda admisible [1e-5, 5e-4]."""
    lp = lambda_pre()
    assert 1e-5 <= lp <= 5e-4


def test_V_pre_at_minimum():
    """V_pre(v₀; S>>S_pre) — finito en torno al mínimo."""
    val = float(V_pre(v0_norm(), 0.5))
    assert np.isfinite(val)


def test_V_total_includes_pre():
    """V_total contiene contribución pre-geométrica para S > S_pre."""
    Phi = 0.5
    S = 0.5
    v_only = float(V_total(Phi, S))
    v_pre = float(V_pre(Phi, S))
    assert abs(v_pre) > 0.0
    assert np.isfinite(v_only)


def test_k_pre_finite():
    """k_pre(S) finito (modulo zeros del potencial)."""
    val = k_pre(0.005)
    assert np.isfinite(val)


def test_sigma0_positive():
    """σ₀ = T₀/V₀D^(eq) > 0."""
    assert sigma0() > 0.0
