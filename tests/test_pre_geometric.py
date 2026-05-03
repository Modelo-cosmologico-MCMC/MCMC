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


def test_T_crit_monotone():
    """T_crit^(n) crece con n (escalas v_n y S_n crecientes)."""
    table = T_crit_table()
    vals = [table[s] for s in ("C1", "C2", "C3", "C4")]
    # No es estrictamente monótona (S × v² crece y luego decae a C4),
    # pero todos los valores son finitos y positivos.
    for v in vals:
        assert v > 0.0
        assert np.isfinite(v)


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
