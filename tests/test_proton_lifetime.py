"""Tests de la vida media del protón (III.H, §K.8)."""

import pytest

from mass_program.proton_decay import (
    proton_lifetime_MCMC, proton_lifetime_uncertainty,
    is_compatible_with_super_k, is_in_hyper_k_window,
    summary, channel_lifetime, EXPERIMENTAL_LIMITS,
)


def test_tau_p_order_of_magnitude():
    """τ_p ≈ 10³⁵ años (Tratado, §K.8)."""
    tau = proton_lifetime_MCMC()
    assert 1e34 < tau < 1e36


def test_compatible_with_super_K():
    """τ_MCMC > 1.6×10³⁴ yr (Super-K 2016)."""
    assert is_compatible_with_super_k()


def test_in_hyper_K_falsification_window():
    """τ_MCMC ∈ [0.5, 5] × 10³⁵ yr (ventana Hyper-K)."""
    assert is_in_hyper_k_window()


def test_uncertainty_propagation():
    """δτ/τ = 4·δM_X/M_X + 2·δα/α ~ 0.4 con incertidumbres nominales."""
    rel = proton_lifetime_uncertainty()
    assert 0.0 < rel < 1.0


def test_summary_keys():
    s = summary()
    for k in ("tau_p_MCMC_yr", "tau_p_rel_err", "M_X_GeV",
              "compatible_super_K", "in_hyper_K_window", "limits"):
        assert k in s


def test_channel_lifetime_eplus_pi0():
    """Canal dominante p→e⁺π⁰: τ_canal ≈ τ_total / BR ≈ τ_total."""
    tau_chan = channel_lifetime("p_to_eplus_pi0")
    assert 1e34 < tau_chan < 1e36
