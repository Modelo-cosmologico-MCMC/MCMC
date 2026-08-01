"""Tests del contraste de los Residuos (frente nº 6, Conj. 9.6 / H.2.5)."""

from cosmology.residues_test import (
    predicted_ratio, tension_sigma, within_bbn_bound, sign_favored, report,
    BBN_REFERENCE,
)


def test_prediction_value():
    """Ec. 9.5 con ε_K = 0.012: G_cosmo/G_N = 0.982 (−1.8%)."""
    assert abs(predicted_ratio(0.012) - 0.982) < 1e-12


def test_within_bbn_at_point_three_sigma():
    """H.2.5: dentro de la cota BBN a 2σ, a ~0.3σ del central."""
    assert within_bbn_bound()
    assert abs(tension_sigma() - 0.32) < 0.05


def test_sign_favored():
    """H.2.5: el central BBN (0.99 < 1) favorece el signo de la
    predicción (también < 1)."""
    assert sign_favored()


def test_report_carries_citation():
    """El contraste lleva su referencia bibliográfica completa."""
    r = report()
    assert "arXiv:1910.10730" in r and "arXiv:1910.10730" in BBN_REFERENCE
    assert "0.32σ" in r or "0.3" in r
