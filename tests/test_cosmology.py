"""Tests del módulo cosmológico.

Los test_regression_* comparan contra los valores calibrados/de referencia
de constants.py (consistencia interna, no contraste observacional).
"""

import numpy as np

from mcmc_ontology import constants as C
from cosmology.background import H_of_z, Lambda_rel
from cosmology.perturbations import f_sigma8


def test_regression_H0_in_range():
    """H(z=0) ≈ H0 de referencia del corpus (el default de H_of_z)."""
    H = H_of_z(0.0)
    assert 68.5 < H < 71.0


def test_regression_Lambda_at_today():
    """Ω_Λ(z=0) ≈ Ω_Λ0 (1 + ε)."""
    Om = Lambda_rel(0.0)
    assert 0.65 < Om < 0.75


def test_Hz_monotone():
    """H(z) creciente en z."""
    z = np.linspace(0.0, 3.0, 50)
    H = H_of_z(z)
    assert np.all(np.diff(H) > 0)


def test_fsigma8_finite():
    z = np.linspace(0.05, 1.5, 10)
    fs = f_sigma8(z)
    assert np.all(np.isfinite(fs))
    assert np.all(fs > 0)


def test_regression_epsilon_value():
    """ε_Λ (amplitud de la transición) sigue siendo el valor calibrado
    del corpus. NO se identifica con δ₀ (regla canónica, ronda 5)."""
    assert abs(C.EPSILON_0 - 0.012) < 1e-9


def test_regression_delta_BIC_reference():
    """ΔBIC de CORPUS_REFERENCE (valor del corpus, no salida del código)."""
    assert C.DELTA_BIC < -5.0
