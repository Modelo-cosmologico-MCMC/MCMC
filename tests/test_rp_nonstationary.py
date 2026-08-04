"""Tests de la RP no estacionaria (frente nº 1, §13.4 — juguete escalar)."""

import numpy as np

from core.rp_nonstationary import (
    STATUS_NONSTATIONARY,
    mirrored_profile,
    rp_min_eig_nonstationary,
    running_profile,
    violation_curve,
)

PHI = np.linspace(-4.0, 4.0, 41)


def test_mirrored_nonstationary_holds():
    """Perfil especular ⟹ RP exacta, aunque los acoplos varíen sitio a
    sitio (la no-estacionariedad no rompe la RP por sí sola)."""
    for seed in (7, 11, 23):
        rng = np.random.default_rng(seed)
        m2, J = mirrored_profile(rng.uniform(0.5, 2.0, size=3),
                                 rng.uniform(0.5, 2.0),
                                 rng.uniform(0.5, 1.5, size=3))
        assert rp_min_eig_nonstationary(PHI, m2, J) > -1e-12


def test_stationary_is_a_mirrored_case():
    """El caso estacionario (g = 0) es especular ⟹ positivo."""
    m2, J = running_profile(3, g_m2=0.0)
    assert rp_min_eig_nonstationary(PHI, m2, J) > -1e-12


def test_monotone_running_violates():
    """Running monótono en m² ⟹ la reflexión ingenua pierde la
    positividad, y la violación crece con el gradiente."""
    m2_a, J_a = running_profile(3, g_m2=0.2)
    m2_b, J_b = running_profile(3, g_m2=0.4)
    va = rp_min_eig_nonstationary(PHI, m2_a, J_a)
    vb = rp_min_eig_nonstationary(PHI, m2_b, J_b)
    assert va < -1e-2          # violación franca
    assert vb < va             # y creciente con g


def test_running_J_also_violates():
    """El running en el acoplo de enlace J también viola."""
    m2, J = running_profile(3, g_J=0.4)
    assert rp_min_eig_nonstationary(PHI, m2, J) < -1e-2


def test_weak_running_almost_positive():
    """Con running débil la violación es minúscula (continuidad): la
    RP se pierde suavemente, no de golpe."""
    m2, J = running_profile(3, g_m2=0.05)
    v = rp_min_eig_nonstationary(PHI, m2, J)
    assert -1e-4 < v < 0.0


def test_modified_reflection_restores():
    """La reflexión modificada (reflejar también el perfil): tomar el
    lado positivo del running y espejarlo restaura la RP exacta —
    la versión escalar de la pregunta real del frente 1."""
    n = 3
    m2_run, J_run = running_profile(n, g_m2=0.4)
    half_m2 = m2_run[n + 1:]           # sitios +1..+n
    half_J = J_run[n:]                 # enlaces (0,1)..(n−1,n)
    m2, J = mirrored_profile(half_m2, m2_run[n], half_J)
    assert rp_min_eig_nonstationary(PHI, m2, J) > -1e-12


def test_violation_curve_monotone():
    """La curva de violación es monótona en el gradiente (medida)."""
    curve = violation_curve([0.1, 0.2, 0.4, 0.8], n_side=3)
    assert np.all(np.diff(curve["min_eigs"]) < 0.0)


def test_status_declares_open_front():
    assert "frente abierto nº 1" in STATUS_NONSTATIONARY
    assert "Wilson" in STATUS_NONSTATIONARY
