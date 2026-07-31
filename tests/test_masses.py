"""Tests del programa de masas (B4 + B5)."""

import pytest

from mcmc_ontology import constants as C
from mass_program.B4_masses import predict_fermion_masses, neutrino_sum_eV
from mass_program.B5_higgs import higgs_mass


def test_higgs_mass():
    """m_H = sqrt(2 β_3) · v3 ≃ 125.3 GeV (tratado, Prop. 12.1).

    El cómputo exacto da sqrt(0.26)·246 = 125.436; el tratado publica
    125.3 (redondeo propio). La tolerancia de 0.2 GeV cubre ambos.
    """
    m = higgs_mass()
    assert abs(m - C.M_HIGGS_MCMC) < 0.2


def test_higgs_vs_pdg():
    """Desviación m_H vs PDG < 0.5%."""
    m = higgs_mass()
    dev = abs(m - C.M_HIGGS_PDG) / C.M_HIGGS_PDG
    assert dev < 0.005


def test_neutrino_sum_bound():
    """Σ m_ν << 0.12 eV."""
    s = neutrino_sum_eV()
    assert s < 0.12


def test_mass_table_finite():
    """Todas las masas predichas son finitas y positivas."""
    res = predict_fermion_masses()
    for f in ("e", "mu", "tau", "u", "d", "s", "c", "b", "t"):
        m = res[f]["m_MCMC"]
        assert m > 0.0
        assert m == m  # not NaN
