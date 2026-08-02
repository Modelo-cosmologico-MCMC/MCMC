"""Tests del sector gravitatorio (cap. 9) y el Ciclo de Victoria (cap. 10)."""

import numpy as np
import pytest

from core.gea import (
    STATUS_ATLAS,
    STATUS_RESIDUOS,
    G_cosmo,
    G_newton,
    atlas_healthy,
    cs2_atlas,
    newton_seal_ratio,
    residues_prediction,
)
from core.victoria import (
    STATUS_LYDIA,
    fertility_condition,
    is_silence,
    iterate_cycles,
    lydia_exponent,
    lydia_gain,
    memory_mode_mass_sq,
    return_map,
)

# ------------------------------ Cap. 9 -----------------------------------

def test_newton_seal_exact_recovery():
    """Prop. 9.3 / §13.5: en (λK, ξ) = (1, 1), G_cosmo = G_N EXACTAMENTE
    — la relatividad general se recupera en el punto fijo."""
    assert abs(newton_seal_ratio(1.0, 1.0) - 1.0) < 1e-15
    assert abs(G_cosmo(2.5, 1.0) - G_newton(2.5, 1.0)) < 1e-15


def test_atlas_propagation_window():
    """Ec. 9.4: c_s² > 0 exige λK > 1 (o λK < 1/3); la aproximación al
    sello es por arriba (λK ↓ 1⁺). Entre 1/3 y 1 el modo es patológico."""
    assert atlas_healthy(1.01)          # rama física, por arriba
    assert atlas_healthy(0.2)           # rama λK < 1/3
    assert not atlas_healthy(0.9)       # ventana patológica
    assert cs2_atlas(1.0) == 0.0        # el sello exacto: modo marginal
    with pytest.raises(ValueError):
        cs2_atlas(1.0 / 3.0)


def test_residues_conjecture_value():
    """Conjetura 9.6 (ec. 9.5): con εK = ε = 0.012, la predicción es
    G_cosmo/G_N − 1 ≈ −1.8% — y está declarada conjetura, no resultado."""
    assert abs(residues_prediction(0.012) - (-0.018)) < 1e-12
    assert "conjetura" in STATUS_RESIDUOS
    assert "condicional" in STATUS_ATLAS


# ------------------------------ Cap. 10 ----------------------------------

def test_memory_mode_exponent_5_2():
    """Ec. 10.1: m_θ² = η/ρ+ ∝ δ0^{5/2} — exponente medido en barrido."""
    deltas = np.logspace(-4, -1, 10)
    m2 = np.array([memory_mode_mass_sq(d) for d in deltas])
    slope = np.polyfit(np.log(deltas), np.log(m2), 1)[0]
    assert abs(slope - 2.5) < 1e-6
    assert memory_mode_mass_sq(0.0) == 0.0


def test_lydia_exponent_and_condition():
    """Ec. H.7: A = γR·ē/m̄², ν = ln(A)/ΔS; ν > 0 ⟺ γR·ē > m̄².
    El módulo EXPONE la condición (frente 4), no la decide."""
    assert lydia_gain(2.0) == 2.0                # con m̄ = ē = 1
    assert lydia_exponent(2.0) > 0.0
    assert lydia_exponent(0.5) < 0.0
    assert fertility_condition(1.5) and not fertility_condition(0.5)
    assert "condicional" in STATUS_LYDIA


def test_eternal_return_spiral():
    """Teo. 10.6 con ν > 0: la imperfección se amplifica vuelta a vuelta
    hasta el atractor (espiral, no círculo): la sucesión es estrictamente
    creciente hasta saturar y nunca revisita un valor anterior."""
    hist = iterate_cycles(1e-3, gamma_R=2.0, n_cycles=30,
                          delta_min=1e-4, delta_sat=0.1)
    assert not is_silence(hist)
    pre_sat = hist[hist < 0.1]
    assert np.all(np.diff(pre_sat) > 0)          # estrictamente creciente
    assert hist[-1] == 0.1                       # atractor en el techo
    assert len(np.unique(hist)) == len(hist) - np.sum(hist == 0.1) + 1


def test_silence_of_victoria():
    """Con ν < 0 el residuo decae bajo el suelo y la cadena no rearranca
    — el Silencio de Victoria como posibilidad estructural (10.5)."""
    hist = iterate_cycles(1e-3, gamma_R=0.5, n_cycles=50,
                          delta_min=1e-4, delta_sat=0.1)
    assert is_silence(hist)
    assert hist[-1] == 0.0
    # y una vez en el Silencio, es terminal:
    assert return_map(0.0, gamma_R=2.0, delta_min=1e-4, delta_sat=0.1) == 0.0


def test_perfection_is_repulsor():
    """Con ν > 0 la perfección (δ=0) es repulsor del mapa: cualquier
    δ > δ_min se aleja de cero; solo bajo el suelo cae al Silencio."""
    above = return_map(2e-4, gamma_R=2.0, delta_min=1e-4, delta_sat=0.1)
    below = return_map(5e-5, gamma_R=2.0, delta_min=1e-4, delta_sat=0.1)
    assert above > 2e-4
    assert below == 0.0
