"""Las cifras comprobables de la lista de erratas para la v36 (docs/erratas_v36_lista_2026-11-12.md; PR R5 del plan del
11-nov). La lista es documentación: este fichero solo vigila que lo que dice «comprobado en el repositorio» lo esté, y
que el álgebra de las fórmulas atribuidas a los tratados de 2025 sea la que el documento escribe (las citas siguen
pendientes de confirmar). Comprobación interna (E8)."""

import math
from pathlib import Path

import numpy as np
import pytest

from core.basal import T0_analytic, kappa_plus, scaled_params
from core.s_clock import kappa1_tilt, radial_landscape
from core.victoria import memory_mode_mass_sq
from cosmology.background import Lambda_rel
from cronos.channels import refresh_channels
from mass_program import B7_empalme as B7
from mcmc_ontology import constants as C

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "erratas_v36_lista_2026-11-12.md"


def test_document_states_what_it_is_and_what_it_is_not():
    text = DOC.read_text(encoding="utf-8")
    assert "No es una fe de erratas" in text and "FE-08-2026" in text and "erratas_epistemologicas_2026-08-04.md" in text
    assert "ninguna de las cinco es una errata hasta que el autor" in text and "confirme la cita" in text
    for section in ("# 1. Erratas que no requieren decisión", "# 2. Pendientes del informe del 5 de noviembre",
                    "# 3. Tratados de 2025: pendiente de confirmar la cita", "# 4. Lo que esta lista no cubre"):
        assert section in text
    # las filas y los tests que cita existen
    for row in ("residuo-falso-vacio-c4", "cruce-victoria-retorno", "cuarta-ecuacion-collado", "rp-camino-onsager-machlup",
                "rp-no-estacionaria", "brazo-dos-sentidos-todas-las-posibilidades"):
        assert f"`{row}`" in text and f"claim_id: {row}\n" in (REPO / "docs" / "claims_registry.yaml").read_text(encoding="utf-8")
    for test in ("test_chi_residue_c4.py", "test_victoria_return.py", "test_rp_nonstationary.py",
                 "test_camino_onsager_machlup.py", "test_erratas_v36.py"):
        assert f"tests/{test}" in text and (REPO / "tests" / test).exists()


def test_entry_2_tilt_is_not_small_relative_to_T0():
    """Prop. 3.4: frente a T₀ la inclinación corrige κ₁·√δ₀ con κ₁ = 1.361: +14.6 % / +30.4 % / +31.1 %."""
    assert kappa1_tilt() == pytest.approx(1.361, abs=5e-4)
    corr = [radial_landscape(d)["T0_full"] / T0_analytic(d) - 1.0 for d in (0.012, 0.0554, 0.058)]
    assert corr == pytest.approx([0.146, 0.304, 0.311], abs=5e-4)


def test_entry_5_channel_refresh_of_B3_does_not_conserve_the_total():
    """Ap. B.3: el total ρ_lat + ρ_id + ρ_m cambia en Δt·(−κ_lat·ρ_lat + Γ_lat + η_dir·ρ_m + Γ_act)."""
    rho_lat, rho_id, rho_m, dt = np.array([1.0]), np.array([2.0]), np.array([3.0]), 0.1
    for kw in ({"kappa_lat": 0.2, "eta_dir": 0.05}, {"kappa_lat": 0.2, "Gamma_lat": 0.1, "eta_dir": 0.05, "Gamma_act": 0.02}):
        new_lat, new_id = refresh_channels(rho_lat, rho_id, rho_m, dt, **kw)
        expected = dt * (-kw["kappa_lat"] * 1.0 + kw.get("Gamma_lat", 0.0) + kw["eta_dir"] * 3.0 + kw.get("Gamma_act", 0.0))
        assert (new_lat + new_id - rho_lat - rho_id)[0] == pytest.approx(expected, rel=1e-12) and expected != 0.0
    # solo cierra si las fuentes se eligen para ello
    new_lat, new_id = refresh_channels(rho_lat, rho_id, rho_m, dt, kappa_lat=0.2, eta_dir=0.05, Gamma_lat=0.05)
    assert (new_lat + new_id - rho_lat - rho_id)[0] == pytest.approx(0.0, abs=1e-15)


def test_entry_6_memory_mode_sits_outside_the_physical_domain():
    """Prop. 10.2: el mínimo angular del anillo de vacíos está en θ = −π/4 (φ_E < 0); en θ = 0, pendiente η/√2."""
    delta0 = 0.01
    eta, rho_p = scaled_params(delta0)["eta"], math.sqrt(kappa_plus() * delta0)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 200001)
    ring = -eta * rho_p * (np.cos(theta) - np.sin(theta)) / math.sqrt(2.0)        # −η·χ sobre ρ = ρ₊
    assert theta[np.argmin(ring)] == pytest.approx(-math.pi / 4.0, abs=1e-5)
    curvature = np.gradient(np.gradient(ring, theta), theta)[np.argmin(ring)] / rho_p ** 2
    assert curvature == pytest.approx(eta / rho_p, rel=1e-6) and eta / rho_p == pytest.approx(6.18e-6, rel=1e-3)
    assert memory_mode_mass_sq(delta0) == pytest.approx(eta / rho_p, rel=1e-12)
    slope_at_boundary = np.gradient(ring, theta)[np.argmin(np.abs(theta))] / rho_p    # por unidad de arco, en θ = 0
    assert slope_at_boundary == pytest.approx(eta / math.sqrt(2.0), rel=1e-6) and slope_at_boundary > 0.0
    # exponente 5/2 de la ley de escala
    assert memory_mode_mass_sq(0.04) / memory_mode_mass_sq(0.01) == pytest.approx(4.0 ** 2.5, rel=1e-12)


def test_entry_7_two_normalizations_of_beta3():
    """(12.1) frente a (H.8) y F.3: 0.13 y 0.0325, un factor 4."""
    assert B7.BETA3_CONVENIO_12_1 == 0.130 and B7.BETA3_CONVENIO_H8 == pytest.approx(0.0325)
    assert B7.BETA3_CONVENIO_12_1 / B7.BETA3_CONVENIO_H8 == pytest.approx(4.0)
    # m_H = √(2β₃)·v y m_H² = 8β₃v² dan la misma masa solo con cada β₃ en su convenio
    assert math.sqrt(2.0 * B7.BETA3_CONVENIO_12_1) * C.V3_GEV == pytest.approx(math.sqrt(8.0 * B7.BETA3_CONVENIO_H8) * C.V3_GEV)
    assert math.sqrt(8.0 * B7.BETA3_CONVENIO_12_1) == pytest.approx(2.0 * math.sqrt(2.0 * B7.BETA3_CONVENIO_12_1))


def test_entry_8_lambda_rel_of_A3_has_w_below_minus_one():
    """Λ_rel de A.3 con los valores del canon: w ≤ −1 en todo z, mínimo −1.027 junto a z_trans, w(0) + 1 = −7e-8."""
    assert (C.EPSILON_LAMBDA, C.Z_TRANS, C.DZ_TRANS) == (0.012, 8.9, 1.5)
    z = np.linspace(0.0, 20.0, 200001)
    w = -1.0 + np.gradient(np.log(Lambda_rel(z)), np.log1p(z)) / 3.0
    assert w.max() <= -1.0 and w.min() == pytest.approx(-1.027, abs=5e-4)
    assert abs(z[np.argmin(w)] - C.Z_TRANS) < 0.2 and w[1] + 1.0 == pytest.approx(-7.4e-8, rel=0.02)


def test_bound_11_5_arithmetic():
    """(v/c)² = 1.1e-9 / 4.5e-7 / 1.0e-6 para v = 10 / 200 / 300 km/s."""
    c_kms = 299792.458
    assert [(v / c_kms) ** 2 for v in (10.0, 200.0, 300.0)] == pytest.approx([1.1e-9, 4.5e-7, 1.0e-6], rel=0.02)


def test_algebra_of_the_formulas_attributed_to_the_2025_treatises():
    """Solo el álgebra de las entradas 9, 10, 11 y 13; que los tratados de 2025 digan eso sigue pendiente."""
    # 9: d/dS[(λ/kα)·ln cosh(S/λ)] = tanh(S/λ)/(kα), sin el factor λ
    lam, k_alpha, S, h = 1e-2, 0.7, 0.013, 1e-7
    t_rel = lambda s: (lam / k_alpha) * math.log(math.cosh(s / lam))  # noqa: E731
    deriv = (t_rel(S + h) - t_rel(S - h)) / (2.0 * h)
    assert deriv == pytest.approx(math.tanh(S / lam) / k_alpha, rel=1e-6)
    assert deriv / (lam * math.tanh(S / lam) / k_alpha) == pytest.approx(100.0, rel=1e-6)      # con λ = 1e-2, un factor 100
    # 10: mínimos de V = −μ²Φ² + λΦ⁴ en ±μ/√(2λ), profundidad −μ⁴/(4λ)
    mu, lam4 = 1.3, 0.4
    phi = np.linspace(-3.0, 3.0, 600001)
    V = -mu ** 2 * phi ** 2 + lam4 * phi ** 4
    assert abs(phi[np.argmin(V)]) == pytest.approx(mu / math.sqrt(2.0 * lam4), abs=1e-5)
    assert V.min() == pytest.approx(-mu ** 4 / (4.0 * lam4), rel=1e-9)
    # 11: η = ε·exp(ε·N) es creciente en N: η ≥ ε_CTP = 0.001/0.099 para N ≥ 0
    eps = 0.001 / 0.099
    assert eps == pytest.approx(0.0101, abs=5e-5)
    N = np.linspace(0.0, 500.0, 5001)
    eta = eps * np.exp(eps * N)
    assert eta.min() == eta[0] == eps and np.all(np.diff(eta) > 0.0) and eta.min() > 6e-10
    # 13: g = ḡ·(1 + Φ_ten/M_Pl) en Φ_ten = 0 da g = ḡ
    assert 1.0 + 0.0 / 1.22e19 == 1.0
