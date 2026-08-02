"""Tests del medio paso del frente nº 5 (opción G): forma del perfil y
compuerta H.2.5 en la malla PM mínima."""

import numpy as np

from cronos.simulation import CronosPM
from cronos.profile_fit import (
    spherical_clump_ic, track_region_density, gate_histories,
    halo_center, radial_profile, fit_shape, shape_comparison,
    friction_drop,
)

R = np.geomspace(0.1, 5.0, 15)


def _noisy(model: np.ndarray, seed: int, sigma_dex: float = 0.03):
    rng = np.random.default_rng(seed)
    return model * 10.0 ** rng.normal(0.0, sigma_dex, size=model.shape)


def test_fit_recovers_cored():
    """Datos cored sintéticos → el ajuste recupera r_c y prefiere cored
    frente a NFW."""
    rho = _noisy(50.0 / (1.0 + (R / 0.8) ** 2), seed=1)
    fit = fit_shape(R, rho, "cored")
    assert abs(fit["scale"] - 0.8) / 0.8 < 0.2
    cmp_ = shape_comparison(R, rho)
    assert cmp_["preferred"] == "cored"


def test_fit_recovers_nfw():
    """Control negativo: datos NFW → prefiere NFW (la comparación
    discrimina en ambos sentidos)."""
    x = R / 0.8
    rho = _noisy(50.0 / (x * (1.0 + x) ** 2), seed=2)
    fit = fit_shape(R, rho, "nfw")
    assert abs(fit["scale"] - 0.8) / 0.8 < 0.25
    cmp_ = shape_comparison(R, rho)
    assert cmp_["preferred"] == "nfw"


def test_radial_profile_uniform_density():
    """Con partículas uniformes, ρ(r) reproduce la densidad media en
    las cáscaras bien pobladas (las internas son ruido de Poisson —
    por eso el perfil devuelve `count`)."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, 10.0, size=(20000, 3))
    mass = np.ones(20000)
    prof = radial_profile(pos, mass, np.array([5.0, 5.0, 5.0]), 10.0)
    well = prof["count"] >= 50
    assert well.sum() >= 4
    assert np.all(np.abs(prof["rho"][well] - 20.0) / 20.0 < 0.25)


def test_halo_center_finds_clump():
    """El centro cae en el grumo, no en el fondo."""
    pos, _, mass = spherical_clump_ic(4, 2048, 10.0, scale=0.5,
                                      clump_frac=0.5)
    c = halo_center(pos, mass, 10.0)
    assert np.all(np.abs(c - 5.0) < 1.0)


def test_friction_drop_pure_function():
    """friction_drop sobre historias sintéticas: caída limpia → cierre
    exacto; historia constante → sin caída."""
    gamma = np.concatenate([np.linspace(0.0, 1e-3, 20), np.zeros(20)])
    d = friction_drop(gamma, i_peak=20, tail_frac=0.25)
    assert d["gate_closed_exactly"] and d["orders_drop"] == np.inf
    assert d["tail_zero_frac"] == 1.0
    flat = friction_drop(np.full(40, 1e-3), i_peak=20, tail_frac=0.25)
    assert flat["orders_drop"] < 0.01


def test_gate_drops_and_old_scheme_persists():
    """H.2.5 en la malla mínima (colapso aislado reproducible): la Γ
    con compuerta cae ≥ 1 orden (máx conservador) y cierra exactamente
    (Γ=0) en la mayoría del tramo virializado; la forma sin compuerta
    (11.3b) persiste (< 0.5 órdenes). Los valores absolutos ≈4e-4→≈7e-9
    del tratado son de SU simulación de producción, no de esta malla."""
    n_p = 1024
    pos, vel, mass = spherical_clump_ic(20260802, n_p, 10.0)
    rho_c = 200.0 * n_p / 1000.0     # umbral de colapso: 200 × media
    sim = CronosPM(pos, vel, mass, grid_n=16, box=10.0,
                   alpha0_inv=1e-6, rho_c=rho_c)
    rho = track_region_density(sim, 350, 0.004, 1.0)
    gh = gate_histories(rho, 0.004, 1e-6, rho_c)
    gated = friction_drop(gh["gated"], gh["i_peak"], tail_frac=0.25)
    ungated = friction_drop(gh["ungated"], gh["i_peak"], tail_frac=0.25)
    assert gated["Gamma_collapse"] > 0.0          # activa en el colapso
    assert gated["orders_drop"] >= 1.0            # cae (máx conservador)
    assert gated["tail_zero_frac"] >= 0.4         # cierre exacto frecuente
    assert gated["Gamma_virial_median"] < 0.01 * gated["Gamma_collapse"]
    assert ungated["orders_drop"] < 0.5           # la antigua persiste
