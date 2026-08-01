"""Tests de la infraestructura de simulación del Apéndice B (v35)."""

import numpy as np
import pytest

from cronos.timestep import entropic_timestep, global_timestep, ETA_CRONOS
from cronos.poisson import solve_poisson, gradient
from cronos.channels import refresh_channels, local_dilation
from cronos.simulation import CronosPM, cic_deposit, cic_gather
from cronos.config import load_config


# ------------------------- paso entrópico (B.3) -------------------------

def test_timestep_reduces_with_density():
    """En regiones densas el reloj entrópico avanza más lento (B.3)."""
    acc = np.ones(3)
    rho = np.array([0.0, 200.0, 2000.0])
    dt = entropic_timestep(acc, 1.0, rho, rho_c=200.0, alpha_cr=0.03)
    assert dt[0] > dt[1] > dt[2]
    # Sin densidad, el paso es el newtoniano η/sqrt(|a|·a):
    assert abs(dt[0] - ETA_CRONOS) < 1e-12


def test_timestep_requires_positive_alpha():
    with pytest.raises(ValueError):
        entropic_timestep(np.ones(1), 1.0, np.ones(1), 200.0, alpha_cr=0.0)


def test_global_timestep_is_minimum():
    acc = np.ones(4)
    rho = np.array([0.0, 100.0, 200.0, 400.0])
    dts = entropic_timestep(acc, 1.0, rho, 200.0, 0.03)
    assert global_timestep(acc, 1.0, rho, 200.0, 0.03) == float(dts.min())


# ------------------------- Poisson modificado (B.2) ---------------------

def test_poisson_single_mode():
    """Para fuente sin(kx), Φ = −4πG·a²·sin(kx)/k² exacto (espectral)."""
    n, box = 32, 2.0 * np.pi
    x = np.arange(n) * box / n
    kx = 2.0 * np.pi / box  # modo fundamental
    source = np.sin(kx * x)[:, None, None] * np.ones((1, n, n))
    a, G = 0.7, 1.0
    phi = solve_poisson(source, box, a_scale=a, G=G)
    phi_exact = -4.0 * np.pi * G * a ** 2 * source / kx ** 2
    assert np.allclose(phi, phi_exact, atol=1e-10)


def test_poisson_zero_mean_source_zero_mean_phi():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(16, 16, 16))
    phi = solve_poisson(src, 10.0)
    assert abs(phi.mean()) < 1e-12


def test_gradient_of_sine():
    """∇ sin(kx) = k·cos(kx) exacto en malla espectral."""
    n, box = 32, 2.0 * np.pi
    x = np.arange(n) * box / n
    field = np.sin(x)[:, None, None] * np.ones((1, n, n))
    g = gradient(field, box)
    assert np.allclose(g[0], np.cos(x)[:, None, None] * np.ones((1, n, n)),
                       atol=1e-10)
    assert np.allclose(g[1], 0.0, atol=1e-12)


# ------------------------- canales y dilatación (B.3) -------------------

def test_refresh_channels_decay_and_source():
    """ρ_lat decae con κ_lat; ρ_id crece con η_dir·ρ_m (B.3 paso 3)."""
    lat = np.full((2, 2, 2), 10.0)
    idg = np.zeros((2, 2, 2))
    m = np.full((2, 2, 2), 5.0)
    lat2, id2 = refresh_channels(lat, idg, m, dt=0.1, kappa_lat=1.0,
                                 eta_dir=2.0)
    assert np.allclose(lat2, 10.0 - 0.1 * 10.0)
    assert np.allclose(id2, 0.1 * 2.0 * 5.0)


def test_local_dilation_saturates():
    """ζ = ζ0·ρ/(ρ+ρ*): 0 sin canal latente, → ζ0 al saturar (B.3)."""
    zeta = local_dilation(np.array([0.0, 1.0, 1e9]), zeta0=0.015,
                          rho_star=1.0)
    assert zeta[0] == 0.0
    assert abs(zeta[1] - 0.0075) < 1e-12
    assert abs(zeta[2] - 0.015) < 1e-6


# ------------------------- CIC ------------------------------------------

def test_cic_mass_conservation():
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 10.0, size=(50, 3))
    mass = np.ones(50)
    grid = cic_deposit(pos, mass, 8, 10.0)
    cell_vol = (10.0 / 8) ** 3
    assert abs(grid.sum() * cell_vol - 50.0) < 1e-9


def test_cic_gather_constant_field():
    field = np.full((8, 8, 8), 3.5)
    pos = np.random.default_rng(3).uniform(0, 10.0, size=(20, 3))
    assert np.allclose(cic_gather(field, pos, 10.0), 3.5)


# ------------------------- ciclo completo -------------------------------

def _tiny_sim(alpha0_inv: float, seed: int = 7) -> CronosPM:
    rng = np.random.default_rng(seed)
    box, n_p = 10.0, 64
    pos = rng.uniform(0, box, size=(n_p, 3))
    # sobredensidad central para inducir colapso
    pos[: n_p // 2] = box / 2 + rng.normal(scale=0.8, size=(n_p // 2, 3))
    vel = np.zeros((n_p, 3))
    return CronosPM(pos, vel, np.ones(n_p), grid_n=8, box=box,
                    alpha0_inv=alpha0_inv, rho_c=1.0)


def test_simulation_deterministic():
    """Misma semilla → misma trayectoria (pares A/B de B.5/B.6)."""
    s1, s2 = _tiny_sim(0.0), _tiny_sim(0.0)
    for _ in range(3):
        s1.step(dt=0.01)
        s2.step(dt=0.01)
    assert np.array_equal(s1.pos, s2.pos)
    assert np.array_equal(s1.vel, s2.vel)


def test_simulation_gate_inactive_without_alpha():
    """Con α0⁻¹ = 0 la compuerta no actúa: Γ = 0 en todos los pasos."""
    sim = _tiny_sim(0.0)
    for _ in range(3):
        diag = sim.step(dt=0.01)
        assert diag["Gamma_max"] == 0.0


def test_simulation_gate_fires_during_collapse():
    """Con α0⁻¹ > 0, la compuerta se activa en algún paso del colapso."""
    sim = _tiny_sim(1e-7)
    fired = False
    for _ in range(6):
        diag = sim.step(dt=0.01)
        if diag["Gamma_max"] > 0.0:
            fired = True
    assert fired


def test_simulation_finite():
    sim = _tiny_sim(1e-7)
    for _ in range(3):
        diag = sim.step(dt=0.01)
    assert np.all(np.isfinite(sim.pos)) and np.all(np.isfinite(sim.vel))
    assert np.isfinite(diag["v_rms"])


# ------------------------- configs --------------------------------------

def test_load_configs():
    """Los tres YAML de caja (B.4) cargan y declaran los campos clave."""
    for name in ("local", "meso", "lss"):
        cfg = load_config(name)
        assert "simulation" in cfg and "cronos" in cfg
        assert cfg["simulation"]["L_box_Mpc_h"] > 0
        assert cfg["cronos"]["delta_S"] == 1e-3
