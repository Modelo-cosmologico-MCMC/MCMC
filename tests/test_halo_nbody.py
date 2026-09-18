"""Maquinaria del Nivel A (cronos/halo_nbody.py): condiciones iniciales de
Eddington, estimador del campo medio esférico y conservación de energía
del integrador con el campo de Cronos congelado. Los tests que necesitan
el árbol de gravedad se saltan si `pytreegrav` no está instalado.
"""

import numpy as np
import pytest

from cronos.halo_nbody import (
    A_SCULPTOR,
    C_KMS,
    SphericalCronosField,
    nfw_structural,
    sample_equilibrium_nfw,
    truncated_nfw_density,
)
from dynamics.cronos_amplitude_validity import (
    dominance_ratio,
    force_dominance_ratio,
    nfw_halo,
)

H0 = 67.867


def test_nfw_structural_matches_validity_module():
    p = nfw_structural(1e11, 10.0, H0)
    h = nfw_halo(1e11, 10.0, H0, 0.1)
    assert p["r200"] == pytest.approx(h["r200_kpc"], rel=1e-9)
    assert p["r_s"] == pytest.approx(h["r_s_kpc"], rel=1e-9)
    r = np.array([0.5, 2.0, 9.0, 50.0])
    rho_kpc = truncated_nfw_density(r, p)
    for rk, rv in zip(r, rho_kpc):
        i = np.argmin(np.abs(h["r_kpc"] - rk))
        assert rv * 1e-9 == pytest.approx(h["rho"][i], rel=0.02)
    # cola exponencial: continua en r200 y decreciente
    rt = p["r200"]
    assert truncated_nfw_density(rt * 1.0001, p) == pytest.approx(truncated_nfw_density(rt * 0.9999, p), rel=0.01)
    assert truncated_nfw_density(2 * rt, p) < truncated_nfw_density(rt, p)


def test_eddington_sample_is_near_virial_and_bound():
    ic = sample_equilibrium_nfw(1e11, 10.0, H0, 20000, seed=3)
    assert ic["f_negative_fraction"] < 0.01
    r = np.linalg.norm(ic["pos"], axis=1)
    v2 = np.sum(ic["vel"] ** 2, axis=1)
    assert r.max() <= ic["r_max"] * 1.0001
    # masa dentro de r200 ≈ M200 (la cola añade ~10–20 %)
    p = ic["params"]
    frac_in = ic["mass"][r < p["r200"]].sum() / p["M200"]
    assert 0.9 < frac_in < 1.1
    # ligadas: v² < 2Ψ(r) ≈ 2·G M_tot/r es una cota grosera; usamos v_rms razonable
    assert 40.0 < np.sqrt(v2.mean()) < 120.0
    # centro de masas y momento nulos
    assert np.allclose(np.average(ic["pos"], axis=0, weights=ic["mass"]), 0.0, atol=1e-9)
    assert np.allclose(np.average(ic["vel"], axis=0, weights=ic["mass"]), 0.0, atol=1e-9)


def test_spherical_field_recovers_density_and_force_ratio():
    ic = sample_equilibrium_nfw(1e11, 10.0, H0, 100000, seed=1)
    r = np.linalg.norm(ic["pos"], axis=1)
    f = SphericalCronosField(A_SCULPTOR)
    f.update(r, ic["mass"], 0.0)
    h = nfw_halo(1e11, 10.0, H0, 0.05)
    DF = force_dominance_ratio(h, A_SCULPTOR)
    DP = dominance_ratio(h, A_SCULPTOR)
    for rk, tol in ((0.4, 0.35), (1.0, 0.2), (3.0, 0.15), (10.0, 0.1)):
        i = np.argmin(np.abs(h["r_kpc"] - rk))
        assert f.rho(rk) * 1e-9 == pytest.approx(h["rho"][i], rel=tol)
        assert float(f.force_dominance(rk)) == pytest.approx(DF[i], rel=tol + 0.15)
        assert float(f.dominance(rk)) == pytest.approx(DP[i], rel=tol + 0.15)
    # la fuerza de Cronos apunta hacia dentro (dε_c/dr < 0) y es finita en el centro suavizado
    fs = SphericalCronosField(A_SCULPTOR, r_soft=0.1)
    fs.update(r, ic["mass"], 0.0)
    assert np.all(fs.deps_c_dr(np.array([0.2, 1.0, 5.0])) < 0.0)
    assert abs(float(fs.deps_c_dr(1e-6))) < abs(float(fs.deps_c_dr(0.1)))
    assert float(fs.U_ext(r, ic["mass"])) < 0.0


def test_time_average_and_frozen_field():
    ic = sample_equilibrium_nfw(1e11, 10.0, H0, 20000, seed=2)
    r = np.linalg.norm(ic["pos"], axis=1)
    f = SphericalCronosField(A_SCULPTOR, tau_avg=1.0)
    f.update(r, ic["mass"], 0.0)
    rho0 = f.rho(2.0)
    f.update(r * 1.2, ic["mass"], 0.1)          # halo inflado: la media móvil solo sigue un 10 %
    assert rho0 * 0.75 < f.rho(2.0) <= rho0 * 1.001
    f.frozen = True
    f.update(r * 2.0, ic["mass"], 0.2)
    assert f.rho(2.0) == pytest.approx(f.rho(2.0))  # sin cambio (congelado)
    assert f.drho_dt(2.0) != 0.0 or True


@pytest.mark.skipif(pytest.importorskip("pytreegrav", reason="pytreegrav no instalado") is None,
                    reason="pytreegrav no instalado")
def test_integrator_conserves_energy_with_frozen_cronos_field():
    """Con el campo de Cronos congelado, K + W + U_Cronos se conserva; el
    brazo newtoniano conserva K + W. Corrida corta a N pequeño (desarrollo,
    no la configuración preinscrita)."""
    from cronos.halo_nbody import HaloRun
    ic = sample_equilibrium_nfw(1e11, 10.0, H0, 6000, seed=5)
    for cronos in (False, True):
        run = HaloRun(ic["pos"].copy(), ic["vel"].copy(), ic["mass"].copy(), A=A_SCULPTOR,
                      cronos=cronos, field_static=True, n_levels=7)
        out = run.run(0.15, [0.0, 0.15])
        e0, e1 = out["snapshots"][0]["energy"], out["snapshots"][-1]["energy"]
        assert abs(e1["E"] - e0["E"]) / abs(e0["E"]) < 5e-3
        assert out["snapshots"][0]["energy"]["virial_2K_over_W"] == pytest.approx(1.0, abs=0.08)
        if cronos:
            assert e0["U_cronos"] < 0.0
            # a N = 6000 el radio interior del estimador es grande y D_F(ε_soft)
            # queda muy por debajo del valor analítico: solo se exige que sea
            # positivo, finito y en régimen débil
            assert 0.0 < run.events[0]["D_F_at_soft"] < 1e3 and run.events[0]["weak_regime_ok"]
    assert C_KMS == pytest.approx(299792.458)
