"""Radio de aceleración nula de la ECV (documento vivo del 24-oct, PR-62): la aceleración neta se anula en R₀ y el
potencial tiene allí su máximo, la ECV dentro de R₀ es M/2, R₀/r200 no depende de la masa, el R₀ autoconsistente del NFW
prolongado y z_acc, con las cifras del documento."""

import math

import numpy as np
import pytest

from dynamics.ecv_border import (
    PLANCK18,
    border_radius,
    ecv_mass_inside,
    net_acceleration,
    potential,
    r0_over_r200,
    r0_selfconsistent,
    rho_ecv,
    z_acceleration,
    zero_acceleration_radius,
)
from dynamics.mcv_equilibrium import nfw_halo


@pytest.mark.parametrize("M", [1e8, 1e12, 1e15])
def test_zero_acceleration_barrier_and_half_mass_of_ecv(M):
    R0 = zero_acceleration_radius(M)
    a = net_acceleration([0.99 * R0, R0, 1.01 * R0], M)
    assert a[0] < 0.0 < a[2] and abs(a[1]) < 1e-12 * abs(a[0])           # hacia dentro antes, hacia fuera después
    r = R0 * np.array([0.98, 0.99, 1.0, 1.01, 1.02])
    assert int(np.argmax(potential(r, M))) == 2                             # cima de la barrera
    assert ecv_mass_inside(R0) == pytest.approx(M / 2.0, rel=1e-12)


def test_r0_over_r200_is_mass_independent():
    assert r0_over_r200(0.0) == pytest.approx(5.27, abs=5e-3)
    assert r0_over_r200(1.0) == pytest.approx(7.76, abs=5e-3)
    for M in (1e9, 1e12, 1e15):
        b = border_radius(M, 10.0)
        assert b["R0_M200"] == pytest.approx(zero_acceleration_radius(M), rel=1e-12)
        assert b["R0_M200"] / b["r200"] == pytest.approx(r0_over_r200(0.0), rel=1e-12)


@pytest.mark.parametrize("c, x_self, mass_ratio", [(5, 37.0, 2.78), (10, 68.3, 2.19), (15, 99.0, 1.97), (20, 129.4, 1.85)])
def test_selfconsistent_radius_of_the_extended_nfw(c, x_self, mass_ratio):
    x = r0_selfconsistent(c)
    assert x == pytest.approx(x_self, abs=0.05) and x > c * r0_over_r200(0.0)
    b = border_radius(1e12, c)
    assert b["M_R0_over_M200"] == pytest.approx(mass_ratio, abs=5e-3)
    M_R0 = b["M_R0_over_M200"] * 1e12                                       # la masa del NFW prolongado fija su propio R₀
    assert zero_acceleration_radius(M_R0) == pytest.approx(b["R0_self"], rel=1e-9)


def test_reference_halo_and_acceleration_epoch():
    b = border_radius(1e12, 10.0)
    assert b["r200"] == pytest.approx(212.0, abs=0.5) and b["R0_self"] == pytest.approx(1446.0, abs=1.0)
    assert b["r200"] == pytest.approx(nfw_halo(1e12, 10.0, H0=100.0 * PLANCK18["h"])["r200"], rel=1e-12)
    assert z_acceleration() == pytest.approx(0.632, abs=5e-4)
    # la densidad media de una burbuja es 2ρ_E: con toda la materia en burbujas, su volumen es el del universo en z_acc
    rho_m_zacc = PLANCK18["Omega_m"] * (1.0 + z_acceleration()) ** 3 * rho_ecv() / PLANCK18["Omega_L"]
    assert rho_m_zacc == pytest.approx(2.0 * rho_ecv(), rel=1e-12)
    assert rho_ecv() == pytest.approx(86.4, abs=0.05) and math.isclose(PLANCK18["Omega_m"] + PLANCK18["Omega_L"], 1.0)
