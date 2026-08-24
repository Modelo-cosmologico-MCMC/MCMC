"""Tests del vector teórico DESI y las convenciones de ajuste
(offline; datos commiteados)."""

import numpy as np
import pytest

from cosmology.desi_background_fit import (
    C_KMS,
    E_of_z,
    chi2_at,
    log_prob_lcdm,
    log_prob_mcmc,
    predict_desi_dr2_vector,
)
from cosmology.desi_bao import load_desi_dr2_all

DATA = load_desi_dr2_all()


def test_vector_order_pinned_to_release():
    """El orden de las 13 componentes es EXACTAMENTE el del fichero
    ingerido (bgs DV primero; después pares DM,DH por bin)."""
    z, quant, _, bins, _ = DATA
    m = predict_desi_dr2_vector(0.3, 14500.0, data=DATA)
    assert m.shape == (13,)
    assert quant[0] == "DV_over_rs" and bins[0] == "bgs"
    # identidades internas del vector, componente a componente:
    for i, (zi, q) in enumerate(zip(z, quant)):
        Ei = float(E_of_z(zi, 0.3))
        if q == "DH_over_rs":
            assert m[i] == pytest.approx(C_KMS / (Ei * 14500.0),
                                         rel=1e-12)


def test_dv_identity():
    """DV³ = z·DM²·DH exacto dentro del vector."""
    z, quant, _, _, _ = DATA
    m = predict_desi_dr2_vector(0.32, 15000.0, data=DATA)
    lookup = {(round(zi, 3), q): v for zi, q, v in zip(z, quant, m)}
    dv = lookup[(0.295, "DV_over_rs")]
    # bgs solo publica DV; verificamos contra DM/DH recalculados:
    Ei = float(E_of_z(0.295, 0.32))
    DH = C_KMS / (Ei * 15000.0)
    # referencia INDEPENDIENTE (scipy.quad adaptativo) — la versión
    # anterior duplicaba el trapecio del integrador viejo dentro del
    # test y se rompió, correctamente, al corregirlo a O(h⁴):
    from scipy.integrate import quad
    integral, _ = quad(lambda x: 1.0 / float(E_of_z(x, 0.32)),
                       0.0, 0.295, epsabs=1e-14, epsrel=1e-13)
    DM = C_KMS / 15000.0 * integral
    assert dv == pytest.approx((0.295 * DM ** 2 * DH) ** (1 / 3),
                               rel=1e-10)


def test_integration_accuracy_vs_quad():
    """El integrador de PRODUCCIÓN (Simpson O(h⁴) + cola GL3 dentro de
    predict_desi_dr2_vector) clava scipy.quad a mejor que 1e-12
    relativo en DM, en TODOS los z_eff del release.

    La versión anterior de este test reconstruía el trapecio RETIRADO
    dentro del test y solo lo comprobaba en z = 2.33 (uno de sus
    puntos favorables: en z = 0.295 el esquema viejo daba 7.3e-8 —
    hallazgo confirmado de la revisión del crosscheck); ahora se mide
    el camino real del vector."""
    from scipy.integrate import quad
    Om, H0rd = 0.31, 10200.0
    z_eff, quant, _, _, _ = DATA
    # fiducial Y el rincón del soporte (ε=−0.05, z_trans=1: la
    # transición dentro del rango — donde el trapecio viejo era peor):
    for eps, zt in [(0.0, 8.9), (-0.05, 1.0)]:
        v = predict_desi_dr2_vector(Om, H0rd, eps=eps, z_trans=zt,
                                    data=DATA)
        checked = 0
        for zi, q, vi in zip(z_eff, quant, v):
            if q != "DM_over_rs":
                continue
            ref, _ = quad(lambda zz: 1.0 / float(
                E_of_z(zz, Om, eps, zt)), 0.0, float(zi),
                limit=200, epsabs=1e-14, epsrel=1e-13)
            assert abs(vi / (C_KMS / H0rd * ref) - 1.0) < 1e-12, \
                (zi, eps)
            checked += 1
        assert checked == 6      # los 6 bins con DM del release


def test_lcdm_recovery_eps_zero():
    """ε = 0 ⟹ el vector MCMC ES el ΛCDM exacto (Prop. A.1 en el
    espacio de observables DESI)."""
    a = predict_desi_dr2_vector(0.3, 14500.0, data=DATA)
    b = predict_desi_dr2_vector(0.3, 14500.0, eps=0.0, z_trans=5.0,
                                data=DATA)
    assert np.array_equal(a, b)


def test_rd_is_common_calibration():
    """El contrato r_d: H0rd entra IDÉNTICAMENTE en ambos modelos —
    reescalar H0rd reescala DM/rd y DH/rd por igual (1/x) en ΛCDM y
    en MCMC, y el código no tiene ningún camino para tratarlo
    distinto (mismo argumento de la misma función)."""
    for eps in (0.0, 0.05):
        m1 = predict_desi_dr2_vector(0.3, 14000.0, eps=eps, data=DATA)
        m2 = predict_desi_dr2_vector(0.3, 28000.0, eps=eps, data=DATA)
        z, quant, _, _, _ = DATA
        sel = [i for i, q in enumerate(quant) if q != "DV_over_rs"]
        assert np.allclose(m1[sel] / m2[sel], 2.0, rtol=1e-12)


def test_chi2_reasonable_at_plausible_point():
    """Sanidad (no benchmark): en un punto plausible el χ² es finito
    y de orden N_data."""
    c = chi2_at((0.2975, 10200.0), "lcdm", DATA)
    assert np.isfinite(c) and 0.0 < c < 100.0


def test_log_probs_finite_and_prior_bounds():
    assert np.isfinite(log_prob_lcdm((0.3, 14500.0), DATA))
    assert log_prob_lcdm((0.9, 14500.0), DATA) == -np.inf
    assert np.isfinite(log_prob_mcmc((0.3, 14500.0, 0.01, 8.9), DATA))
    assert log_prob_mcmc((0.3, 14500.0, 0.2, 8.9), DATA) == -np.inf
