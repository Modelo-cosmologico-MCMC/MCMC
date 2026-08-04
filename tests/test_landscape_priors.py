"""Tests de la sensibilidad de δ_H a los priors del paisaje (v35.1)."""

import numpy as np
import pytest

from core.landscape_priors import (
    DOMAINS,
    delta_H_of,
    domain_bound,
    sample_shapes_prior,
    sensitivity_row,
)

N = 2000


def test_domain_bound_analytic():
    """La cota es λ_H/√(b_max²−4·C0_lo·m_lo²) y rechaza dominios vacíos."""
    b = domain_bound((0.5, 2.0), b_extended=True)
    assert abs(b - 0.130 / np.sqrt(16.0 - 4.0 * 0.5 * 0.25)) < 1e-12
    with pytest.raises(ValueError):
        domain_bound((2.0, 2.0), b_extended=False)   # D ≤ 0 en todo él


def test_sampled_min_respects_bound():
    """Ningún paisaje muestreado baja de la cota analítica (todo prior)."""
    for prior in ("uniform", "loguniform", "normal"):
        shapes = sample_shapes_prior(N, seed=3, prior=prior)
        d_H = delta_H_of(shapes)
        assert d_H.min() >= domain_bound((0.5, 2.0), True)


def test_0012_unreachable_in_all_domains():
    """LA AFIRMACIÓN ROBUSTA: la cota de dominio supera 0.012 en todos
    los dominios O(1) ensayados, con y sin extensión de b̄ — la
    inaccesibilidad del 0.012 no depende del prior (es analítica)."""
    for dom in DOMAINS:
        for ext in (True, False):
            assert domain_bound(dom, ext) > 0.012


def test_priors_shift_median_but_order_survives():
    """LO DEPENDIENTE DEL PRIOR: la mediana se mueve (log-uniforme la
    sube: favorece b̄ pequeño → D menor → δ_H mayor), pero el orden de
    magnitud pocas×10⁻² sobrevive en los tres priors."""
    meds = {}
    for prior in ("uniform", "loguniform", "normal"):
        r = sensitivity_row(N, seed=5, prior=prior, o1_range=(0.5, 2.0))
        meds[prior] = r["median"]
        assert 0.02 < r["median"] < 0.15
    assert meds["loguniform"] > meds["uniform"]


def test_fertile_filter_is_marginal_for_delta_H():
    """El filtro de fertilidad apenas mueve la distribución de δ_H
    (δ_H no depende de ē ni de γR; solo correlación débil vía m̄)."""
    r_on = sensitivity_row(N, seed=7, prior="uniform",
                           o1_range=(0.5, 2.0), fertile_filter=True)
    r_off = sensitivity_row(N, seed=7, prior="uniform",
                            o1_range=(0.5, 2.0), fertile_filter=False)
    assert abs(r_on["median"] - r_off["median"]) < 0.01


def test_no_b_extension_shifts_higgs_to_lower_tail():
    """Sin la extensión de b̄ el dominio empuja δ_H hacia arriba
    (mediana ~0.11) y el valor del empalme 0.0581 cae en la cola
    inferior — la configuración del muestreo IMPORTA y se declara."""
    r = sensitivity_row(N, seed=9, prior="uniform", o1_range=(0.5, 2.0),
                        b_extended=False)
    assert r["median"] > 0.09
    assert 0.0581 < r["p5"]


def test_deterministic():
    """Misma semilla → mismas estadísticas (reproducible)."""
    a = sensitivity_row(500, seed=11, prior="normal", o1_range=(0.5, 2.0))
    b = sensitivity_row(500, seed=11, prior="normal", o1_range=(0.5, 2.0))
    assert a["median"] == b["median"] and a["min"] == b["min"]
