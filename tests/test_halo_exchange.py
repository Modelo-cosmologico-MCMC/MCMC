"""Q(z) sumado sobre los halos y todas las posibilidades del brazo de dos sentidos (documento vivo del 24-oct, PR-65;
decisión del autor del 24-oct): las cifras del documento con su propio modelo, el catálogo completo de combinaciones con
sus consecuencias, el océano libre que apaga las leyes de borde, la normalización declarada de T4, el barrido que se
alimenta del contacto y la cota del objeto crítico."""

import numpy as np
import pytest

from cosmology.halo_exchange import (
    DOC_SETUP,
    ESCAPE,
    LAW_ONTOLOGY,
    TRAPPING,
    Setup,
    c200,
    combine,
    delta_today,
    f_h,
    law_profiles,
    mass_weights,
    two_way_catalogue,
)


@pytest.fixture(scope="module")
def doc_profiles():
    return law_profiles(DOC_SETUP)


@pytest.fixture(scope="module")
def full_profiles():
    return law_profiles(Setup(n_z=121, n_mass=40))


def test_halo_fraction_and_free_ocean_of_the_document(doc_profiles):
    assert f_h(0.0, 1e8) == pytest.approx(0.576, abs=5e-4) and f_h(0.0, 1e8, "PS") == pytest.approx(0.774, abs=5e-4)
    z = doc_profiles["z"]
    assert float(np.interp(1.0, z, doc_profiles["P"])) == pytest.approx(0.428, abs=5e-4)
    Ms, w = mass_weights(0.0, Setup())
    assert float(np.sum(w)) == pytest.approx(f_h(0.0, 1e8), rel=2e-3)          # los pesos discretos suman f_h


@pytest.mark.parametrize("constants, expected", [({"T1": 0.3}, 0.1020), ({"T2": 0.3}, 0.0624), ({"T3": 0.3}, 0.0186),
                                                 ({"T1": 0.3, "E1": 0.3}, -0.1151), ({"T2": 0.3, "E3": 0.3}, -0.0227),
                                                 ({"T3": 0.3, "E2": 1.0}, 0.0071), ({"Tc": 1.0, "E2": 1.0}, 0.0455),
                                                 ({"Tc": 0.5, "E2": 1.0}, 0.0176), ({"Tc": 0.2, "E2": 1.0}, 0.0001)])
def test_the_documents_background_table(doc_profiles, constants, expected):
    q = combine(doc_profiles, constants, sweep_feeds=False)
    assert delta_today(q, doc_profiles["z"]) == pytest.approx(expected, abs=1e-4)


def test_sweep_fed_by_the_contact_and_linearity(doc_profiles):
    z = doc_profiles["z"]
    simple = delta_today(combine(doc_profiles, {"T3": 0.3, "Tc": 1.0, "E2": 1.0}, sweep_feeds=False), z)
    fed = delta_today(combine(doc_profiles, {"T3": 0.3, "Tc": 1.0, "E2": 1.0}), z)
    assert simple == pytest.approx(0.0625, abs=1e-4) and fed > simple                 # la ley de borde completa
    one, two = (delta_today(combine(doc_profiles, {"T1": x}), z) for x in (0.05, 0.10))
    assert two == pytest.approx(2.0 * one, rel=2e-2)                                  # casi lineal en q


def test_catalogue_covers_every_two_way_combination():
    cat = two_way_catalogue()
    assert len(cat) == 24 and len({c["name"] for c in cat}) == 24
    labels = {c["name"]: c["label"] for c in cat}
    assert labels["Tc+E2"] == "contacto" and labels["T3+Tc+E2"].startswith("barrido y contacto")
    assert set(TRAPPING) | set(ESCAPE) == set(LAW_ONTOLOGY)
    for law, o in LAW_ONTOLOGY.items():
        assert o["consecuencia"] and o["encaje"] and isinstance(o["se_apaga_con_las_burbujas"], bool), law
    assert not LAW_ONTOLOGY["T1"]["mecanismo"] and not LAW_ONTOLOGY["E1"]["mecanismo"] and LAW_ONTOLOGY["E2"]["constante"] is None


def test_border_laws_switch_off_when_bubbles_fill_space(full_profiles):
    p = full_profiles
    late, early = p["z"] <= 0.5, p["z"] >= 5.0
    assert float(p["P"][-1]) < 1e-3 < float(p["P"][0])
    for law, fraction in (("Tc", 1e-2), ("E2", 1e-2), ("E3", 0.1), ("acc", 0.1)):           # e^{−N_b} frente a (1+z)³
        assert float(np.max(np.abs(p[law][early]))) < fraction * float(np.max(np.abs(p[law])))
    assert float(np.max(np.abs(p["Tc"][late]))) > 0.05
    assert float(p["T1"][-1]) > 0.01 and float(p["E1"][-1]) == 1.0                    # la ec. 7 tal cual no se apaga


def test_t4_normalization_concentration_and_e4_bound(full_profiles):
    p = full_profiles
    assert float(p["T4"][0]) == pytest.approx(float(p["T2"][0]), rel=1e-12)           # normalización declarada
    assert c200(1e12 / 0.674, 0.0, Setup()) == pytest.approx(10.0 ** 0.905, rel=1e-12) and c200(1e14, 1.0, DOC_SETUP) == 10.0
    assert abs(delta_today(combine(p, {"E4": 1.0}), p["z"])) < 1e-4                   # el objeto crítico no mueve el fondo
    assert 0.15 < float(p["xi_eq"][0]) < 0.25                                         # equilibrio del contacto, NFW prolongado
