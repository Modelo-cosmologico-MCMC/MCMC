"""Tests estructurales del pipeline 5E (fixtures SINTÉTICOS — los
datos reales pertenecen al análisis reproducible, no a la unit suite;
ningún test requiere red).
"""

import dataclasses
import json

import numpy as np
import pytest

from dynamics.disc_cronos import v_cronos_sq
from dynamics.sculptor_transfer import frozen_config
from dynamics.sparc_crossfalsification import (
    aggregate_summary,
    chi_square,
    count_negative_total,
    decide_verdict_arm,
    per_galaxy_row,
    predeclared_subsets,
    predict_baryonic_curve,
    predict_cronos_fixed_A,
    predict_total_curve,
    shape_diagnostics,
)
from dynamics.sparc_data import validate_catalogue

CFG = frozen_config()
R = np.linspace(200.0, 12000.0, 60)          # pc — fixture sintético
RD = 2000.0
SIGMA0 = 200.0


def _cfg_with_A(a):
    return dataclasses.replace(CFG, A_sculptor=a)


def test_cronos_zero_limit():
    """A → 0 ⟹ v_model ≡ v_bar exacto (recuperación)."""
    vb2 = predict_baryonic_curve(30.0 * np.ones_like(R),
                                 80.0 * np.ones_like(R),
                                 np.zeros_like(R), 0.5, 0.7)
    vc2 = predict_cronos_fixed_A(R, SIGMA0, RD, _cfg_with_A(0.0))
    assert np.all(vc2 == 0.0)
    assert np.array_equal(predict_total_curve(vb2, vc2), np.sqrt(vb2))


def test_velocity_squared_composition():
    """Cuadratura, no suma lineal: v_model² = v_bar² + v_cronos²
    exacto, y v_model < v_bar + v_cronos donde ambas > 0."""
    vb2 = np.full_like(R, 100.0 ** 2)
    vc2 = predict_cronos_fixed_A(R, SIGMA0, RD, CFG)
    vm = predict_total_curve(vb2, vc2)
    assert np.allclose(vm ** 2, vb2 + vc2, rtol=1e-12)
    both = vc2 > 0.0
    assert np.all(vm[both] < np.sqrt(vb2[both]) + np.sqrt(vc2[both]))


def test_units():
    """Anclaje dimensional: la clase 5E-A reproduce el término del 5C
    en (km/s)² para un caso pinnado, y χ² es adimensional e invariante
    ante cambio de unidades coherente (v y e escalados igual)."""
    vc2 = predict_cronos_fixed_A(np.array([2000.0]), 500.0, 2000.0,
                                 _cfg_with_A(1e-9))
    ref = v_cronos_sq(np.array([2000.0]), 500.0, 2000.0,
                      CFG.zeta_disc, 1e-9)
    assert vc2[0] == pytest.approx(float(ref[0]), rel=1e-12)
    v_obs = np.array([100.0, 120.0])
    v_mod = np.array([95.0, 125.0])
    e = np.array([5.0, 10.0])
    c1 = chi_square(v_mod, v_obs, e)
    c2 = chi_square(10.0 * v_mod, 10.0 * v_obs, 10.0 * e)
    assert c1 == pytest.approx(c2, rel=1e-12)
    assert c1 == pytest.approx(1.0 + 0.25, rel=1e-12)


def test_exponential_disc_recovers_5c():
    """La clase 5E-A ES la del 5C: identidad exacta sobre un disco
    exponencial sintético con la A congelada real."""
    vc2 = predict_cronos_fixed_A(R, SIGMA0, RD, CFG)
    ref = v_cronos_sq(R, SIGMA0, RD, CFG.zeta_disc, CFG.A_sculptor)
    assert np.array_equal(vc2, np.asarray(ref))


def test_inner_outer_shape_ratio():
    """La identidad ×5 de la clase, medida por el diagnóstico: en un
    disco exponencial sintético, ratio_x5 ≈ 1/sqrt(0.0404) ≈ 4.97."""
    Rfine = np.linspace(50.0, 10.0 * RD, 4000)
    vc = np.sqrt(predict_cronos_fixed_A(Rfine, SIGMA0, RD, CFG))
    vb2 = np.full_like(Rfine, 100.0 ** 2)
    d = shape_diagnostics(Rfine, RD, vc, np.sqrt(vb2) + 20.0, vb2)
    assert d["ratio_x5"] == pytest.approx(4.97, abs=0.05)
    assert d["B_inner"] > 0.0
    assert np.isfinite(d["F_outer"])


def test_shape_diagnostics_out_of_range_is_nan():
    """Si 4R_d cae fuera del rango medido, ratio_x5 es NaN (no se
    extrapola en silencio)."""
    Rshort = np.linspace(200.0, 2.0 * RD, 30)
    vc = np.sqrt(predict_cronos_fixed_A(Rshort, SIGMA0, RD, CFG))
    d = shape_diagnostics(Rshort, RD, vc, 100.0 * np.ones_like(Rshort),
                          np.full_like(Rshort, 90.0 ** 2))
    assert np.isnan(d["ratio_x5"])


def test_sparc_schema():
    """El validador del contrato de ingesta, con fixtures sintéticos:
    acepta lo sano y rechaza NaN, radios ≤ 0, errores ≤ 0 y
    duplicados."""
    good = validate_catalogue(np.array([1.0, 2.0, 1.0]),
                              np.array([50.0, 60.0, 40.0]),
                              np.array([5.0, 5.0, 4.0]),
                              np.array(["g1", "g1", "g2"]))
    assert good == {"n_points": 3, "n_galaxies": 2}
    with pytest.raises(ValueError, match="NaN"):
        validate_catalogue(np.array([1.0, np.nan]), np.ones(2),
                           np.ones(2), np.array(["a", "a"]))
    with pytest.raises(ValueError, match="positivos"):
        validate_catalogue(np.array([0.0, 1.0]), np.ones(2),
                           np.ones(2), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="errores"):
        validate_catalogue(np.array([1.0, 2.0]), np.ones(2),
                           np.array([1.0, 0.0]), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="duplicados"):
        validate_catalogue(np.array([1.0, 1.0]), np.ones(2),
                           np.ones(2), np.array(["a", "a"]))


def test_checksum_manifest(tmp_path):
    """La maquinaria de manifest: solo se escribe tras un fichero
    real, con sha256 y tamaño correctos (fixture sintético local —
    sin red)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "scripts"))
    import download_data as dd
    f = tmp_path / "synthetic.dat"
    f.write_text("# fixture sintético, no observacional\n1 2 3\n")
    digest = dd._sha256(f)
    dd.write_manifest("sparc", "https://example.invalid/x", f, digest)
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man[0]["sha256"] == digest
    assert man[0]["file_size"] == f.stat().st_size
    assert "Lelli" in man[0]["citation"]


def test_per_galaxy_row_locks_A_fitted_false():
    """Toda fila del test principal lleva A_fitted = False y el
    model_scope inequívoco; A_used es la A congelada."""
    d = {"B_inner": 0.5, "F_outer": 0.1, "ratio_x5": 5.0}
    row = per_galaxy_row("SYN-1", 1, 20, 150.0, 2000.0, 30.0, 45.0,
                         d, CFG, "5E-A_exponential")
    assert row["A_fitted"] is False
    assert row["A_used"] == CFG.A_sculptor
    assert row["delta_chi2"] == pytest.approx(15.0)
    assert row["model_scope"] == "5E-A_exponential"


def test_aggregate_summary_and_bootstrap_deterministic():
    """Agregados preinscritos: fracciones, mediana, suma y bootstrap
    reproducible con la semilla congelada."""
    d = np.array([-2.0, -1.0, 0.5, 3.0, 4.0])
    n = np.array([10, 12, 8, 30, 20])
    s1 = aggregate_summary(d, n, CFG)
    s2 = aggregate_summary(d, n, CFG)
    assert s1 == s2
    assert s1["frac_improved"] == pytest.approx(0.4)
    assert s1["frac_worsened"] == pytest.approx(0.6)
    assert s1["median_delta_chi2"] == pytest.approx(0.5)
    assert s1["sum_delta_chi2"] == pytest.approx(4.5)


def test_published_numbers_regression():
    """Candado: la A congelada publicada en la preinscripción."""
    assert CFG.A_sculptor == pytest.approx(6.201589e-7, rel=1e-5)
    assert CFG.A_sculptor_sensitivity[0] == pytest.approx(1.853e-6,
                                                          rel=2e-3)
    assert CFG.A_sculptor_sensitivity[1] == pytest.approx(3.186e-7,
                                                          rel=2e-3)


def test_report_reproducibility(tmp_path):
    """El artefacto DATA_UNAVAILABLE es determinista: dos ejecuciones
    producen bytes idénticos (sin timestamps dentro del informe)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "scripts"))
    from run_front5e_sparc import write_unavailable_verdict
    a = write_unavailable_verdict(tmp_path / "a")
    b = write_unavailable_verdict(tmp_path / "b")
    assert (tmp_path / "a" / "report.md").read_bytes() \
        == (tmp_path / "b" / "report.md").read_bytes()
    assert a == b


def test_negative_total_clip_is_declared_and_counted():
    """El condicional del clip, expuesto: gas con signo puede dar
    v_bar² + v_cronos² < 0; v_model = 0 ahí (elección preinscrita,
    idéntica en M0 y M1) y el punto se CUENTA, no se descarta."""
    vb2 = predict_baryonic_curve(np.array([-30.0]), np.array([5.0]),
                                 np.array([0.0]), 0.5, 0.7)
    assert vb2[0] < 0.0
    vm = predict_total_curve(vb2, np.array([100.0]))
    assert vm[0] == 0.0
    assert count_negative_total(vb2, np.array([100.0])) == 1
    assert count_negative_total(np.array([50.0]), np.array([1.0])) == 0


def test_shape_diagnostics_rejects_unsorted_radii():
    """La guardia de monotonía: radios no crecientes lanzan error en
    vez de dejar que np.interp devuelva basura en silencio."""
    Rbad = np.array([2000.0, 1000.0, 3000.0])
    with pytest.raises(ValueError, match="crecientes"):
        shape_diagnostics(Rbad, RD, np.ones(3), np.ones(3), np.ones(3))


def test_validate_catalogue_rejects_nonmonotonic_per_galaxy():
    """El validador implementa lo que su docstring promete: radios
    decrecientes DENTRO de una galaxia se rechazan."""
    with pytest.raises(ValueError, match="crecientes"):
        validate_catalogue(np.array([2.0, 1.0]), np.ones(2),
                           np.ones(2), np.array(["g1", "g1"]))


def test_decide_verdict_arm_three_arms():
    """La regla de decisión preinscrita, por sus tres brazos, con
    resúmenes sintéticos."""
    base = {"frac_improved": 0.2, "median_delta_chi2": 5.0,
            "bootstrap": {"median_CI95": [2.0, 8.0]}}
    assert decide_verdict_arm(base) == "systematic_worsening"
    sup = {"frac_improved": 0.8, "median_delta_chi2": -4.0,
           "bootstrap": {"median_CI95": [-7.0, -1.0]}}
    assert decide_verdict_arm(sup) == "support"
    ind = {"frac_improved": 0.5, "median_delta_chi2": 0.3,
           "bootstrap": {"median_CI95": [-1.0, 2.0]}}
    assert decide_verdict_arm(ind) == "indeterminate"


def test_predeclared_subsets_never_use_delta_chi2():
    """Los subconjuntos preinscritos se calculan SOLO de observables:
    ninguna referencia a delta_chi2 en la selección (y el candado de
    fuente lo vigila)."""
    rows = [
        {"quality_flag": 1, "Sigma_or_equivalent": 300.0,
         "gas_dominated": False, "delta_chi2": 99.0},
        {"quality_flag": 3, "Sigma_or_equivalent": 50.0,
         "gas_dominated": True, "delta_chi2": -99.0},
    ]
    sub = predeclared_subsets(rows, sigma_median=150.0)
    assert sub["full_sample"] == [0, 1]
    assert sub["quality_sample"] == [0]
    assert sub["HSB_subset"] == [0]
    assert sub["LSB_subset"] == [1]
    assert sub["gas_dominated_subset"] == [1]
