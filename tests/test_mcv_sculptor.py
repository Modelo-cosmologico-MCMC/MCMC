"""La MCV (c) en Sculptor (MCV-b, bloque P0 del autor del 23-oct): la relación masa–concentración de la previa, la rejilla
de la envoltura, el filtro de ramas del banco (dominio y r₁ < r200), σ_los(R) frente a una cuadratura independiente, el
prefiltro del empalme frente a la búsqueda exhaustiva, las letras y sus puertas, un extremo a extremo con un banco
pequeño y bins SINTÉTICOS (ejercitan el instrumento; no son contraste), el analizador que falla cerrado y el candado de
la preinscripción."""

import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import dynamics.mcv_sculptor as ms
from dynamics.mcv_equilibrium import dm_profile, jeans_match, nfw_halo, plummer_mass
from dynamics.mcv_sculptor import (
    bank_point,
    c200_dm14,
    envelope,
    envelope_grid,
    in_prior,
    letters,
    on_box_edge,
    sigma_los_model,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-10-23_mcv_sculptor"


def _script():
    spec = importlib.util.spec_from_file_location("run_mcv_sculptor", REPO / "scripts" / "run_mcv_sculptor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RS = _script()
SPEC = RS.SPEC


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_c200_relation_and_envelope_grid():
    rel = SPEC["c_relation"]
    assert c200_dm14(1e12 / rel["h"], rel) == pytest.approx(10 ** 0.905)
    assert c200_dm14(3e9, rel) == pytest.approx(15.0, abs=0.1)          # la envoltura del dato previo es la mediana
    assert envelope(9.0, 1.0, rel)[1] == pytest.approx(c200_dm14(1e9, rel) * 10 ** 0.11)
    g = envelope_grid(SPEC)
    assert len(g) == len(set(g)) == 295 and sum(in_prior(m, k, SPEC) for m, k in g) == 153
    assert on_box_edge(7.5, 0.0, SPEC) and on_box_edge(9.0, -5.0, SPEC) and not on_box_edge(9.0, 4.0, SPEC)
    assert len(RS._tasks(SPEC)) == 15 * 3 * 295


def test_bank_point_keeps_only_branches_inside_the_domain_and_r200(monkeypatch):
    def fake(M200, c, som, **kw):
        mk = [(10.0, 50.0, 1.0, "con núcleo"), (12.0, 2e10, 1.0, "densa"), (11.0, 1e3, 12.0, "densa"), (0.3, 80.0, 1.0, "densa")]
        return {"halo": {"r200": 10.0}, "branches": [{"sigma0": s, "u": u, "rho0": 1e7 * u, "r1": r1, "branch": b} for s, u, r1, b in mk]}
    monkeypatch.setattr(ms, "jeans_match", fake)
    p = bank_point(1.0, 2.0, 9.0, 0.0, SPEC)
    assert p["branches"] == [[10.0, 50.0, 5e8, 1.0, "con núcleo"]]
    assert (p["dropped_outside_domain"], p["dropped_r1_beyond_r200"]) == (2, 1)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_sigma_los_matches_an_independent_quadrature():
    p = bank_point(0.3, 2.0, 9.0, 0.0, SPEC)
    br = p["branches"][0]
    R = [50.0, 300.0, 900.0]
    model = sigma_los_model(p, br, [0.0], R, SPEC)[0.0]
    a, Ms, G = SPEC["plummer_a_kpc"], 2.0 * SPEC["L_V_Lsun"], 4.30091e-6
    match = {"halo": nfw_halo(p["M200"], p["c"], SPEC["H0"]), "plummer": {"M": Ms, "a": a}, "rho_E": 0.0}
    rg = np.geomspace(1e-5, 40.0, 4000)
    Mg = dm_profile(match, {"sigma0": br[0], "rho0": br[2], "r1": br[3]}, rg)["M_M"] + plummer_mass(rg, Ms, a)

    def nu(r):
        return (1.0 + (r / a) ** 2) ** -2.5

    def nu_s2(r):
        return quad(lambda x: nu(x) * G * float(np.interp(x, rg, Mg)) / x ** 2, r, 31.2, limit=400)[0]

    def s_los(Rk):
        Rk /= 1000.0
        num = quad(lambda z: nu_s2(math.hypot(Rk, z)), 0.0, 31.2, limit=200)[0]
        return math.sqrt(num / quad(lambda z: nu(math.hypot(Rk, z)), 0.0, 31.2, limit=200)[0])
    assert list(model) == pytest.approx([s_los(r) for r in R], rel=2e-3)


def test_prefilter_finds_the_same_branches_inside_the_grid():
    for args, pl in (((1e9, 15.0, 50.0), None), ((1e9, 17.0, 20.0), {"M": 5.2e6, "a": 0.26}),
                     ((1e9, 20.0, 5.0), {"M": 5.2e6, "a": 0.26})):
        full = [(b["sigma0"], b["u"], b["branch"]) for b in jeans_match(*args, plummer=pl)["branches"] if b["u"] <= 1e10]
        fast = [(b["sigma0"], b["u"], b["branch"]) for b in jeans_match(*args, plummer=pl, prefilter=True)["branches"] if b["u"] <= 1e10]
        assert len(full) == len(fast) >= 2
        for x, y in zip(sorted(full), sorted(fast)):
            assert x[0] == pytest.approx(y[0], rel=1e-8) and x[1] == pytest.approx(y[1], rel=1e-8) and x[2] == y[2]


def _rows(fits):
    """Filas sintéticas: χ² alto en todo salvo las combinaciones de `fits` (som, rama, en previas, borde)."""
    rows = []
    for som in SPEC["sigma_over_m_grid"]:
        for branch in ("con núcleo", "densa"):
            for prior in (True, False):
                good = (som, branch, prior) in fits
                rows.append({"som": som, "ups": 2.0, "logM": 9.0 if prior else 11.0, "k": 0.0, "branch": branch, "beta": 0.0,
                             "chi2": 4.0 if good else 60.0, "in_prior": prior})
    return rows


def test_letters_and_gates():
    assert letters(_rows({(0.3, "con núcleo", True)}), 9, SPEC)["letter"] == "A"
    assert letters(_rows({(0.3, "densa", True)}), 9, SPEC)["letter"] == "B"
    assert letters(_rows({(0.3, "con núcleo", False)}), 9, SPEC)["letter"] == "B"
    b20 = letters(_rows({(20.0, "con núcleo", True)}), 9, SPEC)
    assert b20["letter"] == "B" and b20["fits_any_sigma_over_m"] == [20.0]
    assert letters(_rows(set()), 9, SPEC)["letter"] == "C"
    edge = _rows(set())
    edge[0]["chi2"], edge[0]["logM"] = 30.0, 7.5
    assert letters(edge, 9, SPEC)["letter"] == "INDETERMINADO"
    assert letters(_rows({(0.3, "con núcleo", True)}), 3, SPEC)["letter"] == "INDETERMINADO"
    assert letters(_rows({(50.0, "con núcleo", True)}), 9, SPEC)["letter"] == "B"


def test_end_to_end_with_a_small_bank_and_synthetic_bins():
    """Bins sintéticos generados con un punto del banco (β = 0, sin ruido): la letra es A y el punto decisivo es ese."""
    pts = [bank_point(som, ups, lm, k, SPEC) for som in (0.3, 2.0) for ups in (2.0,) for lm, k in ((9.0, 0.0), (9.5, 0.0), (11.5, 0.0))]
    truth = pts[0]
    R = [60.0, 150.0, 250.0, 380.0, 520.0, 700.0, 900.0]
    sig = sigma_los_model(truth, truth["branches"][0], [0.0], R, SPEC)[0.0]
    bins = [{"R_pc": r, "sigma_kms": float(s), "sigma_err_kms": 0.5} for r, s in zip(R, sig)]
    pre = {"spec": SPEC, "reproduction_gate": {"points": [], "sigma_los_reference": [], "R_pc": [], "tolerance_rel": 1e-6}}
    res = RS.analyze_bins(bins, pre, {"points": pts})
    assert res["verdict"] == "A" and res["exhaustive_check"]["point"] == [0.3, 2.0, 9.0, 0.0]
    assert res["letter_detail"]["per_sigma_over_m"]["0.3"]["best_A"]["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert set(res["without_letter_collapse_time"]["0.3"]["t_c_gyr"]) == {"0.45", "0.75"}


def test_analyzer_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(RS, "PREREG", tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="FALLO CERRADO"):
        RS.analyze()


@pytest.mark.skipif(not (OUT / "preregistration.json").exists(), reason="preinscripción MCV-b aún no congelada")
def test_frozen_prereg_lock(tmp_path, monkeypatch):
    pre = json.loads((OUT / "preregistration.json").read_text(encoding="utf-8"))
    assert _sha(OUT / "preregistration.json") in (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert pre["bank"]["sha256"] == _sha(OUT / "model_bank.json") and pre["prefilter_check"]["sha256"] == _sha(OUT / "prefilter_check.json")
    assert pre["prefilter_check"]["differences_in_domain"] == 0
    assert pre["observable"]["annex_sha256"] == _sha(REPO / "results" / "2026-09-21_sculptor_profile" / "binning_annex.json")
    assert pre["observable"]["local_law_prereg_sha256"] == "604f9c254c7e8283949137d23bb75fb4200b5341a369b1b89f06622ce53af002"
    assert pre["spec"] == SPEC and pre["generator_commit_must_precede_freeze"] and pre["expectation_E13"]
    assert "≤ 10" in pre["rules"]["A"] and "≤ 50" in pre["rules"]["B"] and "≤ 50" in pre["rules"]["C"]
    bank = json.loads((OUT / "model_bank.json").read_text(encoding="utf-8"))
    assert bank["spec"] == SPEC and bank["n_points"] == len(RS._tasks(SPEC)) == pre["bank"]["n_points"]
    # el analizador lee la especificación de la preinscripción, no constantes del guion
    body = inspect.getsource(RS.analyze_bins)
    assert 'pre["spec"]' in body and "SPEC" not in body.replace('pre["spec"]', "")
    # sin datos, el analizador registra el fallo cerrado
    monkeypatch.setattr(RS, "OUTDIR", tmp_path)
    assert RS.analyze() == 0
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res["verdict"] == "INDETERMINADO" and "fallo cerrado" in res["reason"]
