"""Candado del frente 2 — la preinscripción no se toca a posteriori.

Vigila: (1) que el cierre y los coeficientes del módulo canónico
coinciden con la preinscripción CONGELADA (cambiarlos tras ver el
espectro rompe aquí); (2) que la preinscripción se generó en un commit
que ya contenía la derivación (pin histórico); (3) que ningún fichero
del frente optimiza hacia s0_target (prohibición de tuning).
"""

import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PREREG = (REPO / "results" / "2026-08-19_front2_fp_beta"
          / "preregistration.json")

# ficheros del frente vigilados contra tuning:
FRONT_FILES = [
    REPO / "core" / "fokker_planck_beta.py",
    REPO / "scripts" / "run_front2_fp_beta.py",
    REPO / "scripts" / "run_front2_fp_delta0.py",
    REPO / "scripts" / "run_front2_prereg.py",
    REPO / "scripts" / "make_front2_report.py",
]
# patrones de USO real (no de mención en prosa — la promesa de la
# preinscripción cita los nombres y no debe dispararse a sí misma);
# reforzados tras la revisión (variantes de import y de optimizador):
FORBIDDEN = [
    "from scipy.optimize", "import scipy.optimize", "scipy.optimize.",
    "from scipy import optimize", "curve_fit(", "minimize(",
    "minimize_scalar(", "fsolve(", "brentq(", "brent(", "golden(",
    "root_scalar(", "fmin", "basinhopping(", "dual_annealing(",
    "least_squares(", "differential_evolution(", "dataclasses.replace",
    "replace(cl", "object.__setattr__",
]


def test_prereg_exists_and_matches_module():
    from core.fokker_planck_beta import (
        BETA_COEFFICIENTS,
        canonical_closure,
        spinodal_canonical_point,
    )
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    cl = canonical_closure()
    c = doc["closure"]
    assert c["a"] == cl.a and c["b"] == cl.b
    assert c["flow_sign"] == cl.flow_sign
    assert c["truncation"] == cl.truncation
    assert c["tau"] == cl.tau
    assert c["g_grid"] == list(cl.g_grid)
    assert c["robustness_window"] == list(cl.robustness_window)
    assert c["s0_band"] == cl.s0_band
    assert c["s0_target"] == cl.s0_target
    # la tabla de coeficientes derivados, congelada (JSON: tuplas→listas):
    assert doc["derivation"]["coefficients"] == {
        k: [list(t) for t in v] for k, v in BETA_COEFFICIENTS.items()}
    p = doc["evaluation_point"]
    assert (p["M0_sq"], p["B"], p["C0"]) == spinodal_canonical_point()


def test_prereg_values_are_the_frozen_ones():
    """Los valores CONCRETOS congelados el 19-ago: banda 10%, ventana
    [0.5, 2], 21 puntos de malla en [0.1, 10], cierre ½/½, signo +1,
    τ = 1, truncamiento cúbico. Cambiar cualquiera es romper la
    preinscripción."""
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    c = doc["closure"]
    assert c["a"] == 0.5 and c["b"] == 0.5
    assert c["flow_sign"] == 1 and c["tau"] == 1.0
    assert c["truncation"] == "cubic"
    assert c["s0_band"] == 0.10
    assert c["s0_target"] == float(np.pi / np.log(10.0))
    assert len(c["g_grid"]) == 21
    assert c["g_grid"][0] == 0.1 and c["g_grid"][-1] == 10.0
    assert 1.0 in c["g_grid"]
    assert c["robustness_window"] == [0.5, 2.0]
    assert set(doc["outcomes"].keys()) == {"A", "B", "C", "D", "rule"}


def test_prereg_commit_contains_derivation():
    """Pin histórico: el code_commit de la preinscripción contiene el
    módulo con las MISMAS β (la preinscripción no es retroactiva)."""
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    sha = doc["code_commit"]
    out = subprocess.run(
        ["git", "show", f"{sha}:core/fokker_planck_beta.py"],
        capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        import pytest
        pytest.skip("commit de la preinscripción no disponible en "
                    "este clone (shallow)")
    src = out.stdout
    assert "-8.0 * a * B - 2.0 * b * M0_sq ** 2" in src
    assert "-24.0 * a * C0 - 8.0 * b * M0_sq * B" in src
    assert "-6.0 * b * (B ** 2 + 2.0 * M0_sq * C0)" in src


def test_no_tuning_in_front_files():
    """Prohibición ejecutable: nada del frente optimiza hacia
    s0_target ni muta el cierre congelado."""
    for f in FRONT_FILES:
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in src, f"{token} en {f.name}"


def test_artifact_locks_and_recompute():
    """CANDADO del desenlace publicado (fp_beta.json): DESENLACE A —
    espectro real en el punto canónico (silla, sin cascada DSI),
    dD/dt = +28 exacto (el flujo canónico sube D, contra Obs. 8.6),
    ventana [0.5, 2] uniformemente real, único punto complejo en el
    extremo g = 10 (donde además dD/dt < 0), cuártico también real.
    Recomputa el espectro con el código vigente: si las β o el punto
    cambian, esto falla."""
    import pytest

    from core.fokker_planck_beta import (
        canonical_closure,
        spinodal_canonical_point,
        stability_matrix,
        stability_matrix_quartic,
    )
    doc = json.loads((REPO / "results" / "2026-08-19_front2_fp_beta"
                      / "fp_beta.json").read_text(encoding="utf-8"))
    assert doc["outcome"] == "A"
    c = doc["canonical"]
    assert c["complex_pair"] is False
    assert c["dD_dt"] == 28.0        # entero del álgebra: −80+36+72
    # recomputación con el código vigente (artefacto ↔ código):
    cl = canonical_closure()
    M = cl.flow_sign * stability_matrix(*spinodal_canonical_point(),
                                        a=cl.a, b=cl.b)
    eigs = np.linalg.eigvals(M)
    assert np.abs(eigs.imag).max() < 1e-12
    assert np.allclose(sorted(eigs.real),
                       sorted(c["eigenvalues_re"]), atol=1e-9)
    assert sorted(c["eigenvalues_re"])[0] == pytest.approx(-18.6603,
                                                           abs=1e-3)
    assert sorted(c["eigenvalues_re"])[2] == pytest.approx(7.6668,
                                                           abs=1e-3)
    # barrido: ventana uniforme real; único complejo en g = 10, con
    # su τ* publicado (revisión: prometido en «se publica SIEMPRE»):
    cplx = [r for r in doc["scan"] if r["complex_pair"]]
    assert len(cplx) == 1 and cplx[0]["g"] == 10.0
    assert cplx[0]["s0_spectral"] == pytest.approx(5.8661, abs=1e-3)
    assert cplx[0]["tau_star_for_lambda10"] == pytest.approx(
        0.2326, abs=1e-3)
    # dD/dt = 44 − 16g en el barrido: negativo para g > 2.75 (6/21
    # puntos), NO solo en g = 10 — lo exclusivo del borde es la
    # conjunción rotación + hundimiento (corrección de la revisión):
    assert doc["n_dD_negative"] == 6
    assert cplx[0]["dD_dt"] < 0.0
    for r in doc["scan"]:
        assert r["dD_dt"] == pytest.approx(44.0 - 16.0 * r["g"],
                                           rel=1e-9), r["g"]
        if 0.5 <= r["g"] <= 2.0:
            assert r["complex_pair"] is False, r["g"]
    # el sistemático cuártico no rescata la cascada:
    Mq = cl.flow_sign * stability_matrix_quartic(
        *spinodal_canonical_point(), 0.0, a=cl.a, b=cl.b)
    assert np.abs(np.linalg.eigvals(Mq).imag).max() < 1e-12
    assert doc["quartic_E4_0"]["complex_pair"] is False
    assert doc["tau_star_for_lambda10"] is None
    # el control de deriva prometido, publicado y dentro del régimen:
    assert doc["drift_control"]["max_rel_error"] < 0.02


def test_addendum_delta0_locks_and_recompute():
    """CANDADO de la adenda δ0 (corrección del hallazgo HIGH): la
    covarianza exacta hace del eje g el eje δ0 (g_ef = δ0⁻³); la
    familia espinodal física con el cierre canónico se complejifica
    para δ0 < δ0* ≈ 0.4961 (g* ≈ 8.189, raíz del discriminante) y en
    el punto físico δ0 = 0.1 las reglas preinscritas darían B
    (contrafactual declarado — el desenlace A preinscrito NO se
    reclasifica). Recomputa espectro y covarianza con el código
    vigente."""
    import pytest

    from core.fokker_planck_beta import (
        beta_functions,
        canonical_closure,
        g_effective,
        physical_spinodal_point,
        stability_matrix,
    )
    doc = json.loads((REPO / "results" / "2026-08-19_front2_fp_beta"
                      / "fp_beta_delta0.json").read_text("utf-8"))
    assert doc["covariance_identity_max_abs_dev"] < 1e-10
    assert doc["g_star_complexification"] == pytest.approx(8.1894,
                                                           abs=1e-3)
    assert doc["delta0_star"] == pytest.approx(0.4961, abs=1e-3)
    assert doc["counterfactual_outcome_at_physical_point"] == "B"
    phys = doc["physical_point"]
    assert phys["complex_pair"] is True
    assert phys["s0_tau1"] == pytest.approx(4.946, abs=5e-3)
    assert phys["tau_star_for_lambda10"] == pytest.approx(0.2759,
                                                          abs=1e-3)
    assert phys["dD_dt"] < 0.0
    # recomputación con el código vigente:
    cl = canonical_closure()
    lam = physical_spinodal_point(0.1)
    M = cl.flow_sign * stability_matrix(*lam, a=cl.a, b=cl.b)
    eigs = np.linalg.eigvals(M)
    assert np.abs(eigs.imag).max() == pytest.approx(phys["s0_tau1"],
                                                    rel=1e-9)
    assert g_effective(0.1) == pytest.approx(1000.0)
    # covarianza puntual: β(D_s·λ) = σ·D_s·β(λ; b·k):
    k, sigma = 0.3, 1.7
    Ds = np.array([k * sigma, k * sigma ** 2, k * sigma ** 3])
    lam0 = np.array([0.8, 1.1, 0.6])
    lhs = beta_functions(*(Ds * lam0), a=0.5, b=0.5)
    rhs = sigma * Ds * beta_functions(*lam0, a=0.5, b=0.5 * k)
    assert np.allclose(lhs, rhs, rtol=1e-12)
    # la transición está entre los puntos 0.45 y 0.50 de la malla:
    rows = {r["delta0"]: r["complex_pair"] for r in doc["rows"]}
    assert rows[0.45] is True and rows[0.5] is False
    # todo punto complejo de la adenda hunde D (la conjunción del
    # corpus, Obs. 8.6, en el régimen físico):
    for r in doc["rows"]:
        if r["complex_pair"]:
            assert r["dD_dt"] < 0.0, r["delta0"]


def test_run_script_consumes_validated_machinery():
    """El barrido usa la maquinaria YA validada del frente 2
    (routes_coincide de core/victoria_exponent.py), no una extracción
    nueva sin validar."""
    f = REPO / "scripts" / "run_front2_fp_beta.py"
    if not f.exists():
        import pytest
        pytest.skip("el script del barrido llega en el commit de "
                    "análisis")
    src = f.read_text(encoding="utf-8")
    assert "routes_coincide" in src or "s0_spectral" in src
