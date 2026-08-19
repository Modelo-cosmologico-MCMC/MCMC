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
    REPO / "scripts" / "run_front2_prereg.py",
]
# patrones de USO real (no de mención en prosa — la promesa de la
# preinscripción cita los nombres y no debe dispararse a sí misma):
FORBIDDEN = [
    "from scipy.optimize", "import scipy.optimize", "scipy.optimize.",
    "curve_fit(", "minimize(", "fsolve(", "brentq(", "least_squares(",
    "differential_evolution(", "dataclasses.replace", "replace(cl",
    "object.__setattr__",
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
    # barrido: ventana uniforme real; único complejo en g = 10:
    cplx = [r for r in doc["scan"] if r["complex_pair"]]
    assert len(cplx) == 1 and cplx[0]["g"] == 10.0
    assert cplx[0]["s0_spectral"] == pytest.approx(5.8661, abs=1e-3)
    assert cplx[0]["dD_dt"] < 0.0    # solo allí se hunde D (Obs. 8.6)
    for r in doc["scan"]:
        if 0.5 <= r["g"] <= 2.0:
            assert r["complex_pair"] is False, r["g"]
    # el sistemático cuártico no rescata la cascada:
    Mq = cl.flow_sign * stability_matrix_quartic(
        *spinodal_canonical_point(), 0.0, a=cl.a, b=cl.b)
    assert np.abs(np.linalg.eigvals(Mq).imag).max() < 1e-12
    assert doc["quartic_E4_0"]["complex_pair"] is False
    assert doc["tau_star_for_lambda10"] is None


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
