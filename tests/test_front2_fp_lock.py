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
