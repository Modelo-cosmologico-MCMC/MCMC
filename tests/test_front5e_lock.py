"""Candados del Frente 5E: la transferencia congelada y la
prohibición de tuning con SPARC.

Estos tests NO requieren red ni datos SPARC: vigilan el protocolo.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dynamics.sculptor_transfer import (
    UPSILON_SCULPTOR_FIDUCIAL,
    frozen_config,
    sculptor_A_req,
)

REPO = Path(__file__).resolve().parent.parent
PREREG = REPO / "results" / "2026-08-16_front5e_sparc" / "preregistration.json"

# Ficheros del análisis 5E vigilados por el candado anti-tuning: TODO
# módulo que toque datos SPARC o componga la predicción, presente y
# futuro (el patrón cubre módulos nuevos con "sparc" en el nombre).
ANALYSIS_GLOBS = ["dynamics/*sparc*.py", "scripts/run_front5e*.py"]

FORBIDDEN = [
    # ajuste directo de A
    "fit_A", "fit_a_to", "optimize_A", "best_A", "A_i_per_galaxy",
    # maquinaria de optimización (no hay nada que optimizar con A fija)
    "scipy.optimize", "curve_fit", "minimize(", "least_squares",
    "brentq", "fmin", "differential_evolution",
    # mutación de la config congelada dentro del análisis
    "dataclasses.replace", "replace(cfg", "object.__setattr__",
    # selección de galaxias por el residuo de Cronos
    "sort_values(\"delta_chi2\"", "sort_values('delta_chi2'",
    "argsort(delta", "drop_worst", "exclude_galaxies",
]


def _analysis_sources():
    files = []
    for pat in ANALYSIS_GLOBS:
        files.extend(REPO.glob(pat))
    return sorted(set(files))


def test_A_is_frozen_from_sculptor():
    """La A del análisis SPARC ES la A_req de Sculptor, producida por
    la misma función canónica — no un número copiado."""
    cfg = frozen_config()
    assert cfg.A_sculptor == pytest.approx(
        sculptor_A_req(UPSILON_SCULPTOR_FIDUCIAL), rel=1e-12)
    assert cfg.A_sculptor == pytest.approx(6.2016e-7, rel=1e-3)


def test_config_is_immutable():
    """El objeto de configuración es inmutable: el análisis lo recibe,
    no puede estimarlo ni modificarlo."""
    cfg = frozen_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.A_sculptor = 1.0


def test_no_sparc_fit_of_A():
    """PROHIBICIÓN ejecutable sobre TODOS los módulos del análisis 5E
    (presentes y futuros que casen los globs): sin fit de A, sin
    maquinaria de optimización, sin mutación de la config congelada,
    sin selección por Δχ².

    LÍMITE DECLARADO: esto es una lista negra de patrones, no un
    verificador semántico — es la primera capa; la segunda es la
    revisión adversarial obligatoria de cada ronda (protocolo §18)."""
    files = _analysis_sources()
    assert files, "globs del candado sin ficheros: revisar ANALYSIS_GLOBS"
    for f in files:
        src = f.read_text(encoding="utf-8")
        for pat in FORBIDDEN:
            assert pat not in src, f"{f.name}: patrón prohibido {pat!r}"
    # El módulo principal recibe A por la config congelada:
    main_mod = REPO / "dynamics" / "sparc_crossfalsification.py"
    src = main_mod.read_text(encoding="utf-8")
    assert "CrossFalsificationConfig" in src
    assert "sculptor_A_req" not in src.replace(
        "from dynamics.sculptor_transfer", "")


def test_prereg_exists_and_matches_pipeline():
    """La preinscripción existe, declara la misma A que produce el
    pipeline hoy, y precede a cualquier ingesta (contrato del 5E)."""
    assert PREREG.exists(), "preinscripción ausente"
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["frozen_config"]["A_sculptor"] == pytest.approx(
        sculptor_A_req(UPSILON_SCULPTOR_FIDUCIAL), rel=1e-9)
    assert "hypothesis_H5E" in doc
    assert "if_data_unavailable" in doc["verdict_scope_precommitment"]


def test_prereg_frozen_fields_pinned():
    """CANDADO HISTÓRICO: los campos congelados de la preinscripción
    quedan fijados aquí con sus valores publicados — una regeneración
    accidental que los cambiara rompe la suite (el sello del registro,
    no solo su coherencia interna)."""
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    fc = doc["frozen_config"]
    assert fc["A_sculptor"] == pytest.approx(6.201589e-7, rel=1e-6)
    assert fc["upsilon_sculptor"] == 2.0
    assert fc["beta_sculptor"] == 0.0
    assert fc["R_half_pc"] == 260.0
    assert fc["sigma_obs_kms"] == 9.2
    assert fc["upsilon_disk_sparc"] == 0.5
    assert fc["upsilon_bulge_sparc"] == 0.7
    assert fc["zeta_disc"] == 0.15
    assert fc["bootstrap_seed"] == 42
    assert fc["bootstrap_n"] == 10000
    assert "decision_rule" in doc, "regla de decisión preinscrita ausente"


def test_missing_data_fails_closed():
    """Sin datos SPARC ingeridos, el análisis debe FALLAR CERRADO
    (excepción con estatuto), nunca fabricar fixtures observacionales.
    (Vigila las MISMAS rutas que usa la maquinaria: data/sparc/.)"""
    manifest = REPO / "data" / "sparc" / "manifest.json"
    if manifest.exists():
        pytest.skip("datos SPARC presentes: este candado aplica al caso sin datos")
    from dynamics.sparc_data import load_sparc_catalogue
    with pytest.raises(FileNotFoundError, match="DATA_UNAVAILABLE"):
        load_sparc_catalogue()
