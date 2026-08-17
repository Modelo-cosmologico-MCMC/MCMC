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
    """PROHIBICIÓN ejecutable: el módulo de análisis 5E no contiene
    ajuste de A — ni fit_A, ni minimización de χ² respecto de A, ni
    A por galaxia. (Escaneo de fuente; si el módulo aún no existe,
    la prohibición rige desde su nacimiento.)"""
    mod = REPO / "dynamics" / "sparc_crossfalsification.py"
    if not mod.exists():
        pytest.skip("módulo 5E aún no creado; el candado rige al crearlo")
    src = mod.read_text(encoding="utf-8")
    forbidden = ["fit_A", "fit_a_to", "optimize_A", "minimize(",
                 "curve_fit", "A_i_per_galaxy", "best_A"]
    for pat in forbidden:
        assert pat not in src, f"patrón de tuning prohibido: {pat}"
    # A debe llegar por la config congelada, no recalcularse allí:
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


def test_missing_data_fails_closed():
    """Sin datos SPARC ingeridos, el análisis debe FALLAR CERRADO
    (excepción con estatuto), nunca fabricar fixtures observacionales."""
    raw = REPO / "data" / "raw" / "sparc"
    manifest = REPO / "data" / "raw" / "sparc" / "manifest.json"
    if manifest.exists():
        pytest.skip("datos SPARC presentes: este candado aplica al caso sin datos")
    try:
        from dynamics.sparc_data import load_sparc_catalogue
    except ImportError:
        pytest.skip("parser 5E aún no creado; el candado rige al crearlo")
    with pytest.raises(FileNotFoundError, match="DATA_UNAVAILABLE"):
        load_sparc_catalogue()
    assert not any(raw.glob("*.dat")) if raw.exists() else True
