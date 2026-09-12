"""El registro canónico de claims y su test de NO-DIVERGENCIA.

La fuente única es docs/claims_registry.yaml; las vistas
(matriz_trazabilidad.md, ontology_nomology_observables.md) se generan.
Si lo commiteado difiere de la regeneración, la suite falla: un cambio
de estatuto en #13, #15 o 5E no puede dejar dos documentos
contradictorios, y nadie puede editar las vistas a mano.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_traceability import (  # noqa: E402
    MATRIX,
    ONTOLOGY,
    load_registry,
    render_matrix,
    render_ontology,
)

REQUIRED_FIELDS = ("claim_id", "entity", "law", "derivation",
                   "implementation", "test", "observable", "dataset",
                   "falsifier", "evidence_level", "status", "artifact")


def _claims():
    return load_registry()["claims"]


def test_no_divergence_matrix():
    """La matriz commiteada ES la regenerada, byte a byte."""
    assert MATRIX.read_text(encoding="utf-8") == \
        render_matrix(load_registry())


def test_no_divergence_ontology():
    """La vista ontología→observables commiteada ES la regenerada."""
    assert ONTOLOGY.read_text(encoding="utf-8") == \
        render_ontology(load_registry())


def test_schema_complete_and_vocabulary():
    """Cada claim trae los 12 campos y un status del vocabulario
    controlado; los claim_id son únicos."""
    reg = load_registry()
    vocab = set(reg["status_vocabulary"])
    ids = [c["claim_id"] for c in reg["claims"]]
    assert len(ids) == len(set(ids)), "claim_id duplicado"
    for c in reg["claims"]:
        for field in REQUIRED_FIELDS:
            assert field in c and str(c[field]).strip() != "", \
                f"{c.get('claim_id', '?')}: campo {field} vacío"
        assert c["status"] in vocab, \
            f"{c['claim_id']}: status fuera del vocabulario"


def test_cited_paths_exist():
    """Toda ruta citada en implementation/test/artifact existe («—»
    es hueco declarado y se acepta)."""
    missing = []
    for c in _claims():
        for field in ("implementation", "test", "artifact"):
            cell = str(c[field]).strip()
            if cell == "—":
                continue
            for path in cell.split(" + "):
                if not (ROOT / path.strip()).exists():
                    missing.append(f"{c['claim_id']}:{path.strip()}")
    assert not missing, f"rutas inexistentes: {missing}"


def test_three_negative_categories_present_and_distinct():
    """Las filas negativas obligatorias existen con sus TRES
    categorías diferenciadas: negativo ≠ no derivado ≠ no ejecutado."""
    by_status = {}
    for c in _claims():
        by_status.setdefault(c["status"], set()).add(c["claim_id"])
    assert {"desi-dr2-6a", "fp-beta-desenlace-a",
            "ajustes-produccion"} <= by_status["resultado-negativo"]
    assert "decada-discriminante" in by_status["claim-no-derivado"]
    assert "5e-falsacion-cruzada" in by_status[
        "experimento-no-ejecutado"]
    # categorías disjuntas por construcción (un status por fila):
    assert by_status["resultado-negativo"].isdisjoint(
        by_status["claim-no-derivado"])
    assert by_status["claim-no-derivado"].isdisjoint(
        by_status["experimento-no-ejecutado"])


def test_canonical_reading_of_front2_pinned():
    """La lectura canónica de #13 queda fijada: desenlace A negativo,
    frontera g* = 8.189, la adenda NO reclasifica, λ = 10 no derivada."""
    c = {x["claim_id"]: x for x in _claims()}
    fp = c["fp-beta-desenlace-a"]["derivation"]
    assert "desenlace A" in fp
    assert "g* = 8.189" in fp
    assert "NO reclasifica" in fp
    dec = c["decada-discriminante"]["derivation"]
    assert "NO derivada" in dec and "τ(S)" in dec


def test_canonical_reading_of_6a_pinned():
    """6A queda fijado como resultado negativo: ε dominada por el
    prior (BAO-only no la identifica) y contraste pro-ΛCDM."""
    c = {x["claim_id"]: x for x in _claims()}
    d = c["desi-dr2-6a"]["derivation"]
    assert "DOMINADA POR EL PRIOR" in d
    assert "pro-ΛCDM" in d


def test_5e_is_not_executed_not_negative():
    """5E es experimento NO ejecutado (fallo cerrado), jamás un
    negativo: la distinción es obligatoria."""
    c = {x["claim_id"]: x for x in _claims()}
    e = c["5e-falsacion-cruzada"]
    assert e["status"] == "experimento-no-ejecutado"
    assert "No es un resultado negativo" in e["derivation"]
    assert "DATA_UNAVAILABLE" in e["dataset"]


def test_frb_does_not_enter():
    """FRB no entra como fila hasta tener dataset con DOI, observable
    consumido y relación con el parámetro bariónico — la exclusión es
    ejecutable."""
    for c in _claims():
        for field in ("claim_id", "entity", "observable", "dataset"):
            assert "frb" not in str(c[field]).lower(), c["claim_id"]
    # y la exclusión queda declarada en la vista generada:
    assert "FRB no entra" in ONTOLOGY.read_text(encoding="utf-8")


def test_registry_yaml_parses_and_is_substantial():
    reg = yaml.safe_load(
        (ROOT / "docs" / "claims_registry.yaml").read_text(
            encoding="utf-8"))
    assert len(reg["claims"]) >= 39
