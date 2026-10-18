"""La tabla de puentes se genera desde el registro (no diverge), cita solo filas existentes, clasifica por la
regla declarada, y la cosmología consume la unidad de S de Florencia en modo declarado."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.s_post_unit import STATUS
from cosmology.dark_channels import S_of_z, S_of_z_declared

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("mdb", REPO / "scripts" / "make_dictionary_bridges.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bridges_view_does_not_diverge_and_cites_existing_rows():
    m = _mod()
    reg = m.load_registry()
    text = m.render(reg)
    assert (REPO / "docs" / "diccionario_puentes.md").read_text(encoding="utf-8") == text
    ids = {r["claim_id"] for r in reg["claims"]}
    for _, _, cited, _, _ in m.BRIDGES:
        assert set(cited) <= ids


def test_phi_ten_split_and_single_post_florencia_gap():
    """Revisión del autor del 30-sep: Φ_ten del fondo es derivado por la Def. 6.4 (N ≡ 1) y el de las perturbaciones
    condicional (Atlas); S_post y C(S) son un solo hueco (ecuación 4). Un grupo de hueco en un puente no ausente falla."""
    m = _mod()
    reg = m.load_registry()
    tab = {b["bridge"]: b for b in m.table(reg)}
    assert tab["Φ_ten (Florencia, fondo)"]["kind"] == "derivado" and "phi-ten-fondo-ley-cronos" in tab["Φ_ten (Florencia, fondo)"]["rows"]
    assert tab["Φ_ten (perturbaciones)"]["kind"] == "condicional"
    post = [b for b in tab.values() if b["gap"] == "ecuación 4"]
    assert {b["bridge"] for b in post} == {"S_post (unidad de S tras Florencia)", "C(S) = d ln a/dS"}
    assert all(b["kind"] == "ausente" for b in post) and m.gap_count(list(tab.values())) == 1
    saved = m.BRIDGES[:]
    try:
        m.BRIDGES[0] = m.BRIDGES[0][:4] + ("ecuación 4",)          # δ₀ (condicional) con grupo de hueco: fallo cerrado
        with pytest.raises(SystemExit):
            m.table(reg)
    finally:
        m.BRIDGES[:] = saved


def test_classification_rule():
    m = _mod()
    mk = lambda s, e="": {"status": s, "evidence_level": e}  # noqa: E731
    assert m.classify([mk("interno")]) == "derivado"
    assert m.classify([mk("interno"), mk("condicional")]) == "condicional"
    assert m.classify([mk("condicional"), mk("calibrado")]) == "calibrado"
    assert m.classify([mk("interno", "parámetros calibrados (F.3)"), mk("condicional")]) == "calibrado"
    assert m.classify([{"status": "interno", "dataset": "— (ancla externa, no ingerida)"}]) == "calibrado"
    assert m.classify([mk("calibrado", "convención declarada en el código")]) == "convencional"
    assert m.classify([mk("interno"), mk("hueco-declarado")]) == "ausente"


def test_S_of_z_declared_consumes_post_florencia_unit():
    z = np.array([0.0, 0.5, 2.0])
    d = S_of_z_declared(z)
    assert np.allclose(d["S"], S_of_z(z)) and d["S_today"] == 95.0
    assert d["unit"]["E2"] == STATUS["E2"] and d["unit"]["E3"] == STATUS["E3"]
    assert d["unit"]["mode"] == "declarado" and "LEGACY_V32" in d["convention"]
