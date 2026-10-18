"""La matriz compacta de claims (es/en) se genera desde el registro y la tabla de puentes: no diverge, contiene cada
fila exactamente una vez, falla cerrado ante un estatuto desconocido y su marco no usa lenguaje de sobreafirmación
(revisión del autor del 30-sep: «lo que se muestra es el registro, no una narrativa»)."""

import copy
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("mcm", REPO / "scripts" / "make_claim_matrix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_matrices_do_not_diverge_from_the_registry():
    m = _mod()
    reg, sha = m.load_registry()
    assert m.OUT_ES.read_text(encoding="utf-8") == m.render(reg, sha, "es")
    assert m.OUT_EN.read_text(encoding="utf-8") == m.render(reg, sha, "en")
    assert sha in m.OUT_ES.read_text(encoding="utf-8")


def test_every_claim_once_and_categories_cover_the_vocabulary():
    m = _mod()
    reg, _ = m.load_registry()
    cats = m.categorize(reg)
    ids = [r["claim_id"] for rows in cats.values() for r in rows]
    assert sorted(ids) == sorted(r["claim_id"] for r in reg["claims"]) and len(ids) == len(set(ids))
    text = m.OUT_ES.read_text(encoding="utf-8")
    matrix = text.split("## 2.")[0]
    for cid in ids:
        assert matrix.count(f"`{cid}`") == 1, cid


def test_unknown_status_fails_closed():
    m = _mod()
    reg, _ = m.load_registry()
    bad = copy.deepcopy(reg)
    bad["claims"][0]["status"] = "demostrado"
    with pytest.raises(SystemExit):
        m.categorize(bad)
    bad = copy.deepcopy(reg)
    bad["status_vocabulary"] = list(bad["status_vocabulary"]) + ["nuevo-estatuto"]
    with pytest.raises(SystemExit):                       # vocabulario sin categoría ni traducción
        m.categorize(bad)


def test_frame_carries_no_overclaim_language():
    m = _mod()
    es = " ".join(list(m.FRAME_ES.values()) + [c[2] for c in m.CATEGORIES]).lower()
    en = " ".join(list(m.FRAME_EN.values()) + [c[3] for c in m.CATEGORIES] + list(m.STATUS_EN.values()) +
                  list(m.BRIDGE_KIND_EN.values())).lower()
    # las frases de las guardas del CI (.github/workflows/ci.yml) ya se comprueban en todo el repositorio, también en las
    # vistas generadas; aquí, las sobreafirmaciones propias de un marco de matriz
    for bad in ("resuelto", "demuestra ", "confirmado por los datos"):
        assert bad not in es, bad
    for pat in (r"\bprov(e|es|en|ed)\b", r"\bsolved\b", r"\bresolved\b", r"\bconfirm", r"\bdemonstrat(es|ed)\b"):
        assert not re.search(pat, en), pat
    assert "not a physical demonstration" in en and "no demostración física" in es


def test_english_edition_translates_statuses_and_quotes_rows():
    m = _mod()
    en = m.OUT_EN.read_text(encoding="utf-8")
    for line in en.splitlines():
        if line.startswith("| `"):
            status = line.split("|")[2].strip()
            assert status in m.STATUS_EN.values(), line[:120]
    assert "quoted verbatim in Spanish" in en
