"""La matriz de trazabilidad es ejecutable: cada ruta citada existe.

Convierte la intención del Apéndice H («cada pieza debe resistir una
prueba objetiva») en verificación: si un módulo o test citado en
docs/matriz_trazabilidad.md se renombra o desaparece, este test falla.
Las celdas «—» son huecos DECLARADOS y se aceptan como tales.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "docs" / "matriz_trazabilidad.md"


def _rows() -> list[list[str]]:
    lines = MATRIX.read_text(encoding="utf-8").splitlines()
    rows = [ln for ln in lines if ln.startswith("|") and "---" not in ln]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in rows[1:]]


def test_matrix_exists_and_is_substantial():
    """La matriz existe y cubre el programa (≥ 30 filas)."""
    assert MATRIX.exists()
    assert len(_rows()) >= 30


def test_every_cited_path_exists():
    """Toda ruta en las columnas Código y Test existe en el árbol."""
    missing = []
    for row in _rows():
        for cell in row[2:4]:
            for path in re.findall(r"`([^`]+)`", cell):
                if not (ROOT / path).exists():
                    missing.append(path)
    assert not missing, f"Rutas citadas inexistentes: {missing}"


def test_declared_gaps_are_marked():
    """Los huecos se declaran con «—», nunca con celdas vacías."""
    for row in _rows():
        for cell in row[2:4]:
            assert cell != "", f"Celda vacía en fila: {row[0]}"


def test_statuses_carry_no_overclaim():
    """Ninguna fila reclama lo prohibido por la v35.1: los estatutos
    usan el vocabulario de E8 («interna demostrada», «juguete»,
    «condicional», «calibrado», «prototipo»...) y jamás «resuelto» ni
    demostración a secas sin cualificar."""
    seen_interna = False
    for row in _rows():
        status = row[4].lower()
        assert "resuelto" not in status, row[0]
        if "demostrad" in status:
            assert "interna" in status or "juguete" in status, row[0]
            seen_interna = True
    assert seen_interna
