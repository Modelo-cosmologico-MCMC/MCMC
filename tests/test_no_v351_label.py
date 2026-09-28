"""Guarda: la etiqueta de versión retirada (la fe de erratas epistemológica del 4-ago-2026 se
cita como FE-08-2026, con sus códigos E intactos) no puede reaparecer fuera de la historia.

Decisión del autor (28-sep-2026): esa versión intermedia no existirá; la siguiente versión del
tratado será la v36. Solo CHANGELOG.md, docs/nota_computacional_I.md y results/ (registros
históricos, inmutables) conservan la cadena. El patrón se construye por partes para que este
fichero no la contenga."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABEL = "v35" + ".1"
ALLOWED = {REPO / "CHANGELOG.md", REPO / "docs" / "nota_computacional_I.md"}
EXTS = {".py", ".md", ".ipynb", ".yaml", ".yml", ".txt", ".html", ".json", ".cfg", ".toml", ".ini", ".rst"}


def _candidates():
    for p in REPO.rglob("*"):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        parts = p.relative_to(REPO).parts
        if parts[0] in {".git", "results", "node_modules", ".venv", "__pycache__"} or ".pytest_cache" in parts:
            continue
        yield p


def test_retired_version_label_absent_outside_history():
    hits = []
    for p in _candidates():
        if p in ALLOWED:
            continue
        if LABEL in p.name:
            hits.append(f"{p.relative_to(REPO)} (nombre de fichero)")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if LABEL in line:
                hits.append(f"{p.relative_to(REPO)}:{i}")
    assert not hits, "etiqueta retirada presente fuera de la historia:\n" + "\n".join(hits)


def test_errata_file_and_label_definition_exist():
    assert (REPO / "docs" / "erratas_epistemologicas_2026-08-04.md").exists()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "FE-08-2026" in readme and "erratas_epistemologicas_2026-08-04.md" in readme
