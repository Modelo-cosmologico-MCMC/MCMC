"""La suite espejo del apéndice H, como test: todo verde y los
condicionales expuestos, no resueltos."""

from validation.appendix_h import run_all, CONDITIONALS


def test_mirror_suite_all_green():
    """Todas las verificaciones de signo, controles negativos y límites
    de recuperación superan (H.1/§13.5)."""
    out = run_all(verbose=False)
    assert out["all_ok"], [r for r in out["rows"] if not r["ok"]]
    assert out["n_total"] >= 17


def test_negative_controls_present():
    """La auditoría incluye controles negativos (Tabla H.1): al menos
    uno por cada verificación de signo principal."""
    out = run_all(verbose=False)
    kinds = [r["kind"] for r in out["rows"]]
    assert kinds.count("control negativo") >= 6
    assert kinds.count("recuperación") == 3


def test_conditionals_exposed_not_resolved():
    """H.3: los siete frentes figuran expuestos con su condición — la
    suite no los verifica como si estuvieran resueltos."""
    assert len(CONDITIONALS) == 7
    for name, cite in CONDITIONALS:
        assert "frente abierto" in cite
