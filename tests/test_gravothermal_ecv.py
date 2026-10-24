"""La fuerza de la ECV en el código gravotérmico y la pared en R₀ (documento vivo del 24-oct, PR-63): con q_E = 0 el código
es el de #61; con la fuerza, el equilibrio inicial, la fuerza neta nula en la pared, la energía con el potencial de la
ECV, el fallo cerrado de una pared más allá de R₀, el acortamiento de t_c para c = 5, 10 y 15 y la comprobación de
resolución. Configuración barata (100 capas hasta 100 r_s, eps = 4·10⁻³); las corridas de referencia, con la resolución
que pasa la comprobación, están en results/2026-10-24_gravothermal_ecv/."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dynamics.ecv_border import r0_selfconsistent
from dynamics.gravothermal import (
    RESOLUTION_TOL,
    GravothermalConfig,
    GravothermalHalo,
    border_shift,
    ecv_config,
    resolution_shift,
    run_collapse,
    zero_acceleration_radius_code,
)

FAST = GravothermalConfig(sigma_hat=0.01, n_shells=100, eps=4e-3)
RESULTS = Path(__file__).resolve().parent.parent / "results" / "2026-10-24_gravothermal_ecv" / "results.json"


def test_without_ecv_the_code_is_that_of_pr61():
    assert run_collapse(FAST)["C_sigma_t_c"] == pytest.approx(152.82316398177247, rel=1e-9)    # el valor de #61


def test_ecv_equilibrium_with_the_wall_at_r0():
    cfg = ecv_config(0.01, 10.0, base=replace(FAST, n_shells=60))
    h = GravothermalHalo(cfg)
    assert cfg.wall == "R0" and cfg.r_border == pytest.approx(r0_selfconsistent(10.0), rel=1e-12)
    assert h.hydrostatic_residual() < 1e-10 and abs(h.wall_net_force()) < 1e-9
    assert zero_acceleration_radius_code(h.M[-1], cfg.q_E) == pytest.approx(h.r[-1], rel=1e-12)   # la masa del código fija R₀
    rc = h.r_centre
    manual = (np.sum(h.u * h.dM) - np.sum((h.M - 0.5 * h.dM) * h.dM / rc) - cfg.q_E / 3.0 * np.sum(rc ** 2 * h.dM))
    assert h.energy() == pytest.approx(manual, rel=1e-12)
    inner = (h.M[-2] - 2.0 / 3.0 * cfg.q_E * h.r[-2] ** 3) / h.M[-2]       # junto a la pared, la ECV casi compensa
    assert 0.0 < inner < 0.3


def test_wall_beyond_r0_fails_closed():
    cfg = ecv_config(0.01, 10.0, base=replace(FAST, n_shells=60))
    with pytest.raises(ValueError, match="más allá de R₀"):
        GravothermalHalo(replace(cfg, wall="fija", r_border=1.25 * cfg.r_border))
    with pytest.raises(ValueError, match="R0"):
        GravothermalHalo(replace(cfg, r_border=0.9 * cfg.r_border))
    with pytest.raises(ValueError, match="pared fija sin la fuerza"):
        border_shift(cfg)
    assert GravothermalHalo(replace(cfg, wall="fija", r_border=0.8 * cfg.r_border)).wall_net_force() > 0.0  # control


@pytest.mark.parametrize("c, shift", [(10.0, -0.029), (15.0, -0.015), (5.0, -0.075)])
def test_ecv_force_shortens_the_collapse(c, shift):
    cfg = ecv_config(0.01, c, base=FAST)
    with_ecv = run_collapse(cfg)
    without = run_collapse(replace(cfg, q_E=0.0, wall="fija"))
    assert with_ecv["t_c"] / without["t_c"] - 1.0 == pytest.approx(shift, abs=1.5e-3)        # cifras del documento
    assert with_ecv["outside_thermal_influence"] and abs(with_ecv["energy_rel_change"]) < 1e-2


def test_resolution_check_is_not_vacuous():
    res = resolution_shift(ecv_config(0.01, 10.0, base=replace(FAST, n_shells=60)))
    assert res["rel"] > RESOLUTION_TOL and not res["converged"]          # con pocas capas la discretización pesa más


@pytest.mark.skipif(not RESULTS.exists(), reason="corridas de referencia aún no escritas")
def test_reference_runs():
    doc = json.loads(RESULTS.read_text(encoding="utf-8"))
    s = doc["summary"]
    assert s["resolution_check_reference_passes_all"] and not s["resolution_check_doc_spacing_passes"]
    for c, shift in (("5", -0.075), ("10", -0.029), ("15", -0.015)):
        assert s["ecv_shift_doc_spacing"][c] == pytest.approx(shift, abs=1e-3)
        assert s["ecv_shift_reference"][c] == pytest.approx(shift, abs=2e-3)
    assert s["wall_beyond_r0_fails_closed"] and abs(s["max_wall_net_force"]) < 1e-9
