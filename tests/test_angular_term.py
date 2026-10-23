"""Cuarta ecuación, brazo «término angular» (bloque P1 del autor del 23-oct): el cierre V_ang = ½μ²χ² en el reloj
(ClockConfig.angular_A) — μ² = 4A(c̄/κ₊)δ₀², gradiente frente a diferencias finitas, sin término nada cambia, las cifras
del guion del autor con A = 1 (θ*, S con medio giro, sin cruce de la diagonal) y el borde superior de la ventana."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from core.basal import c_bar, kappa_plus
from core.s_clock import ClockConfig, SClock, radial_landscape

REPO = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("run_angular_term", REPO / "scripts" / "run_angular_term.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mu2_gradient_and_default_unchanged():
    clk = SClock(ClockConfig(delta0=0.01, angular_A=0.7))
    assert clk.mu2 == pytest.approx(4.0 * 0.7 * c_bar() / kappa_plus() * 0.01 ** 2)
    for q in (np.array([0.11, 0.03]), np.array([0.05, 0.12]), np.array([0.2, 0.2])):
        h = 1e-7
        fd = [(clk._V(q + h * e, clk.lam0) - clk._V(q - h * e, clk.lam0)) / (2 * h) for e in np.eye(2)]
        assert clk._grad(q, clk.lam0) == pytest.approx(fd, rel=1e-6, abs=1e-14)
    plain = SClock(ClockConfig(delta0=0.01))
    assert plain.mu2 == 0.0 and plain.true_vacuum_2d is None
    assert plain.T0 == radial_landscape(0.01)["T0_full"]


@pytest.mark.parametrize("delta0, theta_star, s_half", [(0.003, 0.759, 0.73), (0.01, 0.737, 0.75), (0.0554, 0.672, 0.80)])
def test_author_figures_at_A_equal_one(delta0, theta_star, s_half):
    row = _script().one(delta0, 1.0)
    assert row["finished"] and not row["crosses_diagonal"] and row["theta_max_drop"] <= 1e-12
    assert row["theta_star"] == pytest.approx(theta_star, abs=5e-4)
    assert row["S_half_turn"] == pytest.approx(s_half, abs=5e-3)
    assert row["T0_full_over_law"] > 1.0 and row["theta_final"] < row["theta_star"] < math.pi / 4


def test_window_and_its_upper_edge():
    m = _script()
    for d0 in m.DELTA0:
        t = m.tilt_weight(d0)
        g = m.a_grid(d0)
        assert 1.0 in g and all(t / 4.0 < a < 1.0 + t / math.sqrt(2.0) for a in g) and len(g) == m.N_A + 1
        edge = m.exact_upper_edge(d0)                       # 1 + t/√2 es su primer orden
        assert 1.0 < edge / (1.0 + t / math.sqrt(2.0)) < 1.1
    edge = m.exact_upper_edge(0.01)
    SClock(ClockConfig(delta0=0.01, angular_A=0.999 * edge))
    with pytest.raises(ValueError, match="Sin punto de escape"):
        SClock(ClockConfig(delta0=0.01, angular_A=1.001 * edge))
