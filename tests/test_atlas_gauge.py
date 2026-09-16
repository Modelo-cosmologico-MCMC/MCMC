"""E5_Atlas: forma cerrada de la cola gauge-invariante µ_Δ fijada en
cosmology/mu_eta_atlas.py y la aritmética de la transformación de gauge
sobre una torre sintética (sin sympy: la torre real vive en
validation/atlas_tail_derivation.py, fuera de la suite).
"""

import numpy as np
import pytest

from cosmology.mu_eta_atlas import khronon_cs2, mu_delta_tail_leading


class TestClosedForm:
    def test_equals_minus_alpha_over_cs2(self):
        for lam, al in ((1.012, 1e-6), (1.012, 0.012), (1.05, 0.3), (1.0001, 1e-4)):
            e = np.array([0.005, 0.01, 0.02])
            np.testing.assert_allclose(mu_delta_tail_leading(e, lam, al),
                                       -al * e ** 2 / khronon_cs2(lam, 1.0, al), rtol=1e-14)

    def test_second_order_in_alpha(self):
        """−α_a e²/c_s² = −α_a²(3λ−1)/((2−α_a)(λ−1)) e²: escala como α_a²."""
        lam, e = 1.012, 0.01
        expected = -(1e-4) ** 2 * (3 * lam - 1) / ((2 - 1e-4) * (lam - 1)) * e ** 2
        assert mu_delta_tail_leading(e, lam, 1e-4) == pytest.approx(expected, rel=1e-12)
        ratio = mu_delta_tail_leading(e, lam, 1e-4) / mu_delta_tail_leading(e, lam, 1e-5)
        assert ratio == pytest.approx(100.0, rel=1e-4)

    def test_gr_limit_smooth_and_negative(self):
        assert mu_delta_tail_leading(0.02, 1.012, 0.0) == 0.0
        assert mu_delta_tail_leading(0.02, 1.012, 1e-3) < 0
        # α_a → 0 primero: no hay discontinuidad, la cola se apaga como α_a²
        vals = [abs(float(mu_delta_tail_leading(0.02, 1.012, a))) for a in (1e-2, 1e-4, 1e-6)]
        assert vals[0] > vals[1] > vals[2] and vals[2] < 1e-12

    def test_unobservable_at_ppn_bound(self):
        e = 3.336e-4 / 0.02
        assert abs(float(mu_delta_tail_leading(e, 1.012, 8e-7))) < 1e-12
        with pytest.raises(ValueError):
            mu_delta_tail_leading(0.01, 1.05, 0.3, xi=1.02)


class TestGaugeArithmeticOnSyntheticTower:
    """Ψ_N = ψ + ḃ, Φ_N = φ − Hb sobre X = t^g Σ c_n t^{−n/3}, H = 2/(3t):
    la corrección de un coeficiente c_{b,n} cae en n + 3 con pesos
    (g − n/3) y −2/3. Se comprueba contra diferenciación numérica."""

    def test_shift_by_three_and_weights(self):
        g = 0.7
        cb = {1: -5.8, 3: 10.9}
        tt = np.linspace(0.5, 1.5, 2001)
        bfun = sum(c * tt ** (g - n / 3) for n, c in cb.items())
        bdot = np.gradient(bfun, tt)
        Hb = (2 / (3 * tt)) * bfun
        pred_bdot = sum(c * (g - n / 3) * tt ** (g - (n + 3) / 3) for n, c in cb.items())
        pred_Hb = sum(c * (2 / 3) * tt ** (g - (n + 3) / 3) for n, c in cb.items())
        np.testing.assert_allclose(bdot[10:-10], pred_bdot[10:-10], rtol=1e-5)
        np.testing.assert_allclose(Hb, pred_Hb, rtol=1e-12)
