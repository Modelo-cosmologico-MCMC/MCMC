"""PPN de marco preferido del sector Atlas (E4_Atlas): formas cerradas
fijadas en cosmology/mu_eta_atlas.py (límite khronométrico de
Einstein-aether, β = 0 ⇔ ξ = 1) y la aritmética de cotas. Sin sympy.
"""

import pytest

from cosmology.mu_eta_atlas import (
    G_cosmo_over_G_local,
    alpha_a_max_from_ppn,
    khronon_cs2,
    ppn_alpha1,
    ppn_alpha2,
)


class TestClosedForms:
    def test_alpha1_is_minus_4alpha_at_beta0(self):
        for al in (1e-6, 1e-3, 0.1, 0.3):
            assert ppn_alpha1(al) == pytest.approx(-4 * al, rel=1e-14)
        # forma general 4(α−2β)/(β−1)
        assert ppn_alpha1(0.1, beta=0.05) == pytest.approx(4 * (0.1 - 0.1) / (0.05 - 1))

    def test_alpha2_closed_form_and_leading_order(self):
        al, lam = 0.3, 0.05
        expected = al * (2 * al * lam + al - lam) / (lam * (2 - al))
        assert ppn_alpha2(al, 1 + lam) == pytest.approx(expected, rel=1e-14)
        # orden dominante −α_a/2 cuando α_a ≪ λ
        assert ppn_alpha2(1e-6, 1.012) == pytest.approx(-0.5e-6, rel=1e-3)
        assert ppn_alpha2(1e-8, 1.012) == pytest.approx(-0.5e-8, rel=1e-5)
        with pytest.raises(ValueError):
            ppn_alpha2(0.1, 1.0)

    def test_gr_limit_and_signs(self):
        assert ppn_alpha1(0.0) == 0.0
        assert ppn_alpha2(0.0, 1.012) == 0.0
        assert ppn_alpha1(1e-3) < 0 and ppn_alpha2(1e-3, 1.012) < 0

    def test_mapping_consistency_with_treatise_forms(self):
        """β = 0: c_s² y G_cosmo/G_N del tratado coinciden con las formas
        khronométricas λ(2−α)/(α(2+3λ)) y (2−α)/(2+3λ)."""
        al, lam = 0.3, 0.05
        assert khronon_cs2(1 + lam, 1.0, al) == pytest.approx(
            lam * (2 - al) / (al * (2 + 3 * lam)), rel=1e-12)
        assert G_cosmo_over_G_local(1 + lam, 1.0, al) == pytest.approx(
            (2 - al) / (2 + 3 * lam), rel=1e-12)


class TestBounds:
    def test_solar_spin_governs(self):
        b = alpha_a_max_from_ppn(1e-4, 4e-7, 1.012)
        assert b["governing"] == "alpha2"
        assert b["alpha_a_max_from_alpha1"] == pytest.approx(2.5e-5)
        assert b["alpha_a_max"] == pytest.approx(8.0e-7, rel=2e-3)
        assert b["alpha_a_max"] == pytest.approx(b["leading_order_alpha2_bound"], rel=2e-3)

    def test_weak_alpha2_bound_lets_alpha1_govern(self):
        b = alpha_a_max_from_ppn(1e-4, 1.8e-4, 1.012)
        assert b["governing"] == "alpha1"
        assert b["alpha_a_max"] == pytest.approx(2.5e-5)

    def test_bound_weakly_depends_on_lambda(self):
        vals = [alpha_a_max_from_ppn(1e-4, 4e-7, 1 + lv)["alpha_a_max"]
                for lv in (0.001, 0.012, 0.1)]
        assert max(vals) / min(vals) < 1.01

    def test_sound_tail_at_bound_is_suppressed_in_pole_form(self):
        """En la cota α_a ≈ 8e-7 con λ_K = 1.012 el khronon es superlumínico
        (c_s² ≫ 1) y el residuo del polo (3/2)e²/c_s² a k = 0.02 h/Mpc cae
        por debajo de 1e-7. (El coeficiente COMPLETO lo da la escalera en el
        artefacto E4: en α_a ≪ λ_K − 1 la expansión de E3 no aplica.)"""
        from cosmology.mu_eta_atlas import eta_tail_physical
        e = 3.336e-4 / 0.02
        assert khronon_cs2(1.012, 1.0, 8e-7) > 1e3
        assert eta_tail_physical(e, 1.012, 8e-7) < 1e-7
