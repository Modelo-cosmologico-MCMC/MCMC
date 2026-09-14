"""Contraste de los Residuos vía BBN — unidades SIN los valores
observacionales registrados: el modelo de respuesta (tabla PRyMordial),
la predicción del modelo, la marginalización exacta y la regla de
desenlace, sobre datos sintéticos.
"""

import json

import numpy as np
import pytest

from cosmology.bbn_g import (
    RESPONSE_TABLE,
    RULES,
    classify_outcome,
    delta_G_model,
    log_prob_grid,
    posterior_1d,
    predict,
    response_constants,
    sm_pulls,
)
from cosmology.mu_eta_atlas import residues_ratio_first_order


def synth(dG_true=0.0, yp_sigma=0.002, dh_sigma=0.02e-5, yp_shift=0.0):
    """Datos sintéticos generados por el propio modelo de respuesta."""
    r = response_constants()
    Y, D = predict(dG_true, 0.0, r["omega_b0"], r["tau_n0"], r)
    return {"values": {
        "Y_P_empress_xv": {"value": float(Y) + yp_shift, "sigma": yp_sigma},
        "Y_P_aver_2021": {"value": float(Y), "sigma": yp_sigma},
        "DH_cooke_2018": {"value": float(D), "sigma": dh_sigma},
        "omega_b_planck_2018": {"value": r["omega_b0"], "sigma": 0.00015},
        "tau_n_pdg_2023": {"value": r["tau_n0"], "sigma": 0.5}},
        "official_bytes_verified": False}


class TestModelPrediction:
    def test_delta_G_model_is_minus_1p8_percent(self):
        d = delta_G_model()
        assert d == pytest.approx(2.0 / (3 * 1.012 - 1) - 1.0, rel=1e-12)
        assert d == pytest.approx(-0.0177, abs=2e-4)
        # coherente con la forma de primer orden del frente 6
        assert d == pytest.approx(residues_ratio_first_order(0.012) - 1.0, abs=5e-4)
        # α_a entra como −α_a/2 al primer orden
        assert delta_G_model(alpha_a=0.01) < d


class TestResponseTable:
    def test_table_provenance_and_exponents(self):
        r = response_constants()
        doc = json.loads(RESPONSE_TABLE.read_text(encoding="utf-8"))
        assert r["prymordial_commit"] == doc["prymordial"]["commit"]
        assert len(r["prymordial_commit"]) == 40
        # física: Y_P ∝ G^0.36, D/H ∝ G^0.98 (velocidad de expansión); la
        # correspondencia con ΔN_eff por época da 0.33–0.36 y 0.95–1.0
        assert 0.33 <= r["a_G_YP"] <= 0.38
        assert 0.90 <= r["a_G_DH"] <= 1.02
        assert r["a_w_DH"] < -1.5 and r["a_w_YP"] > 0
        assert r["a_N_YP"] > 0 and r["a_N_DH"] > 0
        assert r["max_rel_resid"] < 2e-3
        assert r["YP0"] == pytest.approx(0.2469, abs=5e-4)     # Fields+2020: 0.2469
        assert 2.40e-5 < r["DH0"] < 2.56e-5

    def test_predict_reproduces_design_runs(self):
        doc = json.loads(RESPONSE_TABLE.read_text(encoding="utf-8"))
        r = response_constants()
        for d, row in zip(doc["design"]["delta_G"], doc["runs"]["delta_G"]):
            Y, D = predict(d, 0.0, resp=r)
            assert Y == pytest.approx(row["YP"], rel=2e-3)
            assert D == pytest.approx(row["DH"], rel=2e-3)
        for dn, row in zip(doc["design"]["delta_Neff"], doc["runs"]["delta_Neff"]):
            Y, D = predict(0.0, dn, resp=r)
            assert Y == pytest.approx(row["YP"], rel=2e-3)
            assert D == pytest.approx(row["DH"], rel=2e-3)

    def test_G_and_Neff_act_in_the_same_direction(self):
        r = response_constants()
        Y0, D0 = predict(0.0, 0.0, resp=r)
        Yg, Dg = predict(0.02, 0.0, resp=r)
        Yn, Dn = predict(0.0, 0.2, resp=r)
        assert Yg > Y0 and Dg > D0 and Yn > Y0 and Dn > D0


class TestLikelihood:
    def test_recovers_injected_delta_G(self):
        data = synth(dG_true=-0.05)
        post = posterior_1d(data)
        assert post["p50"] == pytest.approx(-0.05, abs=3e-3)
        assert post["ci95"][0] < -0.05 < post["ci95"][1]
        assert not post["prior_edge_hit"]

    def test_exact_marginalization_matches_numerical(self):
        """La covarianza aumentada reproduce la integral numérica sobre ω_b
        con su prior gaussiano (respuesta lineal en ln ω_b)."""
        data = synth(dG_true=0.0)
        data["values"]["tau_n_pdg_2023"]["sigma"] = 1e-12     # solo ω_b como nuisance
        r = response_constants()
        grid = np.linspace(-0.1, 0.1, 201)
        lp_exact = log_prob_grid(grid, 0.0, data, r)
        # numérico: integrar sobre ω_b sin el término de nuisance en la cov
        wb0, swb = data["values"]["omega_b_planck_2018"]["value"], 0.00015
        # el prior exacto es gaussiano en ln ω_b (σ_ln = σ/ω): se integra en ω
        # con el mismo prior y el jacobiano 1/ω
        ws = np.linspace(wb0 - 5 * swb, wb0 + 5 * swb, 401)
        prior = np.exp(-0.5 * (np.log(ws / wb0) / (swb / wb0)) ** 2) / ws
        rules_no_w = dict(RULES)
        data_fixed = json.loads(json.dumps(data))
        data_fixed["values"]["omega_b_planck_2018"]["sigma"] = 1e-12
        data_fixed["values"]["tau_n_pdg_2023"]["sigma"] = 1e-12
        num = np.zeros_like(grid)
        for i, g in enumerate(grid):
            like = []
            for w in ws:
                data_fixed["values"]["omega_b_planck_2018"]["value"] = w
                like.append(np.exp(log_prob_grid(g, 0.0, data_fixed, r, rules_no_w)))
            num[i] = np.log(np.trapezoid(np.array(like) * prior, ws))
        # comparar formas normalizadas (constantes aparte)
        a = lp_exact - lp_exact.max()
        b = num - num.max()
        assert np.max(np.abs(a - b)) < 1e-3

    def test_single_probe_and_pulls(self):
        data = synth(dG_true=0.0, yp_shift=-0.006)       # Y_P bajo a mano
        yo = posterior_1d(data, use_DH=False)
        do = posterior_1d(data, use_YP=False)
        assert yo["p50"] < -0.03 and abs(do["p50"]) < 0.01
        pulls = sm_pulls(data)
        assert pulls["pull_YP"] < -2.5 and abs(pulls["pull_DH"]) < 0.5


class TestOutcomeRule:
    def test_order_C_B_A(self):
        m = delta_G_model()
        assert classify_outcome([-0.05, 0.05], m) == "A"
        assert classify_outcome([-0.04, -0.005], m) == "B"      # 0 fuera, modelo dentro
        assert classify_outcome([-0.01, 0.05], m) == "C"        # modelo fuera
        assert classify_outcome([0.01, 0.05], m) == "C"         # ambos fuera → C manda
