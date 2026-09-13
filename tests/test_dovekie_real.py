"""Primera aplicación a Dovekie real — unidades SIN datos reales: todo
sobre vectores sintéticos (la columna MU real no se toca en la suite).
El candado de preinscripción/artefacto vive en
tests/test_dovekie_real_lock.py.
"""

from pathlib import Path

import numpy as np
import pytest

from cosmology.bayesian_fit import BAOData, HzData
from cosmology.dovekie_real_fit import (
    BOUNDS,
    K_PARAMS,
    PRIOR_OMEGA_M,
    chi2_blocks,
    classify_outcome,
    log_prior,
    log_prob,
    n_points,
    official_omega_m,
    theta_full,
)
from cosmology.dovekie_sn import mu_model

REPO = Path(__file__).resolve().parent.parent
RULES = {"C_bench_sigma": 3.0, "C_chi2nu_max": 1.30, "B_dBIC_pro_mcmc": 2.0,
         "A_dBIC_min": 0.0, "A_eps_sd_min": 0.04, "A_bench_sigma": 1.0}


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(5)
    z = np.sort(rng.uniform(0.02, 1.1, 40))
    zhel = z * (1 + 1e-4 * rng.normal(size=40))
    mu_true = mu_model(z, zhel, 0.31)
    sig = 0.12
    mu = mu_true + 0.4 + rng.normal(scale=sig, size=40)
    W = np.eye(40) / sig ** 2
    hz = HzData(z=np.array([0.2, 0.6, 1.0]), H=np.array([75.0, 95.0, 125.0]),
                sig=np.array([5.0, 6.0, 8.0]))
    bao = BAOData(z=np.array([0.5, 0.5]), kind=["DM_over_rd", "DH_over_rd"],
                  value=np.array([13.5, 21.0]), sig=np.array([0.3, 0.5]))
    return {"zHD": z, "zHEL": zhel, "mu": mu, "W": W, "hz": hz, "bao": bao,
            "n_sn": 40, "cov_kind": "SYNTH"}


class TestOfficialReference:
    def test_weighted_omega_m_from_committed_chain(self):
        off = official_omega_m()
        assert off["n_rows"] == 16384
        assert off["n_eff"] > 5000
        assert off["mean"] == pytest.approx(0.3306, abs=0.002)
        assert off["sd"] == pytest.approx(0.0154, abs=0.002)
        assert off["p16"] < off["median"] < off["p84"]
        # el soporte del prior propio es IDÉNTICO al oficial (like-for-like)
        assert off["prior_support"][0] == pytest.approx(PRIOR_OMEGA_M[0], abs=0.001)
        assert off["prior_support"][1] == pytest.approx(PRIOR_OMEGA_M[1], abs=0.001)


class TestModelComposition:
    def test_lcdm_is_mcmc_at_eps_zero(self, synthetic):
        a = chi2_blocks((67.4, 0.31), "lcdm", synthetic)
        b = chi2_blocks((67.4, 0.31, 0.0, 8.9), "mcmc", synthetic)
        for kk in ("sn", "cc", "bao", "total"):
            assert a[kk] == pytest.approx(b[kk], rel=1e-12)

    def test_sn_block_independent_of_H0(self, synthetic):
        """M marginalizada absorbe 5·log10(H0): el bloque SN no depende de
        la H0 del vector; CC y BAO sí."""
        a = chi2_blocks((60.5, 0.31, 0.02, 8.9), "mcmc", synthetic)
        b = chi2_blocks((79.5, 0.31, 0.02, 8.9), "mcmc", synthetic)
        assert a["sn"] == pytest.approx(b["sn"], rel=1e-12)
        assert a["cc"] != b["cc"] and a["bao"] != b["bao"]

    def test_theta_full_and_k(self):
        assert theta_full((67.4, 0.31), "lcdm")[2] == 0.0
        assert K_PARAMS == {"lcdm": 2, "mcmc": 4}
        with pytest.raises(ValueError):
            theta_full((1, 2, 3), "wcdm")

    def test_priors_frozen_bounds(self):
        assert BOUNDS["lcdm"][1] == (0.10, 0.50)
        assert log_prior((67.4, 0.55), "lcdm") == -np.inf      # fuera de Ω_m
        assert log_prior((67.4, 0.31, 0.15, 8.9), "mcmc") == -np.inf
        assert log_prior((67.4, 0.31, 0.012, 25.0), "mcmc") == -np.inf
        assert np.isfinite(log_prior((67.4, 0.31, 0.012, 8.9), "mcmc"))
        # el gaussiano de ε está centrado en 0.012, no en 0
        assert log_prior((67.4, 0.31, 0.012, 8.9), "mcmc") > \
            log_prior((67.4, 0.31, 0.0, 8.9), "mcmc")

    def test_log_prob_finite_and_recovers_truth_region(self, synthetic):
        lp_true = log_prob((67.4, 0.31), "lcdm", synthetic)
        lp_far = log_prob((67.4, 0.45), "lcdm", synthetic)
        assert np.isfinite(lp_true) and lp_true > lp_far
        assert log_prob((67.4, 0.9), "lcdm", synthetic) == -np.inf

    def test_n_points(self, synthetic):
        assert n_points(synthetic) == 40 + 3 + 2
        assert n_points(synthetic, blocks=("sn",)) == 40


class TestOutcomeRule:
    def test_order_C_B_A_indeterminate(self):
        # C domina: benchmark a 4σ aunque todo lo demás sea aburrido
        assert classify_outcome(4.0, 1.0, +5.0, 0.05, True, RULES) == "C"
        assert classify_outcome(0.1, 1.5, +5.0, 0.05, True, RULES) == "C"
        # B: BIC pro-MCMC > 2 y ε separado de 0
        assert classify_outcome(0.1, 1.0, -3.0, 0.02, False, RULES) == "B"
        # A: aburrido
        assert classify_outcome(0.3, 1.02, +6.0, 0.045, True, RULES) == "A"
        # ε separado de cero pero sin preferencia BIC → indeterminado
        assert classify_outcome(0.3, 1.02, +6.0, 0.045, False, RULES) == \
            "INDETERMINADO"
        # prior no dominante (σ pequeña) sin preferencia → indeterminado
        assert classify_outcome(0.3, 1.02, +6.0, 0.02, True, RULES) == \
            "INDETERMINADO"


class TestNoRealMuInTests:
    def test_suite_never_touches_real_mu(self):
        """Los tests de esta aplicación no leen la columna MU real: solo el
        ejecutor, tras las dos barreras."""
        # Comprobación por AST: ningún módulo de test IMPORTA los lectores
        # del HD real (mencionarlos en cadenas de texto es lícito).
        import ast
        forbidden = {"load_dovekie_hd", "load_real_data"}
        for rel in ("tests/test_dovekie_real.py",
                    "tests/test_dovekie_real_lock.py"):
            p = REPO / rel
            if not p.exists():
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported |= {a.name for a in node.names}
                elif isinstance(node, ast.Attribute):
                    imported.add(node.attr)
            assert not (imported & forbidden), f"{rel} importa/usa {imported & forbidden}"
