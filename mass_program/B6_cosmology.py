"""B6 — Resultados cosmológicos del ajuste global.

Re-exporta los resultados clave del ajuste bayesiano: H₀, σ₈, S₈,
ΔBIC, ΔAIC, ε ≡ δ₀ y z_trans. Las funciones físicas H(z), ρ_id(z),
Λ_rel(z) viven en `cosmology.background`.
"""

from mcmc_ontology import constants as C


def global_fit_results() -> dict:
    """Resumen de la Tabla 17 (ajuste global)."""
    return {
        "H0_kms_Mpc":  (C.H0_MCMC, C.H0_ERR),
        "Omega_m":     (0.300, 0.015),
        "epsilon":     (C.EPSILON_0, 0.003),
        "z_trans":     (C.Z_TRANS, 0.4),
        "sigma8":      C.SIGMA8_MCMC,
        "S8":          C.S8_MCMC,
        "delta_chi2":  C.DELTA_CHI2,
        "delta_BIC":   C.DELTA_BIC,
    }


def tensions() -> dict:
    """Tensiones cosmológicas aliviadas por MCMC."""
    return {
        "H0_tension_LCDM_sigma":  4.0,
        "H0_tension_MCMC_sigma":  2.0,
        "S8_tension_LCDM_sigma":  2.5,
        "S8_tension_MCMC_sigma":  1.5,
        "evidence_jeffreys":      "fuerte (ΔBIC < -6)",
    }
