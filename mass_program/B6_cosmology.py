"""B6 — Valores cosmológicos de referencia del corpus.

Re-exporta los VALORES DE REFERENCIA del ajuste global documentado en el
corpus (v32): H₀, σ₈, S₈, ΔBIC, ε (la amplitud ε_Λ de la transición; NO
se identifica con δ₀ — regla canónica) y z_trans. NO son salidas de este
código: reproducirlos requiere los datos observacionales reales en `data/`
y el ajuste bayesiano de producción. Las funciones físicas H(z), ρ_id(z),
Λ_rel(z) viven en `cosmology.background`.
"""

from mcmc_ontology import constants as C


def global_fit_results() -> dict:
    """Valores de referencia de la Tabla 17 del corpus v32 (no salidas del código)."""
    return {
        "H0_kms_Mpc":  (C.H0_MCMC, C.H0_ERR),
        "Omega_m":     (0.300, 0.015),
        "epsilon":     (C.EPSILON_LAMBDA, 0.003),
        "z_trans":     (C.Z_TRANS, 0.4),
        "sigma8":      C.SIGMA8_MCMC,
        "S8":          C.S8_MCMC,
        "delta_chi2":  C.DELTA_CHI2,
        "delta_BIC":   C.DELTA_BIC,
    }


def tensions() -> dict:
    """Estimaciones de tensiones cosmológicas según el corpus (v32).

    Las sigmas y la etiqueta de evidencia son valores de referencia del
    corpus, escritos a mano: este código no las calcula.
    """
    return {
        "H0_tension_LCDM_sigma":  4.0,
        "H0_tension_MCMC_sigma":  2.0,
        "S8_tension_LCDM_sigma":  2.5,
        "S8_tension_MCMC_sigma":  1.5,
        "evidence_jeffreys":      "fuerte (ΔBIC < -6)",
    }
