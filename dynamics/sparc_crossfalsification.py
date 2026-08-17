"""Frente 5E — pipeline de falsación cruzada con A CONGELADA.

Funciones PURAS: reciben arrays y la CrossFalsificationConfig
inmutable (dynamics/sculptor_transfer.py). Este módulo NO estima A,
no la ajusta por galaxia, no minimiza χ² respecto de A y no
selecciona galaxias por Δχ² — el candado test_no_sparc_fit_of_A
escanea esta fuente.

Composición dinámica (consecuencia de la formulación, no elección):
las aceleraciones se suman y v² = R·g, luego

    v_model²(R) = v_bar²(R) + v_cronos²(R)

(en CUADRATURA — nunca suma lineal de velocidades). Con A → 0,
v_model ≡ v_bar (límite de recuperación, testeado).

CONDICIONAL EXPUESTO — totales negativos: con la convención de gas
con signo, v_bar² + v_cronos² puede ser < 0 en puntos interiores de
galaxias dominadas por gas; ahí la identidad NO aplica y
predict_total_curve devuelve v_model = 0 (elección declarada y
preinscrita, idéntica para M0 y M1). El análisis cuenta y publica
esos puntos por galaxia (count_negative_total → n_negative_total);
ninguno se descarta.

APROXIMACIÓN DECLARADA — χ² diagonal: los errores por punto del
catálogo se tratan como independientes; los sistemáticos
correlacionados (distancia, inclinación) entran por las hipótesis de
sensibilidad preinscritas, no por la covarianza del χ².

Alcances (model_scope, preinscritos):
- 5E-A_exponential: exactamente la clase del 5C — ρ del plano medio
  del disco exponencial (Σ0 = Υ_d·L_disk/(2π·R_d²), ζ congelado);
  el test PRINCIPAL.
- 5E-B_extended: extensión bariónica realista (disco+bulbo+gas) SOLO
  si el catálogo trae la fotometría — hipótesis NUEVA, jamás
  presentada como la demostración del 5C.
"""

from __future__ import annotations

import numpy as np

from .disc_cronos import v_cronos_sq
from .sculptor_transfer import CrossFalsificationConfig


def predict_baryonic_curve(v_gas, v_disk, v_bul,
                           upsilon_disk: float,
                           upsilon_bulge: float) -> np.ndarray:
    """v_bar²(R) [(km/s)²] desde las componentes del catálogo.

    Convención estándar SPARC (a CONFIRMAR contra la cabecera real en
    schema_report antes del análisis observacional): las componentes
    van en km/s a Υ = 1 y el gas conserva su signo (v_gas < 0 denota
    densidad superficial negativa efectiva en el interior):
    v_bar² = Υ_d·v_disk² + Υ_b·v_bul² + sign(v_gas)·v_gas²."""
    v_gas = np.asarray(v_gas, float)
    v_disk = np.asarray(v_disk, float)
    v_bul = np.asarray(v_bul, float)
    return (upsilon_disk * v_disk ** 2 + upsilon_bulge * v_bul ** 2
            + np.sign(v_gas) * v_gas ** 2)


def predict_cronos_fixed_A(R_pc, Sigma0: float, R_d_pc: float,
                           cfg: CrossFalsificationConfig) -> np.ndarray:
    """v_cronos²(R) [(km/s)²] de la clase 5E-A: EXACTAMENTE la
    realización exponencial del 5C, con la A congelada de la config
    (nunca un argumento libre A)."""
    return v_cronos_sq(np.asarray(R_pc, float), Sigma0, R_d_pc,
                       cfg.zeta_disc, cfg.A_sculptor)


def predict_total_curve(v_bar_sq: np.ndarray,
                        v_cronos_sq_arr: np.ndarray) -> np.ndarray:
    """v_model = sqrt(v_bar² + v_cronos²) — composición en cuadratura.

    Donde el total es < 0 (gas con signo; condicional expuesto en el
    docstring del módulo) devuelve 0.0 — elección declarada, aplicada
    por igual a M0 (A = 0) y M1; los puntos afectados se cuentan con
    count_negative_total y se publican por galaxia."""
    total = np.asarray(v_bar_sq, float) + np.asarray(v_cronos_sq_arr,
                                                     float)
    return np.sqrt(np.clip(total, 0.0, None))


def count_negative_total(v_bar_sq: np.ndarray,
                         v_cronos_sq_arr: np.ndarray) -> int:
    """Nº de puntos con v_bar² + v_cronos² < 0 (el condicional del
    clip, publicado por galaxia como n_negative_total)."""
    total = np.asarray(v_bar_sq, float) + np.asarray(v_cronos_sq_arr,
                                                     float)
    return int(np.sum(total < 0.0))


def chi_square(v_model: np.ndarray, v_obs: np.ndarray,
               e_v: np.ndarray) -> float:
    """χ² = Σ [(v_model − v_obs)/e_v]² (adimensional)."""
    resid = (np.asarray(v_model, float) - np.asarray(v_obs, float)) \
        / np.asarray(e_v, float)
    return float(np.sum(resid ** 2))


def shape_diagnostics(R_pc, R_d_pc: float, v_cronos: np.ndarray,
                      v_obs: np.ndarray,
                      v_bar_sq: np.ndarray) -> dict:
    """Métricas estructurales preinscritas (por galaxia):

    - B_inner = v_cronos(2R_d/3)/v_obs(2R_d/3)  (interpolado);
    - F_outer = v_cronos/sqrt(max(v_obs² − v_bar², 0)) promediado en
      x ∈ [3, 4] donde el denominador esté definido (NaN si no);
    - ratio_x5 = v_cronos(2R_d/3)/v_cronos(4R_d) — la identidad ×5 de
      la clase 5C convertida en observable (NaN si 4R_d queda fuera
      del rango medido)."""
    R = np.asarray(R_pc, float)
    if R.size > 1 and not np.all(np.diff(R) > 0.0):
        raise ValueError(
            "shape_diagnostics exige radios estrictamente crecientes "
            "(np.interp devuelve basura en silencio si no)")
    vc = np.asarray(v_cronos, float)
    vo = np.asarray(v_obs, float)
    vb2 = np.asarray(v_bar_sq, float)
    x_in = 2.0 * R_d_pc / 3.0
    x_out = 4.0 * R_d_pc
    in_range = R.min() <= x_in <= R.max()
    out_range = R.min() <= x_out <= R.max()
    vc_in = float(np.interp(x_in, R, vc)) if in_range else np.nan
    vo_in = float(np.interp(x_in, R, vo)) if in_range else np.nan
    vc_out = float(np.interp(x_out, R, vc)) if out_range else np.nan
    B_inner = vc_in / vo_in if in_range and vo_in > 0.0 else np.nan
    sel = (R >= 3.0 * R_d_pc) & (R <= 4.0 * R_d_pc)
    deficit = vo[sel] ** 2 - vb2[sel]
    ok = deficit > 0.0
    F_outer = float(np.mean(vc[sel][ok] / np.sqrt(deficit[ok]))) \
        if ok.any() else np.nan
    ratio_x5 = vc_in / vc_out if (in_range and out_range
                                  and vc_out > 0.0) else np.nan
    return {"B_inner": B_inner, "F_outer": F_outer,
            "ratio_x5": ratio_x5}


def per_galaxy_row(galaxy: str, quality_flag, n_points: int,
                   sigma_char: float, R_d_pc: float,
                   chi2_bar: float, chi2_cronos: float,
                   diag: dict, cfg: CrossFalsificationConfig,
                   model_scope: str) -> dict:
    """Fila del per_galaxy.csv preinscrito. A_fitted es SIEMPRE False
    en el test principal (testeado): A llega congelada de la config."""
    return {
        "galaxy": galaxy,
        "quality_flag": quality_flag,
        "N_points": int(n_points),
        "Sigma_or_equivalent": float(sigma_char),
        "Rd": float(R_d_pc),
        "chi2_baryons": float(chi2_bar),
        "chi2_baryons_cronos": float(chi2_cronos),
        "delta_chi2": float(chi2_cronos - chi2_bar),
        "reduced_chi2_baryons": float(chi2_bar / max(n_points, 1)),
        "reduced_chi2_cronos": float(chi2_cronos / max(n_points, 1)),
        "inner_bump_metric": diag["B_inner"],
        "outer_support_metric": diag["F_outer"],
        "shape_x5_ratio": diag["ratio_x5"],
        "n_negative_total": int(diag.get("n_negative_total", 0)),
        "A_used": cfg.A_sculptor,
        "A_fitted": False,
        "model_scope": model_scope,
    }


def aggregate_summary(delta_chi2: np.ndarray, n_points: np.ndarray,
                      cfg: CrossFalsificationConfig) -> dict:
    """Agregados preinscritos + bootstrap entre galaxias (semilla y n
    de la config congelada). La suma global nunca se usa sola."""
    d = np.asarray(delta_chi2, float)
    n = np.asarray(n_points, float)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    idx = rng.integers(0, d.size, size=(cfg.bootstrap_n, d.size))
    boot_median = np.median(d[idx], axis=1)
    boot_sum = np.sum(d[idx], axis=1)
    return {
        "n_galaxies": int(d.size),
        "n_points_total": int(n.sum()),
        "sum_delta_chi2": float(d.sum()),
        "median_delta_chi2": float(np.median(d)),
        "frac_improved": float(np.mean(d < 0.0)),
        "frac_worsened": float(np.mean(d > 0.0)),
        "bootstrap": {
            "median_CI68": [float(np.percentile(boot_median, 16)),
                            float(np.percentile(boot_median, 84))],
            "median_CI95": [float(np.percentile(boot_median, 2.5)),
                            float(np.percentile(boot_median, 97.5))],
            "sum_CI68": [float(np.percentile(boot_sum, 16)),
                         float(np.percentile(boot_sum, 84))],
            "sum_CI95": [float(np.percentile(boot_sum, 2.5)),
                         float(np.percentile(boot_sum, 97.5))],
        },
    }


def decide_verdict_arm(summary: dict) -> str:
    """La REGLA DE DECISIÓN preinscrita (tres brazos, fijada antes de
    ingerir datos; el brazo de éxito existe y es tan explícito como
    el de fallo):

    - "systematic_worsening": mediana(Δχ²) > 0 con CI95 bootstrap
      excluyendo 0 Y fracción mejorada < 1/3 ⟹ camino A del árbol.
    - "support": mediana(Δχ²) < 0 con CI95 excluyendo 0 Y fracción
      mejorada > 1/2 ⟹ compatibilidad que exige validación fuera de
      muestra (camino B; nunca claim universal).
    - "indeterminate": cualquier otro caso ⟹ compatibilidad parcial
      o indecisión; más datos antes de cualquier claim.
    """
    lo, hi = summary["bootstrap"]["median_CI95"]
    med = summary["median_delta_chi2"]
    if med > 0.0 and lo > 0.0 and summary["frac_improved"] < 1.0 / 3.0:
        return "systematic_worsening"
    if med < 0.0 and hi < 0.0 and summary["frac_improved"] > 0.5:
        return "support"
    return "indeterminate"


def predeclared_subsets(rows: list[dict],
                        sigma_median: float) -> dict:
    """Los subconjuntos PREINSCRITOS, calculados solo desde columnas
    observables de las filas (nunca desde Δχ²): full, quality
    (bandera del catálogo ≤ 2 si existe), HSB/LSB (por la mediana
    sigma_median de la columna observable fijada en schema_report,
    calculada ANTES de mirar Δχ²), gas_dominated (bandera observable
    que el parser rellenará). Devuelve índices por subconjunto."""
    full = list(range(len(rows)))
    quality = [i for i, r in enumerate(rows)
               if r.get("quality_flag") is None
               or r["quality_flag"] <= 2]
    hsb = [i for i, r in enumerate(rows)
           if r["Sigma_or_equivalent"] > sigma_median]
    lsb = [i for i, r in enumerate(rows)
           if r["Sigma_or_equivalent"] <= sigma_median]
    gas = [i for i, r in enumerate(rows)
           if r.get("gas_dominated") is True]
    return {"full_sample": full, "quality_sample": quality,
            "HSB_subset": hsb, "LSB_subset": lsb,
            "gas_dominated_subset": gas}
