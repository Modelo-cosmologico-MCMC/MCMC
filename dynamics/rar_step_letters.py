"""Letras del escalón de la RAR (ley local saturante) sobre la predicción congelada `9566a2ac0457` (bloque P0 del autor
del 26-oct-2026, «Las letras del escalón»; sin datos, sin letra, sin congelar).

La predicción (results/2026-09-22_dictionary_eps_c/rar_step_prediction.json, sha256 9566a2ac0457…) no se toca: aquí
solo se escribe cómo leerla en los datos, galaxia a galaxia.

  - Residuo. δ = log g_obs − log g_ref(g_bar), con g_ref = g_bar/(1 − e^{−√(g_bar/g†)}) y g† ajustado una sola vez sobre
    toda la submuestra de calidad (mínimos cuadrados en δ): la nula «función continua de g_bar», con un parámetro.
  - Dónde mirar. x = R/R*, con R* el radio en que la densidad del plano medio del disco, Υ_d·Σ_d(R)/(2h_z), vale ρ*;
    Σ_d(R) = SB_disk(0)·e^{−R/R_d} de la tabla maestra; Υ_d = 0.5 y h_z = ζ·R_d con ζ = 0.15 (convenciones congeladas
    del 5E); sensibilidad ζ = 0.1 y 0.2.
  - Estadístico. Por galaxia, S_g = media de δ en x ∈ [0.8, 1.3] menos media en x ∈ [0.5, 0.7] ∪ [1.6, 2.5]; entran las
    galaxias con al menos dos puntos en cada ventana. S es la media entre galaxias y σ_S su error por bootstrap.
  - Predicción. S_pred es el mismo estadístico sobre δ_pred = log(1 + g_C/g_obs), con g_C = c²·ε_c'(ρ)·ρ/R_d del disco
    de cada galaxia (la lectura aditiva: la ley local como fuerza añadida a lo que ya hay).

HALLAZGO DEL EJECUTOR (27-oct, pendiente del autor). Si la ley aditiva fuera exacta, los datos ya contendrían g_C y el
residuo medido sería log[g_obs/(g_obs − g_C)], no log(1 + g_C/g_obs): coinciden a primer orden en g_C/g_obs, pero con
ρ* = 0.001, donde g_C/g_obs llega a ~1 cerca de R*, la forma del documento queda por debajo del escalón real y la A se
vuelve inalcanzable (galaxias sintéticas: S = 0.19 frente a S_pred = 0.11, unas 6σ_S). Con ρ* = 0.01–0.03 la diferencia
no se ve. Se calculan las dos formas (`DELTA_PRED_FORMS`); cuál entra en la letra es decisión del autor.

Letras por punto (ε_max, ρ*): INDETERMINADO si S_pred < 3σ_S (no comprobable), si hay menos de 20 galaxias con las dos
ventanas, si la letra cambia entre los tres ζ o por las puertas de bytes; A si S ≥ 3σ_S y |S − S_pred| ≤ 2σ_S; C si
S ≤ S_pred − 3σ_S; B el resto. La lectura literal (la ley saturante como única adición a los bariones) se publica
aparte: compara log(g_obs/g_bar) en x ∈ [1.6, 2.5] con la predicha, casi cero allí.

Estatuto: instrumento (E8), sin datos; ninguna cifra es señal (E13).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dynamics.epsilon_c_saturating import deps_c_drho
from dynamics.mcv_sparc import baryon_v2
from dynamics.sparc_bank import KMS2_PER_KPC_SI
from dynamics.weak_field import C_KMS

PREDICTION = Path(__file__).resolve().parent.parent / "results" / "2026-09-22_dictionary_eps_c" / "rar_step_prediction.json"
PREDICTION_SHA256_PREFIX = "9566a2ac0457"

STEP_SPEC = {
    "window_in": [0.8, 1.3], "windows_out": [[0.5, 0.7], [1.6, 2.5]], "min_points": 2, "n_gal_min": 20,
    "zeta": 0.15, "zeta_sensitivity": [0.1, 0.2], "ups_disk": 0.5, "ups_bul": 0.7,
    "testable_sigma": 3.0, "A_tol_sigma": 2.0, "C_sigma": 3.0, "n_boot": 10000, "seed": 20261026,
    "literal_window": [1.6, 2.5], "g_dagger_bounds_si": [1e-11, 1e-9],
}


def frozen_points(path: Path = PREDICTION) -> list[dict]:
    """Los cuatro puntos (ε_max, ρ*) de la predicción congelada, tras comprobar su sha256."""
    raw = Path(path).read_bytes()
    if not hashlib.sha256(raw).hexdigest().startswith(PREDICTION_SHA256_PREFIX):
        raise RuntimeError("la predicción del escalón no es la congelada (sha256 distinto de 9566a2ac0457…)")
    return [{k: p[k] for k in ("label", "eps_max", "rho_star")} for p in json.loads(raw)["points"]]


def g_ref(g_bar, g_dagger):
    """La RAR de un parámetro: g_bar/(1 − e^{−√(g_bar/g†)}) (unidades comunes)."""
    g = np.asarray(g_bar, float)
    return g / -np.expm1(-np.sqrt(g / g_dagger))


def fit_g_dagger(g_bar, g_obs, spec: dict = STEP_SPEC, n_grid: int = 401, n_zoom: int = 8) -> float:
    """g† (m/s²) por mínimos cuadrados de δ = log g_obs − log g_ref sobre todos los puntos de la submuestra de calidad;
    g_bar y g_obs en (km/s)²/kpc. Búsqueda en rejilla de log g† que se estrecha alrededor del mínimo (sin maquinaria de
    optimización, como el resto del análisis de SPARC)."""
    gb, go = np.asarray(g_bar, float), np.asarray(g_obs, float)
    lo, hi = (math.log10(v) for v in spec["g_dagger_bounds_si"])

    def cost(lg):
        return float(np.sum((np.log10(go) - np.log10(g_ref(gb, 10.0 ** lg / KMS2_PER_KPC_SI))) ** 2))
    for _ in range(n_zoom):
        grid = np.linspace(lo, hi, n_grid)
        j = int(np.argmin([cost(x) for x in grid]))
        step = grid[1] - grid[0]
        lo, hi = grid[max(j - 2, 0)], grid[min(j + 2, n_grid - 1)]
        if step < 1e-12:
            break
    return 10.0 ** grid[j]


def galaxy_accelerations(gal, spec: dict = STEP_SPEC) -> dict:
    """g_obs = V_obs²/R y g_bar = V_bar²/R (Υ_d = 0.5, Υ_b = 0.7, D e i del catálogo) en (km/s)²/kpc; solo los puntos con
    V_bar² > 0 (los demás se cuentan)."""
    v2b, _ = baryon_v2(gal.v_gas, gal.v_disk, gal.v_bul, spec["ups_disk"], spec["ups_bul"])
    R = np.asarray(gal.R, float)
    keep = v2b > 0.0
    return {"R": R[keep], "g_obs": np.asarray(gal.v_obs, float)[keep] ** 2 / R[keep], "g_bar": v2b[keep] / R[keep],
            "n_dropped": int((~keep).sum())}


def r_star(gal, rho_star: float, zeta: float, ups_disk: float = 0.5) -> float | None:
    """R* donde Υ_d·Σ_d(R)/(2ζR_d) = ρ* (M☉/pc³); None si el disco no llega a ρ* ni en el centro."""
    if gal.SB_disk0 is None or gal.R_d is None or gal.R_d <= 0.0:
        return None
    rho0 = ups_disk * gal.SB_disk0 / (2.0 * zeta * gal.R_d * 1e3)
    return gal.R_d * math.log(rho0 / rho_star) if rho0 > rho_star else None


def window_statistic(x, delta, spec: dict = STEP_SPEC) -> float | None:
    """S_g = media de δ en la ventana interior menos media en las exteriores; None sin dos puntos en cada una."""
    x, d = np.asarray(x, float), np.asarray(delta, float)
    a, b = spec["window_in"]
    inside = (x >= a) & (x <= b)
    outside = np.zeros_like(inside)
    for lo, hi in spec["windows_out"]:
        outside |= (x >= lo) & (x <= hi)
    if inside.sum() < spec["min_points"] or outside.sum() < spec["min_points"]:
        return None
    return float(np.mean(d[inside]) - np.mean(d[outside]))


DELTA_PRED_FORMS = ("documento", "autoconsistente")


def predicted_delta(gal, acc: dict, point: dict, zeta: float, ups_disk: float = 0.5, form: str = "documento"):
    """δ_pred con g_C = c²·ε_c'(ρ)·ρ/R_d en (km/s)²/kpc y ρ = Υ_d·Σ_d(R)/(2ζR_d) (M☉/pc³): «documento»,
    log(1 + g_C/g_obs); «autoconsistente», log[g_obs/(g_obs − g_C)] (NaN donde g_obs ≤ g_C)."""
    rho = ups_disk * gal.SB_disk0 * np.exp(-acc["R"] / gal.R_d) / (2.0 * zeta * gal.R_d * 1e3)
    g_C = C_KMS ** 2 * deps_c_drho(rho, point["eps_max"], point["rho_star"]) * rho / gal.R_d
    if form == "documento":
        return np.log10(1.0 + g_C / acc["g_obs"]), g_C
    if form == "autoconsistente":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(acc["g_obs"] > g_C, np.log10(acc["g_obs"] / (acc["g_obs"] - g_C)), np.nan), g_C
    raise ValueError(f"forma de δ_pred desconocida: {form!r}")


def step_statistics(galaxies: list, point: dict, zeta: float, g_dagger_si: float, spec: dict = STEP_SPEC,
                    form: str = "documento") -> dict:
    """S, σ_S (bootstrap entre galaxias) y S_pred para un punto (ε_max, ρ*), un ζ y una forma de δ_pred."""
    gd = g_dagger_si / KMS2_PER_KPC_SI
    S, S_pred, names = [], [], []
    for gal in galaxies:
        Rs = r_star(gal, point["rho_star"], zeta, spec["ups_disk"])
        if Rs is None:
            continue
        acc = galaxy_accelerations(gal, spec)
        x = acc["R"] / Rs
        s = window_statistic(x, np.log10(acc["g_obs"]) - np.log10(g_ref(acc["g_bar"], gd)), spec)
        dp, _ = predicted_delta(gal, acc, point, zeta, spec["ups_disk"], form)
        ok = np.isfinite(dp)
        sp = window_statistic(x[ok], dp[ok], spec)
        if s is not None and sp is not None:
            S.append(s)
            S_pred.append(sp)
            names.append(gal.name)
    S, S_pred = np.array(S), np.array(S_pred)
    if S.size == 0:
        return {"n": 0, "S": None, "sigma_S": None, "S_pred": None, "galaxies": []}
    rng = np.random.default_rng(spec["seed"])
    boot = np.mean(S[rng.integers(0, S.size, size=(spec["n_boot"], S.size))], axis=1)
    return {"n": int(S.size), "S": float(S.mean()), "sigma_S": float(boot.std(ddof=1)), "S_pred": float(S_pred.mean()),
            "galaxies": names}


def letter_one(st: dict, spec: dict = STEP_SPEC) -> dict:
    """Letra de un punto con un ζ (sin la puerta de los tres ζ ni la de bytes)."""
    if st["n"] < spec["n_gal_min"]:
        return {"letter": "INDETERMINADO", "reason": f"menos de {spec['n_gal_min']} galaxias con las dos ventanas"}
    S, sS, Sp = st["S"], st["sigma_S"], st["S_pred"]
    if Sp < spec["testable_sigma"] * sS:
        return {"letter": "INDETERMINADO", "reason": "no comprobable: S_pred por debajo de 3σ_S"}
    if S >= spec["testable_sigma"] * sS and abs(S - Sp) <= spec["A_tol_sigma"] * sS:
        return {"letter": "A", "reason": "escalón medido y compatible con el previsto"}
    if S <= Sp - spec["C_sigma"] * sS:
        return {"letter": "C", "reason": "el escalón falta donde la predicción supera el ruido"}
    return {"letter": "B", "reason": "comprobable, ni A ni C"}


def step_letters(galaxies: list, g_dagger_si: float, bytes_ok: bool, points: list[dict] | None = None,
                 spec: dict = STEP_SPEC, form: str = "documento") -> list[dict]:
    """Letras por punto (ε_max, ρ*): las tres ζ y la puerta de que la letra no cambie entre ellas."""
    out = []
    for p in points or frozen_points():
        per_zeta = {}
        for z in [spec["zeta"], *spec["zeta_sensitivity"]]:
            st = step_statistics(galaxies, p, z, g_dagger_si, spec, form)
            per_zeta[str(z)] = {**st, **letter_one(st, spec)}
        letters = {v["letter"] for v in per_zeta.values()}
        if not bytes_ok:
            letter, reason = "INDETERMINADO", "puertas de bytes"
        elif len(letters) > 1:
            letter, reason = "INDETERMINADO", "la letra cambia entre los tres ζ"
        else:
            letter, reason = per_zeta[str(spec["zeta"])]["letter"], per_zeta[str(spec["zeta"])]["reason"]
        out.append({"point": p, "form": form, "letter": letter, "reason": reason, "per_zeta": per_zeta})
    return out


def literal_reading(galaxies: list, point: dict, zeta: float = 0.15, spec: dict = STEP_SPEC) -> dict:
    """La lectura literal, sin letra: media de log(g_obs/g_bar) en x ∈ [1.6, 2.5] frente a la predicha
    log((g_bar + g_C)/g_bar) en la misma ventana."""
    obs, pred = [], []
    lo, hi = spec["literal_window"]
    for gal in galaxies:
        Rs = r_star(gal, point["rho_star"], zeta, spec["ups_disk"])
        if Rs is None:
            continue
        acc = galaxy_accelerations(gal, spec)
        x = acc["R"] / Rs
        w = (x >= lo) & (x <= hi)
        if not w.any():
            continue
        _, g_C = predicted_delta(gal, acc, point, zeta, spec["ups_disk"])
        obs.append(float(np.mean(np.log10(acc["g_obs"][w] / acc["g_bar"][w]))))
        pred.append(float(np.mean(np.log10((acc["g_bar"][w] + g_C[w]) / acc["g_bar"][w]))))
    return {"n": len(obs), "observed_mean_dex": float(np.mean(obs)) if obs else None,
            "predicted_mean_dex": float(np.mean(pred)) if pred else None}
