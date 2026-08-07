#!/usr/bin/env python
"""Ajuste de producción v2 — los cinco bloques: CC + BAO + SNe + CMB + fσ8.

La corrida de la reconciliación (ronda 4, opción B): añade al ajuste de
fondo los dos likelihoods de los que procedía la ventaja del corpus —
la geometría comprimida del CMB y el crecimiento fσ8. Mismo protocolo
que v1: ΛCDM con la MISMA maquinaria (ε=0 exacto, Prop. A.1), criterios
de información con fórmulas explícitas, cadenas versionadas, y el
resultado en portada sea cual sea.

Parámetros: MCMC k=6 (H0, Ωm, ε, z_trans, ω_b, σ8);
            ΛCDM k=4 (H0, Ωm, ω_b, σ8).

SALVEDAD DECLARADA: z* y r_s usan fórmulas de ajuste (Hu & Sugiyama
1996) — sesgo conocido ~0.3% en l_A frente al cálculo exacto de
recombinación, idéntico en ambos modelos: la comparación diferencial
(ΔAIC/ΔBIC) es justa; los posteriores absolutos se leen con esa
salvedad. El likelihood CMB es diagonal (sin correlaciones R-l_A-ω_b).

Uso:
    python scripts/download_data.py all
    python scripts/run_production_fit2.py [--nsteps 2000] [--nwalkers 24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_production_fit import run_sampler, save_chain, summarize  # noqa: E402

from cosmology.bayesian_fit import (  # noqa: E402
    information_criteria,
    load_bao_data,
    load_Hz_data,
    load_sne_data,
    log_like_bao,
    log_like_Hz,
    log_like_sne,
    log_prior,
    log_prior_lcdm,
)
from cosmology.extended_likelihoods import (  # noqa: E402
    cmb_compressed_loglike,
    load_fsigma8_data,
    rsd_loglike,
)
from mcmc_ontology import constants as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"

# Datos (globales del script para los workers de emcee)
_DATA = {}


def _extra_priors(omega_b: float, sigma8: float) -> float:
    if not (0.019 < omega_b < 0.026):
        return -np.inf
    if not (0.50 < sigma8 < 1.10):
        return -np.inf
    return 0.0


def log_prob_mcmc6(theta6) -> float:
    H0, Om, eps, z_trans, wb, s8 = theta6
    lp = log_prior((H0, Om, eps, z_trans))
    lp += _extra_priors(wb, s8)
    if not np.isfinite(lp):
        return -np.inf
    th4 = (H0, Om, eps, z_trans)
    ll = log_like_Hz(th4, _DATA["hz"])
    ll += log_like_sne(th4, _DATA["sne"])
    ll += log_like_bao(th4, _DATA["bao"])
    ll += cmb_compressed_loglike(th4, wb)
    ll += rsd_loglike(th4, s8, *_DATA["rsd"])
    return lp + ll


def log_prob_lcdm4(theta4) -> float:
    H0, Om, wb, s8 = theta4
    lp = log_prior_lcdm((H0, Om))
    lp += _extra_priors(wb, s8)
    if not np.isfinite(lp):
        return -np.inf
    th4 = (H0, Om, 0.0, C.Z_TRANS)   # ε = 0 exacto (Prop. A.1)
    ll = log_like_Hz(th4, _DATA["hz"])
    ll += log_like_sne(th4, _DATA["sne"])
    ll += log_like_bao(th4, _DATA["bao"])
    ll += cmb_compressed_loglike(th4, wb)
    ll += rsd_loglike(th4, s8, *_DATA["rsd"])
    return lp + ll


def _loglike_parts(th4, wb, s8) -> dict:
    parts = {
        "Hz": log_like_Hz(th4, _DATA["hz"]),
        "SNe": log_like_sne(th4, _DATA["sne"]),
        "BAO": log_like_bao(th4, _DATA["bao"]),
        "CMB": cmb_compressed_loglike(th4, wb),
        "fs8": rsd_loglike(th4, s8, *_DATA["rsd"]),
    }
    parts["total"] = float(sum(parts.values()))
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=2000)
    ap.add_argument("--nwalkers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    opts = ap.parse_args()
    burn = opts.nsteps // 2

    _DATA["hz"] = load_Hz_data()
    _DATA["bao"] = load_bao_data()
    _DATA["sne"] = load_sne_data()
    _DATA["rsd"] = load_fsigma8_data()
    n_points = (len(_DATA["hz"].z) + len(_DATA["bao"].z)
                + len(_DATA["sne"].z) + 3 + len(_DATA["rsd"][0]))
    datasets = (f"{len(_DATA['hz'].z)} CC + {len(_DATA['bao'].z)} BAO + "
                f"{len(_DATA['sne'].z)} SNe + CMB comprimido (3) + "
                f"{len(_DATA['rsd'][0])} fσ8")
    print(f"Datos: {datasets}  (n = {n_points})")

    names_m = ["H0", "Omega_m", "epsilon", "z_trans", "omega_b", "sigma8"]
    names_l = ["H0", "Omega_m", "omega_b", "sigma8"]

    print(f"emcee MCMC k=6: {opts.nwalkers}x{opts.nsteps} (semilla {opts.seed})")
    s_m = run_sampler(log_prob_mcmc6,
                      np.array([C.H0_MCMC, 0.31, C.EPSILON_LAMBDA, C.Z_TRANS,
                                0.02236, 0.81]),
                      6, (), opts.nwalkers, opts.nsteps, opts.seed)
    flat_m = s_m.get_chain(discard=burn, thin=4, flat=True)
    lp_m = s_m.get_log_prob(discard=burn, thin=4, flat=True)

    print(f"emcee ΛCDM k=4: {opts.nwalkers}x{opts.nsteps} (misma semilla)")
    s_l = run_sampler(log_prob_lcdm4,
                      np.array([67.4, 0.315, 0.02236, 0.81]),
                      4, (), opts.nwalkers, opts.nsteps, opts.seed)
    flat_l = s_l.get_chain(discard=burn, thin=4, flat=True)
    lp_l = s_l.get_log_prob(discard=burn, thin=4, flat=True)

    def diag(sampler, tag):
        acc = float(np.mean(sampler.acceptance_fraction))
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            tau_s = ", ".join(f"{t:.0f}" for t in tau)
            ok = bool(np.all(opts.nsteps > 50 * tau))
        except Exception as exc:
            tau_s, ok = f"no estimable ({exc})", False
        print(f"  [{tag}] aceptación {acc:.2f}; τ: {tau_s}; "
              f"{'OK' if ok else 'CORTA'}")
        return acc, tau_s, ok

    acc_m, tau_m, ok_m = diag(s_m, "MCMC")
    acc_l, tau_l, ok_l = diag(s_l, "ΛCDM")

    best_m = flat_m[np.argmax(lp_m)]
    best_l = flat_l[np.argmax(lp_l)]
    ll_m = _loglike_parts((best_m[0], best_m[1], best_m[2], best_m[3]),
                          best_m[4], best_m[5])
    ll_l = _loglike_parts((best_l[0], best_l[1], 0.0, C.Z_TRANS),
                          best_l[2], best_l[3])
    ic_m = information_criteria(k=6, n=n_points, loglike_max=ll_m["total"])
    ic_l = information_criteria(k=4, n=n_points, loglike_max=ll_l["total"])
    d_aic = ic_m["AIC"] - ic_l["AIC"]
    d_bic = ic_m["BIC"] - ic_l["BIC"]

    p_m = save_chain("mcmc_v2", flat_m, lp_m, names_m)
    p_l = save_chain("lcdm_v2", flat_l, lp_l, names_l)

    rep = [
        "# Ajuste de producción v2 — cinco bloques (la reconciliación)\n",
        (f"Datos: {datasets} (n = {n_points}). Semilla {opts.seed}, "
         f"{opts.nwalkers} walkers × {opts.nsteps}, descarte {burn}, thin 4."),
        ("ΛCDM con la misma maquinaria (ε=0 exacto, Prop. A.1). Salvedades "
         "declaradas: z*/r_s por Hu–Sugiyama (sesgo ~0.3% en l_A, idéntico "
         "en ambos modelos); CMB diagonal sin correlaciones; r_d fiducial "
         "en BAO; SNe sin ancla (M_B marginalizada en ambos).\n"),
        "## Posteriores (mediana ± 1σ)\n",
        "### MCMC (k = 6)\n\n| Parámetro | Mediana | −/+ 1σ |\n|---|---|---|",
        *summarize(flat_m, names_m),
        (f"\nAceptación {acc_m:.2f}; τ: {tau_m}; "
         f"convergencia {'OK' if ok_m else 'INSUFICIENTE'}\n"),
        "### ΛCDM (k = 4)\n\n| Parámetro | Mediana | −/+ 1σ |\n|---|---|---|",
        *summarize(flat_l, names_l),
        (f"\nAceptación {acc_l:.2f}; τ: {tau_l}; "
         f"convergencia {'OK' if ok_l else 'INSUFICIENTE'}\n"),
        "## χ² por bloque (máximo a posteriori de cada modelo)\n",
        "| Bloque | χ² MCMC | χ² ΛCDM |\n|---|---|---|",
        *[f"| {k} | {-2*ll_m[k]:.2f} | {-2*ll_l[k]:.2f} |" for k in ll_m],
        "\n## Criterios de información\n",
        f"AIC = 2k − 2lnL; BIC = k·ln(n) − 2lnL; k=6 vs k=4; n={n_points}.\n",
        "| Modelo | k | ln L_max | AIC | BIC |\n|---|---|---|---|---|",
        (f"| MCMC | 6 | {ll_m['total']:.2f} | {ic_m['AIC']:.2f} | "
         f"{ic_m['BIC']:.2f} |"),
        (f"| ΛCDM | 4 | {ll_l['total']:.2f} | {ic_l['AIC']:.2f} | "
         f"{ic_l['BIC']:.2f} |"),
        (f"\n**ΔAIC (MCMC − ΛCDM) = {d_aic:+.2f}; "
         f"ΔBIC = {d_bic:+.2f}** (negativo favorece al MCMC). "
         "El resultado se publica sea cual sea.\n"),
        f"Cadenas: `{p_m.name}`, `{p_l.name}`.\n",
    ]
    OUT.mkdir(exist_ok=True)
    report = OUT / "production_fit2_report.md"
    report.write_text("\n".join(rep), encoding="utf-8")
    print(f"\nInforme: {report}")
    print(f"ΔAIC = {d_aic:+.2f}   ΔBIC = {d_bic:+.2f} "
          "(negativo favorece al MCMC)")


if __name__ == "__main__":
    main()
