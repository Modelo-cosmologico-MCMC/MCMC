#!/usr/bin/env python
"""Corrida de producción del ajuste bayesiano: MCMC vs ΛCDM sobre los mismos datos.

Cadena determinista (semilla fijada):
    descarga verificada → likelihoods (CC + BAO + SNe si están) →
    emcee para el MCMC (4 parámetros) y para ΛCDM (2 parámetros, misma
    maquinaria, ε=0 exacto por la Prop. A.1) → cadenas guardadas →
    informe markdown con χ² por dataset y AIC/BIC con fórmula explícita.

Uso:
    python scripts/download_data.py all        # una vez, con red
    python scripts/run_production_fit.py [--nsteps 4000] [--nwalkers 32]

Salidas (en output/, fuera del control de versiones):
    output/chains_mcmc.h5|.npz, output/chains_lcdm.h5|.npz
    output/production_fit_report.md

Avisos de honestidad (se imprimen y van al informe):
- El resultado es el que sea: si el ΔBIC real no favorece al modelo,
  se publica igual.
- ΛCDM se ajusta sobre los mismos datos con la misma maquinaria, no se
  toma de la literatura.
- Con estos resultados publicados, CORPUS_REFERENCE del README debe
  sustituirse por ellos («Resultados del ajuste»).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.bayesian_fit import (  # noqa: E402
    information_criteria,
    load_bao_data,
    load_Hz_data,
    load_sne_data,
    log_like_bao,
    log_like_Hz,
    log_like_sne,
    log_prob,
    log_prob_lcdm,
    theta_lcdm_to_mcmc,
)
from mcmc_ontology import constants as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"


def save_chain(tag: str, chain: np.ndarray, log_prob_vals: np.ndarray,
               names: list[str]) -> Path:
    """Guarda la cadena en HDF5 si h5py está disponible; si no, en NPZ."""
    OUT.mkdir(exist_ok=True)
    try:
        import h5py  # type: ignore

        path = OUT / f"chains_{tag}.h5"
        with h5py.File(path, "w") as fh:
            fh.create_dataset("chain", data=chain)
            fh.create_dataset("log_prob", data=log_prob_vals)
            fh.attrs["params"] = ",".join(names)
        return path
    except ImportError:
        path = OUT / f"chains_{tag}.npz"
        np.savez_compressed(path, chain=chain, log_prob=log_prob_vals,
                            params=np.array(names, dtype=object))
        return path


def run_sampler(log_prob_fn, p0_center: np.ndarray, ndim: int, args: tuple,
                nwalkers: int, nsteps: int, seed: int):
    import emcee

    np.random.seed(seed)  # emcee 3 usa el estado global de NumPy
    rng = np.random.default_rng(seed)
    p0 = p0_center + 1e-3 * rng.normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=args)
    sampler.run_mcmc(p0, nsteps, progress=False)
    return sampler


def summarize(flat: np.ndarray, names: list[str]) -> list[str]:
    lines = []
    for i, nm in enumerate(names):
        lo, med, hi = np.percentile(flat[:, i], [15.865, 50.0, 84.135])
        lines.append(f"| {nm} | {med:.4f} | −{med - lo:.4f} / +{hi - med:.4f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--burn", type=int, default=None,
                    help="descarte inicial (default: nsteps//2)")
    opts = ap.parse_args()
    burn = opts.burn if opts.burn is not None else opts.nsteps // 2

    # --- Datos (los que estén; el informe declara cuáles) ---
    data = load_Hz_data()          # obligatorio: instruye si falta
    try:
        bao = load_bao_data()
    except FileNotFoundError:
        bao = None
    try:
        sne = load_sne_data()
    except FileNotFoundError:
        sne = None
    n_points = len(data.z) + (len(bao.z) if bao else 0) \
        + (len(sne.z) if sne else 0)
    datasets = (f"{len(data.z)} H(z) de cronómetros"
                + (f" + {len(bao.z)} BAO" if bao else "")
                + (f" + {len(sne.z)} SNe (cov completa)" if sne else ""))
    print(f"Datos: {datasets}  (n = {n_points})")
    if sne is None:
        print("[aviso] Sin Pantheon+: ejecuta scripts/download_data.py pantheon")

    names_m = ["H0", "Omega_m", "epsilon", "z_trans"]
    names_l = ["H0", "Omega_m"]

    # --- MCMC (4 parámetros) ---
    print(f"emcee MCMC: {opts.nwalkers} walkers x {opts.nsteps} pasos "
          f"(semilla {opts.seed})...")
    s_m = run_sampler(log_prob, np.array([C.H0_MCMC, 0.300, C.EPSILON_0,
                                          C.Z_TRANS]),
                      4, (data, sne, bao), opts.nwalkers, opts.nsteps,
                      opts.seed)
    flat_m = s_m.get_chain(discard=burn, thin=4, flat=True)
    lp_m = s_m.get_log_prob(discard=burn, thin=4, flat=True)

    # --- ΛCDM (2 parámetros, misma maquinaria, ε = 0 exacto) ---
    print(f"emcee ΛCDM: {opts.nwalkers} walkers x {opts.nsteps} pasos "
          f"(misma semilla)...")
    s_l = run_sampler(log_prob_lcdm, np.array([67.4, 0.300]),
                      2, (data, sne, bao), opts.nwalkers, opts.nsteps,
                      opts.seed)
    flat_l = s_l.get_chain(discard=burn, thin=4, flat=True)
    lp_l = s_l.get_log_prob(discard=burn, thin=4, flat=True)

    # --- Diagnósticos de convergencia ---
    def diag(sampler, tag):
        acc = float(np.mean(sampler.acceptance_fraction))
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            tau_s = ", ".join(f"{t:.0f}" for t in tau)
            ok = bool(np.all(opts.nsteps > 50 * tau))
        except Exception as exc:  # cadenas cortas
            tau_s, ok = f"no estimable ({exc})", False
        print(f"  [{tag}] aceptación media {acc:.2f}; τ_autocorr: {tau_s}")
        return acc, tau_s, ok

    acc_m, tau_m, ok_m = diag(s_m, "MCMC")
    acc_l, tau_l, ok_l = diag(s_l, "ΛCDM")

    # --- Máximos de verosimilitud (sin prior) para AIC/BIC ---
    def loglike_at(theta) -> dict:
        parts = {"Hz": log_like_Hz(theta, data)}
        if sne is not None:
            parts["SNe"] = log_like_sne(theta, sne)
        if bao is not None:
            parts["BAO"] = log_like_bao(theta, bao)
        parts["total"] = float(sum(parts.values()))
        return parts

    best_m = tuple(flat_m[np.argmax(lp_m)])
    best_l = theta_lcdm_to_mcmc(tuple(flat_l[np.argmax(lp_l)]))
    ll_m = loglike_at(best_m)
    ll_l = loglike_at(best_l)
    ic_m = information_criteria(k=4, n=n_points, loglike_max=ll_m["total"])
    ic_l = information_criteria(k=2, n=n_points, loglike_max=ll_l["total"])
    d_aic = ic_m["AIC"] - ic_l["AIC"]
    d_bic = ic_m["BIC"] - ic_l["BIC"]

    p_m = save_chain("mcmc", flat_m, lp_m, names_m)
    p_l = save_chain("lcdm", flat_l, lp_l, names_l)

    # --- Informe ---
    chi2 = {k: -2.0 * v for k, v in ll_m.items()}
    chi2_l = {k: -2.0 * v for k, v in ll_l.items()}
    rep = []
    rep.append("# Ajuste de producción MCMC vs ΛCDM\n")
    rep.append(f"Datos: {datasets} (n = {n_points}). Semilla {opts.seed}, "
               f"{opts.nwalkers} walkers × {opts.nsteps} pasos, descarte "
               f"{burn}, thin 4. Priors: apéndice F (H0 ~ N(67.4, 5²)).\n")
    rep.append("ΛCDM ajustado sobre LOS MISMOS datos con la MISMA maquinaria "
               "(ε = 0 exacto, Prop. A.1); M_B de SNe marginalizada "
               "analíticamente en ambos modelos por igual (no cuenta en k).\n")
    rep.append("## Posteriores (mediana ± 1σ)\n")
    rep.append("### MCMC (k = 4)\n\n| Parámetro | Mediana | −/+ 1σ |\n|---|---|---|")
    rep.extend(summarize(flat_m, names_m))
    rep.append(f"\nAceptación {acc_m:.2f}; τ: {tau_m}; "
               f"convergencia {'OK' if ok_m else 'INSUFICIENTE — subir --nsteps'}\n")
    rep.append("### ΛCDM (k = 2)\n\n| Parámetro | Mediana | −/+ 1σ |\n|---|---|---|")
    rep.extend(summarize(flat_l, names_l))
    rep.append(f"\nAceptación {acc_l:.2f}; τ: {tau_l}; "
               f"convergencia {'OK' if ok_l else 'INSUFICIENTE — subir --nsteps'}\n")
    rep.append("## χ² por dataset (en el máximo a posteriori de cada modelo)\n")
    rep.append("| Dataset | χ² MCMC | χ² ΛCDM |\n|---|---|---|")
    for key in chi2:
        rep.append(f"| {key} | {chi2[key]:.2f} | {chi2_l[key]:.2f} |")
    rep.append("\n## Criterios de información\n")
    rep.append("Fórmulas: AIC = 2k − 2·ln L_max; BIC = k·ln(n) − 2·ln L_max. "
               f"k_MCMC = 4 (H0, Ωm, ε, z_trans); k_ΛCDM = 2 (H0, Ωm); "
               f"n = {n_points}.\n")
    rep.append("| Modelo | k | ln L_max | AIC | BIC |\n|---|---|---|---|---|")
    rep.append(f"| MCMC | 4 | {ll_m['total']:.2f} | {ic_m['AIC']:.2f} | "
               f"{ic_m['BIC']:.2f} |")
    rep.append(f"| ΛCDM | 2 | {ll_l['total']:.2f} | {ic_l['AIC']:.2f} | "
               f"{ic_l['BIC']:.2f} |")
    rep.append(f"\n**ΔAIC (MCMC − ΛCDM) = {d_aic:+.2f}; "
               f"ΔBIC (MCMC − ΛCDM) = {d_bic:+.2f}** "
               "(negativo favorece al MCMC; positivo, a ΛCDM). "
               "El resultado se publica sea cual sea.\n")
    rep.append(f"Cadenas: `{p_m.name}`, `{p_l.name}` (en output/).\n")

    OUT.mkdir(exist_ok=True)
    report = OUT / "production_fit_report.md"
    report.write_text("\n".join(rep), encoding="utf-8")
    print(f"\nInforme: {report}")
    print(f"ΔAIC = {d_aic:+.2f}   ΔBIC = {d_bic:+.2f} "
          "(negativo favorece al MCMC)")


if __name__ == "__main__":
    main()
