#!/usr/bin/env python
"""Ajuste bayesiano emcee: datos reales si están en data/, autoprueba si no.

Con los catálogos descargados (scripts/download_data.py), corre sobre
H(z) de cronómetros cósmicos (+ BAO si está la tabla). Sin ellos, cae a
la AUTOPRUEBA sintética, que es circular por construcción y solo
verifica la maquinaria de inferencia.
"""

from __future__ import annotations

import numpy as np

from cosmology.bayesian_fit import (
    synthetic_selftest_dataset, load_Hz_data, load_bao_data, run_emcee,
)


def main() -> None:
    bao = None
    try:
        data = load_Hz_data()
        try:
            bao = load_bao_data()
        except FileNotFoundError:
            pass
        print(f"Dataset REAL: {len(data.z)} puntos H(z)"
              + (f" + {len(bao.z)} medidas BAO" if bao is not None else ""))
    except FileNotFoundError as exc:
        print(f"[aviso] {exc}")
        data = synthetic_selftest_dataset(n=24, seed=0)
        print(f"Dataset AUTOPRUEBA (sintético, circular): {len(data.z)} puntos H(z)")
        print("El resultado verifica la maquinaria, no contrasta el modelo.")

    sampler = run_emcee(data, nwalkers=24, nsteps=400, seed=42, bao=bao)
    flat = sampler.get_chain(discard=200, thin=2, flat=True)
    medians = np.median(flat, axis=0)
    sigmas  = np.std(flat, axis=0)
    names = ("H0", "Omega_m", "epsilon", "z_trans")
    print("\nResultado del ajuste:")
    for n, m, s in zip(names, medians, sigmas):
        print(f"  {n:<10s} = {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
