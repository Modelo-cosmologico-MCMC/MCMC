#!/usr/bin/env python
"""Ajuste bayesiano emcee con un dataset H(z) sintético."""

from __future__ import annotations

import numpy as np

from cosmology.bayesian_fit import mock_Hz_dataset, run_emcee


def main() -> None:
    data = mock_Hz_dataset(n=24, seed=0)
    print(f"Dataset: {len(data.z)} puntos H(z)")
    sampler = run_emcee(data, nwalkers=24, nsteps=400, seed=42)
    flat = sampler.get_chain(discard=200, thin=2, flat=True)
    medians = np.median(flat, axis=0)
    sigmas  = np.std(flat, axis=0)
    names = ("H0", "Omega_m", "epsilon", "z_trans")
    print("\nResultado del ajuste:")
    for n, m, s in zip(names, medians, sigmas):
        print(f"  {n:<10s} = {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
