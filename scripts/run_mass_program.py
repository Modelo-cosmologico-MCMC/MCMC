#!/usr/bin/env python
"""Reproduce el programa de masas completo: 12/12 fermiones + Higgs."""

from __future__ import annotations

from mass_program.B4_masses import predict_fermion_masses, neutrino_sum_eV
from mass_program.B5_higgs import higgs_mass


def main() -> None:
    print("=" * 60)
    print(" MCMC — Programa de masas")
    print("=" * 60)
    print(f"\nHiggs: m_H = {higgs_mass():.3f} GeV   (PDG: 125.25)")
    print(f"\nΣ m_ν = {neutrino_sum_eV():.3e} eV  (cota Planck+DESI: 0.12)\n")

    res = predict_fermion_masses()
    print(f"{'Fermión':<8}  {'MCMC':>14}  {'PDG':>14}  {'Δ%':>8}")
    print("-" * 50)
    for f in ("e", "mu", "tau", "u", "d", "s", "c", "b", "t"):
        r = res[f]
        print(f"{f:<8}  {r['m_MCMC']:>14.6g}  {r['m_PDG']:>14.6g}  "
              f"{r['dev_pct']:>8.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
