#!/usr/bin/env python
"""Verificación rápida: WKB, masas, cosmología, qudit."""

from __future__ import annotations

from cosmology.background import H_of_z, Lambda_rel
from mass_program.B3_wkb import verify_calibration
from mass_program.B4_masses import predict_fermion_masses
from mass_program.B5_higgs import beta3_calibrated_from_higgs, higgs_mass
from mass_program.M1_qcd_running import K_EW_to_GUT
from quantum.qutip_simulation import simulate_transitions


def main() -> None:
    print("--- Consistencia de la calibración WKB (pesos = entradas, no derivaciones) ---")
    for k, v in verify_calibration().items():
        ok = "OK" if v[2] else "FAIL"
        print(f"  {k:<10s}  computed={v[0]:.4f}  nominal={v[1]:.4f}  [{ok}]")

    print("\n--- Higgs (identidad SM calibrada — Obs. 12.2, no predicción) ---")
    print(f"  m_H = {higgs_mass():.3f} GeV")
    print(f"  β3 calibrado desde m_H medida (PDG) = {beta3_calibrated_from_higgs():.4f}")

    print("\n--- K_QCD(EW → GUT) ---")
    print(f"  K = {K_EW_to_GUT():.4f}  (esperado ≈ 0.5645)")

    print("\n--- Cosmología ---")
    print(f"  H(0)        = {H_of_z(0.0):.2f} km/s/Mpc")
    print(f"  Ω_Λ(0)      = {Lambda_rel(0.0):.4f}")
    # Invariante físico (nivel 3 del CI): este canal estuvo verde
    # mientras H(0) valía 70.09 con H0 = 69.8 — ahora FALLA si el
    # fondo viola su definición (H(0) = H0 para todo θ admisible).
    from mcmc_ontology import constants as C
    h0_dev = abs(float(H_of_z(0.0)) - C.H0_MCMC)
    h0_dev_off = abs(float(H_of_z(0.0, Omega_m=0.35, eps=0.05))
                     - C.H0_MCMC)
    if max(h0_dev, h0_dev_off) > 1e-9:
        raise SystemExit(
            f"INVARIANTE VIOLADO: H(0) != H0 (desv. fiducial {h0_dev:.2e},"
            f" fuera de fiducial {h0_dev_off:.2e})")
    print("  invariante H(0) = H0 (fiducial y fuera de fiducial): OK")

    print("\n--- Espectro fermiónico ---")
    res = predict_fermion_masses()
    for f in ("tau", "mu", "e", "t", "b", "c", "s", "u", "d"):
        r = res[f]
        print(f"  {f:<4s}  m_MCMC={r['m_MCMC']:.4g} GeV   "
              f"PDG={r['m_PDG']:.4g}   Δ={r['dev_pct']:.2f}%")

    print("\n--- Qudit transitions ---")
    fids = simulate_transitions()
    for k, v in fids.items():
        print(f"  {k}: F = {v}")

    print("\n--- Simulación (Apéndice B): mini-ciclo Cronos-KDK ---")
    import numpy as np

    from cronos.simulation import CronosPM
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10.0, size=(64, 3))
    pos[:32] = 5.0 + rng.normal(scale=0.8, size=(32, 3))
    sim = CronosPM(pos, np.zeros((64, 3)), np.ones(64), grid_n=8,
                   box=10.0, alpha0_inv=1e-7, rho_c=1.0)
    diag = None
    for _ in range(4):
        diag = sim.step(dt=0.01)
    ok = np.all(np.isfinite(sim.pos)) and np.isfinite(diag["v_rms"])
    print(f"  4 pasos KDK (B.3): rho_max={diag['rho_max']:.3f}  "
          f"Gamma_max={diag['Gamma_max']:.2e}  [{'OK' if ok else 'FAIL'}]")

    print()
    from validation.appendix_h import run_all
    run_all(verbose=True)


if __name__ == "__main__":
    main()
