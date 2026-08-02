#!/usr/bin/env python
"""Colapso pequeño reproducible con Cronos v3 — primer paso del frente nº 5.

Corre un par A/B de semilla idéntica (v35, B.5/B.6):
    A: α0⁻¹ = 0        (newtoniano puro — control)
    B: α0⁻¹ = 1e-7     (Cronos v3, dentro de la cota de la ec. 11.5)

y registra la fricción de compuerta Γ paso a paso. La firma falsable de
la fig. 11.1: Γ > 0 mientras la región colapsa (ρ̇ > 0) y Γ → 0 al
virializar. El esquema antiguo (v32) la mantenía siempre activa.

Uso:  python scripts/run_halo_collapse.py [--steps 120] [--seed 20260731]

Salida: informe markdown en output/halo_collapse_report.md (fuera del
control de versiones) y resumen por pantalla. Determinista: dos
ejecuciones con la misma semilla producen las mismas trayectorias.

ALCANCE: malla PM pequeña (cronos/simulation.py), no Gadget-4-Cronos.
Las validaciones de producción del corpus (núcleo 2.3 kpc, SPARC,
subhalos) siguen pendientes de reproducción con v3 (frente nº 5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.simulation import CronosPM  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"


def make_ic(seed: int, n_p: int = 512, box: float = 10.0):
    """Sobredensidad esférica fría en el centro + fondo uniforme."""
    rng = np.random.default_rng(seed)
    n_clump = n_p // 2
    pos = rng.uniform(0.0, box, size=(n_p, 3))
    pos[:n_clump] = box / 2.0 + rng.normal(scale=0.6, size=(n_clump, 3))
    vel = np.zeros((n_p, 3))
    return pos % box, vel, np.ones(n_p)


def run(alpha0_inv: float, seed: int, steps: int, dt: float = 0.02):
    pos, vel, mass = make_ic(seed)
    sim = CronosPM(pos, vel, mass, grid_n=16, box=10.0,
                   alpha0_inv=alpha0_inv, rho_c=2.0)
    history = [sim.step(dt=dt) for _ in range(steps)]
    return sim, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--alpha0-inv", type=float, default=1e-7)
    opts = ap.parse_args()

    print(f"Par A/B con semilla {opts.seed} ({opts.steps} pasos)...")
    _, hist_a = run(0.0, opts.seed, opts.steps)            # control
    _, hist_b = run(opts.alpha0_inv, opts.seed, opts.steps)  # Cronos v3

    # Diagnóstico a ESCALA DE HALO (fig. 11.1): Γ del grumo evaluada
    # sobre la historia de su densidad máxima ρ(t) — la compuerta debe
    # estar activa mientras el halo colapsa (ρ̇>0) y apagarse al
    # estabilizarse ρ (virialización).
    from cronos.cronos_v3 import gate_friction

    dt = 0.02
    rho_halo = np.array([h["rho_max"] for h in hist_b])
    rho_dot = np.gradient(rho_halo, dt)
    gam_halo = gate_friction(rho_halo, rho_dot, opts.alpha0_inv, 2.0)
    gam_a_max = max(h["Gamma_max"] for h in hist_a)

    i_peak = int(np.argmax(rho_halo))
    n_on_collapse = int(np.sum(gam_halo[:max(i_peak, 1)] > 0.0))
    n_off_after = int(np.sum(gam_halo[i_peak:] == 0.0))

    lines = [
        "# Colapso reproducible con Cronos v3 (frente abierto nº 5)\n",
        (f"Par A/B de semilla idéntica {opts.seed} (v35, B.5/B.6); "
         f"{opts.steps} pasos; α0⁻¹ = {opts.alpha0_inv:g} "
         "(dentro de la cota de la ec. 11.5).\n"),
        "| Magnitud | Valor |\n|---|---|",
        f"| Γ_max del control (A, α0⁻¹=0) | {gam_a_max:.3e} |",
        f"| Paso del pico de densidad del halo | {i_peak} / {opts.steps} |",
        f"| Pasos de colapso con Γ_halo > 0 | {n_on_collapse} / {i_peak} |",
        f"| Pasos post-pico con Γ_halo = 0 | {n_off_after} / {opts.steps - i_peak} |",
        ("\nFirma falsable (fig. 11.1): la compuerta Γ = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇) "
         "está activa mientras el halo colapsa y se apaga al virializar; el "
         "control newtoniano da Γ≡0 y el esquema v32 la mantenía siempre "
         "activa. La medición en producción (Gadget-4-Cronos, cajas de B.4) "
         "es el frente nº 5.\n"),
    ]
    OUT.mkdir(exist_ok=True)
    report = OUT / "halo_collapse_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"  control A: Γ_max = {gam_a_max:.3e} (esperado 0)")
    print(f"  Cronos B : Γ_halo>0 en {n_on_collapse}/{i_peak} pasos de "
          f"colapso; Γ_halo=0 en {n_off_after}/{opts.steps - i_peak} "
          "pasos post-pico")
    print(f"Informe: {report}")


if __name__ == "__main__":
    main()
