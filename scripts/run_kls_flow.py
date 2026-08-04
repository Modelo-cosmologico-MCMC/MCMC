#!/usr/bin/env python
"""Frente E: el flujo KLS integrado — la ley del walking, medida (cap. 8).

Cuatro medidas sobre el mecanismo del Discriminante, con su estatuto:

1. El «≃» de la ec. 8.4, cuantificado: error del prefactor vs Ω/xR.
2. El exponente de divergencia en la espinodal (esperado −1/2).
3. El Cruce de Victoria como bifurcación dinámica (Obs. 8.6): el
   colapso se dispara tras el cruce D = 0, con seguimiento adiabático
   del vacío antes, y control negativo sin hundimiento.
4. El retraso del colapso escala como rate^(−1/3) (silla-nodo con
   deriva — RESULTADO DEL PROGRAMA, no del tratado).

Lo que NO hace: derivar λ = 10 (frente 2) — eso exige las β-funciones
de los acoplos, que el tratado no da.

Uso: python scripts/run_kls_flow.py

Salida: results/2026-08-02_kls_flow/report.md (+ figura).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decade import fixed_points, s0_from_lambda  # noqa: E402
from core.kls_flow import (  # noqa: E402
    STATUS_LAMBDA,
    cruce_de_victoria,
    delay_scaling,
    divergence_exponent,
    walking_time_analytic,
    walking_time_measured,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "2026-08-02_kls_flow"
B, C0, G = 2.0, 1.0, 1.0   # xR = 1 (parámetros O(1) de demostración)
K = 50.0


def M0_sq_of(om_rel: float) -> float:
    return (B ** 2 + (2.0 * C0 * om_rel) ** 2) / (4.0 * C0)


def main() -> None:
    # 1) La ley del walking, cuantificada
    rows = []
    for om_rel in (0.01, 0.003, 0.001):
        m = walking_time_measured(B, M0_sq_of(om_rel), C0, G, K=K)
        aK = walking_time_analytic(B, M0_sq_of(om_rel), C0, G, K=K)
        rows.append((om_rel, m, aK, abs(m / aK - 1.0)))
        print(f"Ω/xR = {om_rel:g}: error del prefactor de la ec. 8.4 = "
              f"{abs(m / aK - 1.0):.2e}")

    # 2) Exponente de divergencia
    div = divergence_exponent(B, C0, G)
    print(f"Exponente de divergencia en la espinodal: "
          f"{div['exponent']:.4f} (esperado −1/2)")

    # 3) El Cruce de Victoria dinámico + control negativo
    cv = cruce_de_victoria(B, 0.9, rate=1e-3)
    print(f"Cruce: σ* = {cv['sigma_star']:.1f}; colapso en "
          f"σ = {cv['sigma_collapse']:.1f} (retraso {cv['delay']:.2f})")

    # 4) Escalado del retraso
    rates = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ds = delay_scaling(B, 0.9, rates)
    print(f"Retraso ∝ rate^{ds['exponent']:.3f} (silla-nodo con deriva; "
          "esperado −1/3)")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
        # (a) la cascada: x(σ) con el cruce marcado
        sig = np.arange(len(cv["x"])) * cv["d_sigma"]
        axes[0].plot(sig, cv["x"], lw=1, color="#2f6b4f")
        xp = [fixed_points(B, 0.9 + 1e-3 * s, C0)[1]
              if 0.9 + 1e-3 * s < B ** 2 / (4 * C0) else np.nan
              for s in sig[::200]]
        axes[0].plot(sig[::200], xp, ":", color="k", lw=1,
                     label="x+(σ) (vacío instantáneo)")
        axes[0].axvline(cv["sigma_star"], ls="--", color="#8a4a2f",
                        label="σ* (cruce D = 0)")
        axes[0].set_xlabel(r"$\sigma$"); axes[0].set_ylabel("x")
        axes[0].set_title("El Cruce de Victoria (Obs. 8.6): colapso "
                          "tras el cruce")
        axes[0].legend(fontsize=7)
        # (b) divergencia
        axes[1].loglog(div["depths"], div["times"], "o", ms=4,
                       color="#4a6b8a")
        ref = div["times"][0] * (div["depths"] / div["depths"][0]) ** -0.5
        axes[1].loglog(div["depths"], ref, "--", color="k", lw=1,
                       label=r"$|D|^{-1/2}$")
        axes[1].set_xlabel("|D|"); axes[1].set_ylabel(r"$\Delta\sigma_{walk}$")
        axes[1].set_title(f"Divergencia: exponente {div['exponent']:.3f}")
        axes[1].legend(fontsize=8)
        # (c) retraso vs ritmo
        axes[2].loglog(ds["rates"], ds["delays"], "o", ms=4,
                       color="#4a6b8a")
        ref = ds["delays"][0] * (ds["rates"] / ds["rates"][0]) ** (-1 / 3)
        axes[2].loglog(ds["rates"], ref, "--", color="k", lw=1,
                       label=r"rate$^{-1/3}$")
        axes[2].set_xlabel("ritmo de hundimiento de D")
        axes[2].set_ylabel("retraso del colapso")
        axes[2].set_title(f"Bifurcación dinámica: {ds['exponent']:.3f}")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "kls_flow.png", dpi=140)
        print(f"Figura: {OUT / 'kls_flow.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    table = "\n".join(
        f"| {om:g} | {m:.4f} | {a:.4f} | {e:.2e} |"
        for om, m, a, e in rows)
    (OUT / "report.md").write_text(
        "# Frente E: el flujo KLS integrado — la ley del walking, "
        "medida\n\n"
        f"Parámetros O(1) de demostración: B = {B}, C0 = {C0}, G = {G} "
        "(xR = 1). Integración RK4 validada contra la cuadratura exacta "
        "de la ODE separable (ec. 8.3).\n\n"
        "## 1. El «≃» de la ec. 8.4, cuantificado\n\n"
        "| Ω/xR | Δσ medido | Δσ ec. 8.4 (misma ventana) | error |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        "El error del prefactor x ≈ xR decae linealmente con Ω/xR; "
        "frente a la forma π/Ω del tratado queda además el factor de "
        f"ventana 2·arctan({K:g})/π = "
        f"{2.0 * np.arctan(K) / np.pi:.5f}.\n\n"
        "## 2. Divergencia en la espinodal\n\n"
        f"Δσ_walk ∝ |D|^({div['exponent']:.4f}) sobre cuatro décadas "
        "(esperado −1/2, pues Ω = √(−D)/2C0). La ley que sostiene la "
        "log-periodicidad de la Década queda medida.\n\n"
        "## 3. El Cruce de Victoria como bifurcación dinámica\n\n"
        f"Con D(σ) hundiéndose a ritmo 1e-3: el flujo sigue "
        "adiabáticamente el vacío x+(σ) hasta el cruce (σ* = "
        f"{cv['sigma_star']:.1f}, exacto) y colapsa en σ = "
        f"{cv['sigma_collapse']:.1f} — DESPUÉS del cruce, con retraso "
        f"{cv['delay']:.2f}. Control negativo: sin hundimiento el "
        "flujo aparca en x+ y no colapsa (test).\n\n"
        "## 4. El retraso del colapso (resultado del programa)\n\n"
        f"Retraso ∝ rate^({ds['exponent']:.3f}) sobre dos décadas — la "
        "ley −1/3 de la bifurcación silla-nodo con deriva (teoría de "
        "bifurcaciones dinámicas; NO está en el tratado y se declara "
        "como resultado del programa). Formulación CONDICIONAL "
        "correcta: SI la microdinámica real del cruce se reduce a una "
        "silla-nodo con deriva lenta (la forma normal aquí integrada), "
        "ENTONCES el disparo de cada colapso lleva un retraso "
        "∝ ritmo^(−1/3); la condición de pertenencia a esa forma "
        "normal es derivable y queda pendiente.\n\n"
        "## Estatuto (frente 2)\n\n"
        f"{STATUS_LAMBDA}.\n\n"
        f"Para λ = 10, s0 = π/ln(10) = {s0_from_lambda(10.0):.4f} "
        "(Def. 8.3) sigue siendo la relación DEFINIDA, no derivada: "
        "este módulo valida el mecanismo sobre el que descansa, no "
        "decide λ.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
