#!/usr/bin/env python
"""La Circulación de Victoria (F2A–F2D): cuánto giro exige la Década.

Cuatro medidas, con el alcance declarado:
(F2A) el teorema de obstrucción como control negativo — gradiente puro
      (J = 0, G ≻ 0, H simétrica) ⟹ espectro real ⟹ s0 = 0;
(F2B) el mapa α_c(κ) y α_Victoria(κ) sobre la familia declarada
      M(α) = (−I + α·J₀)·H con H = diag(1, κ) — 2D estricto —, clavado
      contra las formas cerradas 2D;
(F2C) la separación tangencial/cruce en el cuello espinodal, exacta
      en el ejemplo construido (G = H = I);
(F2D) la tercera ruta (señal, sin espectro de M) sobre la matriz
      fiducial, junto a la espectral y la dinámica.

Uso: python scripts/run_victoria_circulation.py

Salida: results/2026-08-07_victoria_circulation/report.md (+ figura).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.victoria_circulation import (  # noqa: E402
    S0_TARGET,
    STATUS_CIRCULATION,
    J_plane,
    alpha_c_closed_form_2d,
    alpha_threshold,
    alpha_victoria,
    alpha_victoria_closed_form_2d,
    circulation_flow_matrix,
    max_im_spectrum,
    obstruction_holds,
    s0_signal,
    spinodal_normal,
    tangential_part,
)
from core.victoria_exponent import routes_coincide  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-07_victoria_circulation")
KAPPAS = (1.0, 1.5, 2.0, 3.0, 5.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # F2A — obstrucción sobre 500 sorteos (G ≻ 0, H simétrica)
    rng = np.random.default_rng(20260807)
    ok = 0
    for _ in range(500):
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        ok += obstruction_holds(A @ A.T + 3.0 * np.eye(3),
                                0.5 * (B + B.T))
    print(f"F2A obstrucción: {ok}/500 espectros reales con J = 0")

    # F2B — el mapa α_c(κ), α_V(κ) vs formas cerradas
    rows = []
    for k in KAPPAS:
        H = np.diag([1.0, k])
        a_c = alpha_threshold(np.eye(2), H, J_plane(2))
        a_v = alpha_victoria(np.eye(2), H, J_plane(2))
        rows.append((k, a_c, alpha_c_closed_form_2d(1.0, k),
                     a_v, alpha_victoria_closed_form_2d(1.0, k)))
        print(f"κ = {k}: α_c = {a_c:.6f} (cf {rows[-1][2]:.6f}); "
              f"α_V = {a_v:.6f} (cf {rows[-1][4]:.6f})")

    # F2C — tangencial vs cruce en el cuello
    n = spinodal_normal()
    a = np.array([1.0, 0.0, 0.0])
    t1 = a - (a @ n) * n; t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    J_tang = np.outer(t1, t2) - np.outer(t2, t1)
    J_norm = np.outer(n, t1) - np.outer(t1, n)
    s_pairs = []
    for name, J in (("tangencial", J_tang), ("cruce", J_norm)):
        M = circulation_flow_matrix(np.eye(3), np.eye(3), J, alpha=2.0)
        s_pairs.append((name, max_im_spectrum(M),
                        max_im_spectrum(tangential_part(M, n))))
        print(f"F2C {name}: s0(M) = {s_pairs[-1][1]:.3f}, "
              f"s0(P_T·M·P_T) = {s_pairs[-1][2]:.3f}")

    # F2D — tres rutas sobre la matriz fiducial
    M3 = np.zeros((3, 3))
    M3[:2, :2] = np.array([[0.3, -S0_TARGET], [S0_TARGET, 0.3]])
    M3[2, 2] = -0.7
    M3[0, 2] = 0.4
    r = routes_coincide(M3)
    s_sig = s0_signal(M3)
    print(f"F2D tres rutas: espectral {r['s0_spectral']:.6f} | dinámica "
          f"{r['s0_dynamic']:.6f} | señal {s_sig:.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = np.linspace(1.0, 5.0, 200)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(ks, [alpha_c_closed_form_2d(1.0, k) for k in ks],
                lw=1.5, label="α_c(κ) — umbral de complejificación")
        ax.plot(ks, [alpha_victoria_closed_form_2d(1.0, k) for k in ks],
                lw=1.5, label="α_Victoria(κ) — giro para s0 = π/ln10")
        ax.scatter([r_[0] for r_ in rows], [r_[1] for r_ in rows],
                   marker="o", s=28, zorder=3, label="medido (bisección)")
        ax.scatter([r_[0] for r_ in rows], [r_[3] for r_ in rows],
                   marker="s", s=28, zorder=3)
        ax.set_xlabel("anisotropía κ del hessiano declarado H = diag(1, κ)")
        ax.set_ylabel("α (circulación)")
        ax.set_title("La Década exige giro (en la clase declarada): α_c > 0 "
                     "con paisaje anisótropo;"
                     "\nα_Victoria es un mapa, no una constante")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "victoria_circulation.png", dpi=140)
        print(f"Figura: {OUT / 'victoria_circulation.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    table = "\n".join(
        f"| {k} | {a_c:.6f} | {cf_c:.6f} | {a_v:.6f} | {cf_v:.6f} |"
        for k, a_c, cf_c, a_v, cf_v in rows)
    (OUT / "report.md").write_text(
        "# La Circulación de Victoria (F2A–F2D): cuánto giro exige la "
        "Década\n\n"
        "Implementación de la propuesta v36 «dos niveles del Camino»: el "
        "flujo fundamental desciende (Axioma 4); su proyección efectiva "
        "sobre los acoplos puede girar mientras desciende — "
        "β = (−G⁻¹+J)·∇C conserva dC/dlnS ≤ 0 con J antisimétrica y "
        "permite espectro complejo.\n\n"
        "## F2A. El teorema de obstrucción (control negativo)\n\n"
        f"{ok}/500 sorteos (G ≻ 0, H simétrica, J = 0) con espectro real "
        "— demostrado además por similitud: −G⁻¹H ~ −G^(−1/2)HG^(−1/2). "
        "**Dentro de la clase declarada, s0 ≠ 0 exige J ≠ 0**: la "
        "predicción arquitectónica de la propuesta queda establecida "
        "como teorema del ansatz (comprobación interna, no demostración "
        "física).\n\n"
        "## F2B. La circulación mínima\n\n"
        "| κ | α_c medido | α_c forma cerrada | α_V medido | α_V forma "
        "cerrada |\n|---|---|---|---|---|\n"
        f"{table}\n\n"
        "- **Robusto**: con H isótropa α_c = 0; con H anisótropa el "
        "umbral es finito — la anisotropía del paisaje encarece el giro."
        "\n- **Condicionado**: α_Victoria depende de (H, J₀) declarados "
        "— es un MAPA (aquí clavado contra su forma cerrada 2D con "
        "error < 1e-6), no una constante de la naturaleza; las β reales "
        "(Def. 4.4) decidirán el punto del mapa.\n\n"
        "## F2C. Tangencial al cuello vs cruce\n\n"
        + "\n".join(
            f"- Circulación {name}: s0(M) = {s_f:.3f} → s0(P_T·M·P_T) = "
            f"{s_t:.3f}" for name, s_f, s_t in s_pairs)
        + "\n\nEn el ejemplo construido (G = H = I), la proyección "
        "P_T = I − nnᵀ (n = ∇D/|∇D| en el cuello) separa el walking "
        "tangencial a la espinodal del cruce del discriminante: solo "
        "la circulación tangencial sobrevive a P_T·M·P_T. Con H "
        "genérica la separación NO es exacta — la circulación del "
        "plano de cruce puede sobrevivir a P_T si H acopla n al plano "
        "tangente (P_T·J_cruce·H·P_T = −t₁·(nᵀH·P_T) ≠ 0) —: es una "
        "propiedad del caso construido, no de P_T en general (la "
        "limitación queda ejecutable en el test).\n\n"
        "## F2D. Tres rutas independientes al exponente\n\n"
        f"Sobre la matriz fiducial (par 0.3 ± i·π/ln10 + modo real "
        f"acoplado): espectral {r['s0_spectral']:.6f} | dinámica "
        f"{r['s0_dynamic']:.6f} | señal (cruces por cero, sin espectro "
        f"de M) {s_sig:.6f} — tres estimadores, un exponente.\n\n"
        "## Estatuto\n\n"
        f"{STATUS_CIRCULATION}.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
