#!/usr/bin/env python
"""Frente F: RP con acoplos no estacionarios — el juguete del frente nº 1.

Mide, en la cadena escalar 1D con acoplos dependientes de la posición:
(1) que la RP exige simetría especular del perfil, no estacionariedad;
(2) la curva de violación bajo running monótono (m² y J);
(3) que la reflexión modificada (reflejar también el perfil) restaura
    la positividad exactamente.

Uso: python scripts/run_rp_nonstationary.py

Salida: results/2026-08-02_rp_nonstationary/report.md (+ figura).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rp_nonstationary import (  # noqa: E402
    STATUS_NONSTATIONARY,
    mirrored_profile,
    rp_min_eig_nonstationary,
    running_profile,
    violation_curve,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-02_rp_nonstationary")
PHI = np.linspace(-4.0, 4.0, 41)
N_SIDE = 3


def main() -> None:
    # 1) especular no estacionario (tres semillas): cero exacto
    mirrored_eigs = []
    for seed in (7, 11, 23):
        rng = np.random.default_rng(seed)
        m2, J = mirrored_profile(rng.uniform(0.5, 2.0, size=N_SIDE),
                                 rng.uniform(0.5, 2.0),
                                 rng.uniform(0.5, 1.5, size=N_SIDE))
        mirrored_eigs.append(rp_min_eig_nonstationary(PHI, m2, J))
    print("Especular no estacionario (3 semillas): min eig =",
          ", ".join(f"{v:+.1e}" for v in mirrored_eigs))

    # 2) curvas de violación
    grads = [0.05, 0.1, 0.2, 0.4, 0.8]
    cm = violation_curve(grads, n_side=N_SIDE, which="m2")
    cj = violation_curve(grads, n_side=N_SIDE, which="J")
    for which, c in (("m²", cm), ("J", cj)):
        print(f"Running en {which}: " + ", ".join(
            f"g={g:g}→{v:+.2e}" for g, v in zip(c["gradients"],
                                                c["min_eigs"])))

    # 3) reflexión modificada sobre el running fuerte
    m2_run, J_run = running_profile(N_SIDE, g_m2=0.4)
    m2_mod, J_mod = mirrored_profile(m2_run[N_SIDE + 1:], m2_run[N_SIDE],
                                     J_run[N_SIDE:])
    v_naive = rp_min_eig_nonstationary(PHI, m2_run, J_run)
    v_mod = rp_min_eig_nonstationary(PHI, m2_mod, J_mod)
    print(f"g=0.4: reflexión ingenua {v_naive:+.2e} → "
          f"modificada {v_mod:+.1e}")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.semilogx(cm["gradients"], cm["min_eigs"], "o-",
                    color="#8a4a2f", label="running en m² (ingenua)")
        ax.semilogx(cj["gradients"], cj["min_eigs"], "s-",
                    color="#4a6b8a", label="running en J (ingenua)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.plot(cm["gradients"],
                [rp_min_eig_nonstationary(
                    PHI, *mirrored_profile(
                        running_profile(N_SIDE, g_m2=g)[0][N_SIDE + 1:],
                        running_profile(N_SIDE, g_m2=g)[0][N_SIDE],
                        running_profile(N_SIDE, g_m2=g)[1][N_SIDE:]))
                 for g in cm["gradients"]], "^--", color="#2f6b4f",
                label="reflexión modificada: restaurada")
        ax.set_xlabel("gradiente del running g")
        ax.set_ylabel("autovalor mínimo normalizado")
        ax.set_title("RP no estacionaria (frente nº 1): la simetría "
                     "especular es la hipótesis real")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "rp_nonstationary.png", dpi=140)
        print(f"Figura: {OUT / 'rp_nonstationary.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    rows_m = "\n".join(f"| {g:g} | {v:+.3e} |"
                       for g, v in zip(cm["gradients"], cm["min_eigs"]))
    rows_j = "\n".join(f"| {g:g} | {v:+.3e} |"
                       for g, v in zip(cj["gradients"], cj["min_eigs"]))
    (OUT / "report.md").write_text(
        "# Frente F: RP con acoplos no estacionarios (juguete del "
        "frente nº 1)\n\n"
        f"Cadena escalar de 2·{N_SIDE}+1 sitios, funcionales de la "
        "loncha ±1, contracción exacta por matrices de transferencia. "
        "El sector espinorial de Wilson (Teo. 7.4) NO se toca: esto es "
        "el análogo escalar de la pregunta del frente.\n\n"
        "## 1. La no-estacionariedad no rompe la RP\n\n"
        "Perfiles aleatorios ESPECULARES respecto de la loncha "
        "(acoplos distintos sitio a sitio): min eig = "
        + ", ".join(f"{v:+.1e}" for v in mirrored_eigs)
        + " — cero numérico. Mecanismo: M = diag(v)·LᵀW₀L·diag(v) es "
        "una forma XᵀX. La hipótesis que la RP necesita es la simetría "
        "especular del perfil, no su constancia.\n\n"
        "## 2. El running monótono la viola (medido)\n\n"
        "El caso físico — acoplos que corren con S — con la reflexión "
        "ingenua (que no refleja el perfil):\n\n"
        "| g (running m²) | min eig |\n|---|---|\n" + rows_m + "\n\n"
        "| g (running J) | min eig |\n|---|---|\n" + rows_j + "\n\n"
        "La violación es continua (minúscula a g pequeño) y crece "
        "monótonamente con el gradiente. Curva medida; no se reclama "
        "ley simple.\n\n"
        "## 3. La reflexión modificada restaura la positividad\n\n"
        f"Con g = 0.4 (violación {v_naive:+.2e}), reflejar TAMBIÉN el "
        f"perfil de acoplos da min eig = {v_mod:+.1e} — positividad "
        "exacta. La versión escalar de la pregunta real del frente 1: "
        "¿admite el sector de Wilson una reflexión modificada así? "
        "Queda donde el tratado lo deja: abierta.\n\n"
        f"Estatuto: {STATUS_NONSTATIONARY}.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
