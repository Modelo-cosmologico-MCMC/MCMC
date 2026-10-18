# Candidato 2 del cribado de β: β a un lazo (LPA) del séxtico O(N) en d = 1, 2, 3 — derivación y verificación, sin letra

Commit `33dd9f962`; módulo `core/beta_one_loop.py`. Estatuto: derivación (E8) y verificación numérica; sin letra; no es una corrida preinscrita del reloj.

## Umbrales del flujo de R = B²/(4C₀M₀²)

Formas por defecto: m̄ = 1.0, b̄ = 3.0, C₀ = 1.0 ⟹ R₀ = 2.25; la espinodal exige ∫[2β_B/B − β_C/C − β_M/M] = -0.811 antes de M₀² = 0. T₀ = 0 en R = 4/3, D = 0 en R = 1.

| N | R_c (Wetterich): F > 0 para todo m̃² si R > R_c | R* (Polchinski): crece si R > R* y M₀² > 0 |
|---|---|---|
| 1 | 2.4167 | 1.6667 |
| 2 | 2.0833 | 1.5000 |
| 3 | 1.8833 | 1.4000 |
| 8 | 1.4833 | 1.2000 |
| inf | 1.0833 | 1.0000 |

Ventanas de m̃² en las que R decrece (Wetterich):

- N1_R1.3333: (-0.0125, 0.1596)
- N1_R1.5000: (-0.0058, 0.1325)
- N1_R2.0000: (0.0107, 0.0784)
- N1_R2.2500: (0.0200, 0.0576)
- N2_R1.3333: (-0.0083, 0.1539)
- N2_R1.5000: (0.0000, 0.1250)
- N2_R2.0000: (0.0266, 0.0609)
- N2_R2.2500: ninguna

Mínimo de F sobre todo m̃² en R₀ = 2.25: N = 1: -0.539, N = 2: 0.719

## Flujos desde las condiciones iniciales del reloj (tiempo de RG s; en el reloj s = τ·S)

| esquema | N | d | δ₀ | s(M₀² = 0) | s(T₀ = 0) | s(D = 0) | fracción del exigido | R crece monótonamente con M₀² > 0 |
|---|---|---|---|---|---|---|---|---|
| wetterich_litim | 1 | 1 | 0.01 | 0.000736 | — | — | 0 | sí |
| wetterich_litim | 1 | 1 | 0.03 | 0.00221 | — | — | 0 | sí |
| wetterich_litim | 1 | 2 | 0.01 | 0.000736 | — | — | 0 | sí |
| wetterich_litim | 1 | 2 | 0.03 | 0.00221 | — | — | 0 | sí |
| wetterich_litim | 1 | 3 | 0.01 | 0.000737 | — | — | 0 | sí |
| wetterich_litim | 1 | 3 | 0.03 | 0.00221 | — | — | 0 | sí |
| wetterich_litim | 2 | 1 | 0.01 | 0.000528 | — | — | 0 | sí |
| wetterich_litim | 2 | 1 | 0.03 | 0.00158 | — | — | 0 | sí |
| wetterich_litim | 2 | 2 | 0.01 | 0.000528 | — | — | 0 | sí |
| wetterich_litim | 2 | 2 | 0.03 | 0.00159 | — | — | 0 | sí |
| wetterich_litim | 2 | 3 | 0.01 | 0.000528 | — | — | 0 | sí |
| wetterich_litim | 2 | 3 | 0.03 | 0.00159 | — | — | 0 | sí |
| polchinski | 1 | 1 | 0.01 | 0.00147 | — | — | 0 | sí |
| polchinski | 1 | 1 | 0.03 | 0.00441 | — | — | 0 | sí |
| polchinski | 1 | 2 | 0.01 | 0.00147 | — | — | 0 | sí |
| polchinski | 1 | 2 | 0.03 | 0.00442 | — | — | 0 | sí |
| polchinski | 1 | 3 | 0.01 | 0.00147 | — | — | 0 | sí |
| polchinski | 1 | 3 | 0.03 | 0.00443 | — | — | 0 | sí |
| polchinski | 2 | 1 | 0.01 | 0.00106 | — | — | 0 | sí |
| polchinski | 2 | 1 | 0.03 | 0.00316 | — | — | 0 | sí |
| polchinski | 2 | 2 | 0.01 | 0.00106 | — | — | 0 | sí |
| polchinski | 2 | 2 | 0.03 | 0.00317 | — | — | 0 | sí |
| polchinski | 2 | 3 | 0.01 | 0.00106 | — | — | 0 | sí |
| polchinski | 2 | 3 | 0.03 | 0.00317 | — | — | 0 | sí |

## Candidato 1 bajo el criterio (i) integral (informativo)

Re-corrida de las celdas de la ventana κ̂ ∈ [0.5, 5.0] (cierre 'quadrant', sin escalón) y de los controles canónicos, con el código del reloj de #45. La letra congelada del cribado («no pasa») no cambia.

| δ₀ | τ | κ̂ | fracción del exigido | R crece monótonamente con M₀² > 0 | S(T₀ = 0) | S(D = 0) | S(M₀² = 0) |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.01 | — | 0 | sí | — | — | 0.106 |
| 0.01 | 0.1 | — | 0 | sí | — | — | 0.0107 |
| 0.01 | 1.0 | — | 0 | sí | — | — | 0.00195 |
| 0.03 | 0.01 | — | 0 | sí | — | — | 0.317 |
| 0.03 | 0.1 | — | 0 | sí | — | — | 0.0327 |
| 0.03 | 1.0 | — | 0 | sí | — | — | 0.00326 |
| 0.01 | 0.01 | 1 | 0 | sí | — | — | 0.106 |
| 0.01 | 0.01 | 3.16 | 0 | sí | — | — | 0.106 |
| 0.01 | 0.1 | 1 | 0 | sí | — | — | 0.0107 |
| 0.01 | 0.1 | 3.16 | 0 | sí | — | — | 0.0107 |
| 0.01 | 1.0 | 1 | 0 | sí | — | — | 0.00195 |
| 0.01 | 1.0 | 3.16 | 0 | sí | — | — | 0.00195 |
| 0.03 | 0.01 | 1 | 0 | sí | — | — | 0.317 |
| 0.03 | 0.01 | 3.16 | 0 | sí | — | — | 0.317 |
| 0.03 | 0.1 | 1 | 0 | sí | — | — | 0.0327 |
| 0.03 | 0.1 | 3.16 | 0 | sí | — | — | 0.0327 |
| 0.03 | 1.0 | 1 | 0 | sí | — | — | 0.00326 |
| 0.03 | 1.0 | 3.16 | 0 | sí | — | — | 0.00326 |

## Lectura

Con el campo del reloj (N = 2) y las formas por defecto (R₀ = 9/4 > R_c = 25/12), R solo puede crecer en los dos esquemas y en toda d: el candidato 2 no puede producir D = 0 y el criterio (i) integral no se alcanza. La dimensión entra en la trayectoria de los acoplos pero no en el flujo instantáneo de R. Con un escalar (N = 1) hay una ventana de masa en la que R decrece, pero desde el escalado (3.2) la inestabilidad de masa llega antes de entrar en ella. Derivación, no corrida preinscrita: sin letra sobre λ ni sobre la ley.
