# Frente 5 (b) — serie de resolución del campo: desenlace **INDETERMINADO**

Preinscripción `18c5344d9d0b`; análisis en `4b48879af`. Puertas: {'energy_newton': True, 'energy_self_cronos': False, 'weak_regime': True, 'runs_complete': True}. Corridas ausentes: ninguna.

## Energía y puertas por brazo

| brazo | k_inner | |ΔE_self/E| | |ΔE_naive/E| | |Δ(K+W)/E| | W_fric/|E| | t_final | parada |
|---|---|---|---|---|---|---|---|
| a | 64 | 2.43e-04 | 2.43e-04 | 2.43e-04 | 0.00e+00 | 1.000 | — |
| bAS_k64 | 64 | 2.40e-02 | 2.44e-02 | 2.38e-02 | 6.11e-06 | 1.000 | — |
| bAS_k128 | 128 | 3.96e-02 | 3.99e-02 | 3.93e-02 | 4.33e-06 | 1.000 | — |
| bAS_k256 | 256 | 6.24e-03 | 6.35e-03 | 6.17e-03 | 1.69e-08 | 1.000 | — |
| b005_k64 | 64 | 2.39e-04 | 2.75e-04 | 2.34e-04 | 5.21e-10 | 1.000 | — |
| b005_k128 | 128 | 2.44e-04 | 2.72e-04 | 2.37e-04 | 4.60e-10 | 1.000 | — |
| b005_k256 | 256 | 2.45e-04 | 2.73e-04 | 2.27e-04 | 3.76e-10 | 1.000 | — |

## Serie A_Sculptor: M(<0.4 kpc) final por k_inner

M04 = {'64': 12892877.766130764, '128': 12214305.252123881, '256': 33928625.70034411}; saltos log10: {'64->128': -0.023481095849522914, '128->256': 0.4436974992327127}; converge: False.

## Serie 0.05·A_Sculptor: M(<0.4 kpc) final por k_inner

M04 = {'64': 59035808.718598746, '128': 52250083.578529924, '256': 52928656.09253681}; saltos log10: {'64->128': -0.05302852744613664, '128->256': 0.00560387751799858}; converge: True.

## Cociente b/a por bandas (t final)

| brazo | [0.1,0.4) | [0.4,1.0) | [1.0,2.3) | [2.3,5.0) | [5.0,20.0) |
|---|---|---|---|---|---|
| bAS_k64 | -0.445 | -0.194 | -0.036 | -0.002 | +0.001 |
| bAS_k128 | -0.438 | -0.204 | -0.089 | -0.013 | +0.000 |
| bAS_k256 | -0.046 | -0.074 | -0.064 | -0.004 | +0.001 |
| b005_k64 | +0.332 | +0.056 | -0.005 | -0.007 | -0.000 |
| b005_k128 | +0.166 | +0.069 | -0.005 | -0.003 | -0.000 |
| b005_k256 | +0.126 | +0.076 | -0.015 | -0.005 | +0.000 |

letra bajo la regla congelada; A confirma la predicción del criterio (no convergencia a A_Sculptor, convergencia a 0.05·A_Sculptor); ninguna letra es afirmación sobre el tratado: sitúa la Ley de Cronos débil con A_Sculptor frente a su propia inestabilidad ultravioleta (E8).

## Lo que no decide

- si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): diccionario)
- la amplitud (A_Sculptor es hipótesis del 5E; 0.05 es brazo)
- el interior por debajo de ε_soft = 0.1 kpc
