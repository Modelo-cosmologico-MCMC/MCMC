# Preinscripción — Frente 5 (b), ronda 3: integrador con rango M(<r) actualizado, barrido de suavizado ε_soft, serie ds y brazo de control saturante, bajo puertas heredadas de la cualificación E8-Q

Congelada 2026-09-30T14:25:08.272589+00:00 en el commit `5ec25b043` (contiene el generador). sha256 de `preregistration.json`: `6fba6b53d92910db86e21ed5ae9c8943d9c50f32dcb33b52e3d6eae01bf513ec`. Cualificación citada: `64b9693e6d9a…` (results/2026-09-28_qualification_shells).

**Pregunta**: ¿Depende t_weak de ε_soft como una potencia (la ley) o es independiente (el integrador)? ¿Converge la serie ds a A_Sculptor? ¿Se comporta la forma saturante sub-umbral como predice la ecuación 1 (no sale, converge) y la supra-umbral sale?

**Puertas heredadas** (3 × el suelo medido de la misma clase, rango actualizado): newton ≤ 6.75e-06; AS ≤ 8.88e-02; b005 ≤ 8.34e-04; sat ≤ 8.88e-02. La regla del brief (3 × suelo newtoniano = 6.75e-06) es alcanzable para los brazos de Cronos: {'AS': False, 'b005': False} — se usa el suelo propio. Estacionariedad ≤ 0.1 dex (máximo medido 0.0994). dt_max = 0.01 Myr en los brazos a A_Sculptor.

**Control saturante**: ρ* = ρ_NFW(r = 0.1 kpc) de la tabla de Jeans del halo (M200 = 1e11, c = 10): ρ* = 0.5482 M☉/pc³, ε_max_crit = 1.083e-08; puntos 'sub' ε_max = 5.415e-09 (×0.5, q_máx = 0.50, viable True); 'above' ε_max = 1.624e-08 (×1.5, q_máx = 1.50, viable False).

## Brazos

| brazo | forma | A/A_S | ds | ε_soft | N | t_end [Gyr] |
|---|---|---|---|---|---|---|
| newton_N1e6 | weak | 0.0 | 0.04 | 0.1 | 1000000 | 0.05 |
| sweep_AS_eps0.2 | weak | 1.0 | 0.04 | 0.2 | 1000000 | 0.02 |
| sweep_AS_eps0.1 | weak | 1.0 | 0.04 | 0.1 | 1000000 | 0.02 |
| sweep_AS_eps0.05 | weak | 1.0 | 0.04 | 0.05 | 1000000 | 0.02 |
| sweep_AS_eps0.025 | weak | 1.0 | 0.04 | 0.025 | 1000000 | 0.02 |
| sweep_b005_eps0.2 | weak | 0.05 | 0.04 | 0.2 | 1000000 | 0.05 |
| sweep_b005_eps0.1 | weak | 0.05 | 0.04 | 0.1 | 1000000 | 0.05 |
| sweep_b005_eps0.05 | weak | 0.05 | 0.04 | 0.05 | 1000000 | 0.05 |
| sweep_b005_eps0.025 | weak | 0.05 | 0.04 | 0.025 | 1000000 | 0.05 |
| series_AS_ds0.02 | weak | 1.0 | 0.02 | 0.1 | 1000000 | 0.02 |
| series_AS_ds0.08 | weak | 1.0 | 0.08 | 0.1 | 1000000 | 0.02 |
| sat_sub_ds0.02 | saturating | sub | 0.02 | 0.1 | 1000000 | 0.05 |
| sat_sub_ds0.04 | saturating | sub | 0.04 | 0.1 | 1000000 | 0.05 |
| sat_sub_ds0.08 | saturating | sub | 0.08 | 0.1 | 1000000 | 0.05 |
| sat_above_ds0.04 | saturating | above | 0.04 | 0.1 | 1000000 | 0.05 |
| ctrlN_AS_N1e5 | weak | 1.0 | 0.04 | 0.1 | 100000 | 0.02 |
| ctrlN_b005_N1e5 | weak | 0.05 | 0.04 | 0.1 | 100000 | 0.05 |

## Reglas (congeladas)

- **energy_gate_factor**: 3.0
- **stationarity_dex**: 0.1
- **n_min_shells_within_0p4**: 2000
- **N_control_factor**: 2.0
- **p_power_law_min**: 0.25
- **p_independent_max**: 0.1
- **fit_r2_min**: 0.9
- **sat_convergence_dex**: 0.03
- **letters**: {'A': "barrido: p ≥ p_power_law_min con r² ≥ fit_r2_min en las dos amplitudes; serie ds a A_Sculptor sin convergencia (todas salen y t_weak decrece estrictamente al refinar); control saturante: 'sub' no sale en ninguna ds y M(<0.4) converge con ds (≤ sat_convergence_dex), 'above' sale", 'B': "barrido con p ≥ p_power_law_min pero el control saturante 'sub' TAMBIÉN sale del régimen débil", 'C': 'barrido con |p| ≤ p_independent_max en A_Sculptor (t_weak independiente de ε_soft: el integrador manda; la lectura de la adenda se retira)', 'INDETERMINADO': 'puerta violada (energía, capas interiores, estacionariedad, control de N, corridas ausentes) o p entre p_independent_max y p_power_law_min o r² < fit_r2_min; ningún otro patrón se clasifica'}
- **order**: puertas → C → B → A; ningún umbral se toca tras ver los números

## Expectativas (E13)

- **letter**: A
- **p**: p > 0: r_CJ ∝ A^{2/5}-ish y la cúspide resuelta más adentro colapsa antes (adenda del autor); el brief espera independencia (C)
- **sat_sub**: no sale; M(<0.4) converge con ds
- **sat_above**: sale

## Lo que no puede decidir

- si la Ley de Cronos débil o la saturante es la ley (el brazo saturante es CONTROL: prueba la ecuación 1 como física solo si se cumple la predicción)
- la amplitud (A_Sculptor es hipótesis del 5E)
- los modos no radiales y el interior por debajo de ε_soft
