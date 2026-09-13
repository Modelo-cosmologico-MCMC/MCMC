# µ(k,a), η, Σ desde la ontología — canal Cronos (E1: **A**, E2: **PASS**)

Preinscripción: `2d8bdcadf38b` (congelada ANTES de computar); ejecución: commit `b53bd4812`. Sin datos: es predicción.

- α₀⁻¹ = 1e-06 — la COTA (11.5) saturada: límite superior, no medida. Fondo de referencia: H0 = 67.87, Ω_m = 0.3263, ε = +0.0172, z_trans = 9.09.
- **E1 (cierre comoving)**: max |R_µ − 1| = 1.55e-04 (regla ≤ 0.001) → desenlace A: el aburrido preinscrito. Cierre physical: el crecimiento DIVERGE — ningún número de fσ8 es citable bajo 2b con α₀⁻¹ en la cota (ver sección siguiente).
- **E2 identidades**: Σ−1 = (µ−1)/2 a 1.6e-16; η−1 = −(µ−1) a segundo orden; k² y linealidad en α₀⁻¹ a 0.0e+00; physical/comoving = (1+z)^(9/2) a 2.2e-16; GR exacto: True.

## Tabla µ − 1 (α₀⁻¹ en su cota; k > ventana marcado ✗)

| k [h/Mpc] | ventana | z=0 com. | z=0.5 com. | z=1 com. | z=2 com. | z=3 com. | z=0 phys. | z=0.5 phys. | z=1 phys. | z=2 phys. | z=3 phys. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | ✓ | 9.74e-07 | 6.49e-07 | 4.87e-07 | 3.25e-07 | 2.43e-07 | 9.74e-07 | 4.03e-06 | 1.10e-05 | 4.55e-05 | 1.25e-04 |
| 0.02 | ✓ | 3.90e-06 | 2.60e-06 | 1.95e-06 | 1.30e-06 | 9.74e-07 | 3.90e-06 | 1.61e-05 | 4.41e-05 | 1.82e-04 | 4.99e-04 |
| 0.05 | ✓ | 2.43e-05 | 1.62e-05 | 1.22e-05 | 8.12e-06 | 6.09e-06 | 2.43e-05 | 1.01e-04 | 2.75e-04 | 1.14e-03 | 3.12e-03 |
| 0.1 | ✓ | 9.74e-05 | 6.49e-05 | 4.87e-05 | 3.25e-05 | 2.43e-05 | 9.74e-05 | 4.03e-04 | 1.10e-03 | 4.55e-03 | 1.25e-02 |
| 0.2 | ✓ | 3.90e-04 | 2.60e-04 | 1.95e-04 | 1.30e-04 | 9.74e-05 | 3.90e-04 | 1.61e-03 | 4.41e-03 | 1.82e-02 | 4.99e-02 |
| 0.3 | ✗ | 8.76e-04 | 5.84e-04 | 4.38e-04 | 2.92e-04 | 2.19e-04 | 8.76e-04 | 3.62e-03 | 9.92e-03 | 4.10e-02 | 1.12e-01 |

## Verificación de las estimaciones a mano de la nota (§B.3)

| k | z=0 com. nota → calc. (×) | z=1 phys. nota → calc. (×) | z=3 phys. nota → calc. (×) |
|---|---|---|---|
| 0.05 | 3.00e-05 → 2.43e-05 (×0.81) | 3.00e-04 → 2.75e-04 (×0.92) | 1.00e-02 → 3.12e-03 (×0.31) |
| 0.1 | 1.00e-04 → 9.74e-05 (×0.97) | 1.00e-03 → 1.10e-03 (×1.10) | 5.00e-02 → 1.25e-02 (×0.25) |
| 0.2 | 4.00e-04 → 3.90e-04 (×0.97) | 5.00e-03 → 4.41e-03 (×0.88) | — → 4.99e-02 |

Lectura: ×≈1 confirma la estimación; ×≠1 la corrige — las cifras citables son las calculadas, no las de la nota.

## Dónde se rompe cada cierre (diagnóstico de no-perturbatividad, α₀⁻¹ en la cota)

| cierre | z(ε̄_c = 0.1) | z(ε̄_c = 1) | z(µ−1 = 1), k=0.01 | z(µ−1 = 1), k=0.02 | z(µ−1 = 1), k=0.05 | z(µ−1 = 1), k=0.1 | z(µ−1 = 1), k=0.2 | z(µ−1 = 1), k=0.3 |
|---|---|---|---|---|---|---|---|---|
| comoving | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) | ∞ (no se alcanza) |
| physical | 74.5 | 125.0 | 51.2 | 34.1 | 19.8 | 13.0 | 8.4 | 6.5 |

**Hallazgo estructural (no previsto en la nota)**: el cierre 'physical' con α₀⁻¹ saturando la cota NO es un cierre perturbativo del sector lineal: µ − 1 ∝ (1+z)^{7/2} alcanza O(1) a z de un dígito o dos en toda la ventana de k, la corrección a la lapse ε̄_c supera 1 para z ≳ 125.0, y la ODE de crecimiento diverge desde su z inicial (999). Además, con ρ_c = 200·ρ̄_m(0) el PROPIO fondo supera el umbral para z > 4.85 — el criterio «200× la media» es, por construcción, relativo a la media de cada época, lo que favorece conceptualmente el cierre 'comoving'. Consecuencia enunciable sin datos: bajo 2b, una amplitud viable exige α₀⁻¹ ≪ 1e-6 — la propia consistencia del sector lineal acota 2b por debajo de la cota galáctica, y ese es el discriminador interno de B.3, más fuerte de lo que la nota estimaba. El contraste con datos (CMB-lensing) sigue PROHIBIDO aquí.

**Atlas**: PENDIENTE (frente 3): los coeficientes O(1) de µ_Atlas y η_Atlas no están derivados — sin coeficientes explícitos la contribución es cero y no se cita. Contribución no computada.

**Lo que este artefacto NO afirma**: que µ ≠ 1 esté detectado; que la amplitud sea la de la cota; que Atlas tenga coeficientes; que k² valga fuera de la ventana. Ningún dato de lensing ni de RSD entra aquí (prohibición preinscrita).
