# Diccionario, ecuación 1 — la forma saturante ε_c(ρ) y sus ligaduras (22-sep-2026)

Commit `41af2700f`. Forma DECLARADA: ε_c(ρ) = ε_max·ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}); A_débil ≡ ε_max/ρ*^{3/2}. A_Sculptor = 6.2016e-07 (congelada, 5E).

## Ligaduras

- **(i) Oort–K_z**: Δρ_eff(0) = −c²ε_c'(ρ₀)ρ''(0)/(4πG) ≤ margen + 2σ = 0.0449 M☉/pc³ (ρ₀ = 0.097, losa declarada) ⟹ ε_c'(ρ₀) ≤ 1.741e-08; en la ley débil equivale a A ≤ 0.060·A_Sculptor (el 0.060 de §3.26).
- **(ii) estabilidad** q = c²ρε_c'(ρ)/σ² < 1 en la tabla declarada: plano solar (losa MPH2015) (ρ = 0.097, σ² = 400); Sculptor centro (Plummer, Υ⋆ = 2.0) (ρ = 0.0318, σ² = 84.6); halo NFW 1e11 (r = 0.05 kpc) (ρ = 1.11, σ² = 280); halo NFW 1e11 (r = 0.4 kpc) (ρ = 0.129, σ² = 1.11e+03); halo NFW 1e11 (r = 1.0 kpc) (ρ = 0.046, σ² = 1.76e+03); halo NFW 1e11 (r = 5.0 kpc) (ρ = 0.0049, σ² = 2.73e+03).
- **(iii) perfil de Sculptor**: no evaluable: walker2009 DATA_UNAVAILABLE (fallo cerrado).

Región permitida por (i) ∧ (ii) sobre la malla log ε_max ∈ [−14, −2] × log ρ* ∈ [−5, 2]: 58.0 % (solo Oort 65.0 %, solo estabilidad 58.1 %). Máscara en `allowed_mask.npy`.

## La calibración de Sculptor frente a las ligaduras

Reproducir la FUERZA del 5E en el centro de Sculptor fija ε_c'(ρ_S) = (3/2)A_S ρ_S^{1/2} y define una curva ε_max(ρ*). Sobre ella:

| Υ⋆ | ρ_S [M☉/pc³] | q_S bajo su propia calibración (independiente de la forma) | ρ* con Oort ∧ resto de sistemas | ρ* con Oort ∧ TODOS (incl. Sculptor) | mínimo exceso sobre Oort (ρ*) |
|---|---|---|---|---|---|
| 1.0 | 0.0159 | 1.98 | 72 de 141 | 0 | ×0.07 (ρ* = 1.0e-05) |
| 2.0 | 0.0318 | 5.60 | 12 de 141 | 0 | ×0.59 (ρ* = 1.0e-05) |
| 3.0 | 0.0477 | 10.29 | 0 de 141 | 0 | ×1.98 (ρ* = 1.0e-05) |

calibrar la FUERZA en el centro de Sculptor fija ρ_S·ε_c'(ρ_S) = (3/2)A_S·ρ_S^{3/2} y por tanto q_S = c²ρ_Sε_c'(ρ_S)/σ² sea cual sea la forma ε_c(ρ): la (in)estabilidad de Sculptor bajo su propia calibración no depende del diccionario, solo de A_S, ρ_S(Υ⋆) y σ.

**Lectura**: (a) la saturación SÍ puede reconciliar la fuerza de Sculptor con la vecindad solar si la densidad estelar central de Sculptor es baja (Υ⋆ = 1: ρ_S = 0.016 ≪ ρ₀ = 0.097; con Υ⋆ = 2 también hay ρ* que pasan Oort; con Υ⋆ = 3 ninguno) — la ley débil pura no podía (A_S excluida ×17); (b) pero q_S > 1 para los tres Υ⋆: Sculptor es inestable bajo su propia calibración sea cual sea la forma, porque la calibración fija ρ_S·ε_c'(ρ_S). Ese es el hecho que ninguna forma ε_c(ρ) cambia: o la amplitud del 5E no es la fuerza de Sculptor, o el criterio de Cronos–Jeans no se aplica a Sculptor (σ_los ≠ σ del medio), o la inestabilidad es real y el perfil σ_los(R) lo dirá (ligadura iii, pendiente de bytes).

## Predicción congelada: el escalón en la RAR

`rar_step_prediction.json` (sha256 `9566a2ac0457…`), 12 curvas (4 puntos × 3 discos exponenciales declarados). Con saturación g_C = c²ε_c'(ρ)|∇ρ| solo actúa donde ρ ≈ ρ*: la desviación g_obs − g_bar se concentra en el intervalo de g_bar donde ρ_disco(R) cruza ρ*, no sigue una función continua de g_bar.

| punto | disco Σ₀ [M☉/pc²] | A_débil/A_S | R(ρ = ρ*) [kpc] | máx g_C/g_bar | R del máximo [kpc] |
|---|---|---|---|---|---|
| calibración Sculptor Υ⋆ = 1, ρ* = 0.001 | 100 | 4.15e+03 | 15.35 | 3.98 | 16.39 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.001 | 300 | 4.15e+03 | 18.64 | 1.98 | 19.55 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.001 | 1000 | 4.15e+03 | 14.84 | 1.29 | 15.20 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.01 | 100 | 9.03 | 8.44 | 0.0987 | 9.66 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.01 | 300 | 9.03 | 11.74 | 0.055 | 13.07 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.01 | 1000 | 9.03 | 10.23 | 0.0411 | 10.96 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.03 | 100 | 1.84 | 4.99 | 0.0651 | 6.30 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.03 | 300 | 1.84 | 8.28 | 0.0368 | 9.66 |
| calibración Sculptor Υ⋆ = 1, ρ* = 0.03 | 1000 | 1.84 | 7.93 | 0.0291 | 8.74 |
| borde de Oort–K_z (2σ), ρ* = 0.03 | 100 | 2.79 | 5.14 | 0.0935 | 6.46 |
| borde de Oort–K_z (2σ), ρ* = 0.03 | 300 | 2.79 | 8.44 | 0.0529 | 9.66 |
| borde de Oort–K_z (2σ), ρ* = 0.03 | 1000 | 2.79 | 8.03 | 0.0416 | 8.96 |

## Lo que NO afirma

- la forma (E1) es DECLARADA (mínima compatible con Def. 6.4 y §11.4), no derivada de la ontología
- ningún parámetro se ajusta: el mapa dice qué (ε_max, ρ*) sobreviven a las ligaduras ya derivadas
- la ligadura (iii) (perfil de Sculptor) no se evalúa: walker2009 sigue en fallo cerrado
- la predicción RAR no se contrasta: SPARC sin ingerir; el contraste será una preinscripción propia
