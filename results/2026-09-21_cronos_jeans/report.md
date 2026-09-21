# El criterio de Cronos–Jeans (derivación del autor, 21-sep-2026; reproducida en el repo)

Commit `9efe00f49`. A = A_Sculptor = 6.202e-07 (M☉/pc³)^(−3/2), congelada en 5E.

**Criterio**: q ≡ (3/2)·c²·ε_c(ρ)/σ_1D² ≥ 1 ⟹ inestable a TODA longitud de onda; en el límite fluido ω² = (σ² − (3/2)c²ε_c)k² − 4πGρ, tasa ∝ k·√(q − 1)·σ (ultravioleta, sin longitud de Jeans propia). Un N-cuerpos con q > 1 no puede converger: refinar el campo aumenta la tasa.

## (1) Halo del Nivel A (NFW 1e+11 M☉, c = 10, Jeans isótropo)

| r [kpc] | ρ [M☉/pc³] | (3/2)c²ε_c [km²/s²] | σ_r² | q | D_F |
|---|---|---|---|---|---|
| 0.05 | 1.108e+00 | 97458 | 280 | **348.47** | 1311.21 |
| 0.10 | 5.482e-01 | 33936 | 461 | **73.66** | 232.13 |
| 0.20 | 2.686e-01 | 11641 | 732 | **15.90** | 41.14 |
| 0.30 | 1.756e-01 | 6150 | 941 | **6.54** | 14.96 |
| 0.40 | 1.291e-01 | 3878 | 1112 | **3.49** | 7.30 |
| 0.50 | 1.013e-01 | 2694 | 1256 | **2.14** | 4.18 |
| 0.70 | 6.960e-02 | 1535 | 1492 | **1.03** | 1.80 |
| 0.90 | 5.212e-02 | 995 | 1678 | **0.59** | 0.96 |
| 1.00 | 4.605e-02 | 826 | 1758 | **0.47** | 0.74 |
| 1.50 | 2.803e-02 | 392 | 2064 | **0.19** | 0.27 |
| 2.30 | 1.594e-02 | 168 | 2363 | **0.07** | 0.09 |
| 5.00 | 4.898e-03 | 29 | 2733 | **0.01** | 0.01 |

**r_CJ (q = 1) = 0.709 kpc; M(<r_CJ) = 1.61e+08 M☉.** Con 0.1·A_S: r_CJ = 0.247 kpc; con 0.05·A_S: 0.180 kpc. La regla de parada v_well = 1000 km/s equivale a ρ = 4.32 M☉/pc³: la masa dentro de r_CJ cabría en ≈ 0.21 kpc.

## (2) Vecindad solar (losa MPH15: estrellas 0.043, gas 0.041, MO 0.013 M☉/pc³)

| z [pc] | ρ | c²ε_c [km²/s²] | v_well | g_C [(km/s)²/kpc] | K_z(losa) | g_C/K_z | q (σ_z = 20) |
|---|---|---|---|---|---|---|---|
| 0 | 0.097 | 1684 | 58.0 | 2 | 1 | **3.85** | 6.31 |
| 50 | 0.095 | 1635 | 57.2 | 1909 | 261 | **7.32** | 6.13 |
| 100 | 0.090 | 1503 | 54.8 | 3250 | 512 | **6.35** | 5.63 |
| 200 | 0.074 | 1133 | 47.6 | 3721 | 957 | **3.89** | 4.25 |
| 300 | 0.059 | 805 | 40.1 | 2759 | 1317 | **2.09** | 3.02 |
| 500 | 0.039 | 427 | 29.2 | 1230 | 1837 | **0.67** | 1.60 |
| 800 | 0.024 | 204 | 20.2 | 418 | 2328 | **0.18** | 0.76 |
| 1100 | 0.017 | 126 | 15.9 | 147 | 2652 | **0.06** | 0.47 |

Límite de Oort efectivo de Cronos: **0.75 M☉/pc³** (solo estrellas 0.115) frente al medido ρ_dyn(0) = 0.10 ± 0.01 (referencia, no ingerida). Cotas: g_C ≤ 0.1·K_z en 300 pc ⟹ A ≤ 0.048·A_S; q < 1 en el plano ⟹ A < 0.158·A_S.

## (3) Sculptor con el montaje congelado del 5E (M* = 5.20e+06 M☉, Υ = 2, a₀ = 260 pc, β = 0)

| r [pc] | ρ* | (3/2)c²ε_c | σ_r²(N) | σ_r²(N+C) | q | g_C/g_N |
|---|---|---|---|---|---|---|
| 5 | 7.057e-02 | 1567.2 | 14.33 | 641.24 | **2.44** | 91.11 |
| 50 | 6.443e-02 | 1367.2 | 14.08 | 561.00 | **2.44** | 80.95 |
| 100 | 4.955e-02 | 922.2 | 13.36 | 382.29 | **2.41** | 57.54 |
| 150 | 3.401e-02 | 524.3 | 12.39 | 222.21 | **2.36** | 35.28 |
| 200 | 2.184e-02 | 269.9 | 11.34 | 119.37 | **2.26** | 19.84 |
| 260 | 1.214e-02 | 111.9 | 10.09 | 54.89 | **2.04** | 9.25 |
| 400 | 3.226e-03 | 15.3 | 7.74 | 13.88 | **1.10** | 1.65 |
| 600 | 6.733e-04 | 1.5 | 5.66 | 6.24 | **0.23** | 0.22 |
| 1000 | 7.124e-05 | 0.1 | 3.61 | 3.63 | **0.01** | 0.01 |

| R [pc] | σ_los Newton | σ_los Newton + Cronos (A_S) | observado (referencia) |
|---|---|---|---|
| 10 | 3.56 | **19.14** | ≈ 9–10, plano |
| 30 | 3.55 | **18.74** | ≈ 9–10, plano |
| 60 | 3.51 | **17.47** | ≈ 9–10, plano |
| 100 | 3.44 | **14.96** | ≈ 9–10, plano |
| 150 | 3.31 | **11.50** | ≈ 9–10, plano |
| 200 | 3.17 | **8.51** | ≈ 9–10, plano |
| 260 | 2.99 | **5.95** | ≈ 9–10, plano |
| 350 | 2.75 | **3.87** | ≈ 9–10, plano |
| 500 | 2.42 | **2.63** | ≈ 9–10, plano |
| 700 | 2.10 | **2.13** | ≈ 9–10, plano |
| 1000 | 1.79 | **1.79** | ≈ 9–10, plano |

Inestable (q ≥ 1) hasta r ≈ 420 pc. pico central ≈ 2× lo observado y exterior newtoniano ÷3–4 (β = 0): la forma es la predicción congelada de jeans-dsph. El promedio pesado por luminosidad reproduce σ_obs = 9.2 km/s por construcción.

## Test 1D del umbral (láminas, sin gravedad, σ = ρ₀ = L = 1)

| q | rms(δ) inicial (Poisson) | t = 0.05 | 0.1 | 0.2 | 0.5 | factor | √(q−1) |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.0198 | 0.028 | 0.028 | 0.027 | 0.028 | ×1.4 | 0.00 |
| 1.2 | 0.0198 | 0.043 | 0.061 | 0.072 | 0.091 | ×4.6 | 0.45 |
| 2.0 | 0.0198 | 0.270 | 0.380 | 0.407 | 0.442 | ×22.4 | 1.00 |
| 4.0 | 0.0198 | 0.644 | 0.650 | 0.649 | 0.650 | ×32.9 | 1.73 |

q < 1: rms se queda en el ruido de Poisson; q > 1: crece (y satura pronto para q ≳ 2); el umbral está donde la derivación lo pone.

## Control externo (nota del autor) reproducido

r_CJ 0.709 → 0.709; Oort 0.75 → 0.747; cota K_z 0.048 → 0.048; σ_los(10 pc) 19.14 → 19.14.

## Expectativas declaradas (E13) y lo que NO afirma

- un N-cuerpos con la ley local y q > 1 no converge: refinar la resolución del campo aumenta la tasa
- la masa dentro de r_CJ(A_S) = 0.71 kpc del halo del Nivel A colapsa en ~t_cross hasta la regla de parada (ρ ≈ 4.3 M☉/pc³ ⟺ 1.6e8 M☉ dentro de ≈ 0.2 kpc)
- el propio montaje congelado de Sculptor tiene q ≈ 2–2.4 dentro de ≈ 430 pc: el equilibrio que calibra A_Sculptor es inestable según el criterio
- la vecindad solar exige A ≤ 0.048·A_Sculptor (g_C ≤ 0.1·K_z en 300 pc) y A < 0.16·A_Sculptor (q < 1 en el plano)

- ningún resultado observacional: las referencias (Oort, K_z, perfil de Sculptor) se citan como contexto, no se ingieren
- la ley de la tasa ∝ k no se mide aquí (fase lineal cortísima, modos sembrados por Poisson)
- A_Sculptor no se toca; ninguna preinscripción cambia; el Nivel A sigue INDETERMINADO
