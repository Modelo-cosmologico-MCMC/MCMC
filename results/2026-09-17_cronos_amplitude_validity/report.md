# Amplitud de Cronos: un solo cierre (cálculo de consistencia, 17-sep-2026)

Commit `0f8e37e57`. θ de fondo = medianas congeladas del 12-sep (H0 = 67.87, Ω_m = 0.3263); ρ̄_m(0) = 4.171e-08 M☉/pc³; α₀⁻¹ = 1e-06.

**Lectura obligatoria**: (α₀⁻¹, ρ_c) entran solo por A = α₀⁻¹/ρ_c^(3/2). El cierre cosmológico ρ_c = 200·ρ̄_m da A = 41.5; el galáctico (5E, congelado) A_Sculptor = 6.202e-07: cociente **6.69e+07** (ρ_c galáctica equivalente 1.375 M☉/pc³ = 3.30e+07 ρ̄_m). Dentro de un halo NFW de 1e11 M☉ el cierre cosmológico viola la subdominancia c²ε_c ≲ |Φ_N| (Cor. 11.3c) en todo el halo: EXCLUIDO dinámicamente. El galáctico la cumple en r ≥ 0.1 kpc y coincide con A_max. **La amplitud del modelo operativo es una y es A_Sculptor.**

| halo | cierre | A | D(suav.) | D(0.38 kpc) | D(2.3 kpc) | D(r_s) | D(r200) | r(D = 1) | ε_c(0.38 kpc) | ε_c(r200) | subdominante |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e+11 M☉, c = 10 | galactic | 6.2e-07 | 7.69e-01 | 9.71e-02 | 4.21e-03 | 1.46e-04 | 8.09e-08 | — | 3.1e-08 | 6.4e-15 | sí |
| 1e+11 M☉, c = 10 | A_max | 8.06e-07 | 1.00e+00 | 1.26e-01 | 5.47e-03 | 1.89e-04 | 1.05e-07 | 0.10 kpc | 4.1e-08 | 8.3e-15 | sí |
| 1e+11 M☉, c = 10 | cosmological | 41.5 | 5.15e+07 | 6.50e+06 | 2.82e+05 | 9.76e+03 | 5.42e+00 | 97.75 kpc | 2.1e+00 | 4.3e-07 | NO |
| 1e+10 M☉, c = 13 | galactic | 6.2e-07 | 4.99e+00 | 1.92e-01 | 4.69e-03 | 1.62e-03 | 3.39e-07 | 0.14 kpc | 1.5e-08 | 5.5e-15 | NO |
| 1e+10 M☉, c = 13 | A_max | 1.24e-07 | 1.00e+00 | 3.85e-02 | 9.40e-04 | 3.25e-04 | 6.81e-08 | 0.05 kpc | 2.9e-09 | 1.1e-15 | sí |
| 1e+10 M☉, c = 13 | cosmological | 41.5 | 3.34e+08 | 1.28e+07 | 3.14e+05 | 1.08e+05 | 2.27e+01 | 45.37 kpc | 9.8e-01 | 3.7e-07 | NO |

A_max(1e11, suavizado 0.1 kpc) = 1.30·A_Sculptor; M(< 2.3 kpc) = 1.41e+09 M☉, M(< 0.4 kpc) = 5.38e+07 M☉.

### Cociente de FUERZAS D_F = |c² dε_c/dr| / g_N (lo que «dominar sobre la gravedad» significa en la dinámica)

| halo | cierre | D_F(suav.) | D_F(0.4 kpc) | D_F(1 kpc) | D_F(2.3 kpc) | D_F(r_s) | r(D_F = 1) | subdominante en fuerza |
|---|---|---|---|---|---|---|---|---|
| 1e+11 M☉, c = 10 | galactic | 2.29e+02 | 7.22e+00 | 7.45e-01 | 8.83e-02 | 1.57e-03 | 0.89 kpc | NO |
| 1e+11 M☉, c = 10 | A_max | 2.97e+02 | 9.38e+00 | 9.69e-01 | 1.15e-01 | 2.04e-03 | 0.98 kpc | NO |
| 1e+11 M☉, c = 10 | cosmological | 1.53e+10 | 4.83e+08 | 4.99e+07 | 5.91e+06 | 1.05e+05 | 97.75 kpc | NO |
| 1e+10 M☉, c = 13 | galactic | 1.07e+03 | 6.00e+00 | 5.85e-01 | 5.85e-02 | 1.74e-02 | 0.81 kpc | NO |
| 1e+10 M☉, c = 13 | A_max | 2.15e+02 | 1.20e+00 | 1.17e-01 | 1.17e-02 | 3.50e-03 | 0.43 kpc | NO |
| 1e+10 M☉, c = 13 | cosmological | 7.17e+10 | 4.02e+08 | 3.91e+07 | 3.92e+06 | 1.17e+06 | 45.37 kpc | NO |

**Corrección a la lectura «A_Sculptor está donde (P) dice»**: en una cúspide NFW D_F/D_Φ = (3/2)|dlnρ/dlnr|·|Φ_N|r/(G M(<r)) ≫ 1 (|Φ_N| es finito en el centro; G M(<r)/r → 0). Con A_Sculptor la fuerza de Cronos iguala a la gravedad en r ≈ 0.89 kpc del halo de 1e11 M☉ y la domina dentro: subdominante en potencial, NO en fuerza. Coherente con el origen de A_Sculptor (5A/5B: Cronos sustituye a la materia oscura en Sculptor con bariones solos). El régimen débil (ε_c ≪ 1) sí se cumple; la exclusión del cierre cosmológico es aún más fuerte en fuerza.

Dos matices respecto de la nota del autor: (i) con el cierre cosmológico la parte externa del halo está excluida por subdominancia con ε_c ≪ 1 (la lectura de la nota), pero por debajo de ~r_s se rompe TAMBIÉN el régimen débil (ε_c > 1e-3 en r ≲ r_s, > 1 en r ≲ 0.4 kpc); (ii) en el halo enano de 1e10 M☉ (c = 13, suavizado 50 pc) A_Sculptor deja de ser subdominante por debajo de r(D = 1) = 0.14 kpc (A_max = 0.20·A_Sculptor): la puerta de régimen del Nivel A (D ≤ 1 para r > ε_soft) decide por sí sola qué halo y qué suavizado son admisibles.

## Consecuencia para el sector µ/η del canal Cronos (E1)

ε̄_c(hoy): cosmológico 3.54e-10, galáctico 5.28e-18. µ − 1 en k = 0.2 h/Mpc:

| z | cierre cosmológico (E1 publicado) | cierre galáctico (A_Sculptor) |
|---|---|---|
| 0.0 | 3.90e-04 | 5.82e-12 |
| 1.0 | 1.95e-04 | 6.58e-11 |
| 3.0 | 9.74e-05 | 7.45e-10 |

El artefacto E1 (results/2026-09-13_mu_eta_cronos) no se retoca: sus identidades son exactas para el cierre que declara; lo que cambia es el estatuto de ese cierre (excluido) y, con él, la amplitud de la cola k²: muere por consistencia interna. Control externo del autor reproducido (véase JSON).
