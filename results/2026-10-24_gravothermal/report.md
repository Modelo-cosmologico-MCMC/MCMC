# Código gravotérmico 1D de la MCV (c) con borde de ECV: corridas de comprobación, sin letra

Commit `7a1e11885`. Módulo `dynamics/gravothermal.py`. Borde de ECV: «el borde de ECV corresponde al punto de intercambio entre la burbuja de MCV y la ECV fuera de la influencia del punto gravitacional y térmico (autor, 24-oct)».

Banda declarada antes de la primera corrida del colapso: C·σ̂·t_c ∈ [120, 180] (150/β de Essig et al. (2019) ± 20 %, régimen LMFP).

| corrida | σ̂ | C | capas | borde (r_s) | brazo | C·σ̂·t_c | ρ_c mín (ρ_s) | ΔE/E | flujo al borde | Δu borde |
|---|---|---|---|---|---|---|---|---|---|---|
| referencia | 0.01 | 0.75 | 150 | 100 | cerrado | 151.91 | 2.478 | 2.1e-03 | 8.8e-10 | 1.7e-02 |
| intercambio | 0.01 | 0.75 | 150 | 100 | intercambio | 151.91 | 2.478 | 2.1e-03 | 8.8e-10 | 1.7e-02 |
| capas_300 | 0.01 | 0.75 | 300 | 100 | cerrado | 152.95 | 2.463 | 4.4e-04 | 7.5e-10 | 1.9e-02 |
| eps_5e-4 | 0.01 | 0.75 | 150 | 100 | cerrado | 151.49 | 2.474 | 2.1e-03 | 8.9e-10 | 1.7e-02 |
| sigma_0.001 | 0.001 | 0.75 | 150 | 100 | cerrado | 151.91 | 2.478 | 1.8e-03 | 1.2e-10 | 1.7e-02 |
| C_0.45 | 0.01 | 0.45 | 150 | 100 | cerrado | 151.91 | 2.478 | 2.0e-03 | 6.6e-10 | 1.7e-02 |
| sigma_0.1 | 0.1 | 0.75 | 150 | 100 | cerrado | 152.23 | 2.479 | 5.7e-03 | 2.0e-08 | 1.8e-02 |
| sigma_0.3 | 0.3 | 0.75 | 150 | 100 | cerrado | 154.55 | 2.481 | 1.0e-02 | 9.0e-08 | 1.8e-02 |
| sigma_1 | 1 | 0.75 | 150 | 100 | cerrado | 179.16 | 2.503 | 2.0e-02 | 4.8e-07 | 2.4e-02 |
| borde_10 | 0.01 | 0.75 | 120 | 10 | cerrado | 161.97 | 2.476 | 4.3e-03 | 2.1e-06 | 2.8e-01 |
| borde_20 | 0.01 | 0.75 | 129 | 20 | cerrado | 153.47 | 2.478 | 2.9e-03 | 2.6e-07 | 1.3e-01 |
| borde_40 | 0.01 | 0.75 | 138 | 40 | cerrado | 152.11 | 2.478 | 2.4e-03 | 2.5e-08 | 5.2e-02 |
| borde_80 | 0.01 | 0.75 | 147 | 80 | cerrado | 151.92 | 2.478 | 2.1e-03 | 2.0e-09 | 2.2e-02 |
| borde_160 | 0.01 | 0.75 | 156 | 160 | cerrado | 151.90 | 2.478 | 2.0e-03 | 1.5e-10 | 1.0e-02 |
| borde_200 | 0.01 | 0.75 | 159 | 200 | cerrado | 151.90 | 2.478 | 2.0e-03 | 6.3e-11 | 7.8e-03 |
| pendiente | 0.001 | 0.75 | 400 | 100 | cerrado | 153.10 | 2.461 | 1.7e-04 | 1.0e-10 | 1.9e-02 |
| halo_autor | 0.249427 | 0.75 | 150 | 100 | cerrado | 153.76 | 2.480 | 9.5e-03 | 7.0e-08 | 1.8e-02 |

- Referencia: C·σ̂·t_c = **151.91**, dentro de la banda: **sí**.
- Universalidad LMFP (σ̂ = 0.01 y 0.001; C = 0.75 y 0.45): dispersión relativa 2.6e-05.
- Brazo «intercambio» frente a «cerrado»: -2.9e-08. Resolución (300 capas): +0.68%. Paso (eps/2): -0.28%.
- Duplicar el borde (cambio relativo de t_c; tolerancia 0.001; «fuera» exige además flujo por debajo de 1e-06 en las dos corridas): 10 → 20 r_s: -5.3e-02 (dentro), 20 → 40 r_s: -8.9e-03 (dentro), 40 → 80 r_s: -1.2e-03 (dentro), 80 → 160 r_s: -1.5e-04 (fuera), 100 → 200 r_s: -7.6e-05 (fuera).
- Flujo conductivo al borde por encima de 1e-06 del máximo (dentro de la influencia térmica): borde_10.
- La capa del borde se comprime adiabáticamente al expandirse el interior (columna Δu borde): la influencia mecánica llega a la pared, y la comprobación es que no mueva t_c al duplicar el borde.
- Fase de núcleo: ρ_c mínima 2.478 ρ_s en C·σ̂·t = 18.2 (0.120 t_c).
- Pendiente autosemejante (ρ/ρ_c ∈ [10⁻⁶, 10⁻³], 400 capas, σ̂ = 0.001): -2.183 con ρ_c = 1.0e+06 (53 capas), -2.165 con ρ_c = 1.0e+08 (47 capas); Balberg et al. (2002): −2.19.
- Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g; σ̂ = 0.249): t_c = 46.0 Gyr frente a 44.9 Gyr de la fórmula (cociente 1.025).

La ventana de la pendiente se fijó tras una primera medida: junto al núcleo (ρ/ρ_c ∈ [10⁻³, 10⁻¹]) la pendiente pasa de −2.4, como en la esfera isoterma, y la ley de potencias autosemejante solo aparece más lejos.

**Lectura**: sin letra (E8). El código reproduce el t_c de Essig et al. (2019): C·σ̂·t_c = 151.9 frente a 150 (banda declarada [120, 180]), y la pendiente autosemejante de Balberg et al. (2002): -2.18 frente a −2.19. Con el borde en 100 r_s, el borde queda fuera de la influencia térmica y gravitatoria: flujo al borde 8.8e-10 del máximo, y duplicarlo mueve t_c -7.6e-05. En r200 de un halo con c = 10 (10 r_s) no: flujo 2.1e-06 y duplicarlo mueve t_c -5.3%. El primer borde de la serie que pasa las dos comprobaciones está en 80 r_s.
