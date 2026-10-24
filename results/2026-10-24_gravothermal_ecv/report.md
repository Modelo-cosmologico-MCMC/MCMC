# La fuerza de la ECV en el código gravotérmico y la pared en R₀: corridas de referencia, sin letra

Commit `b18aa8354`. σ̂ = 0.01, C = 0.75, NFW inicial, pared en el R₀ autoconsistente; «sin» es la misma pared fija sin la fuerza de la ECV.

| c | R₀ (r_s) | q_E | capas (doc) | C·σ̂·t_c sin / con (doc) | cambio (doc) | capas (ref) | C·σ̂·t_c sin / con (ref) | cambio (ref) | duplicar capas (ref) | flujo al borde |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 37.0 | 7.878e-05 | 137 | 152.161 / 140.701 | -7.53% | 548 | 153.478 / 141.764 | -7.63% | +3.7e-04 (pasa) | 1.7e-07 |
| 10 | 68.3 | 1.530e-05 | 145 | 151.940 / 147.568 | -2.88% | 580 | 153.238 / 148.761 | -2.92% | +4.0e-04 (pasa) | 1.6e-08 |
| 15 | 99.0 | 5.587e-06 | 150 | 151.916 / 149.611 | -1.52% | 599 | 153.207 / 150.844 | -1.54% | +4.1e-04 (pasa) | 3.8e-09 |
| 20 | 129.4 | 2.687e-06 | 153 | 151.899 / 150.472 | -0.94% | 613 | 153.198 / 151.734 | -0.96% | +4.1e-04 (pasa) | 1.3e-09 |

Serie de resolución de c = 10 con la fuerza de la ECV (duplicar las capas):

- 145→289 capas: +6.5e-03 (no pasa la tolerancia 0.001)
- 289→577 capas: +1.6e-03 (no pasa la tolerancia 0.001)
- 577→1153 capas: +4.0e-04 (pasa la tolerancia 0.001)

- Cambio de t_c por la fuerza de la ECV, frente a las cifras del documento (−7.5 / −2.9 / −1.5 % para c = 5 / 10 / 15): espaciado del documento c = 5: -7.53%, c = 10: -2.88%, c = 15: -1.52%, c = 20: -0.94%; espaciado de referencia c = 5: -7.63%, c = 10: -2.92%, c = 15: -1.54%, c = 20: -0.96%.
- Fuerza neta en la pared: |·| ≤ 4.8e-15 de la atracción en todas las corridas con la ECV.
- Pared en 1.25·R₀ con la fuerza de la ECV: con la fuerza de la ECV, la pared (r_b = 85.4097 r_s) queda más allá de R₀ = 69.82 r_s: sujetaría gas que se escaparía.
- Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g, 600 capas hasta 100 r_s): t_c = 46.38 Gyr sin la fuerza, con la pared fija en 100 r_s como en #61, y 45.05 Gyr con ella y la pared en su R₀ (-2.88%); fórmula de Essig, 44.88 Gyr.

**Lectura**: sin letra (E8). La fuerza de la ECV acorta el colapso, y el cambio no depende de la resolución. La comprobación de resolución del documento (duplicar las capas mueve t_c menos de 10⁻³) no pasa con su espaciado: la discretización de t_c es del 0.6 % con 150 capas. Pasa con el de referencia.
