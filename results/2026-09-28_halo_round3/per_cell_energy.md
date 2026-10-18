# Ronda 3 del halo — |ΔE/E| por celda frente a la cualificación (sin letra)

Tabla anunciada en `declaracion_previa_segunda_ejecucion.md` antes de la segunda ejecución. **No es una puerta ni cambia el desenlace**: la letra la da `halo_round3.json` con la regla congelada (puertas por clase de brazo). Criterio de la revisión del autor del 30-sep: misma celda (clase, N, ds, ε_soft, rango actualizado) y mismo instante de evaluación (Cronos: t_weak; newtoniano: t_end); suelo = el mayor |ΔE/E| de las corridas de la celda que llegaron a ese instante. Preinscripción `6fba6b53d929`; cualificación `64b9693e6d9a`; corridas ausentes: ninguna.

| brazo | clase | N | ds | ε_soft | evaluación | \|ΔE/E\| del brazo | llegó | puerta congelada (clase) | celda en la cualificación | suelo de la celda | cociente |
|---|---|---|---|---|---|---|---|---|---|---|---|
| newton_N1e6 | newton | 1000000 | 0.04 | 0.1 | t_end | 1.97e-07 | sí | 6.75e-06 | no llegó al instante de evaluación | — | — |
| sweep_AS_eps0.2 | AS | 1000000 | 0.04 | 0.2 | t_weak | 2.37e-02 | sí | 8.88e-02 | sin celda equivalente | — | — |
| sweep_AS_eps0.1 | AS | 1000000 | 0.04 | 0.1 | t_weak | 4.03e-03 | sí | 8.88e-02 | no llegó al instante de evaluación | — | — |
| sweep_AS_eps0.05 | AS | 1000000 | 0.04 | 0.05 | t_weak | 3.98e-04 | sí | 8.88e-02 | sin celda equivalente | — | — |
| sweep_AS_eps0.025 | AS | 1000000 | 0.04 | 0.025 | t_weak | 7.65e-04 | sí | 8.88e-02 | sin celda equivalente | — | — |
| sweep_b005_eps0.2 | b005 | 1000000 | 0.04 | 0.2 | t_weak | 1.41e-03 | no | 8.34e-04 | sin celda equivalente | — | — |
| sweep_b005_eps0.1 | b005 | 1000000 | 0.04 | 0.1 | t_weak | 2.87e-03 | no | 8.34e-04 | no llegó al instante de evaluación | — | — |
| sweep_b005_eps0.05 | b005 | 1000000 | 0.04 | 0.05 | t_weak | 1.32e-02 | sí | 8.34e-04 | sin celda equivalente | — | — |
| sweep_b005_eps0.025 | b005 | 1000000 | 0.04 | 0.025 | t_weak | 2.27e-03 | sí | 8.34e-04 | sin celda equivalente | — | — |
| series_AS_ds0.02 | AS | 1000000 | 0.02 | 0.1 | t_weak | 9.31e-04 | sí | 8.88e-02 | sin celda equivalente | — | — |
| series_AS_ds0.08 | AS | 1000000 | 0.08 | 0.1 | t_weak | 3.14e-03 | sí | 8.88e-02 | sin celda equivalente | — | — |
| sat_sub_ds0.02 | sat | 1000000 | 0.02 | 0.1 | t_weak | 3.37e-05 | no | 8.88e-02 | sin celda equivalente | — | — |
| sat_sub_ds0.04 | sat | 1000000 | 0.04 | 0.1 | t_weak | 8.65e-06 | no | 8.88e-02 | sin celda equivalente | — | — |
| sat_sub_ds0.08 | sat | 1000000 | 0.08 | 0.1 | t_weak | 1.31e-06 | no | 8.88e-02 | sin celda equivalente | — | — |
| sat_above_ds0.04 | sat | 1000000 | 0.04 | 0.1 | t_weak | 2.74e-05 | no | 8.88e-02 | sin celda equivalente | — | — |
| ctrlN_AS_N1e5 | AS | 100000 | 0.04 | 0.1 | t_weak | 1.01e-02 | sí | 8.88e-02 | cualificada | 4.31e-03 | 2.34 |
| ctrlN_b005_N1e5 | b005 | 100000 | 0.04 | 0.1 | t_weak | 7.93e-02 | sí | 8.34e-04 | no llegó al instante de evaluación | — | — |

Recuento: cualificada: 1; no llegó al instante de evaluación: 4; sin celda equivalente: 12.

Una celda «no llegó al instante de evaluación» tiene corridas en la cualificación, pero ninguna salió del régimen débil (Cronos) o llegó a t_end (newtoniano) dentro de su pared; sus |ΔE/E| están en `per_cell_energy.json` y no son un suelo en t_weak.
