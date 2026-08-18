# Frente 5E — estado: DATA_UNAVAILABLE

La preinscripción está congelada (`preregistration.{json,md}`, generada ANTES de intentar la ingesta): **A = 6.201589e-07 (M⊙/pc³)^(−3/2)** (Υ⋆ = 2; sensibilidad declarada 1.853e-06 / 3.186e-07).

La fuente primaria y VizieR respondieron CONNECT 403 desde el proxy del entorno (evidencia y desbloqueo en `data/sparc/INGESTA_ESTADO.md`). Por la regla de fallo preinscrita:

- el 5E observacional queda **ABIERTO por ausencia de datos ingeridos**;
- el resultado estructural 5C permanece con su alcance declarado y **no se eleva a veredicto observacional**;
- los checksums siguen abiertos; no se fabrican fixtures observacionales; no se usan mirrors sin prueba de identidad byte a byte.

El pipeline completo (composición en cuadratura, χ², diagnósticos B_inner/F_outer/ley ×5, agregados con bootstrap, fila per-galaxy con `A_fitted = false`) está construido como funciones puras que reciben la config inmutable, y testeado con fixtures sintéticos — la ingesta real es un comando y el análisis observacional otro.
