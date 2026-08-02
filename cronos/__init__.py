"""Cronos — simulación N-body en espacio entrópico S.

Dos esquemas conviven en este paquete:

- ESQUEMA ANTERIOR (v32): integrator.py + kronos_kick.py. La v35
  (Obs. 11.4) lo declara superado: el corpus acopló la lapse a Φ_ten con
  signo positivo (primer error) y escribió el término de velocidad de la
  geodésica con signo negativo (segundo error) — dos errores compensados
  que producían la fenomenología correcta por la ruta incorrecta. Además
  el kick no llevaba la lapse, faltaba la fricción con compuerta Θ(ρ̇) y
  faltaba la fuerza +c²∇ε_c.

- CRONOS V3 (v35, cap. 11): cronos_v3.py — lapse N = 1 + Φ_N/c² − ε_c,
  fricción con compuerta que se apaga al virializar, fuerza +c²∇ε_c y
  cota dura α0⁻¹ ≲ 1e-6 (ec. 11.5). Es el FRENTE ABIERTO Nº 5 del
  tratado (§13.6/cap. 14.5): las simulaciones de producción con este
  esquema están pendientes; el módulo es trabajo en curso, no un
  resultado validado. Las validaciones del corpus (halo 10¹¹ M☉ →
  núcleo 2.3 kpc, SPARC RMSE 12% → 4.5%, subhalos −45%) están
  pendientes de reproducción con v3.

- APÉNDICE B (v35): infraestructura del ciclo de producción —
  timestep.py (paso entrópico B.3, η=0.025), poisson.py (Poisson
  modificado B.2, espectral), channels.py (refresco de mallas y
  dilatación local ζ, B.3), simulation.py (ciclo PM completo B.1-B.3
  con la compuerta del cap. 11) y config.py (cajas Local/Meso/LSS de
  B.4). NO es Gadget-4-Cronos: es la malla PM mínima que demuestra la
  firma falsable de la compuerta en colapsos pequeños reproducibles.
"""
