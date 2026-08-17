# Preinscripción del Frente 5E (antes de tocar SPARC)

- **Commit**: `ffdc77c90e1ee76937493c9844e11c62a4d24eb2`
- **Fecha (UTC)**: 2026-08-17T20:53:34.392825+00:00
- **A transferida (Υ⋆ = 2)**: 6.201589e-07 (M⊙/pc³)^(−3/2) — calculada por `dynamics.sculptor_transfer.sculptor_A_req`, no copiada
- **Sensibilidad declarada**: A(Υ⋆=1) = 1.853e-06, A(Υ⋆=3) = 3.186e-07
- **β (Sculptor)**: 0.0; **R_half**: 260.0 pc; **σ_obs**: 9.2 km/s
- **Convención SPARC congelada**: Υ_disk = 0.5, Υ_bul = 0.7 (3.6 μm, estándar Lelli+16); sensibilidad Υ_disk ∈ (0.3, 0.7)
- **ζ = h/R_d**: 0.15 (sensibilidad (0.1, 0.2))
- **Bootstrap**: n = 10000, semilla = 42

**Hipótesis primaria H_5E**: A_req^Sculptor es compatible con curvas de rotación SPARC reales: añadir Cronos(A congelada) a los bariones no empeora sistemáticamente el ajuste.

**Comparación primaria**: Por galaxia: M0 = bariones vs M1 = bariones + Cronos(A_req^Sculptor). v_model² = v_bar² + v_cronos² (composición en cuadratura — consecuencia de la dinámica: las aceleraciones se suman, v² = R·g). Cronos no añade ningún parámetro ajustado a SPARC.

**Regla central**: no se pregunta qué parámetros hacen funcionar a Cronos en SPARC; se pregunta si la predicción que Sculptor ya fijó sobrevive cuando SPARC no puede modificarla. A queda congelada; el candado es ejecutable (tests/test_front5e_lock.py). El detalle completo de estadísticos, cortes, subconjuntos y alcance del veredicto está en preregistration.json (mismo generador).
