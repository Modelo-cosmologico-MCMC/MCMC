# Preinscripción — µ(k,a), η(k,a), Σ(k,a) desde la ontología (canal Cronos)

- **Commit**: `ee631baa23a6b311bb14603a55e09cd47ac2829d`
- **Fecha (UTC)**: 2026-09-13T12:51:42.718496+00:00
- **α₀⁻¹**: 1e-06 (cota 11.5 saturada — límite superior, no medida); **ρ_c/ρ̄**: 200.0 (cierre 2, ambas lecturas); **ventana**: k ≤ 0.2 h/Mpc (cierre 3)
- **Fondo de referencia**: H0 = 67.87, Ω_m = 0.3263, ε = +0.0172, z_trans = 9.09 (cadenas `e6cbd61b38e4…`)

**E1 (aburrido, control)**: max |R_µ − 1| ≤ 0.001 en k ≤ 0.2, z ≤ 2.0, cierre comoving.
**E2 (firma)**: Σ−1 = (µ−1)/2 exacto; η−1 = −(µ−1) a segundo orden; k²; lineal en α₀⁻¹; physical/comoving = (1+z)^{9/2}; GR exacto en α₀⁻¹ = 0.
**E3 (cadena)**: declarada, no ejecutada (α₀⁻¹ medido exige 5E).

**Tabla de la nota**: estimaciones a mano, a verificar por la implementación; ninguna cifra se cita hasta entonces.

**Atlas**: PENDIENTE (frente 3): los coeficientes O(1) de µ_Atlas y η_Atlas no están derivados — sin coeficientes explícitos la contribución es cero y no se cita.

**Prohibición**: µ, η nunca se eligen desde los datos de lensing; este artefacto no carga datos.
