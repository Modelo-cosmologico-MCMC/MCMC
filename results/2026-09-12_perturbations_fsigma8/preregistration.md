# Preinscripción — predicción perturbativa fσ8 (out-of-sample)

- **Commit**: `6d5f93d50be61858443480e01cbdc092d6a98c19`
- **Fecha (UTC)**: 2026-09-12T21:18:16.735682+00:00
- **Cadenas de fondo (CC+BAO+SNe, sin crecimiento)**: `chains_mcmc.npz` sha256 `e6cbd61b38e4…`
- **σ8 externa (solo contraste absoluto)**: 0.805
- **Banda**: z ∈ [0, 2] (41 puntos), 1000 draws, semilla 20260912

**Desenlace esperado (A, y NO fracaso)**: fσ8^MCMC ≈ fσ8^ΛCDM dentro de la banda del prior de ε_Λ — el control de consistencia del sector lineal. Regla: max_z |R_p50 − 1| ≤ envolvente(ε = ±0.05) y |Δχ²_oos| ≤ 2.0.

**Desenlace contrario (B)**: desviación mayor — error de implementación o física nueva, a discriminar por el procedimiento enumerado (identidad ε = 0; crosscheck independiente del integrador; solo entonces publicación como propiedad del modelo).

**Orden epistemológico declarado, no ejecutado**: ε_c/Atlas → µ, η → fσ8, C_L^φφ, P(k) → datos. **Prohibición**: nunca elegir µ, η desde los datos de lensing.
