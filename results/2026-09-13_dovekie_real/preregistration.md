# Preinscripción — primera aplicación del pipeline SN a Dovekie real

- **Commit**: `1b672b561d8ea043d59e52d53a65c83e27d265ee`
- **Fecha (UTC)**: 2026-09-13T19:10:13.777244+00:00
- **Precondición**: PASS de mocks `3ccdca260152…` (prereg #15 `d7cbabdc6d9f…`)
- **Datos**: Dovekie 1820 SNe (STAT+SYS principal; STATONLY robustez) + 31 CC + 6 BAO; n = 1857
- **Referencia oficial**: Ω_m = 0.3306 ± 0.0154 (chain nautilus flat-ΛCDM SN-only, ponderado)
- **Priors**: H0 ~ N(67.4, 5²) en (60, 80); Ω_m ~ U(0.1, 0.5) (= soporte oficial); ε ~ N(0.012, 0.05²) en (-0.05, 0.1); z_trans ~ U(1.0, 20.0); M_B marginalizada
- **Sampler**: {'nwalkers': 32, 'nsteps': 3000, 'seed': 42, 'burn': 'nsteps//2', 'thin': 4}

**Desenlaces (orden literal C → B → A → INDETERMINADO)**:
- C tensión: |ΔΩ_m| > 3.0σ_of o χ²_ν(ΛCDM) > 1.3.
- B preferencia: BIC_ΛCDM − BIC_MCMC > 2.0 y 0 ∉ CI95(ε) — sospecha de error primero.
- A aburrido (esperado): ΔBIC > 0 pro-ΛCDM, σ(ε) ≥ 0.04, 0 ∈ CI95(ε), |ΔΩ_m| ≤ 1σ_of. Redacción: «las SNe Dovekie no identifican ε_Λ; el fondo MCMC permanece consistente y no preferido».

**Compromiso**: ningún parámetro ni prior cambia tras la primera evaluación sobre datos reales; el resultado entra en el registry con su categoría exacta.
