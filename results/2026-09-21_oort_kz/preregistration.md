# Preinscripción — Oort–K_z: la vecindad solar acota la amplitud de la Ley de Cronos débil por su fuerza vertical

Congelada 2026-09-21T22:55:40.106211+00:00 en el commit `2b5089411` (que contiene el generador). sha256 de `preregistration.json`: `f35e90f77962141fbc00d98a30fe74f409d052574ce9958018e4c36085569749`.

## Erratum

Versión 2 por erratum de `preregistration_v1_2026-09-21.json` (sha256 `5d5a84d26d20e4890f51a8ef58e6741434185332bda053b900f0de6bee4c9366`, congelada 2026-09-21T14:11:19.799697+00:00): cita de Σ(<1.1 kpc) corregida (Bovy & Tremaine 2012 → Bovy & Rix 2013, ApJ 779, 115); ρ_bar(0) declarado como resta; valores verificados por el autor contra los resúmenes de arXiv; el manifest del dataset cambia por esos metadatos, ningún valor ni error cambia. Reglas, piloto y límites copiados verbatim de la v1; el analizador exige que los números coincidan con los de la v1.

## Declarado

- **law**: ε_c = A·ρ^{3/2}; g_C = −c²dε_c/dz (Cor. 11.3, Ley de Cronos débil)
- **effective_density**: Δρ_eff(0) = (1/4πG)·dg_C/dz|₀ = −(3/2)c²A√ρ(0)·ρ''(0)/(4πG); ΔΣ_eff(1.1) = g_C(1.1 kpc)/(2πG)
- **slab**: {'rho_star_0': 0.043, 'h_star_pc': 600.0, 'rho_gas_0': 0.041, 'h_gas_pc': 250.0, 'rho_dm': 0.013}
- **slab_sensitivity_band**: alturas de escala × 0.5 y × 2 (Δρ_eff ∝ h⁻²): banda publicada, no brazo
- **dataset**: local_kz_bounds (transcripción verificada por el autor contra los resúmenes de arXiv; bytes de las tablas NO descargados — aviso propagado)
- **dataset_manifest_sha256**: 39a362141dc1d3131e0de575d307cb867beac7001f608da09031483d8ffb5942
- **room**: ρ_dyn(0) − ρ_bar(0) con ρ_dyn = media ponderada HF2000 + MPH2015; Σ_dyn(1.1) − Σ_bar(1.1) con Bovy & Rix 2013 − MPH2015; errores en cuadratura; TODO el margen para Cronos
- **arms**: {'A_sculptor': 6.201588782789579e-07, 'A_0p05': 3.1007943913947895e-08, 'A_2sigma': 'derivada (lineal en A)'}

## Reglas (congeladas)

- **z_score**: z = (Δ_eff − margen)/σ_margen por observable; z_max = max(plano, columna)
- **compatible**: {'z_max_le': 2.0}
- **tension**: {'z_max_in': [2.0, 5.0]}
- **excluded**: {'z_max_gt': 5.0}
- **indeterminate**: dataset no AVAILABLE, margen ≤ 0 o sha del manifest distinto del congelado
- **publish**: ['A_2σ/A_Sculptor con banda de la losa', 'z del plano y de la columna por separado', 'el aviso de procedencia del dataset']
- **verdict**: letra por brazo con la losa declarada; ninguna letra es afirmación sobre el tratado: la Ley de Cronos débil con A_Sculptor es la hipótesis del 5E y este contraste la sitúa frente a la dinámica local (E8)

## Piloto declarado

ronda del criterio (21-sep mañana): límite de Oort efectivo de Cronos con A_Sculptor ≈ 0.75 M☉/pc³ frente a ρ_dyn(0) ≈ 0.10; cota A ≲ 0.05·A_S por g_C ≤ 0.1·K_z en 300 pc. La estructura de desenlaces se escribió conociéndolo; ningún umbral se ajustó.

## Lo que no puede decidir

- si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): tarea del diccionario)
- la amplitud A del frente 5b (0.05·A_S es brazo, no resultado)
- la losa real (declarada; la banda mide su peso)
