# schema_report — des_dovekie_chains

Generado del fichero ingerido (nunca de memoria). Fuente:
`github.com/des-science/DES-SN5YR @ c9a4fcafc4cbd19bd750dee47fc76194a45c181f`,
ruta `5_COSMOLOGY/chains/flcdm/dovekie_lcdm_nautilus.txt` (cadena
nautilus, flat ΛCDM, **DES-SN only**). El README oficial de 5_COSMOLOGY
declara: M y H0 marginalizados analíticamente; «do not use this sample to
measure H0».

## des_dovekie_chains — `dovekie_lcdm_nautilus.txt`
- Cabecera (línea 1, `#`): `cosmological_parameters--omega_m
  cosmological_parameters--h0 cosmological_parameters--omega_b
  log_weight prior post`
- Metadatos: `#sampler=nautilus`, `#n_varied=3`
- Filas de datos: 16384 (16613 líneas − cabecera y metadatos)
- Pesos: columna `log_weight` (pesos de importancia en log); n_eff =
  1/Σw² = 11253 con w ∝ exp(log_weight)
- Soporte muestreado (prior): Ω_m ∈ [0.100, 0.500]; h0 ∈ [0.55, 0.91]
- **Ω_m oficial (ponderado)**: media 0.3306, σ 0.0154, mediana 0.3304,
  CI68 [0.3153, 0.3461]

## Uso en el programa
Benchmark del brazo ΛCDM propio SN-only (preinscripción
`results/2026-09-13_dovekie_real/`): el posterior de Ω_m de nuestro
pipeline (likelihood validada contra la oficial en #15) frente al oficial.
El soporte del prior de Ω_m del ajuste propio se fija IDÉNTICO al oficial
([0.10, 0.50]) para que la comparación sea like-for-like. Este chain NO
entra en ningún likelihood: es referencia, no dato.
