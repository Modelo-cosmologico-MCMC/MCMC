# Esquema — `local_kz_bounds` (densidad dinámica local y K_z de la vecindad solar)

Fichero: `data/raw/local_kz_bounds/local_kz_bounds_2026.json` (JSON; `values`
con cinco entradas, cada una con `value`, `err`, `unit`, `meaning`, `source`
y `transcribed_from` (y `derived` donde el valor es una resta declarada);
`verification` documenta la verificación del 22-sep; `derived_room` describe
los dos márgenes que el analizador construye).

## PROCEDENCIA Y VERIFICACIÓN

Cinco valores publicados **transcritos a mano** (sesión de código,
21-sep-2026) y **verificados por el autor** (sesión de verificación,
22-sep-2026) contra los resúmenes de arXiv de las fuentes primarias:
astro-ph/9812404, arXiv:1509.05334 y arXiv:1309.0809. Los bytes de las
tablas de los artículos **no se han descargado** (ADS y las revistas no son
alcanzables desde el entorno): `official_bytes_verified: false` se mantiene y
todo artefacto que consuma el fichero lo declara junto con la verificación
(`values_verified_against_arxiv_abstracts: true`).

**Erratum del 22-sep-2026**: la cifra Σ(<1.1 kpc) = 68 ± 4 M☉/pc² es de
**Bovy & Rix 2013** (ApJ 779, 115; arXiv:1309.0809), no de Bovy & Tremaine
2012 como se citó el 21-sep (ese artículo da ρ_DM = 0.008 ± 0.003 M☉/pc³ y no
cita 68 ± 4). La clave se renombra `Sigma_1p1_BT2012` → `Sigma_1p1_BR2013`;
**ningún valor ni error cambia**. ρ_bar(0) = 0.084 ± 0.012 queda declarado
como la resta total − materia oscura de MPH2015 (0.097 − 0.013), no como cifra
citada con ese error.

| clave | valor | unidad | fuente | verificación |
|---|---|---|---|---|
| `rho_dyn_0_HF2000` | 0.102 ± 0.010 | M☉/pc³ | Holmberg & Flynn 2000 | exacto (resumen astro-ph/9812404) |
| `rho_dyn_0_MPH2015` | 0.097 ± 0.013 | M☉/pc³ | McKee, Parravano & Hollenbach 2015 | exacto (resumen arXiv:1509.05334) |
| `rho_bar_0_MPH2015` | 0.084 ± 0.012 | M☉/pc³ | McKee, Parravano & Hollenbach 2015 | consistente, derivado por resta (declarado) |
| `Sigma_1p1_BR2013` | 68 ± 4 | M☉/pc² | Bovy & Rix 2013 | exacto (resumen arXiv:1309.0809); cita corregida |
| `Sigma_bar_1p1_MPH2015` | 47.1 ± 3.4 | M☉/pc² | McKee, Parravano & Hollenbach 2015 | exacto (resumen arXiv:1509.05334) |

Uso preinscrito: el término de Cronos entra como densidad dinámica
EFECTIVA (Δρ_eff(0) = (1/4πG)·dg_C/dz|₀; ΔΣ_eff(1.1) = g_C(1.1 kpc)/(2πG))
y se compara con el MARGEN ρ_dyn − ρ_bar y Σ_dyn − Σ_bar (errores en
cuadratura), concediéndole todo el margen (cota laxa). Ningún valor se
ajusta. La preinscripción v2 (erratum) del 22-sep congela las mismas
reglas que la v1 del 21-sep sobre el manifiesto nuevo; el analizador
comprueba que las reglas son idénticas y que los números coinciden con los
de la v1.
