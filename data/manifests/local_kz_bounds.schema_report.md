# Esquema — `local_kz_bounds` (densidad dinámica local y K_z de la vecindad solar)

Fichero: `data/raw/local_kz_bounds/local_kz_bounds_2026.json` (JSON; `values`
con cinco entradas, cada una con `value`, `err`, `unit`, `meaning`, `source`
y `transcribed_from`; `derived_room` describe los dos márgenes que el
analizador construye).

## AVISO DE PROCEDENCIA

Cinco valores publicados **transcritos a mano** desde la literatura anotada
en la sesión de código (21-sep-2026). El entorno no alcanza arXiv, ADS, DOI
ni revistas (egreso denegado): ningún byte oficial se ha verificado; el
fichero lo declara (`official_bytes_verified: false`) y todo artefacto que
lo consuma lleva el aviso. Verificación: tarea del autor con navegador.

| clave | valor | unidad | fuente |
|---|---|---|---|
| `rho_dyn_0_HF2000` | 0.102 ± 0.010 | M☉/pc³ | Holmberg & Flynn 2000 |
| `rho_dyn_0_MPH2015` | 0.097 ± 0.013 | M☉/pc³ | McKee, Parravano & Hollenbach 2015 |
| `rho_bar_0_MPH2015` | 0.084 ± 0.012 | M☉/pc³ | McKee, Parravano & Hollenbach 2015 |
| `Sigma_1p1_BT2012` | 68 ± 4 | M☉/pc² | Bovy & Tremaine 2012 |
| `Sigma_bar_1p1_MPH2015` | 47.1 ± 3.4 | M☉/pc² | McKee, Parravano & Hollenbach 2015 |

Uso preinscrito: el término de Cronos entra como densidad dinámica
EFECTIVA (Δρ_eff(0) = (1/4πG)·dg_C/dz|₀; ΔΣ_eff(1.1) = g_C(1.1 kpc)/(2πG))
y se compara con el MARGEN ρ_dyn − ρ_bar y Σ_dyn − Σ_bar (errores en
cuadratura), concediéndole todo el margen (cota laxa). Ningún valor se
ajusta.
