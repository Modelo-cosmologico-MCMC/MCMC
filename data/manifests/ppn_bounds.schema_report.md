# Esquema — `ppn_bounds` (cotas PPN de marco preferido para el sector Atlas)

Fichero: `data/raw/ppn_bounds/ppn_bounds_2026.json` (JSON; `values` con
cuatro entradas, cada una con `bound_abs`, `confidence`, `source`, `regime`
y `transcribed_from`).

## AVISO DE PROCEDENCIA

Cuatro cotas publicadas **transcritas a mano** (nota PPN del autor del
15-sep-2026 y literatura anotada en la sesión). El entorno de código no
alcanza arXiv, INSPIRE, DOI ni revistas (egreso denegado), así que ningún
byte oficial se ha verificado; el fichero lo declara
(`official_bytes_verified: false`) y todo artefacto que lo consuma lleva
el aviso. Verificación: tarea del autor con navegador.

| clave | cota | régimen | fuente |
|---|---|---|---|
| `alpha1_llr` | \|α₁\| < 1e-4 | campo débil | Müller, Williams & Turyshev 2008 (LLR) |
| `alpha1_pulsars` | \|α̂₁\| < 4e-5 (95 %) | campo fuerte (con sombrero) | Shao & Wex 2012 |
| `alpha2_solar_spin` | \|α₂\| < 4e-7 | campo débil | Nordtvedt 1987 (giro solar) |
| `alpha2_pulsars` | \|α̂₂\| < 1.8e-4 (95 %) | campo fuerte | Shao et al. 2013 |

Uso preinscrito: la cota que gobierna es la de campo débil más
restrictiva (giro solar sobre α₂); las de campo fuerte se publican como
indicativas. Ningún valor se ajusta.
