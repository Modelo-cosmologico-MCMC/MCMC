# SPARC (curvas de rotación)

Ficheros esperados (vía `python scripts/download_data.py sparc`, o entregados a mano en el raw_dir), según
`data/sources.yaml`, en `data/raw/sparc/`:

- `SPARC_Lelli2016c.mrt` — tabla maestra (distancia, inclinación, R_d, brillo central del disco, calidad Q).
- `MassModels_Lelli2016c.mrt` — modelos de masa de todas las galaxias en una tabla.
- `Rotmod_LTG.zip` — 175 galaxias: r, v_obs, σ_v y componentes bariónicas (Υ = 1).
- `RAR.mrt` y `RARbins.mrt` — la relación de aceleración radial de McGaugh, Lelli y Schombert (2016).

Manifest: `data/manifests/sparc.json` (SCHEMA_UNVERIFIED hasta el schema_report,
`data/manifests/sparc.schema_report.md`). Estado de la ingesta: `INGESTA_ESTADO.md`.

Fuente: http://astroweb.cwru.edu/SPARC/ (uso académico).
Citar: Lelli, McGaugh & Schombert 2016 (AJ 152, 157).
