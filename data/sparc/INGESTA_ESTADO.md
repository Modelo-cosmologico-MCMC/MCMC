# ESTADO DE INGESTA: DATA_UNAVAILABLE (actualizado 27-oct-2026)

ACTUALIZACIÓN 27-oct-2026 (rama mcv/pr-s-sparc-instrument, PR-S): rutas
alineadas con `data/sources.yaml`, sin bytes. `python scripts/download_data.py
sparc` baja (o verifica, si el autor los deja a mano) los CINCO ficheros del
registro (`SPARC_Lelli2016c.mrt`, `MassModels_Lelli2016c.mrt`,
`Rotmod_LTG.zip`, `RAR.mrt`, `RARbins.mrt`) en `data/raw/sparc/` y escribe
`data/manifests/sparc.json` en SCHEMA_UNVERIFIED (antes: dos ficheros en
`data/sparc/` y un `manifest.json` legado que nadie consumía por el registro).
`dynamics/sparc_data.py` solo da el catálogo por
`mcmc_ontology.data_registry.require_available("sparc")`: sin manifest,
DATA_UNAVAILABLE; con el manifest sin esquema confirmado, SCHEMA_UNVERIFIED;
en los dos casos falla cerrado. La primera acción con bytes es
`python -m dynamics.sparc_data --inspect`, que escribe
`data/manifests/sparc.schema_report.md` (descripción byte a byte literal de
cada MRT, cabeceras de Rotmod_LTG y papeles asignables o el motivo por el que
no lo son); el autor lo confirma y el manifest pasa a AVAILABLE. Los bytes
siguen siendo tarea del autor (descarga en el navegador y sha256).


ACTUALIZACIÓN 18-ago-2026 (rama infra/observational-data-registry):
nueva sonda tras el merge del PR #10 — `astroweb.cwru.edu`,
`cdsarc.cds.unistra.fr` y `data.desi.lbl.gov` SIGUEN denegados
(HTTP 000/CONNECT 403 vía proxy). En cambio, la vía git de GitHub SÍ
está abierta: DESI DR2 BAO (CobayaSampler/bao_data), Union3/3.1 y
DES-Dovekie quedaron ingeridos con commit fijado y manifest — ver
`data/sources.yaml` y `python scripts/data_registry.py status`.
El 5E observacional sigue ABIERTO: depende exactamente de estos hosts.


Reintento de la fuente primaria efectuado el 17-ago-2026 (~20:54 UTC)
desde la sesión de trabajo del Frente 5E, DESPUÉS de congelar la
preinscripción (`results/2026-08-16_front5e_sparc/preregistration.json`,
commit `ffdc77c9`):

- `astroweb.cwru.edu` (SPARC) → denegado en AMBOS esquemas: el
  intento `http://` del 31-jul-2026 devolvió el aviso de lista blanca
  del proxy («Host not in allowlist») y el `https://` del 17-ago-2026
  devolvió CONNECT 403 («policy denial» registrado por el gateway).
  El registro de descargas usa la URL `http://` canónica del sitio;
  el bloqueo es por HOST, no por esquema.
- `https://vizier.cds.unistra.fr/` (Walker et al. 2009,
  J/AJ/137/3100) → CONNECT 403 (ídem).

Reglas en vigor (preinscripción, apartado `if_data_unavailable`):

- los checksums de `scripts/download_data.py` permanecen ABIERTOS
  (None) — se fijan solo tras la primera descarga real;
- NO se fabrican fixtures observacionales;
- NO se sustituye la fuente por un mirror salvo prueba byte a byte o
  provenance verificable de identidad con el dataset primario;
- NO se emite veredicto SPARC observacional: el 5E observacional
  queda ABIERTO; el resultado estructural 5C permanece con su
  alcance declarado.

Desbloqueo: añadir `astroweb.cwru.edu` y `cdsarc.cds.unistra.fr` a la
configuración de red del entorno (o entregar los bytes a mano en el
raw_dir: el script no descarga lo que ya está, lo verifica y lo
registra); después `python scripts/download_data.py sparc` genera
`manifest.json` con sha256, tamaño, fecha y cita, y `python
scripts/download_data.py walker2009` (rutas alineadas con
`data/sources.yaml` el 23-oct-2026: ReadMe y table2–5.dat de CDS en
`data/raw/walker2009`) escribe `data/manifests/walker2009.json` en
SCHEMA_UNVERIFIED. La primera acción del análisis es el
`schema_report.md` (inspección del fichero real ANTES de asumir
columnas: `python -m dynamics.sparc_data --inspect`; para Walker,
`python scripts/run_sculptor_profile.py schema`).
