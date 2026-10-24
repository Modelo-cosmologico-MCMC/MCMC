# ESTADO DE INGESTA: DATA_UNAVAILABLE (actualizado 18-ago-2026)

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
