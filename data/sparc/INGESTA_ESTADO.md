# ESTADO DE INGESTA: DATA_UNAVAILABLE (17-ago-2026)

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

Desbloqueo: añadir `astroweb.cwru.edu` y `vizier.cds.unistra.fr` a la
configuración de red del entorno; después
`python scripts/download_data.py sparc && python
scripts/download_data.py walker09` genera `manifest.json` con sha256,
tamaño, fecha y cita — y la primera acción del análisis es el
`schema_report.md` (inspección del fichero real ANTES de asumir
columnas: `python -m dynamics.sparc_data --inspect`).
