# Esquema — `bbn_abundances` (abundancias primordiales para el Contraste de los Residuos)

Fichero: `data/raw/bbn_abundances/abundances_2026.json` (JSON, un objeto
`values` con cinco entradas; cada una lleva `value`, `sigma`, `role`,
`source`, `doi` cuando existe y `transcribed_from`).

## AVISO DE PROCEDENCIA (leer antes de consumir)

Estos NO son bytes de un release: son **cinco números publicados
transcritos a mano**. El 14-sep-2026 el entorno de código no pudo
alcanzar arxiv.org, export.arxiv.org, inspirehep.net, api.crossref.org,
api.semanticscholar.org, doi.org, ui.adsabs.harvard.edu, iopscience ni
academic.oup.com (CONNECT denegado por la política de egreso), así que
ninguna transcripción ni DOI se pudo verificar contra la fuente. El
manifest lo declara (`official_bytes_verified: false` en el fichero;
transformación «bytes oficiales NO verificados» en el manifest) y todo
artefacto que consuma este dataset debe llevar el mismo aviso en su
estatuto. La verificación es tarea del autor con navegador: comprobar
los cinco valores y, si alguno difiere, corregir el fichero y re-ingerir
(el sha256 cambia; los artefactos anteriores quedan invalidados por el
candado).

| clave | valor | σ | rol | fuente | origen de la transcripción |
|---|---|---|---|---|---|
| `Y_P_empress_xv` | 0.2402 | 0.0040 | principal | EMPRESS XV, arXiv:2506.24050 (jun-2026) | mapa del autor (14-sep-2026), verificado por el autor |
| `Y_P_aver_2021` | 0.2453 | 0.0034 | control | Aver et al. 2021, JCAP 03 (2021) 027 | literatura (sesión de código) — **verificar** |
| `DH_cooke_2018` | 2.527e-5 | 0.030e-5 | principal | Cooke, Pettini & Steidel 2018, ApJ 855, 102 | literatura (sesión de código) — **verificar** |
| `omega_b_planck_2018` | 0.02237 | 0.00015 | prior | Planck 2018 VI, A&A 641, A6 (TT,TE,EE+lowE+lensing) | literatura (sesión de código) — **verificar** |
| `tau_n_pdg_2023` | 878.4 s | 0.5 s | prior | PDG 2023 (8 medidas), valor por defecto de PRyMordial | `PRyM_init.py` @ 725d8a8 (bytes presentes en el clone) |

Unidades: Y_P es fracción de masa de helio primordial (convención BBN);
D/H es el cociente de abundancias por número; ω_b ≡ Ω_b h²; τ_n en
segundos. EMPRESS XV publica además N_eff = 2.54 (+0.20/−0.25): es un
parámetro DERIVADO por la colaboración y **no entra** como dato (queda
anotado en `also_reported`).

Roles fijados: Y_P EMPRESS XV = dato principal; Y_P Aver+2021 = control
(muestra el desplazamiento del dato nuevo); D/H = dato principal; ω_b y
τ_n = priors gaussianos. Ningún valor se ajusta ni se combina fuera de
lo preinscrito en `results/2026-09-14_bbn_g/preregistration.json`.
