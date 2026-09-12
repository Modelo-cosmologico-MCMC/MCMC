# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## Sin publicar — septiembre de 2026

- **Registro canónico de claims (12-sep): la trazabilidad se GENERA,
  no se duplica**. `docs/claims_registry.yaml` pasa a ser la fuente
  única (39 claims: entidad, ley, derivación, implementación, test,
  observable, dataset, falsador, estatuto v35.1, categoría,
  artefacto); `scripts/make_traceability.py` genera la matriz
  clásica y la vista entidad → ley → observable → evidencia para la
  v36, y el test de no-divergencia hace imposible que un cambio de
  estatuto (#13, #15, 5E) deje dos documentos contradictorios. Las
  filas negativas llevan TRES categorías diferenciadas y fijadas por
  test — resultado-negativo (6A; #13 desenlace A; ajustes v1/v2) ≠
  claim-no-derivado (λ = 10, depende de τ(S)) ≠
  experimento-no-ejecutado (5E, fallo cerrado) — y FRB queda
  excluido de forma ejecutable hasta tener dataset con DOI,
  observable consumido y relación con el parámetro bariónico.

## Sin publicar — agosto de 2026

- **Crosscheck JAX del fondo (19-ago, 6A prioridad 3): equivalencia
  bajo puertas predeclaradas, y un integrador de producción cazado y
  corregido**. Implementación INDEPENDIENTE en JAX (float64,
  Gauss-Legendre espectral; cero imports de las implementaciones
  NumPy — regla vigilada por test; solo fórmulas declaradas del
  Ap. A + constantes físicas explícitas) de E, H, D_H, D_M, D_V,
  D_L, μ, el mapa S↔z y el vector DESI + χ². Puertas predeclaradas:
  max |ΔX/X| < 1e-8, RMS < 1e-10, |Δχ²| < 1e-8. La PRIMERA ejecución
  (registro en crosscheck_pre_fix.json, con su nota de procedencia:
  se generó con el runner aún sin commitear — declarado) midió el
  integrador DESI de producción (trapecio + interp lineal) en 4.9e-8
  — sobre la puerta — con efecto ≤ 2.8e-6 en los χ² de los argmin
  publicados (veredicto 6A robusto); en vez de relajar la puerta se
  corrigió el integrador a O(h⁴) — verificación contra scipy.quad
  REGISTRADA en el artefacto (integrator_vs_quad: 5.8e-15 en DM,
  fiducial y rincón, con candado < 1e-13) — y se regeneraron los
  artefactos DESI: números publicados idénticos a 3 decimales.
  Resultado final: bloques E/H/D_H y S↔z a precisión de máquina
  (4e-16), vector DESI a 1.2e-14, |Δχ²| ≤ 2.2e-12 en los argmin
  PUBLICADOS (≤ 6.9e-11 incluyendo el rincón de estrés, que no es un
  argmin publicado) — TODAS las puertas PASS. El integrador SNe de
  producción (trapecio 2048 puntos, rondas v1/v2 ya publicadas) se
  midió en 3.0e-6 relativo (≤ 6.6e-6 mag en μ — despreciable frente
  a σ_μ ~ 0.1): publicado como medición sin puerta, sin tocar una
  ronda fusionada (results/2026-08-19_jax_crosscheck/). La ronda pasó
  su revisión adversarial (15 confirmados + 1 parcial, 0 refutados,
  todos aplicados: guard del borde z = 0, test del integrador real,
  independencia por AST, procedencia del registro pre-fix, cotas «≤»
  redondeadas hacia arriba, atribución del |Δχ²| corregida).
  Estatuto: validación interna de implementación (E8) — equivalencia
  numérica entre dos implementaciones, no validación física.

- **Frente 2 (19-ago): las β de Fokker-Planck derivadas — desenlace A
  en el punto preinscrito, y la revisión adversarial destapa la
  cascada en el régimen físico (adenda δ0)**. La reducción canónica
  de la jerarquía de la Def. 4.4 (Polchinski d = 0 con cierres
  DECLARADOS: a = b = ½, truncamiento cúbico con sistemático
  cuártico, sector η O(δ0⁶), diccionario τ no derivable del corpus
  recogido) produce β_M0² = −8aB − 2bM0⁴, β_B = −24aC0 − 8bM0²B,
  β_C0 = −6b(B²+2M0²C0), validadas de punta a punta (jacobiano
  numérico, suavizado gaussiano exacto, FP exacta vía Hopf-Cole; el
  control de deriva prometido, publicado — sustitución
  Langevin→exacto declarada). Los CUATRO desenlaces se
  preinscribieron antes del primer autovalor; en el punto preinscrito
  salió A: espectro de M = ∂β/∂λ enteramente real (silla, sin
  cascada) con dD/dt = +28 > 0 (el signo cambia en g = 2.75 —
  negativo en 6/21 puntos —, el par complejo solo existe para
  g > g* = 8.19, y la conjunción rotación + hundimiento de D
  (Obs. 8.6) vive solo en ese borde; en g = 10, τ* = 0.233 publicado
  como dato). **La revisión adversarial (14 hallazgos confirmados, 0
  refutados) encontró el HIGH: el punto preinscrito escondía
  δ0 ≈ 1** — por la covarianza exacta β(D_s·λ) = σ·D_s·β(λ; b·k), el
  eje g del barrido ES el eje δ0 del escalado (3.2), g_ef = δ0⁻³ — y
  en el régimen perturbativo del corpus (δ0 < δ0* = 0.496) la misma
  reducción canónica SÍ rota y hunde D a la vez. Punto físico
  δ0 = 0.1: s0(τ=1) = 4.946, τ* = 0.276; contrafactual con las
  reglas preinscritas: **B** (declarado como contrafactual — el
  desenlace A preinscrito no se reclasifica; el defecto de diseño de
  la preinscripción se corrige por adenda, no reescribiéndola). El
  mecanismo de la cascada DSI tiene por primera vez una realización
  DERIVADA en el régimen del corpus; **λ = 10 sigue siendo
  calibración** (s0 depende del diccionario τ, no derivable aún).
  Todo con candados de preinscripción, no-ajuste y recomputación
  (results/2026-08-19_front2_fp_beta/).

- **Frente 6A (18/19-ago): DESI DR2 BAO — likelihood validada,
  benchmark superado a 0.04σ y contraste uniforme pro-ΛCDM**. El
  contrato «datos ≠ likelihood ≠ modelo» cruzado con artefactos:
  guard runtime require_available (manifest + estado + esquema +
  sha256 en cada carga científica); equivalencia de χ² contra Cobaya
  3.6.2 (max |Δχ²| = 1.85e-13, identidad de datos COMPUTADA por
  sha256: 16/16 ficheros idénticos a bao_data @ v2.6, el pin de
  Cobaya); benchmark ΛCDM propio vs oficial ANTES del MCMC — con el
  fondo normalizado, Ω_m ES la fracción de materia del ΛCDM plano:
  0.2971 vs 0.2975 ± 0.0086 = 0.04σ, puerta predeclarada PASS; y el
  contraste con r_d como calibración COMÚN (el MCMC no deriva la
  física pre-recombinación): en DESI_ALL y los 7 leave-one-bin-out,
  Δχ² ∈ [−0.33, −0.06] y ΔBIC ∈ [+4.47, +4.90] pro-ΛCDM — veredicto
  UNIFORME, χ²_M como mínimo VERDADERO sobre el cierre del soporte
  del prior (argmin en la frontera ε = −0.05, z_trans = 1 en las 8,
  publicado), ningún bin decisivo en ninguno de los dos modelos
  (incluido lrg-z1 ≡ LRG2), ε_Λ dominado por el prior TRUNCADO
  (cociente de anchuras ≥ 0.99). Razón estructural computada en el
  artefacto: con z_trans = 8.9 y la transición normalizada hoy,
  ε = 0.05 altera E(z ≤ 2.33) en ≤ 4.3e-7 relativo sobre TODO el
  rango, y el residuo χ² tras reabsorber (Ω_m, H0·rd) es 7.6e-11 —
  BAO DR2 no es el test sensible de ε_Λ en esta parametrización; lo
  que castiga es la parsimonia. Mismo veredicto que la repetición
  v1/v2 con el fondo corregido. Candados que ATAN artefacto a código
  (χ² recomputado offline en cada argmin publicado, ancla oficial
  literal, estructural y prior truncado recomputados)
  (results/2026-08-18_desi_dr2_background/).

- **Revisión adversarial del frente 6A aplicada (19-ago)**: seis
  lentes + un verificador independiente por hallazgo — 36
  confirmados, 1 parcial, 0 refutados; el veredicto pro-ΛCDM
  sobrevivió a todos, los números publicados no. Corregido: (1) la
  ronda afirmaba correr sobre el fondo normalizado sin contenerlo —
  la corrección (a29e89f) vivía en una rama sin fusionar; fusionada
  aquí con su suite de invariantes (H(0) = H0 vigilado); (2) los
  χ²_M publicados no eran mínimos del modelo (el refinado sin cotas
  escapaba del soporte del prior) — ahora multistart acotado con
  argmin y frontera publicados; (3) «semilla 42» no reproducía las
  cadenas (emcee 3 usa el estado global de numpy, sin sembrar) —
  ahora cadenas bit-idénticas y test de determinismo; (4) la «razón
  estructural» publicaba el valor de UN punto (z = 2.33) como cota
  del rango, y CHANGELOG/nota citaban un «≤ 3e-5» que no salía de
  ningún cómputo — ahora máximo sobre la malla + residuo tras
  reabsorber, en el artefacto y con candado; (5) la identidad de
  datos y el «logpdf = 0 exacto» eran cadenas fijas — ahora
  computados; (6) el packages path de Cobaya era una ruta efímera de
  sesión — ahora COBAYA_PACKAGES_PATH + script de recreación
  documentado y ejecutable.


- **Frente 5E preinscrito y bloqueado por datos (16/17-ago)**. La
  falsación cruzada Sculptor ↔ SPARC queda CONGELADA antes de tocar
  SPARC: preinscripción computacional (A = 6.201589e-7 (M⊙/pc³)^(-3/2)
  calculada por la función canónica en el commit del merge del PR #9,
  con hipótesis, estadísticos, cortes, subconjuntos y alcance del
  veredicto precomprometidos), configuración inmutable
  (dataclass frozen) que el análisis recibe y no puede estimar,
  candados ejecutables contra tuning (sin fit de A, sin A por
  galaxia, sin selección por Δχ²), pipeline completo de funciones
  puras (cuadratura, χ², B_inner/F_outer/ley ×5, bootstrap
  congelado, A_fitted = False vigilado) testeado con fixtures
  sintéticos sin red, e ingesta reproducible con manifest de
  procedencia y checksum abierto. El reintento de las fuentes
  (astroweb.cwru.edu, VizieR) devolvió CONNECT 403: **el 5E
  observacional queda ABIERTO por ausencia de datos ingeridos**
  (regla de fallo preinscrita — sin fixtures, sin mirrors, sin
  veredicto observacional; el 5C estructural no se eleva).

- **Frente 5, paso 5C estructural + objetivo ρ_id (14-ago)**. La
  mitad del 5C que no necesita datos: en discos exponenciales el
  término de Cronos tiene forma fija v_cronos² ∝ x·e^(−3x/2) (pico en
  x = 2/3), así que aportar V km/s donde la discrepancia de masa vive
  (x ≳ 3) cuesta ×5·V en x = 2/3, PARA CUALQUIER amplitud A — un
  bulto interior incompatible con el carácter bariónico de los discos
  internos densos que la literatura reporta (hipótesis declarada, sin
  ingesta; la confrontación por galaxia es el pendiente). Con la A
  concreta que Sculptor exige, la malla declarada (Σ0 ∈ [50, 800]
  M⊙/pc² × R_d × ζ; los LSB reales bajan más — allí el término se
  hace despreciable y no explica nada) da v_cronos/v_bar hasta ×6.3
  en el pico y 2-82 km/s en x = 4: bultos donde el disco es denso,
  nada donde no lo es. Veredicto estructural 5E (parcial): el término
  débil no puede ser el mecanismo galáctico común; la carga queda en
  ρ_id, y Sculptor la fija como CURVA de degeneración ρ0(r_c)
  medible (M_id(<347 pc) = 1.78e7 M⊙; en el r_c = 0.30 kpc del
  corpus: ρ0 = 0.176 M⊙/pc³). Instrumento de perfiles binados listo
  con autoprueba sintética (la ingesta de Walker+09 y del catálogo
  SPARC sigue bloqueada por el proxy — pendiente declarado con el
  hueco de checksum abierto).

- **Frente 5, medio paso 2 (10-ago): Sculptor contra el potencial
  débil (5A/5B/5D)**. Una sola Φ_eff = Φ_N − c²·ε_c desde la Ley de
  Cronos (Def. 11.1 + Prop. 11.2), Jeans esférico con β constante y
  proyección validados contra la identidad analítica del Plummer
  (error < 2e-3). El resultado, publicado tal cual bajo las DOS
  lecturas declaradas de la ec. (11.5) (literal: solo α₀⁻¹, ρ_c
  libre; subdominancia: c²ε_c ≲ |Φ_N|, de la que la cota nace): solo
  estrellas dan σ_N = 2.1-3.6 km/s; bajo subdominancia el término
  débil de Cronos queda falsado como explicación de Sculptor
  (techo √2·σ_N ≤ 5.1 frente a σ_obs ≈ 9-10); bajo la literal no hay
  violación numérica, pero el término solo alcanza σ_obs como
  potencial DOMINANTE (c²ε_c = ×8-×26·|Φ_N|), el régimen que la
  justificación de la cota excluye. Sculptor no mide ρ_c: mide
  A = α₀⁻¹/ρ_c^(3/2) (con α₀⁻¹ en su cota, ρ_c ≲ 1-2 M⊙/pc³,
  desigualdad unilateral — 5E sigue abierto). La carga explicativa
  queda cuantificada en ρ_id (M_1/2 ≈ 2e7 M⊙ dentro de r_1/2 ≈ 347
  pc), sin perfil derivado a escala dSph — hueco declarado. Datos
  globales de Walker et al. 2009 con procedencia y pendientes
  declarados (`dynamics/dsph_data.py`).

- **Corrección de normalización del fondo (10-ago)**: la forma legacy
  de Λ_rel anclaba Λ0 en el punto medio de la transición (sin dividir
  por F(0)) y congelaba Ω_Λ0 con Ω_m = 0.300 a nivel de módulo —
  H(0) ≈ 1.0042·H0 en el punto fiducial (validate_all imprimía 70.09
  con H0 = 69.8) y clausura no plana al variar Ω_m; en las corridas
  legacy el sesgo de F(0) era SOLO del brazo MCMC (ε hacía doble
  papel). Hoy: transición normalizada hoy (Ω_Λ_rel = Ω_Λ0·F(z)/F(0)) y
  clausura plana por llamada ⟹ H(0) = H0 exacto para todo parámetro
  admisible (dz > 0 declarado).
- **Nivel 3 del CI — invariantes físicos**: suite
  `tests/test_physical_invariants.py` sobre mallas de (Ω_m, ε, z_trans,
  dz) que cubren los priors de los ajustes (ε negativos incluidos), y
  `validate_all` ahora FALLA si H(0) ≠ H0 (el canal que estuvo verde
  durante el bug).
- **Repetición de los ajustes v1 y v2 con el fondo corregido**
  (mismas semillas, datos y configuración): el veredicto diferencial
  se confirma — ΔAIC = +4.00/+4.08, ΔBIC = +14.50/+14.60 pro-ΛCDM,
  ε compatible con 0 — con posteriores absolutos ya sin sesgo
  (`results/2026-08-10_production_fit{,_v2}/`); las corridas
  originales quedan etiquetadas `legacy_pre_normalization` con la
  asimetría del sesgo declarada.

## v0.2.0 — 2 de agosto de 2026 (rondas 2–5)

- **Cadena deductiva ejecutable** (`core/`, caps. 2–10): cada módulo
  implementa las definiciones de su capítulo y sus tests verifican
  identidades, límites y consecuencias numéricas de los teoremas y
  ansätze implementados (comprobación interna, no demostración física —
  v35.1, E8); condicionales expuestos, nunca resueltos en silencio.
  Suite espejo del apéndice H con controles negativos (`validation/`).
- **Ajustes de producción v1 y v2** sobre datos públicos con checksums:
  ΔBIC = +14.5 a favor de ΛCDM en ambos, publicado en portada;
  ε = 0.015 −0.039/+0.043 (compatible con 0); la ventaja del corpus no
  reaparece (la reconciliación respondida).
- **Frente 6 (Residuos), consistencia superada**: −1.8 % dentro de la
  cota BBN (Alvey et al. 2020), a 0.32σ — sin significancia de signo
  (v35.1, E13); refutable (vivo).
- **Frente 7 ejecutado**: empalme C¹ no circular (δ_H ≈ 0.0581 medido
  sin m_H como entrada) + WKB ab initio (una sola κ cubre 13 órdenes).
- **El círculo de δ₀ (ronda 5)**: el atractor de Victoria es el Techo;
  ecuación de cierre W_max = c̄·δ_H³ ≈ 1.65×10⁻⁴ — los ingredientes
  están en el tratado (Lema 10.3 + H.8); la aportación del programa es
  la conexión δ_sat = δ_H y que γR desaparece del punto fijo;
  naturalidad de δ_H (mediana 0.054) y cota δ_H ≳ 0.033.
- **Regla canónica**: retirada la identificación δ₀ ≡ ε_Λ de todo el
  código (los cuatro épsilon del convenio C6 — v35.1, E9 — separados).
- **Medio paso del frente 5**: compuerta H.2.5 con mediana Γ = 0 exacta
  en el halo virializado aislado y control negativo; el veredicto del
  núcleo queda declarado como inaccesible a esta resolución.
- **Cronos v3** (cap. 11), lattice/quantum alineados con apéndices C-D,
  simulación PM mínima del apéndice B con par A/B reproducible.
- Notebook en inglés con salidas ejecutadas; orientación en inglés en el
  README; nota computacional I en `docs/`.
- **Frente E (flujo KLS)**: la ley del walking medida (el «≃» de la
  ec. 8.4 cuantificado; divergencia −1/2), el Cruce de Victoria como
  bifurcación dinámica y el retraso ∝ ritmo^(−1/3) (resultado del
  programa); λ = 10 sigue calibrado (frente 2).
- **Frente 2 instrumentado** (ec. 14.2): dos rutas coincidentes al
  exponente de Victoria (espectro ⟺ walking integrado, error ~1e-14);
  con ansatz O(1), la cascada DSI es genérica (~67 %) pero λ = 10 es
  una selección medible — las β de Fokker-Planck (Def. 4.4) son el
  hueco declarado.
- **Frente F (RP no estacionaria)**: la simetría especular del perfil
  es condición SUFICIENTE para la RP en el juguete (necesidad no
  demostrada) — el running monótono viola la reflexión ingenua y la
  reflexión modificada restaura la positividad exacta; el sector de
  Wilson sigue abierto (frente 1).
- **Matriz de trazabilidad ejecutable** (manuscrito ↔ código ↔ test ↔
  estatuto, con huecos declarados) verificada por test; guardias de
  lenguaje v35.1 en el CI (E4/E6/E8/E13); Lema de Precedencia
  reformulado K(S)=K(ϑS) como condición ejecutable y correspondencias
  E.18/E.19 propuestas para la v36.
- CI con guardias de honestidad y contrato de lint declarado (reglas en
  `pyproject.toml`, versión de ruff fijada).
- 189 tests en verde.

## v0.1.0 — 30 de julio de 2026 (migración v32 → v35)

- Repositorio alineado con el *Tratado de Fundamentos* (v35, DOI
  10.5281/zenodo.20765373): constantes por estatuto epistémico
  (apéndice F), ocho axiomas ejecutables (§1.2), Tabla F.1, transición
  tanh (A.3), Cronos v3 (cap. 11), citas v32 etiquetadas.
- Retiradas las sobreafirmaciones: eliminada la fórmula que negaba la
  existencia de parámetros libres (hoy es guardia de CI),
  auditoría de circularidad del Higgs (Obs. 12.2) recogida, valores del
  corpus separados de las salidas del código, suite declarada como
  regresión interna, test del límite de recuperación (Prop. A.1).
- Descarga de catálogos públicos con SHA-256; likelihoods SNe+BAO+CC;
  CITATION.cff.
