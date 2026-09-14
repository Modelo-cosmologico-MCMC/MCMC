# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## Sin publicar — septiembre de 2026

- **Contraste de los Residuos vía BBN (14-sep, frente 6): G_cosmo/G_N
  libre sobre Y_P de EMPRESS XV + D/H — DESENLACE A (banda compatible,
  sin identificación), PROVISIONAL hasta verificar la procedencia de
  los datos**. Preinscripción congelada ANTES de leer los valores
  (`ec3c8f48597c`, en un commit que contiene su generador). Respuesta
  de las abundancias DERIVADA con PRyMordial (código BBN público,
  clone @ 725d8a8): G_cosmo entra SOLO en la tasa de expansión (masa
  de Planck reescalada con η_b fijo); Y_P ∝ G^0.358, D/H ∝ G^0.977;
  sistemático nuclear NACRE II − PRIMAT en D/H = +0.059e-5, usado como
  σ_th. Predicción: δ_G = (2ξ−α_a)/(3λ_K−1) − 1 = −1.77 % (ε_K = 0.012,
  α_a = 0, ξ = 1). Datos: Y_P = 0.2402 ± 0.0040 (EMPRESS XV), D/H =
  (2.527 ± 0.030)e-5 (Cooke+2018), priors ω_b (Planck 2018) y τ_n
  (PDG) — CINCO VALORES PUBLICADOS TRANSCRITOS: arXiv, INSPIRE, DOI y
  revistas están denegados desde el entorno, los bytes oficiales no se
  han verificado, y el manifest, el schema_report y el artefacto lo
  declaran; la verificación es tarea del autor y hasta entonces el
  desenlace es provisional. Brazo principal (δ_G libre, N_eff = 3.044,
  ω_b y τ_n marginalizados de forma exacta): **δ_G = +0.003 ± 0.025,
  CI95 [−0.045, +0.053]** — el −1.8 % ∈ CI95 (a −0.8 sd del central) y
  0 ∈ CI95 → A. Brazo 0 (SM): pulls Y_P −1.67σ y D/H +1.26σ, χ² =
  4.0 con 2 datos. Brazo 4 (una sonda): solo Y_P da −0.072 ± 0.043 y
  solo D/H +0.036 ± 0.031 — tensión interna de 2.0σ (la conocida entre
  el helio bajo de EMPRESS y el deuterio con tasas PRIMAT): el Y_P bajo
  «apunta» al signo del modelo, pero el conjunto no favorece signo
  alguno y el signo del central no es señal (E13). Brazo 2
  (degeneración G ↔ N_eff, solo informa): cresta dΔN_eff/dδ_G = −6.5
  (la correspondencia por época, 6.1–7.4, reproducida), marginal
  δ_G = +0.11 ± 0.13. Control con Y_P de Aver+2021: +0.016 ± 0.024.
  Lo que cambia en el frente 6: el dato vigente pasa de la cota de
  Alvey+2020 (0.99 +0.06/−0.05) a una banda propia bajo preinscripción
  con la misma conclusión (compatible, sin detección); (9.5) y
  residues_test no cambian. Fronteras: ξ_e no modelada; N_eff casi
  degenerado con G; α_a = 0 en la predicción; transcripción pendiente
  de verificación. Estatuto: contraste interno bajo preinscripción
  (E8) sobre valores publicados transcritos — no demostración física
  (`results/2026-09-14_bbn_g/`).

- **Canal Atlas, cola O(e²) CON el sector de velocidades (13-sep, noche
  2, E3_Atlas): residuos del polo DERIVADOS (E3a = A); confirmación
  numérica E3b = C bajo la regla congelada — publicado sin reajuste**.
  Preinscripción congelada ANTES de ejecutar (`7fb31910a086`, en un
  commit que contiene su generador), con la predicción independiente
  del autor citada (17.07·e² en el punto del arnés). Escalera exacta
  del modo creciente, t^g·Σ c_n t^{−n/3}, sobre el sistema lineal
  completo (ξ = 1; sin truncación cuasi-estática): orden dominante
  µ₀ = 1/(1−α_a/2) y η₀ = 1 exactos en toda la malla; coeficiente de
  e² en (1.05, 0.3): η − 1 → 17.0735 (predicho 17.07), µ_loc − 1 →
  −11.361. Residuos del polo 1/(λ_K−1) por extrapolación cúbica en
  h = λ_K − 1: **P_η = 3α_a/(2−α_a)** (a 4e-16) y **P_µ =
  −P_η·p(2p−1)/3** (a 1e-8) en α_a = 0.1/0.3/0.6; partes regulares
  Q_η = 4.97/6.09/8.36, Q_µ = −4.98/−6.23/−9.42. Lectura revisada
  respecto de #18: el polo es FÍSICO (horizonte de sonido del khronon,
  parámetro pequeño aH/(c_s k)); la truncación QS exagera su residuo
  (2 frente a P_η: ×3.8 en α_a = 0.3) y pierde la dependencia en α_a; el
  «×28» de #18 era un cociente ventana-promediado frente a un arnés
  cuyas ICs QS-consistentes excitan los modos oscilatorios del khronon
  — el control lo muestra: η − 1 oscila entre −4.6e-2 y +5.4e-2 en la
  ventana; el arnés E2 no medía la cola. Arnés con ICs ADIABÁTICAS
  tomadas de la escalera: pendientes s_η = 17.80/17.41/17.23 y s_µ =
  −12.13/−11.72/−11.53 (k/H0 = 66.7/133.3/200), al 0.9–6.8 % de la
  escalera y ninguna compatible con la QS truncada (46.2, −3.79); dos
  brazos PASS, pero el de mayor e (e ≤ 0.02) da exponente log-log 2.14
  ∉ [1.9, 2.1] — curvatura O(e⁴) que el umbral congelado no calibró —
  y la regla clasifica **E3b = C**. No se retoca: estatuto «derivación
  exacta de la escalera, pendiente de confirmación numérica
  independiente»; recalibrar e_fit_max o ajustar con término e⁴ es
  materia de una preinscripción futura. Módulo: tail_pole_residues,
  eta_tail_physical y mu_local_tail_physical (SOLO el residuo del polo;
  Q no es despreciable: en α_a ≈ λ_K − 1 domina); eta_tail_coefficient
  marcada como QS truncada, no física. Magnitudes ilustrativas (z = 0,
  e = (H0/c)/k): (1.05, 0.3) → η − 1 = 7.6e-4 en k = 0.05 h/Mpc;
  (1.012, 0.012) → 2.7e-4 (solo polo: 6.7e-5). Fila `mu-eta-atlas`
  actualizada («colas pendientes» → «residuo del polo derivado; partes
  regulares y confirmación numérica pendientes»). Sin datos; ξ = 1;
  fronteras declaradas. Estatuto: derivación interna (E8)
  (`results/2026-09-13_mu_eta_atlas_tail/`).

- **Primera aplicación del pipeline SN a Dovekie REAL (13-sep):
  DESENLACE INDETERMINADO bajo la regla congelada — publicado sin
  reajuste**. Preinscripción congelada ANTES de la primera lectura de
  la columna MU (`f630834326c0`, en un commit que contiene su
  generador), citando por sha256 el PASS de mocks de #15 como
  precondición; doble barrera ejecutable en el ejecutor. Cinco brazos:
  (1) benchmark SN-only ΛCDM like-for-like con el chain oficial
  nautilus de DES-Dovekie (ingerido con manifest: 16384 filas, mismo
  soporte de prior [0.10, 0.50]): Ω_m propio = 0.3305 (+0.0154
  −0.0150) frente a 0.3306 ± 0.0154 oficial, **Δ = −0.005σ** — el
  pipeline reproduce el número oficial sobre los bytes oficiales;
  (2–5) ΛCDM y MCMC sobre Dovekie + CC (31) + BAO DESI DR2 (6), n =
  1857, con STAT+SYS principal y STATONLY de robustez, emcee 32×3000
  semilla 42, convergidos. STAT+SYS: χ²_ν(ΛCDM) = 0.893, Ω_m = 0.3295 ±
  0.0137, H0 = 67.77 ± 0.61; MCMC gana Δχ²_min = 0.04 con dos
  parámetros más → ΔAIC = +3.96, ΔBIC = +15.01 pro-ΛCDM; ε_Λ = +0.018 ±
  0.037, CI95 [−0.044, +0.091] ∋ 0. STATONLY: ΔAIC = +3.85, ΔBIC =
  +14.90, ε_Λ = +0.012 ± 0.037. La regla preinscrita (orden C → B → A →
  INDETERMINADO) descarta C (sin tensión) y B (sin preferencia); A
  exigía además σ(ε) ≥ 0.04 (posterior dominado por el prior de σ =
  0.05) y se obtiene 0.0371 — el ÚNICO criterio de A que no se cumple;
  el umbral se fijó sin calibración conjunta previa (los mocks de #15
  eran SN-only). La regla no se retoca: el desenlace es INDETERMINADO
  y se publica con el mismo peso que cualquier otro; una futura
  preinscripción distinta podrá recalibrar esa puerta, esta corrida
  no. Lectura honesta de los números (no del veredicto): las SNe
  Dovekie no identifican ε_Λ y el fondo MCMC no es preferido —
  coherente con la nota estructural de 6A (z_trans ≈ 9). Roles
  intactos: Dovekie real = primera aplicación · Unite = benchmark
  armonizado, NO replicación independiente · Union3 = contraste
  externo · Pantheon+ = disección. Estatuto: comprobación interna del
  pipeline sobre datos reales bajo preinscripción (E8) — no
  demostración física (`results/2026-09-13_dovekie_real/`).

- **Validación por mocks del pipeline SN Dovekie (12-sep, PR #15):
  las cuatro puertas preinscritas en PASS y la barrera de datos
  reales armada**. Preinscripción congelada ANTES de ejecutar puerta
  alguna (`d7cbabdc6d9f`), en un commit que contiene su generador
  (lección del 5E v1). Alcance DECLARADO: nuestro pipeline entra a
  nivel HD + covarianza; los 25 mocks fotométricos DES
  (1_SIMULATIONS @ c9a4fcaf) validan las etapas DES aguas arriba
  (SALT3 + BBC) y quedan registrados sin ingerir; la validación
  propia usa 25 realizaciones a nivel HD con la cosmología inyectada
  de los mocks DES (Ω_m = 0.315, ε = 0), los z del HD real
  (N = 1820, VERIFICADO POR BYTES — la cifra 1828 del traspaso era
  errónea) y la covarianza oficial, sin leer jamás la columna MU
  real. Resultados: (1a) fórmula χ² equivalente a la oficial
  ejecutada desde los bytes del release (max |Δχ²| = 1.5e-9;
  conteo 1820 idéntico en ambos parsers y ambos npz); (1b)
  integrador de producción O(h⁴) vs cuadratura independiente
  (max |Δμ| = 1.9e-6 mag); (2) media de pulls de Ω_m = +0.051
  (cota 0.6 = 3·SEM); (3) cobertura k68 = 17/25 ∈ [12, 21],
  k95 = 23/25 ∈ [21, 25] — no se detecta una descalibración
  incompatible con el tamaño de la muestra de mocks; (4) sin falsa
  preferencia: mediana ΔAIC = +3.94 y ΔBIC = +14.95 pro-ΛCDM con
  n(Δ<0) = 0, 0 ∈ CI95(ε) en 25/25, mediana p50(ε) = +0.015 (el
  centro del prior, como preinscribía la nota). BARRERA EJECUTABLE:
  el HD real (columna MU) solo es accesible tras este PASS citando
  la preinscripción por sha256 — vigilada por candado en la suite.
  Roles declarados: mocks = validación · Dovekie real = primera
  aplicación · Unite = benchmark armonizado, NO replicación
  independiente · Union3 = comprobación externa · Pantheon+ =
  disección de Unite. Estatuto: validación interna del pipeline
  (E8) — habilita consumir el dato real, no afirma nada sobre él
  (`results/2026-09-12_dovekie_mocks/`).

- **Sector perturbativo (12-sep): primera predicción out-of-sample
  del crecimiento — DESENLACE A, el aburrido preinscrito**. Con la
  preinscripción congelada ANTES de computar (desenlace esperado:
  fσ8^MCMC ≈ fσ8^ΛCDM dentro de la banda del prior de ε_Λ — control
  de consistencia, NO fracaso), el crecimiento lineal integrado
  EXACTO (extended_likelihoods.growth_D_f, no Linder) propagado
  sobre el posterior de fondo CC+BAO+SNe (ajuste v1 corregido, SIN
  crecimiento — la comparación RSD es out-of-sample y aquí no se
  ajusta nada, guardia ejecutable) da max_z |R_p50(z) − 1| = 4.3e-8
  frente a la envolvente del prior 6.4e-7, y Δχ²_oos = −8.5e-7 sobre
  11 puntos RSD (σ8 externa declarada, idéntica en ambos brazos). La
  predicción con cero parámetros nuevos es la razón R(z) (σ8 se
  cancela). NOTA ESTRUCTURAL heredada de 6A: con z_trans ≈ 9 y la
  transición normalizada hoy, ni el fondo ni el crecimiento lineal
  sondean ε en esta parametrización. El orden epistemológico del
  objetivo discriminante queda declarado, no ejecutado (ε_c/Atlas →
  µ, η → observables → datos; PROHIBIDO elegir µ, η desde lensing).
  Artefacto versionado con banda, semilla y sha256 de la
  preinscripción: `results/2026-09-12_perturbations_fsigma8/`.
  Estatuto: control de consistencia interna del sector lineal sobre
  el fondo (E8) — no validación del modelo frente a ΛCDM.

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

- **Sector perturbativo (13-sep): µ(k,a), η(k,a), Σ(k,a) DERIVADOS del
  canal Cronos, canal Atlas declarado, E1 = A y un hallazgo estructural
  sobre el cierre 'physical'**. Desde los objetos declarados del tratado
  (Ley de Cronos débil 11.1, geodésica 11.2, Gea 9.3), con tres cierres
  nombrados y sin campos ni parámetros nuevos: µ − 1 = ε̄_c(a)·(ck/aH)²/
  Ω_m(a), η = 1/µ (η − 1 = −(µ−1) a segundo orden), Σ = (1+µ)/2
  (Σ − 1 = (µ−1)/2 exacto), cola k², amplitud ∝ α₀⁻¹, límite GR exacto —
  identidades verificadas a precisión de máquina; ε̄_c sale de la
  función CANÓNICA cronos_v3.epsilon_c (una sola fuente para fórmula y
  cota) y la linealización es su derivada (d ln ε_c/d ln ρ = 3/2,
  testeado). Preinscripción congelada ANTES de computar: E1 aburrido
  (≤ 1e-3, cierre comoving), E2 identidades, E3 cadena halos → α₀⁻¹ →
  µ → observables DECLARADA y NO EJECUTADA, tabla a mano de la nota de
  teoría «a verificar». Resultados con α₀⁻¹ = 1e-6 (la COTA 11.5
  saturada — límite superior, no medida): **E1 = A**, max |R_µ − 1| =
  1.55e-4 en k ≤ 0.2 h/Mpc, z ≤ 2 — el sector lineal no inventa señal
  (µ − 1 = 9.7e-5 en z = 0, k = 0.1). La tabla de la nota se recomputó:
  z = 0 y z = 1 correctas (×0.8–1.1); la columna z = 3 del cierre
  physical estaba SOBREESTIMADA ×3–4 (5e-2 → 1.25e-2 en k = 0.1) — las
  cifras citables son las calculadas. HALLAZGO ESTRUCTURAL no previsto:
  el cierre 'physical' (ρ_c = 200·ρ̄_m(0)) con α₀⁻¹ en la cota NO es
  perturbativo — µ − 1 ∝ (1+z)^{7/2} alcanza 1 en z = 8.4 (k = 0.2) …
  51 (k = 0.01), ε̄_c ≥ 1 para z ≳ 125, el propio fondo supera ρ_c para
  z > 4.85 y la ODE de crecimiento diverge (el código FALLA CERRADO,
  nunca publica NaN). Consecuencia enunciable sin datos: bajo 2b una
  amplitud viable exige α₀⁻¹ ≪ 1e-6 — la consistencia del sector lineal
  acota 2b por debajo de la cota galáctica, discriminador interno más
  fuerte de lo que la nota estimaba; el criterio «200× la media»
  favorece conceptualmente el cierre comoving. Atlas: gancho con
  coeficientes O(1) PENDIENTES (frente 3), contribución no computada.
  Sin datos: la prohibición «µ, η nunca desde lensing» es ejecutable
  (test); growth_D_f acepta µ(a) y con µ ≡ 1 es idéntico bit a bit;
  perturbations.mu_modifier/eta_modifier quedan declarados placeholders
  no derivados. Artefacto: `results/2026-09-13_mu_eta_cronos/`.
  Estatuto: derivación interna con cierres declarados (E8) y predicción
  sin contraste — no detección, no validación; amplitud = límite
  superior.

- **Canal Atlas de µ/η DERIVADO desde la acción (13-sep, frente 3):
  desenlace A — la firma sub-horizonte se cancela**. Re-derivación
  simbólica (sympy) desde la Acción de Gea (9.1) + Término de Cronos
  (9.4) en gauge unitario con polvo, sin objetos ajenos, bajo
  preinscripción congelada: las ONCE identidades E1 reproducidas (Sello
  G_cosmo = 2G_B/(3λ_K−1); δ = j0+3φ; µ_QS = 2ξ/[(2ξ−α_a)+3(3λ_K−1)e²];
  η_QS racional; µ_sub = 1/(1−α_a/2ξ); η_sub = 1; GR; G_growth = G_local
  = G_B/(ξ−α_a/2) — la G estática calculada de forma independiente en
  Minkowski con fondo vacío, registrando el residual de Jeans de un
  fondo uniforme no autoconsistente; c_s² = ξ(2ξ−α_a)(λ_K−1)/(α_a(3λ_K−1));
  coef. cinético ∝ (3λ_K−1)/(λ_K−1); c_T² = ξ — sector tensorial añadido
  a los scripts originales). E2, integración del sistema lineal COMPLETO
  con relajación al modo creciente (a_start = 1e-3, pureza por
  constancia de p): µ_num/µ_QS − 1 = +2.8e-04/-5.3e-04/+5.9e-05, η_num − 1 = +3.0e-04/+4.6e-04/-3.3e-05,
  p_num = p_QS = 1.1342 a 6e-04 (k/H0 = 66.7/200/600; umbrales 1 %).
  RESULTADO: el offset α_a/(2ξ) se CANCELA exactamente contra la
  renormalización de la G local — el canal Atlas no deja firma
  sub-horizonte en (µ, η) al orden dominante y la cola k² del canal
  Cronos queda como la ÚNICA firma sub-horizonte del sector
  perturbativo; la expectativa previa «µ_Atlas − 1 = O(1)·ε_K» queda
  SUPERADA (ε_K entra en el fondo: G_cosmo/G_local = (2ξ−α_a)/(3λ_K−1)
  ≈ 1 − (3/2)ε_K − α_a/2, que BBN acota como combinación — fila de los
  Residuos refinada). Dos precisiones: (i) η_QS es 0/0 en λ_K = 1 exacto
  (a e finito daría 1/3): el límite GR toma e → 0 primero; (ii) la cola
  QS de η lleva el polo 1/(λ_K−1) (~170 con ε_K = 0.012), pero el
  sistema completo la da ×~28 menor a e = 0.01 (3.0e-4 frente a
  8.3e-3): el polo es un artefacto de la truncación QS al nivel medido
  y los coeficientes reales de las colas siguen siendo frontera
  declarada (sector de velocidades), junto con el régimen
  superhorizonte, las cotas PPN sobre (ε_K, α_a) y el acoplamiento
  fuerte. Erratum candidata para la v36 (H.2.2): c_s² diverge, no se
  anula, cuando α_a → 0 a λ_K fijo (decisión del autor). Artefacto:
  `results/2026-09-13_mu_eta_atlas/`. Estatuto: derivación interna
  verificada (E8); teoría pura, sin datos.

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
