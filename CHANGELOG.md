# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## Sin publicar — septiembre de 2026

- **Vacío 2D fuera del polo de masa (22-sep): brazos corriente de
  conversión κΣ̇ê_E y rotación de la inclinación η_eff = η₀(1 − f/f_×) en
  el reloj S, con J∇C de control, bajo preinscripción congelada (sha256
  `3990f7bcaee7…`) — INDETERMINADO en los dos brazos y los cuatro δ₀** por
  puertas falladas dentro de la rejilla (κ̂ ≥ 4.5–7.4 no integrable con el
  d_sigma declarado; f_× ≥ 0.75–0.87 sin terminar el descenso; identidad
  > 1e-6 en celdas de f_× pequeño). Publicado sin letra: la corriente
  cruza solo con κ̂ ≥ 2.11 (1.65 en δ₀ ≥ 0.03), siempre con S_cross ≤ 0.87
  y retorno al polo de masa; la rotación cruza en el mismo S que el
  control J∇C (0.931 / 0.880 / 0.807 / 0.753): el S del cruce lo fija la
  descarga, no el término; para f_× ≤ 0.72 el campo se queda en el polo
  de espacio. `ClockConfig.kappa_conv`, `tilt_cross_f`; W_conv y W_tilt
  en la identidad S = f − f₀ + (W_J + W_conv + W_tilt)/T₀; panel Mp/Ep
  derivado; guarda de divergencia del integrador (arreglo del instrumento
  posterior a la preinscripción, declarado; publica `diverged`, no
  reescala). `scripts/run_vacuum_2d.py`, `tests/test_vacuum_2d_lock.py`,
  fila `vacio-2d-conversion-y-rotacion`, Nota I §3.33, artefacto
  `results/2026-09-22_vacuum_2d/`.

- **Decisiones A, B y C del autor (22-sep), derivadas del tratado y
  ejecutadas por el reloj S**. **A** — la T₀ que el Lema 10.3 iguala a
  W_max es la del paisaje completo (el contenido del Lema es T₀ ≤ W_max;
  c̄δ³ es la expresión de la Prop. 3.4, que declara la inclinación como
  corrección). Con la inclinación en los dos lados, δ_H se recalcula como
  raíz de λ_Ad_full(δ) = λ_H (`B7_empalme.sealed_curvature_lambda_full`,
  `delta0_required_full`; espejo `core.delta0_circle.delta0_H_full`):
  δ_H_full = 0.05544 (−4.6 %; la curvatura sellada sube ×1.050 en δ_H) y
  el Techo decidido es W_max = T₀_full(δ_H_full) = 1.869e-4, entre la ley
  3.4 en δ_H_ley (1.652e-4) y el paisaje completo en δ_H_ley (2.167e-4);
  las tres cadenas se publican (`W_max_required_decided`, `DECISION_A`).
  **B** — la nucleación de S₀ es de un grado de libertad (n = 1: sin
  soporte espacial en S₀; el bounce O(4) entrega f₀ = 0.9996 y se traga
  la descarga): `nucleation='kramers'` (Axioma 4, D_ent DECLARADA como
  fracción de la barrera) y `'gamow'` como cota; `'bounce'` queda como
  control negativo etiquetado. Corridas: Γ_K = 8.4e-10 / 6.8e-6 / 1.7e-5
  en δ₀ = 0.01 y 1.7e-8 / 1.4e-4 / 3.4e-4 en δ_H_full para D_ent/ΔV_b =
  0.1 / 1 / 10 (Gamow 1.6e-3 / 7.4e-3), identidad S ≡ f a 2e-13. Lo
  abierto no es n_dim sino D_ent (frente 2). **C** — «el colapso» es
  D = 0 (Cruce de Victoria, Def. 8.4 / Obs. 8.6), el gatillo del estado
  ocupado; M0² = 0 es la desaparición del falso vacío que el campo ya
  abandonó: diagnóstico (`DECLARED_FORMS['collapse_trigger']`). Bajo el
  cierre canónico D no cruza cero en ninguna corrida (M0² = 0 ocurre en
  5): el colapso del tratado exige β del frente 2 y los umbrales siguen
  impuestos. El código ejecuta las decisiones, no las demuestra (E8).
  Artefacto `results/2026-09-21_s_clock/` regenerado con la sección v2;
  filas `circulo-delta0`, `nucleacion-gamma0`, `s-clock-simulador-consistencia`
  y `diagonal-j-requerido` actualizadas; Nota I §3.30; tests nuevos.

- **Frente 5 (b) (21-sep, tarde): serie de resolución del campo de
  Cronos (k_inner 64/128/256) a A_Sculptor y 0.05·A_Sculptor con campo
  instantáneo y balance K + W + (2/5)U_C + W_fric, bajo preinscripción
  congelada — desenlace INDETERMINADO por la puerta de energía**. Tres
  commits (instrumento + generador · congelación · corridas).
  `cronos/halo_nbody.py`: `follow_particles` (malla del campo
  reconstruida de las partículas en cada actualización, sin malla fija
  ni media móvil), `U_self = (2/5)·Σm(−c²ε_c)` (el funcional del que
  deriva la fuerza) y `W_fric` (trabajo de la fricción); la instantánea
  publica `E_self`. Preinscripción (sha256 18c5344d9d0b…): N = 2e5,
  1 Gyr, 7 brazos, convergencia si los dos saltos |log10 M(<0.4 kpc)| ≤
  0.15, letras A/B/C/D/INDETERMINADO con la predicción del criterio
  (A). **Resultado**: INDETERMINADO — |ΔE_self/E| = 2.4 % y 4.0 % en los
  brazos de A_Sculptor con k = 64 y 128 (tolerancia 2 %): con A_Sculptor
  el campo dinámico inyecta energía. Publicado como tabla: la serie a
  A_Sculptor no converge (salto +0.44 dex en 128→256) y la de
  0.05·A_Sculptor converge (−0.05, +0.006 dex), como predice el
  criterio, pero con 18–87 partículas dentro de 0.4 kpc el métrico está
  dominado por el conteo (el newtoniano fluctúa 4.2 → 3.1 ×1e7 M☉) y el
  radio interior del campo cae en 0.41–0.75 kpc. Fricción irrelevante
  (W_fric/|E| ≤ 6e-6). Ronda siguiente: preinscripción nueva con N ≥
  1e6 y métrico resuelto. Fila nueva `halo-serie-resolucion-campo`
  (interno). Nota I §3.29. Registro: 57 claims (sobre main).

- **Test preinscrito del criterio de Cronos–Jeans (frente 5a, 21-sep):
  desenlace INDETERMINADO bajo la regla congelada; donde la fase lineal
  se resuelve, γ/k coincide con la predicción cinética exacta al 1 %**.
  Tres commits (instrumento + predicción · congelación · corridas).
  `cronos/cronos_jeans_kinetic.py`: relación de dispersión de Vlasov con
  la fuerza local, 1 = q[1 + ζZ(ζ)]; modo creciente √π·y·e^{y²}·erfc(y) =
  1 − 1/q, γ = √2kσy(q) (0.149 en q = 1.2 frente a 0.447 del fluido;
  0.612 en q = 2); función de transferencia del instrumento W(k).
  `cronos/cronos_jeans_1d.py`: arranque silencioso multihaz
  (Denavit–Walsh) y siembra de un modo. Preinscripción (sha256
  3dcd7417db06…): q ∈ {0.5, 0.8, 1.2, 2.0} × modos {4, 8, 16}, N = 2e6,
  1024 haces, ventana [3e-5, 3e-4], tolerancias 25 %/25 %/10 %, puertas,
  C/B/A/INDETERMINADO; tres rondas de pilotos declaradas. **Resultado**:
  INDETERMINADO — la puerta «fase lineal resuelta» (≥ 8 puntos con r² ≥
  0.98) falla en q = 1.2 n = 4, q = 2.0 n = 4 y q = 2.0 n = 8 (modos bajos
  a q > 1: batido con modos amortiguados); se retiene sin tocar nada.
  Publicado como tabla: q < 1 estable en los seis modos; q = 1.2: γ/k =
  0.1459 / 0.1375 (n = 8 / 16) frente a 0.1465 / 0.1384 cinético; q = 2,
  n = 16: 0.5976 frente a 0.5994; convergencia en haces 0.7 %. Ronda
  siguiente: preinscripción nueva para los modos bajos. Fila nueva
  `test-criterio-cronos-jeans` (interno). Nota I §3.28. Registro: 57
  claims (sobre la base de esta rama).

- **Frente 5 (c) y (d) (21-sep, tarde): preinscripciones «Oort–K_z» y
  «perfil σ_los(R) de Sculptor» congeladas; la vecindad solar EXCLUYE la
  amplitud única del 5E en el plano y deja A ≲ 0.06·A_Sculptor (banda
  ×4 por la losa); Sculptor queda en fallo cerrado hasta la ingesta**.
  Tres commits (dataset + módulo + generadores · congelación · corridas).
  Dataset `local_kz_bounds` (cinco valores publicados TRANSCRITOS —
  Holmberg & Flynn 2000, McKee, Parravano & Hollenbach 2015, Bovy &
  Tremaine 2012 — bytes oficiales no verificados, aviso propagado).
  `dynamics/local_kz.py`: el término de Cronos como densidad dinámica
  efectiva Δρ_eff(0) = (1/4πG)dg_C/dz|₀ y ΔΣ_eff(1.1) = g_C(1.1 kpc)/(2πG)
  frente al margen ρ_dyn − ρ_bar = 0.016 ± 0.014 M☉/pc³ y Σ_dyn − Σ_bar =
  20.9 ± 5.2 M☉/pc² (todo el margen para Cronos). Resultado bajo la regla
  congelada (`results/2026-09-21_oort_kz/`): A_Sculptor → Δρ_eff(0) =
  0.747 M☉/pc³, z = +50.8, **excluido**; 0.05·A_Sculptor → z = +1.5,
  **compatible**; A_2σ = 0.060·A_Sculptor (banda 0.015 / 0.060 / 0.240 con
  alturas ×½ / ×1 / ×2); la columna a 1.1 kpc no constriñe. Coincide con
  la cota del criterio de Cronos–Jeans (A ≲ 0.048·A_S). Perfil de
  Sculptor (`results/2026-09-21_sculptor_profile/`): predicción congelada
  para β ∈ {−0.5, 0, 0.3} (pico 14–25 km/s, exterior ≈ 2 km/s frente a
  ≈ 9–10 plano), regla A/B/C, **INDETERMINADO por fallo cerrado**
  (walker2009 DATA_UNAVAILABLE: CDS/VizieR 403 vía proxy). Filas nuevas
  `oort-kz-cota-cronos` (condicional) y `sculptor-perfil-sigma-los`
  (experimento-no-ejecutado). Nota I §3.26–3.27. Registro: 58 claims
  (sobre main).

- **La diagonal y el |J| requerido (21-sep, tarde): circulación J∇C del
  Camino de dos niveles en `path_flow` y en el reloj S, bajo
  preinscripción congelada — veredicto MIXTO**. Punto (3) del orden del
  21-sep (tarde), tres commits (generador · congelación · corridas).
  `core/path_flow.py`: `circulation(φ, ∇V, |J|, C)` con J = |J|·ε
  antisimétrica; C = 'V' (J∇V ⊥ ∇V: Monotonía 4.5 exacta) o 'rho2'
  (rotación rígida); `flow(..., J, circulation_C)` devuelve W_J.
  `core/s_clock.py`: `J_circ`/`circulation_C`, identidad S = f − f₀ +
  W_J/T₀, `diagonal_crossing_S`, `J_min_threshold`, `J_required`,
  `DECLARED_FORMS["J_circulation"]`. HALLAZGO: con C = V el término se
  anula en los puntos críticos y el vacío 2D está en el polo de masa —
  θ cruza la diagonal solo si |J| ≥ |J|_min y vuelve a 0: el cruce tiene
  un S máximo (subir |J| lo adelanta). Resultado: |J|_min = 1.358–1.451
  para δ₀ ∈ {0.001, 0.003, 0.01, 0.03, δ_H} (invariante de escala al
  6.8 %), S_max = 0.959 / 0.930 / 0.879 / 0.806 / 0.746 ⟹ **A** en 0.001
  (ventana 1 ± 0.05 alcanzable con |J| ∈ [1.358, 1.396]), **B** en 0.003 y
  0.01, **F** en 0.03 y δ_H (retorno a la frontera φ_E = 0 con W_J/T₀ =
  5.5 / 10.9 y descenso sin completar). La alternativa rígida cruza pero
  con W_J/T₀ = 2–10 y Monotonía rota (coste). La expectativa E13 |J| ≈
  (π/4)/∫|∇V|dσ sobreestima ×2.2–25 (publicado). |J| es calibración
  declarada, no derivación (frente 2). 3 tests + candado. Fila nueva
  `diagonal-j-requerido` (interno). Nota I §3.25. Registro: 58 claims
  (sobre la rama del reloj v1).

- **Nucleación ronda 2 (21-sep, tarde): Γ₀(δ₀) condicional a n_dim ∈
  {1, 3, 4} bajo preinscripción congelada — veredicto (A sí, B sí, C
  no)**. Punto (2) del orden del 21-sep (tarde), tres commits
  (generador · congelación · corridas). `core/nucleation.py`:
  `N_DIM_DECLARED`, `gamow_tunnel` (n = 1: B₁ = 2∫√(2G(V − V_fv))dρ con
  cuadratura adaptativa, prefactor de Gamow ω_fv/2π = m̄δ₀/2π),
  `b1_no_tilt` (0.5178), `gamma0(δ₀, n_dim)`, `path_deformation` (el
  camino de túnel libre de curvarse en (ρ, χ), Nelder–Mead desde el rayo
  θ = 0); `s_clock` con `nucleation='gamow'` (el reloj arranca desde Γ₀:
  publica Γ₀ y σ_nuc = 1/Γ₀). Preinscripción (sha256 46f486e1cc84…):
  ē ∈ {0.5, 1, 2}, δ₀ ∈ [1e-4, 0.9·δ₀_max(ē)], ajuste en los tres puntos
  más bajos, A/B/C/INDETERMINADO con tolerancias a priori. Resultado:
  **A sí** (exponente de Γ₀ con n = 1: 0.9998 / 0.9993 / 0.9978; B₁/δ₀² →
  0.507 / 0.497 / 0.476 frente a b₁ = 0.518), **B sí** (exponente de B
  con n = 4: −1.03 / −1.06 / −1.11), **C no** (B_min/B_ray = 1.0000 en los
  seis casos); n = 3 publicado (B → constante). Las tres leyes cumplen
  Γ₀(0) = 0 por mecanismos distintos: cuál es la del tratado es la
  decisión B del autor (E13). 4 tests + candado. Fila nueva
  `nucleacion-gamma0` (interno-condicional). Nota I §3.24. Registro: 58
  claims.

- **Reloj S v1 (21-sep, tarde): las mejoras que el v0 ya pedía —
  κ₁ en forma cerrada, el Techo con las dos T₀, la raíz del origen,
  δ₀_max ∝ ē⁻² y su naturalidad, la inestabilidad de masa como evento
  del modo diagnóstico, la unidad de S tras Florencia declarada como
  hueco, y el guion v32 archivado**. Punto (1), (4) y (5) del orden del
  21-sep (tarde). `T0_tilt_first_order`: T₀_full/T₀_ley − 1 = κ₁√δ₀ +
  O(δ₀) con κ₁ = ē√κ₊/(√2c̄) = 1.361 (formas por defecto; cociente
  medido/primer orden 0.98 en δ₀ = 0.01, 0.95 en δ_H; residuo ≈
  −0.27·δ₀ publicado). `W_max_required_both`: el Techo del círculo
  publicado con la ley 3.4 (1.652e-4 en δ_H) y con el paisaje completo
  (2.167e-4, +31 %) — cuál nombra el Lema 10.3 es la decisión A del
  autor; la fila `circulo-delta0` hereda la corrección como condicional.
  `radial_landscape` restituye la raíz del origen (sin inclinación el
  origen es el falso vacío exacto y δ₀_max = ∞; la malla de la v0
  arrancaba en 1e-9 y no la veía). δ₀_max en forma cerrada,
  2·g_max(m̄, b̄, C0)²/ē² = 0.1028·ē⁻² (bisección a 3e-8; ē ∈ {0.5, 1,
  2} comprobados) y barrido de naturalidad sobre `landscape_priors`
  (tres priors, n = 4000): la metastabilidad en δ_H la admite una
  minoría de paisajes viables (0.21–0.38 según el prior; 0.51–0.74 en
  0.012) — cartografía publicada sin veredicto.
  Modo emergente: el cruce M0² = 0 ya no detiene el bucle — evento
  `inestabilidad_masa` con S_flip medido frente a la forma cerrada
  δ₀·[b̄ − √(b̄² − 6C0m̄²)]/(24aτC0) = 0.1057·δ₀/τ (cociente 0.94–1.00) y
  continuación hasta C0 ≤ 0; tabla τ_k = S_flip(τ = 1)·δ₀/S_k ≈
  11.8·δ₀·10⁻ᵏ («Década ⟺ τ_k») etiquetada E13: reformulación del
  diccionario τ desde los umbrales calibrados, no derivación; hipótesis
  declarada `tau_per_dim` (τ por dimensión, M0² repuesto en cada
  inestabilidad) publicada como opción sin estatuto. Formas declaradas
  nuevas: `S_post_unit` (la unidad de S tras Florencia no se deriva de
  T₀: fila nueva `diccionario-unidad-S-post-florencia`, hueco),
  `mass_instability`, `tau_per_dim`. `viewer/legacy_v3_storyboard.html`:
  el guion v32 del autor archivado como contrato de interfaz (no
  cálculo), enlazado con esa advertencia desde `viewer/s_clock.html`.
  7 tests nuevos (+1 en el círculo). Filas revisadas
  `s-clock-simulador-consistencia` y `circulo-delta0`. Nota I §3.23.
  Registro: 57 claims.

- **La salida de S₀ (21-sep): `core/nucleation.py` — bounce de Coleman
  O(d) sobre el Basal completo, escape de Kramers del Flujo del Camino
  y el reloj S arrancando en el estado nucleado; hallazgo: con pared
  gruesa la descarga cae dentro de la nucleación**. Punto (3) del orden
  del 21-sep. Instantón O(d) del campo radial sobre el corte θ = 0
  (salida al polo de masa) por disparo overshoot/undershoot con la
  inclinación incluida; acción B y Γ₀/A = e^{−B} (prefactor A
  declarado); ley de escala medida B ∝ δ₀^{−1.49} (d = 4; argumento
  3 − d = −1: Γ₀(0) = 0 sin prefactor) y δ₀^{−0.40} (d = 3; argumento
  0: B → constante, el axioma exige A(δ₀) → 0). Como la barrera vale
  1–8 % de T₀ (pared gruesa; la estimación de pared delgada no es
  aplicable y va solo como referencia), el bounce entrega el campo casi
  en el vacío verdadero: f₀ = 0.9996 (d = 4) y 0.97 (d = 3) en δ₀ =
  0.01, así que los umbrales 0.009/0.099(/0.999) caen DENTRO de la
  nucleación y el reloj S los dispara en σ = 0 con esa nota — la
  descarga no ocurre a lo largo del Flujo del Camino. Escape de
  Kramers (la nucleación propia de la dinámica disipativa de primer
  orden del Axioma 4, con la difusión entrópica de la Def. 4.4 y D_ent
  declarada): prefactor ∝ δ₀^{1.85} (argumento 2), barrera ∝ δ₀^{2.44}
  (argumento 3, erosionado por la inclinación) ⟹ Γ_K(0) = 0 por el
  prefactor, mecanismo opuesto al bounce. Integración conjunta Φ_Ad ⊗
  λ_i desde el bounce (modo emergente): D no cruza cero; S_emergent
  publicado vacío. `s_clock`: `nucleation='bounce'`, f₀ (entropía de
  nucleación) publicada, S ≡ f − f₀ comprobada (2e-13). Convenciones
  declaradas: d, A, normalización de la acción, D_ent, corte θ = 0;
  cuál es la nucleación del tratado es del diccionario. 8 tests. Fila
  nueva `nucleacion-salida-s0` (interno); revisada
  `s-clock-simulador-consistencia`. Nota I §3.22. Registro: 56 claims.

- **Ronda de registro H1 (21-sep): el criterio de Cronos–Jeans — la ley
  local ε_c = A·ρ^{3/2} es ultravioleta-inestable donde q ≡
  (3/2)c²ε_c/σ² > 1, y con A_Sculptor el halo del Nivel A, el propio
  Sculptor y la vecindad solar caen dentro**. Derivación del autor
  (sesión de verificación del 21-sep), reproducida número a número
  (`dynamics/cronos_jeans.py`, `cronos/cronos_jeans_1d.py`,
  `results/2026-09-21_cronos_jeans/`): la respuesta de Boltzmann de un
  medio colisionless isotermo al potencial local −c²ε_c(ρ) admite una
  perturbación estática autoconsistente cuando ρc²ε_c'/σ² ≥ 1; en el
  límite fluido ω² = (σ² − (3/2)c²ε_c)k² − 4πGρ, así que la fuerza de
  Cronos es ultravioleta (tasa ∝ k) y un N-cuerpos con q > 1 no puede
  converger: refinar el campo aumenta la tasa. (1) Halo del Nivel A con
  A_Sculptor: r_CJ = 0.71 kpc, M(<r_CJ) = 1.6e8 M☉, q = 74 en 0.1 kpc y
  3.5 en 0.4 — **no existe solución convergida** del estado inicial; la
  no convergencia de b_res es la predicción, no un fallo del
  instrumento. (2) Vecindad solar: g_C/K_z = 2–7 entre 50 y 300 pc,
  límite de Oort efectivo 0.75 M☉/pc³ frente a 0.10 ± 0.01 (referencia,
  no ingerida); cotas A ≤ 0.048·A_S y A < 0.16·A_S. (3) Sculptor con el
  montaje congelado del 5E: q = 2–2.4 dentro de ≈ 430 pc y σ_los(R)
  predicha 19.1 → 2.6 km/s de 10 a 500 pc (pico central y exterior
  newtoniano; el promedio da 9.2 por construcción) — forma congelada
  ANTES de la ingesta en la fila `jeans-dsph`. Test 1D del umbral
  (láminas sin gravedad): q = 0.8 se queda en el ruido de Poisson, 1.2
  crece, 2 satura en ~2 % del tiempo de cruce, 4 es no lineal desde el
  inicio; la ley ∝ k no se mide aquí. Lectura añadida al Nivel A (fila
  `nivel-a-halo-aislado`): el mecanismo del vaciado y de la ganancia de
  energía es el estimador del campo (malla fija en la primera llamada y
  recorte de la pendiente a 1e-3: el pozo se retira cuando la ley exige
  que se profundice), el retardo modula el ritmo; la fricción y el lapso
  no frenan nada a esta amplitud; con campo autoconsistente instantáneo
  se conserva K + W + (2/5)U_C. Consecuencia: el Nivel A-2 contratado
  (convergencia k_inner, zoom) se retira; lo sustituye el frente 5
  refundado (test del criterio con predicción convergida, halo con A_S
  y 0.05·A_S con balance 2/5, preinscripciones Oort–K_z y perfil de
  Sculptor, ingesta SPARC oficial sin tocar A_Sculptor). Filas: nueva
  `criterio-cronos-jeans` (interno); revisadas `cronos-amplitud-unica`
  (falsador Oort–K_z, A_req/A_Kz ≳ 20), `contenido-materia-lectura-
  operativa` (inestable dentro de r_CJ), `jeans-dsph` (forma predicha
  congelada), `nivel-a-halo-aislado` (estimador; mal puesto dentro de
  r_CJ). Cita de la RAR para futuras preinscripciones: 1609.05917
  (McGaugh, Lelli & Schombert 2016) y **1610.08981** (Lelli et al. 2017;
  1610.06183 es Keller & Wadsley 2017, línea base ΛCDM). Las vistas de
  trazabilidad ya no citan «v35.1» (fe de erratas E1–E13). Registro: 55
  claims. Nada toca A_Sculptor ni ningún umbral; el Nivel A sigue
  INDETERMINADO.

- **El reloj S (21-sep): `core/s_clock.py`, orquestador de la
  trayectoria S₀ → S_{1,001} como SIMULADOR DE CONSISTENCIA — umbrales
  impuestos, cada eslabón no derivado etiquetado; hallazgo: la
  metastabilidad del falso vacío tiene un δ₀ máximo**. Tercer paso del
  orden confirmado (registro → Nivel A → `s_clock`). Un solo bucle en σ
  integra el Flujo del Camino (4.2) sobre el Basal completo (con la
  inclinación −η·χ) desde el punto de escape de Coleman, contabiliza la
  descarga f = (V_fv − V)/T₀ y el índice S = ∫Σ̇dσ/T₀ (normalización
  declarada: S = 1 ⟺ descarga completa), lleva los acoplos por las β de
  Fokker–Planck (opcional), evalúa D(S), dispara los colapsos como
  eventos d → d+1 con el álgebra C(d+1,0), integra los sellados por
  congelación de c_eff (§5.3) y m_eff (§5.4) con FORMA DECLARADA
  (`DECLARED_FORMS`: la ec. 5.3 no está transcrita), sigue θ y en 1,001
  aplica la Rotación de Florencia y ENTREGA el estado inicial
  cosmológico, declarándolo ilegible (diccionario ausente). Derivado y
  superado: T₀ = c̄δ₀³ (1e-15 sin inclinación), monotonía, producción
  entrópica ≥ 0, exclusión, salida al polo de masa, S ≡ f a 2e-13 (el
  flujo se proyecta sobre φ_E ≥ 0 y la ligadura no trabaja: Teo. 4.5
  sobre el dominio físico), álgebras euclidianas anticonmutantes en cada
  colapso, séxtico marginal en d = 3, firma −+++ con control negativo de
  dos giros, RP en la loncha con control J < 0, identidad m_H =
  √(2β₃)v₃, Sello de Newton y recuperación de ΛCDM exactos. Impuesto o
  declarado: los umbrales 0.009/0.099/0.999 (Prop. 8.1, λ = 10 es el
  frente 2), la nucleación (Γ₀ no calculada; σ = 0 en el punto de
  escape), las formas de sellado, el cruce de la diagonal (el flujo del
  Basal NO lo produce: θ → 0 y se queda), los cuantos V3D/Florencia tras
  el residuo de descarga ε_res, Φ_ten = 0, el δ₀ del ciclo siguiente
  (γ_R, frente 4). **Hallazgo**: con η = ē·δ₀³ el falso vacío deja de
  existir por encima de δ₀_max = 0.1028 (formas por defecto) — la Obs.
  8.6 (D > 0) es necesaria, no suficiente — y T₀ medida supera la ley
  c̄δ₀³ en 13 % (δ₀ = 0.01) a 40 % (δ₀ = 0.1) por la inclinación
  (O(δ₀^{1/2}) relativa, Prop. 3.4). **Modo emergente (diagnóstico, sin
  estatuto)**: con las β canónicas y τ ∈ [1e-3, 1] declarado, D se hunde
  pero no cruza cero antes del fin del recorrido (M0² cruza cero antes
  que D para τ grande): los colapsos no emergen del Cruce de Victoria en
  el rango explorado. m_H(δ₀) publicado como dependiente del input (52
  GeV en δ₀ = 0.01; 125.4 en δ_H por construcción). LEDGER obligatorio
  en el artefacto (cada comprobación y evento con su insignia: 16
  derivadas, 4 impuestas, 4 declaradas, 1 publicada, 1 hueco; ninguna
  derivada fallida) y cada colapso con sus dos lecturas S_imposed /
  S_emergent (esta última vacía: D no cruza cero con acoplos
  congelados). Visor `viewer/s_clock.html` (HTML estático, sin
  dependencias) que lee el artefacto y muestra trayectoria, eventos y
  ledger con insignias de estatuto. 12 tests. Fila
  nueva `s-clock-simulador-consistencia` (interno); revisadas
  `c-lieb-robinson-identificacion` (forma declarada integrada; la
  ecuación sigue ausente), `diccionario-primordial-cosmologico` (el
  hueco es ahora un campo vacío en el artefacto) y
  `decada-discriminante` (diagnóstico emergente). Nota I §3.20.
  Registro: 54 claims. Lo que NO afirma: ningún umbral emerge; nada de
  esto es simulación del modelo ni demostración física.

- **Nivel A del frente 5 (17/18-sep): halo aislado NFW de 10¹¹ M☉ con y
  sin Cronos v3 a A_Sculptor — desenlace INDETERMINADO; lo único robusto
  es una contracción inicial del interior de amplitud no convergida**.
  Preinscripción congelada ANTES de generar ninguna condición inicial
  (`72c534dc1c09`, en un commit que contiene su generador), con pilotos
  de desarrollo declarados (N ≤ 5e4 y cortes de 0.1 Gyr a N = 4e5 que
  fijaron el estimador del campo y Δt_min, ningún umbral). Maquinaria
  nueva (`cronos/halo_nbody.py`): condiciones iniciales de Eddington
  para un NFW truncado, campo medio esférico de Cronos con promedio
  temporal, los tres términos del Cor. 11.3 (fuerza +c²∇ε_c, fricción
  con compuerta, lapso), árbol Barnes–Hut (pytreegrav, extra `nbody`),
  pasos individuales y reglas de parada. Seis brazos, diez corridas
  (~14 h de pared): a (newtoniano ×3, 2 Gyr), b (Cronos completo ×3, 2
  Gyr), b_static (campo congelado: |ΔE/E| = 2.6e-4, puerta superada),
  b_res (k_inner = 128), c (cierre cosmológico: parado en t = 0 con
  ε_c,max = 3.5, como la regla exigía — la exclusión del 17-sep queda
  registrada dinámicamente) y c′ (10·A_Sculptor). Puertas superadas:
  energía de a (≤ 1.4e-4), equilibrio de a (0.019 dex), régimen débil en
  b (ε_c ≤ 8e-7). **INDETERMINADO por la letra de la regla**: c′ debía
  desplazar ρ en [0.4, 1) kpc ≥ 0.10 dex a 0.5 Gyr, pero la regla de
  parada (v_well > 1000 km/s, también congelada) lo detuvo en 0.055
  Gyr, cuando esa banda había cambiado +0.05 dex — dos reglas
  congeladas colisionaron y el resultado se retiene entero, sin tocar
  umbrales. c′ sí responde donde la regla no miraba: M(<0.4 kpc) ×7.1,
  v_well 306 → 1000 km/s, ε_c(ε_soft) ×11 en 55 Myr. Números
  descriptivos de b frente a a en 2 Gyr (publicados, sin veredicto):
  log10(ρ_b/ρ_a) = −0.28 dex en [0.4, 1) kpc (−0.39, −0.01, −0.45 por
  semilla), −0.10 en [1, 2.3), ~0 fuera de 2.3 kpc; M_b/M_a(<0.4 kpc)
  1.0 → 4.3 (0.25 Gyr) → 0.68 → 1.08 (2 Gyr). Robusto en todos los
  brazos Cronos: contracción del interior en los primeros 0.25 Gyr
  (×8.4, ×1.3, ×3.3 por semilla; ×2.6 con k_inner = 128; ×2.2 con el
  campo congelado; ×7.1 a 10·A_S) — dirección conforme a la expectativa
  E13 declarada, amplitud dispersa ×1.3–×8. NO convergido: b_res difiere
  de b en +0.83 dex en M(<0.4 kpc) (umbral 0.15) y las corridas con
  campo dinámico ganan energía (+1.7 %, +0.5 %, +1.4 %; b_res +0.25 %)
  que no es U_Cronos (~3e-3 de E): sospecha de trabajo del campo medio
  retardado (τ_avg = 50 Myr frente a t_dyn ≲ 10 Myr dentro de 0.4 kpc),
  declarada como hipótesis para la ronda siguiente. Ajuste de forma: la
  media de b prefiere cored por 0.02 dex (< 0.05: no es C); a prefiere
  NFW. Dos correcciones de código declaradas tras la primera pasada del
  análisis, sin efecto sobre umbrales ni desenlace: el retorno temprano
  del brazo c no escribía la lista de eventos (c se re-ejecutó, 3 s,
  determinista) y la máscara de bandas daba −inf con un bin vacío en
  una semilla (ahora cada par se promedia sobre sus bins poblados y se
  publica el número de bins vacíos). El primer lanzamiento murió con un
  reinicio del contenedor antes de escribir ninguna corrida y se
  relanzó entero. Fila nueva `nivel-a-halo-aislado` (interno,
  INDETERMINADO); revisadas `cronos-v3`, `perfil-compuerta` (el núcleo
  kpc sigue sin veredicto) y `contenido-materia-lectura-operativa`.
  Nota I §3.19. Registro: 53 claims. Lo que NO afirma: ni núcleo ni
  contracción convergida; ninguna predicción del modelo operativo para
  el interior de las enanas sale de esta ronda.

- **Ronda de registro del 17-sep: la amplitud de Cronos es UNA
  (A_Sculptor) y cuatro huecos reciben nombre**. Hallazgo de
  consistencia del autor (sesión de verificación), reproducido en el
  repo (`dynamics/cronos_amplitude_validity.py`,
  `results/2026-09-17_cronos_amplitude_validity/`): (α₀⁻¹, ρ_c) entran
  en toda la dinámica solo por A = α₀⁻¹/ρ_c^(3/2), y el repositorio
  llevaba dos cierres incompatibles — el galáctico A_Sculptor = 6.2e-7
  (5E, congelado) y el cosmológico ρ_c = 200·ρ̄_m de µ/η (A ≈ 41.5;
  elección de la nota de teoría del 13-sep) — que difieren en 6.7e7.
  Dentro de un halo NFW de 1e11 M☉ el cosmológico viola la
  subdominancia c²ε_c ≲ |Φ_N| de la que nace (11.5) en todo el halo
  (D = 5.4 en r200, 1e4 en r_s, 3e5 en 2.3 kpc; y por debajo de ~r_s
  rompe también el régimen débil: ε_c = 2.2 en 0.38 kpc) — EXCLUIDO. El
  galáctico la cumple EN POTENCIAL en r ≥ 0.1 kpc y coincide con
  A_max = 1.3·A_S; en un enano de 1e10 M☉ (suavizado 50 pc) deja de
  ser subdominante bajo 0.14 kpc. Corrección de la sesión de código
  (piloto del Nivel A): la subdominancia en potencial NO es
  subdominancia en fuerza — en la cúspide D_F = |c²dε_c/dr|/g_N es
  20–600 veces D_Φ y con A_S vale 0.73 en 1 kpc, 7.4 en 0.4 kpc, 230 en
  0.1 kpc (D_F = 1 en ≈ 0.9 kpc): +c²∇ε_c domina la gravedad dentro de
  ~0.9 kpc del halo de 1e11 M☉, coherente con que A_S se calibró en
  5A/5B para sustituir a la materia oscura en Sculptor con bariones
  solos. Consecuencias: (i) la amplitud operativa es
  A_Sculptor, reutilizada entre frentes; (ii) el sector µ/η de Cronos
  (E1) se computó con el cierre excluido — sus identidades siguen
  exactas para el cierre que declaran (artefacto intacto), pero con
  A_Sculptor ε̄_c(hoy) = 5.3e-18 y µ − 1 = 5.8e-12 en k = 0.2 h/Mpc: la
  cola k² muere por consistencia interna y la cota k² sobre full-shape
  sale de la hoja de ruta; (iii) `mu_eta_cronos` conserva el 200 solo
  para reproducir E1 y documenta el cierre superado. Filas nuevas:
  `cronos-amplitud-unica` (interno), `c-lieb-robinson-identificacion`
  (claim-no-derivado: c ≡ v_LR no tenía fila), `mapa-s-z-convencion`
  (calibrado: S_today = 95, α = 1, declarado en código y ahora en el
  registro), `diccionario-primordial-cosmologico` (hueco-declarado: no
  hay fórmula de (δ₀, m̄, b̄, C₀, ē, f) a (Ω_id,0, ε_Λ, z_trans, α_n,
  κ_lat, η_lat) ni Λ_ten desde V₀), `contenido-materia-lectura-operativa`
  (condicional: el repo es ΛCDM + canales + Cronos; la lectura fuerte sin
  CDM es hipótesis distinta, no implementada). Filas revisadas:
  `mu-eta-cronos` (cierre superado, amplitud ~1e-12),
  `perfil-compuerta` (el medio paso del 2-ago es INDETERMINADO POR
  RESOLUCIÓN, no «cúspide observada»; y usó el cierre excluido),
  `cronos-v3` y `5e-falsacion-cruzada` (amplitud única), `canales-oscuros`
  (fondo con ρ_cdm; mapa S(z) por convención). Sin datos; cálculo de
  consistencia (E8). Registro: 52 claims.

- **Cierre gauge-invariante de las colas del sector Atlas (15-sep,
  noche, E5_Atlas): η_N ≡ 1 — la cola de η que E4 halló viva era de
  GAUGE UNITARIO; la única cola observable (µ_Δ) es inobservable en la
  cota PPN. E5a = B (la forma cerrada preinscrita falla en α_a ≪ λ_K −
  1: la cola es de primer orden en α_a, no de segundo); E5b = A (primera
  confirmación numérica de la escalera, en invariantes)**. Corrección
  del autor (verificación del 15-sep) que alcanza a E3 y E4: los seis
  campos de la escalera son de gauge unitario y la reparametrización
  temporal al gauge newtoniano (ψ → ψ − Ṫ, b → b + T, φ → φ + HT, δ →
  δ + 3HT, l1 → l1 + mT; derivada de δg → δg − L_ξ ḡ sobre el ansatz
  del módulo) actúa al MISMO orden e²: b arranca en n = 1 y ḃ, Hb caen
  en n = 4. Preinscripción congelada ANTES de ejecutar
  (`e81883295dd2`, en un commit que contiene su generador) con el
  control externo del autor declarado (valores calculados fuera del
  repo antes de la congelación; no fijó umbrales). Invariantes Ψ_N = ψ
  + ḃ, Φ_N = φ − Hb, Δ = δ − 3H l1. **E5a**: en los nueve puntos el
  coeficiente de e² de η_N − 1 es exactamente 0 y η_N,0 = 1; identidad
  de torre Φ_N,n = Ψ_N,n hasta n = 8 (prof 16) en (1.05, 0.3) y (1.012,
  1e-6): teorema «sin estrés anisótropo lineal (a_i a^i cuadrático en
  ∂N; (1−λ_K)K² pura traza) ⟹ η_N ≡ 1» fijado; el «µ_loc·η = 1 +
  O(e⁴)» de E4 era la firma de la reparametrización. La cola de µ_Δ
  (respecto de la G local) sigue −α_a e²/c_s² en α_a ≳ λ_K − 1
  (cocientes 1.0002 y 1.025 en α_a = λ_K − 1; 0.52/0.38 en α_a = 0.3)
  pero NO en α_a ≪ λ_K − 1, donde vale −0.752/−0.774/−0.852·α_a·e² en
  λ_K = 1.001/1.012/1.05 (PRIMER orden en α_a; cocientes 751–39648) →
  **B** por la regla, publicado sin retocar; en la cota PPN (1.012,
  1e-6): µ_Δ − 1 = −2.2×10⁻¹⁰ en k = 0.02 h/Mpc, inobservable (umbral
  1e-5 cumplido). **E5b** (arnés adiabático en invariantes; brazo 66.7
  retirado y ventana a ≥ 0.3, calibrados sobre las pendientes
  UNITARIAS publicadas): max|η_N − 1| = 2.5×10⁻¹⁴ sobre la trayectoria
  en los cuatro brazos (k/H0 = 133–400) y s_µΔ/escalera =
  1.026/0.944/0.976/0.986 → **A**: la escalera queda confirmada
  numéricamente (el control unitario da 0.98–1.01 en η y 0.96–1.02 en
  µ_loc). Consecuencias: la lectura física de E4a = B («cola de η viva
  en la cota») queda SUPERADA — el registro se conserva bajo su
  definición preinscrita —; E3b = C y E4b = C permanecen (arneses en
  gauge unitario); «cancelación de Atlas» en pie y «colas inobservables»
  se restaura con más fuerza: η_N ≡ 1 en el sector Atlas y la firma
  sub-horizonte de (µ, η) es solo la del canal Cronos. Lo que el control
  externo NO anticipó: el orden α_a¹ de µ_Δ en α_a ≪ λ_K − 1 (sus puntos
  pequeños tenían α_a = λ_K − 1); sin forma cerrada aquí. Módulo:
  mu_delta_tail_leading (con su régimen de validez); fila nueva
  `atlas-gauge-invariant`, filas `atlas-ppn` (status → interno,
  observable corregido) y `mu-eta-atlas` actualizadas; ATLAS_STATUS de
  dos módulos y docstrings de E3 marcados «gauge unitario». Sin datos.
  Estatuto: derivación interna (E8)
  (`results/2026-09-15_mu_eta_atlas_gauge/`).

- **Verificación del autor de #21 (BBN-G, 15-sep)**: las dos
  transcripciones observacionales (EMPRESS XV Y_P = 0.2402 ± 0.0040,
  arXiv 2506.24050; Cooke+2018 D/H = (2.527 ± 0.030)×10⁻⁵, arXiv
  1710.11129) comprobadas exactas contra los resúmenes primarios y las
  cifras reproducidas de forma independiente (solo Y_P −7.6 ± 4.5 %
  frente a −7.2 ± 4.3 %; pulls −1.67σ/+1.26σ idénticos). El
  «provisional» se levanta en cuanto a valores publicados; los bytes
  oficiales siguen sin ingerir (`official_bytes_verified: false` en el
  dataset, aviso en manifest y artefacto). Fila `bbn-g-empress`
  actualizada; artefacto intacto.

- **PPN del sector Atlas (15-sep, E4_Atlas): α_a ≤ 8×10⁻⁷ desde el giro
  solar y los Residuos en forma limpia — pero la cola O(e²) de η NO se
  apaga: E4a = B (hallazgo estructural); E4b = C (el cierre de E3b sigue
  abierto)**. Preinscripción congelada ANTES de ejecutar
  (`39b9f56b8eae`, en un commit que contiene su generador; una
  congelación previa inválida y su corrida se deshicieron y quedan
  declaradas en el commit 2/3). α₁, α₂ de Einstein-aether (Foster &
  Jacobson 2006) en el límite khronométrico c_ω → ∞, con el mapeo al
  sector del tratado α_a = α, λ_K = 1 + λ, ξ = 1 ⇔ β = 0 verificado en
  diez identidades (c_s² y G_cosmo/G_N de #18 reproducidos):
  **α₁ = −4α_a, α₂ = −α_a/2 + O(α_a²)**. Cotas de campo débil (LLR
  |α₁| < 1e-4, giro solar |α₂| < 4e-7; TRANSCRITAS, con aviso de
  procedencia): **α_a ≤ 8.0×10⁻⁷** (gobierna α₂; de α₁: 2.5e-5), casi
  independiente de λ; la corrección −α_a/2 a G_cosmo/G_N − 1 es 2×10⁻⁵
  relativa a −(3/2)ε_K: los Residuos quedan en forma limpia. PERO la
  escalera exacta en (λ_K, α_a) = (1.012, 1e-6) da un coeficiente
  COMPLETO de e² en η − 1 de **4.58**, frente a 3.5×10⁻⁸ del residuo del
  polo: la cola no se apaga con α_a → 0 (4.58/4.59/4.71 en α_a =
  1e-6/1e-4/1e-3; 4.51/4.58/4.84 en λ_K = 1.001/1.012/1.05 a α_a =
  1e-6) y µ_loc − 1 = −(η − 1) a cuatro cifras, es decir µ_loc·η = 1 +
  O(e⁴): la cola vive en Ψ y Φ obedece Poisson a O(e²). Magnitud
  (z = 0, e = (H0/c)/k): η − 1 = 1.3×10⁻³ en k = 0.02 h/Mpc, 2.0×10⁻⁴
  en 0.05, 5×10⁻⁵ en 0.1 — alcanza el umbral preinscrito de
  observabilidad (1e-3) → **E4a = B**. La expectativa del mapa («colas
  de sonido inobservables, η − 1 ~ 3×10⁻⁸») usaba solo el residuo del
  polo, que en α_a ≪ λ_K − 1 no es el término dominante (la expansión
  P/(λ−1) + Q de E3 no aplica ahí). CONDICIONADO: el hallazgo descansa
  en la escalera, cuya confirmación numérica independiente sigue
  abierta (E3b/E4b = C); el orden de límites (α_a → 0 con c_s → ∞ y
  después λ_K → 1) no restaura η = 1 de GR — estructura de la clase
  khronométrica que queda declarada, no explicada. E4b (E3b
  re-preinscrito con ajuste e² + e⁴): k/H0 = 133.3 y 200 pasan (s_η
  ×0.945/0.973, s_µ ×0.902/0.952), pero k = 66.7 (e ∈ [0.01, 0.02]) no
  separa e² de e⁴ (s_η ×0.82, s_µ ×0.69) → **C** por la regla; no se
  retoca (lección para una preinscripción futura: ese brazo no sirve
  para un ajuste de dos parámetros). Módulo: ppn_alpha1, ppn_alpha2,
  alpha_a_max_from_ppn; fila `mu-eta-atlas` actualizada (PPN ingerido)
  y fila nueva `atlas-ppn`. Sin datos observacionales nuevos; Λ_sc ~
  M_P√α_a ~ 1e-3 M_P en la cota (H.2.2). Estatuto: derivación interna
  (E8) (`results/2026-09-15_mu_eta_atlas_ppn/`).

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
