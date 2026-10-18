# Matriz compacta de claims del programa MCMC

<!-- GENERADO por scripts/make_claim_matrix.py desde docs/claims_registry.yaml y la tabla de puentes — NO editar a mano: tests/test_claim_matrix.py falla si este fichero difiere de la regeneración. -->

Supuestos, derivaciones completadas, calibraciones, falsadores y huecos, en una página. Cada celda se genera desde el registro canónico (`docs/claims_registry.yaml`) y la tabla de puentes del diccionario (`docs/diccionario_puentes.md`): lo que se muestra es el registro, no una narrativa. Los estatutos siguen la fe de erratas epistemológica FE-08-2026: «interno» es una comprobación interna superada (E8), nunca demostración física.

Fuente: `docs/claims_registry.yaml`, sha256 `6a6f7d53f5c7b818803b9b951fbd5f8025a7e6869527e63101f2f07d94adb0d5`; 76 filas.

## 1. Matriz por categoría

| Categoría | Filas | Claims |
|---|---|---|
| Derivaciones y comprobaciones internas superadas (E8: comprobación interna, no demostración física) | 45 | `axiomas`, `constantes-estatuto`, `plano-dual`, `potencial-basal`, `monotonia-exclusion`, `florencia`, `rp-escalar`, `beta-un-lazo-sextico`, `residuos`, `naturalidad-priors`, `suite-apendice-h`, `jax-crosscheck`, `fondo-cosmologico`, `cmb-fsigma8`, `canales-oscuros`, `mass-gap-reticulo`, `qudit-decoherencia`, `qudit-d5`, `dovekie-mocks-pipeline`, `perturbaciones-fsigma8-oos`, `mu-eta-atlas`, `dovekie-real-primera-aplicacion`, `bbn-g-empress`, `atlas-ppn`, `atlas-gauge-invariant`, `cronos-amplitud-unica`, `phi-ten-fondo-ley-cronos`, `diccionario-epsilon-c-saturante`, `criterio-cronos-jeans`, `test-criterio-cronos-jeans`, `test-criterio-cronos-jeans-ronda2`, `criterio-cronos-jeans-ronda3`, `criterio-cronos-jeans-ronda3b`, `halo-serie-resolucion-campo`, `nucleacion-salida-s0`, `nucleacion-gamma0`, `diagonal-j-requerido`, `vacio-2d-conversion-y-rotacion`, `halo-capas-ronda2`, `cualificacion-laminas-e8q`, `cualificacion-capas-e8q`, `halo-capas-ronda3`, `lecturas-rondas3-3b`, `s-clock-simulador-consistencia`, `nivel-a-halo-aislado` |
| Supuestos: resultados condicionales a hipótesis declaradas | 16 | `rp-no-estacionaria`, `kls-flujo`, `matriz-estabilidad`, `gea-newton-atlas`, `ciclo-victoria`, `circulo-delta0`, `fertilidad`, `cronos-v3`, `ciclo-pm`, `perfil-compuerta`, `jeans-dsph`, `5c-estructural`, `empalme-wkb`, `mu-eta-cronos`, `contenido-materia-lectura-operativa`, `oort-kz-cota-cronos` |
| Calibraciones: números ajustados o anclados fuera, con estatuto declarado | 4 | `funcional-masas`, `higgs-auditoria`, `wkb-calibrado`, `mapa-s-z-convencion` |
| Preinscritos: protocolos congelados todavía sin desenlace | 0 | — |
| Resultados negativos: experimentos ejecutados con desenlace nulo o contrario, publicados como tales | 4 | `beta-extra-cribado`, `fp-beta-desenlace-a`, `desi-dr2-6a`, `ajustes-produccion` |
| Huecos: huecos declarados, afirmaciones sin derivación y experimentos sin ejecutar (datos ausentes, fallo cerrado) | 7 | `decada-discriminante`, `5e-falsacion-cruzada`, `ngen-3`, `c-lieb-robinson-identificacion`, `diccionario-primordial-cosmologico`, `diccionario-unidad-S-post-florencia`, `sculptor-perfil-sigma-los` |

## 2. Puentes del diccionario primordial → cosmológico

Estatuto de cada puente leído de sus filas (`scripts/make_dictionary_bridges.py`). Huecos distintos entre los puentes ausentes: **1**.

| Estatuto | Puentes | Cuáles |
|---|---|---|
| derivado | 2 | f (fracción descargada) y T₀; Φ_ten (Florencia, fondo) |
| condicional | 3 | δ₀; Φ_ten (perturbaciones); ε_K = λ_K − 1 (Residuos) |
| calibrado | 6 | ξ; λ_K, α_a; A (amplitud de Cronos) y ρ*; Ω_id,0, ε_Λ, z_trans; α_n, S_n^post (escalones); κ_lat, η_lat |
| convencional | 2 | T(S) = dt_rel/dS; S_hoy |
| ausente | 2 | S_post (unidad de S tras Florencia); C(S) = d ln a/dS |

## 3. Falsadores

Lo que refutaría o acotaría cada claim, tal como lo declara el registro, agrupado por categoría.

### Derivaciones y comprobaciones internas superadas (E8: comprobación interna, no demostración física)

| Claim | Estatuto | Falsador | Artefacto |
|---|---|---|---|
| `axiomas` | interno | inconsistencia interna entre axiomas y cualquier bloque derivado | — |
| `constantes-estatuto` | interno | uso de una constante fuera de su estatuto declarado | — |
| `plano-dual` | interno | fallo de las identidades del plano en la suite | — |
| `potencial-basal` | interno | exponente medido incompatible con 3 | — |
| `monotonia-exclusion` | interno | contraejemplo numérico de la monotonía | — |
| `florencia` | interno | firma distinta de −+++ en la cadena | — |
| `rp-escalar` | interno | violación de RP en el juguete | — |
| `beta-un-lazo-sextico` | interno | un error en el teorema de signo (un m̃² con F < 0 para R > R_c) o en las β frente al lado derecho exacto lo refutaría; la conclusión no cubre formas iniciales con R₀ < 25/12, truncamientos más altos, el sector de la inclinación η·χ ni esquemas con anomalía distinta de la LPA | results/2026-10-18_beta_one_loop_sextic |
| `residuos` | interno | residuo fuera de la banda declarada; exclusión del −1.8 % por BBN al 95 % | results/2026-09-14_bbn_g |
| `naturalidad-priors` | interno | cota violada bajo priors alternativos declarados | results/2026-08-04_landscape_priors |
| `suite-apendice-h` | interno | cualquier comprobación del Ap. H en rojo | — |
| `jax-crosscheck` | interno | puertas de equivalencia violadas al re-ejecutar | results/2026-08-19_jax_crosscheck |
| `fondo-cosmologico` | interno | H(0) ≠ H0 o pérdida del límite ΛCDM exacto | — |
| `cmb-fsigma8` | interno | incompatibilidad de la geometría comprimida con el fondo ajustado | — |
| `canales-oscuros` | interno | — | — |
| `mass-gap-reticulo` | interno | piso espectral que desaparezca al refinar el retículo | — |
| `qudit-decoherencia` | interno | — | — |
| `qudit-d5` | interno | — | — |
| `dovekie-mocks-pipeline` | interno | cualquier puerta fuera de su cota congelada habría mantenido vedado el HD real; el ajuste real es trabajo posterior | results/2026-09-12_dovekie_mocks |
| `perturbaciones-fsigma8-oos` | interno | una desviación de R(z) mayor que la envolvente del prior habría sido error de implementación o física nueva, a discriminar por el procedimiento enumerado | results/2026-09-12_perturbations_fsigma8 |
| `mu-eta-atlas` | interno | una firma sub-horizonte escala-independiente en (µ, η) medida respecto de la G local refutaría la cancelación; cotas PPN de marco preferido INGERIDAS (fila atlas-ppn): α_a ≤ 8e-7; una medida de η_N − 1 ≠ 0 a gran escala (estrés anisótropo) refutaría el cierre gauge-invariante (fila atlas-gauge-invariant); la cola +4.6·(aH/k)² de E4 era de gauge y ya no es falsador | results/2026-09-13_mu_eta_atlas + results/2026-09-13_mu_eta_atlas_tail + results/2026-09-15_mu_eta_atlas_ppn + results/2026-09-15_mu_eta_atlas_gauge |
| `dovekie-real-primera-aplicacion` | interno | la regla preinscrita enumeraba tensión (C: \|ΔΩ_m\| ≥ 3σ oficial o χ²_ν > 1.30) y preferencia (B: ΔBIC ≤ −2 con 0 ∉ CI95(ε)); ninguna se dio. Recalibrar la puerta σ(ε) es materia de una preinscripción futura y distinta, nunca de esta corrida | results/2026-09-13_dovekie_real |
| `bbn-g-empress` | interno | C: −1.8 % ∉ CI95(δ_G) por BBN (exclusión; prohibido reparametrizar ε_K o invocar α_a a posteriori); B: 0 ∉ CI95 (identificación: sospecha de error primero). Ninguno se dio. Recalibrar priors o errores teóricos es materia de una preinscripción futura, nunca de esta corrida | results/2026-09-14_bbn_g |
| `atlas-ppn` | interno | C: alguna identidad del mapeo falla (no se dio); una cota PPN nueva que empujara α_a por debajo del acoplamiento fuerte declarado. La cola +4.6·(aH/k)² ya no es falsador (era de gauge, E5) | results/2026-09-15_mu_eta_atlas_ppn |
| `atlas-gauge-invariant` | interno | C preinscrito (coef η_N ≠ 0 en algún punto) no se dio; una medida de η_N ≠ 1 a gran escala respecto de la G local refutaría el teorema; una derivación que muestre estrés anisótropo lineal en la clase khronométrica con β = 0 lo refutaría desde la teoría. La forma cerrada de µ_Δ en α_a ≪ λ_K − 1 queda abierta (no se retoca la de E5) | results/2026-09-15_mu_eta_atlas_gauge |
| `cronos-amplitud-unica` | interno | una derivación del tratado que fijara ρ_c de forma independiente y compatible con 200·ρ̄_m dentro de halos refutaría la exclusión; una A_Sculptor revisada por 5E con datos oficiales desplazaría la amplitud única. FALSADOR OORT–K_z (añadido el 21-sep, fila criterio-cronos-jeans): la misma A_Sculptor en la vecindad solar predice un límite de Oort efectivo de Cronos ≈ 0.75 M☉/pc³ (medido 0.10 ± 0.01) y g_C/K_z ≈ 2–7 entre 50 y 300 pc; la cota g_C ≤ 0.1·K_z en 300 pc exige A ≤ 0.048·A_Sculptor — A_req(Sculptor)/A_max(K_z) ≳ 20, incompatibilidad interna de la ley local a amplitud única, pendiente de la preinscripción observacional «Oort–K_z» (frente 5 refundado) | results/2026-09-17_cronos_amplitude_validity |
| `phi-ten-fondo-ley-cronos` | interno | un fondo con r ≠ r̄ en algún punto (inhomogéneo) o una definición de r̄ que no sea la media de r sobre la misma hipersuperficie daría N ≠ 1; dentro del fondo FRW homogéneo no hay falsador observacional porque es una identidad de la definición | — |
| `diccionario-epsilon-c-saturante` | interno | con SPARC ingerido, una desviación g_obs − g_bar continua en g_bar (sin escalón localizado) falsa la forma para todo (ε_max, ρ*) de la región permitida; el perfil σ_los(R) de Sculptor decide si la inestabilidad q_S > 1 es real, si σ_los no es la σ del medio, o si A_S no es la fuerza de Sculptor | results/2026-09-22_dictionary_eps_c |
| `criterio-cronos-jeans` | interno | un N-cuerpos con la ley local que mostrara convergencia del ritmo con q > 1 refutaría el criterio; una v_LR o una escala de no localidad derivadas que cortaran k > 1/L lo acotarían; las preinscripciones Oort–K_z (desenlace C si g_C/K_z > 1 donde el modelo pone ×2–7) y perfil de Sculptor (C si el χ² excluye el pico central a > 3σ) deciden la amplitud única contra datos | results/2026-09-21_cronos_jeans |
| `test-criterio-cronos-jeans` | interno | en una ronda resuelta, γ/k fuera de la tolerancia de la predicción cinética (desenlace B) o crecimiento con q < 1 (C) refutarían la derivación del criterio; esta ronda no decide | results/2026-09-21_cj_criterion_test |
| `test-criterio-cronos-jeans-ronda2` | interno | bajo una preinscripción que resuelva la celda (q = 2, n = 4), una tasa fuera del 25 % de γ_cin(q, k)·W(k) o una dispersión entre modos > 25 % falsaría la predicción cinética del criterio; crecimiento con q = 0.8 falsaría el umbral | results/2026-09-22_cj_criterion_round2 |
| `criterio-cronos-jeans-ronda3` | interno | un instrumento independiente (Vlasov euleriano con banda declarada, o láminas con otro filtro cualificado) que, bajo su propia preinscripción, midiera tasas fuera del 25 % de γ_cin(q, k)·W(k) o crecimiento con q = 0.8 falsaría la reproducción; el criterio mismo solo lo deciden los datos (filas oort-kz y sculptor-perfil) | results/2026-09-28_cj_criterion_round3 |
| `criterio-cronos-jeans-ronda3b` | interno | bajo una preinscripción con la banda inestable por debajo de k_c cualificada, una tasa en q = 1.5 fuera del 25 % de γ_cin(q, k)·W(k) o una dispersión entre modos > 25 % contradiría la ronda 3; el criterio mismo solo lo deciden los datos (filas oort-kz y sculptor-perfil) | results/2026-10-18_cj_criterion_round3b_q1p5 |
| `halo-serie-resolucion-campo` | interno | en una ronda con las puertas superadas y el interior resuelto: la convergencia de la serie a A_Sculptor refutaría la lectura ultravioleta del criterio (letra B); la no convergencia también a 0.05·A_Sculptor señalaría el instrumento (C) | results/2026-09-21_halo_resolution |
| `nucleacion-salida-s0` | interno | una derivación del tratado que fije d, el prefactor o D_ent convertiría Γ₀(δ₀) en predicción del ciclo (Teo. 10.6, frente 4); hasta entonces lo refutable son las leyes de escala medidas y el hecho de que, con pared gruesa, la descarga cae dentro de la nucleación — un Basal con barrera ≫ T₀ (otras formas O(1)) devolvería la descarga al Flujo del Camino | results/2026-09-21_nucleation |
| `nucleacion-gamma0` | interno | condicional a n_dim: con n = 1 la ley lineal se rompería si el prefactor no fuera la frecuencia de intento o si B₁ no escalara como δ₀²; con n = 4 si B no divergiera como δ₀⁻¹; la deformación cambiaría el veredicto C si un término no radial del Basal (fuera de −η·χ) favoreciera caminos curvos | results/2026-09-21_nucleation_gamma0 |
| `diagonal-j-requerido` | interno | una derivación de J desde la ontología (frente 2) que no reproduzca \|J\|_min ≈ 1.36–1.45 (con C = V, G = 1, formas por defecto) o que exija otro generador C; un término del programa que mueva el vacío 2D fuera del polo de masa cambiaría la letra F y el S_max | results/2026-09-21_j_circulation |
| `vacio-2d-conversion-y-rotacion` | interno | una preinscripción nueva con d_sigma y max_steps que superen las puertas en toda la rejilla y sitúen el cruce en S = 1 ± 0.05 con alguno de los dos términos (letra A); o una derivación de κ̂ o f_× desde la ontología (frente 2) que fije su valor sin calibración | results/2026-09-22_vacuum_2d |
| `halo-capas-ronda2` | interno | una ronda 3 preinscrita con rango M(<r) actualizado en los subpasos (o suelo de dt menor y presupuesto de pared mayor) que supere las puertas en las seis celdas de Cronos y clasifique con letra; o una forma de ε_c(ρ) del diccionario (saturante) bajo la que la cúspide no salga del régimen débil | results/2026-09-22_halo_shells |
| `cualificacion-laminas-e8q` | interno | una ronda 3 cuya celda excluida a priori resolviera la fase lineal, o cuya celda no excluida fallara por la banda UV, refutaría el criterio de exclusión como predictor del instrumento (E8-Q, no física) | results/2026-09-28_qualification_sheets |
| `cualificacion-capas-e8q` | interno | una ronda 3 cuyos brazos de Cronos cumplieran la puerta newtoniana (6.8e-6) refutaría el suelo medido; una serie que midiera t_weak distinto de 0.15 Myr con dt_max = 0.15 en ε_soft ≤ 0.05 refutaría la cuantización declarada (E8-Q, no física) | results/2026-09-28_qualification_shells |
| `halo-capas-ronda3` | interno | con condiciones iniciales en equilibrio Newton + Cronos, una contracción de M(<r_u) del brazo saturante sub-umbral frente al newtoniano de la misma celda retiraría la candidatura de la forma saturante; t_weak independiente de ε_soft a A_Sculptor (\|p\| ≤ 0.10) bajo una preinscripción con puertas alcanzables retiraría la lectura de la adenda del autor. Reescrito el 18-oct: el falsador anterior («el brazo saturante sub-umbral saliendo del régimen débil») describía un suceso imposible, porque ε_max = 5.41e-9 < 1e-3 | results/2026-09-28_halo_round3 |
| `lecturas-rondas3-3b` | interno | una ronda 3c con el k_c de cada celda fijado por la regla de banda en la que una celda no excluida cerrara su ventana con la banda bajo k_c por encima de A_nl refutaría la regla como predictor del instrumento (E8-Q); un punto saturante viable con q_eq > 1 en r ≥ 3·ε_soft refutaría L2 | results/2026-10-18_lecturas_rondas3_3b |
| `s-clock-simulador-consistencia` | interno | el simulador se vuelve EMERGENTE (y contrastable como tal) solo cuando el frente 2 entregue τ(S) y el diccionario primordial → cosmológico exista; hasta entonces lo refutable son sus identidades internas (S ≡ f, T₀, álgebras, firma) y el hallazgo δ₀_max(m̄, b̄, ē, C₀), que una derivación de las formas O(1) podría desplazar | results/2026-09-21_s_clock |
| `nivel-a-halo-aislado` | interno | una ronda con preinscripción NUEVA (criterio de c′ compatible con la regla de parada; campo sin retardo o τ_avg ≤ t_dyn interior; serie de convergencia k_inner 64/128/256 como puerta; zoom del interior) que dé A, B o C decide lo que esta ronda retiene. Si el interior contraído resultara convergido, el modelo operativo (ΛCDM + Cronos a A_Sculptor) predeciría enanas más concentradas que ΛCDM en ≲ 1 kpc; si resultara un vaciado convergido, predeciría lo contrario: hoy ninguna de las dos es afirmable | results/2026-09-17_cronos_halo_nivelA |

### Supuestos: resultados condicionales a hipótesis declaradas

| Claim | Estatuto | Falsador | Artefacto |
|---|---|---|---|
| `rp-no-estacionaria` | condicional | contraejemplo de suficiencia; Wilson (frente 1) decidiría necesidad | results/2026-08-02_rp_nonstationary |
| `kls-flujo` | condicional | exponente distinto de −1/2 al refinar la integración | results/2026-08-02_kls_flow |
| `matriz-estabilidad` | condicional | espectro incompatible al sustituir el ansatz por derivación | results/2026-08-05_victoria_exponent |
| `gea-newton-atlas` | condicional | ruptura del sello exacto en la suite | — |
| `ciclo-victoria` | condicional | signo de ν contrario al requerido | — |
| `circulo-delta0` | condicional | W_max incompatible con el círculo (W_max ≠ T₀_full(δ_H_full) = 1.869e-4 con las formas fiduciales) | results/2026-08-02_delta0_circle + results/2026-09-21_s_clock |
| `fertilidad` | condicional | γ_max fuera del rango declarado al refinar | results/2026-08-01_fertility |
| `cronos-v3` | condicional | incapacidad de reproducir la compuerta declarada | — |
| `ciclo-pm` | condicional | — | — |
| `perfil-compuerta` | condicional | núcleo de producción incompatible con la compuerta | results/2026-08-02_profile_shape |
| `jeans-dsph` | condicional | σ_los binada incompatible con la predicción del montaje — en concreto, con la forma congelada (pico central ≈ 2× lo observado, exterior ÷3–4; brazo de sensibilidad β ∈ [−0.5, 0.3]); desenlace C si el χ² de la forma excluye el pico a > 3σ (preinscripción «perfil de Sculptor», frente 5 refundado) | results/2026-08-10_jeans_dsph |
| `5c-estructural` | condicional | forma medida incompatible con shape_x5 ≈ 5 | results/2026-08-14_sparc_structural |
| `empalme-wkb` | condicional | δ_H incompatible al refinar el empalme | results/2026-08-01_empalme_wkb |
| `mu-eta-cronos` | condicional | firma cerrada sin parámetros adicionales — µ − 1 = 2(Σ − 1), η < 1, cola k² — pero con amplitud ~1e-12 a la A única del modelo: ningún contraste RSD+lensing previsto la alcanza; una medida de µ − 1 ≠ 0 escala-dependiente a gran escala NO podría atribuirse al canal Cronos con A_Sculptor. Atlas queda falsable vía el mismo ε_K que BBN | results/2026-09-13_mu_eta_cronos + results/2026-09-17_cronos_amplitude_validity |
| `contenido-materia-lectura-operativa` | condicional | un ajuste del fondo o de halos que exigiera Ω_cdm → 0 con Cronos a A_Sculptor refutaría la lectura operativa a favor de la fuerte; hoy no hay tal ajuste. El Nivel A (halo CDM + Cronos a A_Sculptor) puede refutar la operativa si el interior colapsa de forma incompatible con las enanas observadas — ejecutado el 17/18-sep con desenlace INDETERMINADO (fila nivel-a-halo-aislado), sin veredicto sobre la operativa. Derivación del 21-sep (fila criterio-cronos-jeans): con la ley local ε_c = A·ρ^(3/2) y A_Sculptor, el halo NFW de 1e11 M☉ es INESTABLE dentro de r_CJ = 0.71 kpc (q = (3/2)c²ε_c/σ² > 1, tasa ∝ k): la lectura operativa conserva la CDM para las curvas pero desestabiliza los centros y rompe K_z en la vecindad solar | — |
| `oort-kz-cota-cronos` | condicional | condicional a la losa declarada (banda ×4) y a la forma ε_c = A·ρ^{3/2} (diccionario); los valores ya verificados no movieron ninguna letra; la exclusión de A_Sculptor (z ≈ 51) sobrevive a cualquier altura de la losa en la banda (z ≥ 11.9 con h×2) | results/2026-09-21_oort_kz |

### Calibraciones: números ajustados o anclados fuera, con estatuto declarado

| Claim | Estatuto | Falsador | Artefacto |
|---|---|---|---|
| `funcional-masas` | calibrado | incapacidad de acomodar una masa medida dentro del esquema calibrado | — |
| `higgs-auditoria` | calibrado | — | — |
| `wkb-calibrado` | calibrado | — | — |
| `mapa-s-z-convencion` | calibrado | una C(S) derivada que desplace los umbrales post-Florencia fuera de los escalones calibrados obligaría a recalibrar los canales | — |

### Resultados negativos: experimentos ejecutados con desenlace nulo o contrario, publicados como tales

| Claim | Estatuto | Falsador | Artefacto |
|---|---|---|---|
| `beta-extra-cribado` | resultado-negativo | un candidato, o un cierre de proyección derivado del tratado (no declarado), que pase (i)–(ii) en la ventana plausible; entonces la fila decada-discriminante se revisa. El cribado no excluye el término κΣ̇ê_E como término del Camino (vacío 2D): excluye que su contribución a las β, bajo estos cierres, produzca la Década | results/2026-09-28_beta_screening |
| `fp-beta-desenlace-a` | resultado-negativo | la preinscripción enumeraba desenlaces; A se decidió con el espectro del punto preinscrito | results/2026-08-19_front2_fp_beta |
| `desi-dr2-6a` | resultado-negativo | la señal exigiría ΔAIC/ΔBIC pro-MCMC con ε identificada; se observó lo contrario | results/2026-08-18_desi_dr2_background |
| `ajustes-produccion` | resultado-negativo | la ventaja del corpus (A.6) exigía ΔBIC pro-MCMC; se midió lo contrario | results/2026-08-10_production_fit_v2 |

### Huecos: huecos declarados, afirmaciones sin derivación y experimentos sin ejecutar (datos ausentes, fallo cerrado)

| Claim | Estatuto | Falsador | Artefacto |
|---|---|---|---|
| `decada-discriminante` | claim-no-derivado | una derivación de τ(S) que produjera λ ≠ 10 refutaría la selección | results/2026-09-21_s_clock |
| `5e-falsacion-cruzada` | experimento-no-ejecutado | mediana(Δχ²) > 0 con CI95 excluyendo 0 y frac_improved < 1/3 (camino A preinscrito) | results/2026-08-16_front5e_sparc |
| `ngen-3` | hueco-declarado | construcción explícita con Ngen ≠ 3 en la misma álgebra | — |
| `c-lieb-robinson-identificacion` | claim-no-derivado | una v_LR calculada sobre la cadena discretizada distinta de la c_eff del flujo (5.3) refutaría la identificación | — |
| `diccionario-primordial-cosmologico` | hueco-declarado | — (no hay afirmación contrastable hasta que exista la fórmula) | results/2026-09-21_s_clock |
| `diccionario-unidad-S-post-florencia` | hueco-declarado | — (no hay afirmación contrastable hasta que Σ̇_post y T_sellada se deriven) | results/2026-09-21_s_clock |
| `sculptor-perfil-sigma-los` | experimento-no-ejecutado | C si los dos bins internos del perfil binado excluyen el pico central predicho a > 3σ para todo β del brazo (la expectativa declarada, E13: el perfil observado es plano) | results/2026-09-21_sculptor_profile |
