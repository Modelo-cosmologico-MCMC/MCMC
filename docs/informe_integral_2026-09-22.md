# Informe integral: cómo se comporta computacionalmente cada parte del Modelo Cosmológico de Múltiples Colapsos, leído desde su ontología, y qué falta para probar cada una

> **Advertencia de lectura (cabecera añadida al integrar el informe en el repositorio).**
> Este documento es el informe del autor de la sesión de verificación del 22-sep-2026, con
> su adenda de la misma tarde. Es una *lectura* del estado del programa, no una fuente de
> afirmaciones del registro: las palabras interpretativas («refutada», «ciego a la
> conversión», «catástrofe ultravioleta», «no tiene ningún halo cuspidal estable») son del
> informe y **no están en ninguna fila de `docs/claims_registry.yaml`**. Lo citable es siempre
> la fila: donde el informe dice «refutada» sobre la ley débil local a A_Sculptor, el registro
> afirma «excluido» con letra C bajo la preinscripción v2 de Oort–K_z, con
> `official_bytes_verified: false` (fila `oort-kz-cota-cronos`). Los datos de cabecera
> (`main`, número de claims y de artefactos) corresponden al momento de escribir y a la
> adenda; el estado actual lo da el propio registro. Al integrarlo se han hecho tres cambios
> mecánicos: la fórmula del E13 sobre el signo se escribe «un signo que sale no es una
> señal», porque la guarda de honestidad del CI rechaza la formulación original; se añade el anexo de citas a filas (§17); y se
> añade la nota de verificación de la sesión de código sobre dos números de la adenda (§18).

**Sesión de verificación, 22-sep-2026.** Base verificada al escribir: `main` = `aa927cb` (PR #39 fusionado; #35–#39 de hoy incluidos). **Adenda al final (misma tarde): `main` = `330b2e4`, 67 claims, 42 artefactos, §3.34; tres correcciones de la sesión de código aceptadas — en particular, donde este informe dice «refutada» el registro afirma «excluida bajo regla congelada (letra C) con bytes no descargados», y esa es la formulación citable.** Suite local sobre el estado anterior (`41af270`): 601 passed + 2 skipped; registro: **66 claims** (36 internos, 16 condicionales, 4 calibrados, 3 resultados negativos, 3 huecos declarados, 2 claims no derivados, 2 experimentos no ejecutados). Artefactos: 41 directorios en `results/`. Nota computacional I: §3.1–§3.33. En curso, sin fusionar: la ronda 2 de la serie del halo con el código de capas esféricas (diez brazos corriendo bajo preinscripción congelada `1d6682f968fa…`).

**Cómo está escrito este informe.** Para cada parte del modelo sigo el mismo orden: qué afirma la ontología (el texto del tratado, con su axioma), cómo está realizada en el código, qué muestra el cómputo (con los números), si está bien desarrollada (verificación independiente, correcciones históricas, defectos), y qué haría falta para *probarla* — con el vocabulario del programa: derivado ≠ condicional ≠ calibrado ≠ convención ≠ claim no derivado ≠ experimento no ejecutado; comprobación interna (E8) ≠ demostración física; un signo que sale no es una señal (E13). «Probar una parte» tiene aquí un sentido preciso: que exista una predicción de esa parte, congelada antes de mirar, contrastada con un instrumento cuya validez se haya demostrado aparte, y que el desenlace sea una letra y no una lectura. Con ese criterio, al final del informe digo qué partes están probadas, cuáles solo son consistentes, cuáles están calibradas y cuáles han sido refutadas en su forma actual.

---

## 0. El mapa: axioma → realización → módulo → artefacto → estatuto

La Tabla 1.1 del tratado enlaza cada axioma con su realización matemática y con el resultado que lo sostiene. El repositorio realiza esa tabla módulo a módulo; lo que sigue es el mismo mapa con la columna que el tratado no tiene: el estatuto computacional hoy.

| Axioma | Realización (tratado) | Módulos | Artefactos decisivos | Estatuto computacional |
|---|---|---|---|---|
| 1 Unidad dual | Plano Dual, Fibrado Dual (Cap. 2) | `core/dual_plane`, `florencia.chain_generators` | suite Ap. H | interno demostrado (identidades) |
| 2 Imperfección | escalado canónico de δ₀; estructura de vacíos (Cap. 3) | `core/basal`, `landscape_priors`, `s_clock.radial_landscape` | 2026-08-04_landscape_priors, 2026-09-21_s_clock | derivado + hallazgos nuevos (δ₀_max, corrección √δ₀) |
| 3 Tensión | Potencial Basal; T₀ = c̄δ₀³ (Cap. 3) | `core/basal`, `nucleation` | 2026-09-21_nucleation(_gamma0) | derivado; Γ₀ condicional a D_ent |
| 4 Camino | Flujo del Camino; Monotonía; Exclusión (Cap. 4) | `core/path_flow`, `s_clock` | 2026-09-21_j_circulation, 2026-09-22_vacuum_2d | derivado (radial); transporte azimutal **no producido** |
| 5 Sellos | puntos fijos β_i = 0; Década; Discriminante (Cap. 8) | `decade`, `kls_flow`, `fokker_planck_beta`, `victoria_exponent` | 2026-08-02_kls_flow, 08-05_victoria_exponent, 08-19_front2_fp_beta | λ = 10 **calibrado**; colapsos **impuestos**; cierre canónico no produce D = 0 |
| 6 Atemporalidad | Cadena de Álgebras; Z real; elipticidad (Cap. 5) | `florencia`, `lattice/*`, `mcmc_ontology/clifford_algebra` | — (suite) | interno demostrado; sellos de c y c² con **forma declarada** |
| 7 Cronos | γ⁰ = iγ_S; MDR; lapse N (Cap. 6–7, 11) | `florencia`, `reflection_positivity`, `rp_nonstationary`, `dynamics/weak_field`, `cronos/*` | 08-02_rp_nonstationary; 09-17…09-22 (Cronos débil) | firma derivada; RP juguete; **ley débil refutada en el plano a A_Sculptor** (lectura; el registro dice «excluido», letra C); forma abierta |
| 8 Victoria | híbrido flujo⊕reinicio; F_Vic (Cap. 10) | `victoria`, `fertility_map`, `delta0_circle` | 08-01_fertility, 08-02_delta0_circle | condicional (ν, W_max) |
| Gea/Atlas (Cap. 9) | Sello de Newton; Atlas; Residuos | `core/gea`, `cosmology/mu_eta_atlas` | 09-13…09-15 Atlas (E1–E5) | derivado al orden observable; ΛCDM-lineal |
| Masas (Cap. 12) | funcional de masas; empalme | `mass_program/*` | 08-01_empalme_wkb | calibrado; δ_H medido; no predicción |
| Cosmología (Ap. A) | Friedmann tensional; canales; S(z) | `cosmology/*`, `mcmc_ontology/S_map` | 08-10_production_fit, 08-18_desi, 09-12/13 Dovekie, 09-14_bbn | casi-ΛCDM; negativos publicados; S(z) convención |
| Estructura (Ap. B) | Cronos v3 en N-cuerpos | `cronos/*`, `halo_nbody`, `halo_shells` | 08-02_profile_shape, 09-17_nivelA, 09-21_halo_resolution | prototipos; Nivel A/serie INDETERMINADOS; instrumento en construcción |
| Reloj S (C1) | trayectoria única S₀ → S₁,₀₀₁ | `core/s_clock`, `viewer/` | 09-21_s_clock | simulador de consistencia, no emergente |
| Diccionario (frente nuevo) | primordial → cosmológico | `dynamics/epsilon_c_saturating`, `core/s_post_unit` | 09-22_dictionary_eps_c | ecuación 1 acotada; 2–3 declaradas |

---

## 1. Axioma 1 — la unidad dual: el Plano Dual y el Fibrado

**Ontología.** «Contenido» (Mp) y «continente» (Ep) son dos proyecciones de un mismo valor en una fibra interna R² con coordenadas (φ_M, φ_E); la fase de conversión θ = arctan(φ_E/φ_M) mide «cuánto de lo condensado se ha vuelto apertura»; el modo de desequilibrio χ = (φ_M − φ_E)/√2 «siente la inclinación»; la diagonal θ = π/4 es el lugar fijo de la reflexión Mp ↔ Ep, «el lugar de equilibrio Mp = Ep». El dominio físico es el primer cuadrante (C4). El Lema de Ortogonalidad Dimensional separa la dimensión de la fibra (2) de la del soporte (0…3): la dualidad existe sobre un punto.

**Realización.** `core/dual_plane.py` implementa (2.1) y la reflexión dual; `to_dual`, `on_diagonal`, `is_physical` son las utilidades que consumen el reloj y los experimentos del vacío 2D. La cadena de álgebras C(d+1, 0) sobre el soporte (`florencia.chain_generators(d)`) se construye independientemente de la fibra, y el reloj lo comprueba en cada colapso (anticonmutación, firma euclidiana, dimensión de la representación): la Ortogonalidad Dimensional está realizada como hecho de código.

**Comportamiento.** Nada que decidir: son identidades y se cumplen a precisión de máquina en toda la suite. Lo interesante es lo que la fibra hace *bajo la dinámica*, y eso pertenece al Axioma 4 (§3): el cómputo muestra que el descenso ocurre en ρ y que θ apenas se mueve.

**Desarrollo.** Correcto y cerrado. **Prueba.** No es una parte contrastable por sí misma: es el escenario. Su única consecuencia observable es a través de Prop. 3.5 (transporte azimutal), que hoy no se produce (§3).

---

## 2. Axiomas 2 y 3 — la imperfección y la tensión: el Potencial Basal y la salida de S₀

**Ontología.** El estado perfecto (δ₀ = 0) es un pozo plano hasta sexto orden, «simétrico, inerte y sin nada que empuje fuera»; cualquier imperfección δ₀ > 0 que cumpla la casi-cancelación b̄² > (16/3)C₀m̄² abre un vacío verdadero más hondo y convierte el origen en falso vacío; la diferencia es la Tensión Primordial T₀ = c̄δ₀³ (Prop. 3.4), y T₀ = 0 ⟺ δ₀ = 0 (Axioma 3). La inclinación −ηχ, η = ēδ₀³, orienta la primera salida hacia el polo de masa; la salida es nucleación de Coleman con Γ₀(0) = 0 (Prop. 3.5).

**Realización.** `core/basal.py`: V₀(ρ, χ; δ₀) con el escalado canónico (M₀² = m̄²δ₀², B = b̄δ₀, η = ēδ₀³), c̄, κ₊, T₀ analítica y numérica, residuo ⟨χ⟩. `core/s_clock.radial_landscape`: los puntos estacionarios del paisaje **completo** (con inclusión de la inclinación) sobre el corte θ = 0: falso vacío desplazado a ρ_fv ≃ η/(√2M₀²), barrera, vacío verdadero, punto de escape V = V_fv. `core/nucleation.py`: bounce O(d) por disparo (d = 3, 4), túnel 0+1 de Gamow con prefactor ω_fv/2π, y escape de Kramers sobreamortiguado con D_ent declarada. `core/landscape_priors.py`: naturalidad bajo tres priors.

**Comportamiento.** Cinco resultados, en orden de importancia:

1. **La ley T₀ = c̄δ₀³ es exacta sin inclinación (1e-15) y es solo el orden dominante con ella.** Forma cerrada verificada por dos vías: T₀_full = c̄δ₀³[1 + κ₁√δ₀ + O(δ₀)], κ₁ = ē√κ₊/(√2c̄) = 1.361 (formas por defecto); medido +7.4 % (0.003), +13.3 % (0.01), +31 % (δ_H), +40 % (0.1). El tratado llama a la inclinación «subdominante O(δ₀^{7/2})»: lo es paramétricamente, pero en los δ₀ que el programa maneja (0.012–0.058) pesa entre el 15 % y el 31 %. Decisión A (ejecutada, §3.30): el Lema 10.3 y el empalme usan el paisaje completo; δ_H pasa de 0.0581 a **0.0554** y el Techo decidido es W_max = 1.869×10⁻⁴.
2. **La metastabilidad de S₀ tiene un δ₀ máximo.** Con la inclinación, la barrera desaparece para δ₀ > δ₀_max = 2g_max²/ē² = 0.1028·ē⁻² (forma cerrada; verificado ē = 0.5 → 0.411, ē = 2 → 0.0257). La Obs. 8.6 (D(S₀) > 0) es necesaria, no suficiente. Sobre los paisajes viables de los priors F.2, la fracción con S₀ metastable en δ_H es minoritaria (0.21–0.38) y en 0.012 va de 0.51 a 0.74: **la existencia del falso vacío no es genérica** — es una ligadura conjunta (δ₀, ē) que el tratado no enunciaba.
3. **La barrera es somera**: barrera/T₀ = 0.088 (1e-3) → 0.054 (0.01) → 0.010 (δ_H): régimen de pared gruesa; la pared delgada no aplica.
4. **La nucleación depende de la dimensionalidad del instantón, y la ontología la fija.** n = 4: B ∝ δ₀^{−1.03…−1.11}, Γ₀ exponencialmente suprimida — pero el bounce entrega el campo con f₀ = 0.9996: *toda la descarga cabría dentro de la nucleación*; n = 3: B → constante; n = 1 (túnel en σ): B₁ = 0.5δ₀² ≪ 1 y Γ₀ ≈ (m̄δ₀/2π)e^{−B₁} se anula **linealmente por el prefactor** (exponente medido 0.9998/0.9993/0.9978); Kramers (Axioma 4): prefactor ∝ δ₀^{1.85}, barrera ∝ δ₀^{2.44}/D_ent. Decisión B (ejecutada): n = 1 con Kramers y Gamow como cota; O(3)/O(4) son controles negativos porque en S₀ no hay soporte espacial. Lo que queda abierto es D_ent: con D_ent/ΔV_b = 0.1 / 1 / 10 la espera σ_nuc va de 1.2×10⁹ a 6×10⁴ (δ₀ = 0.01) — **la salida de S₀ es exponencialmente sensible a una constante que el diccionario no da**.
5. **El camino de túnel no se curva**: la optimización de θ(s) devuelve el rayo θ = 0 con B_min/B_ray = 1.0000 en las seis (δ₀, ē): la salida hacia el polo de masa (Prop. 3.5) es un hecho del paisaje, no una elección.

**Desarrollo.** Correcto y verificado de forma independiente (mis cálculos reproducen los cinco puntos; el único defecto — la raíz del origen sin inclinación — está corregido en v1). La Prop. 3.4 debe reenunciarse en v36 con su corrección; el Lema 10.3 con T₀_full.

**Prueba.** El Axioma 3 no tiene observable propio; se prueba por consistencia (T₀ → 0 con δ₀, Γ₀(0) = 0) y las dos cosas están hechas. Lo contrastable es indirecto: el valor de δ₀ (vía δ_H del empalme y δ₀* de Victoria) y el hecho de que el paisaje F.2 no garantice la metastabilidad: si el frente 4 fija W_max y con él δ₀*, la ligadura ē < 0.32/√δ₀* es una **predicción sobre las formas O(1)** que la Ruta L del Maestro (espumas de espín) debería reproducir. Y D_ent es la primera constante que el frente 2 debe entregar.

---

## 3. Axioma 4 — el Camino: el Flujo del Camino, la monotonía y el transporte azimutal que no ocurre

**Ontología.** «La dinámica fundamental no es una fuerza aplicada sobre el Campo, sino el descenso por la vía de mínima resistencia», dΦ_Ad/dσ = −G⁻¹(S)δV/δΦ_Ad con G ≻ 0 (4.2); S = ∫Σ̇dσ es monótona (Teo. 4.5); Exclusión (4.7); completación estocástica (Def. 4.4). Y la lectura del Camino: «la primera salida es hacia el polo de masa: el ego se constituye antes, y solo después viene el transporte azimutal hacia la apertura — la conversión paulatina de masa en espacio que cruza el Centro» (Prop. 3.5: θ crece monótonamente y cruza π/4 en S ≃ 1).

**Realización.** `core/path_flow.py` (flujo de gradiente disipativo; tests de Monotonía y Exclusión); en el reloj, el flujo **proyectado** sobre el dominio físico (en la frontera φ_i = 0 con velocidad saliente la componente se anula y la ligadura no trabaja) integra Φ_Ad(σ) desde el punto de escape; el contador de descarga f = (V_fv − V)/T₀ y S = ∫Σ̇dσ/T₀ con normalización declarada «S = 1 ⟺ descarga completa». Los tres términos candidatos para el transporte azimutal están implementados como brazos: circulación J∇C (C = V y rotación rígida), corriente de conversión κΣ̇ê_E y rotación de la inclinación η_eff = η₀(1 − f/f_×).

**Comportamiento.**

- **Lo derivado se cumple**: Monotonía (max ΔV/T₀ = −4.7×10⁻⁵), producción entrópica ≥ 0, Exclusión, salida al polo de masa (θ_final = 0.000), y la identidad **S ≡ f** a 2×10⁻¹³ — que es un teorema del flujo de gradiente (Σ̇ = −dV/dσ) y no una hipótesis. Descenso completo en σ̂ = σδ₀² = 0.83 (δ₀ = 0.01), 1.32 (δ_H), 5.6 (0.1: la barrera somera hace lento el arranque).
- **El transporte azimutal no ocurre en el Basal**: el flujo de gradiente deja θ = 0 y el reloj impone θ_imp(S) para la entrega. Los tres mecanismos ensayados bajo preinscripción coinciden en lo esencial: J∇C con C = V cruza solo si |J| ≥ |J|_min = 1.36–1.45 y **vuelve** al polo, con S_max del cruce = 0.96 (δ₀ = 0.001) → 0.75 (δ_H); la corriente de conversión cruza con κ̂ ≥ 1.65–2.11 y también vuelve, con S_cross ≤ 0.87; la rotación de la inclinación cruza para f_× ≤ 0.84 **en el mismo S que el control J∇C** (0.931/0.880/0.807/0.753) y para f_× ≤ 0.72 el campo pasa al polo de espacio y se queda. Veredictos: MIXTO (§3.25) e INDETERMINADO en los dos brazos (§3.33). La frase que resume los tres experimentos es de la sesión de código y es exacta: «el S del cruce lo fija la descarga f(σ) — cuándo θ llega a π/4 depende de cuánto ha bajado el campo, no del término».

**Lo que esto enseña sobre la ontología** (el punto central de este informe). En el Potencial Basal la tensión se descarga en ρ («cuánta realidad se ha desprendido del punto simétrico»), y la conversión Mp → Ep vive en θ, donde el único término es la inclinación. Dos números lo cuantifican: la rigidez angular frente a la radial, m_θ²/V''_ρ = (η/ρ₊)/V''(ρ₊) = 2.8×10⁻³ (δ₀ = 0.003), 5.1×10⁻³ (0.01), 1.2×10⁻² (δ_H) — el modo θ es paramétricamente blando (Prop. 10.2: O(δ₀^{5/2}), la memoria de Victoria) y en un flujo sobreamortiguado eso significa **cientos de veces más lento** que el descenso radial; y la energía del recorrido angular completo, 2ηρ₊/T₀ = 0.20 / 0.34 / 0.70 en los mismos δ₀ — que no es despreciable, y hace del polo de masa el mínimo global por un margen de ≈ 0.12–0.25 T₀ frente a la diagonal. Consecuencia: el reloj S, que cuenta tensión liberada, **es ciego a la conversión**, y cualquier término que empuje θ durante la descarga se apaga con ella y el campo vuelve al polo; cualquier término que invierta la inclinación lleva el campo al polo de espacio, porque la diagonal no es mínimo de nada. La Prop. 3.5 («cruza la diagonal en S ≃ 1») y la Def. 2.2 («lugar de equilibrio Mp = Ep») son afirmaciones de simetría y de narrativa que el Basal + Camino, tal como están escritos, no realizan: **la ontología dice que la tensión es el desequilibrio Mp/Ep; la matemática pone la tensión en ρ y deja Mp/Ep como fase libre.** Eso no es un fallo de código: es el hallazgo más importante que el reloj ha producido, y es una decisión del tratado, no del repositorio: o bien θ_imp(S) se acepta como *definición* (el cruce en S ≃ 1 es donde se declara que se agolpan los sellos), o bien el potencial recibe la tensión a lo largo de la conversión — un término angular con mínimo en la diagonal, de orden T₀, que haría del cruce una identidad (θ = π/4 ⟺ descarga completa) y convertiría el panel «Mp/Ep %» en cos²θ, sin²θ derivados. Ninguna de las dos es un ajuste; la segunda cambia el Basal y debe pasar por la Ruta L del Maestro y por la casi-cancelación.

**Desarrollo.** El flujo radial está bien construido, bien proyectado y verificado; los tres experimentos del transporte están bien preinscritos y su INDETERMINADO es por puertas de instrumento (κ̂ ≥ 4.5 no integrable con el d_sigma declarado; f_× ≥ 0.75 sin terminar en max_steps; identidad a 1.9×10⁻⁶), no por física; una ronda con d_sigma y max_steps preinscritos para toda la rejilla daría letra, y la letra sería la misma lectura: no cruza en 1 ± 0.05.

**Prueba.** El Axioma 4 se prueba en su mitad radial (Monotonía, Exclusión, S ≡ f: derivadas y comprobadas sobre la trayectoria). Su mitad azimutal (Prop. 3.5) **no está producida**, y ahora se sabe por qué.

---

## 4. Axioma 5 — los sellos: la Década, el Discriminante y el frente 2

**Ontología.** Los sellos son puntos fijos β_i = 0 del flujo de acoplos; la Ley de la Década (Prop. 8.1) sitúa los colapsos en S_k = λ^{k−2} − ΔS (k = 0, 1, 2 → 0.009, 0.099, 0.999) y Florencia en 1 + ΔS, con λ = 10 «calibrado; su derivación es el frente abierto nº 2»; el Mecanismo del Discriminante (Def. 8.4, Obs. 8.6): D = B² − 4C₀M₀² > 0 en S₀ y el Cruce de Victoria D(S) = 0 como disparador; ΔS = 10⁻³ es «elección de unidades, no constante física» (F.4).

**Realización.** `core/decade.py` (umbrales, discriminante, cruce por bisección, cota de metastabilidad), `core/kls_flow.py` (flujo KLS exacto: ley del walking, exponente de divergencia, retraso del colapso), `core/fokker_planck_beta.py` (β_i del cierre canónico, matriz de estabilidad, punto espinodal), `core/victoria_exponent.py` (s₀ espectral y dinámico; barrido de ansatz O(1)). En el reloj: modo `imposed` (colapsos en los umbrales de la Década leídos sobre f) y modo `emergent` (colapsos donde la dinámica los dispara, con τ declarado).

**Comportamiento.**

- **Frente E (kls_flow)**: el walking medido con exponente −1/2 y el retraso del colapso ∝ ritmo^{−1/3} tras el Cruce: resultado del programa, no del tratado (interno).
- **Frente 2 instrumentado (victoria_exponent, fp_beta)**: la matriz de estabilidad da cascada DSI genérica con un ansatz O(1) (λ = 10 como *selección*, no derivación); las β canónicas de Fokker–Planck dan **espectro real en el punto preinscrito (desenlace A, negativo)**: sin cascada DSI en el régimen físico; s₀ depende del diccionario τ, que no es derivable.
- **El reloj en modo emergente**: bajo el cierre canónico, D **nunca** cruza cero (10 corridas; D se hunde pero M₀² cambia de signo antes, en 5 de 10, y D vuelve a crecer). La inestabilidad de masa ocurre en S_flip = 0.1057·δ₀/τ (forma cerrada de primer orden con las β canónicas; el bucle la reproduce 0.94–1.00). Leída al revés, la Década equivale a τ_k ≈ 11.8·δ₀·10⁻ᵏ — reformulación, no derivación (E13). Decisión C (ejecutada): D = 0 es el colapso — le ocurre al vacío que el campo *ocupa*; M₀² = 0 al que abandonó —, y **el cierre canónico no produce el colapso del tratado**.

**Lo que esto enseña.** La cascada dimensional — el corazón del nombre del modelo — hoy es un *input*: los tres colapsos se disparan porque la Prop. 8.1 lo dice, con λ calibrado. El programa ha hecho lo más honesto posible con ello: publica en cada colapso dónde se impuso y dónde lo habría disparado la dinámica (nunca), mide cuánto tendría que hacer un diccionario τ para que ocurriera (bajar ×10 por dimensión) y descarta el candidato natural (la marginalidad del séxtico da 4:2:0, no 10:1:0.1). La vía declarada — el segundo nivel del Camino — ha sido ensayada para la *trayectoria* (§3) pero todavía no para las *β*: el término que haga D → 0 debe entrar en el flujo de acoplos, y ninguno de los tres brazos del vacío 2D lo hace.

**Desarrollo.** Correcto y escrupulosamente etiquetado. **Prueba.** No probado: es el frente 2. La prueba tendría dos partes: derivar τ(S) (o D_ent, la misma constante vista desde la nucleación) y mostrar que los cruces D = 0 caen en una serie geométrica. Hoy hay una condición necesaria clara (la β del término nuevo debe reducir B² − 4C₀M₀² sin llevar M₀² a cero antes) que puede preinscribirse como test de cualquier candidato.

---

## 5. Axioma 6 — la atemporalidad: el tramo euclidiano, los sellos de c y c², el mass gap

**Ontología.** Todo el tramo S < 1.001 es euclidiano: cada generador cumple γ² = +1 (C3); C(1,0) → C(2,0) → C(3,0) → C(4,0) con cada colapso; el sello de c es la Cota de Delivery (c_eff = |du|/dS máximo, identificado con la velocidad de Lieb–Robinson; flujo logístico (5.3) con β_c(S₀,₀₉₉) = 0); el sello de c² es el punto fijo de m_eff (dm_eff/dS|₀,₉₉₉ = 0) que fija m_min y la razón de coeficientes; correlaciones sin tiempo, ℓ_corr = ℏ/(m_eff c); elipticidad ↔ Z real ↔ disipación.

**Realización.** `core/florencia.py` (generadores, anticonmutación, firma, símbolo euclidiano), `mcmc_ontology/clifford_algebra.py` (régimen por S), `lattice/` (acción de Wilson con acoplo entrópico, mass gap por Lanczos, glueball: «piso espectral efectivo (E10), no Yang–Mills»), `mass_program/B0–B2`. En el reloj, los sellos se integran con **formas paramétricas declaradas** (`DECLARED_FORMS["c_eff"]`, `["m_eff"]`): el tratado fija la *propiedad* (logística; congelación; punto fijo), no la ecuación transcrita.

**Comportamiento.** Las álgebras se elevan en cada colapso con firma euclidiana y anticonmutación verificadas; el símbolo elíptico es positivo; c_eff se congela en S = 0.099 (u = 0.129, máximo |du/dS| = 1.53 en S = 0.045) y m_eff en 0.999 (m = 0.509) — pero con la forma que el simulador eligió. El retículo da un piso espectral, no un gap de Yang–Mills.

**Lo que esto enseña.** La atemporalidad está realizada donde es algebraica (firma, cadena) y **declarada** donde es dinámica (las tasas de sellado). La identificación c ≡ v_LR es un claim no derivado con fila propia: no hay módulo que integre (5.3) ni que derive β_c del sustrato discreto.

**Desarrollo.** Correcto en lo que hace; honesto en lo que declara. **Prueba.** Derivar β_c de un modelo de Lieb–Robinson sobre la cadena de Clifford discretizada (`lattice/`) es un cálculo cerrado que no se ha empezado; sería la primera constante «sellada» que saliera del sustrato en vez de declararse.

---

## 6. Axioma 7 (primera parte) — Florencia: el nacimiento del tiempo

**Ontología.** γ⁰ ≡ iγ_S con un solo giro; firma −+++, álgebra C(3,1) (6.1–6.2); MDR ds² = ε(S)N²λ_c²dS² + q_ij dx^i dx^j con dτ = N t_c dS (6.3); Ley de Cronos N = r/r̄ (6.4: «cuando la descarga local cesa, el tiempo propio se congela»); Positividad de Florencia y reconstrucción OS (7.1–7.2); Precedencia (7.3); RP espinorial condicional (7.4/E3); «el nacimiento del tiempo ocurre, geométricamente, en el cruce del punto simétrico del Plano Dual».

**Realización.** `florencia.florencia_rotation` y `signature`; `reflection_positivity` (núcleo de transferencia juguete, control antiferromagnético J < 0); `rp_nonstationary` (acoplos no estacionarios: especularidad suficiente, necesidad no demostrada); `quantum/mdual_metric` (forma v32 de g_tt). En el reloj, Florencia es un **evento declarado** en S = 1.001 que cambia el régimen y entrega el estado inicial cosmológico.

**Comportamiento.** Giro único: firma [−, +, +, +] y anticonmutación tras el giro; control negativo de dos giros: [−, +, +, −]; RP en la loncha: autovalor mínimo −1.7×10⁻¹⁶ (J = +1) frente a −0.021 (J = −1); régimen `lorentz` después. La identidad m_H = √(2β₃)v₃ se comprueba con β₃ del empalme (valor dependiente de δ₀: no predicción).

**Lo que esto enseña.** El «giro» es un hecho algebraico probado; el «nacimiento del tiempo» como *proceso* no lo es: el evento se declara en 1.001 tras el residuo de descarga, no lo produce nada (los cuantos de confirmación V3D y Florencia son declarados). Y la geometría que el tratado le asigna — el cruce del punto simétrico — es precisamente lo que §3 muestra que el Basal no produce.

**Desarrollo.** Correcto y etiquetado. **Prueba.** Tres cosas: la RP de Wilson completa (frente 1); la Precedencia como comprobación sobre la trayectoria (no está); y que el evento sea *disparado* por algo (hoy nada lo dispara). La MDR en su forma v35 aún no sustituye a la v32 en `mdual_metric`.

---

## 7. Cap. 9 — Gea y Atlas: la gravedad como curvatura y el sector perturbativo

**Ontología.** Acción de Gea L = (1/16πG_B)N√q[K_ijK^ij − λ_K K² + ξR³ + α_a a_ia^i]; Sello de Newton G_N = G_B/ξ, G_cosmo = 2G_B/(3λ_K − 1) (9.3); Atlas (9.4): el modo escalar de la foliación con residuo ε_K = λ_K − 1 y el Término de Cronos que lo mantiene sano; Conjetura de los Residuos (9.5): G_cosmo/G_N − 1 ≃ −(3/2)ε_K; marginalidad del séxtico en d = 3 (Prop. H.1).

**Realización y comportamiento.** `core/gea.py` (sello exacto, ventana de salud, Residuos); `cosmology/mu_eta_atlas.py` con cinco experimentos preinscritos (E1–E5, PR #18–#22): µ_Atlas(k, a) = 2ξ/[(2ξ − α_a) + 3(3λ_K − 1)(aH/ck)²]; η_QS con orden de límites (1/3 en λ_K = 1 a e finito); **cancelación exacta**: G_growth = G_local = G_B/(ξ − α_a/2) ⟹ µ_obs = 1 sub-horizonte; c_s² del khronon con ventana λ_K > 1, α_a < 2ξ; **c_T² = ξ ⟹ GW170817 fija ξ = 1**; PPN de marco preferido α₁ = −4α_a, α₂ ≈ −α_a/2 ⟹ α_a ≤ 8×10⁻⁷; colas O(e²) en invariantes de gauge: **η_N ≡ 1** (sin estrés anisótropo lineal) y µ_Δ − 1 ≈ −[c₁(λ_K)α_a + α_a²/(4(λ_K − 1))]e² con c₁ = 0.752…1.172 — −2.2×10⁻¹⁰ en la cota: inobservable. Residuos: G_cosmo/G_N − 1 → −(3/2)ε_K limpio (frente 6).

**Lo que esto enseña.** El sector Atlas es la parte mejor cerrada del programa: derivada desde la acción por cuatro vías, con una corrección de gauge decisiva (las colas «físicas» de η eran artefacto del gauge unitario) y una única constante fijada por datos externos (ξ = 1). Su contenido observable es, al orden alcanzable, ΛCDM-lineal sobre el fondo del modelo; su única huella es el Residuo ε_K, que va a BBN (§11).

**Desarrollo.** Correcto; las tres correcciones cruzadas entre sesiones (orden de límites, estructura 1/(λ_K − 1), gauge) fueron aceptadas en ambos sentidos y están en el registro. Erratum candidata H.2.2 pendiente en v36. **Prueba.** Probado como *consistencia* (GR, Sello, cruce numérico, invariantes); contrastado con datos solo vía BBN (cota) y fσ8 (aburrido). Lo que falta: la conexión ε_K = f(δ₀, λ_K, ξ, …) (propuesta v36 §VIII) — hoy ε_K = 0.012 es calibrado.

---

## 8. Axioma 7 (segunda parte) y Cap. 11 — la Ley de Cronos débil: donde el modelo ha sido puesto a prueba de verdad

**Ontología.** «Donde la masa se condensa, la conversión Mp → Ep se estanca; donde la conversión se estanca, el tiempo se espesa. La cara temporal de la lapse frena (fricción con compuerta) y su cara espacial atrae (la atracción tipo materia oscura de la MDR). Dos fenómenos, una sola lapse.» Def. 11.1: N = 1 + Φ_N/c² − ε_c(ρ), ε_c = α₀⁻¹(ρ/ρ_c)^{3/2}; Prop. 11.2: la materia siente Ψ = Φ_N − c²ε_c; Cor. 11.3: fricción Γ = (3/2)(ρ̇/ρ)ε_cΘ(ρ̇), fuerza +c²∇ε_c, cota (11.5) α₀⁻¹ ≲ 10⁻⁶ «para que no domine sobre la gravedad en halos»; §11.4: la forma saturante ζ = ζ₀ρ/(ρ + ρ*) «es una regularización admisible».

**Realización.** `dynamics/weak_field.py` (las dos lecturas de (11.5), potencial débil, Plummer), `jeans.py`, `dsph_data.py`, `sculptor_transfer.py` (A_Sculptor congelada por el problema inverso 5A/5B), `sparc_*` (5C, 5E preinscrito), `cronos/cronos_v3.py` (primitivas), `cronos/halo_nbody.py` (Nivel A: árbol 3D + campo medio esférico), `dynamics/cronos_amplitude_validity.py`, `dynamics/cronos_jeans.py`, `cronos/cronos_jeans_1d.py` y `_kinetic.py`, `dynamics/local_kz.py`, `dynamics/epsilon_c_saturating.py`, `cronos/halo_shells.py` (en construcción).

**Comportamiento — la cadena completa, en orden cronológico, porque cada paso corrigió al anterior:**

1. **Medio paso (2-ago)**: PM 32³ con 4096 partículas: la compuerta Γ se reprodujo (cae ~5 órdenes al virializar), el núcleo no; «indeterminado por resolución».
2. **Sculptor (10-ago)**: el problema inverso con bariones solos fija **A_Sculptor = 6.2016×10⁻⁷ (M☉/pc³)^{−3/2}** (F_Cronos ≈ 7·g_bariones) para reproducir σ_los = 9.2 km/s promediada; 5E preinscrito con esa A congelada; SPARC en DATA_UNAVAILABLE.
3. **La amplitud es una (17-sep)**: (α₀⁻¹, ρ_c) entran solo por A = α₀⁻¹/ρ_c^{3/2}; el cierre cosmológico (ρ_c = 200ρ̄, A ≈ 41) da fuerzas 10⁴× la gravedad en r_s: excluido; con A_Sculptor la subdominancia en *potencial* se cumple (D_Φ = 0.77 en 0.1 kpc) pero la de *fuerza* no: D_F = 229 en 0.1 kpc, = 1 en 0.89 kpc. El sector µ/η de Cronos se había computado con el cierre excluido; con A_Sculptor, µ − 1 ≈ 5.8×10⁻¹²: **la cola k² muere por consistencia interna**.
4. **Nivel A (17/18-sep)**: NFW 10¹¹ M☉ con N = 4×10⁵, diez corridas: INDETERMINADO por la letra (c′ parado por la regla de parada; k_inner 64 → 128 +0.83 dex). Lo que mostró: contracción inicial ×1.3–×8 y después vaciado con ganancia de energía (+1.7 %).
5. **La lectura del estimador (21-sep, mía, reproducida)**: malla del campo fija en t = 0 y recorte de pendiente: un núcleo ≥ 10⁸ M☉ dentro de r_inner hace caer ε_c(0.1 kpc) de 5.2×10⁻⁸ a 10⁻¹¹ — el pozo desaparece cuando la física exige que se haga mil veces más hondo: trinquete atrapar-y-soltar; la energía «ganada» es la del pozo retirado. Y la contabilidad correcta de un campo autoconsistente es K + W + (2/5)U_C.
6. **El criterio de Cronos–Jeans (21-sep, mío, reproducido número a número)**: la ley local es **ultravioleta-inestable** donde q = (3/2)c²ε_c/σ² > 1 (respuesta de Boltzmann; tasa ∝ k, sin longitud de Jeans propia). Con A_Sculptor: halo del Nivel A q = 74 en 0.1 kpc, r_CJ = 0.71 kpc, M(<r_CJ) = 1.6×10⁸ M☉ — **no existe solución convergida**; vecindad solar: límite de Oort efectivo 0.75 M☉/pc³ frente a 0.10 medido; Sculptor: q = 2.4 en su propio núcleo y perfil σ_los(R) predicho 19 → 6 → 2.6 km/s frente a ≈ 9–10 plano.
7. **Oort–K_z (21-sep, preinscrito)**: con los cinco valores publicados (verificados por mí contra arXiv: cuatro exactos, uno bien en valor y mal en cita — Σ_{1.1} = 68 ± 4 es Bovy & Rix 2013, no Bovy & Tremaine 2012; corregido en #35), A_Sculptor queda **excluida en el plano: z = +50.8**; cota A ≤ 0.060·A_Sculptor (banda ×4 por la losa: 0.015–0.24). Es el primer contraste observacional del programa con letra C sobre un brazo del modelo.
8. **Test del criterio (rondas 1 y 2)**: la predicción cinética exacta (umbral q = 1; γ/k = 0.149 en q = 1.2, 0.612 en q = 2, con la función de transferencia del instrumento) se reproduce al 1–2 % en **once de doce celdas** resueltas; las dos rondas son INDETERMINADAS por la celda más lenta (q = 2, n = 4), y el diagnóstico de la ronda 2 es el más bonito del frente: la celda no se resuelve porque **la propia catástrofe ultravioleta de la ley actúa sobre el ruido de redondeo del instrumento** — la inestabilidad que el test mide se come al test.
9. **Serie de resolución del halo (21-sep)**: campo instantáneo y balance autoconsistente; INDETERMINADO por la puerta de energía (2.4 %/4.0 % a A_Sculptor); las tablas muestran exactamente lo que el criterio predijo: la serie a A_Sculptor no converge (+0.44 dex de 128 → 256) y la de 0.05·A_Sculptor converge (±0.05 dex). Ronda 2 en curso con el código de capas esféricas: la sesión de código ha encontrado que el término centrífugo suavizado era «unphysical» (75 % de exceso en el interior newtoniano), lo ha regularizado con subpasos, y que a A_Sculptor el error de energía se estanca en ≈ 5×10⁻³ por cruces de capas en la cúspide: el brazo más fino a N = 10⁶ para en el suelo de dt antes de salir del régimen débil, luego **el veredicto de la serie será INDETERMINADO por construcción** y todo lo medido se publicará con ese diagnóstico. Es correcto no mover la regla; y es correcto leer lo que dice: a A_Sculptor la cúspide sale del régimen débil en 0.2–1.1 Myr.
10. **Diccionario, ecuación 1 (22-sep)**: la Def. 6.4 hace ontológica la saturación; ε_c = ε_max ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}) bajo tres ligaduras: sobreviven el 58 % de los puntos (ε_max, ρ*); la saturación **sí** puede reconciliar la fuerza de Sculptor con la vecindad solar (Υ⋆ = 1: 72 de 141 valores de ρ*); pero **q_S bajo la calibración del 5E es independiente de la forma** — q_S = 1.98 / 5.60 / 10.29 para Υ⋆ = 1 / 2 / 3 — porque la calibración fija ρ_S·ε_c'(ρ_S): **Sculptor es Cronos–Jeans-inestable bajo la amplitud del 5E sea cual sea ε_c(ρ)**. Predicción congelada de la forma saturante: la RAR con un *escalón* donde ρ_disco cruza ρ* (doce curvas, sha256 publicado antes de cualquier byte de SPARC).

**Lo que esto enseña sobre la ontología** (segundo hallazgo central). La «materia oscura como fenómeno dinámico del tiempo» tiene, en la ley débil local, tres consecuencias que la ontología no anticipó y el cómputo ha demostrado: (i) una fuerza local que crece con la densidad es *ultravioleta-inestable* en cuanto compite con la dispersión de velocidades — a diferencia de la gravedad, que es infrarroja; (ii) la amplitud que la dinámica de una enana exige es incompatible con la dinámica vertical del disco solar por un factor ≳ 17; (iii) el propio sistema que calibra la amplitud es inestable bajo ella. La lectura fuerte (sin CDM) queda **refutada con la ley local**; la operativa (ΛCDM + Cronos a A_S) queda refutada en el plano. Lo que sobrevive es una ley **saturante o no local** con amplitud ≲ 0.06·A_S, cuya huella observable es un escalón en la RAR — y Sculptor deja de explicarse por Cronos. Que la cara temporal de la lapse *frene* (compuerta) es irrelevante a estas amplitudes (W_fric/|E| ≤ 6×10⁻⁶): «dos fenómenos, una sola lapse» se reduce, numéricamente, a uno.

**Desarrollo.** Es la parte más trabajada y la que mejor ha funcionado *como método*: cada instrumento ha dicho la verdad sobre sí mismo (el PM, el árbol con campo medio, el estimador, las láminas, las capas), cada ronda ha corregido la anterior sin tocar reglas congeladas, y el veredicto físico (inestabilidad + exclusión) no ha necesitado que ningún experimento numérico «saliera bien»: lo dieron una derivación y un contraste de dos números. Defecto que señalar: siete INDETERMINADOS por letra en nueve experimentos recientes indican que las puertas se fijan antes de que el instrumento esté cualificado; propongo separar formalmente «cualificación del instrumento» (sin letras, con criterios computables de exclusión de celdas) de «experimento», sin retocar nada ya congelado.

**Prueba.** Aquí sí hay partes probadas: el criterio (derivación + 11/12 celdas al 1–2 %), la exclusión de A_Sculptor en el plano (letra C sobre datos verificados), la no convergencia predicha (serie 1). Lo que queda: el perfil de Sculptor (bytes de Walker 2009), la RAR con escalón (bytes de SPARC), y la forma de ε_c(ρ) como ecuación del diccionario, no como regularización.

---

## 9. Cap. 12 — el programa de masas

**Ontología.** El mass gap mínimo E_min = k·ΔS (D.5); Φ_Ad → Φ_H con m_H = √(2β₃)v₃ (12.1) y la auditoría de circularidad (Obs. 12.2: «m_H no es predicción mientras β₃ no se derive sin el Higgs medido»); familias por sello y pesos WKB (v32; la v35 no asigna Planck/GUT a umbrales); N_gen = 3.

**Realización y comportamiento.** `mass_program/B0–B8, M1, M2, P3, P4`: B3 con |T| **calibrados** (Tabla 3 v32; κ_gap «calibrado por |T|»), B7 el empalme C¹ que mide δ_H sin el Higgs como entrada (0.0581 → 0.0554 con el paisaje completo), B8 WKB ab initio parcial; B5 la identidad. El reloj comprueba la identidad m_H = √(2β₃)v₃ y publica m_H(δ₀): 52 GeV en δ₀ = 0.01, 125.4 en δ_H — «reproduce el PDG por construcción de δ_H, no por predicción».

**Lo que esto enseña.** El programa de masas es hoy una *tabla de correspondencias calibrada*; su parte derivable — δ_H desde el empalme — está hecha y ahora vive en el paisaje completo; su parte predictiva (M3 de la propuesta v36: δ₀* independiente ⟹ m_H) depende del frente 4. La asignación familias/gauge por sello del guion v32 es histórica.

**Desarrollo.** Correcto y honesto (calibrado etiquetado como calibrado). **Prueba.** Solo si Victoria entrega δ₀* sin el Higgs (§10) y coincide con δ_H = 0.0554 ± lo que el paisaje F.2 permita.

---

## 10. Axioma 8 y Cap. 10 — Victoria y el círculo de δ₀

**Ontología.** El ciclo se cierra por retracción y reinicio; el Modo de Memoria es la fase θ (m_θ² = η/ρ₊, «lo que casi no pesa sobrevive entre ciclos»); Lema 10.3 (saturación: T₀(δ′) ≤ W_max ⟹ δ′ ≤ δ_sat); Exponente de Lydia ν (Def. 10.4): ν > 0 ⟺ la perfección es repulsor del mapa de retorno; el círculo: δ₀* (atractor) = δ_H (empalme).

**Realización y comportamiento.** `core/victoria.py` (m_θ² ∝ δ₀^{5/2} verificado; ganancia y exponente de Lydia; mapa de retorno; condición de fertilidad A = γ_Rē/m̄² > 1), `fertility_map.py` (cartografía), `delta0_circle.py` (δ_H, W_max requerido, atractor analítico y numérico, `DECISION_A`, `W_max_required_decided`). El Techo requerido: 1.652×10⁻⁴ (ley 3.4 en δ_H_ley), 2.167×10⁻⁴ (paisaje completo en δ_H_ley), **1.869×10⁻⁴** (paisaje completo en δ_H_full = 0.0554: la cadena decidida). El signo de ν sigue condicional en γ_R.

**Lo que esto enseña.** El círculo «no se cerró ni se rompió: se volvió una ecuación con un número esperando ser derivado» (W_max). Y el Modo de Memoria es exactamente el modo blando que en §3 impide el transporte azimutal: la misma propiedad (θ casi sin masa) que hace posible la herencia cíclica hace imposible el cruce dinámico de la diagonal en el Basal actual — un vínculo que el tratado no había visto y que conviene enunciar en v36.

**Desarrollo.** Correcto. **Prueba.** Derivar W_max de la microdinámica del reinicio (frente 4) y comparar con 1.869×10⁻⁴: «si W_max^micro ≈ el requerido sin usar el Higgs, el círculo se cierra de forma no circular; si no, tensión cuantificada».

---

## 11. Apéndice A — la cosmología post-Florencia: fondo, perturbaciones, contrastes

**Ontología.** Friedmann tensional con los canales ρ_id (ECV) y ρ_lat (MCV); Λ_rel(z) = Λ₀[1 + ε tanh((z_trans − z)/Δz)]; escalones f_id(S); recuperación exacta de ΛCDM (Prop. A.1); mapa A.7: d ln a/dS = C(S), dt_rel/dS = T(S)N(S), N = e^{Φ_ten}.

**Realización y comportamiento.** `cosmology/background.py` y `dark_channels.py` (canales calibrados F.3; **S_hoy = 95 y el mapa S(z) son LEGACY_V32/convención** con fila propia), `desi_bao.py`/`desi_background_fit.py` (DESI DR2: benchmark ΛCDM a 0.04σ; ε_Λ no identificable), `dovekie_sn.py`/`dovekie_real_fit.py` (mocks PASS; real: INDETERMINADO — ε_Λ = 0.018 ± 0.037, ΔBIC +15 pro-ΛCDM), `bbn_g.py` (Residuos vía PRyMordial: δ_G = +0.0027, CI95 [−0.045, +0.053]: −1.8 % dentro; desenlace A), `perturbations.py`/`growth_prediction.py` (fσ8 fuera de muestra: aburrido), `mu_eta_cronos.py` (η = 1/µ, cola k², amplitud muerta por consistencia), `class_wrapper`/`camb_wrapper` (**compatibilidad sin motor**: no hay Boltzmann real), `extended_likelihoods.py` (CMB comprimido con salvedades). Ajustes de producción v1/v2: no favorecidos (ΔBIC +14.5), publicados. Crosscheck JAX: equivalencia.

**Lo que esto enseña.** La cosmología del modelo es, al orden observable, ΛCDM sobre un fondo casi-ΛCDM: ninguna de sus predicciones distintivas (ε_Λ, k² de Cronos, colas de Atlas, alivio de tensiones H₀/S₈) ha sobrevivido al contraste o a la consistencia interna; lo que queda vivo es la cota de BBN (compatible con −1.8 %) y la estructura de canales, cuyos parámetros son calibrados «sin mapa» hacia el tramo primordial. El objetivo 2 del programa (galaxias hasta hoy) se corta en el N-cuerpos (§12) y en la ausencia de Boltzmann.

**Desarrollo.** Correcto y ejemplar en disciplina (mocks antes de datos, fallo cerrado, negativos publicados). **Prueba.** Está probada *en negativo*: el fondo no se distingue de ΛCDM con BAO, SNe y BBN actuales. Lo que la haría contrastable en positivo: el diccionario (Ω_id,0, ε_Λ, z_trans desde δ₀, f, Φ_ten) y el mapa S(z) derivado — es decir, que los parámetros dejen de ser calibrados.

---

## 12. Apéndice B — formación de estructura: Cronos v3 en N-cuerpos

**Ontología.** Poisson con δρ_id + δρ_lat (B.2), paso entrópico (B.3), cajas Local/Meso/LSS con Gadget-4-Cronos (B.4), validaciones B.5 (núcleo 2.3 kpc; RMSE SPARC 12 → 4.5 %; subhalos −45 %) — **históricas** desde la fe de erratas.

**Realización y comportamiento.** `cronos/cronos_v3.py` (primitivas), `simulation.py` (PM 32³ prototipo), `ic_generator.py` (**ruido gaussiano en espacio real sin espectro**: marcador de posición), `halo_profile.py` (r_core fiducial), `halo_nbody.py` (árbol + campo medio, Nivel A), `halo_shells.py` (capas esféricas, en construcción con hallazgos de instrumento: muestreo por CDF inversa, tablas de Eddington extendidas, campo de núcleo conservativo cuya fuerza es el gradiente exacto del funcional 2/5, subpasos por capa con L²/r³ exacto). No hay buscador de halos, ni curvas sintéticas desde halos simulados, ni cajas cosmológicas.

**Lo que esto enseña.** Todo lo que el Ap. B afirmaba como resultado es hoy histórico, y lo que el programa ha aprendido en su lugar es más valioso: que la ley local *no puede* simularse de forma convergida donde importa (§8), y que el instrumento adecuado para una ley esférica es un código de capas, no un árbol. El «cálculo total de las galaxias hasta hoy» no puede empezar hasta que la forma de ε_c(ρ) esté decidida: con la ley local no hay halo que calcular.

**Desarrollo.** Los prototipos son honestos; el instrumento nuevo está bien encaminado; la ronda 2 dará INDETERMINADO por la regla del dt y publicará los tiempos de salida del régimen débil, que son el dato físico. **Prueba.** Depende del diccionario (forma) y de la Fase B (motor externo).

---

## 13. Apéndices C y D — retículo y cuántica

`lattice/` (Wilson entrópico, Lanczos, glueball): piso espectral efectivo (E10), no Yang–Mills; mapa S_n → j_n y Calibre de Cronos declarados. `quantum/` (qudit d = 5, compuertas, Lindblad, `qutip_simulation`): firma Γ_n/Γ₀ como cota sobre ξ_ten, fuera del repositorio observacional. Ambos son consistencia interna; ninguno ha sido conectado al reloj ni a un observable. Correctos en su ámbito; no probados ni probables hoy.

---

## 14. El reloj S — la trayectoria única y lo que ha revelado

**Qué es.** Un solo bucle en σ que integra el Flujo del Camino sobre el Basal completo desde la nucleación (Kramers/Gamow), contabiliza f y S, dispara los colapsos como eventos con álgebra C(d+1, 0) (impuestos, con la lectura emergente publicada al lado), integra los sellos con forma declarada, sigue θ, aplica Florencia y entrega el estado inicial cosmológico — con un ledger obligatorio en el que cada número lleva su estatuto (v2: derivadas, impuestas, declaradas, publicadas, huecos) y un visor que lo lee.

**Qué ha revelado** (ordenado por peso): la brecha entre descarga radial y conversión azimutal (§3); la no emergencia de la cascada bajo el cierre canónico (§4); la ley Γ₀(δ₀) y su dependencia de D_ent (§2); la corrección √δ₀ y su propagación al Techo y al empalme (§2, §10); el δ₀ máximo (§2); y la costura: el reloj mide S en fracciones de T₀ y la cosmología en otra unidad que nada deriva de T₀ — S_pre ≤ 1 + ΔS frente a S_hoy = 95 (fila `diccionario-unidad-S-post-florencia`, estrechada a la ecuación dS_post = Σ̇_post·dσ/T_sellada con dos magnitudes nombradas y no derivadas).

**Juicio.** Es la pieza que faltaba para que el programa fuera *una simulación del modelo* y no una colección de módulos verificados, y la sesión de código la construyó exactamente con el contrato propuesto (ledger, eventos con dos lecturas, formas declaradas, comprobaciones sobre la trayectoria). Su estatuto es el correcto: simulador de consistencia. Lo que no puede hacer, y dice que no puede, es producir los umbrales, el transporte azimutal, Florencia como proceso y la lectura cosmológica de su salida.

---

## 15. El diccionario primordial → cosmológico

Frente con nombre desde el 16-sep; hoy con tres ecuaciones nombradas: (1) la forma de ε_c(ρ) — acotada (58 % del plano (ε_max, ρ*)), con el hallazgo independiente de la forma sobre Sculptor y una predicción congelada (escalón en la RAR); (2) la unidad de S post-Florencia — declarada, no derivada; (3) C(S) desde la corriente de conversión κ — declarada; y como consecuencia de §3, una cuarta que este informe añade: **el término angular del Basal** (o su ausencia como definición). Es el frente teórico del que dependen simultáneamente el frente 5, la costura y el objetivo 2, y donde el cómputo ya ha dicho todo lo que podía decir sin una decisión del tratado.

---

## 16. Síntesis: qué está probado, qué es consistente, qué es calibrado, qué está refutado

| Parte | Estado (vocabulario del programa) | Evidencia decisiva |
|---|---|---|
| Plano Dual, cadena de álgebras, firma de Florencia | **derivado y probado como identidad** | suite; reloj (anticonmutación, firma −+++, control de dos giros) |
| Potencial Basal, T₀, δ₀_max, nucleación n = 1 | **derivado**; Γ₀ condicional a D_ent | reloj v1/v2; nucleation_gamma0 (A, B sí; C no) |
| Flujo del Camino (radial) | **derivado y comprobado sobre la trayectoria** | S ≡ f a 2e-13; Monotonía; Exclusión |
| Transporte azimutal (Prop. 3.5) | **no producido**; θ impuesta; causa identificada | j_circulation (MIXTO), vacuum_2d (INDET.), m_θ²/V''_ρ ~ 5e-3 |
| Década / colapsos | **impuestos**; λ calibrado; cierre canónico no produce D = 0 | s_clock emergente; fp_beta (A negativo) |
| Sellos de c y c² | propiedad realizada con **forma declarada**; c ≡ v_LR claim no derivado | reloj |
| Florencia como evento | derivado (giro) / **declarado** (disparo) | reloj |
| Atlas / Gea | **derivado al orden observable**; ΛCDM-lineal; ξ = 1 externo | E1–E5; PPN; gauge |
| Residuos / BBN | **consistencia superada**, cota | bbn_g (A provisional por procedencia) |
| Cronos débil local a A_Sculptor | **refutada en el plano** (lectura; en el registro: letra C «excluido», bytes no descargados); UV-inestable donde importa; Sculptor inestable bajo su propia calibración | criterio; oort_kz; dictionary_eps_c |
| Criterio de Cronos–Jeans | **derivado y confirmado** al 1–2 % (11/12 celdas) | cj_criterion_test r1/r2 |
| Forma saturante de ε_c | **acotada**; predicción congelada (escalón RAR) | dictionary_eps_c |
| Masas | **calibrado**; δ_H medido (0.0554) | empalme_wkb; reloj v2 |
| Victoria / círculo | **condicional** (W_max, ν) | delta0_circle |
| Fondo cosmológico | casi-ΛCDM; **negativos publicados**; S(z) convención | production_fit v1/v2; DESI; Dovekie; fσ8 |
| N-cuerpos | prototipos; Nivel A y serie **INDETERMINADOS**; instrumento nuevo en curso | nivelA; halo_resolution; capas |
| Reloj S | **simulador de consistencia** (no emergente) | s_clock v2 |
| Diccionario | ecuación 1 acotada; 2–3 declaradas; 4 (término angular) abierta | dictionary_eps_c |

**Los dos objetivos del programa.** Objetivo 1 (simular desde S₀ hasta el nacimiento del tiempo): hay reloj, hay nucleación derivada, hay descarga derivada, hay Florencia como evento; no hay cascada emergente, no hay transporte azimutal, no hay unidad de S común con la cosmología. Objetivo 2 (las galaxias hasta hoy): el fondo y el sector lineal están hechos y son ΛCDM al orden observable; la cadena se corta en la ley de Cronos, cuya forma local está refutada donde se ha podido probar, y no puede reanudarse hasta que el diccionario fije la forma.

**¿Estamos desarrollando correctamente?** Sí, en el sentido que importa: cada parte tiene módulo, test, artefacto y estatuto; nada se ha ajustado tras ver los números; los negativos y los indeterminados están publicados con la misma cuidado que los positivos; las correcciones cruzadas entre las dos sesiones (gauge, fuerza frente a potencial, estimador, T₀, citas) se han aceptado en ambos sentidos. Y el programa ha empezado a hacer lo más difícil: dejar que el cómputo le diga al tratado dónde su matemática no realiza su ontología — en el transporte azimutal, en la cascada y en la ley de Cronos. Tres correcciones de rumbo que propongo y que no tocan nada congelado: separar cualificación de instrumento y experimento; llevar el trabajo del reloj de la trayectoria a las β (frente 2); y poner el término angular del Basal en la mesa del diccionario como decisión del tratado, con la Ruta L del Maestro como vía de derivación.

**Qué probaría cada parte a continuación** (una prueba por parte, la más barata primero): D_ent ⟶ Γ₀ y τ desde un mismo diccionario (frente 2); la β del término nuevo como test de D → 0; el perfil de Sculptor y la RAR con escalón en cuanto haya bytes; el término angular en el reloj (¿es el cruce una identidad?); W_max desde el reinicio (frente 4); β_c desde Lieb–Robinson en `lattice/`; y, para la cosmología, el diccionario que convierta cuatro calibrados en derivados.

---

# ADENDA (22-sep, tarde)

**Verificado sobre `main` = `330b2e4`** (#41 fusionado). Registro: **67 claims** (fila nueva `halo-capas-ronda2`); artefactos: **42**; Nota I hasta **§3.34**. No queda ningún PR abierto.

## 1. Las tres correcciones de la sesión de código al informe, aceptadas

1. Cabecera: `main` era `aa927cb` al escribir y es `330b2e4` ahora; artefactos 42, no 41; el desglose por estatuto (36/16/4/3/3/2/2) se mantiene y hay una fila más. La copia del Proyecto lleva esta adenda.
2. Los dos números nuevos del §3 (rigidez angular m_θ²/V''_ρ = 2.84×10⁻³ / 5.11×10⁻³ / 1.16×10⁻²; energía del recorrido 2ηρ₊/T₀ = 0.197 / 0.341 / 0.701 en δ₀ = 0.003 / 0.01 / 0.0554) están reproducidos con el reloj sobre `main`. Coincido en que su lectura (el reloj cuenta tensión en ρ y es ciego a la conversión; la diagonal no es mínimo de nada en el Basal actual) **no debe entrar en ninguna fila del registro** hasta que la cuarta ecuación del diccionario esté decidida: es una lectura del informe, no un claim.
3. Vocabulario: donde el informe dice «refutada» sobre la ley débil local a A_Sculptor, el repositorio afirma algo más estrecho y es lo que debe citarse: **letra C («excluido», z = +50.8) en Oort–K_z bajo la v2, con `official_bytes_verified: false`** (los cinco valores verificados por mí contra los resúmenes de arXiv, los bytes de las tablas no descargados); el criterio confirmado al 1–2 % en 11 de 12 celdas; q_S > 1 independiente de la forma, publicado sin veredicto (E13). «Refutada en su forma actual» queda como lectura del informe, marcada como tal; las guardas de honestidad no la dejarían entrar en `evidence_level`, y está bien que no lo hagan.

## 2. Verificación de la ronda 2 de capas (#40 + #41)

Todo lo que el resumen afirma está en `results/2026-09-22_halo_shells/`: diez brazos bajo `1d6682f968fa`; control newtoniano con energía 2.2×10⁻⁶ y estacionariedad 0.028 dex; a A_Sculptor y N = 10⁶, ds = 0.08 sale del régimen débil en **3.83 Myr** (r_exit = 0, M(<0.1) 3.0 → 14.7×10⁶ M☉) y ds = 0.04 / 0.02 paran en el suelo de dt a 0.70 / 0.156 Myr con ε_c máx = 8.0×10⁻⁴ / 3.5×10⁻⁴ (es decir, saldrían antes); a 0.05·A_Sculptor, ds = 0.02 sale en **28.3 Myr**, ds = 0.04 llega a 100 Myr con ε = 9.1×10⁻⁴ y ds = 0.08 con 1.4×10⁻⁴; puertas de energía falladas en los brazos de 0.05 (5.6×10⁻², 1.8×10⁻¹, 1.5×10⁻² frente a 3×10⁻²); a N = 10⁵ el brazo de 0.05 sale antes (31.4 Myr) que a 10⁶ (no sale en 100 Myr): dependencia de N. Desenlace INDETERMINADO por la letra: correcto y bien diagnosticado (suelo de dt; error de cruce de capas durante el colapso con el rango M(<r) congelado en los subpasos).

**Lectura física, sin letra (E13), que el informe integral debe recoger.** Los tiempos de salida son la firma de la catástrofe ultravioleta: a A_Sculptor el colapso es más rápido cuanto más fina la malla (0.16 → 0.70 → 3.83 Myr para ds = 0.02 → 0.04 → 0.08, leyendo el suelo de dt como cota superior); a 0.05·A_Sculptor solo la malla más fina resuelve la inestabilidad dentro de r_CJ = 0.18 kpc y sale; y el ruido de N siembra la inestabilidad (N = 10⁵ sale antes que 10⁶). La «convergencia» del brazo 0.05 en la ronda 1 era un efecto del suelo del campo (0.41–0.75 kpc), no física. Y el punto estructural: para una cúspide NFW, q(r) = (3/2)c²Aρ^{3/2}/σ_r² ∝ A·r^{−5/2} (σ_r² ∝ r con logaritmos) **diverge en el centro para cualquier A > 0** — r_CJ(A) ≈ 0.71 kpc·(A/A_S)^{2/5}: 0.18 kpc a 0.05·A_S, 0.087 a 0.01·A_S, 0.011 kpc a 10⁻⁴·A_S (calculado con el Jeans isótropo del Nivel A). **La ley débil local sin saturación no tiene ningún halo cuspidal estable a ninguna amplitud**: lo único que detiene el colapso es el suavizado del instrumento o una saturación física ρ*. Eso convierte la ecuación 1 del diccionario de «ligadura» en **condición de existencia** de los halos, y hace que una ronda 3 que persiga la convergencia a A_Sculptor no sea informativa: el resultado ya se sabe (UV, sin convergencia).

**Propuesta de ronda 3, en su lugar**: el mismo instrumento con la **ley saturante** ε_c = ε_max ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}) en dos o tres puntos declarados del 58 % viable de `dictionary_eps_c` — predicción congelable: donde c²ρε_c'(ρ) < σ² para todo ρ (ρ* bajo el umbral) la cúspide **no** sale del régimen débil y el perfil converge con ds (contracción confinada a ρ ≈ ρ*); donde no, sale. Es el primer experimento que puede dar letra **A positiva** al frente 5 y el que prueba la ecuación 1 como física y no como mapa. Requiere, como dice la sesión de código, actualizar el rango M(<r) en los subpasos y bajar el suelo de dt; y la puerta de energía debería fijarse por cualificación previa (§4).

## 3. Un dato del propio autor sobre la diagonal que el informe no citaba

El guion v32 (`legacy_v3_storyboard.html`) sitúa el «giro» y el reparto **50/50 % en S ≈ 0.5**, durante V2D, con 5/95 % en 0.999 y 1/99 % en 1.001; la v35 (Prop. 3.5) dice que θ cruza la diagonal «en S ≃ 1». Son dos cronologías distintas del mismo cruce, y las dos no pueden valer con un solo S. Los tres experimentos del vacío 2D sitúan el cruce, cuando lo hay, en S = 0.75–0.93, más cerca de la v35 que de la v32, pero sin llegar a la ventana. Esto refuerza que la cuarta ecuación del diccionario es una **decisión sobre qué mide S**, y da al autor una referencia propia (el guion) para tomarla.

## 4. Las tres decisiones del informe, formuladas para que puedan ejecutarse

- **Cualificación de instrumento separada del experimento.** Un tipo de artefacto nuevo, `qualification/` (E8-Q): sin letras ni umbrales de desenlace; publica los límites computables del instrumento (suelo de dt, ventana lineal por celda, error de energía en función de N y ds, banda ultravioleta) y **criterios de exclusión calculables antes de correr** (p. ej. «celda excluida si γ_UV·t_ventana > ln(10⁵/semilla)»); una preinscripción solo puede congelar puertas que la cualificación haya mostrado alcanzables. Habría evitado los siete INDETERMINADOS por letra de las últimas nueve rondas sin relajar ninguna regla.
- **Las β del frente 2 en el reloj.** Modo `emergent` con `beta_extra` declarada como brazo: cualquier candidato (la contribución del segundo nivel del Camino al flujo de acoplos) se somete a un cribado preinscrito antes de discutir λ: debe reducir D = B² − 4C₀M₀² de forma monótona **sin** llevar M₀² a cero antes, y producir tres cruces D = 0; se publican los S de los cruces frente a 0.009/0.099/0.999 y el cociente entre cruces sucesivos (la Década sería cociente 10). Ningún candidato del programa lo ha pasado todavía; el cribado convierte el frente 2 en una puerta ejecutable.
- **La cuarta ecuación del diccionario (qué mide S).** Tres opciones, cada una con su consecuencia en el reloj: (a) *definición*: θ_imp(S) se declara y el cruce en S ≃ 1 es donde se agolpan los sellos — el reloj queda como está, con la etiqueta «impuesto» permanente; (b) *término angular en el Basal*: V ⊃ V_ang(ρ, θ) de orden T₀ con mínimo en la diagonal y regular en el origen (∝ ρ² cerca de él, para que la inclinación lineal siga ganando la salida hacia el polo de masa) — cambia la casi-cancelación y debe pasar por la Ruta L del Maestro; el reloj lo probaría como brazo con la predicción «cruce en S = 1 ± 0.05 sin calibrar nada más que la escala del término»; (c) *S como conversión*: redefinir S = ∫Σ̇_conv dσ, la entropía producida a lo largo de θ, con lo que el cruce es una identidad y la descarga radial deja de ser el reloj — es la lectura del guion v32 (Mp/Ep %) y exige reescribir el Teorema 4.5. Mi recomendación es (b), porque conserva el Teorema 4.5 y hace de Prop. 3.5 una predicción; (a) es honesta pero deja la ontología sin realizar; (c) rompe más de lo que arregla.

## 5. Sobre llevar el informe al repositorio

Sí, conviene: `docs/informe_integral_2026-09-22.md` con las tres correcciones de §1, las citas a filas y esta adenda, y con la advertencia en cabecera de que las lecturas interpretativas (las palabras «refutada», «ciego a la conversión») son del informe y no del registro. Es una decisión del autor; la sesión de código ya se ha ofrecido a integrarlo.

---

## 17. Anexo (añadido al integrar): citas a las filas del registro

Cada sección del informe se apoya en filas de `docs/claims_registry.yaml`; lo citable es la fila, con su `status` y su `evidence_level`, no la frase del informe.

| Sección | Filas del registro |
|---|---|
| §1 Plano Dual | `axiomas`, `plano-dual`, `suite-apendice-h` |
| §2 Basal y salida de S₀ | `potencial-basal`, `naturalidad-priors`, `nucleacion-salida-s0`, `nucleacion-gamma0`, `s-clock-simulador-consistencia` |
| §3 Camino y transporte azimutal | `monotonia-exclusion`, `diagonal-j-requerido`, `vacio-2d-conversion-y-rotacion`, `s-clock-simulador-consistencia` |
| §4 Sellos, Década, frente 2 | `decada-discriminante`, `kls-flujo`, `matriz-estabilidad`, `fp-beta-desenlace-a`, `s-clock-simulador-consistencia` |
| §5 Atemporalidad, sellos de c y c² | `florencia`, `mass-gap-reticulo`, `c-lieb-robinson-identificacion`, `s-clock-simulador-consistencia` |
| §6 Florencia | `florencia`, `rp-escalar`, `rp-no-estacionaria` |
| §7 Gea y Atlas | `gea-newton-atlas`, `residuos`, `mu-eta-atlas`, `atlas-ppn`, `atlas-gauge-invariant` |
| §8 Ley de Cronos débil | `cronos-v3`, `ciclo-pm`, `perfil-compuerta`, `jeans-dsph`, `5c-estructural`, `5e-falsacion-cruzada`, `cronos-amplitud-unica`, `mu-eta-cronos`, `nivel-a-halo-aislado`, `criterio-cronos-jeans`, `oort-kz-cota-cronos`, `sculptor-perfil-sigma-los`, `test-criterio-cronos-jeans`, `test-criterio-cronos-jeans-ronda2`, `halo-serie-resolucion-campo`, `halo-capas-ronda2`, `diccionario-epsilon-c-saturante`, `contenido-materia-lectura-operativa` |
| §9 Masas | `funcional-masas`, `higgs-auditoria`, `ngen-3`, `empalme-wkb`, `wkb-calibrado` |
| §10 Victoria y círculo de δ₀ | `ciclo-victoria`, `circulo-delta0`, `fertilidad` |
| §11 Cosmología | `fondo-cosmologico`, `canales-oscuros`, `ajustes-produccion`, `desi-dr2-6a`, `jax-crosscheck`, `cmb-fsigma8`, `perturbaciones-fsigma8-oos`, `dovekie-mocks-pipeline`, `dovekie-real-primera-aplicacion`, `bbn-g-empress`, `mapa-s-z-convencion` |
| §12 Estructura | `cronos-v3`, `nivel-a-halo-aislado`, `halo-serie-resolucion-campo`, `halo-capas-ronda2` |
| §13 Retículo y cuántica | `mass-gap-reticulo`, `qudit-decoherencia`, `qudit-d5` |
| §14 Reloj S | `s-clock-simulador-consistencia`, `constantes-estatuto` |
| §15 Diccionario | `diccionario-primordial-cosmologico`, `diccionario-unidad-S-post-florencia`, `diccionario-epsilon-c-saturante` |

## 18. Nota de verificación de la sesión de código (añadida al integrar, 28-sep-2026)

Sobre `main` = `330b2e4`, con `dynamics.cronos_jeans.nfw_jeans_table` (Jeans isótropo del halo del Nivel A):

- r_CJ(0.05·A_S) = 0.180 kpc y r_CJ(0.01·A_S) = 0.087 kpc se reproducen; r_CJ(10⁻³·A_S) = 0.031 kpc. El exponente efectivo entre 0.05 y 0.01 es 0.45, así que «r_CJ ∝ A^{2/5}» es la escala aproximada que da σ_r² ∝ r, no una ley exacta (con la ley pura, 0.05·A_S daría 0.214 kpc).
- r_CJ(10⁻⁴·A_S) = 0.011 kpc **no es reproducible con el módulo**: su horquilla de raíces empieza en 0.02 kpc y a esa amplitud q < 1 en toda ella. El valor queda como cálculo del autor, sin verificar aquí.
- Los números de la ronda 2 de capas citados en la adenda §2 coinciden con `results/2026-09-22_halo_shells/halo_shells.json`.
