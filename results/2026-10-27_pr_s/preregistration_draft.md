# PR-S — BORRADOR de la preinscripción (no congelado)

> BORRADOR — no congelado; no es una preinscripción. Se congela solo con las decisiones del autor (decisiones_autor.json) y antes de cualquier byte de SPARC o de Li+2020.

Generado por `scripts/run_pr_s.py draft` en el commit `390e8292dd54307b8df96c1d7d49ad713828d757` (2026-10-28).

## Decisiones pendientes del autor

| Id | Decisión | Recomendación | Alternativa | En el código |
| --- | --- | --- | --- | --- |
| 27oct-1 | Enmiendas 1–4 (σ_x/m efectivo y alcance de las letras; memoria del halo; rama densa y colapso; lapse) | las cuatro, antes de congelar | dejarlas para otra ronda, con el riesgo de leer una A sin su alcance | las cuatro están implementadas; ninguna toca la letra |
| 27oct-2 | Relación de época de formación para t_edad | c = c₁·a₀/a_c de Wechsler et al. (2002) con c₁ congelado antes de los bytes (Wechsler: c₁ ≈ 4.1 para c_vir) | tres valores fijos (6, 10 y 13 Gyr) como sensibilidad, sin relación | los dos modos; falta el valor de c₁ |
| 27oct-3 | Rama densa | lectura de coherencia con t_c y la variante restringida (solo t_c ≤ t_edad) como sensibilidad, sin letra | publicar solo la fracción | las dos (mcv_sparc.restricted_fit) |
| 27oct-4 | Ley local en la lapse de la MCV | mantener la separación del 18-oct con la lectura cruzada de q_CJ(0) | brazo combinado sin letra con la forma saturante de ρ* = 0.01–0.03 | la lectura cruzada; el brazo combinado no (ver el hallazgo H3, que cambia su premisa) |
| 26oct-1 | J.6.2 en el banco | control MCMC-J62 con r⋆ libre en A; el factor de época, señalado como incoherencia del Maestro | retirarlo y comparar solo la pendiente | el brazo MCMC-J62 con r⋆ en rejilla |
| 26oct-2 | Forma de σ_x/m(v) | ley de potencias (σ₁₀₀, a) con la constante como sub-brazo anidado | solo constante | la rejilla (σ₁₀₀, a) completa; a = 0 es la constante |
| 26oct-3 | Reparto A/B | por sha256 del identificador | aleatorio con semilla congelada | los dos (sparc_bank.split_halves) |
| 26oct-4 | Letras del escalón | lectura aditiva con ventanas; la literal, aparte | solo la literal | las dos (dynamics/rar_step_letters.py) |
| 26oct-5 | Previas del halo | c–M de Dutton y Macciò (0.11 dex), las de Sculptor | la c–M de 2610.05871, si difiere | DM14; la otra exige transcribirla |
| 26oct-6 | Lo que pide #65 (afueras, E3, datos del fondo) | afueras = capa del borde del código; E3 fuera hasta tener vacíos; datos del fondo, del autor | — | no es de PR-S |

## Decisiones del 28-oct (no bloquean PR-S)

| Id | Decisión | Recomendación | En PR-S |
| --- | --- | --- | --- |
| 28oct-1 | Lectura de la energía sellada | (a) operativa para los ajustes y (b) como frente de derivación de C(S), sin decidir la ley de Cronos antes de Walker | nada: PR-S usa (a) |
| 28oct-2 | Ley constante de la evaluación | brazo de control en el catálogo de #65, como ξ_0, fuera de PR-S | nada (fuera de PR-S y no ahora) |
| 28oct-3 | S_local en PR-S | lectura operativa con t_edad(c) y ΔN(0), sin letra | texto «S_local operativa» (punto 10), sin parámetros nuevos |
| 28oct-4 | Respaldo de la rama | sí, con las cinco condiciones | hecho: mcv/pr-s-sparc-instrument-wip, commit «borrador sin congelar» |
| 28oct-5 | Nombres | ξ_0 para la ley constante, q_dec para la deceleración y q_CJ(0) en PR-S | q_CJ(0) en el código y en el texto; ε solo con subíndice (punto 12) |

## Hallazgos del ejecutor

| Id | Hallazgo | Opciones | Efecto |
| --- | --- | --- | --- |
| H1 | Con la rejilla sola, χ²_mín queda cuantizado: el paso de las molestias (Υ y D en ±2σ) y el de (ρ₀, σ₀) mueven χ²_mín más de 0.5 en galaxias con errores pequeños. | rejilla sola (documento), o rejilla y refinado local en rejillas que se estrechan (propuesta, FIT_SPEC['zoom']) | la comprobación de resolución de scripts/run_pr_s.py checks da las cifras |
| H2 | En las letras del escalón, δ_pred = log(1 + g_C/g_obs) del documento queda por debajo del escalón que deja la propia ley aditiva cuando g_C ~ g_obs (ρ* = 0.001): la A sería inalcanzable. | «documento», o «autoconsistente», log[g_obs/(g_obs − g_C)] | solo ρ* = 0.001 |
| H3 | Para la forma saturante, 1.5·c²·ε_c/σ₀² (fórmula de C3 y de la enmienda 7) no es el q que decide la estabilidad de Cronos–Jeans, c²·ρ·ε_c'(ρ)/σ₀²; coinciden solo en la ley débil. Con ρ ≫ ρ* el segundo es cientos de veces menor: con ρ* = 0.001, q < 1 en vez de 0.5–9.3. | publicar los dos (hecho) | cambia la premisa de la alternativa de la decisión 27oct-4 |
| H4 | El punto fijo t_edad = t_W(c(t_edad)) tiene dos raíces por núcleo: una junto al borde (c ≈ c₁, t_W → 0, inestable) y la estable; las dos son envolturas autoconsistentes. | todas, y el ajuste toma el mejor χ² (hecho), o solo la estable | lectura de memoria |
| H5 | «ECV por debajo de 1e-4 de la atracción dentro de 10·R_d» depende del tamaño del disco (1.7e-4 con R_d = 2 kpc en 1e11 M☉); la forma de C1 (≤ 6e-5 hasta 0.1·r200) se cumple. | test con C1 (hecho) | ninguno sobre las letras |
| H6 | El candado anti-ajuste del 5E vigila dynamics/*sparc*.py y prohíbe la maquinaria de optimización: la física general del brazo (raíces del empalme) va en dynamics/mcv_baryons.py y el ajuste a SPARC es en rejilla. | mantener el candado tal cual (hecho), o ampliarlo a mano | ninguno |
| H7 | Precisiones del ejecutor que el documento no fija: χ² = datos + previas (−2 ln de las previas); ΛCDM-1 con log c ∈ [0, 2]; Υ_b solo con bulbo; r₁ < r200; rangos de (ρ₀, σ₀); paso de los nodos (0.04 en ln r); χ²_ν de E8-Q = χ²_datos/(N − k_halo); la a₀ poblacional con la RAR de un parámetro. | aceptarlas o fijar otras | todas entran en la preinscripción |
| H8 | Faltan entradas del corpus y de la literatura: la Tabla 45 del Unificado, el valor de c₁, las previas exactas de Li+2020 y el montaje de 2610.05871 (arxiv.org no es accesible desde la sesión). | que el autor las transcriba | E8-Q y la lectura r_c(M200) fallan cerrado sin ellas |
| H9 | Con discos sintéticos de tres tamaños (0.5, 1 y 2 veces R_ref) y h_z = 0.15·R_d, el escalón previsto S_pred vale 0.30 dex con ρ* = 0.001 y 0.024–0.032 dex con ρ* = 0.01–0.03, entre 2 y 8 veces lo del documento del 26-oct (0.10–0.14 y 0.004–0.007, con los discos de la predicción congelada): g_C ∝ 1/R_d crece en los discos compactos. Con σ_S ≈ 0.005 dex, el escalón con ρ* = 0.01–0.03 podría ser comprobable. Con ρ* = 0.001, la ley predice g_C ≥ g_obs en la mitad de esos discos. | revisar la expectativa E13 del escalón con el reparto de R_d de SPARC antes de congelar, o dejarla como está | solo la expectativa E13; las letras no cambian |

## Alcance de cada letra

- **MCV (c), letra sobre la mitad B.** Prueba que la MCV en equilibrio por autointercambio, con un σ_x/m(v) único para todas las galaxias, describe los discos fuera de la muestra. No puede decir: nada sobre el intercambio con la ECV, el borde R₀, el punto de ruptura ni el origen de la MCV; un modelo de materia oscura autointeractuante sin ontología daría la misma letra (Ren et al. 2019).
- **MCMC-C (5E).** Prueba la ley local ε_c = A_S·ρ^(3/2) con bariones solos y A congelada. No puede decir: nada sobre la MCV (c); A_S ya está «excluida bajo regla congelada» por Oort–K_z, con bytes no verificados.
- **Escalón de la RAR.** Prueba la forma saturante de ε_c y su densidad ρ*. No puede decir: con ρ* = 0.01–0.03 la letra esperada es INDETERMINADO por no comprobable; no informa sobre la MCV.
- **MCMC-J62.** Prueba el tratado escrito (Maestro J.6.2), con r⋆ libre. No puede decir: no es la MCV (c): su pendiente 1/3 no sale del equilibrio.
- **Lecturas sin letra.** Prueba universalidad de σ_x/m, proximidad al colapso (rama densa), escala r_c(M200), cruce con Sculptor, historia física acumulada del halo (memoria), lapse y q_CJ(0) cruzado. No puede decir: ninguna da letra.

## Lecturas sin letra

- globalidad de σ_x/m (calibración con B dentro de la región del 95 % de A)
- ramas: fracción en la rama densa y con c a más de 2σ, frente al brillo superficial
- r_c(M200) frente a J.6.2 (control), la pendiente 0.35 de U (166), «∼0.4 en SPARC» y la Tabla 45 del Unificado
- SPARC → Sculptor: σ_x/m(v) calibrado en el ⟨v_rel⟩ de Sculptor
- memoria del halo: t_edad por galaxia desde la concentración (c₁ congelado), recalibración en A, prueba en B, Δχ² frente a t_edad = 10 Gyr y cambio de la calibración; o 6/10/13 Gyr
- rama densa y colapso: t_c por galaxia (β = 0.75 y 0.45) con el σ_x/m calibrado y la envoltura ajustada; la rama densa con t_c > t_edad, marcada; variante con la rama densa solo si t_c ≤ t_edad
- lapse: ΔN(0) = |Φ(0)|/c² y σ₀ por galaxia
- q_CJ(0) cruzado (antes q0): si el 5E o el escalón reciben A, q_CJ(0) = 1.5c²ε_c/σ₀² (y c²ρε_c'/σ₀²) de cada galaxia
- banco: BIC con k completo, preferencias por pares, ΔBIC frente a ΛCDM-2, a₀ por galaxia frente al brillo

## Cosmología fija y declarada

- fondo: PLANCK18 de dynamics/ecv_border.py
- h: 0.674
- Omega_m: 0.315
- Omega_L: 0.685
- rho_E: Ω_Λ·ρ_crit,0, constante
- H0: H0_DEFAULT = 67.4 de dynamics/mcv_equilibrium.py
- leyes_de_Q_o_de_H: ninguna (ni las de #65 ni la constante de la evaluación)
- t_edad: con el t(a) de ese fondo; retroceso hasta z = 1 = 7.95 Gyr
- motivo: con la ley constante de la evaluación (ξ_0 = 0.06), R₀ en z = 1 cambiaría ×0.986 y t_edad entre +0.15 % y +0.74 %, frente al ±29 % de la c–M y al 30–55 % que la edad mueve el núcleo

## S_local operativa

S_local = ∫N·dS (Fundamentos, 6.3). En PR-S su análogo operativo es t_edad(c), la historia física acumulada del halo, y ΔN(0) mide cuánto más despacio corre el centro: S_local y S_g coinciden a 3e-6. Sin letra y sin parámetros nuevos.

## Símbolos

ε solo con subíndice (ε_c, ε_max); el q₀ del 27-oct es q_CJ(0), el del criterio de Cronos–Jeans; q no se usa para la deceleración.

## Respaldo

rama mcv/pr-s-sparc-instrument-wip (commit «borrador sin congelar», paso 0 del bloque del 28-oct); la congelación se hace en mcv/pr-s-sparc-instrument, con el texto final y antes de los bytes.

## Fuera y declarado

- cuspide_del_agujero_negro: la cúspide de MCV junto al agujero negro (ec. 5 del bloque) no entra en PR-S: con M_BH de 1e5–1e6 M☉ y σ₀ = 20–50 km/s, r_h = 0.2–11 pc; con 1e7–1e8 M☉ y σ₀ = 100–150 km/s, r_h = 2–43 pc; como mucho 43 pc, por debajo de los radios de SPARC
- gap: un gap que cambiara las masas de forma diferencial está acotado en torno a 1e-7 (Δμ/μ en z = 0.89, Bagdonaite et al. 2013, PRL 111, 231101); uno universal lo absorben Υ y las unidades: no entra en PR-S
- cronos/halo_profile.py: fuera de PR-S: calibración fiducial sin origen en el corpus (12.6 kpc en 1e11 M☉)
- intercambio_con_la_ECV: no se ve en SPARC (C1); se prueba en el fondo y en el borde

## Expectativas E13 (antes de congelar)

- MCV (c): A o B (Ren et al. 2019 ajustan 135 galaxias con σ/m = 3 cm²/g y la c–M de Planck).
- Riesgo: si SPARC calibra σ_x/m ~ 2–3 cm²/g con a ≥ 0, la rama con núcleo de Sculptor se queda corta; en la rama densa, Sculptor tendría t_c de 150–740 Gyr (incoherente salvo que σ_x/m crezca mucho a baja velocidad).
- Escalón: con ρ* = 0.001, comprobable; con ρ* = 0.01–0.03, INDETERMINADO por no comprobable (documento del 26-oct, con los discos de la predicción congelada). Salvedad: con discos compactos el escalón previsto es entre 2 y 8 veces mayor y podría ser comprobable (hallazgo H9).
- Lectura literal del escalón: C.

## No se toca

- 5E (ley local con A_S): `results/2026-08-16_front5e_sparc/preregistration.json`, sha256 `4b2ecdff2f39cbd0e1f8326989cd90fef3621eab0de8ad736f143e0e615ad704`
- Sculptor con la MCV (c): `results/2026-10-23_mcv_sculptor/preregistration.json`, sha256 `2ed76f7f0bf4e8c5364d5bc9811a67e7712bdac95bc915ccf22093f7c4aee006`
- escalón de la RAR (predicción): `results/2026-09-22_dictionary_eps_c/rar_step_prediction.json`, sha256 `9566a2ac04570e07b3b588cb73abbef774fc4545265aad54db5e401bfb074548`
- A_Sculptor: 6.201588782789579e-07
- ρ*, q_S y la RAR congelada: intactos

El resto (brazos, especificaciones, letras y escalón) está en `preregistration_draft.json`.
