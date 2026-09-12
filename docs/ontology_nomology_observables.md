# Ontología → nomología → observables (vista para la v36)

<!-- GENERADO por scripts/make_traceability.py desde docs/claims_registry.yaml — NO editar a mano: el test de no-divergencia (tests/test_claims_registry.py) falla si este fichero difiere de la regeneración. -->

Cada entidad del programa con su ley, su observable (si lo tiene), el dataset que lo decide y su falsador declarado. Vocabulario de estatutos v35.1; tres categorías negativas DISTINTAS — resultado negativo ≠ claim no derivado ≠ experimento no ejecutado — porque colapsan responsabilidades distintas: lo medido y adverso, lo aún no derivable, y lo bloqueado por datos.

**FRB no entra**: sin dataset con versión/DOI persistente, sin observable consumido por el MCMC y sin relación precisa con el parámetro bariónico que restringiría, es literatura relevante, no evidencia del modelo (control bariónico anclado hoy: kSZ 2604.19744/45).

| Entidad | Ley | Observable | Dataset | Falsador | Estatuto | Categoría | Artefacto |
|---|---|---|---|---|---|---|---|
| Ocho axiomas | §1.2 | — | — | inconsistencia interna entre axiomas y cualquier bloque derivado | ancla conceptual | `interno` | — |
| Constantes por estatuto | Ap. F | — | — | uso de una constante fuera de su estatuto declarado | F.2-F.4 + C6 | `interno` | — |
| Plano Dual | Cap. 2 | — | — | fallo de las identidades del plano en la suite | interna demostrada | `interno` | — |
| Potencial Basal, T₀ = c̄δ₀³ | Cap. 3 | — | — | exponente medido incompatible con 3 | interna demostrada (exponente medido) | `interno` | — |
| Monotonía y Exclusión | Cap. 4 | — | — | contraejemplo numérico de la monotonía | interna demostrada | `interno` | — |
| Cadena de Álgebras y Florencia | Caps. 5-6 | — | — | firma distinta de −+++ en la cadena | interna demostrada (firma −+++) | `interno` | — |
| RP escalar | Cap. 7 (Teo. 7.1) | — | — | violación de RP en el juguete | juguete demostrado | `interno` | — |
| RP no estacionaria | §7.5 (Prop. 7.4, E3) | — | — | contraejemplo de suficiencia; Wilson (frente 1) decidiría necesidad | juguete (especularidad suficiente, necesidad no demostrada); Wilson abierto (frente 1) | `condicional` | results/2026-08-02_rp_nonstationary |
| Década y Discriminante | Cap. 8 | — | — | una derivación de τ(S) que produjera λ ≠ 10 refutaría la selección | interna demostrada; λ=10 calibrado | `claim-no-derivado` | — |
| Flujo KLS integrado | Cap. 8 (frente E) | — | — | exponente distinto de −1/2 al refinar la integración | resultado numérico (−1/2; −1/3 condicional) | `condicional` | results/2026-08-02_kls_flow |
| Matriz de estabilidad | §14.2 (frente 2) | — | — | espectro incompatible al sustituir el ansatz por derivación | instrumentado; ansatz O(1): DSI genérica, λ = 10 selección | `condicional` | results/2026-08-05_victoria_exponent |
| β de Fokker-Planck (frente 2) | Def. 4.4 + §14.2 | — | — | la preinscripción enumeraba desenlaces; A se decidió con el espectro del punto preinscrito | A en el punto preinscrito (δ0 ≈ 1, espectro real); adenda δ0 (covarianza exacta, g_ef = δ0⁻³): cascada + hundimiento de D para δ0 < 0.496 con el cierre canónico — contrafactual B declarado; λ = 10 sigue calibrado (τ no derivable) | `resultado-negativo` | results/2026-08-19_front2_fp_beta |
| Gea, Sello de Newton, Atlas | Cap. 9 | — | — | ruptura del sello exacto en la suite | sello exacto; Atlas condicional | `condicional` | — |
| Contraste de los Residuos | Conj. 9.6 (frente 6) | residuos de escala entre sectores | — | residuo fuera de la banda declarada | consistencia superada (E13); refutable vivo | `interno` | — |
| Ciclo de Victoria | Cap. 10 | — | — | signo de ν contrario al requerido | condicional (signo de ν, frente 4) | `condicional` | — |
| Círculo de δ₀ | Cap. 10 + H.8 | — | — | W_max incompatible con el círculo | condicional (W_max, frente 4) | `condicional` | results/2026-08-02_delta0_circle |
| Fertilidad | H.2.3 (frente 4) | — | — | γ_max fuera del rango declarado al refinar | cartografía; γ_max declarado | `condicional` | results/2026-08-01_fertility |
| Naturalidad y priors | Nota I §3.2 | — | — | cota violada bajo priors alternativos declarados | robusto (cota) / fiducial (mediana) | `interno` | results/2026-08-04_landscape_priors |
| Suite espejo del Ap. H | Ap. H | — | — | cualquier comprobación del Ap. H en rojo | comprobaciones internas (E8) | `interno` | — |
| Cronos v3 | Cap. 11 | — | — | incapacidad de reproducir la compuerta declarada | implementado; producción abierta (frente 5) | `condicional` | — |
| Ciclo PM del Ap. B | Ap. B | — | — | — | prototipo (no producción) | `condicional` | — |
| Perfil y compuerta | H.2.5 (frente 5) | perfil ρ(r) de halos | — | núcleo de producción incompatible con la compuerta | compuerta reproducida; núcleo en producción | `condicional` | results/2026-08-02_profile_shape |
| Potencial débil y Jeans dSph | Def. 11.1 + Prop. 11.2 (frente 5, 5A/5B/5D) | σ_los de esferoidales enanas | walker2009 (DATA_UNAVAILABLE: host denegado) | σ_los binada incompatible con la predicción del montaje | medio paso: veredicto exacto en el montaje declarado; perfil binado y 5C/5E pendientes | `condicional` | results/2026-08-10_jeans_dsph |
| 5C estructural y objetivo ρ_id | Cor. 11.3c + Ap. A (frente 5, 5C/5E) | curvas de rotación de discos | sparc (DATA_UNAVAILABLE: host denegado) | forma medida incompatible con shape_x5 ≈ 5 | parcial: ley de forma exacta; catálogo SPARC pendiente de ingesta (proxy) | `condicional` | results/2026-08-14_sparc_structural |
| Falsación cruzada 5E (A congelada) | frente 5, 5E | Δχ² por galaxia (bariones vs bariones + Cronos) | sparc + walker2009 (DATA_UNAVAILABLE; congelado, sin mirrors) | mediana(Δχ²) > 0 con CI95 excluyendo 0 y frac_improved < 1/3 (camino A preinscrito) | preinscrito; DATA_UNAVAILABLE — observacional abierto (proxy) | `experimento-no-ejecutado` | results/2026-08-16_front5e_sparc |
| DESI DR2 BAO (frente 6A) | Ap. A + registro de datos | DH/rd, DM/rd, DV/rd (13 componentes DR2) | desi_dr2_bao (AVAILABLE @ commit fijado) | la señal exigiría ΔAIC/ΔBIC pro-MCMC con ε identificada; se observó lo contrario | likelihood validada (Cobaya 3.6.2); benchmark 0.04σ (fondo normalizado); contraste uniforme pro-ΛCDM; ε no sondeable por BAO en esta parametrización (z_trans ≈ 8.9; estructural, computado en el artefacto) | `resultado-negativo` | results/2026-08-18_desi_dr2_background |
| Crosscheck JAX del fondo (6A-P3) | Ap. A + A.7 | — | — | puertas de equivalencia violadas al re-ejecutar | equivalencia bajo puertas predeclaradas (PASS: álgebra 4e-16, vector 1.2e-14, \|Δχ²\| ≤ 2.2e-12 en los argmin publicados y ≤ 6.9e-11 incluyendo el rincón de estrés); integrador DESI de producción cazado (4.9e-8) y corregido a O(h⁴); integrador SNe medido (3e-6, sin puerta) — interna (E8) | `interno` | results/2026-08-19_jax_crosscheck |
| Fondo cosmológico | Ap. A | H(z) | — | H(0) ≠ H0 o pérdida del límite ΛCDM exacto | límite de recuperación exacto (Prop. A.1) | `interno` | — |
| Ajustes de producción v1/v2 | A.6 → E5 | H(z) CC + μ(z) SNe + BAO | pantheonplus + boss_eboss (AVAILABLE) | la ventaja del corpus (A.6) exigía ΔBIC pro-MCMC; se midió lo contrario | no favorecido (ΔBIC +14.5), publicado | `resultado-negativo` | results/2026-08-10_production_fit_v2 |
| CMB comprimido y fσ8 | Ap. A (opción B) | (R, l_A, ω_b) + fσ8(z) | planck2018 comprimido + compilación RSD (AVAILABLE, legacy) | incompatibilidad de la geometría comprimida con el fondo ajustado | salvedades declaradas (HS96, diagonal) | `interno` | — |
| Canales oscuros | Ap. A.1 | — | — | — | esquema de dos canales | `interno` | — |
| Funcional del Camino (masas) | Cap. 12 | cocientes de masas SM | — | incapacidad de acomodar una masa medida dentro del esquema calibrado | calibrado; carácter predictivo por establecer (12.5) | `calibrado` | — |
| Higgs y auditoría | Prop. 12.1 / Obs. 12.2 | m_H | — | — | identidad SM calibrada, no predicción | `calibrado` | — |
| Ngen = 3 | Prop. 12.4 → E4 | número de generaciones | — | construcción explícita con Ngen ≠ 3 en la misma álgebra | identificación estructural, sin test dedicado (hueco declarado; sub-frente del 7) | `hueco-declarado` | — |
| Empalme C¹ y WKB ab initio | H.2.4 (frente 7) | — | — | δ_H incompatible al refinar el empalme | no circular; mide δ_H (≠ δ₀*, C6); valor de β₃ condicional | `condicional` | results/2026-08-01_empalme_wkb |
| WKB calibrado (v32) | Tabla 3 v32 | — | — | — | entradas calibradas, no derivación | `calibrado` | — |
| Mass gap en retículo | Ap. D | — | — | piso espectral que desaparezca al refinar el retículo | piso espectral efectivo (E10), no Yang-Mills | `interno` | — |
| Qudit y decoherencia | Ap. C | — | — | — | firma de consistencia del canal (E10) | `interno` | — |
| Qudit d=5 | Ap. C.1 | — | — | — | base implementada | `interno` | — |

## Resultados negativos (experimento EJECUTADO, desenlace nulo o contrario, publicado como tal)

- **β de Fokker-Planck (frente 2)** (fp-beta-desenlace-a): derivación validada por la suite (álgebra ⟺ jacobiano ⟺ suavizado gaussiano ⟺ FP exacta vía Hopf-Cole). RESULTADO PRIMARIO, PREINSCRITO Y NEGATIVO — desenlace A: espectro enteramente real en el punto preinscrito (silla sin rotación, sin cascada DSI). Par complejo solo en 1/21 puntos del barrido (frontera exacta g* = 8.189; τ* ≈ 0.2326 en el único punto con cascada). La adenda δ0 identifica una región físicamente interesante con par complejo pero NO reclasifica el desenlace A; es vía abierta posterior, no éxito retrospectivo.
- **DESI DR2 BAO (frente 6A)** (desi-dr2-6a): RESULTADO NEGATIVO EJECUTADO Y PUBLICADO: el contraste uniforme es pro-ΛCDM (ΔAIC/ΔBIC positivos) y ε_Λ está DOMINADA POR EL PRIOR — BAO-only no la identifica en esta parametrización (z_trans ≈ 8.9 deja la transición fuera del rango efectivo BAO; estructural, computado en el artefacto). El benchmark del brazo ΛCDM propio quedó a 0.042σ del oficial.
- **Ajustes de producción v1/v2** (ajustes-produccion): RESULTADO NEGATIVO EJECUTADO Y PUBLICADO: repetición con el fondo normalizado (10-ago-2026, mismas semillas) mantiene el veredicto pro-ΛCDM (ΔAIC = +4.00/+4.08, ΔBIC = +14.50/+14.60); ε compatible con 0.

## Claims no derivados (la afirmación existe; su derivación, no — distinto de un negativo: no hay test ejecutable que la decida aún)

- **Década y Discriminante** (decada-discriminante): λ = 10 NO derivada: depende de τ(S), que no está derivado — calibración, no predicción

## Experimentos no ejecutados (protocolo preinscrito, datos ausentes, fallo cerrado — sin veredicto en ningún sentido)

- **Falsación cruzada 5E (A congelada)** (5e-falsacion-cruzada): protocolo COMPLETO preinscrito (A = 6.201589e-7 congelada desde Sculptor, regla de decisión ejecutable) y candado anti-tuning activo; SIN VEREDICTO: los bytes oficiales de SPARC/Walker no están ingeridos (fallo cerrado, sin mirrors). No es un resultado negativo: el experimento no se ha ejecutado.
