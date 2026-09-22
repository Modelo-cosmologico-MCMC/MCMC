# El reloj S como simulador de consistencia — recorrido S₀ → S_{1,001} (21-sep-2026)

Commit `41af2700f`. Modo con estatuto: umbrales IMPUESTOS (Prop. 8.1). δ₀ primario = 0.01 (valor de prueba: el tratado no asigna valor a δ₀). Formas del Basal: m̄ = 1.0, b̄ = 3.0, ē = 1.0, C0 = 1.0.

**Lectura obligatoria**: simulador de consistencia (E8): umbrales de la Década impuestos (Prop. 8.1, λ = 10 calibrado — frente 2), nucleación declarada en el punto de escape, sellados con forma paramétrica declarada, diagonal impuesta para la entrega; el estado entregado en S = 1,001 no es legible por la cosmología (diccionario ausente).

## Hallazgo del simulador: la metastabilidad tiene un δ₀ máximo

Con la inclinación −η·χ (η = ē·δ₀³) el falso vacío deja de existir por encima de **δ₀_max = 0.1028** para las formas por defecto: la Obs. 8.6 (D(S₀) > 0) es necesaria, no suficiente. Por debajo, el falso vacío está en ρ_fv ≃ η/(√2·M0²) sobre el eje θ = 0 (residuo ⟨χ⟩ de la Prop. 3.4) y la barrera vale 0.054·T₀ en δ₀ = 0.01. T₀ medida en el paisaje completo supera la ley c̄·δ₀³ en 13.3 % (corrección de la inclinación, O(δ₀^{1/2}) relativa; la ley sin inclinación se reproduce a 1e-15).

## Recorrido primario

| estación | S (reloj f) | S = ∫Σ̇dσ/T₀ | σ | álgebra | firma | estatuto |
|---|---|---|---|---|---|---|
| colapso_1D | 0.009 | 0.00891 | 322.8 | C(2,0) | [1, 1] | impuesto (Prop. 8.1) |
| colapso_2D | 0.099 | 0.09883 | 2042.1 | C(3,0) | [1, 1, 1] | impuesto (Prop. 8.1) |
| colapso_3D | 0.999 | 0.99899 | 8269.0 | C(4,0) | [1, 1, 1, 1] | impuesto (Prop. 8.1) |
| V3D | 1.000 | — | 8270.0 | — | — | declarado (cuanto de anticipación/confirmación, Prop. 8.1) |
| Florencia | 1.001 | — | 8270.0 | C(3,1) | [-1, 1, 1, 1] | derivado (giro, firma, RP juguete, identidad) — evento declarado en S = 1,001 |

Identidad S ≡ f (Teo. 4.5, flujo proyectado): |S − f|_max = 1.9e-13. Monotonía: max ΔV/T₀ = -4.7e-05; producción entrópica mínima 2.7e-12; exclusión: sí; salida al polo de masa: θ_final = 0.000. Descenso hasta f = 1 − ε_res en σ̂ = σ·δ₀² = 0.827 (1654 pasos).

Sellados (forma declarada): c_eff congelado en S = 0.099 con u = 0.1289 (máximo |du/dS| = 1.526 en S = 0.045); m_eff congelado en S = 0.999 con m = 0.5091.

Diagonal: el flujo NO la cruza (θ_max = 0.000); para la entrega se impone θ_imp(S) con cruce en S = 1.0.

Florencia (S = 1,001): C(4,0) → C(3,1), firma [-1, 1, 1, 1]; control negativo (dos giros): [-1, 1, 1, -1]; RP en la loncha: autovalor mínimo -1.7e-16 (J = +1) frente a -0.021 (J = −1, control negativo); identidad m_H = √(2β₃)v₃ comprobada con β₃ = 0.0224 ⟹ m_H = 52.0 GeV en δ₀ = 0.01 (depende de δ₀: no es predicción).

Apretón de manos con la cosmología: Sello de Newton G_cosmo/G_N = 1.0 en (1,1); Atlas sano en λ_K = 1 + ε_K; recuperación de ΛCDM con ε = 0: desviación máxima 0.0e+00. **Diccionario primordial → cosmológico: NO EXISTE** — el estado entregado no es legible por `cosmology/`.

## Barrido en δ₀

| δ₀ | metastable | barrera/T₀ | corrección inclinación | σ̂ descenso | m_H [GeV] | diagonal por flujo |
|---|---|---|---|---|---|---|
| 0.0030 (valor de prueba) | sí | 0.0754 | +7.4 % | 0.763 | 28.5 | no |
| 0.0100 (valor de prueba) | sí | 0.0537 | +13.3 % | 0.827 | 52.0 | no |
| 0.0300 (valor de prueba) | sí | 0.0269 | +22.7 % | 0.990 | 90.1 | no |
| 0.0581 (δ_H (empalme H.2.4, valor requerido)) | sí | 0.0099 | +31.1 % | 1.319 | 125.4 | no |
| 0.1000 (valor de prueba) | sí | 0.0001 | +39.9 % | 5.640 | 164.5 | no |
| 0.2000 (valor de prueba) | NO | — | — | — | — | — (sin falso vacío metastable (δ0 > δ0_max = 0.1028): la inclinación borra la barrera) |

m_H(δ_H) reproduce el PDG por construcción de δ_H (valor requerido, auditoría de circularidad Obs. 12.2), no por predicción.

## Modo emergente (diagnóstico, sin estatuto)

Acoplos por las β de Fokker–Planck (cierre canónico) con diccionario τ declarado; reloj S = ∫Σ̇dσ/T₀; colapso donde D = 0.

| δ₀ | τ | D inicial | D mínimo | cruces D = 0 (S) | S final | fin |
|---|---|---|---|---|---|---|
| 0.01 | 0 | 5.00e-04 | 5.00e-04 | — | 0.9990 | descenso completado (residuo de descarga alcanzado) |
| 0.01 | 0.001 | 5.00e-04 | 3.69e-04 | — | 0.6241 | max_steps agotado |
| 0.01 | 0.01 | 5.00e-04 | 2.26e-04 | — | 0.1766 | max_steps agotado |
| 0.01 | 0.1 | 5.00e-04 | 2.04e-04 | — | 0.0251 | max_steps agotado |
| 0.01 | 1 | 5.00e-04 | 2.42e-04 | — | 0.0028 | max_steps agotado |
| 0.05 | 0 | 1.25e-02 | 1.25e-02 | — | 0.9990 | descenso completado (residuo de descarga alcanzado) |
| 0.05 | 0.001 | 1.25e-02 | 1.15e-02 | — | 0.8927 | max_steps agotado |
| 0.05 | 0.01 | 1.25e-02 | 7.74e-03 | — | 0.4953 | max_steps agotado |
| 0.05 | 0.1 | 1.25e-02 | 5.04e-03 | — | 0.1180 | max_steps agotado |
| 0.05 | 1 | 1.25e-02 | 5.13e-03 | — | 0.0157 | max_steps agotado |

Lectura: con el cierre canónico D se hunde (Obs. 8.6) pero en ninguna corrida cruza cero antes del fin del recorrido: para τ > 0 M0² cruza cero ANTES que D (el falso vacío se destabiliza por la masa, no por la espinodal; evento `inestabilidad_masa`, tabla siguiente) y el bucle sigue sobre un potencial aplanado hasta agotar el presupuesto de pasos con D > 0. En el rango explorado los colapsos NO emergen del Cruce de Victoria — resultado diagnóstico que depende del diccionario τ (frente 2), publicado sin verdicto.

## v1 (21-sep, tarde): corrección de la inclinación a primer orden

κ₁ = ē·√κ₊/(√2·c̄) = **1.361** con las formas por defecto; T₀_full/T₀_ley − 1 = κ₁·√δ₀ + O(δ₀).

| δ₀ | medido | κ₁√δ₀ | cociente | residuo/δ₀ |
|---|---|---|---|---|
| 0.0030 | 0.0737 | 0.0745 | 0.989 | -0.273 |
| 0.0100 | 0.1333 | 0.1361 | 0.980 | -0.274 |
| 0.0300 | 0.2273 | 0.2357 | 0.964 | -0.280 |
| 0.0581 | 0.3113 | 0.3281 | 0.949 | -0.288 |
| 0.1000 | 0.3994 | 0.4303 | 0.928 | -0.309 |

## v1: el Techo W_max con las dos T₀ (círculo de δ₀)

En δ_H = 0.0581: ley 3.4 c̄·δ_H³ = **1.6523e-04**; paisaje completo V_fv − V_tv = **2.1667e-04** (+31.1 %). En 0.012: 1.4530e-06 frente a 1.6648e-06. Cuál nombra el Lema 10.3 es decisión del autor (A); la fila circulo-delta0 hereda la corrección como condicional, no la resuelve.

## v1: δ₀_max en forma cerrada y naturalidad

δ₀_max = 2·g_max(m̄, b̄, C0)²/ē² ∝ ē⁻² (el origen es el falso vacío exacto sin inclinación: δ₀_max = ∞ con ē = 0, raíz que la malla de la v0 no veía).

| ē | bisección | forma cerrada | δ₀_max·ē² |
|---|---|---|---|
| 0.5 | 0.41113 | 0.41113 | 0.10278 |
| 1.0 | 0.10278 | 0.10278 | 0.10278 |
| 2.0 | 0.02570 | 0.02570 | 0.10278 |

Sobre los paisajes viables de `landscape_priors` (n = 4000 por prior; filtro fértil como en la cartografía):

| prior | fracción con falso vacío en δ₀ = 0.012 | en δ₀ = δ_H = 0.0581 | mediana δ₀_max | forma cerrada vs bisección |
|---|---|---|---|---|
| uniform | 0.557 | 0.257 | 0.0160 | 5.9e-06 |
| loguniform | 0.508 | 0.206 | 0.0123 | 8.6e-06 |
| normal | 0.739 | 0.377 | 0.0347 | 5.8e-06 |

Lectura: la metastabilidad en δ_H NO es genérica sobre el prior (una fracción minoritaria de paisajes la admite); cartografía publicada sin veredicto — el prior es declarado.

## v1: inestabilidad de masa (modo diagnóstico) y la tabla τ_k (E13)

En modo emergente el cruce M0² = 0 se publica como evento y el bucle continúa (se detiene solo con C0 ≤ 0). Forma cerrada de primer orden: S_flip ≃ δ₀·[b̄ − √(b̄² − 6C0m̄²)]/(24·a·τ·C0) = 0.1057·δ₀/τ.

| δ₀ | τ | S_flip medido | S_flip 1er orden | cociente | S final | fin |
|---|---|---|---|---|---|---|
| 0.01 | 0 | — | — | — | 0.9990 | descenso completado (residuo de descarga alcanzado) |
| 0.01 | 0.001 | — | — | — | 0.6241 | max_steps agotado |
| 0.01 | 0.01 | 0.10556 | 0.10566 | 0.999 | 0.1766 | max_steps agotado |
| 0.01 | 0.1 | 0.01049 | 0.01057 | 0.993 | 0.0251 | max_steps agotado |
| 0.01 | 1 | 0.00100 | 0.00106 | 0.944 | 0.0028 | max_steps agotado |
| 0.05 | 0 | — | — | — | 0.9990 | descenso completado (residuo de descarga alcanzado) |
| 0.05 | 0.001 | — | — | — | 0.8927 | max_steps agotado |
| 0.05 | 0.01 | — | — | — | 0.4953 | max_steps agotado |
| 0.05 | 0.1 | 0.05278 | 0.05283 | 0.999 | 0.1180 | max_steps agotado |
| 0.05 | 1 | 0.00525 | 0.00528 | 0.994 | 0.0157 | max_steps agotado |

Tabla τ_k — el τ que haría caer la inestabilidad de masa en cada umbral de la Década, τ_k = S_flip(τ = 1)·δ₀/S_k (≈ 11.8·δ₀·10⁻ᵏ): es el diccionario τ leído al revés desde los umbrales calibrados, **no una derivación** (E13: el número no es señal).

| δ₀ | τ₀ (S = 0.009) | τ₁ (S = 0.099) | τ₂ (S = 0.999) |
|---|---|---|---|
| 0.01 | 1.1740e-01 | 1.0673e-02 | 1.0577e-03 |
| 0.05 | 5.8701e-01 | 5.3365e-02 | 5.2884e-03 |

Hipótesis declarada τ_d (opcional): τ por dimensión igual a la tabla τ_k y M0² repuesto en cada inestabilidad — publica dónde caerían los colapsos:

- δ₀ = 0.01: colapsos en S = [0.0089] con disparos ['inestabilidad de masa M0² = 0 (hipótesis τ_d declarada, E13)']; fin: max_steps agotado.
- δ₀ = 0.05: colapsos en S = [0.009] con disparos ['inestabilidad de masa M0² = 0 (hipótesis τ_d declarada, E13)']; fin: max_steps agotado.

## v2 (22-sep): las decisiones A, B y C del autor, ejecutadas

**A — qué T₀ nombra el Lema 10.3: la del paisaje completo.** El contenido físico del Lema es T₀(δ′) ≤ W_max; c̄δ′³ es la expresión de la Prop. 3.4, cuya corrección el reloj midió. Con la inclinación en los dos lados (Techo y empalme H.8), δ_H se recalcula como raíz de λ_Ad_full(δ) = λ_H:

| cadena | δ_H | W_max requerido |
|---|---|---|
| ley 3.4 en δ_H_ley | 0.05814 | 1.6523e-04 |
| paisaje completo en δ_H_ley | 0.05814 | 2.1667e-04 |
| **decidida**: paisaje completo en δ_H_full | **0.05544** (-4.6 %) | **1.8695e-04** |

el Lema 10.3 iguala W_max a T₀_full; con δ_H recalculado sobre el mismo paisaje (λ_Ad_full = λ_H) el Techo requerido es T₀_full(δ_H_full); se publican las tres cadenas para la trazabilidad.

**B — n = 1 con Kramers (Axioma 4) y Gamow como cota; el bounce es control negativo.** D_ent DECLARADA como fracción de la altura de la barrera:

| δ₀ | D_ent/ΔV_b | prefactor | ΔV_b/D_ent | Γ_K | σ_espera = 1/Γ_K | Γ₀ Gamow (ref.) | S ≡ f |
|---|---|---|---|---|---|---|---|
| 0.0100 | 0.1 | 1.848e-05 | 10.00 | 8.392e-10 | 1.192e+09 | 1.554e-03 | 1.9e-13 |
| 0.0100 | 1 | 1.848e-05 | 1.00 | 6.800e-06 | 1.471e+05 | 1.554e-03 | 1.9e-13 |
| 0.0100 | 10 | 1.848e-05 | 0.10 | 1.673e-05 | 5.979e+04 | 1.554e-03 | 1.9e-13 |
| 0.0554 | 0.1 | 3.795e-04 | 10.00 | 1.723e-08 | 5.804e+07 | 7.363e-03 | 2.1e-13 |
| 0.0554 | 1 | 3.795e-04 | 1.00 | 1.396e-04 | 7.163e+03 | 7.363e-03 | 2.1e-13 |
| 0.0554 | 10 | 3.795e-04 | 0.10 | 3.434e-04 | 2.912e+03 | 7.363e-03 | 2.1e-13 |

Control negativo: CONTROL NEGATIVO (decisión B): bounce O(4), excluido por la ontología (sin soporte espacial en S₀) (f₀ = 0.9996: la descarga entera caería dentro de la nucleación). Γ_K(0) = 0 por el prefactor ∝ δ₀²; D_ent no la fija el corpus (diccionario τ, frente 2): la ley de Γ₀ es condicional a D_ent, no a n_dim.

**C — «el colapso» es D = 0 (Cruce de Victoria).** bajo el cierre canónico D no cruza cero en ninguna corrida; M0² = 0 sí ocurre y se publica como diagnóstico (no-evento para el estado ocupado); el colapso del tratado exige β del frente 2 (cruces D = 0 en las corridas emergentes: ninguno; eventos M0² = 0: 5).

## Lo que NO afirma

- ningún umbral emerge: los colapsos se disparan en 0.009/0.099/0.999 por la Ley de la Década (calibrada, frente 2)
- la nucleación (Γ₀) no se calcula en la corrida primaria: el reloj arranca en el punto de escape con σ = 0 declarado; con la decisión B (n = 1) Γ₀ se publica en modo 'kramers' (D_ent declarada) o 'gamow' (cota), y 'bounce' es control negativo
- las leyes de sellado de c_eff y m_eff tienen forma paramétrica declarada, no la ec. (5.3)
- la diagonal θ = π/4 no la cruza el flujo del Basal: se impone para la entrega
- V3D y Florencia son cuantos declarados tras el residuo de descarga
- el estado entregado en S = 1,001 no es legible por la cosmología (diccionario ausente); m_H depende de δ₀ (input) y β₃ es condicional
- el modo emergente es diagnóstico: el diccionario τ no es derivable
- la inestabilidad de masa (M0² = 0) es un evento publicado del modo diagnóstico, no un colapso con estatuto; la tabla τ_k reformula el diccionario, no lo deriva (E13)
- W_max se publica con las tres cadenas (ley 3.4 en δ_H_ley; paisaje completo en δ_H_ley; paisaje completo en δ_H_full): la decisión A del autor (22-sep) nombra T₀_full, pero el valor de W_max sigue sin asignar en el tratado (frente 4)
- las decisiones A, B y C son del autor, derivadas del texto del tratado y declaradas en DECLARED_FORMS/DECISION_A: el código las ejecuta, no las demuestra (E8); D_ent y las β que hagan D → 0 siguen siendo del frente 2
- la unidad de S tras Florencia no se deriva de T₀ (hueco: fila diccionario-unidad-S-post-florencia)
