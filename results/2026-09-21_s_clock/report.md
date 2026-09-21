# El reloj S como simulador de consistencia — recorrido S₀ → S_{1,001} (21-sep-2026)

Commit `3a1cff846`. Modo con estatuto: umbrales IMPUESTOS (Prop. 8.1). δ₀ primario = 0.01 (valor de prueba: el tratado no asigna valor a δ₀). Formas del Basal: m̄ = 1.0, b̄ = 3.0, ē = 1.0, C0 = 1.0.

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
| 0.01 | 0.01 | 5.00e-04 | 3.00e-04 | — | 0.1057 | acoplos fuera del dominio (C0 ≤ 0 o M0² ≤ 0) |
| 0.01 | 0.1 | 5.00e-04 | 3.03e-04 | — | 0.0105 | acoplos fuera del dominio (C0 ≤ 0 o M0² ≤ 0) |
| 0.01 | 1 | 5.00e-04 | 3.12e-04 | — | 0.0011 | acoplos fuera del dominio (C0 ≤ 0 o M0² ≤ 0) |
| 0.05 | 0 | 1.25e-02 | 1.25e-02 | — | 0.9990 | descenso completado (residuo de descarga alcanzado) |
| 0.05 | 0.001 | 1.25e-02 | 1.15e-02 | — | 0.8927 | max_steps agotado |
| 0.05 | 0.01 | 1.25e-02 | 7.74e-03 | — | 0.4953 | max_steps agotado |
| 0.05 | 0.1 | 1.25e-02 | 7.50e-03 | — | 0.0530 | acoplos fuera del dominio (C0 ≤ 0 o M0² ≤ 0) |
| 0.05 | 1 | 1.25e-02 | 7.54e-03 | — | 0.0053 | acoplos fuera del dominio (C0 ≤ 0 o M0² ≤ 0) |

Lectura: con el cierre canónico D se hunde (Obs. 8.6) pero en ninguna corrida cruza cero antes del fin del recorrido: para τ grande M0² cruza cero ANTES que D (el falso vacío se destabiliza por la masa, no por la espinodal) y para τ pequeño el descenso termina, o se agota el presupuesto de pasos con el flujo ya muy lento sobre un potencial aplanado, con D > 0. En el rango explorado los colapsos NO emergen del Cruce de Victoria — resultado diagnóstico que depende del diccionario τ (frente 2), publicado sin verdicto.

## Lo que NO afirma

- ningún umbral emerge: los colapsos se disparan en 0.009/0.099/0.999 por la Ley de la Década (calibrada, frente 2)
- la nucleación (Γ₀) no se calcula: el reloj arranca en el punto de escape con σ = 0 declarado
- las leyes de sellado de c_eff y m_eff tienen forma paramétrica declarada, no la ec. (5.3)
- la diagonal θ = π/4 no la cruza el flujo del Basal: se impone para la entrega
- V3D y Florencia son cuantos declarados tras el residuo de descarga
- el estado entregado en S = 1,001 no es legible por la cosmología (diccionario ausente); m_H depende de δ₀ (input) y β₃ es condicional
- el modo emergente es diagnóstico: el diccionario τ no es derivable
