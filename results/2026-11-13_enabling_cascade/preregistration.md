# Preinscripción — Cribado de la cascada de habilitación: tres direcciones latentes con pesos por una ley de escala de un número, en el reloj S completo

Congelada 2026-11-13T11:29:38.911237+00:00 en el commit `4b40300e5` (contiene el generador), ANTES de ninguna corrida con letra. sha256 de `preregistration.json`: `9040df79900a505a8f96282d5b2a80dfc3785533820209f3fd95f713333d1286`.

**Pregunta**: Con el Basal del reloj y tres direcciones latentes cuyos pesos salen de la ley de escala con λ = 10, ¿hay tres colapsos completos, en orden, con los dos cocientes de umbrales en [7, 14], y se cumplen la Monotonía y los controles?

## Decisiones del autor (13-nov-2026)

Acta: `docs/decisiones_autor_2026-11-13.md` (sha256 `f947e1aa4fd565abf0c50f246cc2d8fee4f1940dc58505b84aa19353f1e234be`).

- **Decisión 4**: «Latente primero».
- **Decisión 5**: «Los tres q_k no representan tres nuevas esencias fundamentales independientes. Los interpreto como parámetros de orden asociados a modos de estructuración emergente de la Unidad Dual. […] El modelo R6 puede utilizar tres amplitudes q_k como descripción efectiva de estas transiciones, pero su interpretación como modos emergentes deberá justificarse mediante una dinámica común derivada de la Unidad Dual. Su mera introducción no constituye una demostración de emergencia dimensional.»
- **Decisión 7**: «Adopto provisionalmente una progresión geométrica de razón λ para los pesos del Basal, q_1 y q_2, con un peso residual para q_3, de forma que los cuatro permanezcan normalizados. Para λ = 10, la distribución propuesta es 0,9 %, 9 %, 90 % y 0,1 %, respectivamente. […] El resultado deberá contrastarse mediante simulaciones del sistema completo, pruebas de estabilidad y controles negativos.»

## Declarado

- **delta0_grid**: [0.003, 0.01, 0.0554]
- **clock**: reloj por defecto (core/s_clock.py): rayo θ_nuc = 0, punto de escape, acoplos fijos, con la inclinación (ē = 1), G = 1, d_sigma_hat = 5e-4
- **model**: V̂ = V̂0 + Σ_k W_k·[½·m_k²·q̂_k² + ¼·q̂_k⁴], m_k² = −tanh((Y_{k−1} − θ_g)/Δ_g), Y_0 = ρ̂²/ρ̂₊², Y_k = q̂_k² (core/enabling_cascade.py); flujo de gradiente puro en las cinco variables con la rigidez G del reloj
- **base**: {'lam_scale': 10.0, 'gate_theta': 0.9, 'gate_Delta': 0.05, 'seeds': [1e-06, 1e-06, 1e-06], 'dS_step': 0.0002, 'noise_floor': 0.0, 'tension_reference': 'measured', 'gated': True}
- **base_clock**: {'d_sigma_hat': 0.0005}
- **scale_law**: fracciones (λ−1)/λ³, (λ−1)/λ², (λ−1)/λ, 1/λ³ de la tensión nominal; con λ = 10: [0.009, 0.09, 0.9, 0.001]; W_k = 4·(profundidad nominal), respecto de la T₀ MEDIDA
- **lambda**: λ = 10 del canon (mcmc_ontology.constants.LAMBDA_DECADE): CALIBRADO, no derivado
- **S**: S ≡ f = (V_inicial − V)/T_total, con T_total del potencial ampliado (mínimo en el dominio físico)
- **events**: gatillo_k: el autovalor del hessiano reducido (dirección radial del Plano Dual y direcciones latentes) asociado a q_k pasa de positivo a no positivo; habilitacion_k: Y_k cruza ½; colapso_kD: los dos, con el S del gatillo
- **integrator**: RK4 con paso h = min(d_sigma_hat, dS_step·T_total/Σ̇), proyectado sobre φ̂ ≥ 0 como el reloj
- **letter_arms**: ['base']
- **gate_arms**: ['convergencia']
- **control_arms**: ['sin_semilla', 'semilla_1e-3', 'semilla_1e-9', 'suelo_1e-18', 'sin_compuerta', 'jerarquia_rota', 'pesos_iguales']
- **sensitivity_arms**: ['lambda_5', 'lambda_7', 'lambda_14', 'lambda_20', 'referencia_ley', 'theta_0.8', 'theta_0.95', 'Delta_0.02', 'Delta_0.1']
- **reference_without_letter**: juguete radial del oráculo: δ₀ = 0.01 con ē = 0 (cifras ya públicas en la revisión del 11-nov)

### Brazos

| Brazo | Cambia en el modelo | Cambia en el reloj |
|---|---|---|
| base | — | — |
| convergencia | {"dS_step": 0.0001} | {"d_sigma_hat": 0.00025} |
| semilla_1e-3 | {"seeds": [0.001, 0.001, 0.001]} | — |
| semilla_1e-9 | {"seeds": [1e-09, 1e-09, 1e-09]} | — |
| sin_semilla | {"seeds": [0.0, 0.0, 0.0]} | — |
| suelo_1e-18 | {"noise_floor": 1e-18} | — |
| sin_compuerta | {"gated": false} | — |
| jerarquia_rota | {"seeds": [0.0, 1e-06, 1e-06]} | — |
| pesos_iguales | {"shares": [1.0, 1.0, 1.0, 1.0]} | — |
| lambda_5 | {"lam_scale": 5.0, "noise_floor": 1e-18} | — |
| lambda_7 | {"lam_scale": 7.0, "noise_floor": 1e-18} | — |
| lambda_14 | {"lam_scale": 14.0, "noise_floor": 1e-18} | — |
| lambda_20 | {"lam_scale": 20.0, "noise_floor": 1e-18} | — |
| referencia_ley | {"tension_reference": "law"} | — |
| theta_0.8 | {"gate_theta": 0.8} | — |
| theta_0.95 | {"gate_theta": 0.95} | — |
| Delta_0.02 | {"gate_Delta": 0.02} | — |
| Delta_0.1 | {"gate_Delta": 0.1} | — |

## Reglas (congeladas)

- **evaluated_on**: los S de los gatillos de los tres colapsos completos (gatillo y habilitación) del brazo base, por δ₀
- **A_interval**: [7.0, 14.0]
- **n_collapses_required**: 3
- **letters**:
    - A: los dos cocientes S_{k+1}/S_k en [7, 14]
    - B: uno de los dos en [7, 14]
    - C: ninguno en [7, 14], o menos de tres colapsos completos
    - INDETERMINADO: alguna puerta numérica o algún control obligatorio falla
- **order**: puertas → controles → C → B → A
- **per_cell**: letra por δ₀; la global es la peor con el orden A < B < INDETERMINADO < C
- **gates**:
    - finished: el descenso del brazo base termina (velocidad proyectada ≈ 0, ninguna dirección sembrada inestable), sin divergencia y sin subdesbordamiento de una dirección sembrada
    - monotonia_max_dV_over_T_total: 1e-12
    - identity_max_abs_diff: 1e-06
    - trigger_is_min_eigenvalue: en cada colapso el autovalor que cruza cero es el mínimo del hessiano reducido
    - hierarchy: los colapsos salen en el orden V1D, V2D, V3D
    - final_state: hessiano reducido definido positivo al final y |V_final − V_mín| ≤ final_V_tol·T_total
    - final_V_tol: 1e-09
    - convergence_rel: 0.0001
    - convergence: con el paso a la mitad (d_sigma_hat y dS_step) hay el mismo número de colapsos y cada S de gatillo cambia ≤ convergence_rel en valor relativo
- **controls**:
    - sin_semilla: con las tres semillas a cero no hay ningún colapso completo y f final coincide con la parte del Basal, T₀/T_total, a no_seed_rel
    - no_seed_rel: 1e-06
    - semilla: con semilla 1e-3 y 1e-9 hay los mismos colapsos y cada S de gatillo cambia ≤ seed_rel respecto de 1e-6
    - seed_rel: 0.001
    - suelo: con un suelo de ruido de 1e-18 (fuente continua en lugar de semilla inicial) cada S de gatillo cambia ≤ seed_rel
    - sin_compuerta: sin compuerta (m_k² ≡ +1) no hay ningún gatillo ni colapso
    - jerarquia_rota: con q_1 sin sembrar y q_2, q_3 sembradas no hay ningún colapso completo
    - pesos_iguales: con el reparto 1:1:1:1 la regla de letras da C (la A no sale por construcción)
- **sensitivity_without_letter**: barrido de λ (5, 7, 14, 20; con el suelo de 1e-18 para que el flotante no apague q_2), referencia de los pesos c̄·δ₀³ en lugar de la T₀ medida, y compuerta (θ_g = 0.8 y 0.95; Δ_g = 0.02 y 0.1): se publican con la lectura de la regla, sin letra
- **published_without_letter**:
    - desviación de cada gatillo respecto de 0.009 / 0.099 / 0.999
    - recuento de números elegidos: con la ley de escala, λ, θ_g y Δ_g; con pesos independientes, tres pesos, θ_g y Δ_g
    - el juguete radial del oráculo (sin inclinación), como referencia ya pública
- **no_letter_on_lambda**: True

## Comprobaciones antes de congelar

el instrumento se comprobó con el juguete radial sin inclinación (cifras ya públicas: gatillos, tensión total, retraso por semilla, control sin semilla, sin compuerta, jerarquía rota y pesos iguales) y con el reloj completo SIN semilla en los tres δ₀ (sin cascada: reproduce el descenso del reloj por defecto a 1e-13). Ninguna celda con inclinación y semilla se corrió antes de congelar.

en el juguete, el tercer gatillo del oráculo (σ̂ = 1.675) depende de un suelo numérico de su integrador (≈ 1e-18): con la dinámica exacta de una semilla inicial, q_2 decae e^−517 con la compuerta cerrada y el gatillo cae en σ̂ = 3.18; en S el umbral es el mismo (0.99006). Por eso el brazo base usa la dinámica exacta y el suelo es un control.

## Expectativa (E13, escrita antes de congelar)

A en los tres δ₀, con cocientes próximos a 11.3 y 10.1. Es lo que da el juguete radial, ya público, y aquí los pesos se fijan respecto de la Tensión Primordial MEDIDA del Basal, de modo que la inclinación del reloj casi no mueve el reparto: la A es casi segura y no prueba nada sobre λ, que entra a mano. Lo que puede fallar, y es lo que el cribado añade, son las puertas y los controles en el reloj completo (Plano Dual, inclinación, frontera φ_E = 0): que la Monotonía se cumpla paso a paso, que cada gatillo sea el autovalor mínimo, que sin semilla o sin compuerta no haya cascada y que los umbrales en S no dependan de la semilla. Una B, una C o un INDETERMINADO serían el resultado con contenido. En el barrido de λ se espera que los cocientes sigan a 1 + λ y a λ + 1/(1 + λ): la A solo para λ entre 7 y 13, que es decir que la Década entra por λ y no sale de la dinámica.

## Lo que no puede decidir

- λ = 10: entra como número calibrado de la ley de escala; la coincidencia con la Ley de la Década es una hipótesis pendiente de derivación (decisión 7)
- que un parámetro de orden q_k sea una dimensión espacial: son tres amplitudes añadidas; su interpretación como modos emergentes exige una dinámica común derivada de la Unidad Dual (decisión 5)
- la compuerta: es un acople postulado, con dos números elegidos
- si el colapso es nucleado o espinodal (decisión 6, pendiente; R7)
- nada observable: recorrido interno del reloj (E8)
