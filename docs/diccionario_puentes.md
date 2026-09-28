# Diccionario primordial → cosmológico: la tabla de puentes

<!-- GENERADO por scripts/make_dictionary_bridges.py desde docs/claims_registry.yaml — NO editar a mano: tests/test_dictionary_bridges.py falla si este fichero difiere de la regeneración. -->

Cada puente es una magnitud que el tramo pre-geométrico debería entregar a la cosmología (o una constante que la cosmología usa y debería salir de él). Su estatuto se LEE de las filas de `docs/claims_registry.yaml` que lo sostienen, con la regla de `scripts/make_dictionary_bridges.py` (ausente ≻ convencional ≻ calibrado ≻ condicional ≻ derivado). Es una vista, no un juicio: cambia cuando cambian las filas. Ninguna celda «derivado» es demostración física (E8); las cuatro ecuaciones del diccionario (1: forma de ε_c; 2: unidad de S post-Florencia; 3: C(S) desde κ; 4: qué mide S) quedan donde el registro las tiene.

| Puente | De → a | Estatuto | Filas | Nota |
|---|---|---|---|---|
| δ₀ | imperfección primordial → todo el tramo pre-geométrico y δ_H del empalme | **condicional** | `potencial-basal`, `empalme-wkb`, `circulo-delta0` | δ_H = 0.0554 medido por el empalme C¹ (paisaje completo); el atractor δ₀* de Victoria es condicional a W_max |
| f (fracción descargada) y T₀ | Potencial Basal → reloj S (S ≡ f) | **derivado** | `potencial-basal`, `s-clock-simulador-consistencia` | S = f − f₀ + W/T₀ comprobado a 2e-13 sobre la trayectoria; T₀ = c̄δ₀³[1 + κ₁√δ₀ + …] |
| Φ_ten (Florencia) | estado entregado en S = 1,001 → N = e^{Φ_ten} de la cosmología | **ausente** | `s-clock-simulador-consistencia`, `diccionario-primordial-cosmologico` | el reloj entrega Φ_ten = 0 DECLARADO (DECLARED_FORMS); la cosmología no lo lee (readable_by_cosmology = False) |
| ε_K = λ_K − 1 (Residuos) | Gea/Atlas → G_cosmo/G_N (BBN) | **condicional** | `gea-newton-atlas`, `residuos`, `bbn-g-empress` | ε_K = 0.012 viene de la Conj. 9.6 (−1.8 %), no de δ₀; la conexión ε_K = f(δ₀, λ_K, ξ, …) no existe (propuesta v36 §VIII) |
| ξ | acción de Gea → c_T² = ξ | **calibrado** | `mu-eta-atlas`, `atlas-gauge-invariant` | ξ = 1 fijado por GW170817: constante externa, no derivada del tramo pre-geométrico |
| λ_K, α_a | acción de Gea → µ_Atlas, c_s², PPN | **calibrado** | `mu-eta-atlas`, `atlas-ppn`, `atlas-gauge-invariant` | ventana λ_K > 1, α_a ≤ 8e-7 (PPN): cotas, no valores derivados |
| A (amplitud de Cronos) y ρ* | Ley de Cronos débil → dinámica galáctica | **calibrado** | `cronos-amplitud-unica`, `jeans-dsph`, `oort-kz-cota-cronos`, `diccionario-epsilon-c-saturante` | A_Sculptor calibrada por el problema inverso 5A/5B; excluida en el plano (letra C, bytes no descargados); ρ* acotado al 58 % del plano |
| S_post (unidad de S tras Florencia) | reloj S → cronología post-geométrica | **ausente** | `diccionario-unidad-S-post-florencia` | ecuación 2 declarada: dS_post = Σ̇_post·dσ/T_sellada, con Σ̇_post y T_sellada nombradas y no derivadas |
| C(S) = d ln a/dS | corriente de conversión κ → ley de expansión (A.7) | **ausente** | `diccionario-unidad-S-post-florencia`, `vacio-2d-conversion-y-rotacion`, `mapa-s-z-convencion` | ecuación 3 declarada; κ̂ es calibración del brazo (ii) del vacío 2D (INDETERMINADO); S_hoy = 95 sigue siendo convención |
| T(S) = dt_rel/dS | Ley de Cronos de fondo → tiempo cosmológico (A.7) | **convencional** | `mapa-s-z-convencion` | t_rel = C·(S − S_birth)^α con α = 1: convención operativa LEGACY_V32 |
| S_hoy | mapa S(z) → escalones f_id(S) de los canales | **convencional** | `mapa-s-z-convencion` | S_today = 95 (LEGACY_V32): convención declarada, sin derivación |
| Ω_id,0, ε_Λ, z_trans | canales ECV/MCV → fondo (A.4–A.6) | **calibrado** | `canales-oscuros`, `fondo-cosmologico`, `ajustes-produccion` | parámetros del ajuste (F.3); ε_Λ no identificable con DESI DR2 ni Dovekie real |
| α_n, S_n^post (escalones) | colapsos post-Florencia → f_id(S) (A.5) | **calibrado** | `canales-oscuros` | el tratado no fija valores: parámetros del ajuste; α = () apaga los escalones |
| κ_lat, η_lat | canal latente → w_lat(z) (A.6) | **calibrado** | `canales-oscuros` | calibrados en el ajuste de producción |

## Recuento

- **derivado**: 1
- **condicional**: 2
- **calibrado**: 6
- **convencional**: 2
- **ausente**: 3

## Conexión declarada de la unidad de S (ecuaciones 2 y 3)

`cosmology.dark_channels.S_of_z_declared` devuelve el mapa S(z) vigente junto con la procedencia de su unidad: la convención S_hoy = 95 (LEGACY_V32) y el estatuto de `core.s_post_unit` (E2 y E3 declaradas, Σ̇_post y T_sellada no derivadas, κ pendiente). La cosmología CONSUME así la unidad de S de Florencia en modo declarado: el número no cambia; cambia lo que el artefacto dice de él.
