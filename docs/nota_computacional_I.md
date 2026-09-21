# Nota computacional I del programa MCMC

**La ecuación de cierre del círculo de δ₀, la naturalidad de δ_H y los
contrastes de producción**

- **Autor**: Adrián Martínez Estellés — ORCID
  [0009-0009-4314-9642](https://orcid.org/0009-0009-4314-9642)
- **Fecha**: 2 de agosto de 2026
- **Código**: [github.com/Modelo-cosmologico-MCMC/MCMC](https://github.com/Modelo-cosmologico-MCMC/MCMC),
  Apache-2.0, release v0.2.0 (DOI del código: pendiente de la primera
  release con integración Zenodo)
- **Obra de referencia**: *Tratado de Fundamentos* (v35, junio 2026),
  DOI [10.5281/zenodo.20765373](https://doi.org/10.5281/zenodo.20765373);
  serie con DOI de concepto
  [10.5281/zenodo.14167831](https://doi.org/10.5281/zenodo.14167831)

**Abstract (English).** We report the original results of the MCMC
verification program — the executable implementation of the *Tratado de
Fundamentos* (v35). Iterating the Victoria return map with the Ceiling
inside the fertile region shows that its attractor is the Ceiling itself,
δ∞ = (W_max/c̄)^{1/3}, independent of the restart gain and of the initial
imperfection. This yields a consistency equation not present in the
treatise: the δ₀ circle closes if and only if W_max = c̄·δ_H³ ≈ 1.65×10⁻⁴,
linking Thm. 10.6 to the Higgs splice (eq. H.8) through a single number the
treatise declares but does not quantify. A Monte Carlo over the fertile
O(1) landscape shows δ_H = O(0.05) is generic (median 0.054, 90% within
[0.037, 0.116]) and bounded below by δ_H ≳ 0.033 — no O(1) shape closes the
splice at 0.012, structurally retiring the v32 identification δ₀ ≡ ε_Λ.
We also report two production Bayesian fits on public data (ΔBIC = +14.5
favoring ΛCDM in both, published as-is), one observational consistency
check passed (G_cosmo/G_N − 1 ≈ −1.8% vs. the BBN bound, 0.32σ — no
detection, no sign significance; v35.1, E13), and the half-step of
front 5 (gate friction closes exactly at virialization in the minimal PM
mesh; the cored-core verdict requires production resolution). Unfavorable
and favorable outcomes are reported with identical weight.

**Resumen.** Esta nota documenta los resultados ORIGINALES del programa de
verificación computacional del MCMC — los que nacieron del código y no
están en el tratado — junto con su estatuto exacto: qué demuestran, qué
condicionan y qué no afirman. Todos los números provienen de cadenas y
resultados versionados en el repositorio, reproducibles con los comandos
de la §6.

---

## 1. Propósito y contrato de honestidad

El repositorio MCMC implementa el *Tratado de Fundamentos* (v35) como
cadena deductiva ejecutable: cada módulo de `core/` implementa las
definiciones de su capítulo y sus tests verifican identidades, límites
y consecuencias numéricas de los teoremas y ansätze implementados —
comprobación interna de la implementación, no demostración física
(v35.1, E8); donde el tratado declara un condicional, el módulo expone
el parámetro con su condición — nunca lo resuelve en silencio. El contrato de trabajo: cada
commit deja el repositorio afirmando exactamente lo que hace; los
desenlaces desfavorables se publican en portada con el mismo tono que los
favorables. Esta nota hereda ese contrato.

## 2. Métodos

Implementación de referencia en Python (NumPy/emcee), 189 tests, CI con
guardias de honestidad (frases prohibidas verificadas por grep en cada
push) y contrato de lint declarado. Datos observacionales públicos con
SHA-256 fijados (Pantheon+; compilaciones de cronómetros cósmicos, BAO
DR12, fσ8 y geometría comprimida de Planck con referencia por punto).
Los resultados de producción viven versionados en `results/` con semilla,
convergencia y salvedades declaradas en cada informe.

## 3. Resultados originales del programa

### 3.1 La ecuación de cierre del círculo de δ₀

El empalme C¹ (ec. H.8) mide, sin la masa del Higgs como entrada, el δ₀
que cierra en m_H ≃ 125.3 GeV: **δ_H = 0.130/√(b̄²−4C₀m̄²) ≈ 0.0581** con
las formas fiduciales (la relación es no circular: un test verifica que
perturbar el valor PDG no altera el cálculo). Iterando el mapa de retorno
de Victoria (`core/victoria.py`) con Techo en la región fértil:

> **El atractor es el Techo**: δ∞ = (W_max/c̄)^{1/3}, independiente de γR
> y de la imperfección inicial (desde 0.012 se alcanza en 4 vueltas).

De ahí la ecuación de consistencia. Matiz de autoría (auditoría
v35.1): los ingredientes están en el tratado — el Lema 10.3 ya deriva
δ_sat = (W_max/c̄)^{1/3} y H.8 da δ_H; la aportación del programa es la
CONEXIÓN (igualar δ_sat con δ_H como condición de cierre) y la
observación de que γR desaparece del punto fijo, consecuencia directa
de la forma saturada del mapa:

> **El círculo se cierra ⟺ W_max = c̄·δ_H³ ≈ 1.652×10⁻⁴** (formas
> fiduciales, c̄ = 0.8408).

La ecuación liga el Teo. 10.6 (el atractor de Victoria) con la ec. H.8
(el empalme del Higgs) a través del único número que el tratado declara
pero no cuantifica. **Estatuto**: condición falsable, no resultado — el
control negativo lo demuestra (con W_max/2 el atractor es 0.0461 ≠ δ_H).
Tres desenlaces cuando el frente 4 derive W_max de la microdinámica del
reinicio: ≈1.65×10⁻⁴ cierra la cadena; otro valor deja empalme y Victoria
en tensión cuantificada (publicable igualmente); mientras tanto, es la
predicción condicional más nítida del programa. Consecuencias
verificadas si δ₀ = δ_H: T₀ escala ×113.7 (Prop. 3.4) y m_θ² ×51.7
(ec. 10.1). La fertilidad (ν > 0) no depende de δ₀ (ec. H.7); la
contención W_max ≥ c̄·δ_H³ queda cartografiada como curva por paisaje.

### 3.2 La naturalidad de δ_H y la cota inferior

Monte Carlo sobre el paisaje O(1) fértil (40 000 muestras; fracción
fértil 0.720 con γ_max = 3 declarado):

- **δ_H = O(0.05) es genérico**: mediana 0.054, 90 % central en
  [0.037, 0.116]. El valor que el Higgs exige cae en el centro de la
  distribución que las formas naturales del Basal producen por defecto —
  una propiedad de naturalidad que el tratado no reclamó y ahora está
  medida.
- **δ_H ≳ 0.033 en toda la región O(1)**: el discriminante acota el
  mínimo alcanzable; ninguna forma O(1) cierra el empalme en 0.012. La
  identificación δ₀ ≡ ε_Λ heredada de la v32 no es solo dudosa: es
  estructuralmente imposible dentro del paisaje natural del propio
  modelo. La regla canónica (no identificar δ₀ con ε_Λ) queda demostrada
  desde dentro.

**Robustez frente a priors (análisis de sensibilidad, 4-ago-2026;
tarea nacida de la auditoría v35.1).** El barrido priors (uniforme,
log-uniforme, normal) × dominios O(1) × extensión de b̄ × filtro fértil
(`results/2026-08-04_landscape_priors/`) separa lo robusto de lo
fiducial: la **inaccesibilidad del 0.012 es analítica** — la cota de
dominio δ_H ≥ λ_H/√(b_max²−4·C0_lo·m_lo²) supera 0.012 en todas las
configuraciones, para cualquier prior — y el orden de magnitud
δ_H = pocas×10⁻² es robusto (medianas 0.028–0.113); la **mediana
precisa 0.054 es la lectura fiducial**, no un invariante del paisaje
(en la configuración sin extensión de b̄, 0.0581 cae bajo el p5). La
afirmación de naturalidad queda así condicionada: genérico el orden,
fiducial el centrado.

### 3.3 Los ajustes de producción (v1 y v2) — el desenlace desfavorable

Dos ajustes bayesianos sobre datos públicos, con ΛCDM ajustado por la
misma maquinaria (ε = 0 exacto, Prop. A.1) y criterios de información
explícitos. Las corridas originales (jul/ago 2026) precedían a la
corrección de normalización del fondo (H(0) = H0 exacto y clausura
plana por llamada; en las corridas legacy el sesgo de F(0) era solo
del brazo MCMC) y quedan etiquetadas `legacy_pre_normalization` en
`results/`; la tabla cita la **repetición del 10-ago-2026 con el
fondo corregido** (mismas semillas, datos y configuración), que
confirma el veredicto:

| Ajuste (repetición 10-ago-2026) | Datos (n) | ΔAIC | ΔBIC (MCMC − ΛCDM) |
|---|---|---|---|
| v1 (legacy jul 2026: +4.03/+14.53) | CC + BAO + SNe (1408) | +4.00 | **+14.50** |
| v2 «la reconciliación» (legacy ago 2026: +4.01/+14.53) | + CMB comprimido + fσ8 (1422) | +4.08 | **+14.60** |

Positivo favorece a ΛCDM. En v2 (24×8000 pasos, convergencia 50·τ):
ε = 0.018 −0.041/+0.043 — compatible con cero y con el 0.012 del corpus
(sin necesidad, no excluida); H₀ = 68.19 ± 0.40 y σ₈ = 0.799 ± 0.029,
planckianos (posteriores ya sin el sesgo de normalización).
**La reconciliación queda respondida**: la ventaja del
corpus (ΔBIC = −6.1) no reaparece al añadir los dos bloques de los que
procedía; solo podría residir en el C_ℓ completo, la lente débil o su
metodología. Los valores del corpus son referencia histórica, no
citables como vigentes. Salvedades declaradas en los informes: z*/r_s
por Hu–Sugiyama (~0.3 % en l_A, idéntico en ambos modelos), CMB diagonal,
r_d fiducial en BAO, M_B marginalizada.

### 3.4 El contraste BBN (frente 6) — consistencia superada

La Conjetura de los Residuos (ec. 9.5) predice G_cosmo/G_N − 1 ≃
−(3/2)·ε_K ≈ −1.8 %. Frente a la cota de nucleosíntesis vigente
G_BBN/G₀ = 0.99 +0.06/−0.05 a 2σ (Alvey, Sabti, Escudero & Fairbairn
2020, Eur. Phys. J. C 80, 148; arXiv:1910.10730): la predicción cae
dentro, a 0.32σ del valor central. A este nivel de significancia **no
hay detección ni preferencia de signo** (v35.1, E13): el resultado es
una **consistencia superada** — el contraste podía excluir el valor
predicho y no lo excluye —, declarada sin significancia; la
coincidencia de lado con el central (ambos < 1) es descriptiva. Se
volvería decisivo si las cotas llegan al 1–2 %. Estatuto: conjetura
declarada (9.6), frente abierto nº 6, refutable (vivo).

### 3.5 El medio paso del frente 5 — el instrumento dijo la verdad

En la malla PM mínima (par A/B de semilla idéntica, α₀⁻¹ en la cota de
la ec. 11.5): el colapso frío aislado emerge cuspy en ambas corridas
(NFW preferida, 0.32 vs 0.04–0.06 dex), la diferencia A/B es <1 % y los
radios de escala caen bajo la celda — **la malla no puede discriminar el
veredicto del núcleo**, que queda en producción, como el frente declara.
La compuerta sí habló: en el halo aislado virializado la mediana de Γ
del tramo final es **cero exacto** (Γ = 0 en el 57 % de los pasos) y la
forma antigua sin compuerta persiste (el control negativo de H.2.5).
Hallazgo de validez documentado: ρ_c debe ser el umbral de colapso
(~200× la media); con ρ_c ~ media, ε_c alcanza O(1) — fuera del régimen
débil de la Def. 11.1 — y la fricción condensa el halo en un punto
(artefacto verificado antes de descartarse). Certificado de régimen
publicado: ε_c máx = 3×10⁻⁵ ≪ 1.

### 3.6 El frente 2 instrumentado (ec. 14.2)

La matriz de estabilidad del flujo de acoplos en el cuello espinodal —
el objeto que §14.2 declara como la tarea del frente 2 — queda
instrumentada (`core/victoria_exponent.py`): las dos rutas al exponente
de Victoria (espectral, s0 = |Im μ(M)|, y dinámica, el periodo del
walking medido en el flujo integrado) coinciden con error ~10⁻¹⁴; con
ansatz O(1) declarado, la cascada DSI es **genérica** (~67 % de las
matrices complejifican sus exponentes, con independencia de la escala
del ansatz) pero **λ = 10 es una selección medible, no una
consecuencia**: la banda ±10 % de s0 = π/ln10 captura solo unas
unidades por ciento, y esa fracción depende además de la escala del
ansatz. Desenlace del ansatz en `results/2026-08-05_victoria_exponent/`.

**Las β reales, derivadas — A en el punto preinscrito y la cascada en
el régimen físico (19-ago-2026).** La ronda del frente derivó las β de
(M0², B, C0) desde la reducción canónica de la jerarquía de
Fokker-Planck (Def. 4.4): Polchinski d = 0 con cierres declarados
(a = b = ½; truncamiento cúbico con el cuártico como sistemático;
sector η O(δ0⁶) despreciado; diccionario τ = dt/dlnS declarado no
derivable del corpus recogido). Con los cuatro desenlaces preinscritos
ANTES del primer autovalor (`core/fokker_planck_beta.py`,
`results/2026-08-19_front2_fp_beta/`), en el punto preinscrito salió el
desfavorable: **desenlace A** — espectro de M = ∂β/∂λ enteramente real
(μ = −18.66, −1.01, +7.67: silla sin rotación) y dD/dt = +28 > 0 (la
Obs. 8.6 exige hundirlo; el signo cambia en g = 2.75 — 6/21 puntos del
barrido con dD/dt < 0 — y el par complejo solo existe para
g > g* = 8.19: la conjunción rotación + hundimiento vive solo en el
borde, donde λ = 10 exigiría τ* = 0.233, publicado como dato). **La
revisión adversarial de la ronda (14 confirmados, 0 refutados) destapó
el hallazgo HIGH**: las β cumplen la covarianza exacta
β(D_s·λ; a, b) = σ·D_s·β(λ; a, b·k), así que el eje g del barrido ES
el eje δ0 del escalado (3.2) (g_ef = δ0⁻³) y el punto preinscrito
fijaba implícitamente δ0 ≈ 1, fuera del régimen perturbativo del
corpus. La adenda δ0 (misma derivación congelada, preinscripción
intacta): para δ0 < δ0* = 0.496 la reducción canónica produce par
complejo Y dD/dt < 0 a la vez; en δ0 = 0.1, s0(τ=1) = 4.946 y
τ* = 0.276 (el corpus no fija τ; elegirlo a posteriori sería tuning,
por eso τ* se publica como dato). Contrafactual con las reglas
preinscritas en el punto físico: **B** — hay cascada, fuera de la
banda de λ = 10. La derivación queda validada de punta a punta
(álgebra ⟺ jacobiano ⟺ suavizado exacto ⟺ FP exacta vía Hopf-Cole,
con el control de deriva publicado y la sustitución Langevin→exacto
declarada). **Estatuto**: interno (E8) y condicional a los cierres
declarados — el mecanismo de la cascada tiene por primera vez una
realización derivada en el régimen del corpus, y λ = 10 sigue siendo
una calibración; el defecto de diseño de la preinscripción (el punto
escondía δ0 ≈ 1) queda declarado y corregido por adenda.

<!-- 3.7 = la Circulación de Victoria (rama theory/front2-gradient-victoria) -->

### 3.8 Sculptor contra el potencial débil (frente 5, medio paso 2)

El potencial débil de la Ley de Cronos (Def. 11.1 + Prop. 11.2),
Φ_eff = Φ_N − c²·ε_c(ρ), queda instrumentado con Jeans esférico y
proyección (`dynamics/`, validado contra la identidad analítica del
Plummer isótropo, error < 2×10⁻³) y confrontado con los datos globales
de Sculptor (Walker et al. 2009; procedencia y pendientes declarados
en `dynamics/dsph_data.py`). El resultado, publicado tal cual BAJO
LAS DOS LECTURAS DECLARADAS de la ec. (11.5) — la literal L (acota
solo α₀⁻¹, ρ_c libre) y la de subdominancia P («para que no domine
sobre la gravedad en halos», Cor. 11.3c), no equivalentes con ρ_c
libre —: solo estrellas dan σ_N = 2.1–3.6 km/s (Υ⋆ ∈ [1,3]); **bajo
P el término débil de Cronos queda falsado como explicación de
Sculptor** (techo saturado √2·σ_N ≤ 5.1 km/s frente a σ_obs ≈ 9–10);
bajo L no hay violación numérica de la desigualdad, pero el término
solo alcanza σ_obs siendo el potencial DOMINANTE — c²ε_c =
×8–×26·|Φ_N| —, el régimen que la justificación declarada de la cota
excluye. El invariante entre lecturas: **el término solo explica
Sculptor dejando de ser una corrección subdominante de campo débil**.
Sculptor no mide ρ_c — mide la amplitud A = α₀⁻¹/ρ_c^(3/2); con α₀⁻¹
en su cota, la exigencia es la desigualdad unilateral ρ_c ≲ 1–2
M⊙/pc³, COMPATIBLE con la receta de validez de la malla PM del §3.5
(ρ_c ≈ umbral de colapso — requisito de régimen en unidades de
código, no una medida): el contraste cruzado 5E sigue plenamente
abierto. La carga explicativa de los dSph queda cuantificada para el
sector ρ_id (M_1/2 ≈ 2×10⁷ M⊙ dentro del radio de media luz 3D
r_1/2 ≈ 347 pc, estimador de Wolf declarado), cuyo perfil a esa
escala el tratado no deriva — hueco declarado. **Estatuto**:
veredicto exacto dentro del montaje declarado (trazador Plummer, β
constante, datos globales); el perfil binado σ_los(R), poblaciones
múltiples y la ingesta de la tabla original son los pendientes del
siguiente paso; 5C (SPARC con la MISMA A) decidirá la falsación
cruzada.

### 3.9 El 5C estructural y el objetivo ρ_id como curva

La mitad del 5C que no necesita el catálogo (cuya ingesta sigue
bloqueada por el proxy — pendiente declarado): en discos exponenciales
el término de Cronos tiene FORMA fija, v_cronos²(R) =
(3c²/2)·(R/R_d)·A·ρ(R)^(3/2) ∝ x·e^(−3x/2) con pico en x = 2/3 —
identidad de la Def. 11.1, independiente de A, Σ0, R_d y ζ. De ahí la
ley que decide: v²(x=4)/v²(pico) = 0.040, así que **aportar V km/s
donde la discrepancia de masa vive (x ≳ 3) cuesta 5·V km/s de bulto
en x = 2/3** — ninguna amplitud A produce curvas planas con este
término; que los discos internos densos son bariónicos (sin hueco
para tal bulto) es el resultado estándar de la literatura, usado como
HIPÓTESIS DECLARADA sin ingesta — la confrontación por galaxia es el
pendiente. Con la A que Sculptor exige (recalculada, no copiada:
6.2×10⁻⁷ para Υ⋆ = 2), la malla declarada (Σ0 ∈ [50, 800] M⊙/pc²;
los LSB reales bajan más — allí el término se hace despreciable y no
explica nada) da v_cronos/v_bar hasta ×6.3 en el pico interior y
2–82 km/s en x = 4 (bultos donde el disco es denso, nada donde no),
con ε_c ≤ 5×10⁻⁶ (el régimen débil intacto: falla la fenomenología,
no la expansión). **Veredicto estructural 5E (parcial)**: junto con
el medio paso 2, el término débil de Cronos no puede ser el mecanismo
galáctico común a presión y rotación; la carga es de ρ_id — y
Sculptor la deja MEDIDA como curva de degeneración: con el perfil
cored del corpus, M_id(<347 pc) = 1.78×10⁷ M⊙ define ρ0(r_c) (en el
r_c = 0.30 kpc del corpus: ρ0 = 0.176 M⊙/pc³), la curva que cualquier
derivación futura de ρ_id — o la relación rcore(M, z) de B.5
calibrada en SPARC — debe atravesar. Instrumento de perfiles binados
listo con autoprueba sintética declarada
(`dynamics/binned_profile.py`).

**El 5E queda preinscrito y congelado (16/17-ago)**: la amplitud
transferida A = 6.201589×10⁻⁷ (M⊙/pc³)^(−3/2) se calculó con la
función canónica ANTES de intentar la ingesta, con hipótesis
primaria, estadísticos, cortes y alcance del veredicto
precomprometidos (`results/2026-08-16_front5e_sparc/`), candados
ejecutables contra tuning y pipeline puro testeado en sintético. Las
fuentes (SPARC, VizieR) siguen denegadas por el proxy: **el 5E
observacional está ABIERTO por ausencia de datos ingeridos** — regla
de fallo preinscrita: sin fixtures, sin mirrors, sin veredicto
observacional.

### 3.10 DESI DR2 BAO: benchmark y contraste (frente 6A)

Con la capa de datos reproducibles (PR #11) y el guard runtime
require_available, la likelihood BAO propia quedó validada contra la
referencia externa fijada (Cobaya 3.6.2: máx |Δχ²| = 1.85×10⁻¹³ sobre
vectores sintéticos; identidad de datos computada por sha256 — 16/16
ficheros idénticos a bao_data @ v2.6, el pin de Cobaya) y el ΛCDM
propio reprodujo el valor oficial publicado ANTES de tocar el MCMC —
con el fondo normalizado (H(0) = H0, clausura plana por llamada),
Ω_m es la fracción de materia del ΛCDM plano de verdad:
0.2971 −0.0085/+0.0086 vs 0.2975 ± 0.0086 (arXiv:2503.14738) =
0.04σ; puerta predeclarada PASS. El contraste (r_d como calibración
común — el MCMC no deriva física pre-recombinación; χ²_min por
multistart acotado al soporte del prior, argmin publicado):
**veredicto uniforme en DESI_ALL y los 7 leave-one-bin-out** —
Δχ² ∈ [−0.33, −0.06], ΔBIC ∈ [+4.47, +4.90] pro-ΛCDM, el argmin del
MCMC en la frontera del soporte (ε = −0.05, z_trans = 1) en las 8
configuraciones (publicado), ningún bin decisivo en ninguno de los
dos modelos, ε_Λ dominado por el prior truncado (cociente de
anchuras posterior/prior ≥ 0.99). Razón estructural computada en el
artefacto: con z_trans = 8.9 y la transición normalizada hoy,
ε = 0.05 altera E(z ≤ 2.33) en ≤ 4.3×10⁻⁷ relativo sobre TODO el
rango, y tras reabsorber (Ω_m, H0·rd) el residuo χ² es 7.6×10⁻¹¹ —
BAO DR2 constriñe la forma de la expansión donde el fondo del corpus
es casi degenerado con ΛCDM; no es el test sensible de ε_Λ en esta
parametrización y lo que el dato castiga es la parsimonia. Mismo
veredicto que la repetición v1/v2 con el fondo corregido. La ronda
pasó su revisión adversarial (36 hallazgos confirmados aplicados —
fondo sin fusionar, mínimos no acotados, semilla que no reproducía,
números estructurales de un solo punto; el veredicto sobrevivió a
todos, los números publicados se regeneraron). **Estatuto**:
BAO-only; el cruce con cadenas oficiales completas pendiente del
host bloqueado; SNe (mocks Dovekie primero) y combinaciones DESI+SNe
son el siguiente paso del frente.

### 3.11 Crosscheck JAX del fondo (6A, prioridad 3)

Implementación INDEPENDIENTE en JAX (float64, Gauss-Legendre
espectral; cero imports de las implementaciones NumPy — regla
vigilada por test) de E, H, D_H, D_M, D_V, D_L, μ, el mapa S↔z y el
vector DESI + χ², contrastada bajo puertas PREDECLARADAS
(max |ΔX/X| < 10⁻⁸, RMS < 10⁻¹⁰, |Δχ²| < 10⁻⁸) en los θ donde viven
los números publicados (los argmin de benchmark.json y
contrast.json). La primera ejecución CAZÓ al integrador DESI de
producción: trapecio + interpolación lineal en 4.9×10⁻⁸ — sobre la
puerta — con efecto ≤ 2.8×10⁻⁶ en los χ² de los argmin publicados (el
veredicto 6A nunca dependió de él); en vez de relajar la puerta, el
integrador se corrigió a O(h⁴) — verificación contra scipy.quad
registrada en el artefacto (integrator_vs_quad: 5.8×10⁻¹⁵ en DM, con
candado) — y los artefactos DESI se regeneraron: números publicados
idénticos a 3 decimales. Resultado final: **todas las puertas PASS**
(álgebra a 4×10⁻¹⁶, vector a 1.2×10⁻¹⁴, |Δχ²| ≤ 2.2×10⁻¹² en los
argmin publicados y ≤ 6.9×10⁻¹¹ incluyendo el rincón de estrés). El
integrador SNe de producción (rondas v1/v2, ya fusionadas) queda
medido en 3.0×10⁻⁶ relativo (≤ 6.6×10⁻⁶ mag en μ — despreciable
frente a σ_μ ~ 0.1): publicado como medición sin puerta, sin tocar
una ronda cerrada. El registro pre-fix lleva su nota de procedencia
(generado con el runner aún sin commitear — hallazgo declarado de la
revisión adversarial de la ronda: 15 confirmados + 1 parcial, 0
refutados, todos aplicados).
**Estatuto**: validación interna de implementación (E8) —
equivalencia numérica entre dos implementaciones, no validación
física (`results/2026-08-19_jax_crosscheck/`).

### 3.12 El sector perturbativo lineal: la primera predicción out-of-sample (12-sep-2026)

Con la preinscripción congelada ANTES de computar (desenlace esperado
enumerado: fσ8^MCMC ≈ fσ8^ΛCDM dentro de la banda del prior de ε_Λ —
el control de consistencia, NO un fracaso; desenlace contrario
enumerado con su procedimiento de discriminación), el crecimiento
lineal integrado exacto sobre el H(z) del modelo, propagado sobre el
posterior de fondo CC+BAO+SNe (v1 corregido, sin crecimiento en el
ajuste), produce el **desenlace A**: max_z |R_p50(z) − 1| = 4.3×10⁻⁸
con R(z) = fσ8^MCMC/fσ8^ΛCDM (la predicción con cero parámetros
nuevos: σ8 se cancela), dentro de la envolvente del prior
(6.4×10⁻⁷), y Δχ² = −8.5×10⁻⁷ out-of-sample sobre la compilación RSD
(σ8 externa declarada, idéntica en ambos brazos; aquí no se ajusta
nada — guardia ejecutable). La magnitud ~10⁻⁷ es la nota estructural
de 6A heredada: con z_trans ≈ 9 y la transición normalizada hoy, ni
el fondo ni el crecimiento lineal sondean ε en esta parametrización.
El objetivo discriminante queda declarado y NO ejecutado: µ(k,z),
η(k,z) deben derivarse de Atlas/ε_c (frentes 3 y 10), con prohibición
preinscrita de elegirlos desde los datos de lensing.
**Estatuto**: control de consistencia interna del sector lineal
sobre el fondo (E8) — no validación del modelo frente a ΛCDM
(`results/2026-09-12_perturbations_fsigma8/`).

### 3.13 La validación por mocks del pipeline SN Dovekie (12-sep-2026, PR #15)

Con la preinscripción congelada ANTES de ejecutar
puerta alguna — tolerancias, semillas y enteros binomiales exactos
para N = 25 — el pipeline SN propio superó sus cuatro puertas sobre
25 mocks a nivel HD (el punto de entrada del pipeline; los mocks
fotométricos DES validan las etapas DES aguas arriba y quedan
registrados sin ingerir): equivalencia con la fórmula oficial
ejecutada desde los bytes del release (max |Δχ²| = 1.5×10⁻⁹, conteo
1820 verificado por bytes), integradores cruzados (1.9×10⁻⁶ mag),
recovery de Ω_m (media de pulls +0.051, cota 3·SEM = 0.6), cobertura
binomial exacta (k68 = 17/25 ∈ [12, 21], k95 = 23/25 ∈ [21, 25] —
no se detecta una descalibración incompatible con el tamaño de la
muestra de mocks) y ausencia de falsa preferencia por la extensión
(mediana ΔAIC = +3.94 pro-ΛCDM, n(Δ<0) = 0; 0 ∈ CI95(ε) en 25/25).
La BARRERA es ejecutable: la columna MU del HD real solo es accesible
citando por sha256 la preinscripción del PASS, y el candado vive en
la suite.
**Estatuto**: validación interna del pipeline (E8) — habilita
consumir el dato real; no afirma nada sobre el dato real
(`results/2026-09-12_dovekie_mocks/`).

### 3.14 µ, η y Σ desde la ontología: el canal Cronos derivado y dónde se rompe cada cierre (13-sep-2026)

Con la preinscripción congelada ANTES de computar (E1, el desenlace
aburrido, como control; E2 identidades; E3, la cadena halos → α₀⁻¹ →
µ → observables, declarada y no ejecutada; la tabla a mano de la nota de
teoría como «a verificar»), la Ley de Cronos débil (11.1), la geodésica
(11.2) y Gea (9.3) dan, linealizando la ε_c CANÓNICA
(d ln ε_c/d ln ρ = 3/2), el canal Cronos cerrado:
**µ − 1 = ε̄_c(a)·(ck/aH)²/Ω_m(a), η = 1/µ, Σ = (1+µ)/2** — cola k²,
amplitud ∝ α₀⁻¹, µ − 1 = 2(Σ − 1) y η < 1 sin parámetros adicionales,
límite GR exacto; identidades verificadas a precisión de máquina. Con
α₀⁻¹ = 10⁻⁶ (la cota 11.5 saturada — límite superior, no medida) y el
cierre comoving (ρ_c = 200× la media de cada época): **E1 = A**,
max |R_µ − 1| = 1.55×10⁻⁴ en k ≤ 0.2 h/Mpc, z ≤ 2 — el sector lineal no
inventa señal (µ − 1 = 9.7×10⁻⁵ en z = 0, k = 0.1 h/Mpc). La tabla de
la nota se recomputó: z = 0 y z = 1 correctas (×0.8–1.1); la columna
z = 3 del cierre physical estaba sobreestimada ×3–4 (5×10⁻² → 1.25×10⁻²
en k = 0.1) — las cifras citables son las calculadas. **Hallazgo
estructural no previsto**: el cierre physical (ρ_c = 200·ρ̄_m(0)) con
α₀⁻¹ en la cota no es perturbativo: µ − 1 ∝ (1+z)^{7/2} alcanza 1 en
z = 8.4 (k = 0.2) a 51 (k = 0.01), ε̄_c ≥ 1 para z ≳ 125, el propio
fondo supera ρ_c para z > 4.85 y la ODE de crecimiento diverge (el
código falla cerrado, nunca publica NaN). Consecuencia enunciable sin
datos: bajo 2b una amplitud viable exige α₀⁻¹ ≪ 10⁻⁶ — la consistencia
del sector lineal acota 2b por debajo de la cota galáctica, y el
criterio «200× la media» favorece conceptualmente el cierre comoving.
El canal Atlas queda como gancho con coeficientes O(1) pendientes del
frente 3 (contribución no computada); lo enunciable sin el cierre: el
mismo ε_K de G_cosmo/G_N − 1 ≈ −(3/2)ε_K fija el offset de crecimiento
a gran escala. Ningún dato entra aquí: la prohibición «µ, η nunca se
eligen desde los datos de lensing» es ejecutable en la suite.
**Estatuto**: derivación interna con cierres declarados (E8) y
predicción sin contraste — no detección, no validación; amplitud =
límite superior (`results/2026-09-13_mu_eta_cronos/`).

### 3.15 El canal Atlas de µ/η derivado desde la acción: la firma sub-horizonte se cancela (13-sep-2026)

Bajo preinscripción congelada (once identidades E1 a reproducir; arnés
E2 con umbrales del 1 %; desenlaces A/B; fronteras declaradas), la
Acción de Gea (9.1) con el Término de Cronos (9.4), en gauge unitario y
con polvo, se expandió a segundo orden y se resolvió en el límite
cuasi-estático: **µ_Atlas(e) = 2ξ/[(2ξ−α_a)+3(3λ_K−1)e²]**, η_Atlas → 1,
y **G_growth = G_local = G_B/(ξ−α_a/2)** — la G estática calculada de
forma independiente en Minkowski con fondo vacío (el residual de Jeans
de un fondo uniforme no autoconsistente queda registrado). La acción
reproduce el Sello de Newton (G_cosmo = 2G_B/(3λ_K−1)), la ventana de
salud (9.4) (no-fantasma ⟺ λ_K > 1; c_s² = ξ(2ξ−α_a)(λ_K−1)/(α_a(3λ_K−1))
> 0 ⟺ α_a < 2ξ) y c_T² = ξ (GW170817 ⟹ ξ = 1). La integración del
sistema lineal completo, sin aproximación QS y con relajación al modo
creciente, da µ_num/µ_QS − 1 = +2.8e-04/-5.3e-04/+5.9e-05 y η_num − 1 = +3.0e-04/+4.6e-04/-3.3e-05
(k/H0 = 66.7/200/600), con el índice de crecimiento p_num = p_QS =
1.1342 a 6e-04: **desenlace A**. Lectura: el offset α_a/(2ξ) se cancela
exactamente contra la renormalización de la G local, así que el canal
Atlas no deja firma sub-horizonte en (µ, η) al orden dominante y la cola
k² del canal Cronos (§3.14) es la ÚNICA firma sub-horizonte del sector
perturbativo; la expectativa previa «µ_Atlas − 1 = O(1)·ε_K» queda
superada — registro de una expectativa corregida por derivación, que es
parte del método. ε_K entra en el fondo: G_cosmo/G_local =
(2ξ−α_a)/(3λ_K−1) ≈ 1 − (3/2)ε_K − α_a/2, y BBN acota la combinación
(§3.4 refinada). Dos precisiones: η_QS es 0/0 en λ_K = 1 exacto (el
límite GR toma e → 0 primero); y la cola QS de η lleva el polo
1/(λ_K−1), pero el sistema completo la da ×~28 menor a e = 0.01 — el
polo es un artefacto de la truncación QS al nivel medido y los
coeficientes reales de las colas son frontera declarada (sector de
velocidades), como el régimen superhorizonte, las cotas PPN sobre
(ε_K, α_a) y el acoplamiento fuerte. Erratum candidata para la v36
(H.2.2): c_s² diverge, no se anula, cuando α_a → 0 a λ_K fijo.
**Estatuto**: derivación desde la acción verificada por cuatro vías
(E8); teoría pura, sin datos (`results/2026-09-13_mu_eta_atlas/`).

**Adenda E3 (13-sep, noche 2): la cola O(e²) CON el sector de
velocidades.** Bajo preinscripción congelada (escalera E3a y arnés E3b
con desenlaces separados), el modo creciente exacto como serie
t^g·Σ c_n t^{−n/3} sobre el sistema lineal completo (ξ = 1) reproduce
en el orden dominante µ₀ = 1/(1−α_a/2) y η₀ = 1 exactos en toda la
malla, y en el orden e² da los coeficientes verdaderos: en el punto del
arnés (1.05, 0.3) η − 1 = 17.0735·e² — la cifra que la derivación
independiente del autor había predicho (17.07). Los residuos del polo
1/(λ_K−1) salen exactos: **P_η = 3α_a/(2−α_a)** (a 4e-16) y **P_µ =
−P_η·p(2p−1)/3** (a 1e-8) en α_a = 0.1/0.3/0.6, con partes regulares
Q_η = 4.97/6.09/8.36 y Q_µ = −4.98/−6.23/−9.42 (**E3a = A**). Se revisa
la lectura anterior: el polo es real en los potenciales de gauge
unitario — el parámetro pequeño es aH/(c_s k), el horizonte de sonido
del khronon — *[y, como fija la adenda E5, de gauge en η: en invariantes
η_N ≡ 1 y el polo sobrevive solo en µ_Δ]*; lo que la truncación
QS exagera es el residuo (2 frente a P_η: ×3.8 en α_a = 0.3, ×12.7 en
0.1) y su dependencia en α_a, y el «×28» de #18 era el cociente
ventana-promediado frente a un arnés cuyas condiciones iniciales
QS-consistentes excitaban los modos oscilatorios del khronon (el control
muestra η − 1 oscilando entre −4.6×10⁻² y +5.4×10⁻² en la ventana: el
arnés E2 no medía la cola). La confirmación numérica con condiciones
iniciales adiabáticas tomadas de la propia escalera da pendientes de
η − 1 y µ_loc − 1 frente a e² al 0.9–6.8 % de la escalera en los tres k
(17.80/17.41/17.23 frente a 17.07; −12.13/−11.72/−11.53 frente a
−11.36), ninguna compatible con la QS truncada (46.2, −3.79); pero el
brazo de mayor e (e ≤ 0.02, k/H0 = 66.7) da exponente log-log 2.14
∉ [1.9, 2.1] — curvatura O(e⁴) que el umbral congelado no calibró — y
la regla clasifica **E3b = C (abierto)**. No se retoca: el estatuto
queda «derivación exacta de la escalera, pendiente de confirmación
numérica independiente»; una preinscripción futura y distinta podrá
fijar e_fit_max menor o un ajuste con término e⁴. Magnitudes
ilustrativas (z = 0, e = (H0/c)/k, escalera completa): (1.05, 0.3) →
η − 1 = 7.6×10⁻⁴ en k = 0.05 h/Mpc y 4.8×10⁻³ en 0.02; (1.012, 0.012) →
2.7×10⁻⁴ en 0.05, donde la parte regular domina (solo polo:
6.7×10⁻⁵). Fronteras: Q solo numéricas, ξ ≠ 1, superhorizonte, O(e⁴),
PPN. **Estatuto**: derivación interna (E8) sin datos
(`results/2026-09-13_mu_eta_atlas_tail/`).

**Adenda E4 (15-sep): PPN del sector y lo que la cota hace — y no
hace — con las colas.** Bajo preinscripción congelada (`39b9f56b8eae`),
los parámetros de marco preferido se derivaron como límite khronométrico
(c_ω → ∞) de Einstein-aether (Foster & Jacobson 2006), con el mapeo al
sector del tratado α_a = α, λ_K = 1 + λ, ξ = 1 ⇔ β = 0 verificado en diez
identidades (c_s² y G_cosmo/G_N de la derivación desde la acción
reproducidos): **α₁ = −4α_a, α₂ = −α_a/2 + O(α_a²)**. Las cotas de campo
débil (LLR |α₁| < 10⁻⁴; giro solar |α₂| < 4×10⁻⁷; transcritas, con
aviso de procedencia) dan **α_a ≤ 8×10⁻⁷**, casi independiente de λ, y
la corrección −α_a/2 a G_cosmo/G_N − 1 queda en 2×10⁻⁵ relativa: el
Contraste de los Residuos toma su forma limpia −(3/2)ε_K. Pero la
expectativa «colas de sonido inobservables» no sobrevive a la escalera
exacta: en (λ_K, α_a) = (1.012, 10⁻⁶) el coeficiente COMPLETO de e² en
η − 1 es **4.58**, frente a 3.5×10⁻⁸ del residuo del polo, y no se apaga
con α_a → 0 (4.58/4.59/4.71 en α_a = 10⁻⁶/10⁻⁴/10⁻³; 4.51/4.58/4.84 en
λ_K = 1.001/1.012/1.05), con µ_loc − 1 = −(η − 1) a cuatro cifras
(µ_loc·η = 1 + O(e⁴): la cola vive en Ψ y Φ obedece Poisson a O(e²)).
En α_a ≪ λ_K − 1 la expansión P/(λ_K−1) + Q de E3 no aplica y el polo no
es el término dominante. Magnitud (z = 0, e = (H0/c)/k): η − 1 =
1.3×10⁻³ en k = 0.02 h/Mpc, 2.0×10⁻⁴ en 0.05, 5×10⁻⁵ en 0.1 — alcanza el
umbral preinscrito de observabilidad (10⁻³) y la regla clasifica **E4a =
B, hallazgo estructural**, publicado sin retocar. Dos precisiones de
honestidad: (i) el hallazgo descansa en la escalera, cuya confirmación
numérica independiente sigue abierta — E4b re-preinscribió E3b con un
ajuste e² + e⁴ y volvió a dar **C**: los brazos k/H0 = 133.3 y 200 pasan
al 3–10 % de la escalera, pero el de 66.7 (e ∈ [0.01, 0.02]) no separa
e² de e⁴ (lección para una preinscripción futura, no retoque de esta);
(ii) el orden de límites (α_a → 0 con c_s → ∞ y luego λ_K → 1) no
restaura η = 1 de GR — estructura de la clase khronométrica que queda
declarada, no explicada. Para el correo técnico: la frase «cancelación
de Atlas» sigue en pie al orden dominante; la frase «colas
inobservables» no debe usarse. **Estatuto**: derivación interna (E8)
con cotas transcritas; hallazgo condicionado
(`results/2026-09-15_mu_eta_atlas_ppn/`). *[Superado en su lectura
física por la adenda E5, abajo: la cola de η era de gauge unitario; la
parte PPN queda verificada punto por punto por el autor.]*

**Adenda E5 (15-sep, noche): la corrección de gauge — η_N ≡ 1 y la cola
de η de E3/E4 no era observable.** La verificación del autor detectó el
error de fondo, común a su Parte C.2 y a E3/E4: los seis campos de la
escalera son los de gauge unitario, y la reparametrización temporal al
gauge newtoniano actúa al mismo orden e² que las colas — b arranca en
n = 1 de la torre y ḃ, Hb caen exactamente en n = 4. Las reglas (ψ → ψ
− Ṫ, b → b + T, φ → φ + HT, δ → δ + 3HT, l1 → l1 + mT) se re-derivaron
de δg_μν → δg_μν − L_ξ ḡ_μν sobre el ansatz del propio módulo y dan los
invariantes Ψ_N = ψ + ḃ, Φ_N = φ − Hb, Δ = δ − 3H l1. Bajo preinscripción
congelada (`e81883295dd2`) con el control externo del autor declarado
(sus valores, calculados fuera del repo antes de la congelación, no
fijaron umbrales), la MISMA escalera exacta en invariantes da, en los
nueve puntos, **coeficiente de e² de η_N − 1 exactamente 0** y η_N,0 =
1, con la identidad de torre Φ_N,n = Ψ_N,n hasta n = 8: el teorema «sin
estrés anisótropo lineal (a_i a^i es cuadrático en ∂N; (1−λ_K)K² es pura
traza) ⟹ la ecuación ij sin traza es la de GR ⟹ η_N ≡ 1» queda fijado.
El polo y la parte regular de la cola de η eran de gauge, y el patrón
µ_loc·η = 1 + O(e⁴) que E4 observó es la firma de la reparametrización
temporal, no física. La única cola observable es la de µ_Δ, y aquí la
regla congelada hace su trabajo: la forma cerrada preinscrita −α_a
(aH/(c_s k))² se cumple en α_a ≳ λ_K − 1 (cocientes escalera/forma
1.0002 en (1.0001, 10⁻⁴) y 1.025 en (1.012, 0.012); 0.52 y 0.38 en
α_a = 0.3 por correcciones O(α_a)) pero **falla en α_a ≪ λ_K − 1**,
donde la cola es de PRIMER orden en α_a — −0.752/−0.774/−0.852·α_a·e²
en λ_K = 1.001/1.012/1.05, cocientes 751–39648 — y no de segundo, así
que **E5a = B**, publicado sin retocar. Su magnitud en la cota PPN
(1.012, 10⁻⁶) es µ_Δ − 1 = −2.2×10⁻¹⁰ en k = 0.02 h/Mpc: inobservable
(umbral 10⁻⁵ cumplido). El arnés adiabático en invariantes (brazo 66.7
retirado y ventana a ≥ 0.3, calibrados sobre las pendientes UNITARIAS
ya publicadas) da **max|η_N − 1| = 2.5×10⁻¹⁴** sobre la trayectoria en
los cuatro brazos y s_µΔ/escalera = 1.026/0.944/0.976/0.986 → **E5b =
A**: la primera confirmación numérica de la escalera (el control
unitario en la misma trayectoria queda al 1–4 %, sin gobernar). Lo que
cambia: la lectura física de E4a = B queda superada (el registro se
conserva bajo su definición preinscrita), E3b = C y E4b = C permanecen
como arneses en gauge unitario, «cancelación de Atlas» sigue en pie y
«colas inobservables» se restaura con más fuerza — η_N ≡ 1 en el sector
Atlas y toda la firma sub-horizonte de (µ, η) es del canal Cronos. Lo
que el control externo no anticipó: el orden α_a¹ de µ_Δ cuando α_a ≪
λ_K − 1 (sus puntos pequeños tenían α_a = λ_K − 1); no hay forma cerrada
aquí y es materia de una preinscripción futura, nunca de esta.
**Estatuto**: derivación interna (E8) sin datos
(`results/2026-09-15_mu_eta_atlas_gauge/`).

### 3.16 La primera aplicación a Dovekie real: desenlace INDETERMINADO bajo la regla congelada (13-sep-2026)

Con la preinscripción congelada ANTES de la primera lectura de la
columna MU del HD (sha256 `f630834326c0`, en un commit que contiene su
generador; precondición: el PASS de mocks de §3.13 citado por hash), el
pipeline SN entró por primera vez en el dato real. Benchmark SN-only
ΛCDM like-for-like con el chain oficial nautilus de DES-Dovekie (mismo
soporte de prior [0.10, 0.50]): Ω_m = 0.3305 (+0.0154 −0.0150) frente a
0.3306 ± 0.0154, Δ = −0.005σ — el pipeline reproduce el número oficial
sobre los bytes oficiales. Contraste conjunto Dovekie + CC + BAO (n =
1857), STAT+SYS principal: χ²_ν(ΛCDM) = 0.893; MCMC gana Δχ²_min = 0.04
con dos parámetros más, ΔAIC = +3.96, ΔBIC = +15.01 pro-ΛCDM; ε_Λ =
+0.018 ± 0.037 con 0 ∈ CI95; STATONLY (robustez) idéntico en signo y
orden (ΔBIC = +14.90, ε_Λ = +0.012 ± 0.037). La regla preinscrita
(C → B → A → INDETERMINADO) descarta tensión (C) y preferencia (B); el
desenlace aburrido A exigía además σ(ε) ≥ 0.04 y se obtiene 0.0371, el
único criterio de A no cumplido — un umbral fijado sin calibración
conjunta previa (los mocks de §3.13 eran SN-only). **La regla no se
retoca**: el desenlace se publica como INDETERMINADO con el mismo peso
que los demás; recalibrar esa puerta es materia de una preinscripción
futura y distinta, no de esta corrida. Lo que los números dicen por sí
mismos, sin veredicto: las SNe Dovekie no identifican ε_Λ y el fondo
MCMC no es preferido, en línea con la nota estructural de 6A (z_trans ≈
9). Roles inalterados (Unite = benchmark armonizado, NO replicación
independiente).
**Estatuto**: comprobación interna del pipeline sobre datos reales bajo
preinscripción (E8) — no demostración física
(`results/2026-09-13_dovekie_real/`).

### 3.17 El Contraste de los Residuos vía BBN: G_cosmo/G_N libre sobre EMPRESS XV + D/H (14-sep-2026)

El observable más directo del modelo (§3.4, refinado en §3.15):
δ_G = G_cosmo/G_N − 1 = (2ξ−α_a)/(3λ_K−1) − 1 = −1.77 % con ε_K = 0.012,
α_a = 0 y ξ = 1. BBN solo ve G en la tasa de expansión, así que la
respuesta de Y_P y D/H se derivó con un código BBN público (PRyMordial,
clone a commit fijado) reescalando la masa de Planck en H con η_b fijo:
Y_P ∝ G^0.358 y D/H ∝ G^0.977, con el sistemático nuclear NACRE II −
PRIMAT (+0.059×10⁻⁵ en D/H) como error teórico. Con la preinscripción
congelada ANTES de leer los valores (`ec3c8f48597c`), el brazo principal
— δ_G libre, N_eff = 3.044, ω_b (Planck 2018) y τ_n (PDG) marginalizados
de forma exacta — sobre Y_P = 0.2402 ± 0.0040 (EMPRESS XV, jun-2026) y
D/H = (2.527 ± 0.030)×10⁻⁵ (Cooke+2018) da **δ_G = +0.003 ± 0.025, CI95
[−0.045, +0.053]**: el −1.8 % y el 0 están dentro → **desenlace A**, la
banda compatible sin identificación. Los brazos de contexto explican el
número: solo Y_P da −0.072 ± 0.043 (el helio bajo de EMPRESS «apunta» al
signo del modelo) y solo D/H da +0.036 ± 0.031, con una tensión interna
de 2.0σ — la conocida entre EMPRESS y el deuterio con tasas PRIMAT —, de
modo que el conjunto no favorece signo alguno; el signo del central no
es señal (E13). La degeneración G ↔ N_eff queda medida (cresta
dΔN_eff/dδ_G = −6.5, la correspondencia por época) y el control con
Aver+2021 da +0.016 ± 0.024. Lo que cambia: el dato vigente del frente 6
pasa de la cota de Alvey+2020 a una banda propia bajo preinscripción con
la misma conclusión. **Aviso de procedencia**: los cinco valores son
publicados y transcritos a mano — arXiv, INSPIRE, DOI y revistas no son
alcanzables desde el entorno de código. El 15-sep el autor comprobó las
dos transcripciones observacionales contra los resúmenes primarios
(exactas) y reprodujo las cifras de forma independiente (solo Y_P −7.6 ±
4.5 % frente a −7.2 ± 4.3 %; pulls idénticos): el «provisional» se
levanta en cuanto a valores publicados; los bytes oficiales siguen sin
ingerir y el manifest y el artefacto lo declaran. Fronteras: ξ_e no
modelada; α_a = 0.
**Estatuto**: contraste interno bajo preinscripción (E8) sobre valores
transcritos — no demostración física (`results/2026-09-14_bbn_g/`).

### 3.18 La amplitud de Cronos es una: el cierre cosmológico queda excluido dentro de los halos y la cola k² muere por consistencia (17-sep-2026)

Los parámetros (α₀⁻¹, ρ_c) de la Ley de Cronos débil (Def. 11.1) entran
en toda la dinámica solo por la amplitud A ≡ α₀⁻¹/ρ_c^(3/2), con
ε_c = A·ρ^(3/2) (§3.8, `dynamics/weak_field.py`). El repositorio llevaba,
sin decirlo, dos cierres incompatibles: el galáctico, A_Sculptor =
6.2×10⁻⁷ (M☉/pc³)^(−3/2), obtenido del problema inverso en Sculptor y
congelado en la preinscripción 5E (§3.8), y el cosmológico, ρ_c =
200·ρ̄_m con α₀⁻¹ = 10⁻⁶ (A ≈ 41.5), elegido en la nota de teoría del
13-sep e implementado en §3.14. Difieren en 6.7×10⁷. La condición de la
que nace la cota (11.5) es la subdominancia punto a punto c²ε_c ≲ |Φ_N|
(lectura (P), Cor. 11.3c), y el cociente D ≡ c²ε_c/|Φ_N| decide: dentro
de un halo NFW de 10¹¹ M☉ (c = 10) el cierre cosmológico da D = 5.4 en
r₂₀₀, 10⁴ en r_s y 3×10⁵ en 2.3 kpc — **excluido dinámicamente**; en la
parte externa con ε_c ≪ 1 (falla la subdominancia, no el régimen débil),
y por debajo de ~r_s rompiendo también el régimen débil (ε_c = 2.2 en
0.38 kpc). El galáctico cumple la subdominancia en r ≥ 0.1 kpc (D = 0.77
en 0.1 kpc, 4×10⁻³ en 2.3 kpc) y coincide con la amplitud máxima
subdominante en potencial de ese halo (A_max = 1.3·A_Sculptor). Pero la
subdominancia en potencial no es subdominancia en fuerza: en una
cúspide NFW el cociente dinámico D_F = |c²dε_c/dr|/g_N vale
(3/2)|dlnρ/dlnr|·|Φ_N|r/(GM(<r)) veces D_Φ — un factor 20–600, porque
|Φ_N| es finito en el centro mientras GM(<r)/r → 0 — y con A_Sculptor
D_F = 0.73 en 1 kpc, 7.4 en 0.4 kpc y 230 en 0.1 kpc: el término
+c²∇ε_c domina la gravedad dentro de ~0.9 kpc del halo de 10¹¹ M☉
(hallazgo de la sesión de código al pilotar el Nivel A). Es coherente con
el origen de A_Sculptor: en 5A/5B se calibró para que Cronos sustituya a
la materia oscura en Sculptor con bariones solos, así que sumada a un
halo CDM domina su centro. En un halo enano de 10¹⁰ M☉ (c = 13,
suavizado 50 pc) también la subdominancia en potencial se rompe por
debajo de 0.14 kpc. El cálculo de partida es del autor (sesión de
verificación del 17-sep) y se reproduce en el repo de forma independiente
(`dynamics/cronos_amplitude_validity.py`), sin desenlaces preinscritos:
es consistencia, no experimento. **Consecuencias**: la amplitud del
modelo operativo es una y es A_Sculptor, reutilizada entre frentes; el
sector µ/η del canal Cronos (§3.14, E1) se computó con el cierre excluido
— sus identidades siguen exactas para el cierre que declaran y el
artefacto no se retoca —, pero con A_Sculptor ε̄_c(hoy) = 5.3×10⁻¹⁸ y
µ − 1 = 5.8×10⁻¹² en k = 0.2 h/Mpc, z = 0: la cola k² muere por
consistencia interna, no solo por falta de sensibilidad, y la cota k²
sobre full-shape sale de la hoja de ruta. En la misma ronda el registro
recibe cuatro filas que nombran huecos: c ≡ v_LR (claim no derivado, sin
fila hasta hoy), el mapa S(z) (convención S_today = 95, α = 1), el
diccionario primordial → cosmológico (hueco declarado: nadie ha empezado
la fórmula que lleva el estado en S = 1.001 a los parámetros
cosmológicos) y la lectura operativa del contenido de materia (ΛCDM +
canales + Cronos; la lectura fuerte sin CDM es hipótesis distinta, no
implementada). **Estatuto**: cálculo de consistencia interna (E8)
(`results/2026-09-17_cronos_amplitude_validity/`).

### 3.19 Nivel A del frente 5: halo aislado con y sin Cronos v3 a A_Sculptor — INDETERMINADO (17/18-sep-2026)

Primera corrida de producción del frente 5 con la amplitud única:
un halo NFW aislado de 10¹¹ M☉ (c = 10, N = 4×10⁵, suavizado 0.1 kpc)
en equilibrio de Eddington, integrado 2 Gyr con y sin la ley completa
del Cor. 11.3 (fuerza +c²∇ε_c, fricción con compuerta y lapso, con
ε_c = A_Sculptor·ρ^{3/2} sobre un campo medio esférico). La
preinscripción se congeló antes de generar ninguna condición inicial
(`72c534dc1c09`, commit con el generador), con pilotos de desarrollo
declarados que fijaron el instrumento y ningún umbral. Seis brazos: a
(newtoniano, tres semillas), b (Cronos completo, tres semillas),
b_static (campo congelado, única corrida Cronos con energía conservada
por construcción: |ΔE/E| = 2.6×10⁻⁴, puerta superada), b_res (control
de convergencia del campo, k_inner = 128), c (cierre cosmológico, que
la regla exigía parar en t = 0 y paró: ε_c,max = 3.5 — la exclusión de
§3.18 queda registrada dinámicamente) y c′ (10·A_Sculptor, control de
respuesta). Las puertas de energía (brazo a, ≤ 1.4×10⁻⁴), equilibrio
(0.019 dex en [1, 10] kpc) y régimen débil (ε_c ≤ 8×10⁻⁷ en todas las
comprobaciones de b) se superaron.

**Desenlace INDETERMINADO por la letra de la regla**. El control c′
debía desplazar ρ en [0.4, 1) kpc al menos 0.10 dex «a 0.5 Gyr»; la
regla de parada, igualmente congelada (v_well = √(2c²ε_c) > 1000 km/s:
el modelo ha abandonado cualquier régimen de enana), lo detuvo en 0.055
Gyr, y en ese instante esa banda había cambiado +0.05 dex. Dos reglas
congeladas colisionaron; el resultado se retiene entero y ningún umbral
se toca. Lo que c′ sí muestra, publicado sin veredicto, es que la
implementación responde a 10·A_Sculptor donde la regla no miraba:
M(<0.4 kpc) ×7.1, v_well 306 → 1000 km/s y ε_c(ε_soft) ×11 en 55 Myr.

**Números descriptivos** (sin veredicto; E13: el signo no es señal).
En 2 Gyr, log10(ρ_b/ρ_a) = −0.28 dex en [0.4, 1) kpc (−0.39, −0.01,
−0.45 por semilla), −0.10 en [1, 2.3) kpc y ~0 fuera de 2.3 kpc; el
cociente M_b/M_a dentro de 0.4 kpc pasa por 1.0 → 4.3 (0.25 Gyr) →
0.68 (0.5 Gyr) → 1.08 (2 Gyr) en media de semillas, con una semilla
que apenas se mueve (×1.3) y dos que se contraen ×3–×8 y luego vacían
el interior. Lo único robusto en todos los brazos Cronos es la
**contracción inicial** del interior en los primeros 0.25 Gyr (×8.4,
×1.3, ×3.3 por semilla; ×2.6 con k_inner = 128; ×2.2 con el campo
congelado; ×7.1 a 10·A_S en 55 Myr): la dirección coincide con la
expectativa declarada en la preinscripción (D_F > 1 dentro de 0.9 kpc),
la amplitud dispersa ×1.3–×8 entre semillas y resoluciones. **No está
convergido**: el control b_res difiere de b en +0.83 dex en M(<0.4 kpc)
a 2 Gyr (umbral 0.15), y las corridas con campo dinámico ganan energía
(+1.7 %, +0.5 %, +1.4 %; b_res +0.25 %) que no es la energía potencial
de Cronos (U_Cronos ~ 3×10⁻³ de E). La sospecha, declarada como
hipótesis para la ronda siguiente y no como resultado, es trabajo del
campo medio retardado (τ_avg = 50 Myr frente a t_dyn ≲ 10 Myr dentro
de 0.4 kpc): un potencial que llega tarde a un colapso bombea energía y
el vaciado posterior coincide con la ganancia. El ajuste de forma en
[0.5, 5] kpc no es C: la media de b prefiere cored por 0.02 dex (< 0.05)
y a prefiere NFW.

Dos correcciones de código se declaran tras la primera pasada del
análisis, sin efecto sobre umbrales ni desenlace: el retorno temprano
del brazo c no escribía la lista de eventos (c se re-ejecutó,
determinista, en 3 s) y la máscara de bandas daba −inf con un bin
vacío en una semilla. **Lo que la ronda siguiente necesita** (decisión
del autor, con preinscripción nueva): criterio de c′ compatible con la
regla de parada; campo sin retardo o con τ_avg ≤ t_dyn interior; serie
de convergencia k_inner 64/128/256 como puerta; zoom del interior (N ≥
10⁶ o multimasa); y el trabajo del campo publicado para separar la
física del retardo. **Estatuto**: experimento numérico interno (E8)
preinscrito y ejecutado, INDETERMINADO — ni núcleo ni contracción
convergida; ninguna predicción del modelo operativo para el interior de
las enanas sale de esta ronda
(`results/2026-09-17_cronos_halo_nivelA/`).

### 3.20 El reloj S como simulador de consistencia: la trayectoria S₀ → S_{1,001} en un solo bucle, con cada eslabón no derivado etiquetado (21-sep-2026)

Tercer paso del orden confirmado el 17-sep. `core/s_clock.py` recorre
la trayectoria pre-geométrica en un solo bucle en σ: el Flujo del
Camino (4.2) sobre el Basal completo, con la inclinación −η·χ, desde el
punto de escape de Coleman (el punto más allá de la barrera donde V
vuelve al valor del falso vacío); el contador de descarga f = (V_fv −
V)/T₀ y el índice entrópico S = ∫Σ̇dσ/T₀, normalizado para que la
descarga completa sea S = 1 (el número del cuanto es convencional,
F.4); los acoplos (M0², B, C0) por las β de Fokker–Planck de la Def.
4.4 cuando se pide; el discriminante D(S) y los colapsos como eventos
d → d + 1 con el álgebra C(d+1,0); los sellados por congelación de
c_eff (§5.3) y m_eff (§5.4); la fase θ; y en S = 1,001 la Rotación de
Florencia, que entrega el estado inicial cosmológico. Cada comprobación
lleva su etiqueta: derivado, impuesto, declarado o publicado.

**Derivado y superado** (δ₀ = 0.01, valor de prueba; el tratado no
asigna valor a δ₀): T₀ = c̄δ₀³ a 10⁻¹⁵ sin inclinación; monotonía del
Camino, producción entrópica ≥ 0 y exclusión; salida al polo de masa;
la identidad S ≡ f a 2×10⁻¹³ — que exige proyectar el flujo sobre el
dominio físico φ_E ≥ 0 y contar solo el trabajo de las componentes
activas (la ligadura no produce entropía: Teo. 4.5 sobre el primer
cuadrante); álgebras C(2,0), C(3,0), C(4,0) anticonmutantes y
euclidianas en cada colapso; el séxtico marginal exactamente en d = 3;
en Florencia la firma −+++ con un solo giro y el control negativo de
dos giros (dos direcciones temporales), la positividad por reflexión en
la loncha con el control J < 0 que la rompe, la identidad m_H =
√(2β₃)v₃; el Sello de Newton en (1,1) y la recuperación de ΛCDM con
ε = 0, exactos.

**Impuesto o declarado, y por qué**: los umbrales 0.009/0.099/0.999
(Prop. 8.1: λ = 10 es el frente 2); la nucleación (Γ₀(δ₀) no está
calculada: el reloj arranca en el punto de escape con σ = 0); las
formas de sellado (el tratado fija la propiedad — logística, β_c = 0 en
S_0,099, dm_eff/dS = 0 en S_0,999 — pero la ec. 5.3 no está transcrita
en el repo: `DECLARED_FORMS` las publica y sustituirlas es un cambio
local); el cruce de la diagonal θ = π/4 (el flujo del Basal lleva θ al
polo de masa y lo mantiene: el transporte azimutal de la Prop. 3.5 no
está implementado y se impone para la entrega); los cuantos V3D en
1,000 y Florencia en 1,001 tras el residuo de descarga ε_res (F.3);
Φ_ten(S_1,001) = 0 por normalización; el δ₀ del ciclo siguiente (γ_R,
frente 4). El estado entregado — constantes selladas, λ_Ad y β₃, f
residual, θ y χ, canales iniciales = ninguno — se declara **ilegible
para la cosmología**: el diccionario primordial → cosmológico (§3.18)
es ahora un campo vacío en un artefacto, no una frase.

**Hallazgo del simulador**: la metastabilidad del falso vacío tiene un
δ₀ máximo. Con la inclinación η = ē·δ₀³ el falso vacío se desplaza a
ρ_fv ≃ η/(√2·M0²) (el residuo ⟨χ⟩ de la Prop. 3.4) y la barrera, de
altura ∝ δ₀³ frente a una inclinación ∝ δ₀^{7/2}, desaparece por encima
de δ₀_max = 0.1028 para las formas por defecto (m̄ = 1, b̄ = 3, ē = 1,
C₀ = 1): la condición de la Obs. 8.6 (D(S₀) > 0) es necesaria, no
suficiente. Por la misma razón la Tensión Primordial medida en el
paisaje completo supera la ley c̄δ₀³ en un 13 % (δ₀ = 0.01) a un 40 %
(δ₀ = 0.1): la corrección es O(δ₀^{1/2}) relativa, como la Prop. 3.4
anticipa al llamar subdominante a η·ρ₊. **Modo emergente (diagnóstico,
sin estatuto)**: con las β canónicas de Fokker–Planck y un diccionario
τ ∈ [10⁻³, 1] declarado, D(S) se hunde (Obs. 8.6) pero no cruza cero
antes del fin del recorrido — para τ grande M0² cruza cero antes que D
(el falso vacío se destabiliza por la masa, no por la espinodal) y para
τ pequeño el descenso termina con D > 0 — así que en el rango explorado
los colapsos no emergen del Cruce de Victoria; el resultado depende de
τ y se publica sin veredicto. m_H(δ₀) se publica como función del
input: 52 GeV en δ₀ = 0.01, 125.4 GeV en δ_H = 0.0581 por construcción
de δ_H (Obs. 12.2), no por predicción. El artefacto lleva un **ledger**
obligatorio (cada comprobación y cada evento con su insignia de
estatuto y el recuento por insignia) y cada colapso publica sus dos
lecturas, S_imposed y S_emergent, siempre juntas; el visor estático
`viewer/s_clock.html` lee el artefacto y las muestra. **Estatuto**:
comprobación interna (E8) de consistencia entre estaciones; el
interruptor emergente solo tendrá estatuto cuando el frente 2 entregue
τ(S) y el diccionario exista (`results/2026-09-21_s_clock/`).

### 3.21 El criterio de Cronos–Jeans: la ley local es ultravioleta-inestable donde q > 1, y la amplitud única cae dentro en el halo del Nivel A, en Sculptor y en la vecindad solar (21-sep-2026)

Derivación del autor (sesión de verificación del 21-sep), reproducida
en el repositorio número a número. Un medio colisionless isotermo con
dispersión σ y potencial efectivo Ψ = Φ_N − c²ε_c(ρ) (Prop. 11.2)
responde a una perturbación estática con δρ = −ρ δΨ/σ²; la parte de
Cronos es autoconsistente, δΨ_C = −c²ε_c'(ρ) δρ, y la ecuación δρ =
+ρc²ε_c'δρ/σ² tiene solución no trivial cuando ρc²ε_c'/σ² ≥ 1. Con
ε_c = A·ρ^{3/2}, ρε_c' = (3/2)ε_c:

    q(ρ, σ) ≡ (3/2)·c²·ε_c(ρ)/σ_1D² ≥ 1  ⟹  inestable a toda longitud de onda.

En el límite fluido ω² = (σ² − (3/2)c²ε_c)k² − 4πGρ: la gravedad es
infrarroja (tasa independiente de k) y la fuerza de Cronos, local, es
ultravioleta (tasa ∝ k·√(q−1)·σ). No hay longitud de Jeans propia: por
encima del umbral la escala más pequeña resuelta crece más deprisa, y
un N-cuerpos con esta ley **no puede converger** cuando q > 1 — refinar
la resolución del campo aumenta la tasa, y el ruido de Poisson del
estimador la siembra (semillas distintas, desenlaces distintos).

**Tres aplicaciones con la amplitud única** (`dynamics/cronos_jeans.py`;
ninguna ingiere datos). (1) El halo del Nivel A (NFW de 10¹¹ M☉, c = 10,
Jeans isótropo): q = 348 en 0.05 kpc, 74 en 0.1, 3.5 en 0.4 y 1.03 en
0.7; r_CJ = 0.71 kpc con 1.6×10⁸ M☉ dentro (0.25 kpc con 0.1·A_S, 0.18
con 0.05·A_S). Respuesta a la pregunta encuadrada del brief («¿existe
una solución de halo Cronos convergida con A_Sculptor?»): no para ese
estado inicial — el problema continuo es mal puesto dentro de r_CJ; la
regla de parada v_well = 1000 km/s equivale a ρ = 4.3 M☉/pc³, es decir
1.6×10⁸ M☉ dentro de ≈ 0.2 kpc. (2) La vecindad solar (losa de McKee,
Parravano & Hollenbach 2015): g_C/K_z = 7.3 en 50 pc, 6.4 en 100, 2.1
en 300 y 0.67 en 500; límite de Oort efectivo de Cronos 0.75 M☉/pc³
(solo estrellas 0.115) frente a 0.10 ± 0.01 medido (referencia, no
ingerida); cotas A ≤ 0.048·A_S (g_C ≤ 0.1·K_z en 300 pc) y A < 0.16·A_S
(q < 1 en el plano con σ_z = 20 km/s): A_req(Sculptor)/A_max(K_z) ≳ 20.
(3) Sculptor con el montaje congelado del 5E: q = 2.0–2.4 dentro de
≈ 430 pc — el equilibrio que calibra A_Sculptor es inestable según el
criterio — y σ_los(R) predicha de 19.1 km/s en 10 pc a 2.6 en 500 pc
(pico central y exterior newtoniano; el promedio pesado por
luminosidad da 9.2 por construcción): esa forma queda congelada en la
fila `jeans-dsph` antes de la ingesta. Test 1D del umbral (láminas sin
gravedad, `cronos/cronos_jeans_1d.py`): q = 0.8 se queda en el ruido de
Poisson, q = 1.2 crece, q = 2 satura en ~2 % del tiempo de cruce, q = 4
es no lineal desde el inicio; la ley de la tasa ∝ k no se mide aquí (es
la puerta del test preinscrito del frente 5 refundado).

**Lo que el Nivel A muestra en realidad** (lectura añadida a §3.19):
el mecanismo del vaciado y de la ganancia de energía es el estimador del
campo — la malla logarítmica se fija en la primera llamada (r_inner =
0.251 kpc toda la corrida) y la pendiente se recorta a ≥ 10⁻³, de modo
que cuando la masa se concentra bajo r_inner la densidad estimada cae
~100× y ε_c ~1000×: el pozo se retira exactamente cuando la ley exige
que se profundice (trinquete: atrapar con pozo hondo, soltar con pozo
retirado; ΔE ≈ M_atrapada·v_well²/2 cuadra en orden de magnitud). El
retardo τ_avg modula el ritmo; no es la causa. La fricción y el lapso no
frenan nada a esta amplitud (∫Γdt ~ 10⁻⁴ en 2 Gyr; N − 1 ~ −5×10⁻⁶); y
con campo autoconsistente instantáneo la energía conservada es K + W +
(2/5)U_C, no K + W + U_C (con campo congelado sí: por eso b_static
conserva mejor). El desenlace INDETERMINADO se mantiene; lo que cambia
es qué debe preguntar la ronda siguiente: el Nivel A-2 contratado
(convergencia k_inner, zoom) no puede converger y se retira; lo
sustituye el frente 5 refundado (test del criterio con predicción
convergida; halo con A_S y 0.05·A_S y balance 2/5; preinscripciones
observacionales Oort–K_z y perfil de Sculptor; ingesta SPARC oficial
sin tocar A_Sculptor; cita de la RAR: 1609.05917 y 1610.08981). Las
salidas que el propio tratado admite (§11.4) — regularización
saturante ζ = ζ₀ρ/(ρ+ρ*), no localidad a escala L, amplitud pequeña —
son tres estructuras distintas, y decidir cuál es la del modelo es
trabajo del diccionario. **Estatuto**: derivación (E8) con control
externo del autor reproducido, expectativas declaradas (E13) y sin
datos ingeridos; nada es resultado observacional; A_Sculptor y todos
los umbrales intactos (`results/2026-09-21_cronos_jeans/`).

### 3.22 La salida de S₀: bounce de Coleman, escape de Kramers y el reloj S arrancando en el estado nucleado (21-sep-2026)

El eslabón inicial del reloj S que el repaso del 16-sep declaró
ausente. `core/nucleation.py` resuelve el instantón O(d) del campo
radial sobre el corte θ = 0 del Plano Dual (la salida ocurre hacia el
polo de masa, Prop. 3.5) con el paisaje completo, inclinación incluida:
φ'' + (d−1)φ'/r = V'(φ), φ'(0) = 0, φ(∞) = φ_fv, por disparo
overshoot/undershoot; publica la acción B, Γ₀/A = e^{−B} (el prefactor
A es dimensional y el tratado no lo fija), el centro del bounce y la
fracción de T₀ ya descargada allí, f₀. Ley de escala medida sobre
δ₀ ∈ [0.002, 0.064]: B ∝ δ₀^{−1.49} para d = 4 (argumento sin
inclinación 3 − d = −1: Γ₀(0) = 0 sin necesidad de prefactor, la
inercia eterna del perfecto) y δ₀^{−0.40} para d = 3 (argumento 0: B
tiende a constante y el axioma exige A(δ₀) → 0). **Hallazgo**: la
barrera del falso vacío vale entre el 1 % y el 8 % de T₀ (pared gruesa;
la pared delgada no es aplicable y se publica solo como referencia), y
en ese régimen el instantón entrega el campo casi en el vacío
verdadero — f₀ = 0.9996 con d = 4 y 0.97 con d = 3 en δ₀ = 0.01 — de
modo que la descarga y sus umbrales 0.009 y 0.099 (y 0.999 para d = 4)
caen **dentro de la nucleación**, no a lo largo del Flujo del Camino:
el reloj S con `nucleation='bounce'` los dispara en σ = 0 con esa nota,
publica f₀ como entropía de nucleación y comprueba S ≡ f − f₀ (2×10⁻¹³).

La otra nucleación es la propia de la dinámica del tratado: el Flujo
del Camino es disipativo y de primer orden (Axioma 4) y su completación
estocástica (Def. 4.4) añade difusión entrópica, así que el escape del
falso vacío es un problema de Kramers sobreamortiguado, Γ_K =
√(V''_fv|V''_b|)/(2πG)·exp(−ΔV_b/D_ent), con D_ent declarada (el
diccionario t ↔ σ no la fija). Medido: prefactor ∝ δ₀^{1.85} (argumento
2) y barrera ∝ δ₀^{2.44} (argumento 3, erosionado por la inclinación):
Γ_K(0) = 0 por el prefactor — el paisaje plano de δ₀ = 0 no tiene
curvatura que fije un ritmo — mecanismo opuesto al bounce, donde B → ∞.
Cuál de las dos es la nucleación del tratado (instantón conservativo o
escape disipativo) es decisión del diccionario, no del código. La
integración conjunta Φ_Ad ⊗ λ_i desde el bounce (modo emergente, τ ∈
[10⁻³, 10⁻¹]) no produce ningún cruce D = 0: los eventos S_emergent se
publican vacíos, junto a los S_imposed. Convenciones declaradas: la
dimensión d (3 y 4 publicadas; el tramo pre-geométrico no tiene
espacio-tiempo), el prefactor, la normalización de la acción (G = ħ =
1), D_ent y el corte θ = 0. **Estatuto**: cálculo de consistencia (E8)
con la expectativa E13 cumplida por vías distintas en cada mecanismo
(`results/2026-09-21_nucleation/`).

## 4. Lo que estos resultados NO afirman

- El círculo de δ₀ **no se cerró ni se rompió**: se volvió una ecuación
  con un número esperando ser derivado (frente 4).
- Los ajustes v1/v2 **no muestran alivio** de las tensiones H₀/S₈; el
  fondo queda certificado como límite de recuperación de ΛCDM.
- m_H ≃ 125.3 GeV **no es una predicción** mientras β₃ no se derive sin
  el Higgs medido como entrada (Obs. 12.2; el empalme acota el frente,
  no lo cierra: el valor de β₃ con (M₀², B, C₀) sellados sigue
  condicional).
- El núcleo cored, los cinco órdenes de H.2.5 y las validaciones SPARC
  del corpus siguen **pendientes de producción** (frente 5); el medio
  paso del 2-ago es indeterminado por resolución, no cúspide observada,
  y el Nivel A del 17/18-sep es INDETERMINADO (§3.19): ni núcleo ni
  contracción convergida del interior.
- La cola k² del canal Cronos **no es una predicción con amplitud
  observable**: con la amplitud única A_Sculptor, µ − 1 ~ 10⁻¹² (§3.18).
  Las identidades de forma (η = 1/µ, µ − 1 = 2(Σ − 1)) siguen exactas.
- λ = 10 sigue calibrado (frente 2 — las β canónicas de Fokker-Planck
  dan cascada DSI en el régimen físico δ0 < 0.496 pero s0 depende del
  diccionario τ, no derivable); el signo de ν, condicional (frente
  4); la RP no estacionaria, abierta (frente 1).

## 5. Prioridades que esta nota deja planteadas

1. Derivar W_max desde la microdinámica del reinicio (frente 4) — decide
   el círculo; valor esperado 1.652×10⁻⁴.
2. Núcleo cored en cajas de producción (frente 5) — el Nivel A (halo
   aislado de 10¹¹ M☉ con y sin Cronos v3 a A_Sculptor) se ejecutó con
   desenlace INDETERMINADO (§3.19); la ronda siguiente exige
   preinscripción nueva con criterio de c′ compatible con la regla de
   parada, campo sin retardo y serie de convergencia del campo como
   puerta.
2b. El reloj S (`core/s_clock.py`) existe como simulador de
   consistencia (§3.20); lo que lo volvería emergente son los frentes
   con nombre de §3.18: τ(S) (frente 2), el diccionario primordial →
   cosmológico, el mapa S(z), c ≡ v_LR — y el transporte azimutal de
   la Prop. 3.5, que el reloj exhibe como ausente. La nucleación
   Γ₀(δ₀) existe desde §3.22 con convenciones declaradas (d, prefactor,
   D_ent): decidir cuál mecanismo es el del tratado es del diccionario.
3. Residuos al 1–2 % de precisión futura (la consistencia superada se
   volvería decisiva).
4. Los medios pasos de los frentes E y F quedaron ejecutados el mismo
   día de esta nota: el flujo KLS integrado mide la ley del walking
   (el «≃» de la ec. 8.4 cuantificado, divergencia −1/2, y el retraso
   del colapso ∝ ritmo^(−1/3) tras el Cruce — resultado del programa),
   y el juguete de RP no estacionaria establece que la simetría
   especular del perfil es condición SUFICIENTE para la RP (la
   necesidad no está demostrada): el running monótono viola la
   reflexión ingenua y reflejar también el perfil restaura la
   positividad exacta. λ = 10 (frente 2) y el
   sector de Wilson (frente 1) siguen abiertos, como declara el
   tratado. Desenlaces en `results/2026-08-02_kls_flow/` y
   `results/2026-08-02_rp_nonstationary/`.

La v35.1 (fe de erratas epistemológica, 4 de agosto de 2026) adoptó el
blindaje como convenio C6 — los CUATRO épsilon: δ₀ (input, sin valor
asignado), ε_Λ (transición), ε_K (Sello de Newton) y ε_c(ρ) (Cronos) —
junto con las reclasificaciones E1–E13 (`docs/erratas_v35.1.md`). Para
la v36 queda: cuantificar W_max o declarar la ecuación de cierre como
predicción condicional del modelo.

## 6. Reproducibilidad

```bash
pip install -e ".[dev]"
pytest tests/                          # 173 tests
python scripts/validate_all.py         # suite espejo del apéndice H
python scripts/run_delta0_circle.py    # §3.1-3.2 (semilla 20260802)
python scripts/download_data.py all    # datos con SHA-256 fijados
python scripts/run_production_fit.py --nsteps 8000   # §3.3 v1
python scripts/run_production_fit2.py --nsteps 8000  # §3.3 v2
python -c "from cosmology.residues_test import report; print(report())"  # §3.4
python scripts/run_profile_shape.py    # §3.5
python scripts/run_kls_flow.py         # §5.4 (frente E)
python scripts/run_rp_nonstationary.py # §5.4 (frente F)
python scripts/run_dovekie_real_prereg.py  # §3.16 preinscripción (no lee MU)
python scripts/run_dovekie_real.py     # §3.16 cinco brazos tras la doble barrera
python scripts/run_mu_eta_atlas_tail_prereg.py  # §3.15 adenda E3 (preinscripción)
python scripts/run_mu_eta_atlas_tail.py         # §3.15 adenda E3 (escalera + arnés)
python -m validation.bbn_response       # §3.17 respuesta BBN (requiere external/PRyMordial)
python scripts/run_bbn_g_prereg.py     # §3.17 preinscripción (no lee datos)
python scripts/run_bbn_g.py            # §3.17 contraste tras la barrera
python scripts/run_atlas_ppn_prereg.py # §3.15 adenda E4 (preinscripción)
python scripts/run_atlas_ppn.py        # §3.15 adenda E4 (PPN + cierre E3b)
python scripts/run_atlas_gauge_prereg.py # §3.15 adenda E5 (preinscripción)
python scripts/run_atlas_gauge.py      # §3.15 adenda E5 (invariantes de gauge)
python scripts/run_cronos_amplitude_validity.py  # §3.18 amplitud única de Cronos
pip install -e ".[nbody]"                                 # §3.19 (pytreegrav + numba)
python scripts/run_cronos_halo_prereg.py                  # §3.19 preinscripción (no genera ICs)
python scripts/run_cronos_halo_nivelA.py run              # §3.19 diez corridas (~14 h, 4 núcleos)
python scripts/run_cronos_halo_nivelA.py analyze          # §3.19 regla congelada
python scripts/run_s_clock.py                             # §3.20 el reloj S (consistencia)
python scripts/run_cronos_jeans.py                        # §3.21 criterio de Cronos–Jeans (derivación + test 1D)
python scripts/run_nucleation.py                          # §3.22 salida de S₀ (bounce, Kramers, reloj desde el bounce)
```

Desenlaces versionados: `results/2026-07-31_production_fit/`,
`results/2026-08-01_production_fit_v2/`, `results/2026-08-01_fertility/`,
`results/2026-08-01_empalme_wkb/`, `results/2026-08-02_delta0_circle/`,
`results/2026-08-02_profile_shape/`, `results/2026-08-02_kls_flow/`,
`results/2026-08-02_rp_nonstationary/`.

## Referencias

- Martínez Estellés, A. (2026). *Modelo Cosmológico de Múltiples
  Colapsos (MCMC) — Tratado de Fundamentos* (v35).
  DOI 10.5281/zenodo.20765373.
- Alvey, J., Sabti, N., Escudero, M. & Fairbairn, M. (2020). *Improved
  BBN constraints on the variation of the gravitational constant*.
  Eur. Phys. J. C 80, 148. arXiv:1910.10730.
- Scolnic, D. et al. (2022). ApJ 938, 113; Brout, D. et al. (2022). ApJ
  938, 110 (Pantheon+). Alam, S. et al. (2017). MNRAS 470, 2617 (BOSS
  DR12). Chen, L., Huang, Q.-G. & Wang, K. (2019). JCAP 02, 028
  (geometría comprimida de Planck 2018). Compilaciones de cronómetros
  cósmicos y fσ8: referencias por punto en `scripts/download_data.py`.
