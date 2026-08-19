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
ansatz. Las β reales (jerarquía de Fokker-Planck, Def. 4.4) siguen
siendo el hueco declarado; el módulo está listo para consumirlas.
Desenlace en `results/2026-08-05_victoria_exponent/`.

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
  del corpus siguen **pendientes de producción** (frente 5).
- λ = 10 sigue calibrado (frente 2); el signo de ν, condicional
  (frente 4); la RP no estacionaria, abierta (frente 1).

## 5. Prioridades que esta nota deja planteadas

1. Derivar W_max desde la microdinámica del reinicio (frente 4) — decide
   el círculo; valor esperado 1.652×10⁻⁴.
2. Núcleo cored en cajas de producción (frente 5).
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
