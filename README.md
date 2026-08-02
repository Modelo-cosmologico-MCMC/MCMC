# MCMC — Modelo Cosmológico de Múltiples Colapsos

[![CI](https://github.com/Modelo-cosmologico-MCMC/MCMC/actions/workflows/ci.yml/badge.svg)](https://github.com/Modelo-cosmologico-MCMC/MCMC/actions/workflows/ci.yml)

> **Estado de versión (julio 2026).** Esta implementación está alineada con el
> **Tratado de Fundamentos** (v35, junio 2026, DOI [10.5281/zenodo.20765373](https://doi.org/10.5281/zenodo.20765373)),
> registro canónico vigente del modelo, que lo refunda sobre ocho axiomas
> ontológicos. La alineación desde el *Tratado Unificado* (v32) se completó en
> julio de 2026 y está documentada commit a commit, incluida una revisión
> adversarial del núcleo numérico. Las constantes están organizadas por estatuto
> epistémico (Apéndice F), los valores de referencia del corpus se declaran como
> tales —no como salidas del código—, la parametrización superada se conserva en
> el bloque `LEGACY_V32`, y la integración continua mantiene esos criterios como
> guardias permanentes.
> Enlace permanente a la última versión del tratado: [10.5281/zenodo.14167831](https://doi.org/10.5281/zenodo.14167831).

Implementación de referencia en Python del **Modelo Cosmológico de Múltiples
Colapsos (MCMC)**, basado en el *Tratado Unificado* (2025-2026) de
Adrián Martínez Estellés. El repositorio reproduce la ontología completa del
modelo: dualidad primordial Mp/Ep, sellos ontológicos, Campo de Adrián,
WKB espinorial, programa de masas, cosmología con Λ_rel(z), Cronos N-body,
mass gap lattice, simulación cuántica de qudits y visualización.

## Estructura

```
mcmc_ontology/    Ontología pura: axiomas (v35 §1.2), constantes, S-map, potencial, Clifford, sellos
core/             La cadena deductiva ejecutable (caps. 2-10): Plano Dual, Basal,
                  Flujo, Florencia, RP (juguete), Discriminante, Gea, Victoria
validation/       Suite espejo del apéndice H (verificaciones + controles negativos)
mass_program/     B0–B6, P3, P4, M1 (running QCD), M2 (CKM)
cosmology/        H(z), Λ_rel, canales v35 (Ap. A), ajuste bayesiano y de producción
cronos/           Cronos v3 (cap. 11) + ciclo PM del Ap. B: paso entrópico,
                  Poisson modificado, canales, colapso reproducible
lattice/          Mass gap (D.5-D.6), β(S) entrópico, mapa Sₙ→jₙ, spinfoam (D.4)
quantum/          Qudit d=5 (C.1), compuertas Û(θ) (C.2-C.3), Lindblad (C.4), Γn/Γ0 (C.5)
visualization/    Animaciones, espectro de masas, plots cosmológicos
data/             Datos observacionales (vacío; poblar con scripts/download_data.py)
configs/          Parámetros CLASS/CAMB y simulaciones
tests/            Tests unitarios y de integración
notebooks/        Notebooks de análisis
scripts/          Scripts de reproducción
```

## Instalación

```bash
pip install -e .
# o
pip install -r requirements.txt
```

Requisitos: Python ≥ 3.10, numpy, scipy, matplotlib, astropy, emcee, qutip
(opcional para `quantum/`).

## Uso rápido

```python
from mcmc_ontology import constants as C
from mass_program.B4_masses import predict_fermion_masses
from mass_program.B5_higgs import higgs_mass

print(higgs_mass())                 # ≈ 125.4 GeV (el tratado publica 125.3 — Obs. 12.2)
print(predict_fermion_masses())     # tabla 12/12 fermiones
```

## Cronología tensional (Tabla F.1, Tratado de Fundamentos v35)

| S        | Rasgo dominante                                        |
|----------|--------------------------------------------------------|
| S₀       | Unidad dual Mp/Ep, sin espacio ni tiempo               |
| S₀,₀₀₁   | Emergencia Mp/Ep, V0D, proto-gravedad                  |
| S₀,₀₁₀   | V1D, partícula/antipartícula, energía dinámica         |
| S₀,₀₉₉   | **Sello de c** (Cota de Delivery)                      |
| S₀,₁₀₀   | V2D, giro y rotación                                   |
| S₀,₉₉₉   | **Sello de c²** (preparación volumétrica)              |
| S₁,₀₀₀   | Colapso V3D, gravedad como curvatura                   |
| S₁,₀₀₁   | V3+1D, mass gap mínimo, Ley de Cronos, ΦAd → ΦH        |

**Ley de la Década** (Prop. 8.1): con λ = 10 y ΔS = 10⁻³,

```
S_k^col = λ^(k−2) − ΔS   (k = 0,1,2 → colapsos 1D, 2D, 3D)
S_Florencia = λ⁰ + ΔS
→ 0.009, 0.099, 0.999, 1.001
```

Los cuatro umbrales se reducen a tres datos: la razón λ, el calibre ΔS y la
regla de ±1 cuanto (anticipación/confirmación). El exponente de Victoria es
s0 = π/ln(10) ≈ 1.3644 (derivado para λ = 10). En código:
`mcmc_ontology.constants.decade_thresholds()` y `S0_VICTORIA`.

Nota v35: los sellos S₀,₀₉₉ y S₀,₉₉₉ son sellos de *c* y de *c²*, no
emergencias de familias fermiónicas; la supervivencia de cada modo a través
de los sellos la codifican los pesos c_in del **Funcional del Camino**
(Def. 12.3). La asociación familia↔sello y las escalas Planck/GUT por sello
eran presentación del Tratado Unificado (v32) y se conservan solo en el
bloque `LEGACY_V32` de `constants.py`.

## Resultados del ajuste de producción (julio 2026)

Primer ajuste bayesiano de este repositorio sobre **datos reales**
(`scripts/run_production_fit.py`, semilla 42, 32 walkers × 4000 pasos,
convergencia verificada por autocorrelación): 31 puntos H(z) de cronómetros
cósmicos + 6 medidas BAO del consenso BOSS DR12 (r_d fiducial 147.09 Mpc,
fijo) + 1371 SNe de Pantheon+ con covarianza completa (M_B marginalizada
analíticamente; sin ancla cefeida). ΛCDM se ajustó sobre **los mismos datos
con la misma maquinaria** (ε = 0 exacto, Prop. A.1); priors del Apéndice F.

| Parámetro | MCMC (k=4) | ΛCDM (k=2) |
|---|---|---|
| H₀ [km/s/Mpc] | 66.1 ± 1.7 | 66.6 ± 1.3 |
| Ω_m | 0.344 ± 0.026 | 0.340 ± 0.024 |
| ε | 0.014 −0.038/+0.042 — **compatible con 0** | ≡ 0 |
| z_trans | 9.2 ± 4.7 (dominado por el prior) | — |

χ² total en el máximo: 1226.07 (MCMC) frente a 1226.03 (ΛCDM).
**ΔAIC = +4.0 y ΔBIC = +14.5 (positivo favorece a ΛCDM)**: con estos tres
catálogos y esta metodología, los dos parámetros extra del MCMC no mejoran el
ajuste y los criterios de información favorecen a ΛCDM. El resultado se
publica tal cual — el compromiso del proyecto es con el contraste, no con el
desenlace.

Informe completo, fórmulas y cadenas: `results/2026-07-31_production_fit/`.
Reproducción determinista: `python scripts/download_data.py all &&
python scripts/run_production_fit.py`.

### La corrida de la reconciliación (v2, cinco bloques — agosto 2026)

La pregunta que dejó abierta la v1 —¿reaparece la ventaja del corpus cuando
entran los likelihoods que sondean la transición?— está respondida:
`scripts/run_production_fit2.py` añade la **geometría comprimida del CMB**
(R, l_A, ω_b; Chen, Huang & Wang 2019) y el **crecimiento fσ₈** (11 puntos
RSD, ODE exacta de D(a)), n = 1422, MCMC k=6 frente a ΛCDM k=4, convergencia
50·τ verificada (24×8000 pasos, semilla 42):

| Parámetro | MCMC (k=6) | ΛCDM (k=4) |
|---|---|---|
| H₀ | 67.1 ± 1.5 | 67.6 ± 0.7 |
| Ω_m | 0.323 ± 0.016 | 0.318 ± 0.009 |
| ε | 0.015 −0.039/+0.043 — **compatible con 0** | ≡ 0 |
| σ₈ | 0.802 ± 0.029 | 0.800 ± 0.030 |

χ² total 1235.04 vs 1235.03 (idéntico bloque a bloque, incluidos CMB y fσ₈);
**ΔAIC = +4.0, ΔBIC = +14.5 a favor de ΛCDM** — el mismo veredicto que la
v1: la penalización por parsimonia decide porque los parámetros extra no
compran nada. **La ventaja del corpus no reaparece con fondo + geometría +
crecimiento.** Lo que queda fuera de esta corrida: el espectro C_ℓ completo
y la lente débil (el CMB entra comprimido y diagonal; z*/r_s por
Hu–Sugiyama, sesgo ~0.3% idéntico en ambos modelos). Informe y cadenas:
`results/2026-08-01_production_fit_v2/`.

**Relación con el corpus.** Los valores del corpus v32 (H₀ = 69.8, σ₈ = 0.805,
ΔBIC = −6.1 a favor del MCMC) permanecen documentados como referencia
histórica en `CORPUS_REFERENCE` (`mcmc_ontology/constants.py`). Tras la v2,
la reconciliación queda acotada con precisión: la ventaja que el corpus
reportaba tendría que proceder del C_ℓ completo, de la lente débil o de su
propia metodología — ya no del fondo, la geometría comprimida ni el
crecimiento, donde ambos modelos son indistinguibles y ε sale compatible
con cero. H₀ intermedio (~69–70) y σ₈ reducido no emergen en esta corrida.

Sobre m_H ≃ 125.3 GeV: es la identidad del Modelo Estándar con λ_H = β3 y no
constituye por sí sola una predicción mientras β3 no se derive sin usar m_H
como entrada (Obs. 12.2; frente abierto nº 7). Σm_ν ≈ 4.6×10⁻⁵ eV (seesaw
tensional, muy por debajo de la cota < 0.12 eV).

## La cadena deductiva ejecutable (`core/`)

Los capítulos 2–10 del tratado están implementados como código cuyo
principio de diseño es: **cada módulo implementa las definiciones de su
capítulo y sus tests verifican los teoremas**. La ley de escala
T₀ = c̄·δ₀³ se mide (exponente ajustado = 3); la Monotonía del Camino
(Teo. 4.5) y la Exclusión (Lema 4.7) se comprueban sobre trayectorias —
«la flecha del tiempo se demuestra, no se postula» tiene test; la
Rotación de Florencia produce la firma (−,+,+,+) girando un solo
generador; el Discriminante se fusiona en la espinodal y el walking
diverge al acercarse a ella; el Sello de Newton recupera GR exactamente;
y el Ciclo de Victoria itera en espiral con ν>0 o cae al Silencio con
ν<0. Donde el tratado declara condicional (signo de Lydia, ventana de
Atlas, Conjetura de los Residuos, RP espinorial), el módulo **expone el
parámetro con su condición — nunca lo resuelve en silencio**.

La suite espejo del apéndice H (`validation/appendix_h.py`, invocada al
final de `validate_all.py`) ejecuta las verificaciones de signo de
§13.5, los tres límites de recuperación y sus **controles negativos** —
el caso que debe fallar y falla — con salida en el formato del tratado:
qué queda demostrado, qué condicional, qué refutable.

### El círculo de δ₀ (ronda 5 — agosto 2026)

El empalme C¹ (H.2.4) mide δ_H ≈ 0.0581 con formas fiduciales; el
Teo. 10.6 da al retorno de Victoria un atractor δ₀*. ¿Son el mismo
número? `core/delta0_circle.py` lo calcula: **en toda la región fértil
el atractor es el Techo, δ∞ = (W_max/c̄)^{1/3}** — independiente de γR
y del δ inicial (desde 0.012 sube en 4 vueltas). El círculo se cierra
⟺ **W_max = c̄·δ_H³ ≈ 1.65×10⁻⁴**: una ecuación de consistencia que
liga el Teo. 10.6 con la ec. H.8 y **transfiere la pregunta a W_max**,
que el tratado no cuantifica (frente abierto nº 4) — el cálculo no
cierra ni rompe el círculo, y ambos desenlaces siguen abiertos. Sobre
el paisaje O(1) fértil, δ_H = O(0.05) es genérico (mediana 0.054,
5–95% en [0.037, 0.116]) y **ninguna forma O(1) cierra el empalme en
δ₀ = 0.012** — refuerzo independiente de la regla canónica que separa
δ₀ de ε_Λ. Si δ₀ = δ_H: T₀ escala ×114 y m_θ² ×52 (consecuencias a
rastrear). Reproducción: `python scripts/run_delta0_circle.py`;
desenlace en `results/2026-08-02_delta0_circle/`.

## Simulaciones (Apéndice B / frente abierto nº 5)

El ciclo Cronos-KDK del Apéndice B está implementado en malla PM mínima:
paso temporal entrópico (ec. B.3, η=0.025), Poisson modificado (ec. B.2),
refresco de canales tensionales y dilatación local (B.3), con la fricción
de compuerta del cap. 11. La demo reproducible:

```bash
python scripts/run_halo_collapse.py
```

corre un par A/B de semilla idéntica (B.5/B.6) y exhibe la **firma
falsable de la fig. 11.1**: la compuerta activa durante el colapso y
apagada al virializar (el control newtoniano da Γ≡0).

**Alcance declarado**: esto NO es Gadget-4-Cronos (la variante de
producción del tratado con árbol octal, MPI y cajas 512³–1024³ de B.4,
cuyas configuraciones están en `configs/cronos_*.yaml`). Las validaciones
del corpus (núcleo de 2.3 kpc, SPARC RMSE 12%→4.5%, subhalos −45%) siguen
**pendientes de reproducción** con el esquema corregido — es el frente
abierto nº 5 (§13.6/cap. 14.5).

## Tests

```bash
pytest tests/
```

## Cómo citar

Ver `CITATION.cff` (GitHub genera la cita automáticamente). Registro
canónico vigente: **Tratado de Fundamentos** (v35),
DOI [10.5281/zenodo.20765373](https://doi.org/10.5281/zenodo.20765373).
DOI de concepto de la serie (siempre la última versión):
[10.5281/zenodo.14167831](https://doi.org/10.5281/zenodo.14167831).
No citar como vigentes los depósitos superados (v29, v32).
ORCID de referencia: [0009-0009-4314-9642](https://orcid.org/0009-0009-4314-9642).

## Licencia

Apache License 2.0 — ver `LICENSE`.
