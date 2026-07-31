# MCMC — Modelo Cosmológico de Múltiples Colapsos

> **Estado de versión (julio 2026).** Esta implementación sigue el *Tratado Unificado*
> (v32, marzo 2026). El registro canónico vigente del modelo es el **Tratado de
> Fundamentos** (v35, junio 2026, DOI [10.5281/zenodo.20765373](https://doi.org/10.5281/zenodo.20765373)),
> que refunda el MCMC sobre ocho axiomas ontológicos. La alineación del código con la
> v35 está en curso y se documenta commit a commit; hasta completarse, los valores y
> etiquetas de este repositorio pueden diferir del tratado vigente.
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
mass_program/     B0–B6, P3, P4, M1 (running QCD), M2 (CKM)
cosmology/        H(z), ρ_id, ρ_lat, Λ_rel, ajuste bayesiano
cronos/           N-body en S, fricción entrópica, halos cored
lattice/          Mass gap E_min = k·ΔS, espectro glueball SU(3)
quantum/          Qudit d=5, H_MCMC ontológico, simulación QuTiP
visualization/    Animaciones, espectro de masas, plots cosmológicos
data/             Datos observacionales de referencia
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

## Valores de referencia del corpus

**Estos valores NO son salidas de este código**: proceden de los ajustes
documentados en el corpus del modelo (v32). Los directorios de `data/` están
vacíos; reproducirlos requiere cargar los catálogos observacionales reales
en `data/` y ejecutar el ajuste bayesiano de producción.
Hasta entonces se usan como valores fiduciales de referencia
(`CORPUS_REFERENCE` en `mcmc_ontology/constants.py`).

| Observable    | Corpus MCMC         | PDG / ΛCDM       |
|---------------|---------------------|------------------|
| m_H           | ≃ 125.3 GeV ¹       | 125.25 GeV       |
| H₀            | 69.8 ± 1.1 km/s/Mpc | 67.7 ± 0.7       |
| σ₈            | 0.805               | 0.830            |
| ΔBIC vs ΛCDM  | −6.1                | —                |
| Σm_ν          | 4.6×10⁻⁵ eV         | < 0.12 eV        |

¹ m_H = √(2·β3)·v3 es la identidad del Modelo Estándar con λ_H = β3: el acuerdo
numérico no constituye por sí solo una predicción mientras β3 no se derive sin
usar m_H como entrada (Tratado de Fundamentos, Obs. 12.2; frente abierto nº 7).

## Tests

```bash
pytest tests/
```

## Licencia

Ver `LICENSE`.
