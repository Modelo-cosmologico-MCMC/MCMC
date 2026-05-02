# MCMC — Modelo Cosmológico de Múltiples Colapsos

Implementación de referencia en Python del **Modelo Cosmológico de Múltiples
Colapsos (MCMC)**, basado en el *Tratado Unificado* (2025-2026) de
Adrián Martínez Estellés. El repositorio reproduce la ontología completa del
modelo: dualidad primordial Mp/Ep, sellos ontológicos, Campo de Adrián,
WKB espinorial, programa de masas, cosmología con Λ_rel(z), Cronos N-body,
mass gap lattice, simulación cuántica de qudits y visualización.

## Estructura

```
mcmc_ontology/    Ontología pura: constantes, S-map, potencial, Clifford, sellos
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

print(higgs_mass())                 # 125.44 GeV
print(predict_fermion_masses())     # tabla 12/12 fermiones
```

## Sellos ontológicos

| Sello | S      | Evento                              | Gauge                        |
|-------|--------|-------------------------------------|------------------------------|
| C₀    | 0.001  | Colapso puntual, V₀D                | —                            |
| C₁    | 0.009  | Sello V₀D, emerge F₁ (τ,t,b,ν_τ)    | —                            |
| C₂    | 0.099  | Sello V₁D, emerge F₂ (μ,c,s,ν_μ)    | SO(10)→GPS                   |
| C₃    | 0.999  | Sello V₂D, emerge F₃ (e,u,d,ν_e)    | GSM                          |
| C₄    | 1.001  | Big Bang, nace el tiempo, Higgs     | SM completo                  |

## Resultados principales

| Observable    | MCMC                | PDG / ΛCDM       |
|---------------|---------------------|------------------|
| m_H           | 125.44 GeV          | 125.25 GeV       |
| H₀            | 69.8 ± 1.1 km/s/Mpc | 67.7 ± 0.7       |
| σ₈            | 0.805               | 0.830            |
| ΔBIC vs ΛCDM  | −6.1                | —                |
| Σm_ν          | 4.6×10⁻⁵ eV         | < 0.12 eV        |

## Tests

```bash
pytest tests/
```

## Licencia

Ver `LICENSE`.
