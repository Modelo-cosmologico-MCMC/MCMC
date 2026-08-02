# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## v0.2.0 — 2 de agosto de 2026 (rondas 2–5)

- **Cadena deductiva ejecutable** (`core/`, caps. 2–10): cada módulo
  implementa las definiciones de su capítulo y sus tests verifican los
  teoremas; condicionales expuestos, nunca resueltos en silencio. Suite
  espejo del apéndice H con controles negativos (`validation/`).
- **Ajustes de producción v1 y v2** sobre datos públicos con checksums:
  ΔBIC = +14.5 a favor de ΛCDM en ambos, publicado en portada;
  ε = 0.015 −0.039/+0.043 (compatible con 0); la ventaja del corpus no
  reaparece (la reconciliación respondida).
- **Frente 6 (Residuos) ganado**: −1.8 % dentro de la cota BBN
  (Alvey et al. 2020), a 0.32σ, signo favorecido.
- **Frente 7 ejecutado**: empalme C¹ no circular (δ_H ≈ 0.0581 medido
  sin m_H como entrada) + WKB ab initio (una sola κ cubre 13 órdenes).
- **El círculo de δ₀ (ronda 5)**: el atractor de Victoria es el Techo;
  ecuación de cierre W_max = c̄·δ_H³ ≈ 1.65×10⁻⁴ (original del
  programa); naturalidad de δ_H (mediana 0.054) y cota δ_H ≳ 0.033.
- **Regla canónica**: retirada la identificación δ₀ ≡ ε_Λ de todo el
  código (los tres épsilon de F.3/F.4 separados).
- **Medio paso del frente 5**: compuerta H.2.5 con mediana Γ = 0 exacta
  en el halo virializado aislado y control negativo; el veredicto del
  núcleo queda declarado como inaccesible a esta resolución.
- **Cronos v3** (cap. 11), lattice/quantum alineados con apéndices C-D,
  simulación PM mínima del apéndice B con par A/B reproducible.
- Notebook en inglés con salidas ejecutadas; orientación en inglés en el
  README; nota computacional I en `docs/`.
- CI con guardias de honestidad y contrato de lint declarado (reglas en
  `pyproject.toml`, versión de ruff fijada).
- 173 tests en verde.

## v0.1.0 — 30 de julio de 2026 (migración v32 → v35)

- Repositorio alineado con el *Tratado de Fundamentos* (v35, DOI
  10.5281/zenodo.20765373): constantes por estatuto epistémico
  (apéndice F), ocho axiomas ejecutables (§1.2), Tabla F.1, transición
  tanh (A.3), Cronos v3 (cap. 11), citas v32 etiquetadas.
- Retiradas las sobreafirmaciones: eliminada la fórmula que negaba la
  existencia de parámetros libres (hoy es guardia de CI),
  auditoría de circularidad del Higgs (Obs. 12.2) recogida, valores del
  corpus separados de las salidas del código, suite declarada como
  regresión interna, test del límite de recuperación (Prop. A.1).
- Descarga de catálogos públicos con SHA-256; likelihoods SNe+BAO+CC;
  CITATION.cff.
