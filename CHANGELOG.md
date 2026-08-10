# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## Sin publicar — agosto de 2026

- **Corrección de normalización del fondo (10-ago)**: la forma legacy
  de Λ_rel anclaba Λ0 en el punto medio de la transición (sin dividir
  por F(0)) y congelaba Ω_Λ0 con Ω_m = 0.300 a nivel de módulo —
  H(0) ≈ 1.0042·H0 en el punto fiducial (validate_all imprimía 70.09
  con H0 = 69.8) y clausura no plana al variar Ω_m; en las corridas
  legacy el sesgo de F(0) era SOLO del brazo MCMC (ε hacía doble
  papel). Hoy: transición normalizada hoy (Ω_Λ_rel = Ω_Λ0·F(z)/F(0)) y
  clausura plana por llamada ⟹ H(0) = H0 exacto para todo parámetro
  admisible (dz > 0 declarado).
- **Nivel 3 del CI — invariantes físicos**: suite
  `tests/test_physical_invariants.py` sobre mallas de (Ω_m, ε, z_trans,
  dz) que cubren los priors de los ajustes (ε negativos incluidos), y
  `validate_all` ahora FALLA si H(0) ≠ H0 (el canal que estuvo verde
  durante el bug).
- **Repetición de los ajustes v1 y v2 con el fondo corregido**
  (mismas semillas, datos y configuración): el veredicto diferencial
  se confirma — ΔAIC = +4.00/+4.08, ΔBIC = +14.50/+14.60 pro-ΛCDM,
  ε compatible con 0 — con posteriores absolutos ya sin sesgo
  (`results/2026-08-10_production_fit{,_v2}/`); las corridas
  originales quedan etiquetadas `legacy_pre_normalization` con la
  asimetría del sesgo declarada.

## v0.2.0 — 2 de agosto de 2026 (rondas 2–5)

- **Cadena deductiva ejecutable** (`core/`, caps. 2–10): cada módulo
  implementa las definiciones de su capítulo y sus tests verifican
  identidades, límites y consecuencias numéricas de los teoremas y
  ansätze implementados (comprobación interna, no demostración física —
  v35.1, E8); condicionales expuestos, nunca resueltos en silencio.
  Suite espejo del apéndice H con controles negativos (`validation/`).
- **Ajustes de producción v1 y v2** sobre datos públicos con checksums:
  ΔBIC = +14.5 a favor de ΛCDM en ambos, publicado en portada;
  ε = 0.015 −0.039/+0.043 (compatible con 0); la ventaja del corpus no
  reaparece (la reconciliación respondida).
- **Frente 6 (Residuos), consistencia superada**: −1.8 % dentro de la
  cota BBN (Alvey et al. 2020), a 0.32σ — sin significancia de signo
  (v35.1, E13); refutable (vivo).
- **Frente 7 ejecutado**: empalme C¹ no circular (δ_H ≈ 0.0581 medido
  sin m_H como entrada) + WKB ab initio (una sola κ cubre 13 órdenes).
- **El círculo de δ₀ (ronda 5)**: el atractor de Victoria es el Techo;
  ecuación de cierre W_max = c̄·δ_H³ ≈ 1.65×10⁻⁴ — los ingredientes
  están en el tratado (Lema 10.3 + H.8); la aportación del programa es
  la conexión δ_sat = δ_H y que γR desaparece del punto fijo;
  naturalidad de δ_H (mediana 0.054) y cota δ_H ≳ 0.033.
- **Regla canónica**: retirada la identificación δ₀ ≡ ε_Λ de todo el
  código (los cuatro épsilon del convenio C6 — v35.1, E9 — separados).
- **Medio paso del frente 5**: compuerta H.2.5 con mediana Γ = 0 exacta
  en el halo virializado aislado y control negativo; el veredicto del
  núcleo queda declarado como inaccesible a esta resolución.
- **Cronos v3** (cap. 11), lattice/quantum alineados con apéndices C-D,
  simulación PM mínima del apéndice B con par A/B reproducible.
- Notebook en inglés con salidas ejecutadas; orientación en inglés en el
  README; nota computacional I en `docs/`.
- **Frente E (flujo KLS)**: la ley del walking medida (el «≃» de la
  ec. 8.4 cuantificado; divergencia −1/2), el Cruce de Victoria como
  bifurcación dinámica y el retraso ∝ ritmo^(−1/3) (resultado del
  programa); λ = 10 sigue calibrado (frente 2).
- **Frente 2 instrumentado** (ec. 14.2): dos rutas coincidentes al
  exponente de Victoria (espectro ⟺ walking integrado, error ~1e-14);
  con ansatz O(1), la cascada DSI es genérica (~67 %) pero λ = 10 es
  una selección medible — las β de Fokker-Planck (Def. 4.4) son el
  hueco declarado.
- **Frente F (RP no estacionaria)**: la simetría especular del perfil
  es condición SUFICIENTE para la RP en el juguete (necesidad no
  demostrada) — el running monótono viola la reflexión ingenua y la
  reflexión modificada restaura la positividad exacta; el sector de
  Wilson sigue abierto (frente 1).
- **Matriz de trazabilidad ejecutable** (manuscrito ↔ código ↔ test ↔
  estatuto, con huecos declarados) verificada por test; guardias de
  lenguaje v35.1 en el CI (E4/E6/E8/E13); Lema de Precedencia
  reformulado K(S)=K(ϑS) como condición ejecutable y correspondencias
  E.18/E.19 propuestas para la v36.
- CI con guardias de honestidad y contrato de lint declarado (reglas en
  `pyproject.toml`, versión de ruff fijada).
- 189 tests en verde.

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
