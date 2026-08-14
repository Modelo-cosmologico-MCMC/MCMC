# Historial de versiones

Formato: cada versión lista lo que el repositorio AFIRMA tras ella —
el contrato de honestidad manda («cada commit deja el repositorio en un
estado en que afirme exactamente lo que hace»).

## Sin publicar — agosto de 2026

- **Frente 5, paso 5C estructural + objetivo ρ_id (14-ago)**. La
  mitad del 5C que no necesita datos: en discos exponenciales el
  término de Cronos tiene forma fija v_cronos² ∝ x·e^(−3x/2) (pico en
  x = 2/3), así que aportar V km/s donde la discrepancia de masa vive
  (x ≳ 3) cuesta ×5·V en x = 2/3, PARA CUALQUIER amplitud A — un
  bulto interior incompatible con el carácter bariónico de los discos
  internos densos que la literatura reporta (hipótesis declarada, sin
  ingesta; la confrontación por galaxia es el pendiente). Con la A
  concreta que Sculptor exige, la malla declarada (Σ0 ∈ [50, 800]
  M⊙/pc² × R_d × ζ; los LSB reales bajan más — allí el término se
  hace despreciable y no explica nada) da v_cronos/v_bar hasta ×6.3
  en el pico y 2-82 km/s en x = 4: bultos donde el disco es denso,
  nada donde no lo es. Veredicto estructural 5E (parcial): el término
  débil no puede ser el mecanismo galáctico común; la carga queda en
  ρ_id, y Sculptor la fija como CURVA de degeneración ρ0(r_c)
  medible (M_id(<347 pc) = 1.78e7 M⊙; en el r_c = 0.30 kpc del
  corpus: ρ0 = 0.176 M⊙/pc³). Instrumento de perfiles binados listo
  con autoprueba sintética (la ingesta de Walker+09 y del catálogo
  SPARC sigue bloqueada por el proxy — pendiente declarado con el
  hueco de checksum abierto).

- **Frente 5, medio paso 2 (10-ago): Sculptor contra el potencial
  débil (5A/5B/5D)**. Una sola Φ_eff = Φ_N − c²·ε_c desde la Ley de
  Cronos (Def. 11.1 + Prop. 11.2), Jeans esférico con β constante y
  proyección validados contra la identidad analítica del Plummer
  (error < 2e-3). El resultado, publicado tal cual bajo las DOS
  lecturas declaradas de la ec. (11.5) (literal: solo α₀⁻¹, ρ_c
  libre; subdominancia: c²ε_c ≲ |Φ_N|, de la que la cota nace): solo
  estrellas dan σ_N = 2.1-3.6 km/s; bajo subdominancia el término
  débil de Cronos queda falsado como explicación de Sculptor
  (techo √2·σ_N ≤ 5.1 frente a σ_obs ≈ 9-10); bajo la literal no hay
  violación numérica, pero el término solo alcanza σ_obs como
  potencial DOMINANTE (c²ε_c = ×8-×26·|Φ_N|), el régimen que la
  justificación de la cota excluye. Sculptor no mide ρ_c: mide
  A = α₀⁻¹/ρ_c^(3/2) (con α₀⁻¹ en su cota, ρ_c ≲ 1-2 M⊙/pc³,
  desigualdad unilateral — 5E sigue abierto). La carga explicativa
  queda cuantificada en ρ_id (M_1/2 ≈ 2e7 M⊙ dentro de r_1/2 ≈ 347
  pc), sin perfil derivado a escala dSph — hueco declarado. Datos
  globales de Walker et al. 2009 con procedencia y pendientes
  declarados (`dynamics/dsph_data.py`).

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
