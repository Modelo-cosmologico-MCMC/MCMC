"""Programa de masas del MCMC.

Bloques:
  B0 — Imperfección primordial δ₀ ≡ ε.
  B1 — VEVs por sello, pesos ϑ_n.
  B2 — Modos espinoriales / N_gen = 3.
  B3 — WKB espinorial → |T_n^(i)|.
  B4 — Fórmula maestra de masas fermiónicas.
  B5 — Higgs y mass gap.
  B6 — Valores cosmológicos de referencia del corpus.
  P3 — Curvatura V''_total y Δm_eff.
  P4 — Cascada SO(10), Yukawa GUT.
  M1 — Running QCD entrópico (corrección K_QCD).
  M2 — Mezcla CKM (factores de supresión).

ESTATUTO (Tratado de Fundamentos v35, 12.5):

- Estructural (consecuencia de la ontología, no ajustable): Ngen = 3
  (Prop. 12.4); el seesaw tensional (Prop. 12.5); la forma del Funcional
  del Camino, m_i = Σ_n c_in · ϑ_n · v_n (Def. 12.3).
- Pendiente o potencialmente circular (a resolver antes de reclamar
  predicción): el cálculo WKB explícito de los pesos c_in; la derivación
  independiente de β3 (Obs. 12.2).

Cierre literal del tratado (12.5): «Si el cálculo WKB se completa con
valores de c_in derivados de primeros principios, el Funcional del Camino
constituiría la primera derivación de las constantes de Yukawa desde una
cosmología; hasta entonces, el espectro de la tabla 12.1 es un ajuste
exitoso cuyo carácter predictivo está por establecer.»
"""
