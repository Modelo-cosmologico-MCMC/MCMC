"""core — La cadena deductiva ejecutable (Tratado de Fundamentos v35, caps. 2-10).

Principio de diseño: cada módulo implementa las definiciones de su
capítulo y sus tests verifican identidades, límites y consecuencias
numéricas de los teoremas y ansätze implementados — COMPROBACIÓN
INTERNA de la implementación, no demostración física (v35.1, E8).
Donde el tratado declara condicional, el módulo expone el parámetro y
lo dice — nunca lo resuelve en silencio.

    dual_plane.py             Cap. 2  — el escenario (ρ, θ, χ, ς)
    basal.py                  Cap. 3  — el paisaje (V0, T0 = c̄·δ0³)
    path_flow.py              Cap. 4  — la dinámica (Monotonía, Exclusión)
    florencia.py              Cap. 5-6 — la Cadena de Álgebras y el giro
    reflection_positivity.py  Cap. 7  — la condición del presente (juguete)
    rp_nonstationary.py       Cap. 7  — RP con acoplos corriendo (frente F)
    decade.py                 Cap. 8  — Discriminante y flujo log-periódico
    kls_flow.py               Cap. 8  — el flujo KLS integrado (frente E)
    victoria_exponent.py      §14.2   — matriz de estabilidad (frente 2)
    gea.py                    Cap. 9  — Gea / Sello de Newton / Atlas
    victoria.py               Cap. 10 — el cierre del ciclo (Lydia, suelo)
    delta0_circle.py          Cap. 10 + H.8 — el círculo de δ₀ (ronda 5)

La suite espejo del apéndice H vive en validation/appendix_h.py.
"""
