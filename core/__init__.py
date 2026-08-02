"""core — La cadena deductiva ejecutable (Tratado de Fundamentos v35, caps. 2-10).

Principio de diseño: cada módulo implementa las definiciones de su
capítulo y sus tests verifican los teoremas. Donde el tratado demuestra,
el test comprueba numéricamente; donde el tratado declara condicional,
el módulo expone el parámetro y lo dice — nunca lo resuelve en silencio.

    dual_plane.py             Cap. 2  — el escenario (ρ, θ, χ, ς)
    basal.py                  Cap. 3  — el paisaje (V0, T0 = c̄·δ0³)
    path_flow.py              Cap. 4  — la dinámica (Monotonía, Exclusión)
    florencia.py              Cap. 5-6 — la Cadena de Álgebras y el giro
    reflection_positivity.py  Cap. 7  — la condición del presente (juguete)
    decade.py                 Cap. 8  — Discriminante y flujo log-periódico
    gea.py                    Cap. 9  — Gea / Sello de Newton / Atlas
    victoria.py               Cap. 10 — el cierre del ciclo (Lydia, suelo)

La suite espejo del apéndice H vive en validation/appendix_h.py.
"""
