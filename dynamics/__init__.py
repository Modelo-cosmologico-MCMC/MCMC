"""dynamics — El potencial débil y los sistemas soportados por presión.

Frente 5 (Cronos: de halo rotante a dinámica universal), secuencia de
la propuesta v36 §VII (registro de trabajo del 7-ago-2026, documento
del programa):

    5A  potencial débil — una sola Φ_eff(r) desde la Ley de Cronos
        (Def. 11.1) y la geodésica completa (Prop. 11.2)
    5B  aceleración g_eff = −∇Φ_eff
    5D  sistemas de presión — Jeans esférico,
        d(ν·σ_r²)/dr + (2β/r)·ν·σ_r² = −ν·dΦ_eff/dr
    5E  falsación cruzada — los MISMOS parámetros físicos para
        sistemas rotacionales y de presión; si solo funciona con
        retuning por sistema, el sector local pierde capacidad
        explicativa (estatuto declarado de antemano)

    weak_field.py   5A-5B: Φ_eff = Φ_N − c²·ε_c(ρ), perfil de Plummer
    jeans.py        5D: solución esférica con β constante y proyección
    dsph_data.py    la primera muestra dSph (Sculptor) con procedencia

El fallo del contraste conjunto cuenta como falsación del mecanismo
galáctico propuesto, no como llamada a ajustar por sistema.
"""
