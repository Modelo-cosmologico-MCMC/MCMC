# Preinscripción — Γ₀(δ₀) condicional a n_dim ∈ {1, 3, 4}: túnel de Gamow (n = 1), bounce O(n) (n = 3, 4), deformación de camino en (ρ, χ) como segunda pasada

Congelada 2026-09-21T12:47:37.060092+00:00 en el commit `62a030ea1` (que contiene el generador). sha256 de `preregistration.json`: `46f486e1cc84d350bd6d1f4a8128176ec831690ca55c62b02db52fb1c8d36a5c`.

## Declarado

- n_dim ∈ [1, 3, 4]; G = ħ = 1; corte θ = 0; formas m̄ = 1.0, b̄ = 3.0, C0 = 1.0; ē ∈ [0.5, 1.0, 2.0].
- Mallas δ₀ (8 puntos log entre 1e-4 y 0.9·δ₀_max(ē) = 0.9·0.1028·ē⁻²): ē = 0.5: [1.0e-04, 3.700e-01]; ē = 1.0: [1.0e-04, 9.250e-02]; ē = 2.0: [1.0e-04, 2.313e-02].
- Prefactor n = 1: Gamow: ω_fv/(2π), ω_fv = √(V''(φ_fv)/G) = m̄·δ₀ (frecuencia de intento). n = 3, 4: A dimensional, NO fijado: se publica Γ₀/A = e^{−B}.
- Deformación: θ(s) = θ_end·s + Σ a_k sin(kπs) recortado a [0, π/2]; ρ(s) lineal ρ_fv → ρ_esc(θ_end); B = 2∫√(2G·max(V − V_fv, 0))|dΦ/ds|ds; Nelder–Mead desde el rayo θ = 0.

## Predicciones (argumento de escala)

- n = 1: B₁ = b₁·δ₀²·(1 + O(√δ₀)); Γ₀ = (m̄δ₀/2π)·e^{−B₁} → lineal en δ₀; b₁(sin inclinación) = 0.5178.
- n = 3: B₃ → constante (argumento 3 − d = 0): Γ₀(0) = 0 solo si A(δ₀) → 0.
- n = 4: B₄ ∝ δ₀^{−1} (argumento 3 − d = −1): Γ₀(0) = 0 sin prefactor.
- Deformación: cociente B_min/B_ray ≃ 1: la inclinación −η·χ es máxima sobre θ = 0 dentro del dominio φ ≥ 0, el rayo recto es el candidato natural.
- la inclinación corrige las leyes puras en O(√δ₀) relativo, AMPLIFICADO por T₀/altura de barrera (la barrera vale 1–8 % de T₀): por eso los exponentes se miden en la década BAJA del barrido.

## Reglas (congeladas)

- Ventana de ajuste: década baja: los TRES puntos más bajos de cada malla (δ₀ ∈ [1e-4, ~1e-3]). mínimos cuadrados en log–log; residuo relativo máximo del ajuste = tol_fit.
- **A**: A si, para TODO ē, |p_Γ₀ − 1| ≤ tol_A y |B₁/δ₀²(δ₀ mín) − b₁| ≤ 0.10·b₁ (tol_A = 0.05).
- **B**: B si, para TODO ē, |p_B4 + 1| ≤ tol_B (tol_B = 0.25).
- **C**: C si en algún (δ₀, ē) B_min/B_ray < 0.5 con el optimizador convergido.
- Puertas: bounce convergido en todos los puntos; cuadratura convergida; residuo del ajuste ≤ 0.1; optimizador convergido.
- Veredicto: tupla (A: sí/no, B: sí/no, C: sí/no) publicada entera; INDETERMINADO si alguna puerta falla; la fila nucleacion-gamma0 queda condicional a n_dim (decisión B del autor); ningún desenlace se convierte en afirmación sobre el tratado (E8, E13).

## Pilotos declarados

tres comprobaciones de instrumento ANTES de congelar, con n_grid = 4001 y trapecio: B₁/δ₀² = 0.4966 en δ₀ = 1e-4 (ē = 1), bounce n = 4 en δ₀ = 1e-4 convergido en 43 disparos (~2 s), deformación en δ₀ ∈ {0.01, 0.05} con cociente 1.000. Fijaron la cuadratura (quad adaptativa por la singularidad √ en los extremos) y la ventana de ajuste; NINGUNA tolerancia se ajustó a ellos (tol_A = 0.05 y tol_B = 0.25 se declaran a priori por el tamaño esperado de la corrección O(√δ₀)).

## Lo que no puede decidir

- n_dim (decisión B del autor)
- el prefactor A de n ∈ {3, 4}
- cuál nucleación es la del tratado (instantón conservativo, túnel 0+1 o escape de Kramers)
- el diccionario t ↔ σ
