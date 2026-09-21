# Preinscripción — La diagonal y el |J| requerido: circulación J∇C del Camino de dos niveles en el reloj S

Congelada 2026-09-21T13:24:39.085657+00:00 en el commit `cd27d8997` (que contiene el generador). sha256 de `preregistration.json`: `4f56704d076cbfeec124369a4e363f0c651622658c3c01fb3c6b76b401184f48`.

## Declarado

- Camino de dos niveles (propuesta v36): dΦ/dσ = −G⁻¹∇V + J∇C con J = |J|·ε antisimétrica (ε = [[0, −1], [1, 0]]) y |J| DECLARADO; C = V por defecto (J∇V ⊥ ∇V: la Monotonía 4.5 y la identidad S ≡ f se conservan EXACTAMENTE; Σ̇ cuenta solo el nivel disipativo) o C = ρ²/2 (rotación rígida; trabaja contra la inclinación: W_J = ∫∇V·J∇C dσ ≠ 0, publicado). Orientación de J fijada para que θ crezca hacia la diagonal durante el descenso. El reloj publica el |J| que sitúa el cruce de la diagonal en S_target (calibración con significado, no derivación: J desde la ontología es el frente 2).
- C principal: V; alternativa: rho2 (C = δ₀²ρ²/2; |J| adimensional en σ̂). Orientación: θ crece hacia la diagonal durante el descenso.
- Malla δ₀: [0.001, 0.003, 0.01, 0.03, 0.05813776741499453]; ventana [0.95, 1.05]; fin del descenso S = 0.999; intervalo de |J| [0.05, 50.0]; bisección al 0.002.

## Reglas (congeladas)

- **F_per_delta0**: con |J|_min (C = V) el flujo vuelve a la frontera φ_E = 0 (θ_final ≤ 1e-9) — se evalúa ANTES que A/B
- **A_per_delta0**: no F y S_max_cross(δ₀; C = V) ≥ 0.95
- **B_per_delta0**: no F y S_max_cross(δ₀; C = V) < 0.95
- **global**: A/B/F si todo δ₀ comparte la letra; MIXTO si cambia (publicar el mayor δ₀ con A y el menor con F)
- **C_rigid**: el umbral rígido (rho2) tiene S_max ∈ ventana Y (W_J/T₀ > 1e-6 o |S − f − W_J/T₀|_max > 1e-6 o Monotonía rota): la alternativa rígida sitúa el cruce pero con coste publicado
- **scale_invariance**: {'quantity': 'max(J_min)/min(J_min) − 1 sobre la malla (C = V)', 'tol': 0.1}
- **E13_estimate**: cociente |J|_min / [(π/4)/∫|∇V|dσ] publicado sin veredicto
- **gates**: {'bracket_ok_all': True, 'identity_interior_C_V': 1e-06, 'monotonia_C_V': True}
- **verdict**: letra global + tabla por δ₀ + coste de C rígida + invariancia; INDETERMINADO si falla una puerta; |J| es calibración declarada, no derivación (frente 2, E8)

## Piloto declarado

δ₀ ∈ {0.003, 0.01, 0.03} con C = V antes de congelar: |J|_min ≈ 1.362 / 1.369 / 1.389, S del cruce en el umbral 0.930 / 0.879 / 0.806, θ vuelve hacia 0 (vacío verdadero 2D en el polo de masa); en 0.03 llega a la frontera φ_E = 0 con W_J/T₀ = 5.5 y el descenso no completa (letra F). Con C = V y |J| = 2.5 (0.01) llega a la frontera φ_M = 0. C rígida con |J| ∈ {0.5, 1, 2} (0.01): cruce en 0.88 / 0.88 / 0.32, W_J/T₀ = 6 / 12 / 24, |S − f| ≈ 1.3. La estructura A/B/F/MIXTO se escribió conociendo esto; lo no medido y preinscrito: 0.001, δ_H, la invariancia, los δ₀ de cambio y el umbral rígido.

## Lo que no puede decidir

- J desde la ontología (frente 2)
- si el tratado quiere la diagonal como estado final (requiere que el vacío 2D no esté en el polo de masa: otro término, no |J|)
