# Preinscripción — El vacío 2D fuera del polo de masa: corriente de conversión κΣ̇ê_E y rotación de la inclinación η_eff = η₀(1 − f/f_×) como brazos, J∇C de control

Congelada 2026-09-22T00:29:12.248808+00:00 en el commit `fa8db8dc0` (contiene el generador), ANTES de ninguna corrida de producción. sha256 de `preregistration.json`: `3990f7bcaee7203d958fc16ee4eecb643b1f757cd50d389b0e85d023e51adf37`.

**Pregunta**: ¿Sitúa alguno de los dos cierres compatibles con el tratado el cruce de la diagonal θ = π/4 en S = 1 ± 0.05 con θ monótona, para un intervalo de su único número declarado (κ̂ o f_×) de anchura ≥ 20 %?

**Forma declarada**: El vacío verdadero 2D del Basal está sobre el polo de masa (θ = 0) y ningún término que se anule en el vacío (J∇C) lo mueve (§3.25). Dos cierres DECLARADOS como brazos: (ii) corriente de conversión dΦ/dσ = −G⁻¹∇V + κΣ̇ê_E, κ = κ̂·ρ₊/T₀ (la conversión Mp → Ep es creación de espacio, Prop. 3.5: deriva a lo largo de ê_E proporcional a la producción entrópica; θ monótona porque Σ̇ ≥ 0, Teo. 4.5; se apaga cuando la descarga termina, no en el vacío; identidad S = f − f₀ + (W_J + W_conv)/T₀ con W_conv = ∫∇V·κΣ̇ê_E dσ); (i) rotación de la inclinación η_eff = η₀(1 − f/f_×) (en f_× el potencial es O(2)-simétrico, Prop. 10.2; después el polo de espacio es el mínimo), con W_tilt = ∫(∇V_ref − ∇V_eff)·dΦ publicado. κ̂ o f_× son CALIBRACIONES con significado (el cruce de la diagonal en S = 1); derivarlos es el frente 2.

## Declarado

- **delta0_grid**: {'0.003': 0.003, '0.01': 0.01, '0.03': 0.03, 'delta_H_full': 0.055444201949872325}
- **kappa_hat_grid**: [0.05, 0.06418, 0.08238, 0.10574, 0.13572, 0.17421, 0.22361, 0.28701, 0.3684, 0.47287, 0.60696, 0.77908, 1.0, 1.28357, 1.64755, 2.11474, 2.71442, 3.48414, 4.47214, 5.74029, 7.36806, 9.45742, 12.13924, 15.58156, 20.0]
- **f_cross_grid**: [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.9, 0.93, 0.96, 0.99]
- **J_push_arm_i**: 0.1
- **circulation_C**: V
- **control_J_min_from_3_25**: {'0.003': 1.363, '0.01': 1.37, '0.03': 1.389}
- **G**: 1.0
- **theta_nuc**: 0.0
- **kappa_definition**: κ = κ̂·ρ₊/T₀ (κ̂ adimensional); espacio acumulado ∫κΣ̇dσ = κ̂·ρ₊·f
- **stop_rule_arm_i**: descenso terminado cuando f ≥ 1 − ε_res o |∇V_eff| < 1e-6·T₀/ρ₊
- **identity**: S = f − f₀ + (W_J + W_conv + W_tilt)/T₀ con W_tilt = ∫(∇V_ref − ∇V_eff)·dΦ

## Reglas (congeladas)

- **S_window**: [0.95, 1.05]
- **width_ratio_A**: 1.2
- **tol_theta_drop_rad**: 0.01
- **tol_identity**: 1e-06
- **monotonia_required_arm_ii**: True
- **monotonia_required_arm_i**: False
- **letters**: {'A': 'cruce en la ventana para un intervalo de κ̂ (o f_×) con máx/mín ≥ width_ratio_A, θ monótona (caída ≤ tol_theta_drop) y, en (ii), Monotonía 4.5', 'B': 'intervalo no vacío pero máx/mín < width_ratio_A (ajuste fino)', 'C': 'ninguna corrida cruza en la ventana con θ monótona (y Monotonía en (ii))', 'INDETERMINADO': 'identidad > tol_identity en alguna corrida, descenso no terminado, o corridas ausentes'}
- **global**: la mejor letra de los dos brazos (A > B > C); INDETERMINADO si cualquiera de los dos lo es

## Expectativas (E13)

- **kappa_hat**: ≈ 1 (κ ≈ ρ₊/T₀ en orden de magnitud, §5 del texto del autor); geométricamente ≈ 0.7 para que κ̂ρ₊f ≈ ρ₊/√2
- **arm_ii**: A (θ monótona por construcción, según el texto del autor); el piloto vio que θ vuelve al polo
- **arm_i**: B o C (el cruce depende de un empujón externo y de f_×)

## Piloto declarado

δ₀ = 0.01: κ̂ ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3} → cruce solo para κ̂ ≥ 2 (S_cross 0.81, 0.66), θ sube y vuelve al polo (caída ≈ θ_max), Monotonía conservada, identidad ≤ 3e-10; f_× ∈ {0.5, 0.8, 0.9, 0.99} con |J| = 0.1 → cruce en S = 0.880 para 0.5 y 0.8 (θ → π/2 y 1.16), sin cruce para 0.9 y 0.99, identidad ≤ 5.6e-7, descenso sin terminar en f (⟹ regla de parada ∇V_eff ≈ 0).

solo el instrumento: mallas, empujón, regla de parada, tolerancia de identidad; ninguna letra ni ventana

## Lo que no puede decidir

- κ̂ o f_× son calibraciones con significado, no derivaciones (frente 2)
- cuál de los dos cierres es el del tratado: la letra dice cuál sitúa el cruce, no cuál es verdadero
- nada observable: recorrido interno (E8)
