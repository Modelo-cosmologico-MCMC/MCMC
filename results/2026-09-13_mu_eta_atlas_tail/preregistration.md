# Preinscripción E3_Atlas — E3_Atlas — cola O(e²) del canal Atlas con el sector de velocidades: escalera exacta del modo creciente y arnés numérico con condiciones iniciales adiabáticas

Congelada 2026-09-13T21:00:11.454264+00:00 en el commit `e54e6e106` (contiene el generador). ANTES de ejecutar la escalera de producción y el arnés adiabático.

## E3a — escalera exacta (ξ = 1)

Malla (λ_K, α_a): [[1.05, 0.3], [1.0125, 0.3], [1.1, 0.3], [1.05, 0.1], [1.05, 0.6], [1.012, 0.012]]. Orden dominante: µ₀ = 1/(1−α_a/2) y η₀ = 1 exactos. Residuos del polo en α_a = [0.1, 0.3, 0.6] con pasos h = [0.005, 0.01, 0.02, 0.04] (ajuste cúbico): |P/P_cerrado − 1| ≤ 0.0001 para P_η = 3α/(2−α) y P_µ = −P_η·p(2p−1)/3.

## E3b — arnés con ICs adiabáticas

Punto {'lambda_K': 1.05, 'alpha_a': 0.3, 'xi': 1.0}, k/H0 = [66.667, 133.333, 200.0], a_start = 0.1, ajuste en e ≤ 0.02. Reglas: pendientes de η−1 y µ_loc−1 frente a e² dentro del 10% de la escalera en cada k; exponente log-log ∈ [1.9, 2.1]; pureza de modo ≤ 0.01. Predicción independiente del autor (adenda C.2): 17.07·e² en η.

## Desenlaces

E3a: A (residuos derivados) / B (no fijados). E3b: A (cola confirmada) / B (gana la QS truncada: error en la escalera) / C (abierto). Nunca se ajustan los umbrales; todo se publica.

Control no vinculante: ICs QS-consistentes de #18 (suelo de ruido).

Sin datos; ξ = 1; fronteras declaradas en el JSON.
