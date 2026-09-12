# Preinscripción PR #15 — validación por mocks del pipeline SN Dovekie

- **Commit**: `6b2abf17cdb8ca6b71738a8775ac4f58c55dc324`
- **Fecha (UTC)**: 2026-09-12T21:08:29.744680+00:00
- **N SNe (verificado por bytes)**: 1820 — la cifra 1828 del traspaso era errónea
- **Mocks**: 25 realizaciones a nivel HD, semilla 20260912, covarianza STAT+SYS, cosmología inyectada Ω_m = 0.315, ε = 0 (la de los mocks DES @ c9a4fcaf)

**Puertas** (tolerancias y enteros CONGELADOS aquí, antes de ejecutar nada):
1a. |Δχ²| ≤ 1e-6 entre A, B y la fórmula oficial ejecutada desde los bytes del release; conteo 1820 idéntico.
1b. |Δμ| ≤ 1e-5 mag (integrador producción vs quad independiente); |Δχ̃²| ≤ 0.01.
2. |media de pulls de Ω_m| ≤ 0.6 (3·SEM, SEM = 1/√25).
3. cobertura: k₆₈ ∈ [12, 21], k₉₅ ∈ [21, 25] (binomial exacta, N = 25; σ_p̂ = 0.093).
4. mediana(ΔAIC) > 0, mediana(ΔBIC) > 0, n(Δ<0) ≤ 7; 0 ∈ CI95(ε) en ≥ 21 de 25 y |mediana p50(ε)| ≤ 0.05.

**Redacción obligatoria si la puerta 3 pasa**: «no se detecta una descalibración incompatible con el tamaño de la muestra de mocks» — nunca «coverage validado al X %».

**Alcance**: la validación cubre la etapa cosmológica (HD + covarianza), el punto de entrada de nuestro pipeline; los mocks fotométricos DES validan las etapas DES aguas arriba y quedan fuera (registrados, no ingeridos).

**Barrera**: el HD real (columna MU) no entra en ningún script de inferencia sin mock_validation.json en PASS citando el sha256 de ESTE fichero (candado ejecutable + candado de protocolo en la suite).
