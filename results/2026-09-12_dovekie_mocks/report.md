# PR #15 — validación por mocks del pipeline SN Dovekie: **PASS**

Preinscripción: `d7cbabdc6d9f` (congelada ANTES de ejecutar puerta alguna); ejecución: commit `196b40a6d`, 699.7 s.

- **1a** fórmula χ²: max|Δχ²| = 1.46e-09 (≤ 1e-6), max|ΔW| = 0e+00, conteo 1820 idéntico en parser A, parser B y ambos npz → PASS
- **1b** distancias: max|Δμ| = 1.91e-06 mag (≤ 1e-5), |Δχ̃²| = 1.31e-03 (≤ 0.01) → PASS
- **2** recovery: media de pulls = +0.051 (cota 0.6 = 3·SEM); sd = 1.070 (diagnóstico, sin puerta) → PASS
- **3** cobertura: k68 = 17/25 ∈ [12, 21], k95 = 23/25 ∈ [21, 25] → PASS; no se detecta una descalibración incompatible con el tamaño de la muestra de mocks
- **4** falsa preferencia: mediana ΔAIC = +3.94, mediana ΔBIC = +14.95, n(ΔAIC<0) = 0 ≤ 7, 0 ∈ CI95(ε) en 25/25 (≥ 21), mediana p50(ε) = +0.0149 (|·| ≤ 0.05) → PASS

Roles declarados: mocks = validación del pipeline · Dovekie real = primera aplicación (solo tras PASS) · Unite = benchmark armonizado de robustez, NO replicación independiente · Union3 = comprobación externa · Pantheon+ = disección de Unite.

La tabla íntegra por mock está en per_mock.csv; el veredicto ejecutable (la llave de la barrera) en mock_validation.json.
