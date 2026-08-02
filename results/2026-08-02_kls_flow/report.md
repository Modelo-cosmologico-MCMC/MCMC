# Frente E: el flujo KLS integrado — la ley del walking, medida

Parámetros O(1) de demostración: B = 2.0, C0 = 1.0, G = 1.0 (xR = 1). Integración RK4 validada contra la cuadratura exacta de la ODE separable (ec. 8.3).

## 1. El «≃» de la ec. 8.4, cuantificado

| Ω/xR | Δσ medido | Δσ ec. 8.4 (misma ventana) | error |
|---|---|---|---|
| 0.01 | 155.6139 | 155.0799 | 3.44e-03 |
| 0.003 | 517.0795 | 516.9330 | 2.83e-04 |
| 0.001 | 1550.8475 | 1550.7990 | 3.13e-05 |

El error del prefactor x ≈ xR decae linealmente con Ω/xR; frente a la forma π/Ω del tratado queda además el factor de ventana 2·arctan(50)/π = 0.98727.

## 2. Divergencia en la espinodal

Δσ_walk ∝ |D|^(-0.4997) sobre cuatro décadas (esperado −1/2, pues Ω = √(−D)/2C0). La ley que sostiene la log-periodicidad de la Década queda medida.

## 3. El Cruce de Victoria como bifurcación dinámica

Con D(σ) hundiéndose a ritmo 1e-3: el flujo sigue adiabáticamente el vacío x+(σ) hasta el cruce (σ* = 100.0, exacto) y colapsa en σ = 115.5 — DESPUÉS del cruce, con retraso 15.54. Control negativo: sin hundimiento el flujo aparca en x+ y no colapsa (test).

## 4. El retraso del colapso (resultado del programa)

Retraso ∝ rate^(-0.330) sobre dos décadas — la ley −1/3 de la bifurcación silla-nodo con deriva (teoría de bifurcaciones dinámicas; NO está en el tratado y se declara como resultado del programa). Consecuencia falsable dentro del modelo: el disparo de cada colapso de la Década no es instantáneo en el cruce — lleva un retraso universal fijado por el ritmo del flujo de acoplos.

## Estatuto (frente 2)

condicional (frente 2, Obs. 8.7): el flujo valida las ec. 8.3-8.4 cuantitativamente; convertir el periodo del walking en λ = e^{π/s0} requiere las funciones de flujo de los acoplos, que el tratado no da — este módulo mide todo lo medible sin ellas.

Para λ = 10, s0 = π/ln(10) = 1.3644 (Def. 8.3) sigue siendo la relación DEFINIDA, no derivada: este módulo valida el mecanismo sobre el que descansa, no decide λ.
