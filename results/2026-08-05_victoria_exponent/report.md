# Frente 2 instrumentado: la matriz de estabilidad de la ec. 14.2

Semilla 20260805, n = 20000 matrices por escala; ansatz DECLARADO: entradas i.i.d. U(−a, a) sobre λi = (M0², B, C0) — las β reales (Fokker-Planck, Def. 4.4) son el hueco del frente y este módulo queda listo para consumirlas.

## 1. La maquinaria, validada

Las dos rutas al exponente de Victoria coinciden sobre la matriz de prueba con modo real acoplado: s0 espectral = 1.364376 vs s0 dinámico (periodo del walking del flujo integrado, unwrap del ángulo) = 1.364376 — error relativo 1.53e-14; λ = e^(π/s0) = 10.0000.

## 2-3. El barrido del ansatz O(1)

| escala a | fracción DSI | s0 mediano | banda ±10% de π/ln10 |
|---|---|---|---|
| 1.0 | 0.674 | 0.520 | 0.0037 |
| 1.5 | 0.674 | 0.780 | 0.1134 |
| 2.0 | 0.674 | 1.040 | 0.1567 |

- **La cascada DSI es GENÉRICA y robusta**: ~2/3 de las matrices O(1) complejifican sus exponentes, con independencia de la escala del ansatz (la complejidad del espectro es invariante de escala).
- **λ = 10 es una SELECCIÓN, no una consecuencia**: la banda ±10% de s0 = π/ln10 = 1.3644 captura solo unas unidades por ciento — y esa fracción DEPENDE de la escala del ansatz (s0 escala linealmente con a), así que ni siquiera es un número invariante: sin las β reales no hay predicción de λ. Exactamente la disyuntiva de §14.2: «si resulta λ ≠ 10, el diez era convención».

Estatuto: condicional (§14.2, frente abierto nº 2): la maquinaria espectro ⟺ walking está validada; las β-funciones reales (jerarquía de Fokker-Planck, Def. 4.4) son el hueco declarado — con ansatz O(1), la cascada DSI es genérica pero λ = 10 es una selección medible, no una consecuencia.
