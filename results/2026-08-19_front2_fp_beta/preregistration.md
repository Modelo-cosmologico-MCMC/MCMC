# Preinscripción — frente 2: β de Fokker-Planck (Def. 4.4)

Commit: `1c72eb193843`. CONGELADA antes de computar ningún autovalor.

## Derivación (cierres declarados)

dV/dt = a·ΔV − b·(∇V)² (Polchinski d = 0); truncamiento cúbico
(cuártico = sistemático); sector η·χ O(δ0⁶) despreciado con
declaración; diccionario τ = dt/dlnS no derivable del corpus recogido
(reescala s0; el tipo espectral y Q = |Im|/|Re| son invariantes).

Coeficientes derivados y congelados (por unidad de t):

    β_M0² = −8·a·B − 2·b·M0⁴
    β_B   = −24·a·C0 − 8·b·M0²·B
    β_C0  = −6·b·(B² + 2·M0²·C0)

## Punto de evaluación

Espinodal D = 0 (Def. 8.4) en unidades naturales: (M0², B, C0) =
(1.0, 2.0,
1.0).

## Cierre canónico y espacio de robustez

a = 0.5, b = 0.5, signo = +1, τ = 1.0;
malla g = a/b: 21 puntos log-espaciados en
[0.1, 10.0]; ventana de robustez
g ∈ [0.5, 2.0]; banda de s0: ±10% de
π/ln10 = 1.3644.

## Los cuatro desenlaces (regla cuantitativa, congelada)

- **A** — sin par complejo en el punto canónico: la reducción no
  produce la cascada DSI (desfavorable; portada).
- **B** — par complejo, s0(τ=1) fuera de banda: λ ≠ 10 bajo el
  diccionario canónico; se publica τ*.
- **C** — par complejo, s0(τ=1) en banda: candidato (condicional al
  diccionario).
- **D** — el tipo espectral cambia en g ∈ [0.5, 2.0]:
  dependiente del cierre; mapa completo, sin veredicto sobre λ.

Se publican SIEMPRE: el barrido entero, Q, τ*, el signo de dD/dt
(Obs. 8.6), la sensibilidad al truncamiento y el control de Langevin.

## Compromiso

Sin ajuste de coeficientes tras conocer s0; banda/malla/ventana
congeladas; publicación íntegra sea cual sea el desenlace. Candado:
`tests/test_front2_fp_lock.py`.
