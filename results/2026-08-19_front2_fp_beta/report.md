# Frente 2: el espectro de las β de Fokker-Planck — desenlace A

Preinscripción: `1c72eb193843` (congelada ANTES del
primer autovalor); espectro: `c0868cdeaeea`. Derivación
validada por la suite (álgebra ⟺ jacobiano numérico ⟺ suavizado
gaussiano exacto ⟺ FP exacta vía Hopf-Cole).

## El resultado, tal cual

**DESENLACE A** (regla preinscrita): en el punto canónico
(espinodal D = 0, cierre Polchinski a = b = 0.5, truncamiento
cubic, τ = 1.0) el espectro de M = ∂β/∂λ es
**enteramente real**: μ = (-18.6603, -1.0065, +7.6668) — una silla sin rotación. La
reducción de Fokker-Planck de la Def. 4.4, con el cierre canónico,
**no produce el par complejo de la cascada DSI**: el exponente de
Victoria no emerge de esta β.

Doble negativa coherente con el corpus: además del espectro real,
**dD/dt = +28.0 > 0** en la espinodal — el flujo canónico
SUBE el discriminante, cuando la Obs. 8.6 exige hundirlo (D debe
cruzar a negativo para disparar cada colapso).

## Robustez (todo publicado, caiga donde caiga)

- Ventana preinscrita g ∈ [0.5, 2.0]: tipo espectral UNIFORME
  (real) — el desenlace A es robusto, no D.
- Barrido completo (21 puntos, g ∈
  [0.1, 10.0]): 1/21 puntos
  con par complejo:
- g = 10.0: s0(τ=1) = 5.8661 (dos rutas, error rel. 6.0e-09), Q = |Im/Re| = 0.2864, dD/dt = -116.0
  El único punto complejo es el extremo de difusión dominante — y es
  también el ÚNICO donde dD/dt < 0: las dos condiciones del corpus
  (rotación + hundimiento de D) solo aparecen JUNTAS en el borde
  g ≫ 1, lejos del cierre canónico. Incluso allí, s0 no es π/ln10
  = 1.3644 sin un diccionario τ ajustado a mano
  (prohibido por la preinscripción).
- Sistemático del truncamiento: la variante cuártica (E4 = 0) da
  espectro también real (μ = -31.667, -6.325, +4.794, +13.199)
  — el desenlace no es un artefacto del cierre cúbico.

## Qué significa (y qué no)

- **Significa**: dentro de esta reducción (Polchinski d = 0 con kernel
  isótropo, jerarquía truncada, sector η despreciado), la cascada DSI
  NO es genérica: λ = 10 sigue siendo una CALIBRACIÓN (§14.2: «si
  resulta λ ≠ 10, el diez era convención» — pero aquí ni siquiera hay
  λ que resultar en el cierre canónico). El resultado ACOTA dónde
  podrían vivir unas β que sí roten: cierres fuertemente anisótropos
  o dominados por difusión (g ≳ 10), o física fuera de esta reducción.
- **No significa**: que el mecanismo del tratado esté refutado — el
  alcance preinscrito es interno (E8): UNA reducción con cierres
  declarados de la Def. 4.4, no la única posible. Si el corpus fija
  otro cierre (G anisótropa, temperatura entrópica dependiente de S,
  diccionario τ concreto), se recalcula con él — los cierres están
  expuestos como parámetros, no resueltos en silencio.

## Estatuto del frente 2 tras esta ronda

Sigue ABIERTO, con contenido nuevo: antes «las β no existen» (hueco
declarado); ahora «las β de la reducción canónica existen y NO
producen Victoria» (negativa concreta, con candados). La carga de la
derivación de λ = 10 queda sobre cierres no canónicos o sobre física
adicional del corpus — cuantificado, no vago.
