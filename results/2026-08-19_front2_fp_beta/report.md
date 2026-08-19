# Frente 2: el espectro de las β de Fokker-Planck — desenlace A en el punto preinscrito, con corrección de alcance (adenda δ0)

Preinscripción: `1c72eb193843` (congelada ANTES del
primer autovalor); espectro: `10e7f56e62b0`; adenda:
`10e7f56e62b0`. Derivación validada por la suite (álgebra
⟺ jacobiano numérico ⟺ suavizado gaussiano exacto ⟺ FP exacta vía
Hopf-Cole).

## 1. El resultado en el punto preinscrito, tal cual

**DESENLACE A** (regla preinscrita, en su orden literal):
en el punto preinscrito (M0², B, C0) = (1, 2, 1) — espinodal D = 0 —
con el cierre canónico (Polchinski a = b = 0.5, truncamiento
cubic, τ = 1.0), el espectro de M = ∂β/∂λ es
**enteramente real**: μ = (-18.6603, -1.0065, +7.6668) — silla sin rotación, sin cascada
DSI. Además dD/dt = +28.0 > 0 en ese punto (Obs. 8.6
exige < 0). El sistemático cuártico (E4 = 0) da espectro también real
(μ = -31.667, -6.325, +4.794, +13.199).

Barrido preinscrito completo (21 puntos, g ∈
[0.1, 10.0]), publicado íntegro:
1/21 puntos con par complejo —
- g = 10.0: s0(τ=1) = 5.8661 (dos rutas, error rel. 6.0e-09), Q = |Im/Re| = 0.2864, τ* = 0.2326, dD/dt = -116.0
  El signo de dD/dt NO es exclusivo de ese punto: dD/dt = 44 − 16g
  sobre el barrido (cero en g = 2.75), negativo en 6 de
  21 puntos; lo exclusivo del borde es la CONJUNCIÓN
  rotación + hundimiento de D (par complejo solo para
  g > g* = 8.189, frontera exacta por
  el discriminante del polinomio característico). La ventana
  preinscrita g ∈ [0.5, 2.0] es uniformemente real.

## 2. CORRECCIÓN (revisión adversarial): el eje g es el eje δ0

La revisión adversarial de la ronda encontró (hallazgo HIGH,
confirmado por verificador independiente) que la primera versión
afirmaba «la dependencia en δ0 entra por el diccionario τ(δ0)» — es
FALSO. Las β cumplen la covarianza exacta β(D_s·λ; a, b) =
σ·D_s·β(λ; a, b·k) (verificada numéricamente: desviación máxima
9.1e-13), de modo que la familia
espinodal FÍSICA del escalado (3.2), (m̄²δ0², b̄δ0, C0), equivale
espectralmente a (1, 2, 1) con g_ef = δ0⁻³: **el barrido en g era, en
secreto, un barrido en δ0**, y el punto preinscrito fija
implícitamente δ0 ≈ 1 — fuera del régimen perturbativo δ0 ≪ 1 donde
vive el corpus del Basal. (Coherencia interna restaurada: el
despreciar η por O(δ0⁶) exige δ0 ≪ 1, incompatible con el punto
preinscrito; la adenda evalúa donde el argumento vale.)

**La adenda δ0** (fp_beta_delta0.json; misma derivación congelada,
sin tocar coeficientes ni reglas): a lo largo de la familia física
con el cierre canónico, el espectro se complejifica para
δ0 < δ0* = 0.4961 (= g*^(−1/3)) y allí dD/dt < 0 —
**en el régimen físico del corpus, la reducción canónica SÍ produce
las dos condiciones del corpus a la vez** (rotación + hundimiento de
D). En el punto físico de referencia δ0 = 0.1:
s0(τ=1) = 4.9461, Q = 1.3352,
dD/dt = -1.596, y λ = 10 exigiría
τ* = 0.2758 — publicado como dato: el
corpus recogido no fija τ; elegirlo a posteriori para forzar
s0 = π/ln10 sería tuning (compromiso preinscrito), por eso τ* se
publica, no se deriva. Aplicando las reglas preinscritas AL PUNTO
FÍSICO (clasificación contrafactual, declarada como tal — el
desenlace A del punto preinscrito NO se reclasifica):
**B** — hay cascada,
con s0 fuera de la banda de λ = 10 bajo el diccionario canónico.

## 3. Control de deriva (secundario prometido, cumplido y declarado)

La preinscripción prometía un «control de Langevin» (deriva medida de
los acoplos vs β predichas). Se cumplió con un control determinista
MÁS FUERTE y la sustitución se declara: la FP exacta vía Hopf-Cole
(e^{−V_t} = e^{tΔ/2}·e^{−V0}, sin ruido Monte Carlo), Richardson
t → 0, base hasta x⁵. Resultado (artefacto `drift_control`): error
relativo máximo 0.1% en el régimen declarado.

## 4. Qué significa (y qué no)

- **En el punto preinscrito (δ0 ≈ 1)**: desenlace A — la
  reducción canónica no produce la cascada allí. Se publicó tal cual
  y no se reclasifica.
- **En el régimen físico (δ0 ≲ 0.50)**: la misma
  reducción canónica SÍ rota y hunde D — favorable al MECANISMO de la
  cascada; pero s0(τ=1) ≠ π/ln10: **λ = 10 sigue sin derivarse** (el
  diccionario τ y su posible dependencia δ0 quedan declarados como no
  derivables del corpus recogido).
- **No significa**: ni que el mecanismo esté refutado (punto
  preinscrito) ni que λ = 10 esté derivado (punto físico). El alcance
  es interno (E8) y condicional a los cierres declarados; si el
  corpus fija otro cierre o el diccionario, se recalcula.

## 5. Estatuto del frente 2 tras esta ronda

Sigue ABIERTO, con contenido nuevo en las dos direcciones: la
reducción canónica de la Def. 4.4 produce cascada DSI exactamente en
el régimen perturbativo del corpus (δ0 < δ0* ≈ 0.50)
— el mecanismo tiene por primera vez una realización derivada, no
asumida — y λ = 10 sigue siendo una calibración (s0 depende del
diccionario τ, no derivable aún). La preinscripción tuvo un defecto
de diseño (el punto de evaluación escondía δ0 ≈ 1) que la revisión
adversarial destapó; se corrige por adenda, con la preinscripción
intacta y el defecto declarado — no reescrito.
