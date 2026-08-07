# Correspondencias E.18 y E.19 — propuesta para la v36 (Apéndice E)

Dos correspondencias nuevas con el Arte del Camino, nacidas de
resultados computacionales del programa (no del tratado), para su
inserción tras E.9 y E.12. Respetan la regla del propio apéndice: **no
demuestran física nueva — traducen una estructura técnica ya
identificada**, y el estatuto matemático queda separado de la lectura
subjetiva.

---

## E.18 — El fantasma del suelo perdido

**Estructura matemática.** El retraso dinámico tras el cruce de la
espinodal: con D(σ) hundiéndose, la anulación del discriminante y el
comienzo efectivo de la caída no son el mismo acontecimiento — el
colapso llega con un retraso ∝ ritmo^(−1/3) (silla-nodo con deriva;
formulación condicional declarada). Tres momentos distintos:
aproximación al umbral, anulación del discriminante, caída efectiva
posterior.

*Anclaje:* `core/kls_flow.py` (`cruce_de_victoria`, `delay_scaling`);
desenlace en `results/2026-08-02_kls_flow/`.

**Lectura subjetiva.** Una estructura puede haber dejado de sostenernos
antes de que seamos capaces de caer fuera de ella.

**Idea central.**

> La tensión desborda antes; la caída llega después.

No todo retraso es permanencia. A veces es la memoria dinámica de un
equilibrio que ya no existe.

---

## E.19 — La integridad como simetría especular

**Estructura matemática.** La positividad por reflexión conservada por
un perfil especular: la hipótesis real no es la estacionariedad
(∂_S K = 0) sino la simetría K(S) = K(ϑS) — el perfil constante es el
caso especular trivial; el variable pero especular conserva la
positividad; el monótono sin reflexión modificada la rompe; el
reflejado junto con el campo la restaura exactamente. (Alcance: juguete
escalar hasta resolver Wilson.)

*Anclaje:* `core/rp_nonstationary.py` (`is_mirror_symmetric`,
`PRECEDENCE_REFORMULATED`); desenlace en
`results/2026-08-02_rp_nonstationary/`.

**Lectura subjetiva.** La integridad no exige permanecer idéntico;
exige que la transformación permita reconocer una continuidad entre el
antes y el después.

**Idea central.**

> Cambiar no rompe el presente. Lo rompe que el antes y el después ya
> no puedan reconocerse a través del acto que los separa.
