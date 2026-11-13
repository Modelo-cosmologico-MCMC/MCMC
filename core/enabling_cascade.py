"""Cascada de habilitación — modo opcional del reloj S en el que los colapsos de la Década no se imponen: los
produce la pérdida de estabilidad de tres direcciones latentes sobre un espacio de estados fijo (R6 del plan del
11-nov-2026; revisión profunda, continuación; comprobación interna, E8).

DECISIONES DEL AUTOR (13-nov-2026; texto literal en `docs/decisiones_autor_2026-11-13.md`).
- Decisión 4: «Latente primero». Un solo potencial sobre un espacio de estados fijo que ya contiene las direcciones
  q_k; cada colapso es la pérdida de estabilidad de una dirección. La lectura «creado» queda en reserva.
- Decisión 5: «Los tres q_k no representan tres nuevas esencias fundamentales independientes. Los interpreto como
  parámetros de orden asociados a modos de estructuración emergente de la Unidad Dual.» Y también: «El modelo R6 puede
  utilizar tres amplitudes q_k como descripción efectiva de estas transiciones, pero su interpretación como modos
  emergentes deberá justificarse mediante una dinámica común derivada de la Unidad Dual. Su mera introducción no
  constituye una demostración de emergencia dimensional.»
- Decisión 7: «Adopto provisionalmente una progresión geométrica de razón λ para los pesos del Basal, q_1 y q_2, con un
  peso residual para q_3, de forma que los cuatro permanezcan normalizados.» Y también: «La coincidencia entre la razón
  de escala de los pesos y la Ley de la Década se considera una hipótesis pendiente de derivación, no una identidad
  física demostrada.»

EL MODELO. Al Potencial Basal completo del reloj (Plano Dual, con la inclinación −η·χ) se le añaden tres parámetros de
orden q_1, q_2, q_3, uno por tramo dimensional (V1D, V2D, V3D). En las variables escaladas del Basal (Φ = √δ0·Φ̂,
V = δ0³·V̂, σ̂ = σ·δ0²):

    V̂ = V̂0(φ̂_M, φ̂_E) + Σ_k W_k·[½·m_k²·q̂_k² + ¼·q̂_k⁴],   m_k² = −tanh((Y_{k−1} − θ_g)/Δ_g),

con Y_0 = ρ̂²/ρ̂₊² la descarga del Basal (ρ̂₊ es el vacío verdadero del paisaje completo) e Y_k = q̂_k² la habilitación del
modo k. La masa de cada modo es positiva hasta que el tramo anterior ha avanzado una fracción θ_g; ahí se anula y la
dirección se abre: una estructura dimensional superior no puede realizarse antes que las inferiores. Un solo potencial,
acoplos fijos y Flujo del Camino de gradiente puro: la Monotonía (Teo. 4.5) sigue siendo teorema y este modo la comprueba
paso a paso, a diferencia del modo 'emergent' del reloj, donde los acoplos se mueven.

LOS PESOS (decisión 7). La profundidad nominal de cada sector es W_k/4. Con la ley de escala de un número λ, el Basal,
q_1 y q_2 están en progresión geométrica de razón λ y q_3 lleva el resto: fracciones (λ−1)/λ³, (λ−1)/λ², (λ−1)/λ y 1/λ³
de la tensión total nominal T₀·λ³/(λ−1), con T₀ la Tensión Primordial del Basal MEDIDA en el paisaje completo. Con
λ = 10: 0.9 %, 9 %, 90 % y 0.1 %. Son coeficientes relativos de contribución al modelo efectivo, no fracciones de masa,
espacio ni energía física. λ = 10 es el valor del canon (`mcmc_ontology.constants.LAMBDA_DECADE`), CALIBRADO: aquí
no se deriva.

EL RELOJ. S ≡ f = (V_inicial − V)/T_total, con T_total = V_inicial − V_mínimo del potencial AMPLIADO: S = 1 ⟺ descarga
completa del modelo entero (si S contara solo la tensión del Basal, los umbrales no serían los de la Década). El campo
arranca en el punto de escape del reloj, con cada q̂_k en su semilla.

LOS EVENTOS. El hessiano se toma sobre la dirección radial del Plano Dual y las tres direcciones latentes. La dirección
angular del plano queda fuera: su curvatura V′(ρ)/ρ es geométrica (negativa en todo descenso radial) y su dinámica es la
de la cuarta ecuación, no la de la cascada.
- gatillo_k: el autovalor del hessiano reducido asociado a q_k pasa de positivo a no positivo (M_k² = 0: el origen de la
  dirección nueva pierde estabilidad). Se publica si en ese momento es el autovalor mínimo.
- habilitacion_k: Y_k cruza ½ hacia arriba.
- colapso_kD: un gatillo seguido de su habilitación. Su S es el del gatillo.
Sin semilla el gatillo existe y la habilitación no: si V es par en q_k, el estrato q_k = 0 es invariante bajo el Camino.

LO QUE NO ES. No deriva λ = 10 ni demuestra que un parámetro de orden q_k sea una dimensión espacial: son tres
amplitudes añadidas y un acople postulado (la compuerta tanh). Los colapsos son espinodales, no nucleados (la variante
nucleada es R7 y espera la decisión 6). No toca el cruce de la diagonal ni la Rotación de Florencia: hasta S = 1.001 la
variable S ordena la estructuración pregeométrica sin ser un reloj físico. El comportamiento por defecto del reloj S no
cambia: este modo vive en este módulo y toma del reloj el paisaje, el punto de escape, T₀ y el esquema de integración
(RK4 de paso fijo en σ̂, proyectado sobre el cuadrante físico); con las semillas a cero reproduce su descenso paso a paso.

Estatuto: comprobación interna (E8), con letra de consistencia solo bajo preinscripción congelada
(`scripts/run_enabling_cascade.py`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize

from mcmc_ontology import constants as C

from .florencia import all_anticommute, chain_generators, signature
from .s_clock import ClockConfig, SClock, to_jsonable

N_MODES = 3
TRAMOS = ("V0D", "V1D", "V2D", "V3D")
SQRT2 = math.sqrt(2.0)

STATUS = ("comprobación interna (E8): cascada de habilitación con direcciones latentes (decisión 4 del autor, 13-nov-2026), "
          "tres parámetros de orden q_k como descripción efectiva (decisión 5) y pesos por una ley de escala de un número "
          "(decisión 7); λ calibrado, compuerta postulada, colapsos espinodales; no es una derivación de la Década ni de "
          "la emergencia dimensional")

DECLARED_FORMS = {
    "sector": "sector cuártico por modo: W_k·[½·m_k²·q̂_k² + ¼·q̂_k⁴], par en q̂_k (el estrato q̂_k = 0 es invariante). "
              "FORMA DECLARADA: con otra forma cambian las cifras, no la existencia.",
    "gate": "compuerta m_k² = −tanh((Y_{k−1} − θ_g)/Δ_g) con Y_0 = ρ̂²/ρ̂₊² e Y_k = q̂_k²: acople POSTULADO entre cada tramo y "
            "el siguiente. θ_g y Δ_g son dos números elegidos (0.9 y 0.05 por defecto).",
    "scale_law": "decisión 7 del autor: fracciones (λ−1)/λ³, (λ−1)/λ², (λ−1)/λ, 1/λ³ de la tensión nominal T₀·λ³/(λ−1); "
                 "W_k = 4·(profundidad nominal del sector k). λ = 10 del canon, CALIBRADO: hipótesis pendiente de derivación.",
    "S_normalization": "S ≡ f = (V_inicial − V)/T_total con T_total del potencial ampliado: S = 1 ⟺ descarga completa del "
                       "modelo entero. Declarado (si S contara solo la tensión del Basal los umbrales no serían los de la Década).",
    "rigidity": "la rigidez escalar G del reloj vale también para las direcciones latentes, que escalan como el campo "
                "(q = √δ0·q̂). Declarado.",
    "seed": "cada q̂_k arranca en una semilla declarada (1e-6 por defecto). Sin semilla ninguna dirección se abre: el "
            "gatillo necesita la imperfección o la difusión para producir un colapso.",
    "event": "gatillo_k: el autovalor del hessiano reducido (dirección radial del Plano Dual y direcciones latentes) asociado "
             "a q_k pasa de positivo a no positivo; habilitacion_k: Y_k cruza ½; colapso_kD: gatillo y habilitación, con el S "
             "del gatillo. La dirección angular del Plano Dual queda fuera del hessiano reducido.",
    "nucleation": "el Basal sale de S₀ como en el reloj (punto de escape de Coleman); los tres colapsos de la cascada son "
                  "espinodales, sin barrera. La variante nucleada es R7 (decisión 6, pendiente).",
}


def scale_law_shares(lam: float) -> tuple[float, float, float, float]:
    """Reparto de la ley de escala de un número (decisión 7 del autor, 13-nov-2026): progresión geométrica de razón λ
    para el Basal, q_1 y q_2 y peso residual para q_3, normalizados a 1.

    Devuelve las fracciones de (V0D, V1D, V2D, V3D): ((λ−1)/λ³, (λ−1)/λ², (λ−1)/λ, 1/λ³). Con λ = 10: 0.9 %, 9 %, 90 % y
    0.1 %. Son coeficientes relativos del modelo efectivo; λ no se deriva."""
    if not lam > 1.0:
        raise ValueError("la ley de escala exige λ > 1 (el peso residual 1/λ³ y la razón λ dejan de estar ordenados)")
    return ((lam - 1.0) / lam ** 3, (lam - 1.0) / lam ** 2, (lam - 1.0) / lam, 1.0 / lam ** 3)


def cumulative_thresholds(shares) -> list[float]:
    """Umbrales nominales de un reparto: fracción acumulada al terminar V0D, V1D y V2D. Con la ley de escala son
    λ^(k−3) − λ^(−3) (k = 1, 2, 3): los de la Ley de la Década para λ = 10 (0.009, 0.099, 0.999)."""
    s = [float(x) for x in shares]
    tot = sum(s)
    return [sum(s[:k]) / tot for k in (1, 2, 3)]


def gate_mass(Y: float, theta_g: float, Delta_g: float) -> tuple[float, float, float]:
    """Compuerta del modo k: m_k² = −tanh(u), u = (Y_{k−1} − θ_g)/Δ_g, con sus dos derivadas respecto de Y_{k−1}.
    Positiva (dirección cerrada) hasta que el tramo anterior alcanza la fracción θ_g; ahí se anula."""
    u = (Y - theta_g) / Delta_g
    t = math.tanh(u)
    s2 = 1.0 - t * t
    return -t, -s2 / Delta_g, 2.0 * s2 * t / Delta_g ** 2


def ratios(S_list) -> list[float]:
    """Cocientes S_{k+1}/S_k entre gatillos sucesivos."""
    return [float(S_list[i + 1] / S_list[i]) for i in range(len(S_list) - 1) if S_list[i] > 0.0]


def cascade_letter(S_triggers, rules: dict) -> dict:
    """Letra de una celda (regla del plan del 11-nov, la del candidato 3 aplicada a los gatillos): A si los dos
    cocientes S_{k+1}/S_k caen en el intervalo A; B si cae uno; C si no cae ninguno o hay menos de tres colapsos. Las
    puertas y los controles se aplican antes y fuera de esta función."""
    lo, hi = rules["A_interval"]
    n_req = int(rules["n_collapses_required"])
    if len(S_triggers) < n_req:
        return {"letter": "C", "ratios": ratios(S_triggers), "n_in_interval": 0,
                "reason": f"{len(S_triggers)} colapsos completos; hacen falta {n_req}"}
    r = ratios(list(S_triggers)[:n_req])
    inside = sum(1 for x in r if lo <= x <= hi)
    letter = "A" if inside == len(r) else "B" if inside >= 1 else "C"
    return {"letter": letter, "ratios": r, "n_in_interval": inside,
            "reason": f"{inside} de {len(r)} cocientes en [{lo:g}, {hi:g}]"}


@dataclass(frozen=True)
class CascadeConfig:
    """Entradas del modo cascada. `clock` da el Basal (formas, δ0, rayo y nucleación); lo demás es del modelo efectivo."""
    clock: ClockConfig = field(default_factory=ClockConfig)
    lam_scale: float = C.LAMBDA_DECADE          # λ de la ley de escala (decisión 7); calibrado, no derivado
    shares: tuple | None = None                 # reparto explícito de (V0D, V1D, V2D, V3D) para controles; None = ley de escala
    tension_reference: str = "measured"         # referencia de los pesos: 'measured' (T₀ del paisaje completo, la del reloj)
                                                # | 'law' (c̄·δ0³, la ley de la Prop. 3.4: sin la corrección de la inclinación)
    gate_theta: float = 0.9                     # θ_g: fracción del tramo anterior que abre el siguiente
    gate_Delta: float = 0.05                    # Δ_g: anchura de la compuerta
    seeds: tuple = (1e-6, 1e-6, 1e-6)           # semilla de cada q̂_k (0 = dirección sin sembrar)
    gated: bool = True                          # False: control negativo sin compuerta (m_k² ≡ +1)
    noise_floor: float = 0.0                    # suelo de ruido: |q̂_k| ≥ suelo en cada paso para las direcciones sembradas
                                                # (0 = dinámica exacta de una semilla inicial; > 0 emula una fuente continua)
    dS_step: float = 2e-4                       # paso: h = min(d_sigma_hat del reloj, dS_step·T_total/Σ̇); resuelve los colapsos rápidos
    amplitude_representation: str = "linear"    # 'linear': se integra q̂_k (ronda 1) | 'log': se integra ln q̂_k, que no
                                                # se agota cuando la semilla decae por debajo de lo que el flotante representa
    max_sigma_hat: float = 400.0                # guarda del recorrido en σ̂
    v_tol: float = 1e-10                        # fin: velocidad proyectada < v_tol·T_total/ρ̂₊
    record_every: int = 5


class CascadeClock:
    """El bucle del modo cascada. `run()` devuelve trayectoria, eventos, colapsos y comprobaciones con su etiqueta."""

    def __init__(self, cfg: CascadeConfig | None = None):
        self.cfg = cfg or CascadeConfig()
        k = self.cfg
        c = k.clock
        if c.thresholds != "imposed" or c.couplings_flow:
            raise ValueError("el modo cascada exige acoplos fijos (thresholds='imposed', couplings_flow=False): solo así la "
                             "Monotonía es teorema")
        if c.J_circ != 0.0 or c.kappa_conv != 0.0 or c.tilt_cross_f is not None or c.beta_extra is not None:
            raise ValueError("el modo cascada es flujo de gradiente puro: sin circulación, corriente de conversión, rotación "
                             "de la inclinación ni Δβ")
        if c.nucleation == "bounce":
            raise ValueError("el modo cascada arranca en el punto de escape (el bounce es control negativo del reloj)")
        if len(k.seeds) != N_MODES or any(s < 0.0 for s in k.seeds):
            raise ValueError("seeds: tres semillas no negativas, una por dirección latente")
        if k.noise_floor < 0.0:
            raise ValueError("noise_floor ≥ 0")
        if not k.dS_step > 0.0:
            raise ValueError("dS_step > 0")
        if k.amplitude_representation not in ("linear", "log"):
            raise ValueError("amplitude_representation: 'linear' | 'log'")
        if not (k.gate_Delta > 0.0 and 0.0 < k.gate_theta):
            raise ValueError("compuerta: θ_g > 0 y Δ_g > 0")
        self.base = SClock(c)                               # paisaje, punto de escape y T₀ del reloj, sin tocarlo
        d = c.delta0
        self.T0_hat = self.base.T0 / d ** 3
        self.x_plus_hat = self.base.x_plus0 / d
        self.x_esc_hat = self.base.x_esc / d
        self.m2, self.b, self.C0, self.G = c.m_bar ** 2, c.b_bar, c.C0, c.G
        self.tilt_hat = c.e_bar * math.sqrt(d)              # −η·χ = δ0³·(−ē√δ0·χ̂)
        self.mu2_hat = self.base.mu2 / d ** 2
        self.shares = tuple(float(x) for x in (k.shares if k.shares is not None else scale_law_shares(k.lam_scale)))
        if len(self.shares) != N_MODES + 1 or any(s <= 0.0 for s in self.shares):
            raise ValueError("shares: cuatro fracciones positivas (V0D, V1D, V2D, V3D)")
        if k.tension_reference not in ("measured", "law"):
            raise ValueError("tension_reference: 'measured' | 'law'")
        # profundidad nominal de cada sector latente en unidades de δ0³, relativa a la Tensión Primordial del Basal: la
        # MEDIDA en el paisaje completo (la del reloj) o la de la ley c̄·δ0³ (control de sensibilidad)
        self.T_ref_hat = self.T0_hat if k.tension_reference == "measured" else float(self.base.T0_law) / d ** 3
        self.depth_hat = [self.T_ref_hat * self.shares[j] / self.shares[0] for j in (1, 2, 3)]
        self.W = [4.0 * x for x in self.depth_hat]
        self.T_nominal_hat = self.T0_hat + sum(self.depth_hat)
        self.y_min, self.V_min = self._global_minimum()

    # ------------------------------------------------------------------ potencial, gradiente y hessiano (escalados)
    def _gate(self, Y: float) -> tuple[float, float, float]:
        if not self.cfg.gated:
            return 1.0, 0.0, 0.0
        return gate_mass(Y, self.cfg.gate_theta, self.cfg.gate_Delta)

    def V_hat(self, y) -> float:
        a, e = y[0], y[1]
        x = a * a + e * e
        chi = (a - e) / SQRT2
        v = 0.5 * self.m2 * x - 0.25 * self.b * x * x + (self.C0 / 6.0) * x ** 3 - self.tilt_hat * chi + 0.5 * self.mu2_hat * chi * chi
        Y = x / self.x_plus_hat
        for j in range(N_MODES):
            q = y[2 + j]
            v += self.W[j] * (0.5 * self._gate(Y)[0] * q * q + 0.25 * q ** 4)
            Y = q * q
        return v

    def grad_hat(self, y) -> list[float]:
        a, e, q1, q2, q3 = y
        x = a * a + e * e
        R = self.m2 - self.b * x + self.C0 * x * x
        ang = 0.5 * self.mu2_hat * (a - e)
        ga = R * a - self.tilt_hat / SQRT2 + ang
        ge = R * e + self.tilt_hat / SQRT2 - ang
        W1, W2, W3 = self.W
        m1, d1, _ = self._gate(x / self.x_plus_hat)
        c1 = W1 * q1 * q1 * d1 / self.x_plus_hat            # ∂V/∂x por la compuerta del modo 1
        ga += c1 * a
        ge += c1 * e
        m2_, d2, _ = self._gate(q1 * q1)
        m3, d3, _ = self._gate(q2 * q2)
        g1 = W1 * (m1 * q1 + q1 ** 3) + W2 * q2 * q2 * d2 * q1
        g2 = W2 * (m2_ * q2 + q2 ** 3) + W3 * q3 * q3 * d3 * q2
        g3 = W3 * (m3 * q3 + q3 ** 3)
        return [ga, ge, g1, g2, g3]

    def hessian_hat(self, y) -> np.ndarray:
        a, e, q1, q2, q3 = y
        x = a * a + e * e
        R = self.m2 - self.b * x + self.C0 * x * x
        Rp = -self.b + 2.0 * self.C0 * x
        W1, W2, W3 = self.W
        xp = self.x_plus_hat
        m1, d1, dd1 = self._gate(x / xp)
        m2_, d2, dd2 = self._gate(q1 * q1)
        m3, d3, dd3 = self._gate(q2 * q2)
        H = np.zeros((5, 5))
        cx = W1 * q1 * q1 * d1 / xp                          # la compuerta del modo 1 en el gradiente: ∂T₁/∂φ̂_i = cx·φ̂_i
        cxx = W1 * q1 * q1 * dd1 / xp ** 2                   # y su derivada: ∂cx/∂φ̂_j = 2·cxx·φ̂_j
        H[0, 0] = R + 2.0 * Rp * a * a + 0.5 * self.mu2_hat + cx + 2.0 * cxx * a * a
        H[1, 1] = R + 2.0 * Rp * e * e + 0.5 * self.mu2_hat + cx + 2.0 * cxx * e * e
        H[0, 1] = H[1, 0] = 2.0 * Rp * a * e - 0.5 * self.mu2_hat + 2.0 * cxx * a * e
        H[0, 2] = H[2, 0] = 2.0 * W1 * q1 * d1 * a / xp
        H[1, 2] = H[2, 1] = 2.0 * W1 * q1 * d1 * e / xp
        H[2, 2] = W1 * (m1 + 3.0 * q1 * q1) + W2 * q2 * q2 * (d2 + 2.0 * dd2 * q1 * q1)
        H[2, 3] = H[3, 2] = 2.0 * W2 * q2 * d2 * q1
        H[3, 3] = W2 * (m2_ + 3.0 * q2 * q2) + W3 * q3 * q3 * (d3 + 2.0 * dd3 * q2 * q2)
        H[3, 4] = H[4, 3] = 2.0 * W3 * q3 * d3 * q2
        H[4, 4] = W3 * (m3 + 3.0 * q3 * q3)
        return H

    def reduced_hessian(self, y) -> np.ndarray:
        """Hessiano sobre la dirección radial del Plano Dual y las tres direcciones latentes (4×4). Con el campo en el
        origen del plano, la dirección radial es la del rayo de nucleación."""
        H = self.hessian_hat(y)
        rho = math.hypot(y[0], y[1])
        th = self.cfg.clock.theta_nuc
        er = (y[0] / rho, y[1] / rho) if rho > 0.0 else (math.cos(th), math.sin(th))
        P = np.zeros((5, 4))
        P[0, 0], P[1, 0] = er
        P[2, 1] = P[3, 2] = P[4, 3] = 1.0
        return P.T @ H @ P

    def mode_curvatures(self, y) -> tuple[list[float], float, int]:
        """Autovalor del hessiano reducido asociado a cada q_k (el de autovector con mayor componente en q_k), el
        autovalor mínimo y el índice de la componente dominante de su autovector (0 = radial, k = dirección latente k)."""
        w, U = np.linalg.eigh(self.reduced_hessian(y))
        kap = []
        for j in (1, 2, 3):
            kap.append(float(w[int(np.argmax(np.abs(U[j, :])))]))
        i_min = int(np.argmin(w))
        return kap, float(w[i_min]), int(np.argmax(np.abs(U[:, i_min])))

    def _global_minimum(self) -> tuple[list[float], float]:
        """Mínimo del potencial ampliado en el dominio físico (φ̂ ≥ 0), desde el vacío nominal: fija T_total."""
        th = self.cfg.clock.theta_nuc
        rp = math.sqrt(self.x_plus_hat)
        y0 = np.array([rp * math.cos(th), rp * math.sin(th), 1.0, 1.0, 1.0])
        sc = self.T_nominal_hat
        r = minimize(lambda z: self.V_hat(z) / sc, y0, jac=lambda z: np.array(self.grad_hat(z)) / sc,
                     bounds=[(0.0, None), (0.0, None)] + [(None, None)] * N_MODES, method="L-BFGS-B",
                     options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 5000})
        y = [float(v) for v in r.x]
        # pulido de Newton sobre las variables libres (las del Plano Dual que no están sobre la frontera y las latentes)
        for _ in range(60):
            g = np.array(self.grad_hat(y))
            free = [i for i in range(5) if not (i < 2 and y[i] <= 0.0 and g[i] > 0.0)]
            Hf = self.hessian_hat(y)[np.ix_(free, free)]
            try:
                step = np.linalg.solve(Hf, -g[free])
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)) or float(np.max(np.abs(step))) > 0.5:
                break
            for i, s in zip(free, step):
                y[i] += float(s)
            y[0], y[1] = max(y[0], 0.0), max(y[1], 0.0)
            if float(np.max(np.abs(step))) < 1e-15:
                break
        return y, float(self.V_hat(y))

    # ------------------------------------------------------------------ flujo
    def _velocity(self, y) -> tuple[list[float], float]:
        """Flujo del Camino de gradiente proyectado sobre el dominio físico del Plano Dual, como el reloj: sobre la
        frontera φ̂_i = 0 con velocidad saliente la componente se anula y la ligadura no trabaja."""
        g = self.grad_hat(y)
        v = [-gi / self.G for gi in g]
        if y[0] <= 0.0 and v[0] < 0.0:
            v[0] = 0.0
        if y[1] <= 0.0 and v[1] < 0.0:
            v[1] = 0.0
        return v, -sum(gi * vi for gi, vi in zip(g, v))

    def latent_rates(self, y) -> list[float]:
        """r_k tal que ∂V̂/∂q̂_k = r_k·q̂_k: todo el gradiente latente es proporcional a la amplitud (V̂ es par en q̂_k), de
        modo que d ln q̂_k/dσ̂ = −r_k/G es regular aunque q̂_k sea diminuta."""
        a, e, q1, q2, q3 = y
        W1, W2, W3 = self.W
        m1 = self._gate((a * a + e * e) / self.x_plus_hat)[0]
        m2_, d2, _ = self._gate(q1 * q1)
        m3, d3, _ = self._gate(q2 * q2)
        return [W1 * (m1 + q1 * q1) + W2 * q2 * q2 * d2, W2 * (m2_ + q2 * q2) + W3 * q3 * q3 * d3, W3 * (m3 + q3 * q3)]

    def _velocity_log(self, s) -> tuple[list[float], float]:
        """El mismo flujo en las variables (φ̂_M, φ̂_E, ln q̂_1, ln q̂_2, ln q̂_3): d ln q̂_k/dσ̂ = −r_k/G. Una dirección sin
        sembrar (ln q̂ = −∞) no se mueve. Σ̇ es la misma cantidad que en la representación lineal."""
        y = [s[0], s[1]] + [math.exp(v) if v > -math.inf else 0.0 for v in s[2:]]
        g = self.grad_hat(y)
        va, ve = -g[0] / self.G, -g[1] / self.G
        if y[0] <= 0.0 and va < 0.0:
            va = 0.0
        if y[1] <= 0.0 and ve < 0.0:
            ve = 0.0
        r = self.latent_rates(y)
        dl = [-rj / self.G if v > -math.inf else 0.0 for rj, v in zip(r, s[2:])]
        sprod = -(g[0] * va + g[1] * ve) + sum((rj * qj) ** 2 for rj, qj in zip(r, y[2:])) / self.G
        return [va, ve] + dl, sprod

    def run(self) -> dict:
        k, c = self.cfg, self.cfg.clock
        th = c.theta_nuc
        rho_esc = math.sqrt(self.x_esc_hat)
        y = [rho_esc * math.cos(th), rho_esc * math.sin(th)] + [float(s) for s in k.seeds]
        V_start = self.V_hat(y)
        T_total = V_start - self.V_min
        h_max = c.d_sigma_hat                           # el paso del reloj; solo se acorta en los colapsos rápidos
        v_scale = T_total / math.sqrt(self.x_plus_hat)
        keys = ("sigma_hat", "S", "f", "V_hat", "Y0", "theta", "Y1", "Y2", "Y3", "lam_min", "Sprod")
        rec = {kk: [] for kk in keys}
        events, triggers, halves = [], {}, {}
        kap, lam_min, _ = self.mode_curvatures(y)
        sigma, S_int, f_val, V_prev = 0.0, 0.0, 0.0, V_start
        max_dV, min_sprod = -math.inf, math.inf
        max_id_diff = 0.0
        W_floor = 0.0                                   # trabajo del suelo de ruido (0 sin suelo)
        Yk = [q * q for q in y[2:]]
        n, finished, stop_reason = 0, False, None
        max_steps = int(math.ceil(k.max_sigma_hat / h_max)) * 20
        underflow = []                                  # direcciones sembradas que el flotante ha llevado a 0 exacto
        h_min_used = h_max
        log_repr = k.amplitude_representation == "log"
        vel = self._velocity_log if log_repr else self._velocity
        ln_floor = math.log(k.noise_floor) if k.noise_floor > 0.0 else -math.inf

        def to_y(st):
            return [st[0], st[1]] + [math.exp(v) if v > -math.inf else 0.0 for v in st[2:]] if log_repr else st

        # estado integrado: las amplitudes o sus logaritmos (−∞ para una dirección sin sembrar)
        st = ([y[0], y[1]] + [math.log(q) if q > 0.0 else -math.inf for q in y[2:]]) if log_repr else y

        def record(sprod):
            x = y[0] * y[0] + y[1] * y[1]
            vals = (sigma, S_int, f_val, V_prev, x / self.x_plus_hat, math.atan2(y[1], y[0]), Yk[0], Yk[1], Yk[2], lam_min, sprod)
            for kk, vv in zip(keys, vals):
                rec[kk].append(vv)

        v1, s1 = vel(st)
        record(s1)
        while n < max_steps and sigma < k.max_sigma_hat:
            n += 1
            # el reloj no avanza más de dS_step por paso: con Σ̇ grande (colapso rápido de un sector) el paso se acorta
            h = h_max if s1 <= 0.0 else min(h_max, k.dS_step * T_total / s1)
            h_min_used = min(h_min_used, h)
            st2 = [yi + 0.5 * h * vi for yi, vi in zip(st, v1)]
            v2, s2 = vel(st2)
            st3 = [yi + 0.5 * h * vi for yi, vi in zip(st, v2)]
            v3, s3 = vel(st3)
            st4 = [yi + h * vi for yi, vi in zip(st, v3)]
            v4, s4 = vel(st4)
            st_new = [yi + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4) for yi, a1, a2, a3, a4 in zip(st, v1, v2, v3, v4)]
            st_new[0], st_new[1] = max(st_new[0], 0.0), max(st_new[1], 0.0)
            y_new = to_y(st_new)
            if not all(math.isfinite(v) for v in y_new):
                stop_reason = f"divergencia numérica en el paso {n} (se publica, no se reescala)"
                break
            S_new = S_int + (h / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / T_total
            V_new = self.V_hat(y_new)
            max_dV = max(max_dV, V_new - V_prev)          # Monotonía del flujo (antes del suelo de ruido, si lo hay)
            if k.noise_floor > 0.0:
                # fuente continua declarada: una dirección sembrada no baja del suelo; su trabajo se publica
                touched = False
                for j in range(N_MODES):
                    if k.seeds[j] > 0.0 and (st_new[2 + j] < ln_floor if log_repr else abs(y_new[2 + j]) < k.noise_floor):
                        st_new[2 + j] = ln_floor if log_repr else k.noise_floor
                        touched = True
                if touched:
                    y_new = to_y(st_new)
                    V_floor = self.V_hat(y_new)
                    W_floor += V_floor - V_new
                    V_new = V_floor
            for j in range(N_MODES):
                # en la representación lineal el flotante puede llevar la amplitud a 0 exacto (criterio de la ronda 1; un
                # subnormal que ya no crece no llega a 0 y se ve en que la corrida no termina)
                if not log_repr and k.seeds[j] > 0.0 and y_new[2 + j] == 0.0 and (j + 1) not in underflow:
                    underflow.append(j + 1)
            f_new = (V_start - V_new) / T_total
            min_sprod = min(min_sprod, s1, s2, s3, s4)
            max_id_diff = max(max_id_diff, abs(S_new - f_new - W_floor / T_total))
            kap_new, lam_new, dom_new = self.mode_curvatures(y_new)
            Y_new = [q * q for q in y_new[2:]]
            for j in range(N_MODES):
                # gatillo: el autovalor asociado a q_k pasa de positivo a no positivo
                if (j + 1) not in triggers and kap[j] > 0.0 >= kap_new[j]:
                    w = kap[j] / (kap[j] - kap_new[j])
                    x_new = y_new[0] ** 2 + y_new[1] ** 2
                    ev = {"kind": f"gatillo_{j + 1}", "mode": j + 1, "opens": TRAMOS[j + 1], "S": S_int + w * (S_new - S_int),
                          "sigma_hat": sigma + w * h,
                          # habilitación del tramo anterior al dispararse: Y_0 del Basal o Y_{k−1} del modo previo
                          "Y_prev_at_trigger": float(x_new / self.x_plus_hat) if j == 0 else float(Y_new[j - 1]),
                          "is_min_eigenvalue": bool(lam_new <= kap_new[j] + 1e-12 * abs(kap_new[j]) and dom_new == j + 1),
                          "status": "derivado (pérdida de estabilidad del hessiano reducido en la dirección latente)"}
                    triggers[j + 1] = ev
                    events.append(ev)
                # habilitación: Y_k cruza ½ hacia arriba
                if (j + 1) not in halves and Yk[j] < 0.5 <= Y_new[j]:
                    w = (0.5 - Yk[j]) / (Y_new[j] - Yk[j])
                    ev = {"kind": f"habilitacion_{j + 1}", "mode": j + 1, "opens": TRAMOS[j + 1], "S": S_int + w * (S_new - S_int),
                          "sigma_hat": sigma + w * h, "status": "derivado (Y_k = ½: media habilitación)"}
                    halves[j + 1] = ev
                    events.append(ev)
            y, st, S_int, f_val, V_prev, sigma = y_new, st_new, S_new, f_new, V_new, sigma + h
            kap, lam_min, Yk = kap_new, lam_new, Y_new
            v1, s1 = vel(st)
            # la velocidad del criterio de parada es la del espacio de estados (φ̂, q̂) en las dos representaciones
            vnorm = math.sqrt(sum(vi * vi for vi in (self._velocity(y)[0] if log_repr else v1)))
            # una dirección sembrada y todavía inestable no ha terminado aunque su velocidad sea diminuta
            pending = any((k.seeds[j] > 0.0 if log_repr else y[2 + j] != 0.0) and kap[j] <= 0.0 for j in range(N_MODES))
            finished = vnorm < k.v_tol * v_scale and not pending
            if n % k.record_every == 0 or finished:
                record(s1)
            if finished:
                stop_reason = "descenso completado (velocidad proyectada ≈ 0 y ninguna dirección sembrada inestable)"
                break
        if not finished and stop_reason is None:
            stop_reason = "max_sigma_hat o max_steps agotado"
        collapses = []
        for j in (1, 2, 3):
            if j in triggers and j in halves:
                gens = chain_generators(j)
                collapses.append({"kind": f"colapso_{j}D", "opens": TRAMOS[j], "S": float(triggers[j]["S"]),
                                  "S_half_enabling": float(halves[j]["S"]), "sigma_hat": float(triggers[j]["sigma_hat"]),
                                  "sigma_hat_half_enabling": float(halves[j]["sigma_hat"]),
                                  "is_min_eigenvalue": triggers[j]["is_min_eigenvalue"],
                                  "algebra": f"C({j + 1},0)", "dim_rep": int(gens[0].shape[0]),
                                  "anticommute": bool(all_anticommute(gens)), "signature": signature(gens),
                                  "status": "derivado en el modelo efectivo (gatillo y habilitación); sin letra sobre λ"})
        S_trig = [cc["S"] for cc in collapses]
        # una dirección superior no puede abrirse antes que las inferiores: orden estricto en σ̂ y sin huecos
        ordered = (all(collapses[i]["sigma_hat"] < collapses[i + 1]["sigma_hat"] for i in range(len(collapses) - 1))
                   and [cc["kind"] for cc in collapses] == [f"colapso_{j}D" for j in range(1, len(collapses) + 1)])
        rec_np = {kk: np.asarray(vv) for kk, vv in rec.items()}
        g_end = self.grad_hat(y)
        v_end, _ = self._velocity(y)
        w_end = np.linalg.eigvalsh(self.reduced_hessian(y))
        thr = C.decade_thresholds()[:3]

        def at_S(arr, S_target):
            S_arr = rec_np["S"]
            return float(np.interp(S_target, S_arr, arr)) if S_arr.size > 1 and S_arr[-1] >= S_target else None

        checks = {
            "monotonia_4_5": {"max_dV_over_T_total": float(max_dV / T_total) if n else 0.0,
                              "pass": bool(n == 0 or max_dV <= 1e-12 * T_total),
                              "note": "un solo potencial y acoplos fijos: V es función de Lyapunov; la comprobación queda activa",
                              "status": "derivado"},
            "produccion_entropica_4_5": {"min": float(min_sprod) if n else 0.0, "pass": bool(n == 0 or min_sprod >= 0.0),
                                         "status": "derivado"},
            "S_equals_f_identity": {"max_abs_diff": float(max_id_diff), "pass": bool(max_id_diff < 1e-6),
                                    "W_floor_over_T_total": float(W_floor / T_total),
                                    "note": "Σ̇ = −dV/dσ ⟹ S = f exactamente en el flujo de gradiente; con suelo de ruido, "
                                            "S = f + W_suelo/T_total",
                                    "status": "derivado"},
            "secuencia_jerarquica": {"order": [cc["kind"] for cc in collapses], "pass": bool(ordered),
                                     "note": "V1D, V2D y V3D en este orden: cada compuerta depende del tramo anterior",
                                     "status": "derivado"},
            "gatillo_es_autovalor_minimo": {"per_collapse": [cc["is_min_eigenvalue"] for cc in collapses],
                                            "pass": bool(all(cc["is_min_eigenvalue"] for cc in collapses)) if collapses else None,
                                            "status": "derivado"},
            "estado_final": {"finished": bool(finished), "grad_norm_projected_over_scale": float(math.sqrt(sum(vi * vi for vi in v_end)) * self.G / v_scale),
                             "grad_norm_full_over_scale": float(math.sqrt(sum(gi * gi for gi in g_end)) / v_scale),
                             "reduced_hessian_min_eigenvalue": float(w_end.min()),
                             "stable": bool(w_end.min() > 0.0),
                             "V_final_minus_V_min_over_T_total": float((V_prev - self.V_min) / T_total),
                             "pass": bool(finished), "status": "derivado"},
            "sellos_como_asentamientos": {"Y1_at_S_0.099": at_S(rec_np["Y1"], thr[1]), "Y2_at_S_0.999": at_S(rec_np["Y2"], thr[2]),
                                          "Y_final": [float(v) for v in Yk], "Y0_final": float((y[0] ** 2 + y[1] ** 2) / self.x_plus_hat),
                                          "note": "c en S_0,099 y c² en S_0,999 aparecerían como asentamientos de q_1 y q_2; el "
                                                  "asentamiento no es exacto (retroacción del tramo siguiente)",
                                          "status": "publicado"},
        }
        cfg_out = {**{kk: vv for kk, vv in asdict(k).items() if kk != "clock"}, "clock": {**asdict(c), "fp": asdict(c.fp)}}
        return to_jsonable({
            "config": cfg_out, "status": STATUS, "mode": "cascade", "declared_forms": DECLARED_FORMS,
            "delta0": c.delta0, "T0_basal_hat": self.T0_hat, "T_ref_hat": self.T_ref_hat, "T_total_hat": T_total,
            "T_total_nominal_hat": self.T_nominal_hat,
            "shares": list(self.shares), "nominal_thresholds": cumulative_thresholds(self.shares),
            "decade_thresholds": thr, "depth_hat": self.depth_hat, "W": self.W,
            "x_esc_hat": self.x_esc_hat, "x_plus_hat": self.x_plus_hat, "y_min": self.y_min,
            "trajectory": {kk: vv.tolist() for kk, vv in rec_np.items()},
            "events": events, "collapses": collapses, "S_triggers": S_trig,
            "S_half_enabling": [cc["S_half_enabling"] for cc in collapses], "ratios": ratios(S_trig),
            "n_triggers": len(triggers), "n_collapses": len(collapses),
            "checks": checks, "steps": n, "finished": bool(finished), "stop_reason": stop_reason,
            "h_max": h_max, "h_min_used": h_min_used, "underflow_modes": underflow,
            "amplitude_representation": k.amplitude_representation,
            "sigma_hat_final": sigma, "S_final": S_int, "f_final": f_val})


def run_cascade(delta0: float = 0.01, clock_kwargs: dict | None = None, **kwargs) -> dict:
    """Atajo: un recorrido del modo cascada con el reloj por defecto en δ0 (rayo θ = 0, punto de escape, acoplos fijos)
    y la ley de escala del canon. `clock_kwargs` pasa entradas al Basal del reloj; `kwargs`, al modelo efectivo."""
    return CascadeClock(CascadeConfig(clock=ClockConfig(delta0=delta0, **(clock_kwargs or {})), **kwargs)).run()
