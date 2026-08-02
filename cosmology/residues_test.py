"""El contraste de los Residuos — frente nº 6 (v35, Conj. 9.6 y H.2.5).

EL ÚNICO FRENTE DONDE EL MODELO YA PASÓ UN CONTRASTE OBSERVACIONAL REAL.

Predicción (Conjetura 9.6, ec. 9.5): si el residuo del sello ε_K ≡
λK(S_act) − 1 es del orden del residuo de descarga ε = 0.012 del ajuste,

    G_cosmo/G_N − 1 ≃ −(3/2)·ε_K ≈ −1.8%

Cota observacional (H.2.5): la nucleosíntesis primordial da

    G_BBN/G_0 = 0.99 +0.06/−0.05  (a 2σ)

    Alvey, Sabti, Escudero & Fairbairn (2020), «Improved BBN constraints
    on the variation of the gravitational constant», Eur. Phys. J. C 80,
    148; arXiv:1910.10730.

La predicción (ratio 0.982) cae holgadamente dentro de la cota, a ~0.3σ
del valor central 0.99 — QUE ADEMÁS FAVORECE SU SIGNO (el central está
por debajo de 1, en la misma dirección). El modelo sobrevive a un
contraste real y queda a la espera de medidas más finas.

ESTATUTO: la relación ε_K ≈ ε es la Conjetura 9.6 (condicional); el
contraste de este módulo es real pero no la demuestra — la acota.
"""

from __future__ import annotations

from mcmc_ontology import constants as C

# Cota BBN (Alvey, Sabti, Escudero & Fairbairn 2020; arXiv:1910.10730):
BBN_RATIO_CENTRAL = 0.99
BBN_ERR_UP_2SIGMA = 0.06     # +0.06 a 2σ
BBN_ERR_DOWN_2SIGMA = 0.05   # −0.05 a 2σ
BBN_REFERENCE = ("Alvey, Sabti, Escudero & Fairbairn (2020), "
                 "Eur. Phys. J. C 80, 148; arXiv:1910.10730")


def predicted_ratio(eps_K: float = C.EPSILON_0) -> float:
    """G_cosmo/G_N predicho: 1 − (3/2)·ε_K (ec. 9.5). Con ε_K = 0.012:
    0.982 (−1.8%)."""
    return 1.0 - 1.5 * eps_K


def tension_sigma(eps_K: float = C.EPSILON_0) -> float:
    """Desviación de la predicción respecto del central BBN, en σ.

    Se usa el error del lado correspondiente (asimétrico), convertido de
    2σ a 1σ. Con ε_K = 0.012: |0.982 − 0.99|/0.025 ≈ 0.32σ.
    """
    pred = predicted_ratio(eps_K)
    err_2s = BBN_ERR_DOWN_2SIGMA if pred < BBN_RATIO_CENTRAL \
        else BBN_ERR_UP_2SIGMA
    return abs(pred - BBN_RATIO_CENTRAL) / (err_2s / 2.0)


def within_bbn_bound(eps_K: float = C.EPSILON_0) -> bool:
    """¿Cae la predicción dentro de la cota BBN a 2σ?"""
    pred = predicted_ratio(eps_K)
    return (BBN_RATIO_CENTRAL - BBN_ERR_DOWN_2SIGMA
            <= pred <= BBN_RATIO_CENTRAL + BBN_ERR_UP_2SIGMA)


def sign_favored(eps_K: float = C.EPSILON_0) -> bool:
    """¿Favorece el central BBN el signo de la predicción? Sí cuando el
    central está del mismo lado de 1 que la predicción (H.2.5)."""
    pred = predicted_ratio(eps_K)
    return (pred - 1.0) * (BBN_RATIO_CENTRAL - 1.0) > 0.0


def report(eps_K: float = C.EPSILON_0) -> str:
    """El contraste en una línea, con su cita."""
    return (f"G_cosmo/G_N predicho = {predicted_ratio(eps_K):.3f} "
            f"({100.0 * (predicted_ratio(eps_K) - 1.0):+.1f}%); cota BBN "
            f"{BBN_RATIO_CENTRAL} +{BBN_ERR_UP_2SIGMA}/−{BBN_ERR_DOWN_2SIGMA}"
            f" (2σ): dentro={within_bbn_bound(eps_K)}, "
            f"tensión={tension_sigma(eps_K):.2f}σ, "
            f"signo favorecido={sign_favored(eps_K)} "
            f"[{BBN_REFERENCE}]")
