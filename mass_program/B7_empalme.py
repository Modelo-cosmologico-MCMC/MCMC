"""B7 — El empalme C¹ en el sello: β3 desde la curvatura sellada (H.2.4).

LA RELACIÓN NO CIRCULAR. La continuidad de V y V' al identificarse
ΦAd → ΦH impone (ec. H.8):

    m_H² = V_H''(v) = 8·β3·v²   ⟹   β3 = m_H²/(8v²) = ¼·λ_H

y «la curvatura sellada del tramo previo V_Ad''(Φ*) determina m_H, no al
revés (en la normalización |Φ|⁴, λ_H = m_H²/2v² = 0.130)».

AVISO DE NORMALIZACIÓN (dos convenios en el tratado, consistentes):
  - Convenio de (12.1): m_H = √(2β3)·v con β3 ≡ λ_H = 0.130 — es el de
    constants.BETA["C3"] y B5_higgs. CANÓNICO en este repositorio.
  - Convenio de (H.8): β3 ≡ λ_H/4 ≈ 0.0325 (de m_H² = 8β3v²).
  Ambos se exponen explícitamente para que ningún resultado parezca
  discrepar por un factor 4 sin estarlo.

EL CÁLCULO SIN m_H COMO ENTRADA. La curvatura del Basal en el anillo de
vacíos, en la normalización |Φ|⁴:

    λ_Ad ≡ V_Ad''(ρ+)/(2·ρ+²)

se calcula desde core.basal (numérica y analíticamente: λ_Ad = √D =
δ0·√(b̄² − 4C0·m̄²)) — la masa medida del Higgs NO entra en ninguna
parte del cálculo.

RESULTADO (el desenlace se publica sea cual sea, y es este):
con las formas fiduciales O(1) (m̄=1, b̄=3, C0=1) el cierre numérico
λ_Ad = 0.130 exige δ0 ≈ 0.0581 — NO el 0.012 de ε_Λ que el v32
identificaba con δ0 (identificación superada — regla canónica), con el
que λ_Ad ≈ 0.0268 (m_H ≈ 57 GeV). El frente queda
ACOTADO cuantitativamente: la relación de empalme es no circular y está
implementada; el valor numérico de β3 con (M0², B, C0) sellados sigue
siendo CONDICIONAL, exactamente como declara el Estado de H.2.4. La
auditoría de circularidad de B5_higgs (Obs. 12.2) se conserva.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from core.basal import V0, kappa_plus, M_BAR, B_BAR, C0_DEFAULT

# Los dos convenios, explícitos:
BETA3_CONVENIO_12_1 = 0.130          # β3 ≡ λ_H  (ec. 12.1; canónico aquí)
BETA3_CONVENIO_H8 = 0.130 / 4.0      # β3 ≡ λ_H/4 (ec. H.8)


def lambda_H_measured() -> float:
    """λ_H = m_H²/(2v²) desde el valor MEDIDO (PDG) — solo para el
    CONTRASTE final, jamás como entrada del cálculo."""
    return C.M_HIGGS_PDG ** 2 / (2.0 * C.V3_GEV ** 2)


def sealed_curvature_lambda(delta0: float, m_bar: float = M_BAR,
                            b_bar: float = B_BAR,
                            C0: float = C0_DEFAULT,
                            numeric: bool = False) -> float:
    """λ_Ad = V_Ad''(ρ+)/(2ρ+²) — la curvatura sellada del tramo previo,
    SIN la masa del Higgs como entrada (H.2.4).

    Analítica: λ_Ad = √D = δ0·√(b̄²−4C0m̄²). Con numeric=True se calcula
    por diferencias finitas sobre core.basal.V0 (verificación cruzada).
    """
    if not numeric:
        disc = b_bar ** 2 - 4.0 * C0 * m_bar ** 2
        if disc < 0.0:
            raise ValueError("Sin vacío sellado: D < 0")
        return delta0 * float(np.sqrt(disc))
    rho_p = float(np.sqrt(kappa_plus(m_bar, b_bar, C0) * delta0))
    h = 1e-6 * rho_p
    vpp = (V0(rho_p + h, 0.0, delta0, m_bar, b_bar, 0.0, C0)
           - 2.0 * V0(rho_p, 0.0, delta0, m_bar, b_bar, 0.0, C0)
           + V0(rho_p - h, 0.0, delta0, m_bar, b_bar, 0.0, C0)) / h ** 2
    return float(vpp / (2.0 * rho_p ** 2))


def beta3_derived(delta0: float, convention: str = "12.1",
                  m_bar: float = M_BAR, b_bar: float = B_BAR,
                  C0: float = C0_DEFAULT) -> float:
    """β3 derivado del empalme, en el convenio pedido ('12.1' o 'H.8')."""
    lam = sealed_curvature_lambda(delta0, m_bar, b_bar, C0)
    if convention == "12.1":
        return lam
    if convention == "H.8":
        return lam / 4.0
    raise ValueError("convention: '12.1' o 'H.8'")


def m_H_predicted(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                  C0: float = C0_DEFAULT) -> float:
    """m_H = √(2·λ_Ad)·v3 [GeV] — la predicción del empalme, sin m_H
    medido como entrada."""
    lam = sealed_curvature_lambda(delta0, m_bar, b_bar, C0)
    return float(np.sqrt(2.0 * lam) * C.V3_GEV)


def delta0_required(target_lambda: float = BETA3_CONVENIO_12_1,
                    m_bar: float = M_BAR, b_bar: float = B_BAR,
                    C0: float = C0_DEFAULT) -> float:
    """El δ0 que el cierre exige: λ_Ad(δ0) = target ⟹ δ0* =
    target/√(b̄²−4C0m̄²). Con las formas fiduciales: δ0* ≈ 0.0581."""
    disc = b_bar ** 2 - 4.0 * C0 * m_bar ** 2
    return target_lambda / float(np.sqrt(disc))


def audit_report(delta0_cosmo: float = C.EPSILON_0) -> str:
    """El desenlace del empalme, cuantificado y publicable tal cual."""
    lam_at_eps = sealed_curvature_lambda(delta0_cosmo)
    d0_star = delta0_required()
    return (
        f"Empalme C¹ (H.8), sin m_H como entrada: λ_Ad(δ0={delta0_cosmo}) = "
        f"{lam_at_eps:.4f} → m_H = {m_H_predicted(delta0_cosmo):.1f} GeV "
        f"(≠ 125.3). El cierre λ_Ad = 0.130 exige δ0* = {d0_star:.4f} con "
        "las formas fiduciales O(1). CONCLUSIÓN: relación no circular "
        "implementada; el valor numérico de β3 con (M0²,B,C0) sellados "
        "sigue CONDICIONAL (Estado de H.2.4) — el frente queda acotado: "
        "o δ0 ≈ 0.058 en el sello, o formas no fiduciales. La auditoría "
        "de la Obs. 12.2 se conserva."
    )
