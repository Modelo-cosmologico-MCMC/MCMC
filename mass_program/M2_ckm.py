"""M2 — Mezcla CKM (Ec. 600, Tratado Unificado v32; cascada SO(10)).

Convención base:
    U_d = I  (down diagonal exacto)
    U_u = V_CKM  (up quarks mezclan)

Factor a aplicar en la fórmula maestra de masas:

    V_CKM(n, f) = 1                  si n = n_e(f)        (diagonal, sin supresión)
                = |V_{f, dom(n)}|^2  si n ≠ n_e(f)        (inter-familia, suprimido)

donde dom(n) es el fermión dominante en el sello n (mismo tipo).
"""

from __future__ import annotations

from mcmc_ontology import constants as C


# Fermión dominante por sello y tipo
DOMINANT = {
    "C1": {"up": "t", "down": "b", "lepton": "tau", "nu": "nu_tau"},
    "C2": {"up": "c", "down": "s", "lepton": "mu",  "nu": "nu_mu"},
    "C3": {"up": "u", "down": "d", "lepton": "e",   "nu": "nu_e"},
    "C4": {"up": "u", "down": "d", "lepton": "e",   "nu": "nu_e"},
}

# Sello de emergencia por fermión
EMERGENCE_SEAL = {
    "tau": "C1", "t": "C1", "b": "C1", "nu_tau": "C1",
    "mu":  "C2", "c": "C2", "s": "C2", "nu_mu":  "C2",
    "e":   "C3", "u": "C3", "d": "C3", "nu_e":   "C3",
}


def V_CKM_factor(seal: str, fermion: str) -> float:
    """V_CKM(n, f) — factor de mezcla a aplicar en la fórmula maestra.

    · Si n es el sello de emergencia de f → 1 (diagonal).
    · Si los fermiones son leptones o neutrinos → 1 (sin mezcla CKM).
    · Si los fermiones son quarks → |V_{f, dom(n)}|^2 (inter-familia).
    """
    if EMERGENCE_SEAL[fermion] == seal:
        return 1.0
    # Sin CKM para leptones cargados ni neutrinos en este modelo
    is_quark = fermion in {"t", "b", "c", "s", "u", "d"}
    if not is_quark:
        return 1.0
    f_type = "up" if fermion in {"t", "c", "u"} else "down"
    dom = DOMINANT[seal][f_type]
    key = tuple(sorted((fermion, dom)))
    # Reordenar a (up, down) para usar tabla CKM
    if fermion in {"t", "c", "u"}:
        u_q, d_q = fermion, dom
    elif dom in {"t", "c", "u"}:
        u_q, d_q = dom, fermion
    else:
        # Down-down (no físico en CKM); devolver 1 por convención
        return 1.0
    val = C.CKM_SQ.get((u_q, d_q))
    if val is None:
        # Si la combinación no está tabulada (mismo tipo), no aplica
        return 1.0
    return float(val)
