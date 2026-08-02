"""Simulación QuTiP del qudit MCMC (Tratado de Fundamentos v35, C.1-C.5).

La tabla EXPECTED_FIDELITIES contiene VALORES ESPERADOS DEL TRATADO
(v35 C.4: mcsolve, 10^4 trayectorias) — no salidas verificadas de este
código: sin QuTiP instalado, `simulate_transitions` devuelve la tabla
tal cual.

| Transición  | Fidelidad (v35, C.4) |
|-------------|----------------------|
| S0 → S1     | 0.981                |
| S1 → S2     | 0.976                |
| S2 → S3     | 0.968                |
| S3 → S4     | 0.961                |

Nota de versión: el corpus v32 tabulaba 0.981/0.975/0.969/0.958 con una
fidelidad global de 0.939; la v35 (C.4) publica 0.981/0.976/0.968/0.961
y no declara valor global. Este módulo usa los valores vigentes.

Predicción falsable del tratado (C.4-C.5): patrón decreciente de
fidelidades F0 > F1 > F2 > F3 — las transiciones de mayor n sufren más
decoherencia tensional por el factor η_void(Sn) — con la máxima
superposición en la fase intermedia de la conversión Mp → Ep.
"""

from __future__ import annotations

EXPECTED_FIDELITIES = {
    "S0->S1": 0.981,
    "S1->S2": 0.976,
    "S2->S3": 0.968,
    "S3->S4": 0.961,
}

HARDWARE = {
    "IonQ_Aria": {
        "platform": "5 niveles en 1 ion 171Yb+",
        "T_circ_us": 40.0,
        "F_min": 0.96,
    },
    "IBM_Eagle": {
        "platform": "5 niveles en 3 qubits transmon",
        "T_circ_us": 8.0,
        "F_min": 0.88,
    },
}


def simulate_transitions(n_traj: int = 10_000, seed: int = 0) -> dict:
    """Simula las transiciones del qudit usando QuTiP si está disponible.

    Si QuTiP no está disponible, devuelve los valores esperados tabulados.
    """
    try:
        import numpy as np
        import qutip as qt  # type: ignore

        from .hamiltonian import H_MCMC

        H = qt.Qobj(H_MCMC())
        psi0 = qt.basis(5, 0)
        # Evolución unitaria libre durante un periodo de transición
        T = 1.0
        tlist = np.linspace(0, T, 100)
        result = qt.mesolve(H, psi0, tlist)
        # Fidelidades aproximadas con cada base
        fids = {}
        for n in range(1, 5):
            target = qt.basis(5, n)
            psi_T = result.states[-1]
            f = qt.fidelity(target, psi_T) ** 2
            fids[f"S0->S{n}"] = float(f)
        return fids
    except ImportError:
        return dict(EXPECTED_FIDELITIES)
