"""III.B — Boltzmann en variable S (scaffold).

Tratado §3.19.8 Etapa III. Reescribe el esquema Boltzmann usando S
como variable de integración mediante la Ley de Cronos.

ESTADO: scaffold. La integración completa requiere CLASS/CAMB modificados;
este módulo provee la API y las ecuaciones operativas (κ_lat, Γ_lat,
ρ_lat(S)) que conectan con el Boltzmann externo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import odeint

from mcmc_ontology import constants as C


@dataclass
class BoltzmannS:
    """Pipeline Boltzmann en variable entrópica S.

    Atributos:
        epsilon, z_trans, dz_trans: parámetros de Λ_rel(z).
        alpha_lat, beta_lat, gamma_lat: tasas de conversión latente
            (Ecs. 48-52 del Tratado revisado).
    """
    epsilon:    float = C.EPSILON_0
    z_trans:    float = C.Z_TRANS
    dz_trans:   float = 1.5
    alpha_lat:  float = 0.05
    beta_lat:   float = 0.01
    gamma_lat:  float = 0.005
    S_seals:    list[float] = field(default_factory=lambda: [0.009, 0.099, 0.999, 1.001])
    lambda_ont: float = C.LAMBDA_ONT

    def kappa_lat(self, S: float) -> float:
        """Tasa de decaimiento latente κ_lat(S) (Ec. 54).

        κ_lat(S) ∝ Σ_n Θ'_λ(S - S_n) · U_n / E_*(S)
        """
        total = 0.0
        for sn in self.S_seals:
            arg = (S - sn) / self.lambda_ont
            if abs(arg) > 350.0:
                continue  # cosh overflow → contribución despreciable
            total += (0.5 / self.lambda_ont) / np.cosh(arg) ** 2
        return float(self.alpha_lat * total)

    def Gamma_lat(self, S: float, rho) -> float:
        """Fuente Γ_lat(S, ρ) (Ec. 55, simplificada)."""
        rho_scalar = float(np.asarray(rho).item() if hasattr(rho, "shape")
                           else rho)
        return self.beta_lat * rho_scalar

    def rho_lat_evolution(self, S_arr: np.ndarray, rho_lat_0: float = 0.05) -> np.ndarray:
        """Evolución ρ_lat(S) (Ec. 55):

            dρ_lat/dS = -κ_lat(S) ρ_lat + Γ_lat(S, ρ_lat)
        """
        def drho(rho, S):
            return -self.kappa_lat(S) * rho + self.Gamma_lat(S, rho)

        sol = odeint(drho, rho_lat_0, np.asarray(S_arr, dtype=float))
        return sol[:, 0]

    def transfer_function(self, k: np.ndarray, z_arr: np.ndarray):
        """Función de transferencia T(k, z). Requiere CLASS/CAMB modificado."""
        raise NotImplementedError(
            "Integración Boltzmann completa requiere CLASS/CAMB modificado. "
            "Use cosmology.class_wrapper o camb_wrapper para producción."
        )
