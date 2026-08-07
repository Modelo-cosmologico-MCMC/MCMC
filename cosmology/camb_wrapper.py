"""Interfaz mínima a CAMB modificado para MCMC."""

from __future__ import annotations

from mcmc_ontology import constants as C


def mcmc_camb_params(H0: float = C.H0_MCMC,
                     Omega_m: float = 0.300,
                     eps: float = C.EPSILON_LAMBDA,
                     z_trans: float = C.Z_TRANS) -> dict:
    """Parámetros para un run CAMB con Λ_rel(z)."""
    return {
        "H0": H0,
        "ombh2": 0.0489 * (H0 / 100.0) ** 2,
        "omch2": (Omega_m - 0.0489) * (H0 / 100.0) ** 2,
        "ns": 0.965,
        "sigma8_target": C.SIGMA8_MCMC,
        "mcmc_eps": eps,
        "mcmc_z_trans": z_trans,
    }


def run_camb(params: dict | None = None):
    """Ejecuta CAMB si está disponible."""
    try:
        import camb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "El paquete `camb` no está instalado. `pip install camb`."
        ) from exc
    p = params or mcmc_camb_params()
    pars = camb.set_params(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], ns=p["ns"])
    return camb.get_results(pars)
