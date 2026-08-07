"""Interfaz mínima al código CLASS modificado para MCMC.

Si la librería `classy` no está disponible, ofrece una capa de
compatibilidad que escribe ficheros .ini reproducibles para uso
posterior (CLASS standalone).
"""

from __future__ import annotations

from pathlib import Path

from mcmc_ontology import constants as C


def mcmc_class_params(H0: float = C.H0_MCMC,
                      Omega_m: float = 0.300,
                      eps: float = C.EPSILON_LAMBDA,
                      z_trans: float = C.Z_TRANS) -> dict:
    """Parámetros para un run CLASS con Λ_rel(z) y growth modificado."""
    return {
        "h": H0 / 100.0,
        "Omega_m": Omega_m,
        "Omega_b": 0.0489,
        "n_s": 0.965,
        "sigma8": C.SIGMA8_MCMC,
        # Extensión MCMC: parametrización de Λ_rel
        "mcmc_eps": eps,
        "mcmc_z_trans": z_trans,
        # Hooks:  S, ε aplicados en el módulo `background.c` modificado
        "output": "tCl,pCl,lCl,mPk",
        "P_k_max_h/Mpc": 10.0,
        "lensing": "yes",
    }


def write_ini(path: str | Path, params: dict | None = None) -> Path:
    """Escribe un fichero .ini para CLASS con los parámetros MCMC."""
    if params is None:
        params = mcmc_class_params()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k} = {v}" for k, v in params.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def run_class(params: dict | None = None):
    """Ejecuta CLASS si `classy` está instalado; en caso contrario, lanza ImportError."""
    try:
        from classy import Class  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "El paquete `classy` no está instalado. Para ejecutar CLASS "
            "compila CLASS modificado y `pip install classy`."
        ) from exc
    cosmo = Class()
    cosmo.set(params or mcmc_class_params())
    cosmo.compute()
    return cosmo
