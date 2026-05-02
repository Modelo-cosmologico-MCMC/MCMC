#!/usr/bin/env python
"""Lanza un run de CLASS modificado con los parámetros nominales del MCMC."""

from __future__ import annotations

from cosmology.class_wrapper import mcmc_class_params, write_ini


def main() -> None:
    p = mcmc_class_params()
    path = write_ini("output/class_mcmc.ini", p)
    print(f"Parámetros CLASS escritos en {path}")
    try:
        from cosmology.class_wrapper import run_class
        cosmo = run_class(p)
        print("CLASS ejecutado:", cosmo)
    except ImportError as e:
        print(f"[skip] {e}")


if __name__ == "__main__":
    main()
