"""
Lattice - Teoría de Gauge en la Red MCMC
=========================================

Implementación de Yang-Mills en la red para el modelo MCMC.

El acoplamiento gauge depende del sello entrópico S:
    β(S) = β0 + β1 × exp[-bS × (S - S3)]

Esto conecta la ontología MCMC con QCD y el problema del mass gap.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com
"""

from .bloque4_ym_lattice import (
    beta_MCMC,
    mass_gap,
    accion_wilson,
    LatticeYM,
    su2_elemento_aleatorio,
    BETA_0,
    BETA_1,
    B_S,
    LAMBDA_QCD,
    ALPHA_H,
)

__all__ = [
    "beta_MCMC",
    "mass_gap",
    "accion_wilson",
    "LatticeYM",
    "su2_elemento_aleatorio",
    "BETA_0",
    "BETA_1",
    "B_S",
    "LAMBDA_QCD",
    "ALPHA_H",
]
