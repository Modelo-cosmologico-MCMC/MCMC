"""
Simulations - Simulaciones Astrofísicas MCMC
=============================================

Simulaciones N-body con fricción entrópica y perfiles de halos.

Este módulo implementa:
- Fricción entrópica: η(ρ) = α × (ρ/ρc)^1.5
- Perfil de Burkert para halos de materia oscura
- Relación r_core-masa galáctica

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com
"""

from .bloque3_nbody import (
    friccion_entropica,
    perfil_burkert,
    radio_core,
    Particula,
    SimulacionNBody,
    ALPHA_FRICCION,
    RHO_CRITICA,
)

__all__ = [
    "friccion_entropica",
    "perfil_burkert",
    "radio_core",
    "Particula",
    "SimulacionNBody",
    "ALPHA_FRICCION",
    "RHO_CRITICA",
]
