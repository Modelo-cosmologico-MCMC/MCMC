"""La vecindad solar como contraste de la Ley de Cronos débil: el término
de Cronos como densidad dinámica EFECTIVA (frente 5c, «Oort–K_z»).

La fuerza local +c²∇ε_c (Cor. 11.3) tiene, en una losa estratificada
ρ(z), componente vertical g_C(z) = −c²·dε_c/dz = −(3/2)c²Aρ^{1/2}ρ'(z),
dirigida hacia el plano como la gravedad. Una estrella trazadora no
distingue su origen: la cinemática vertical mide K_z,tot = K_z,N + g_C y
la lee como densidad dinámica. Por tanto el término de Cronos aparece
en el límite de Oort como

    Δρ_eff(0) = (1/4πG)·dg_C/dz|₀ = −(3/2)c²A·ρ(0)^{1/2}·ρ''(0)/(4πG)

y en la columna hasta z como ΔΣ_eff(z) = g_C(z)/(2πG). La observación
deja un MARGEN: ρ_dyn(0) − ρ_bar(0) y Σ_dyn(1.1) − Σ_bar(1.1), que Cronos
comparte con la materia oscura local; la cota publicada le concede todo
el margen (la más laxa). La losa ρ(z) es DECLARADA (formas sech² con
las densidades centrales de McKee, Parravano & Hollenbach 2015 y alturas
de escala declaradas); su sensibilidad se publica variando las alturas
×½ y ×2, porque Δρ_eff(0) ∝ ρ''(0) ∝ h⁻².

Estatuto: cálculo de consistencia (E8) que consume la tabla transcrita
`local_kz_bounds` (bytes oficiales no verificados: aviso propagado);
ningún parámetro se ajusta; A_Sculptor no se toca.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dynamics.sculptor_transfer import frozen_config
from dynamics.weak_field import C_KMS, G_PC
from mcmc_ontology.data_registry import require_available

A_SCULPTOR = frozen_config().A_sculptor

# losa DECLARADA (densidades centrales MPH2015; alturas de escala del
# criterio de Cronos–Jeans, dynamics.cronos_jeans.solar_neighbourhood_table)
SLAB_DECLARED = {"rho_star_0": 0.043, "h_star_pc": 600.0, "rho_gas_0": 0.041, "h_gas_pc": 250.0, "rho_dm": 0.013}


def load_bounds() -> dict:
    """La tabla transcrita (falla cerrado si el dataset no está AVAILABLE)."""
    raw = require_available("local_kz_bounds")
    files = sorted(Path(raw).glob("*.json"))
    if not files:
        raise FileNotFoundError("DATA_UNAVAILABLE: local_kz_bounds sin fichero JSON")
    return json.loads(files[0].read_text(encoding="utf-8"))


def slab_density(z_pc, slab: dict = SLAB_DECLARED, h_factor: float = 1.0):
    z = np.asarray(z_pc, dtype=float)
    return (slab["rho_star_0"] / np.cosh(z / (slab["h_star_pc"] * h_factor)) ** 2
            + slab["rho_gas_0"] / np.cosh(z / (slab["h_gas_pc"] * h_factor)) ** 2 + slab["rho_dm"])


def cronos_increments(A: float, slab: dict = SLAB_DECLARED, h_factor: float = 1.0, z_max_pc: float = 1100.0) -> dict:
    """Δρ_eff(0) [M☉/pc³] y ΔΣ_eff(z_max) [M☉/pc²] del término de Cronos
    con amplitud A sobre la losa declarada (alturas × h_factor)."""
    z = np.linspace(0.0, 1500.0, 30001)
    rho = slab_density(z, slab, h_factor)
    eps = A * rho ** 1.5
    gC = -C_KMS ** 2 * np.gradient(eps, z)              # (km/s)²/pc, hacia el plano (> 0 aquí por el signo de ρ')
    rho2 = -(2.0 * slab["rho_star_0"] / (slab["h_star_pc"] * h_factor) ** 2
             + 2.0 * slab["rho_gas_0"] / (slab["h_gas_pc"] * h_factor) ** 2)   # ρ''(0)
    d_rho_eff = -1.5 * C_KMS ** 2 * A * np.sqrt(rho[0]) * rho2 / (4.0 * np.pi * G_PC)
    i = int(np.searchsorted(z, z_max_pc))
    d_sigma_eff = float(gC[i] / (2.0 * np.pi * G_PC))
    return {"A": A, "h_factor": h_factor, "d_rho_eff_0_msun_pc3": float(d_rho_eff),
            "d_sigma_eff_msun_pc2": d_sigma_eff, "z_max_pc": z_max_pc,
            "linear_in_A": True}


def room(bounds: dict) -> dict:
    """El margen observacional (valor ± error en cuadratura) en el plano
    (media ponderada de HF2000 y MPH2015 menos bariones MPH2015) y en la
    columna a 1.1 kpc (BT2012 menos bariones MPH2015)."""
    v = bounds["values"]
    r1, r2 = v["rho_dyn_0_HF2000"], v["rho_dyn_0_MPH2015"]
    w1, w2 = 1.0 / r1["err"] ** 2, 1.0 / r2["err"] ** 2
    rho_dyn = (w1 * r1["value"] + w2 * r2["value"]) / (w1 + w2)
    rho_dyn_err = (w1 + w2) ** -0.5
    rb = v["rho_bar_0_MPH2015"]
    s, sb = v["Sigma_1p1_BT2012"], v["Sigma_bar_1p1_MPH2015"]
    return {"rho_dyn_0": rho_dyn, "rho_dyn_0_err": rho_dyn_err,
            "rho_room_0": rho_dyn - rb["value"], "rho_room_0_err": float(np.hypot(rho_dyn_err, rb["err"])),
            "Sigma_room_1p1": s["value"] - sb["value"], "Sigma_room_1p1_err": float(np.hypot(s["err"], sb["err"]))}


def significance(A: float, bounds: dict, h_factor: float = 1.0) -> dict:
    """Nº de σ en que el incremento de Cronos excede el margen (z-score
    del exceso; ≤ 0 si cabe dentro del margen central) en el plano y en la
    columna, y la A máxima compatible a 2σ (lineal en A)."""
    inc = cronos_increments(A, h_factor=h_factor)
    rm = room(bounds)
    z_rho = (inc["d_rho_eff_0_msun_pc3"] - rm["rho_room_0"]) / rm["rho_room_0_err"]
    z_sig = (inc["d_sigma_eff_msun_pc2"] - rm["Sigma_room_1p1"]) / rm["Sigma_room_1p1_err"]
    A_2s_rho = A * (rm["rho_room_0"] + 2.0 * rm["rho_room_0_err"]) / inc["d_rho_eff_0_msun_pc3"]
    A_2s_sig = A * (rm["Sigma_room_1p1"] + 2.0 * rm["Sigma_room_1p1_err"]) / inc["d_sigma_eff_msun_pc2"]
    return {"A": A, "h_factor": h_factor, "increments": inc, "room": rm,
            "z_rho_0": float(z_rho), "z_Sigma_1p1": float(z_sig), "z_max": float(max(z_rho, z_sig)),
            "A_2sigma_rho_0": float(A_2s_rho), "A_2sigma_Sigma_1p1": float(A_2s_sig),
            "A_2sigma_over_A_sculptor": float(min(A_2s_rho, A_2s_sig) / A_SCULPTOR)}
