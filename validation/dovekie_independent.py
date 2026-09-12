"""Implementación B — verificador independiente del pipeline SN Dovekie.

Patrón del crosscheck #14 (validation/jax_background.py): una SEGUNDA
implementación de cada pieza del cálculo, sin imports compartidos con
la ruta de producción (cosmology/dovekie_sn.py) en el cálculo del χ²:

  - parser propio del HD (línea a línea, sin _parse_hd),
  - desempaquetado propio del npz (índices triangulares derivados por
    bucle explícito, no np.triu_indices),
  - E(z) reimplementada desde la fórmula declarada (Ap. A.2/A.3 con
    clausura plana por llamada y transición tanh normalizada hoy),
  - D_M por cuadratura adaptativa (scipy.integrate.quad) punto a punto,
  - χ̃² con M marginalizada por la fórmula A9-A12 escrita de forma
    distinta (sumas explícitas vía einsum).

Este módulo NO es ruta de producción: existe para las puertas 1a/1b de
la preinscripción del PR #15. PROHIBICIÓN (vigilada por el candado):
jamás importa los módulos de producción cosmology/dovekie_sn ni
cosmology/background.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent.parent
C_KMS = 299792.458
H0_FIXED = 70.0
OMEGA_R0 = 9.2e-5          # el mismo valor declarado del fondo (Ap. A)


# ---------------------------------------------------------------------
# Parser independiente del HD
# ---------------------------------------------------------------------

def parse_hd_independent(path: Path) -> dict:
    """Parser propio del formato SNANA (VARNAMES + filas separadas por
    espacios), con el corte oficial zHD > 0 aplicado al final.

    BARRERA PR #15: en esta fase el parser devuelve SOLO el diseño
    (redshifts y conteo) — la columna MU real no sale de aquí hasta el
    PASS de la validación por mocks (la extensión post-PASS añadirá la
    lectura del dato para el crosscheck del ajuste real)."""
    header: list[str] = []
    data: list[dict] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line[0] == "#":
            continue
        parts = line.split()
        if parts[0] == "VARNAMES:":
            header = parts[1:]
            continue
        if parts[0] != "SN:":       # solo filas con clave SNANA
            continue
        data.append(dict(zip(header, parts[1:])))
    zhd = np.array([float(r["zHD"]) for r in data])
    zhel = np.array([float(r["zHEL"]) for r in data])
    sel = zhd > 0.0
    return {"zHD": zhd[sel], "zHEL": zhel[sel],
            "n_sn": int(np.count_nonzero(sel))}


def unpack_inv_cov_independent(path: Path) -> np.ndarray:
    """Desempaquetado propio: recorre la triangular superior aplanada
    fila a fila (bucle explícito de índices, sin np.triu_indices)."""
    d = np.load(path)
    n = int(np.asarray(d[d.files[0]]).ravel()[0])
    flat = np.asarray(d[d.files[1]], dtype=float)
    if flat.size != n * (n + 1) // 2:
        raise ValueError("longitud triangular inconsistente con nsn")
    W = np.empty((n, n))
    k = 0
    for i in range(n):
        m = n - i
        W[i, i:] = flat[k:k + m]
        W[i:, i] = flat[k:k + m]
        k += m
    return W


# ---------------------------------------------------------------------
# Fondo y distancias independientes
# ---------------------------------------------------------------------

def E_independent(z: float, Omega_m: float, eps: float,
                  z_trans: float, dz: float = 1.5) -> float:
    """E(z) reescrita desde la fórmula declarada: clausura plana por
    llamada (Ω_DE,0 = 1 − Ω_m − Ω_r) y transición tanh normalizada en
    z = 0 (Ap. A.2/A.3, corrección ago-2026)."""
    F = 1.0 + eps * math.tanh((z_trans - z) / dz)
    F0 = 1.0 + eps * math.tanh(z_trans / dz)
    Omega_DE0 = 1.0 - Omega_m - OMEGA_R0
    arg = (Omega_m * (1.0 + z) ** 3 + OMEGA_R0 * (1.0 + z) ** 4
           + Omega_DE0 * F / F0)
    return math.sqrt(arg)


def mu_model_independent(zHD: np.ndarray, zHEL: np.ndarray,
                         Omega_m: float, eps: float = 0.0,
                         z_trans: float = 8.9) -> np.ndarray:
    """μ teórica con D_M por cuadratura adaptativa punto a punto
    (misma convención oficial: μ = 5·log10[(1+zHEL)·D_M(zHD)] + 25)."""
    zHD = np.asarray(zHD, float)
    zHEL = np.asarray(zHEL, float)
    # cuadratura por z único (los HD repiten pocos z; el orden se
    # restaura por índice inverso)
    z_unique, inv = np.unique(zHD, return_inverse=True)
    dm_unique = np.empty(len(z_unique))
    for i, zi in enumerate(z_unique):
        val, _ = quad(lambda x: 1.0 / E_independent(x, Omega_m, eps,
                                                    z_trans),
                      0.0, float(zi), epsabs=1e-12, epsrel=1e-12,
                      limit=300)
        dm_unique[i] = (C_KMS / H0_FIXED) * val
    DM = dm_unique[inv]
    return 5.0 * np.log10((1.0 + zHEL) * DM) + 25.0


def chi2_marginalized_independent(mu_mod: np.ndarray,
                                  mu_data: np.ndarray,
                                  W: np.ndarray) -> float:
    """χ̃² (A9-A12) con las sumas escritas en einsum — sin reutilizar
    la composición de la ruta de producción."""
    delta = np.asarray(mu_mod, float) - np.asarray(mu_data, float)
    chit2 = float(np.einsum("i,ij,j->", delta, W, delta))
    B = float(np.einsum("i,ij->", delta, W))
    S = float(np.einsum("ij->", W))
    return chit2 - (B ** 2) / S + math.log(S / (2.0 * math.pi))
