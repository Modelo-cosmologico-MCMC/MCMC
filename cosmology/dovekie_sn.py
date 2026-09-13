"""Pipeline SN Dovekie (DES-SN5YR) — etapa cosmológica, implementación A.

ALCANCE DECLARADO: este módulo consume el release de nivel Hubble
diagram (HD + matrices de covarianza inversa) de 4_DISTANCES_COVMAT —
el punto de entrada de nuestro pipeline. Las etapas DES aguas arriba
(ajuste SALT3 de curvas de luz, BBC/bias corrections) NO se
reimplementan aquí: las validan los mocks fotométricos de DES dentro
del pipeline de DES. La validación por mocks del PIPELINE PROPIO
(PR #15) opera en este mismo nivel HD (validation/dovekie_mocks.py).

BARRERA EJECUTABLE (patrón 5E, preinscrita): la columna MU del HD real
solo es accesible vía load_dovekie_hd(), que FALLA CERRADO salvo que
exista results/2026-09-12_dovekie_mocks/mock_validation.json con
status PASS y el sha256 de la preinscripción congelada. El generador
de mocks usa load_dovekie_design(), que devuelve únicamente los
redshifts (el diseño del experimento), nunca MU.

Fórmula de la likelihood (idéntica a la oficial,
data/raw/des_dovekie/DES-Dovekie-SN_Likelihood.py, ec. A9-A12 de
Goliath et al. astro-ph/0104009 — M marginalizada analíticamente):

    χ̃² = Δᵀ·W·Δ − B²/S + ln(S/2π),   Δ = μ_model − μ_data,
    B = 1ᵀ·W·Δ,  S = 1ᵀ·W·1,  W = C⁻¹ (la que distribuye el release).

M es totalmente degenerada con H0: H0 queda FIJADA a 70 (la del HD,
que asume H0 = 70) y no se mide — declarado, igual que en el script
oficial ("Do not try to measure H0 with SN data").

μ teórica (convención EXACTA del script oficial):
    μ(z) = 5·log10[(1 + zHD)·(1 + zHEL)·D_A(zHD)] + 25
         = 5·log10[(1 + zHEL)·D_M(zHD)] + 25       [D_M comóvil, Mpc]
con E(z) del fondo corregido (cosmology.background.H_of_z, clausura
plana por llamada; ΛCDM ≡ ε = 0 exacto, Prop. A.1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cosmology.background import H_of_z
from mcmc_ontology.data_registry import require_available

ROOT = Path(__file__).resolve().parent.parent
MOCK_VALIDATION = (ROOT / "results" / "2026-09-12_dovekie_mocks"
                   / "mock_validation.json")
PREREG = (ROOT / "results" / "2026-09-12_dovekie_mocks"
          / "preregistration.json")

C_KMS = 299792.458
H0_FIXED = 70.0            # la del HD oficial; M-marginalización ⇒ no medible
N_SN_EXPECTED = 1820       # conteo verificado por bytes (npz nsn y HD);
                           # la cifra 1828 del informe de traspaso era errónea
Z_GRID = np.linspace(0.0, 1.5, 3001)   # malla del integrador (z_max HD ≈ 1.13)


# ---------------------------------------------------------------------
# Carga del release (registro de datos 6A.1: manifest + sha256)
# ---------------------------------------------------------------------

def _parse_hd(path: Path) -> dict:
    """Parsea el HD oficial (formato SNANA: cabecera «VARNAMES:» y
    filas separadas por espacios). Devuelve todas las columnas como
    arrays; el corte del script oficial (zHD > 0) se aplica aquí."""
    names: list[str] = []
    rows: list[list[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
                continue
            if line.startswith("SN:"):     # clave de fila SNANA
                rows.append(line.split()[1:])
    if not names:
        raise ValueError(f"HD sin cabecera VARNAMES: {path}")
    cols = {n: [] for n in names}
    for r in rows:
        for n, v in zip(names, r):
            cols[n].append(v)
    z = np.array(cols["zHD"], dtype=float)
    keep = z > 0.0                       # corte del script oficial
    out = {}
    for n in names:
        if n in ("CID",):
            out[n] = np.array(cols[n])[keep]
        else:
            out[n] = np.array(cols[n], dtype=float)[keep]
    return out


def load_dovekie_design() -> dict:
    """Diseño del experimento SIN el dato: redshifts (zHD, zHEL) y N.

    Es la ÚNICA función que el generador de mocks puede usar: no
    devuelve MU ni MUERR. La barrera de mock_validation NO aplica aquí
    (el diseño no es el dato); el registro sha256 sí."""
    raw = require_available("des_dovekie")
    hd = _parse_hd(raw / "DES-Dovekie_HD.csv")
    return {"zHD": hd["zHD"], "zHEL": hd["zHEL"], "n_sn": len(hd["zHD"])}


def load_dovekie_inv_cov(kind: str = "STAT+SYS") -> np.ndarray:
    """Matriz de covarianza INVERSA oficial, desempaquetada con la
    convención del release (triangular superior aplanada → simétrica).
    kind ∈ {"STATONLY", "STAT+SYS"}."""
    if kind not in ("STATONLY", "STAT+SYS"):
        raise ValueError(kind)
    raw = require_available("des_dovekie")
    d = np.load(raw / f"{kind}.npz")
    n = int(d[d.files[0]][0])
    W = np.zeros((n, n))
    W[np.triu_indices(n)] = d[d.files[1]]
    lower = np.tril_indices(n, -1)
    W[lower] = W.T[lower]
    return W


def require_mock_validation_pass() -> dict:
    """Barrera ejecutable del PR #15: los datos reales de Dovekie no
    entran en ningún script de inferencia sin un mock_validation.json
    con status PASS que cite el sha256 de la preinscripción congelada.
    Falla cerrado en cualquier otro caso."""
    if not MOCK_VALIDATION.exists():
        raise RuntimeError(
            "MOCK_VALIDATION_REQUIRED: no existe "
            f"{MOCK_VALIDATION} — el pipeline SN no puede consumir el "
            "HD real de Dovekie sin superar las cuatro puertas sobre "
            "mocks (scripts/run_dovekie_mocks.py)")
    doc = json.loads(MOCK_VALIDATION.read_text(encoding="utf-8"))
    if doc.get("status") != "PASS":
        raise RuntimeError(
            f"MOCK_VALIDATION_{doc.get('status', 'ABSENT')}: la "
            "validación por mocks no está en PASS — falla cerrado")
    if not PREREG.exists():
        raise RuntimeError(
            "PREREGISTRATION_MISSING: mock_validation.json existe pero "
            "la preinscripción que cita no está en el árbol")
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if doc.get("preregistration_sha256") != sha:
        raise RuntimeError(
            "PREREGISTRATION_MISMATCH: el sha256 de la preinscripción "
            "no coincide con el citado por mock_validation.json — "
            "falla cerrado (¿preinscripción alterada tras el PASS?)")
    return doc


def load_dovekie_hd() -> dict:
    """HD real COMPLETO (incluye MU). Guardado por la barrera: solo
    accesible tras el PASS de la validación por mocks."""
    require_mock_validation_pass()
    raw = require_available("des_dovekie")
    hd = _parse_hd(raw / "DES-Dovekie_HD.csv")
    hd["n_sn"] = len(hd["zHD"])
    return hd


# ---------------------------------------------------------------------
# Modelo de distancias (implementación A — producción)
# ---------------------------------------------------------------------

def _cumulative_invE(Omega_m: float, eps: float,
                     z_trans: float) -> np.ndarray:
    """Cumulativa de 1/E sobre Z_GRID a O(h⁴): Simpson por pares en los
    puntos pares y media-regla (5, 8, −1)/12 en los impares — el mismo
    esquema que el integrador DESI de producción corregido tras el
    crosscheck JAX (#14)."""
    E = np.asarray(H_of_z(Z_GRID, H0=1.0, Omega_m=Omega_m, eps=eps,
                          z_trans=z_trans))
    f = 1.0 / E
    h = float(Z_GRID[1] - Z_GRID[0])
    e0, mid, e2 = f[0:-1:2], f[1::2], f[2::2]
    out = np.empty(len(Z_GRID))
    out[0] = 0.0
    out[2::2] = np.cumsum(h / 3.0 * (e0 + 4.0 * mid + e2))
    out[1::2] = out[0:-1:2] + h / 12.0 * (5.0 * e0 + 8.0 * mid - e2)
    return out


def mu_model(zHD: np.ndarray, zHEL: np.ndarray, Omega_m: float,
             eps: float = 0.0, z_trans: float = 8.9) -> np.ndarray:
    """μ teórica en la convención del script oficial (ver docstring del
    módulo), con D_M por interpolación lineal de la cumulativa O(h⁴)
    (el residuo de interpolación, medido contra la implementación
    independiente, entra en la puerta 1b de la preinscripción)."""
    zHD = np.asarray(zHD, float)
    zHEL = np.asarray(zHEL, float)
    if np.any(zHD > Z_GRID[-1]):
        raise ValueError("zHD fuera de la malla del integrador")
    invE = _cumulative_invE(Omega_m, eps, z_trans)
    DM = (C_KMS / H0_FIXED) * np.interp(zHD, Z_GRID, invE)
    return 5.0 * np.log10((1.0 + zHEL) * np.maximum(DM, 1e-300)) + 25.0


def chi2_sn_marginalized(mu_mod: np.ndarray, mu_data: np.ndarray,
                         W: np.ndarray) -> float:
    """χ̃² con M marginalizada analíticamente (fórmula oficial A9-A12,
    incluido el término ln(S/2π)). Invariante EXACTA bajo
    μ_data → μ_data + const (propiedad testeada)."""
    delta = np.asarray(mu_mod, float) - np.asarray(mu_data, float)
    Wd = W @ delta
    chit2 = float(delta @ Wd)
    B = float(np.sum(Wd))
    S = float(np.sum(W))
    return chit2 - B * B / S + float(np.log(S / (2.0 * np.pi)))


def loglike_sn(mu_mod: np.ndarray, mu_data: np.ndarray,
               W: np.ndarray) -> float:
    """log L = −χ̃²/2 (la misma cantidad que devuelve la función
    oficial cov_log_likelihood)."""
    return -0.5 * chi2_sn_marginalized(mu_mod, mu_data, W)
