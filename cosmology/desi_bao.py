"""Likelihood BAO DESI DR2 — ingesta oficial, bins con identificador canónico.

Datos: data/raw/desi_dr2/ (ingeridos de CobayaSampler/bao_data —
la likelihood BAO pública que DESI enlaza — con commit fijado y
sha256 en data/manifests/desi_dr2_bao.json; esquema confirmado en
desi_dr2_bao.schema_report.md ANTES de escribir este parser).

Vector ALL_GCcomb (13 componentes; cantidades ADIMENSIONALES /r_d):
    DV/rs (bgs), DM/rs+DH/rs × {lrg-z0, lrg-z1, lrgpluselg, elg,
    qso, lya}.

IDENTIFICADORES OFICIALES (la ambigüedad LRG1/LRG2 queda resuelta y
el identificador oficial se guarda siempre):
    bgs (0.295), lrg-z0 (0.510) ≡ LRG1, lrg-z1 (0.706) ≡ LRG2,
    lrgpluselg (0.934), elg (1.321), qso (1.484), lya (2.330).

El leave-one-bin-out deja de ser manual: subset_indices produce
DESI_ALL y DESI_MINUS_<bin> eliminando las componentes del bin y
recortando la covarianza en consecuencia.

χ² gaussiano con la covarianza oficial completa (correlaciones
DM-DH por bin incluidas): chi2 = rᵀ C⁻¹ r.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "desi_dr2"

# z_eff → identificador oficial (schema_report; redondeo a 3 decimales)
OFFICIAL_BINS = {
    0.295: "bgs",
    0.510: "lrg-z0",       # ≡ LRG1
    0.706: "lrg-z1",       # ≡ LRG2
    0.934: "lrgpluselg",
    1.321: "elg",
    1.484: "qso",
    2.330: "lya",
}


def load_desi_dr2_all():
    """(z, cantidad, valor, bin_oficial, C) del vector ALL_GCcomb.

    Falla cerrado (FileNotFoundError) si la ingesta no está presente."""
    mean_f = RAW / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_f = RAW / "desi_gaussian_bao_ALL_GCcomb_cov.txt"
    if not mean_f.exists() or not cov_f.exists():
        raise FileNotFoundError(
            "DATA_UNAVAILABLE: vector DESI DR2 no ingerido "
            "(data/raw/desi_dr2/ + manifest)")
    z, val, quant = [], [], []
    for line in mean_f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        z.append(float(parts[0]))
        val.append(float(parts[1]))
        quant.append(parts[2])
    C = np.loadtxt(cov_f)
    z = np.asarray(z)
    val = np.asarray(val)
    if C.shape != (z.size, z.size):
        raise ValueError(f"cov {C.shape} incompatible con mean {z.size}")
    bins = [OFFICIAL_BINS[round(zi, 3)] for zi in z]
    return z, quant, val, bins, C


def subset_indices(bins: list[str], drop: str | None = None) -> np.ndarray:
    """Índices del subconjunto: DESI_ALL (drop=None) o
    DESI_MINUS_<bin> (drop = identificador oficial)."""
    if drop is not None and drop not in set(bins):
        raise ValueError(f"bin desconocido: {drop!r} "
                         f"(oficiales: {sorted(set(bins))})")
    return np.array([i for i, b in enumerate(bins) if b != drop],
                    dtype=int)


def loo_configurations(bins: list[str]) -> dict[str, np.ndarray]:
    """Todas las configuraciones leave-one-bin-out, automatizadas:
    DESI_ALL + DESI_MINUS_<BIN> para cada bin oficial presente."""
    out = {"DESI_ALL": subset_indices(bins)}
    for b in dict.fromkeys(bins):
        key = "DESI_MINUS_" + b.upper().replace("-", "_").replace("+", "PLUS")
        out[key] = subset_indices(bins, drop=b)
    return out


def chi2_bao(model_values: np.ndarray, data_values: np.ndarray,
             C: np.ndarray, idx: np.ndarray | None = None) -> float:
    """χ² = rᵀ C⁻¹ r sobre el subconjunto idx (todo si None), con la
    covarianza oficial recortada coherentemente."""
    m = np.asarray(model_values, float)
    d = np.asarray(data_values, float)
    if idx is not None:
        m, d = m[idx], d[idx]
        C = C[np.ix_(idx, idx)]
    r = m - d
    return float(r @ np.linalg.solve(C, r))
