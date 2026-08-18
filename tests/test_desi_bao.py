"""Tests de la ingesta DESI DR2 BAO (offline: los ficheros están
COMMITEADOS con manifest — un clone limpio ejecuta esto sin red).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmology.desi_bao import (
    OFFICIAL_BINS,
    chi2_bao,
    load_desi_dr2_all,
    loo_configurations,
    subset_indices,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifests" / "desi_dr2_bao.json"


def test_manifest_checksums_hold():
    """Los sha256 del manifest (computados al ingerir, nunca a mano)
    siguen siendo los de los ficheros commiteados."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from data_registry import sha256_of
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert doc["version"].startswith("bb0c1c9009dc")
    assert doc["state"] == "AVAILABLE"
    for e in doc["files"]:
        f = REPO / "data" / "raw" / "desi_dr2" / e["file"]
        assert f.exists(), e["file"]
        assert sha256_of(f) == e["sha256"], e["file"]


def test_vector_structure_matches_schema_report():
    """13 componentes, los 7 z_eff oficiales, covarianza 13×13
    simétrica definida positiva."""
    z, quant, val, bins, C = load_desi_dr2_all()
    assert z.size == 13
    assert C.shape == (13, 13)
    assert sorted(set(bins)) == sorted(OFFICIAL_BINS.values())
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > 0.0)
    assert quant[0] == "DV_over_rs" and bins[0] == "bgs"
    assert quant.count("DM_over_rs") == 6
    assert quant.count("DH_over_rs") == 6


def test_first_value_pinned_against_release():
    """Candado contra el release oficial: DV/rs(bgs, z=0.295) =
    7.94167639 (fichero ingerido, commit bb0c1c9)."""
    _, _, val, _, _ = load_desi_dr2_all()
    assert val[0] == pytest.approx(7.94167639, abs=1e-8)


def test_loo_configurations_automated():
    """DESI_ALL + 7 DESI_MINUS_*: tamaños correctos (13; 12 al quitar
    bgs, 11 al quitar cualquier bin de dos componentes)."""
    _, _, _, bins, _ = load_desi_dr2_all()
    loo = loo_configurations(bins)
    assert len(loo) == 8
    assert loo["DESI_ALL"].size == 13
    assert loo["DESI_MINUS_BGS"].size == 12
    for k, idx in loo.items():
        if k not in ("DESI_ALL", "DESI_MINUS_BGS"):
            assert idx.size == 11, k
    assert loo["DESI_MINUS_LRG_Z1"].size == 11   # LRG2 ≡ lrg-z1
    with pytest.raises(ValueError, match="desconocido"):
        subset_indices(bins, drop="lrg2")


def test_chi2_zero_at_data_and_positive_off():
    """χ²(datos, datos) = 0; alejarse en una componente con la
    covarianza oficial da χ² > 0; el subconjunto recorta coherente."""
    _, _, val, bins, C = load_desi_dr2_all()
    assert chi2_bao(val, val, C) == pytest.approx(0.0, abs=1e-20)
    off = val.copy()
    off[0] += 0.1
    full = chi2_bao(off, val, C)
    assert full > 0.0
    idx = subset_indices(bins, drop="bgs")
    assert chi2_bao(off, val, C, idx=idx) == pytest.approx(0.0, abs=1e-20)
