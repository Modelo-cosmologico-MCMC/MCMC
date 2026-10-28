"""Ingesta SPARC alineada con data/sources.yaml (PR-S, 27-oct): rutas y manifest del registro, fallo cerrado sin bytes y
sin esquema confirmado, y el parser byte a byte con fallo cerrado ante la ambigüedad. Los ficheros de estas pruebas son
SINTÉTICOS y declarados (dos galaxias inventadas, «SINT-1» y «SINT-2»): ejercitan el lector, no son datos."""

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynamics import sparc_data as sd
from dynamics.walker_binning import SchemaError

REPO = Path(__file__).resolve().parent.parent

MASTER_COLS = [("Galaxy", "A11", "---", "Galaxy name"), ("T", "I2", "---", "Hubble type"), ("D", "F6.2", "Mpc", "Distance"),
               ("e_D", "F5.2", "Mpc", "Mean error on D"), ("Inc", "F4.1", "deg", "Inclination"),
               ("e_Inc", "F4.1", "deg", "Mean error on Inc"), ("Rdisk", "F5.2", "kpc", "Disk scale length"),
               ("SBdisk", "F8.2", "solLum/pc2", "Disk central surface brightness"), ("Q", "I3", "---", "Quality flag")]
MM_COLS = [("ID", "A11", "---", "Galaxy identifier"), ("D", "F6.2", "Mpc", "Assumed distance"), ("R", "F6.2", "kpc", "Radius"),
           ("Vobs", "F6.2", "km/s", "Observed velocity"), ("e_Vobs", "F5.2", "km/s", "Uncertainty in Vobs"),
           ("Vgas", "F6.2", "km/s", "Gas velocity"), ("Vdisk", "F6.2", "km/s", "Disk velocity"),
           ("Vbul", "F6.2", "km/s", "Bulge velocity")]

CURVES = {"SINT-1": {"D": 10.0, "R": [0.5, 1.0, 2.0, 4.0, 6.0], "Vobs": [40.0, 70.0, 95.0, 110.0, 112.0],
                     "errV": [5.0, 4.0, 3.0, 3.0, 4.0], "Vgas": [-5.0, 10.0, 20.0, 25.0, 24.0],
                     "Vdisk": [50.0, 80.0, 95.0, 85.0, 75.0], "Vbul": [0.0] * 5},
          "SINT-2": {"D": 25.5, "R": [1.0, 3.0, 5.0, 8.0, 12.0, 16.0], "Vobs": [90.0, 160.0, 190.0, 200.0, 205.0, 203.0],
                     "errV": [8.0, 6.0, 5.0, 5.0, 6.0, 7.0], "Vgas": [5.0, 15.0, 25.0, 30.0, 32.0, 30.0],
                     "Vdisk": [100.0, 150.0, 160.0, 150.0, 135.0, 120.0], "Vbul": [80.0, 60.0, 45.0, 35.0, 28.0, 24.0]}}
MASTER = {"SINT-1": [5, 10.0, 1.5, 55.0, 4.0, 1.8, 250.0, 1], "SINT-2": [3, 25.5, 3.0, 70.0, 3.0, 3.1, 900.0, 2]}


def _mrt(name, cols, rows):
    """Texto MRT con la descripción byte a byte (formato de CDS/AAS) y filas de ancho fijo."""
    spec, pos, widths = [], 1, []
    for label, fmt, unit, expl in cols:
        w = int(fmt[1:].split(".")[0])
        spec.append(f"{pos:4d}-{pos + w - 1:3d} {fmt:<6s} {unit:<10s} {label:<8s} {expl}")
        widths.append((w, fmt))
        pos += w
    sep = "-" * 80
    lines = ["Title: SINTETICO (prueba del lector, no son datos)", "=" * 80,
             f"Byte-by-byte Description of file: {name}", sep, " Bytes Format Units      Label    Explanations", sep,
             *spec, sep, "Note (1): sintético.", sep]
    for row in rows:
        out = ""
        for (w, fmt), v in zip(widths, row):
            if fmt[0] == "A":
                out += f"{v:<{w}s}"
            elif fmt[0] == "I":
                out += f"{int(v):>{w}d}"
            else:
                out += f"{v:>{w}.{int(fmt.split('.')[1])}f}"
        lines.append(out)
    return "\n".join(lines) + "\n"


def _rotmod(c):
    head = [f"# Distance = {c['D']} Mpc", "# Rad\tVobs\terrV\tVgas\tVdisk\tVbul\tSBdisk\tSBbul",
            "# kpc\tkm/s\tkm/s\tkm/s\tkm/s\tkm/s\tL/pc^2\tL/pc^2"]
    rows = ["\t".join(f"{x:.2f}" for x in r) for r in zip(c["R"], c["Vobs"], c["errV"], c["Vgas"], c["Vdisk"], c["Vbul"],
                                                          [100.0] * len(c["R"]), [0.0] * len(c["R"]))]
    return "\n".join(head + rows) + "\n"


def _write_raw(tmp_path, curves=CURVES, master=MASTER, mm_shift=0.0, master_cols=MASTER_COLS):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = [[n, *v] for n, v in master.items()]
    if master_cols is not MASTER_COLS:
        rows = [[r[0], *r[1:len(master_cols)]] for r in rows]
    (raw / "SPARC_Lelli2016c.mrt").write_text(_mrt("SPARC_Lelli2016c.mrt", master_cols, rows), encoding="latin-1")
    mm_rows = [[n, c["D"], r, v + mm_shift, e, g, d, b] for n, c in curves.items()
               for r, v, e, g, d, b in zip(c["R"], c["Vobs"], c["errV"], c["Vgas"], c["Vdisk"], c["Vbul"])]
    (raw / "MassModels_Lelli2016c.mrt").write_text(_mrt("MassModels_Lelli2016c.mrt", MM_COLS, mm_rows), encoding="latin-1")
    with zipfile.ZipFile(raw / "Rotmod_LTG.zip", "w") as z:
        for n, c in curves.items():
            z.writestr(f"{n}_rotmod.dat", _rotmod(c))
    for f in ("RAR.mrt", "RARbins.mrt"):
        (raw / f).write_text(_mrt(f, [("gbar", "F7.3", "[m/s2]", "log gbar"), ("gobs", "F7.3", "[m/s2]", "log gobs")],
                                  [[-10.5, -10.2], [-11.0, -10.5]]), encoding="latin-1")
    return raw


def test_paths_follow_the_registry():
    src = yaml.safe_load((REPO / "data" / "sources.yaml").read_text(encoding="utf-8"))["datasets"]["sparc"]
    spec = importlib.util.spec_from_file_location("download_data", REPO / "scripts" / "download_data.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    entries = dl.DOWNLOADS["sparc"]
    assert [d.name for _, d, _ in entries] == src["artifacts"] == list(sd.FILES)
    assert all(d.parent == REPO / src["raw_dir"] for _, d, _ in entries) and sd.RAW_DIR == REPO / src["raw_dir"]
    assert all(sha is None for _, _, sha in entries) and "sparc" in dl.REGISTRY_DATASETS


def test_fails_closed_without_manifest_or_without_confirmed_schema(tmp_path, monkeypatch):
    import mcmc_ontology.data_registry as dr
    monkeypatch.setattr(dr, "MANIFESTS", tmp_path / "manifests")
    monkeypatch.setattr(sd, "manifest_path", dr.manifest_path)
    with pytest.raises(FileNotFoundError, match="DATA_UNAVAILABLE"):
        sd.load_sparc_catalogue()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "sparc.json").write_text(json.dumps({"state": "SCHEMA_UNVERIFIED", "files": []}),
                                                       encoding="utf-8")
    with pytest.raises(RuntimeError, match="SCHEMA_UNVERIFIED"):
        sd.load_sparc_catalogue()


def test_parser_builds_the_galaxies_from_the_byte_by_byte_description(tmp_path):
    gals = sd.build_galaxies(_write_raw(tmp_path))
    assert [g.name for g in gals] == ["SINT-1", "SINT-2"]
    g1, g2 = gals
    assert g1.D == 10.0 and g1.e_D == 1.5 and g1.inc == 55.0 and g1.R_d == 1.8 and g1.SB_disk0 == 250.0 and g1.quality == 1
    assert np.allclose(g1.v_gas, CURVES["SINT-1"]["Vgas"]) and not g1.has_bulge and g2.has_bulge


def test_ambiguity_fails_closed(tmp_path):
    with pytest.raises(SchemaError, match="distinto"):
        sd.build_galaxies(_write_raw(tmp_path / "a", mm_shift=0.5))
    bad_d = {**CURVES, "SINT-1": {**CURVES["SINT-1"], "D": 11.0}}
    with pytest.raises(SchemaError):
        sd.build_galaxies(_write_raw(tmp_path / "b", curves=bad_d))
    dup = [*MASTER_COLS[:-1], ("Q", "I3", "---", "Quality flag")]
    dup[2] = ("D", "F6.2", "kpc", "Distance in the wrong unit")
    with pytest.raises(SchemaError, match="'D'"):
        sd.build_galaxies(_write_raw(tmp_path / "c", master_cols=dup))
    with pytest.raises(SchemaError, match="Rad"):
        sd.parse_rotmod("# Distance = 1 Mpc\n# R\tVobs\n# kpc\tkm/s\n1.0\t2.0\n", "x_rotmod.dat")
    with pytest.raises(SchemaError, match="Distance"):
        sd.parse_rotmod("# Rad\tVobs\terrV\tVgas\tVdisk\tVbul\n# kpc\tkm/s\tkm/s\tkm/s\tkm/s\tkm/s\n", "x_rotmod.dat")


def test_schema_report_is_the_first_action(tmp_path):
    rep = sd.inspect_schema(_write_raw(tmp_path))
    assert "SCHEMA_UNVERIFIED" in rep and "Byte-by-byte Description of file: SPARC_Lelli2016c.mrt" in rep
    assert "2 curvas leídas" in rep and "Filas de datos: 2" in rep and "Columnas derivadas" in rep
    with pytest.raises(FileNotFoundError, match="DATA_UNAVAILABLE"):
        sd.inspect_schema(tmp_path / "vacío")
