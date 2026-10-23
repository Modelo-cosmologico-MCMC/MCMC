"""Binning de Walker et al. 2009 (PR-W, bloque P0 del autor del 23-oct): el lector del ReadMe de CDS y la asignación de
papeles con fallo cerrado, la unidad «estrella» con épocas repetidas, los bins de igual número con ⟨e²⟩ restado, el orden
de las letras y el analizador de extremo a extremo sobre catálogos SINTÉTICOS (ejercitan el instrumento; no son
contraste). Candados del anexo congelado y de las rutas alineadas con data/sources.yaml."""

import hashlib
import importlib.util
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynamics.cronos_jeans import sculptor_profile_table
from dynamics.walker_binning import (
    SchemaError,
    angular_separation,
    assign_roles,
    equal_number_bins,
    frozen_letter,
    parse_readme,
    readme_record_count,
    stars_from_catalogue,
)

REPO = Path(__file__).resolve().parent.parent
SC = REPO / "results" / "2026-09-21_sculptor_profile"


def _script(name):
    spec = importlib.util.spec_from_file_location(name[:-3], REPO / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RS = _script("run_sculptor_profile.py")
PRE = json.loads((SC / "preregistration.json").read_text(encoding="utf-8"))
AX = {"numbers": RS.ANNEX_NUMBERS, "table_file": "table4.dat"}

README = """J/AJ/137/3100   ReadMe SINTETICO de prueba (imita el formato de CDS; no es el ReadMe oficial)
================================================================================
File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records  Explanations
--------------------------------------------------------------------------------
ReadMe          80        .  This file
table4.dat      71   {n:>6d}  Sculptor (sintetico)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table[2-5].dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1-  8  A8    ---     Target  Target identifier
  10- 11  I2    h       RAh     Hour of Right Ascension (J2000)
  13- 14  I2    min     RAm     Minute of Right Ascension (J2000)
  16- 20  F5.2  s       RAs     Second of Right Ascension (J2000)
      22  A1    ---     DE-     Sign of the Declination (J2000)
  23- 24  I2    deg     DEd     Degree of Declination (J2000)
  26- 27  I2    arcmin  DEm     Arcminute of Declination (J2000)
  29- 32  F4.1  arcsec  DEs     Arcsecond of Declination (J2000)
  34- 43  F10.4 d       HJD     Heliocentric Julian date minus 2450000
  45- 50  F6.1  km/s    HV      Heliocentric line-of-sight velocity
  52- 55  F4.1  km/s    e_HV    Error in HV
  57- 61  F5.2  0.1nm   SigMg   Magnesium index
                                (continuation of the explanation)
  63- 66  F4.2  0.1nm   e_SigMg Error in SigMg
  68- 71  F4.2  ---     PM      ? Membership probability (1)
--------------------------------------------------------------------------------
Note (1): synthetic.
================================================================================
"""
C_RA, C_DEC = RS.ANNEX_NUMBERS["centre_ra_deg"], RS.ANNEX_NUMBERS["centre_dec_deg"]


def _line(target, ra, dec, hjd, v, e, pm):
    t = round(ra / 15.0 * 3600.0, 2)
    h, m = int(t // 3600), int((t % 3600) // 60)
    a = round(abs(dec) * 3600.0, 1)
    d, dm = int(a // 3600), int((a % 3600) // 60)
    v_s = " " * 6 if v is None else f"{v:6.1f}"
    pm_s = " " * 4 if pm is None else f"{pm:4.2f}"
    return (f"{target:<8s} {h:02d} {m:02d} {t - 3600 * h - 60 * m:05.2f} {'-' if dec < 0 else '+'}{d:02d} {dm:02d} "
            f"{a - 3600 * d - 60 * dm:04.1f} {hjd:10.4f} {v_s} {e:4.1f} {0.4:5.2f} {0.05:4.2f} {pm_s}")


def _write(tmp_path, lines):
    (tmp_path / "ReadMe").write_text(README.format(n=len(lines)), encoding="ascii")
    (tmp_path / "table4.dat").write_text("\n".join(lines) + "\n", encoding="ascii")
    return (tmp_path / "ReadMe").read_text(encoding="ascii"), tmp_path / "table4.dat"


def _galaxy(tmp_path, sigma_of_R, n_mem=1500, n_rep=200, n_non=300, e_range=(1.0, 3.0), seed=1):
    """Catálogo sintético: miembros con perfil de Plummer proyectado (a = 260 pc a 79 kpc), dispersión σ(R) dada,
    épocas repetidas de la misma estrella, no miembros de velocidad ancha (algunos sin P_M)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.94, n_mem)
    R = 260.0 * np.sqrt(u / (1.0 - u))
    phi = rng.uniform(0.0, 2.0 * math.pi, n_mem)
    th = np.degrees(R / 79000.0)
    dec = C_DEC + th * np.sin(phi)
    ra = C_RA + th * np.cos(phi) / math.cos(math.radians(C_DEC))
    r_non = rng.uniform(0.0, 0.35, n_non)
    ra = np.concatenate([ra, C_RA + r_non * np.cos(np.arange(n_non))])
    dec = np.concatenate([dec, C_DEC + r_non * np.sin(np.arange(n_non))])
    while True:                                   # estrellas distintas a > 2″ (en el sintético no hay pares casuales)
        u = np.stack([np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                      np.sin(np.radians(dec))], axis=-1)
        close = np.argwhere(np.triu(u @ u.T > math.cos(math.radians(2.0 / 3600.0)), 1))
        if not close.size:
            break
        dec[close[:, 1]] += 3.0 / 3600.0
    e = rng.uniform(*e_range, n_mem)
    v_true = 111.0 + rng.normal(0.0, 1.0, n_mem) * sigma_of_R(R)
    pm = rng.uniform(0.92, 1.0, n_mem)
    lines = [_line(f"S{i:05d}", ra[i], dec[i], 4700.0 + i, v_true[i] + rng.normal() * e[i], e[i], pm[i]) for i in range(n_mem)]
    for i in rng.choice(n_mem, n_rep, replace=False):
        e2 = rng.uniform(*e_range)
        lines.append(_line(f"S{i:05d}", ra[i], dec[i], 5100.0 + i, v_true[i] + rng.normal() * e2, e2, pm[i]))
    for j in range(n_non):
        lines.append(_line(f"N{j:05d}", ra[n_mem + j], dec[n_mem + j], 4800.0 + j, rng.normal(60.0, 70.0),
                           rng.uniform(*e_range), None if j % 10 == 0 else rng.uniform(0.0, 0.5)))
    return _write(tmp_path, lines)


def _cols(*specs):
    return [{"start": 0, "end": 0, "kind": k, "unit": u, "label": lab, "explanation": x} for k, u, lab, x in specs]


BASE = (("F", "deg", "RAdeg", "Right ascension"), ("F", "deg", "DEdeg", "Declination"),
        ("F", "km/s", "Vlos", "line-of-sight velocity"), ("F", "km/s", "e_Vlos", "error"),
        ("F", "---", "Pmem", "membership probability"))


def test_readme_parser_and_roles():
    cols = parse_readme(README.format(n=3), "table4.dat")
    assert [c["label"] for c in cols] == ["Target", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "HJD", "HV", "e_HV",
                                          "SigMg", "e_SigMg", "PM"]
    by = {c["label"]: c for c in cols}
    assert (by["DE-"]["start"], by["DE-"]["end"], by["DE-"]["kind"]) == (22, 22, "A")
    assert (by["HV"]["start"], by["HV"]["end"], by["HV"]["unit"]) == (45, 50, "km/s")
    assert by["SigMg"]["explanation"].endswith("(continuation of the explanation)")
    assert assign_roles(cols) == {"position": {"format": "sexagesimal", "labels": ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]},
                                  "velocity": "HV", "velocity_error": "e_HV", "membership": "PM", "identifier": "Target"}
    assert readme_record_count(README.format(n=1818), "table4.dat") == 1818
    assert readme_record_count(README.format(n=1818), "table2.dat") is None
    with pytest.raises(SchemaError, match="byte a byte"):
        parse_readme(README.format(n=3), "table9.dat")


def test_roles_fail_closed_on_ambiguity():
    r = assign_roles(_cols(*BASE))
    assert r["position"]["format"] == "deg" and r["velocity"] == "Vlos" and r["identifier"] is None
    helio = assign_roles(_cols(*BASE, ("F", "km/s", "Vhel", "heliocentric velocity"), ("F", "km/s", "e_Vhel", "error")))
    assert helio["velocity"] == "Vhel"
    with pytest.raises(SchemaError, match="velocidad"):
        assign_roles(_cols(*BASE, ("F", "km/s", "Vgsr", "Galactic standard of rest velocity")))
    with pytest.raises(SchemaError, match="error de la velocidad"):
        assign_roles(_cols(*[s for s in BASE if s[2] != "e_Vlos"]))
    with pytest.raises(SchemaError, match="pertenencia"):
        assign_roles(_cols(*[s for s in BASE if s[2] != "Pmem"], ("I", "---", "Mem", "membership flag")))
    with pytest.raises(SchemaError, match="pertenencia"):
        assign_roles(_cols(*BASE, ("F", "---", "Pmem2", "second membership probability")))
    with pytest.raises(SchemaError, match="posición"):
        assign_roles(_cols(*BASE[2:]))
    two_ids = assign_roles(_cols(*BASE, ("A", "---", "ID", "Star identifier"), ("A", "---", "Field", "Field name")))
    assert two_ids["identifier"] is None


def test_epochs_combine_into_stars(tmp_path):
    p1, p2, p4, p6 = (C_RA, C_DEC), (C_RA + 0.1, C_DEC), (C_RA, C_DEC + 0.1), (C_RA - 0.1, C_DEC)
    lines = [_line("S1", *p1, 4700.0, 100.0, 2.0, 0.95), _line("S1", *p1, 4800.0, 104.0, 1.0, 0.97),
             _line("S2", *p2, 4700.0, 90.0, 2.0, 0.95), _line("S3", p2[0], p2[1] + 0.5 / 3600.0, 4701.0, 94.0, 2.0, 0.80),
             _line("S4", C_RA, C_DEC - 0.1, 4700.0, None, 2.0, 0.99),
             _line("S5", *p4, 4700.0, 80.0, 3.0, None), _line("S5", *p4, 4700.0, 80.0, 3.0, None),
             _line("S6", *p6, 4700.0, 70.0, 2.0, 0.99), _line("S6", p6[0], p6[1] + 5.0 / 3600.0, 4900.0, 74.0, 2.0, 0.99)]
    text, table = _write(tmp_path, lines)
    st = stars_from_catalogue(text, table, "table4.dat", 1.0)
    d = st["diagnostics"]
    assert (d["lines"], d["identical_lines_dropped"], d["records_dropped_missing"], d["stars"]) == (9, 1, 1, 4)
    assert (d["stars_with_repeat_epochs"], d["merges_by_position_only"], d["identifiers_with_records_beyond_merge_radius"]) == (3, 1, 1)
    assert st["v_kms"][0] == pytest.approx((100.0 / 4 + 104.0) / 1.25) and st["e_kms"][0] == pytest.approx(1.25 ** -0.5)
    assert st["v_kms"][1] == pytest.approx(92.0) and st["e_kms"][1] == pytest.approx(math.sqrt(2.0))
    assert st["pm"][1] == pytest.approx(0.80) and st["pm_max"][1] == pytest.approx(0.95) and math.isnan(st["pm"][2])
    assert list(st["n_epochs"]) == [2, 2, 1, 2]
    with pytest.raises(SchemaError, match="declarados"):
        (tmp_path / "ReadMe").write_text(README.format(n=8), encoding="ascii")
        stars_from_catalogue((tmp_path / "ReadMe").read_text(encoding="ascii"), table, "table4.dat", 1.0)
    bad = [ln[:67] + "1.50" for ln in lines]
    text, table = _write(tmp_path, bad)
    with pytest.raises(SchemaError, match=r"\[0, 1\]"):
        stars_from_catalogue(text, table, "table4.dat", 1.0)


def test_equal_number_bins_and_corrected_sigma():
    rng = np.random.default_rng(3)
    R = np.concatenate([rng.uniform(0.0, 1000.0, 1000), rng.uniform(1000.5, 1500.0, 80)])
    v, e = rng.normal(0.0, 9.0, R.size), rng.uniform(1.0, 3.0, R.size)
    bins = equal_number_bins(R, v, e, 150, 1000.0)
    assert [b["N"] for b in bins] == [143] * 6 + [142]
    order = np.argsort(R[:1000], kind="stable")
    for b, idx in zip(bins, np.array_split(order, 7)):
        s2, e2 = float(np.var(v[idx], ddof=1)), float(np.mean(e[idx] ** 2))
        assert b["R_pc"] == pytest.approx(float(np.median(R[idx]))) and b["s2"] == pytest.approx(s2)
        assert b["sigma_kms"] == pytest.approx(math.sqrt(s2 - e2))
        assert b["sigma_err_kms"] == pytest.approx(math.sqrt(s2 - e2) / math.sqrt(2.0 * b["N"]))
    assert [len(equal_number_bins(np.zeros(n), np.arange(n), np.ones(n), 150, 1.0)) for n in (50, 224, 225, 374, 375)] == [1, 1, 2, 2, 3]
    tie = equal_number_bins(np.zeros(4), [1.0, 2.0, 30.0, 40.0], np.ones(4), 2, 1.0)
    assert [b["v_mean_kms"] for b in tie] == [1.5, 35.0]
    cold = equal_number_bins(np.arange(4.0), [0.0, 0.1, 0.0, 0.1], np.ones(4), 4, 10.0)
    assert math.isnan(cold[0]["sigma_kms"]) and math.isnan(cold[0]["sigma_err_kms"])


def _bins(n, sig=10.0, err=1.0):
    return [{"sigma_kms": sig, "sigma_err_kms": err} for _ in range(n)]


def test_frozen_letter_order():
    kw = {"chi2_A_max": 1.5, "chi2_B_max": 3.0, "C_z_min": 3.0, "C_inner_bins": 2, "min_bins": 4}
    both = {b: [14.0, 14.0] + [10.0] * 23 for b in ("-0.5", "0.0")}          # χ²_ν = 32/25 ≤ 1.5 y C: C precede
    L = frozen_letter(_bins(25), both, **kw)
    assert L["letter"] == "C" and L["per_beta"]["0.0"]["chi2_nu"] == pytest.approx(32 / 25)
    assert frozen_letter(_bins(4), {"0.0": [11.6] * 4}, **kw)["letter"] == "B"
    assert frozen_letter(_bins(4), {"0.0": [10.5] * 4}, **kw)["letter"] == "A"
    some = frozen_letter(_bins(4), {"-0.5": [12.5] * 4, "0.3": [11.0] * 4}, **kw)
    assert some["letter"] == "A" and some["best_beta"] == "0.3"
    fall = frozen_letter(_bins(4), {"0.0": [10.0, 10.0, 15.0, 15.0]}, **kw)
    assert fall["letter"] == "INDETERMINADO" and "ninguna letra" in fall["reason"]
    not_all = frozen_letter(_bins(25), {"-0.5": [14.0, 14.0] + [10.0] * 23, "0.0": [12.0, 14.0] + [10.0] * 23}, **kw)
    assert not_all["letter"] == "A"
    assert frozen_letter(_bins(3), {"0.0": [10.0] * 3}, **kw)["letter"] == "INDETERMINADO"
    nan = _bins(4)
    nan[2]["sigma_kms"] = math.nan
    assert "σ del bin" in frozen_letter(nan, {"0.0": [10.0] * 4}, **kw)["reason"]


def test_synthetic_flat_profile_end_to_end(tmp_path):
    """Perfil plano de 9.5 km/s (la forma de la referencia de la literatura): el instrumento lo recupera y la regla
    congelada da C frente al pico predicho."""
    text, table = _galaxy(tmp_path, lambda R: np.full_like(R, 9.5))
    res = RS.analyze_catalogue(text, table, PRE, AX)
    assert res["verdict"] == "C" and res["prediction_reproduction_max_dev_kms"] <= 1e-6
    rd = res["reading"]
    assert (rd["stars"], rd["stars_with_repeat_epochs"], rd["merges_by_position_only"]) == (1800, 200, 0)
    assert res["members"] == 1500 and res["member_centroid_offset_arcmin"] < 1.0
    n_in = res["members_within_R_max"]
    assert len(res["bins"]) == math.floor(n_in / 150 + 0.5) and sum(b["N"] for b in res["bins"]) == n_in
    sig = np.array([b["sigma_kms"] for b in res["bins"]])
    err = np.array([b["sigma_err_kms"] for b in res["bins"]])
    assert np.all(np.abs(sig - 9.5) < 4.0 * err)
    assert float(np.sum(sig / err ** 2) / np.sum(err ** -2)) == pytest.approx(9.5, abs=0.4)
    assert all(min(v["z_inner"]) > 3.0 for v in res["letter_detail"]["per_beta"].values())
    assert set(res["without_letter_distance_sensitivity"]) == {"75.0", "83.0"}


def test_synthetic_predicted_profile_is_not_rejected(tmp_path):
    """Catálogo generado con la propia predicción congelada (β = 0): la regla no lo excluye (C no se cumple y χ²_ν es de
    orden uno) — el instrumento no fabrica la letra C."""
    grid = np.geomspace(2.0, 1200.0, 80)
    s0 = np.array([p["sigma_los_newton_cronos"] for p in sculptor_profile_table(beta=0.0, R_pc=tuple(grid))["profile"]])
    text, table = _galaxy(tmp_path, lambda R: np.interp(np.log(R), np.log(grid), s0), e_range=(0.5, 1.5), seed=7)
    res = RS.analyze_catalogue(text, table, PRE, AX)
    per = res["letter_detail"]["per_beta"]
    assert res["verdict"] in ("A", "B") and per["0.0"]["chi2_nu"] < 3.0
    assert not all(min(v["z_inner"]) > 3.0 for v in per.values())


def test_schema_and_gates_fail_closed(tmp_path):
    text, table = _galaxy(tmp_path, lambda R: np.full_like(R, 9.5), n_mem=300, n_rep=0, n_non=0)
    broken = text.replace("e_HV    Error in HV", "eHV     Error in HV")
    assert "esquema no confirmado" in RS.analyze_catalogue(broken, table, PRE, AX)["reason"]
    off = {"numbers": {**RS.ANNEX_NUMBERS, "centre_dec_deg": C_DEC + 0.5}, "table_file": "table4.dat"}
    res = RS.analyze_catalogue(text, table, PRE, off)
    assert res["verdict"] == "INDETERMINADO" and "centroide" in res["reason"]
    few = RS.analyze_catalogue(text, table, PRE, AX)                       # 300 miembros → 2 bins < 4
    assert few["verdict"] == "INDETERMINADO" and "N_bins" in few["reason"]
    drift = json.loads(json.dumps(PRE))
    drift["declared"]["prediction_profiles_kms"]["0.0"][0]["newton_cronos"] += 1e-3
    assert "no reproducida" in RS.analyze_catalogue(text, table, drift, AX)["reason"]


def test_generators_refuse_to_overwrite_or_to_freeze_after_bytes(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="ya está congelada"):
        RS.prereg()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "table4.dat").write_text("x\n", encoding="ascii")
    monkeypatch.setattr(RS, "ANNEX", tmp_path / "binning_annex.json")
    monkeypatch.setattr(RS, "RAW", raw)
    with pytest.raises(SystemExit, match="antes de los bytes"):
        RS.annex()
    monkeypatch.setattr(RS, "RAW", tmp_path / "absent")
    monkeypatch.setattr(RS, "OUTDIR", tmp_path)
    assert RS.annex() == 0
    doc = json.loads((tmp_path / "binning_annex.json").read_text(encoding="utf-8"))
    assert doc["numbers"] == RS.ANNEX_NUMBERS and doc["of_preregistration"]["sha256"] == _sha(SC / "preregistration.json")
    assert _sha(tmp_path / "binning_annex.json") in (tmp_path / "binning_annex.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="ya está congelado"):
        RS.annex()


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_analyze_wiring_with_an_available_catalogue(tmp_path, monkeypatch):
    """analyze() de punta a punta con un raw_dir sintético dado por require_available: cita los sha de la preinscripción
    y del anexo, escribe la tabla de bins, y falla cerrado si el anexo cambia tras congelarse."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    _galaxy(raw, lambda R: np.full_like(R, 9.5))
    monkeypatch.setattr(RS, "OUTDIR", out)
    monkeypatch.setattr(RS, "ANNEX", out / "binning_annex.json")
    monkeypatch.setattr(RS, "RAW", tmp_path / "absent")
    assert RS.annex() == 0
    monkeypatch.setattr(RS, "require_available", lambda dataset: raw)
    assert RS.analyze() == 0
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["verdict"] == "C" and res["annex_sha256"] == _sha(out / "binning_annex.json")
    assert res["preregistration_sha256"] == _sha(SC / "preregistration.json") and len(res["bins"]) == 10
    assert "| 10 | 149 |" in (out / "report.md").read_text(encoding="utf-8")
    doc = json.loads((out / "binning_annex.json").read_text(encoding="utf-8"))
    doc["numbers"]["pm_min"] = 0.5
    (out / "binning_annex.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="no es el congelado"):
        RS.analyze()


@pytest.mark.skipif(not (SC / "binning_annex.json").exists(), reason="anexo de binning aún no congelado")
def test_frozen_annex_lock():
    ax = json.loads((SC / "binning_annex.json").read_text(encoding="utf-8"))
    assert _sha(SC / "binning_annex.json") in (SC / "binning_annex.md").read_text(encoding="utf-8")
    assert ax["of_preregistration"]["sha256"] == _sha(SC / "preregistration.json") == "604f9c254c7e8283949137d23bb75fb4200b5341a369b1b89f06622ce53af002"
    v = ax["frozen_rule_verbatim"]
    assert v["rules"] == PRE["rules"] and v["membership"] == PRE["declared"]["membership"] and v["binning"] == PRE["declared"]["binning"]
    N = ax["numbers"]
    assert N == RS.ANNEX_NUMBERS and ax["table_file"] == "table4.dat"
    # los umbrales del anexo son los del texto congelado
    assert f"≥ {N['pm_min']}" in v["membership"] and f"≈ {N['target_per_bin']} estrellas" in v["binning"] and "≤ 1 kpc" in v["binning"]
    assert f"≤ {N['chi2_A_max']}" in v["rules"]["A"] and f"≤ {N['chi2_B_max']:g}" in v["rules"]["B"]
    assert f"> {N['C_z_min']:g}" in v["rules"]["C"] and "dos bins" in v["rules"]["C"] and N["C_inner_bins"] == 2
    assert f"N_bins < {N['min_bins']}" in v["rules"]["INDETERMINADO"]
    # el analizador lee los números del anexo, no constantes
    body = inspect.getsource(RS.analyze_catalogue)
    for k in ("pm_min", "target_per_bin", "R_max_pc", "distance_kpc", "chi2_A_max", "C_z_min", "centroid_gate_arcmin"):
        assert f'N["{k}"]' in body
    assert "ANNEX_NUMBERS" not in body and re.search(r"\b(0\.9|150|79\.0|1\.5)\b", body) is None


def test_download_routes_aligned_with_sources(tmp_path, monkeypatch):
    dd = _script("download_data.py")
    src = yaml.safe_load((REPO / "data" / "sources.yaml").read_text(encoding="utf-8"))["datasets"]["walker2009"]
    entries = dd.DOWNLOADS["walker2009"]
    assert "walker09" not in dd.DOWNLOADS and {d.name for _, d, _ in entries} == set(src["artifacts"])
    assert all(d.parent == REPO / src["raw_dir"] for _, d, _ in entries)
    assert all(src["host"] in u and src["catalog"] in u for u, _, _ in entries) and all(s is None for *_, s in entries)
    import mcmc_ontology.data_registry as dr
    monkeypatch.setattr(dr, "MANIFESTS", tmp_path / "manifests")
    f = tmp_path / "table4.dat"
    f.write_text("fixture sintetico, no observacional\n", encoding="ascii")
    p = dd.write_registry_manifest("walker2009", [("https://cdsarc.cds.unistra.fr/ftp/J/AJ/137/3100/table4.dat", f, dd._sha256(f))])
    man = json.loads(p.read_text(encoding="utf-8"))
    assert p == tmp_path / "manifests" / "walker2009.json" and man["state"] == "SCHEMA_UNVERIFIED"
    assert man["files"] == [{"file": "table4.dat", "sha256": dd._sha256(f), "bytes": f.stat().st_size}] and "Walker" in man["citation"]


def test_angular_separation_small_angles():
    assert float(angular_separation(C_RA, C_DEC, C_RA, C_DEC + 1.0 / 3600.0)) == pytest.approx(math.pi / 648000.0, rel=1e-9)
    assert float(angular_separation(10.0, 0.0, 190.0, 0.0)) == pytest.approx(math.pi)
