"""Ingesta SPARC alineada con data/sources.yaml (PR-S, 27-oct-2026) — falla cerrado, inspecciona antes de asumir.

CONTRATO (preinscripción del 5E + regla de fallo + capa de datos de data/sources.yaml):

1. data/sources.yaml manda: cinco ficheros (SPARC_Lelli2016c.mrt, MassModels_Lelli2016c.mrt, Rotmod_LTG.zip, RAR.mrt,
   RARbins.mrt) en data/raw/sparc/. `python scripts/download_data.py sparc` los baja de astroweb.cwru.edu (o, si el autor
   los entrega a mano en el raw_dir, no descarga: verifica y registra) y escribe data/manifests/sparc.json en
   SCHEMA_UNVERIFIED, con los sha256 computados.
2. Sin manifest, TODO falla cerrado (DATA_UNAVAILABLE); con el manifest en SCHEMA_UNVERIFIED, también: el único acceso
   científico es `load_sparc_catalogue`, que pasa por mcmc_ontology.data_registry.require_available. Nunca se fabrican
   fixtures observacionales ni se asume un mirror.
3. La PRIMERA acción con bytes es el schema_report: `python -m dynamics.sparc_data --inspect` escribe
   data/manifests/sparc.schema_report.md con la descripción byte a byte de cada tabla MRT, las cabeceras de Rotmod_LTG
   y los papeles que el parser asignaría (o el motivo por el que no puede). Solo cuando el autor lo confirma, el
   manifest pasa a AVAILABLE con la transformación «esquema confirmado: data/manifests/sparc.schema_report.md».
4. El parser no supone columnas: lee la descripción byte a byte de cada MRT (el formato de CDS, el mismo lector que
   dynamics/walker_binning.py) y asigna papeles por etiqueta y unidad; en Rotmod_LTG lee la línea de nombres y la de
   unidades de cada fichero. Toda ambigüedad falla cerrado (SchemaError → INDETERMINADO «esquema no confirmado»).
   Conserva unidades, identificador y bandera de calidad; valida radios > 0 y crecientes, errores > 0, sin NaN ni
   duplicados (validate_catalogue); comprueba que la distancia de cada Rotmod coincide con la de la tabla maestra y
   que las curvas coinciden con MassModels_Lelli2016c (crosscheck_mass_models).

El estatuto vigente de la ingesta vive en data/sparc/INGESTA_ESTADO.md.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from dynamics.walker_binning import SchemaError, parse_readme
from mcmc_ontology.data_registry import manifest_path, require_available

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "sparc"
FILES = ("SPARC_Lelli2016c.mrt", "MassModels_Lelli2016c.mrt", "Rotmod_LTG.zip", "RAR.mrt", "RARbins.mrt")
SCHEMA_REPORT = ROOT / "data" / "manifests" / "sparc.schema_report.md"

_UNAVAILABLE = (
    "DATA_UNAVAILABLE: el catálogo SPARC no está ingerido (sin data/manifests/sparc.json). Estatuto y desbloqueo: "
    "data/sparc/INGESTA_ESTADO.md. El análisis observacional falla CERRADO por contrato — no usa fixtures ni mirrors."
)

# Papeles exigidos: etiqueta del MRT → unidades admitidas (la etiqueta y la unidad deben aparecer una sola vez).
MASTER_ROLES = {"Galaxy": ("---",), "D": ("Mpc",), "e_D": ("Mpc",), "Inc": ("deg",), "e_Inc": ("deg",),
                "Rdisk": ("kpc",), "SBdisk": ("solLum/pc2", "Lsun/pc2", "L/pc2"), "Q": ("---",)}
MASSMODEL_ROLES = {"ID": ("---",), "D": ("Mpc",), "R": ("kpc",), "Vobs": ("km/s",), "e_Vobs": ("km/s",),
                   "Vgas": ("km/s",), "Vdisk": ("km/s",), "Vbul": ("km/s",)}
ROTMOD_ROLES = {"Rad": "kpc", "Vobs": "km/s", "errV": "km/s", "Vgas": "km/s", "Vdisk": "km/s", "Vbul": "km/s"}
_DIST = re.compile(r"^#\s*Distance\s*=\s*([0-9.eE+-]+)\s*Mpc\s*$")


# ------------------------------------------------------------------ tablas MRT (formato byte a byte de CDS/AAS)
def mrt_columns(path: Path) -> list[dict]:
    """Columnas de un fichero MRT según su propia descripción byte a byte."""
    p = Path(path)
    return parse_readme(p.read_text(encoding="latin-1"), p.name)


def mrt_data_lines(path: Path) -> list[str]:
    """Las líneas de datos de un MRT: las que siguen a la última línea separadora (solo guiones) de la cabecera."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    seps = [i for i, ln in enumerate(lines) if re.fullmatch(r"-{20,}", ln.strip() or "x")]
    if not seps:
        raise SchemaError(f"{Path(path).name}: sin separador de cabecera")
    return [ln for ln in lines[seps[-1] + 1:] if ln.strip()]


def read_mrt(path: Path) -> tuple[list[dict], dict]:
    """Valores por etiqueta de un MRT (A → str o None; I/F/E → float, NaN si vacío) y sus columnas."""
    cols = mrt_columns(path)
    lines = mrt_data_lines(path)
    out = {}
    for c in cols:
        vals = []
        for ln in lines:
            s = ln[c["start"] - 1:c["end"]].strip()
            if c["kind"] == "A":
                vals.append(s or None)
            elif not s:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(s))
                except ValueError as exc:
                    raise SchemaError(f"{Path(path).name}, {c['label']}: valor no numérico {s!r}") from exc
        out[c["label"]] = vals if c["kind"] == "A" else np.asarray(vals, dtype=float)
    return cols, out


def assign_mrt_roles(cols: list[dict], required: dict, name: str) -> dict:
    """Comprueba que cada etiqueta exigida aparece una sola vez y con una unidad admitida."""
    for label, units in required.items():
        hits = [c for c in cols if c["label"] == label]
        if len(hits) != 1:
            raise SchemaError(f"{name}: la etiqueta {label!r} aparece {len(hits)} veces (se exige una)")
        if hits[0]["unit"] not in units:
            raise SchemaError(f"{name}: {label!r} en {hits[0]['unit']!r}, se esperaba {' o '.join(units)}")
    return {label: label for label in required}


# ------------------------------------------------------------------ Rotmod_LTG
def parse_rotmod(text: str, member: str) -> dict:
    """Un fichero <galaxia>_rotmod.dat: la distancia de la cabecera, la línea de nombres y la de unidades, y las
    columnas por papel. Fallo cerrado si falta algo o si los papeles no son únicos."""
    lines = text.splitlines()
    dist = [float(m.group(1)) for m in (_DIST.match(ln.strip()) for ln in lines) if m]
    if len(dist) != 1:
        raise SchemaError(f"{member}: {len(dist)} líneas «# Distance = … Mpc» (se exige una)")
    head = [ln.lstrip("#").split() for ln in lines if ln.startswith("#") and not _DIST.match(ln.strip())]
    names = [h for h in head if "Rad" in h and "Vobs" in h]
    if len(names) != 1:
        raise SchemaError(f"{member}: no hay una única línea de nombres con Rad y Vobs")
    names = names[0]
    units = head[head.index(names) + 1] if head.index(names) + 1 < len(head) else []
    if len(units) != len(names):
        raise SchemaError(f"{member}: la línea de unidades no casa con la de nombres")
    idx = {}
    for role, unit in ROTMOD_ROLES.items():
        hits = [i for i, n in enumerate(names) if n == role]
        if len(hits) != 1:
            raise SchemaError(f"{member}: la columna {role!r} aparece {len(hits)} veces")
        if units[hits[0]] != unit:
            raise SchemaError(f"{member}: {role!r} en {units[hits[0]]!r}, se esperaba {unit!r}")
        idx[role] = hits[0]
    rows = [ln.split() for ln in lines if ln.strip() and not ln.startswith("#")]
    if any(len(r) != len(names) for r in rows):
        raise SchemaError(f"{member}: filas con un número de columnas distinto de la cabecera")
    data = np.array([[float(x) for x in r] for r in rows], dtype=float)
    return {"distance_mpc": dist[0], **{role: data[:, i] for role, i in idx.items()}}


def read_rotmod_zip(path: Path) -> dict:
    """Todas las curvas de Rotmod_LTG.zip, por galaxia (el nombre sale del fichero, <galaxia>_rotmod.dat)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.endswith("_rotmod.dat")]
        if not members:
            raise SchemaError("Rotmod_LTG.zip no contiene ficheros *_rotmod.dat")
        for m in members:
            name = Path(m).name[: -len("_rotmod.dat")]
            if name in out:
                raise SchemaError(f"Rotmod_LTG.zip: la galaxia {name!r} aparece dos veces")
            out[name] = parse_rotmod(z.read(m).decode("latin-1"), m)
    return out


# ------------------------------------------------------------------ validación y catálogo
def validate_catalogue(radii: np.ndarray, v_obs: np.ndarray, e_v: np.ndarray, galaxy_ids: np.ndarray) -> dict:
    """Validación física por galaxia: radios > 0 y estrictamente crecientes, errores de velocidad > 0, sin NaN y sin
    duplicados (galaxy_id, R). Lanza ValueError si algo falla; devuelve el resumen."""
    r = np.asarray(radii, float)
    v = np.asarray(v_obs, float)
    e = np.asarray(e_v, float)
    g = np.asarray(galaxy_ids)
    if np.isnan(r).any() or np.isnan(v).any() or np.isnan(e).any():
        raise ValueError("NaN en el catálogo")
    if (r <= 0.0).any():
        raise ValueError("radios no estrictamente positivos")
    if (e <= 0.0).any():
        raise ValueError("errores de velocidad no positivos")
    keys = list(zip(g.tolist(), r.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicados (galaxia, R)")
    for gal in np.unique(g):
        rg = r[g == gal]
        if rg.size > 1 and not np.all(np.diff(rg) > 0.0):
            raise ValueError(f"radios no crecientes dentro de la galaxia {gal!r}")
    return {"n_points": int(r.size), "n_galaxies": int(np.unique(g).size)}


KPC_TO_PC = 1000.0      # SPARC publica radios en kpc; el 5E trabaja en pc (columna DERIVADA, nunca implícita)


def crosscheck_mass_models(curves: dict, mm: dict, tol_kms: float = 0.051, tol_kpc: float = 0.0051) -> list[str]:
    """Galaxias cuyas curvas de Rotmod_LTG no coinciden con MassModels_Lelli2016c (radios, Vobs y componentes, al
    redondeo de las tablas). Lista vacía si todo casa."""
    bad = []
    ids = np.asarray(mm["ID"], dtype=object)
    for name, cv in curves.items():
        sel = ids == name
        if not sel.any():
            bad.append(f"{name}: no está en MassModels")
            continue
        pairs = (("R", "Rad", tol_kpc), ("Vobs", "Vobs", tol_kms), ("e_Vobs", "errV", tol_kms), ("Vgas", "Vgas", tol_kms),
                 ("Vdisk", "Vdisk", tol_kms), ("Vbul", "Vbul", tol_kms))
        for a, b, tol in pairs:
            x, y = np.asarray(mm[a])[sel], cv[b]
            if x.size != y.size or np.max(np.abs(x - y)) > tol:
                bad.append(f"{name}: {a} distinto")
                break
    return bad


def build_galaxies(raw_dir: Path) -> list:
    """Galaxias (dynamics.mcv_sparc.Galaxy) desde la tabla maestra y Rotmod_LTG, con las comprobaciones del contrato."""
    from dynamics.mcv_sparc import Galaxy
    raw = Path(raw_dir)
    mcols, master = read_mrt(raw / "SPARC_Lelli2016c.mrt")
    assign_mrt_roles(mcols, MASTER_ROLES, "SPARC_Lelli2016c.mrt")
    curves = read_rotmod_zip(raw / "Rotmod_LTG.zip")
    gcols, mm = read_mrt(raw / "MassModels_Lelli2016c.mrt")
    assign_mrt_roles(gcols, MASSMODEL_ROLES, "MassModels_Lelli2016c.mrt")
    bad = crosscheck_mass_models(curves, mm)
    if bad:
        raise SchemaError("Rotmod_LTG y MassModels_Lelli2016c no casan: " + "; ".join(bad[:5]))
    names = master["Galaxy"]
    if len(set(names)) != len(names):
        raise SchemaError("tabla maestra: identificadores repetidos")
    out = []
    for i, name in enumerate(names):
        if name not in curves:
            raise SchemaError(f"{name}: está en la tabla maestra y no en Rotmod_LTG")
        cv = curves[name]
        if abs(cv["distance_mpc"] - master["D"][i]) > 0.0051:
            raise SchemaError(f"{name}: la distancia de Rotmod ({cv['distance_mpc']}) no es la de la tabla maestra")
        validate_catalogue(cv["Rad"], cv["Vobs"], cv["errV"], np.full(cv["Rad"].size, name))
        out.append(Galaxy(name, cv["Rad"], cv["Vobs"], cv["errV"], cv["Vgas"], cv["Vdisk"], cv["Vbul"],
                          float(master["D"][i]), float(master["e_D"][i]), float(master["Inc"][i]),
                          float(master["e_Inc"][i]), R_d=float(master["Rdisk"][i]), SB_disk0=float(master["SBdisk"][i]),
                          quality=int(master["Q"][i])))
    extra = sorted(set(curves) - set(names))
    if extra:
        raise SchemaError(f"Rotmod_LTG trae galaxias fuera de la tabla maestra: {extra[:5]}")
    return out


def load_sparc_catalogue(raw_dir: Path | None = None) -> list:
    """Punto de entrada del análisis observacional (5E y PR-S): exige el dataset «sparc» AVAILABLE con el esquema
    confirmado y los sha256 re-verificados (require_available); si no, falla cerrado."""
    if raw_dir is None:
        if not manifest_path("sparc").exists():
            raise FileNotFoundError(_UNAVAILABLE)
        raw_dir = require_available("sparc")
    return build_galaxies(raw_dir)


# ------------------------------------------------------------------ schema_report
def inspect_schema(raw_dir: Path = RAW_DIR, n_lines: int = 40) -> str:
    """El schema_report de los cinco ficheros, ANTES de cualquier parser: la descripción byte a byte literal de cada MRT
    y su número de filas, la lista de Rotmod_LTG con la cabecera literal del primero, y los papeles que el parser
    asignaría o el motivo por el que no puede."""
    raw = Path(raw_dir)
    missing = [f for f in FILES if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(_UNAVAILABLE + f" (faltan {missing})")
    parts = ["# schema_report — SPARC (data/raw/sparc)\n",
             "Estado: SCHEMA_UNVERIFIED hasta que el autor confirme este informe.\n"]
    for f in FILES:
        p = raw / f
        parts.append(f"\n## {f}\n")
        if f.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                parts.append(f"Miembros: {len(names)}\n\n```\n" + "\n".join(names[:10]) + "\n```\n")
                first = next((m for m in names if m.endswith("_rotmod.dat")), None)
                if first:
                    head = z.read(first).decode("latin-1").splitlines()[:n_lines]
                    parts.append(f"\nCabecera literal de {first}:\n\n```\n" + "\n".join(head) + "\n```\n")
            try:
                parts.append(f"\nPapeles de Rotmod: {len(read_rotmod_zip(p))} curvas leídas con {list(ROTMOD_ROLES)}\n")
            except SchemaError as exc:
                parts.append(f"\nPapeles de Rotmod: NO ASIGNABLES ({exc})\n")
            continue
        text = p.read_text(encoding="latin-1").splitlines()
        cut = next((i for i, ln in enumerate(text) if "Byte-by-byte" in ln), 0)
        parts.append("```\n" + "\n".join(text[cut:cut + n_lines]) + "\n```\n")
        try:
            parts.append(f"\nFilas de datos: {len(mrt_data_lines(p))}\n")
            roles = {"SPARC_Lelli2016c.mrt": MASTER_ROLES, "MassModels_Lelli2016c.mrt": MASSMODEL_ROLES}.get(f)
            if roles:
                assign_mrt_roles(mrt_columns(p), roles, f)
                parts.append(f"Papeles: {list(roles)} (directos del catálogo)\n")
        except SchemaError as exc:
            parts.append(f"\nPapeles: NO ASIGNABLES ({exc})\n")
    parts.append("\nColumnas derivadas (nunca implícitas): R en pc = 1000·R en kpc (5E); V_bar² con los signos de SPARC; "
                 "M_b = r·V_bar²/G (PR-S).\n")
    return "".join(parts)


if __name__ == "__main__":
    import sys
    if "--inspect" in sys.argv:
        SCHEMA_REPORT.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_REPORT.write_text(inspect_schema(), encoding="utf-8")
        print(f"schema_report: {SCHEMA_REPORT}")
