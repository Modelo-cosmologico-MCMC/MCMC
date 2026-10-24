"""Perfil binado σ_los(R) desde el catálogo ESTELAR de Walker, Mateo & Olszewski 2009 (CDS J/AJ/137/3100): el binning
de la preinscripción «perfil σ_los(R) de Sculptor» (frente 5d) y de las que reutilicen ese observable.

La regla congelada (results/2026-09-21_sculptor_profile/preregistration.json, sha256 604f9c25…) fija: miembros con
probabilidad ≥ 0.9 (columna del catálogo); bins de igual número (≈ 150 estrellas) en R proyectado ≤ 1 kpc; σ_los por bin
= desviación típica corregida por errores de medida (⟨e²⟩ restado); error del bin σ/√(2N). Lo que esa regla no fijaba
(lectura del esquema, la unidad «estrella» con épocas repetidas, centro, distancia, R del bin, reparto del resto, orden
de las letras) lo fija el ANEXO congelado antes de los bytes (binning_annex.json, mismo directorio). Este módulo ejecuta
el anexo y nada más: sin recorte de binarias, sin corrección de rotación ni de perspectiva, sin recorte sigma.

El esquema se lee del ReadMe oficial (descripción byte a byte de CDS), no se supone: los papeles (posición, velocidad
heliocéntrica y su error, probabilidad de pertenencia, identificador) se asignan con las reglas de `assign_roles` y toda
ambigüedad falla cerrado (SchemaError → INDETERMINADO, «esquema no confirmado»).

Estado: sin bytes (walker2009 DATA_UNAVAILABLE); las pruebas usan catálogos sintéticos que ejercitan el instrumento y no
son contraste.
"""

from __future__ import annotations

import fnmatch
import math
import re
from pathlib import Path

import numpy as np

ARCSEC_RAD = math.pi / (180.0 * 3600.0)
ARCMIN_RAD = 60.0 * ARCSEC_RAD


class SchemaError(RuntimeError):
    """El ReadMe o la tabla no permiten asignar sin ambigüedad lo que el anexo exige (esquema no confirmado)."""


_SECTION = re.compile(r"^Byte-by-byte Description of file:\s*(.+?)\s*$")
_COLUMN = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s+([AIFE])(\d+)(?:\.\d+)?\s+(\S+)\s+(\S+)(?:\s+(.*))?$")
_FLAG = re.compile(r"^[nflqru]_")                       # prefijos de CDS para notas, banderas y límites


def parse_readme(text: str, filename: str) -> list[dict]:
    """Columnas de `filename` según la sección «Byte-by-byte Description of file: …» del ReadMe de CDS (admite los
    patrones de CDS, p. ej. table[2-5].dat). Cada columna: start y end (1-indexados, inclusivos), kind (A/I/F/E), unit,
    label y explanation (con las líneas de continuación unidas)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m and any(fnmatch.fnmatchcase(filename, p) for p in re.split(r"[\s,]+", m.group(1)) if p):
            start = i + 1
            break
    if start is None:
        raise SchemaError(f"el ReadMe no describe {filename} byte a byte")
    cols, in_body = [], False
    for line in lines[start:]:
        if not in_body:
            in_body = "Bytes" in line and "Label" in line
            continue
        if not line.strip() or line.startswith("=") or (line.startswith("-") and set(line.strip()) == {"-"}):
            if cols:
                break
            continue
        m = _COLUMN.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            cols.append({"start": a, "end": b, "kind": m.group(3), "unit": m.group(5), "label": m.group(6),
                         "explanation": (m.group(7) or "").strip()})
        elif cols:
            cols[-1]["explanation"] = (cols[-1]["explanation"] + " " + line.strip()).strip()
    if not cols:
        raise SchemaError(f"la sección byte a byte de {filename} no tiene columnas legibles")
    return cols


def readme_record_count(text: str, filename: str) -> int | None:
    """Número de registros que el «File Summary» del ReadMe declara para `filename` (None si no lo declara)."""
    in_summary = False
    for line in text.splitlines():
        if line.startswith("File Summary"):
            in_summary = True
            continue
        if in_summary:
            if line.startswith("Byte-by-byte") or line.startswith("See also"):
                break
            m = re.match(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s", line + " ")
            if m and fnmatch.fnmatchcase(filename, m.group(1)):
                return int(m.group(3))
    return None


def _is_error(label: str) -> bool:
    return label.startswith(("e_", "E_", "s_"))


def assign_roles(cols: list[dict]) -> dict:
    """Papeles exigidos por el anexo, con las reglas congeladas en él:
    - posición: RAh/RAm/RAs + DE-/DEd/DEm/DEs (sexagesimal de CDS) o RAdeg/DEdeg (o RAJ2000/DEJ2000 en grados);
    - velocidad: la única columna en km/s que no es un error; si hay varias, la única heliocéntrica;
    - su error: la columna e_<velocidad>;
    - pertenencia: la única columna real (F/E) que no es un error y cuya explicación habla de pertenencia
      («member…», sin «proper») o cuya etiqueta es PM/Pmem/Prob;
    - identificador (opcional): la única columna de texto que identifica la estrella; si no hay una sola, la estrella
      se define solo por posición.
    Cualquier otra situación falla cerrado (SchemaError)."""
    by = {c["label"]: c for c in cols}
    if all(k in by for k in ("RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs")):
        position = {"format": "sexagesimal", "labels": ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]}
    elif all(k in by for k in ("RAdeg", "DEdeg")):
        position = {"format": "deg", "labels": ["RAdeg", "DEdeg"]}
    elif all(k in by and by[k]["unit"] == "deg" for k in ("RAJ2000", "DEJ2000")):
        position = {"format": "deg", "labels": ["RAJ2000", "DEJ2000"]}
    else:
        raise SchemaError("posición: ni RAh/RAm/RAs/DE-/DEd/DEm/DEs ni RAdeg/DEdeg en la descripción byte a byte")
    vel = [c for c in cols if c["unit"] == "km/s" and c["kind"] in "FEI" and not _is_error(c["label"])]
    if len(vel) > 1:
        vel = [c for c in vel if "helio" in c["explanation"].lower() or c["label"].lower() in ("hv", "hrv", "vhel", "rvhel")]
    if len(vel) != 1:
        raise SchemaError(f"velocidad: {len(vel)} columnas candidatas en km/s (se exige una, heliocéntrica)")
    v = vel[0]["label"]
    if "e_" + v not in by:
        raise SchemaError(f"error de la velocidad: falta la columna e_{v}")
    pm = [c for c in cols if c["kind"] in "FE" and not _is_error(c["label"])
          and (("member" in c["explanation"].lower() and "proper" not in c["explanation"].lower())
               or c["label"].lower() in ("pm", "pmem", "p_mem", "prob"))]
    if len(pm) != 1:
        raise SchemaError(f"pertenencia: {len(pm)} columnas candidatas (se exige una probabilidad real)")
    ids = [c for c in cols if c["kind"] == "A" and c["label"] not in ("DE-", "DE+") and not _FLAG.match(c["label"])
           and (re.search(r"identif|designation|\bname\b|target|\bstar\b|\bid\b", c["explanation"], re.I)
                or c["label"].lower() in ("id", "name", "target", "star", "seq"))]
    return {"position": position, "velocity": v, "velocity_error": "e_" + v, "membership": pm[0]["label"],
            "identifier": ids[0]["label"] if len(ids) == 1 else None}


def read_table(path: Path, cols: list[dict]) -> tuple[dict, dict]:
    """Valores por etiqueta (A → str o None si vacío; I/F/E → float, NaN si vacío) y contadores de lectura. Las líneas
    idénticas byte a byte cuentan una vez (regla del anexo)."""
    raw = [ln for ln in Path(path).read_text(encoding="latin-1").splitlines() if ln.strip()]
    seen, lines = set(), []
    for ln in raw:
        if ln not in seen:
            seen.add(ln)
            lines.append(ln)
    out = {}
    for c in cols:
        vals = []
        for ln in lines:
            s = ln[c["start"] - 1:c["end"]].strip()
            if c["kind"] == "A":
                vals.append(s or None)
            elif not s:
                vals.append(math.nan)
            else:
                try:
                    vals.append(float(s))
                except ValueError as exc:
                    raise SchemaError(f"{c['label']}: valor no numérico {s!r}") from exc
        out[c["label"]] = vals if c["kind"] == "A" else np.asarray(vals, dtype=float)
    return out, {"lines": len(raw), "identical_lines_dropped": len(raw) - len(lines)}


def positions_deg(t: dict, roles: dict) -> tuple[np.ndarray, np.ndarray]:
    lab = roles["position"]["labels"]
    if roles["position"]["format"] == "deg":
        return np.asarray(t[lab[0]], float), np.asarray(t[lab[1]], float)
    ra = 15.0 * (t["RAh"] + t["RAm"] / 60.0 + t["RAs"] / 3600.0)
    sign = np.array([-1.0 if s == "-" else 1.0 for s in t["DE-"]])
    return ra, sign * (t["DEd"] + t["DEm"] / 60.0 + t["DEs"] / 3600.0)


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def angular_separation(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Separación angular en radianes (forma atan2, exacta también a separaciones pequeñas)."""
    u, w = _unit(ra1, dec1), _unit(ra2, dec2)
    return np.arctan2(np.linalg.norm(np.cross(u, w), axis=-1), np.sum(u * w, axis=-1))


def group_stars(ra, dec, ident, merge_arcsec: float) -> tuple[np.ndarray, dict]:
    """Unidad «estrella» del anexo: dos registros son la misma estrella si comparten identificador o si distan menos de
    `merge_arcsec` (cierre transitivo). Devuelve el índice de estrella de cada registro (estrellas numeradas por orden
    de primera aparición) y diagnósticos."""
    n = len(ra)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    if ident is not None:
        first = {}
        for i, s in enumerate(ident):
            if s is not None:
                if s in first:
                    union(first[s], i)
                else:
                    first[s] = i
    u = _unit(ra, dec)
    cos_lim = math.cos(merge_arcsec * ARCSEC_RAD)
    by_position = 0
    for i0 in range(0, n, 512):
        ii, jj = np.nonzero(u[i0:i0 + 512] @ u.T > cos_lim)
        for a, b in zip(ii + i0, jj):
            if a < b:
                if ident is None or ident[a] is None or ident[a] != ident[b]:
                    by_position += find(a) != find(b)
                union(int(a), int(b))
    roots = [find(i) for i in range(n)]
    index, star = {}, np.empty(n, dtype=int)
    for i, r in enumerate(roots):
        star[i] = index.setdefault(r, len(index))
    split_ids = 0
    if ident is not None:
        by_id = {}
        for i, s in enumerate(ident):
            if s is not None:
                by_id.setdefault(s, []).append(i)
        for idx in (np.array(m) for m in by_id.values() if len(m) > 1):
            sep = angular_separation(ra[idx][:, None], dec[idx][:, None], ra[idx][None, :], dec[idx][None, :])
            split_ids += bool(float(np.max(sep)) > merge_arcsec * ARCSEC_RAD)
    return star, {"merges_by_position_only": int(by_position), "identifiers_with_records_beyond_merge_radius": split_ids}


def stars_from_catalogue(readme_text: str, table_path: Path, filename: str, merge_arcsec: float) -> dict:
    """Del ReadMe y la tabla a una fila por estrella: velocidad combinada por media pesada por inverso de varianza de sus
    épocas (error (Σ e⁻²)^−½), posición media, pertenencia = mínimo de los P_M presentes de sus épocas (NaN si ninguno),
    número de épocas. Se descartan los registros sin velocidad, sin error positivo o sin posición."""
    cols = parse_readme(readme_text, filename)
    roles = assign_roles(cols)
    t, counts = read_table(table_path, cols)
    declared = readme_record_count(readme_text, filename)
    if declared is not None and declared != counts["lines"]:
        raise SchemaError(f"{filename}: {counts['lines']} registros frente a {declared} declarados en el ReadMe")
    ra, dec = positions_deg(t, roles)
    v, e = t[roles["velocity"]], t[roles["velocity_error"]]
    pm = t[roles["membership"]]
    finite_pm = pm[np.isfinite(pm)]
    if finite_pm.size and (finite_pm.min() < 0.0 or finite_pm.max() > 1.0):
        raise SchemaError(f"pertenencia fuera de [0, 1] en {roles['membership']}: no es una probabilidad")
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0) & np.isfinite(ra) & np.isfinite(dec)
    keep = np.flatnonzero(ok)
    ident = [t[roles["identifier"]][i] for i in keep] if roles["identifier"] else None
    star, gdiag = group_stars(ra[keep], dec[keep], ident, merge_arcsec)
    ns = int(star.max()) + 1 if star.size else 0
    w = 1.0 / e[keep] ** 2
    sw = np.bincount(star, weights=w, minlength=ns)
    v_star = np.bincount(star, weights=w * v[keep], minlength=ns) / sw
    uvec = np.zeros((ns, 3))
    np.add.at(uvec, star, _unit(ra[keep], dec[keep]))
    uvec /= np.linalg.norm(uvec, axis=1)[:, None]
    pm_min = np.full(ns, np.inf)
    pm_max = np.full(ns, -np.inf)
    pk = pm[keep]
    fin = np.isfinite(pk)
    np.minimum.at(pm_min, star[fin], pk[fin])
    np.maximum.at(pm_max, star[fin], pk[fin])
    pm_min[np.isinf(pm_min)] = np.nan
    n_ep = np.bincount(star, minlength=ns)
    return {"roles": roles, "columns": cols,
            "ra_deg": np.degrees(np.arctan2(uvec[:, 1], uvec[:, 0])) % 360.0, "dec_deg": np.degrees(np.arcsin(uvec[:, 2])),
            "v_kms": v_star, "e_kms": sw ** -0.5, "pm": pm_min, "pm_max": pm_max, "n_epochs": n_ep,
            "diagnostics": {**counts, "records_dropped_missing": int((~ok).sum()), "stars": ns,
                            "stars_with_repeat_epochs": int((n_ep > 1).sum()), **gdiag,
                            "records_with_membership": int(np.isfinite(pm).sum())}}


def equal_number_bins(R_pc, v, e, target_per_bin: int, R_max_pc: float) -> list[dict]:
    """Bins de igual número en R ≤ R_max: n_bins = ⌊N/objetivo + ½⌋ (mínimo 1 si N > 0); orden estable por R (empates por
    orden de aparición); reparto con el resto en los bins internos (np.array_split). Por bin: N, R mediana, R mínimo y
    máximo, media propia del bin, s² (ddof = 1), ⟨e²⟩, σ = √(s² − ⟨e²⟩) (NaN si s² ≤ ⟨e²⟩) y σ_err = σ/√(2N)."""
    R_pc, v, e = (np.asarray(x, dtype=float) for x in (R_pc, v, e))
    sel = np.flatnonzero(R_pc <= R_max_pc)
    order = sel[np.argsort(R_pc[sel], kind="stable")]
    n = order.size
    n_bins = int(math.floor(n / target_per_bin + 0.5)) if n else 0
    if n and n_bins == 0:
        n_bins = 1
    bins = []
    for idx in (np.array_split(order, n_bins) if n_bins else []):
        vb, eb = v[idx], e[idx]
        s2 = float(np.var(vb, ddof=1)) if idx.size > 1 else math.nan
        e2 = float(np.mean(eb ** 2))
        sig = math.sqrt(s2 - e2) if s2 > e2 else math.nan
        bins.append({"N": int(idx.size), "R_pc": float(np.median(R_pc[idx])), "R_min_pc": float(R_pc[idx].min()),
                     "R_max_pc": float(R_pc[idx].max()), "v_mean_kms": float(vb.mean()), "s2": s2, "e2_mean": e2,
                     "sigma_kms": sig, "sigma_err_kms": math.nan if math.isnan(sig) else sig / math.sqrt(2.0 * idx.size)})
    return bins


def binned_profile(stars: dict, centre_ra_deg: float, centre_dec_deg: float, distance_kpc: float, pm_min: float,
                   R_max_pc: float, target_per_bin: int) -> dict:
    """Miembros (P_M de la estrella ≥ pm_min), R = D·θ circular desde el centro fijado, bins de igual número; con el
    desplazamiento del centroide de los miembros respecto del centro (diagnóstico y puerta del anexo)."""
    pm = stars["pm"]
    mem = np.isfinite(pm) & (pm >= pm_min)
    pm_max = stars["pm_max"]
    straddle = int((np.isfinite(pm) & (pm < pm_min) & (pm_max >= pm_min)).sum())
    ra, dec = stars["ra_deg"][mem], stars["dec_deg"][mem]
    R = 1000.0 * distance_kpc * angular_separation(ra, dec, centre_ra_deg, centre_dec_deg)
    if mem.any():
        c = _unit(ra, dec).sum(axis=0)
        c_ra, c_dec = math.degrees(math.atan2(c[1], c[0])) % 360.0, math.degrees(math.asin(c[2] / np.linalg.norm(c)))
        offset = float(angular_separation(c_ra, c_dec, centre_ra_deg, centre_dec_deg)) / ARCMIN_RAD
    else:
        offset = math.nan
    bins = equal_number_bins(R, stars["v_kms"][mem], stars["e_kms"][mem], target_per_bin, R_max_pc)
    return {"bins": bins, "members": int(mem.sum()), "members_within_R_max": int((R <= R_max_pc).sum()),
            "stars_pm_straddling_threshold": straddle, "member_centroid_offset_arcmin": offset}


def frozen_letter(bins: list[dict], pred_by_beta: dict, chi2_A_max: float, chi2_B_max: float, C_z_min: float,
                  C_inner_bins: int, min_bins: int) -> dict:
    """Letra con el orden del anexo (puertas → C → B → A → INDETERMINADO) sobre la regla congelada:
    χ² = Σ_bins [(σ_pred − σ_obs)/σ_err]², ν = N_bins; A: χ²_ν ≤ chi2_A_max para algún β; B: chi2_A_max < χ²_ν(mejor β)
    ≤ chi2_B_max; C: en los C_inner_bins bins internos (σ_pred − σ_obs)/σ_err > C_z_min para todo β. Puertas: N_bins <
    min_bins o algún bin sin σ definida. Si ninguna letra aplica, INDETERMINADO (sin reasignar a B ni a C)."""
    if len(bins) < min_bins:
        return {"letter": "INDETERMINADO", "reason": f"N_bins = {len(bins)} < {min_bins}"}
    sig = np.array([b["sigma_kms"] for b in bins])
    err = np.array([b["sigma_err_kms"] for b in bins])
    if not np.all(np.isfinite(sig)):
        return {"letter": "INDETERMINADO", "reason": "algún bin con s² ≤ ⟨e²⟩: σ del bin sin definir"}
    per = {}
    for beta, pred in pred_by_beta.items():
        z = (np.asarray(pred, dtype=float) - sig) / err
        chi2 = float(np.sum(z ** 2))
        per[beta] = {"chi2": chi2, "chi2_nu": chi2 / len(bins), "z": z.tolist(), "z_inner": z[:C_inner_bins].tolist()}
    best = min(per, key=lambda b: per[b]["chi2_nu"])
    c_holds = all(min(per[b]["z_inner"]) > C_z_min for b in per)
    x = per[best]["chi2_nu"]
    if c_holds:
        letter, reason = "C", f"los {C_inner_bins} bins internos excluyen la predicción a > {C_z_min}σ para todo β"
    elif chi2_A_max < x <= chi2_B_max:
        letter, reason = "B", f"{chi2_A_max} < χ²_ν(mejor β) = {x:.3g} ≤ {chi2_B_max}"
    elif x <= chi2_A_max:
        letter, reason = "A", f"χ²_ν(mejor β) = {x:.3g} ≤ {chi2_A_max}"
    else:
        letter, reason = "INDETERMINADO", f"ninguna letra A/B/C aplica: χ²_ν(mejor β) = {x:.3g} > {chi2_B_max} sin la exclusión de C"
    return {"letter": letter, "reason": reason, "best_beta": best, "per_beta": per}
