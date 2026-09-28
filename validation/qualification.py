"""Cualificación de instrumentos (E8-Q) — tipo de artefacto del programa (orden del autor, 28-sep-2026).

Una cualificación NO es un experimento: no tiene letras ni umbrales de desenlace. Publica los límites
computables de un instrumento (suelo de error de energía, banda ultravioleta, ventana lineal, coste)
en función de sus parámetros, y CRITERIOS DE EXCLUSIÓN calculables antes de correr. Regla nueva del
programa: una preinscripción solo puede congelar puertas que una cualificación haya mostrado alcanzables
(`assert_gate_attainable`); no relaja ninguna regla ya congelada.

Artefacto: results/<fecha>_qualification_<instrumento>/qualification.{json,md} con
`artifact_type = "qualification"`, sin ninguna de las claves prohibidas (`verdict`, `letter`, `letters`).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_TYPE = "qualification"
E8Q = ("E8-Q: cualificación de instrumento — sin letras ni umbrales de desenlace; publica límites computables "
       "del instrumento y criterios de exclusión calculables antes de correr; una preinscripción solo puede "
       "congelar puertas que la cualificación haya mostrado alcanzables")
FORBIDDEN_KEYS = {"verdict", "letter", "letters", "desenlace", "outcome"}


def _check_no_letters(obj, path="") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"una cualificación no puede llevar '{k}' ({path or 'raíz'}): E8-Q")
            _check_no_letters(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_letters(v, f"{path}[{i}]")


def _fmt(v, depth: int = 0) -> str:
    """Representación legible para el .md: números con 4 cifras, dicts pequeños en línea, grandes resumidos."""
    if isinstance(v, float):
        return "∞" if math.isinf(v) else ("NaN" if math.isnan(v) else f"{v:.4g}")
    if isinstance(v, dict):
        if len(v) > 12:
            return f"({len(v)} entradas; véase la tabla o el JSON)"
        if depth > 0 and len(v) > 4:
            return f"({len(v)} entradas; véase el JSON)"
        return "{" + ", ".join(f"{k}: {_fmt(x, depth + 1)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x, depth + 1) for x in v) + "]" if len(v) <= 12 else f"[{len(v)} elementos]"
    return str(v)


def write_qualification(outdir: Path, instrument: str, code_commit: str, grid: dict, runs: list,
                        limits: dict, exclusion_criteria: dict, notes: list, md_extra: list | None = None) -> str:
    """Escribe qualification.json y qualification.md; devuelve el sha256 del JSON."""
    doc = {"artifact_type": ARTIFACT_TYPE, "status": E8Q, "instrument": instrument, "code_commit": code_commit,
           "written_utc": datetime.now(timezone.utc).isoformat(), "grid": grid, "runs": runs,
           "limits": limits, "exclusion_criteria": exclusion_criteria, "notes": notes}
    _check_no_letters(doc)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "qualification.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    md = [f"# Cualificación del instrumento «{instrument}» (E8-Q)", "",
          f"Commit `{code_commit[:9]}`; `qualification.json` sha256 `{sha}`. {E8Q}.", "",
          "## Rejilla", ""] + [f"- **{k}**: {_fmt(v)}" for k, v in grid.items()] + ["", "## Límites medidos", ""] + \
         [f"- **{k}**: {_fmt(v)}" for k, v in limits.items()] + ["", "## Criterios de exclusión (calculables antes de correr)", ""] + \
         [f"- **{k}**: {_fmt(v)}" for k, v in exclusion_criteria.items()] + ["", "## Notas", ""] + [f"- {n}" for n in notes]
    if md_extra:
        md += [""] + md_extra
    (outdir / "qualification.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return sha


def load_qualification(path: Path) -> dict:
    p = Path(path)
    p = p / "qualification.json" if p.is_dir() else p
    if not p.exists():
        raise SystemExit(f"FALLO CERRADO: falta la cualificación {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("artifact_type") != ARTIFACT_TYPE:
        raise SystemExit(f"FALLO CERRADO: {p} no es una cualificación")
    _check_no_letters(doc)
    doc["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return doc


def assert_gate_attainable(qual: dict, limit_key: str, gate_value: float, factor: float = 3.0, larger_is_looser: bool = True) -> dict:
    """Regla del programa: la puerta preinscrita debe ser ≥ factor × el suelo medido (para tolerancias que
    crecen al relajarse, p. ej. |ΔE/E|). Devuelve el registro para la preinscripción; falla cerrado si no."""
    floor = qual["limits"][limit_key]
    if isinstance(floor, dict):
        floor = floor["value"]
    ok = gate_value >= factor * floor if larger_is_looser else gate_value <= floor / factor
    if not ok:
        raise SystemExit(f"FALLO CERRADO: la puerta {limit_key} = {gate_value:g} no es alcanzable según la cualificación "
                         f"(suelo {floor:g}, factor {factor:g})")
    return {"limit_key": limit_key, "gate": gate_value, "measured_floor": floor, "factor": factor,
            "qualification_sha256": qual["_sha256"], "qualification_instrument": qual["instrument"]}


def sheet_cell_excluded(gamma_uv: float, t_window: float, amp_nonlinear: float, amp_noise: float) -> dict:
    """Criterio de exclusión de las láminas: la banda UV, creciendo desde el ruido de redondeo A_ruido a la
    tasa γ_UV, alcanza la amplitud no lineal A_nl antes de que cierre la ventana lineal del modo sembrado
    (t_ventana) si γ_UV·t_ventana > ln(A_nl/A_ruido). Calculable antes de correr con la γ_UV medida."""
    budget = math.log(amp_nonlinear / amp_noise) if amp_noise > 0 else math.inf
    return {"gamma_uv_times_t_window": gamma_uv * t_window, "ln_budget": budget,
            "excluded": bool(gamma_uv * t_window > budget)}
