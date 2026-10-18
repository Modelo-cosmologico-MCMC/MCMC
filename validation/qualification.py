"""Cualificación de instrumentos (E8-Q) — tipo de artefacto del programa (orden del autor, 28-sep-2026).

Una cualificación NO es un experimento: no tiene letras ni umbrales de desenlace. Publica los límites
computables de un instrumento (suelo de error de energía, banda ultravioleta, ventana lineal, coste)
en función de sus parámetros, y CRITERIOS DE EXCLUSIÓN calculables antes de correr. Regla nueva del
programa: una preinscripción solo puede congelar puertas que una cualificación haya mostrado alcanzables
(`assert_gate_attainable`); no relaja ninguna regla ya congelada.

Artefacto: results/<fecha>_qualification_<instrumento>/qualification.{json,md} con
`artifact_type = "qualification"`, sin ninguna de las claves prohibidas (`verdict`, `letter`, `letters`).

REGLA POR CELDA (revisión del autor del 30-sep a #46; para las preinscripciones congeladas desde el 18-oct, no
retroactiva). Una puerta ya no se compara con el máximo de la rejilla sino con el suelo de la MISMA celda, definida por
(brazo, N, ds, ε_soft, rango) — `CELL_FIELDS` —, medido en el MISMO instante de evaluación:
  * energía de los brazos de Cronos evaluada en t_weak: solo cuentan las corridas de la celda que salieron del régimen
    débil (su |ΔE/E| es el acumulado hasta la salida); las cortadas antes por pared o por el suelo de dt no miden ese
    suelo (`assert_cell_gate_attainable`);
  * presupuesto de pared declarado por celda: alcanzable si la misma celda llegó a su instante de evaluación en
    ≤ presupuesto/margen (`assert_wall_budget_attainable`);
  * estacionariedad por celda: el brazo newtoniano de la misma celda, medido hasta el mismo t
    (`assert_cell_stationarity_attainable`).
Si varias corridas de la celda llegan al instante de evaluación (p. ej. dt_min distinto), el suelo es el mayor y se
publican todas. Todo falla cerrado: sin celda equivalente, o sin corrida que llegue al instante de evaluación, la
puerta no se puede congelar. `cell_report` da el mismo diagnóstico sin fallar, para publicar tablas.
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


# ------------------------------------------------------------------ regla por celda (revisión del autor, 30-sep)
CELL_FIELDS = ("arm", "N", "ds", "eps_soft", "rank_update")
EVALUATIONS = ("t_weak", "t_end")


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    return a == b


def _cell_label(cell: dict, fields=CELL_FIELDS) -> str:
    return ", ".join(f"{f} = {cell.get(f)}" for f in fields)


def cell_report(qual: dict, cell: dict, metric: str = "dE_rel_max", evaluation: str = "t_weak",
                t_eval_myr: float | None = None, fields=CELL_FIELDS) -> dict:
    """Diagnóstico sin fallar: qué corridas de la cualificación son la MISMA celda y cuáles llegaron al instante de
    evaluación ('t_weak': salieron del régimen débil; 't_end': llegaron a t_eval_myr). status ∈ {'cualificada',
    'sin celda equivalente', 'no llegó al instante de evaluación'}; floor = el mayor de las que llegaron."""
    missing = [f for f in fields if f not in cell]
    if missing:
        raise ValueError(f"la celda no declara {missing}")
    if evaluation not in EVALUATIONS:
        raise ValueError(f"evaluación {evaluation!r} no es {EVALUATIONS}")
    if evaluation == "t_end" and t_eval_myr is None:
        raise ValueError("evaluación 't_end' sin t_eval_myr")
    same = [r for r in qual["runs"] if all(f in r and _same(r[f], cell[f]) for f in fields)]
    if evaluation == "t_weak":
        reached = [r for r in same if r.get("t_weak_myr") is not None]
    else:
        reached = [r for r in same if r.get("t_final_myr") is not None and r["t_final_myr"] >= t_eval_myr * (1.0 - 1e-9)]
    status = "cualificada" if reached else ("sin celda equivalente" if not same else "no llegó al instante de evaluación")
    return {"cell": {f: cell[f] for f in fields}, "metric": metric, "evaluation": evaluation, "t_eval_myr": t_eval_myr,
            "status": status, "floor": max(float(r[metric]) for r in reached) if reached else None,
            "runs_same_cell": [{"key": r.get("key"), "t_weak_myr": r.get("t_weak_myr"), "t_final_myr": r.get("t_final_myr"),
                                "stop_reason": r.get("stop_reason"), "wall_s": r.get("wall_s"), metric: r.get(metric)} for r in same],
            "runs_reaching_evaluation": [r.get("key") for r in reached],
            "wall_s_to_evaluation": max(float(r["wall_s"]) for r in reached) if reached else None,
            "qualification_sha256": qual.get("_sha256"), "qualification_instrument": qual.get("instrument")}


def _require_qualified(rep: dict, what: str) -> None:
    if rep["status"] != "cualificada":
        at = "" if rep["t_eval_myr"] is None else " a {} Myr".format(rep["t_eval_myr"])
        keys = [x["key"] for x in rep["runs_same_cell"]] or "ninguna"
        raise SystemExit(f"FALLO CERRADO: {what} de la celda ({_cell_label(rep['cell'], tuple(rep['cell']))}) no es congelable: "
                         f"{rep['status']} en la cualificación {rep['qualification_instrument']} "
                         f"(evaluación {rep['evaluation']}{at}; corridas de la celda: {keys})")


def assert_cell_gate_attainable(qual: dict, cell: dict, gate_value: float, metric: str = "dE_rel_max", factor: float = 3.0,
                                evaluation: str = "t_weak", t_eval_myr: float | None = None, fields=CELL_FIELDS) -> dict:
    """Regla por celda: la puerta de la celda debe ser ≥ factor × el suelo medido en la MISMA celda y en el MISMO
    instante de evaluación (energía de Cronos: 't_weak'). Devuelve el registro para la preinscripción; falla cerrado."""
    rep = cell_report(qual, cell, metric, evaluation, t_eval_myr, fields)
    _require_qualified(rep, f"la puerta {metric}")
    if not gate_value >= factor * rep["floor"]:
        raise SystemExit(f"FALLO CERRADO: la puerta {metric} = {gate_value:g} de la celda ({_cell_label(rep['cell'], fields)}) "
                         f"es menor que {factor:g} × el suelo de la celda ({rep['floor']:g})")
    return {"rule": "por celda (revisión del autor, 30-sep)", "gate": gate_value, "factor": factor, "measured_floor": rep["floor"],
            **{k: rep[k] for k in ("cell", "metric", "evaluation", "t_eval_myr", "runs_reaching_evaluation", "runs_same_cell",
                                   "qualification_sha256", "qualification_instrument")}}


def assert_wall_budget_attainable(qual: dict, cell: dict, budget_s: float, evaluation: str = "t_weak",
                                  t_eval_myr: float | None = None, margin: float = 1.0, fields=CELL_FIELDS) -> dict:
    """Presupuesto de pared declarado por celda: alcanzable si la MISMA celda llegó a su instante de evaluación en la
    cualificación con ≤ budget_s/margin segundos de pared (el mayor si hay varias). Falla cerrado si ninguna llegó: la
    cualificación no muestra que la celda alcance su instante de evaluación con ningún presupuesto."""
    rep = cell_report(qual, cell, "wall_s", evaluation, t_eval_myr, fields)
    _require_qualified(rep, "el presupuesto de pared")
    need = rep["wall_s_to_evaluation"]
    if not budget_s >= margin * need:
        raise SystemExit(f"FALLO CERRADO: presupuesto {budget_s:g} s < {margin:g} × {need:g} s que la celda "
                         f"({_cell_label(rep['cell'], fields)}) necesitó en la cualificación")
    return {"rule": "presupuesto de pared por celda (revisión del autor, 30-sep)", "budget_s": budget_s, "margin": margin,
            "wall_s_measured": need, **{k: rep[k] for k in ("cell", "evaluation", "t_eval_myr", "runs_reaching_evaluation",
                                                           "qualification_sha256", "qualification_instrument")}}


def assert_cell_stationarity_attainable(qual: dict, cell: dict, gate_dex: float, t_eval_myr: float, factor: float = 1.0,
                                        fields=CELL_FIELDS) -> dict:
    """Estacionariedad por celda: el brazo newtoniano de la MISMA celda, medido hasta t_eval_myr; puerta ≥ factor ×
    lo medido (factor 1: la regla del programa para la estacionariedad es «≥ el máximo medido»)."""
    newton = dict(cell, arm="newton")
    rep = cell_report(qual, newton, "stationarity_dex", "t_end", t_eval_myr, fields)
    _require_qualified(rep, "la puerta de estacionariedad")
    if not gate_dex >= factor * rep["floor"]:
        raise SystemExit(f"FALLO CERRADO: estacionariedad {gate_dex:g} dex < {factor:g} × {rep['floor']:g} dex medidos en la celda "
                         f"({_cell_label(rep['cell'], fields)}) hasta {t_eval_myr} Myr")
    return {"rule": "estacionariedad por celda (revisión del autor, 30-sep)", "gate_dex": gate_dex, "factor": factor,
            "measured_dex": rep["floor"], **{k: rep[k] for k in ("cell", "t_eval_myr", "runs_reaching_evaluation",
                                                                "qualification_sha256", "qualification_instrument")}}


def sheet_cell_excluded(gamma_uv: float, t_window: float, amp_nonlinear: float, amp_noise: float) -> dict:
    """Criterio de exclusión de las láminas: la banda UV, creciendo desde el ruido de redondeo A_ruido a la
    tasa γ_UV, alcanza la amplitud no lineal A_nl antes de que cierre la ventana lineal del modo sembrado
    (t_ventana) si γ_UV·t_ventana > ln(A_nl/A_ruido). Calculable antes de correr con la γ_UV medida."""
    budget = math.log(amp_nonlinear / amp_noise) if amp_noise > 0 else math.inf
    return {"gamma_uv_times_t_window": gamma_uv * t_window, "ln_budget": budget,
            "excluded": bool(gamma_uv * t_window > budget)}
