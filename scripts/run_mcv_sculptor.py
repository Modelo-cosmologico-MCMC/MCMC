#!/usr/bin/env python
"""Preinscripción «la MCV (c) en Sculptor» (MCV-b, bloque P0 del autor del 23-oct): el perfil σ_los(R) de Sculptor,
binado con PR-W, frente al equilibrio de la MCV (c) — Jeans isotermo con autointercambio empalmado con una envoltura NFW
de molestia, con las estrellas de Plummer (Υ⋆ ∈ {1, 2, 3}) en el potencial y σ_x/m global.

    python scripts/run_mcv_sculptor.py bank      # banco de modelos sobre la rejilla congelada (antes de los bytes)
    python scripts/run_mcv_sculptor.py check     # prefiltro del empalme frente a la búsqueda exhaustiva (240 puntos)
    python scripts/run_mcv_sculptor.py prereg    # congela la preinscripción, que cita el sha256 del banco
    python scripts/run_mcv_sculptor.py analyze   # FALLA CERRADO mientras walker2009 no esté AVAILABLE

Letras (del autor; precisiones del ejecutor en la preinscripción):
    A — cabe con σ_x/m ≤ 10 cm²/g, envoltura dentro de las previas y rama con núcleo
    B — solo cabe en la rama densa o con la envoltura fuera de las previas (o con 10 < σ_x/m ≤ 50)
    C — no cabe con ningún σ_x/m ≤ 50 cm²/g
    INDETERMINADO — puertas

La preinscripción congelada de la ley local («perfil σ_los(R) de Sculptor», sha256 604f9c25…) no se toca: las dos corren
el mismo día sobre el mismo perfil binado (anexo de PR-W, sha256 2a067315…).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.mcv_equilibrium import (  # noqa: E402
    G_KPC,
    dm_profile,
    jeans_match,
    plummer_mass,
)
from dynamics.mcv_sculptor import (  # noqa: E402
    bank_point,
    chi2_rows,
    envelope_grid,
    gravothermal_reference,
    letters,
    sigma_los_model,
)
from dynamics.walker_binning import (  # noqa: E402
    SchemaError,
    binned_profile,
    stars_from_catalogue,
)
from mcmc_ontology.data_registry import manifest_path, require_available  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-23_mcv_sculptor"
PREREG = OUTDIR / "preregistration.json"
BANK = OUTDIR / "model_bank.json"
CHECK = OUTDIR / "prefilter_check.json"
WALKER_DIR = ROOT / "results" / "2026-09-21_sculptor_profile"
ANNEX = WALKER_DIR / "binning_annex.json"
LOCAL_LAW_PREREG = WALKER_DIR / "preregistration.json"
RAW = ROOT / "data" / "raw" / "walker2009"

SPEC = {
    "t_age_gyr": 10.0, "H0": 67.4, "rho_E": 0.0,
    "plummer_a_kpc": 0.26, "L_V_Lsun": 2.6e6,
    "upsilon": [1.0, 2.0, 3.0], "beta_arm": [-0.5, 0.0, 0.3],
    "sigma_over_m_grid": [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0],
    "sigma_over_m_A_max": 10.0, "sigma_over_m_C_max": 50.0,
    "c_relation": {"a": 0.905, "b": -0.101, "h": 0.674, "scatter_dex": 0.11},
    "prior": {"logM": [8.5, 10.5], "k_abs_max": 2.0},
    "grid_prior": {"logM": [8.5, 10.5], "dlogM": 0.125, "k_abs_max": 2.0, "dk": 0.5},
    "grid_box": {"logM": [7.5, 11.5], "dlogM": 0.25, "k_abs_max": 5.0, "dk": 1.0},
    "domain": {"sigma0_range": [0.5, 300.0], "u_max": 1e10},
    "jeans": {"n_r": 800, "r_max_over_a": 120.0},
    "fit": {"p_min": 0.05, "n_fitted": 2, "nu_min": 2},
    "prefilter": True,
}
# Puntos del banco que la puerta de reproducción recalcula el día del análisis (incluye ramas densas y casos sin rama).
REPRO_POINTS = [(0.3, 2.0, 9.0, 0.0), (2.0, 2.0, 9.5, -1.0), (10.0, 1.0, 9.0, 1.0), (50.0, 3.0, 10.5, 0.0),
                (0.01, 1.0, 8.5, 2.0), (5.0, 3.0, 11.0, -3.0), (20.0, 2.0, 8.0, 0.0), (1.0, 3.0, 10.0, 2.0)]
REPRO_SIGMA_POINTS = [(0.3, 2.0, 9.0, 0.0), (2.0, 2.0, 9.5, -1.0), (10.0, 1.0, 9.0, 1.0)]
REPRO_R_PC = [50.0, 100.0, 200.0, 400.0, 700.0]
REPRO_TOL = 1e-6


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _refuse_if_bytes() -> None:
    present = [p.name for p in RAW.glob("*")] if RAW.exists() else []
    if manifest_path("walker2009").exists() or present:
        raise SystemExit(f"FALLO CERRADO: hay manifest o bytes de walker2009 ({present}); esto se congela antes de los bytes")


def _bank_task(args):
    return bank_point(*args, SPEC, prefilter=SPEC["prefilter"])


def _tasks(spec: dict) -> list[tuple]:
    return [(som, ups, lm, k) for som in spec["sigma_over_m_grid"] for ups in spec["upsilon"] for lm, k in envelope_grid(spec)]


def bank(processes: int | None = None) -> int:
    """Banco de modelos: todas las ramas del empalme en cada punto de la rejilla congelada (sin datos)."""
    if BANK.exists() or PREREG.exists():
        raise SystemExit("FALLO CERRADO: el banco o la preinscripción ya existen; no se sobrescriben")
    _refuse_if_bytes()
    tasks = _tasks(SPEC)
    n = processes or min(4, os.cpu_count() or 1)
    points = []
    with Pool(n) as pool:
        for i, p in enumerate(pool.imap(_bank_task, tasks, chunksize=4)):
            points.append(p)
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {"generator": "scripts/run_mcv_sculptor.py bank", "code_commit": _git_head(),
           "generated_utc": datetime.now(timezone.utc).isoformat(), "spec": SPEC, "n_points": len(points), "points": points}
    tmp = BANK.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.rename(BANK)
    print(f"banco: {BANK} ({len(points)} puntos) sha256 {_sha(BANK)}")
    return 0


def _check_task(point):
    fast = bank_point(*point, SPEC, prefilter=True)
    full = bank_point(*point, SPEC, prefilter=False)
    return {"point": list(point), "same_in_domain": _same_branches(fast["branches"], full["branches"], REPRO_TOL),
            "n_fast": len(fast["branches"]), "n_full": len(full["branches"]),
            "outside_domain_fast": fast["dropped_outside_domain"], "outside_domain_full": full["dropped_outside_domain"]}


def check(n: int = 240, seed: int = 20261023, processes: int | None = None) -> int:
    """El prefiltro del empalme frente a la búsqueda exhaustiva en n puntos al azar de la caja (σ_x/m log-uniforme en
    [0.01, 50], Υ⋆ de la rejilla, log10 M200 y k uniformes): las ramas dentro del dominio deben coincidir."""
    if CHECK.exists() or PREREG.exists():
        raise SystemExit("FALLO CERRADO: la comprobación o la preinscripción ya existen; no se sobrescriben")
    _refuse_if_bytes()
    rng = np.random.default_rng(seed)
    g = SPEC["grid_box"]
    pts = [(float(10.0 ** rng.uniform(-2.0, math.log10(50.0))), float(rng.choice(SPEC["upsilon"])),
            float(rng.uniform(*g["logM"])), float(rng.uniform(-g["k_abs_max"], g["k_abs_max"]))) for _ in range(n)]
    with Pool(processes or min(4, os.cpu_count() or 1)) as pool:
        rows = pool.map(_check_task, pts, chunksize=2)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {"generator": "scripts/run_mcv_sculptor.py check", "code_commit": _git_head(),
           "generated_utc": datetime.now(timezone.utc).isoformat(), "n": n, "seed": seed,
           "differences_in_domain": sum(1 for r in rows if not r["same_in_domain"]),
           "outside_domain_roots_only_exhaustive": sum(r["outside_domain_full"] - r["outside_domain_fast"] for r in rows),
           "rows": rows}
    CHECK.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"comprobación: {doc['differences_in_domain']} diferencias dentro del dominio en {n} puntos; sha256 {_sha(CHECK)}")
    return 0


def _bank_index(doc: dict) -> dict:
    return {(p["som"], p["ups"], p["logM"], p["k"]): p for p in doc["points"]}


def _same_branches(a: list, b: list, tol: float) -> bool:
    key = sorted
    a, b = key((x[0], x[1], x[4]) for x in a), key((x[0], x[1], x[4]) for x in b)
    return len(a) == len(b) and all(abs(p[0] / q[0] - 1.0) <= tol and abs(p[1] / q[1] - 1.0) <= tol and p[2] == q[2]
                                    for p, q in zip(a, b))


def _sigma_reference(idx: dict) -> list[dict]:
    out = []
    for key in REPRO_SIGMA_POINTS:
        p = idx[key]
        for br in p["branches"]:
            s = sigma_los_model(p, br, SPEC["beta_arm"], REPRO_R_PC, SPEC)
            out.append({"point": list(key), "branch": br[4], "rho0": br[2],
                        "sigma_los_kms": {str(b): [float(x) for x in v] for b, v in s.items()}})
    return out


def dato_previo() -> dict:
    """Reproducción SIN LETRA del dato previo del autor: masa dentro de r_½ = (4/3)·R_half de la rama con núcleo para las
    envolturas que cita, frente a M_half = 4σ²R_half/G (σ = 9.2 km/s, R_half = 260 pc)."""
    r_half = 4.0 / 3.0 * SPEC["plummer_a_kpc"]
    M_half = 4.0 * 9.2 ** 2 * SPEC["plummer_a_kpc"] / G_KPC
    rows = []
    for M200, c in ((1e9, 20.0), (3e9, 15.0)):
        for som in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
            for ups in (None, 1.0, 2.0, 3.0):
                pl = None if ups is None else {"M": ups * SPEC["L_V_Lsun"], "a": SPEC["plummer_a_kpc"]}
                m = jeans_match(M200, c, som, t_age=SPEC["t_age_gyr"], H0=SPEC["H0"], plummer=pl, prefilter=True)
                core = [b for b in m["branches"] if b["branch"] == "con núcleo"]
                if not core:
                    rows.append({"M200": M200, "c": c, "som": som, "ups": ups, "M_dm": None, "M_tot": None})
                    continue
                M_dm = float(dm_profile(m, core[0], [r_half])["M_M"][0])
                M_st = 0.0 if pl is None else float(plummer_mass(r_half, pl["M"], pl["a"]))
                rows.append({"M200": M200, "c": c, "som": som, "ups": ups, "M_dm": M_dm, "M_tot": M_dm + M_st,
                             "sigma0": core[0]["sigma0"], "r1_kpc": core[0]["r1"]})
    return {"r_half_3d_kpc": r_half, "M_half_wolf_msun": M_half, "rows": rows,
            "author_statement": "la rama con núcleo llega a 2.05×10⁷ M☉ con σ_x/m ≈ 0.3–0.5 cm²/g y envolturas de 10⁹ (c = 20) a "
                                "3×10⁹ M☉ (c = 15); con σ_x/m ≥ 2 se queda en 0.7–1.4×10⁷"}


E13 = ("A, con σ_x/m pequeño. El dato previo del autor (reproducido aquí sin letra) dice que la rama con núcleo alcanza la "
       "masa de media luz de Sculptor (M(<r_½) ≈ 2.05×10⁷ M☉, el estimador 4σ²R_half/G con σ = 9.2 km/s) con σ_x/m ≈ "
       "0.3–0.5 cm²/g y envolturas de 10⁹–3×10⁹ M☉ dentro de las previas, y que con σ_x/m ≥ 2 se queda corta; un perfil casi "
       "plano de ≈ 9–10 km/s hasta 1 kpc es lo que da una esfera isoterma con una envoltura así. Si el perfil binado exige "
       "más masa central que esas envolturas, la expectativa pasa a B (rama densa o envoltura fuera de las previas); C no se "
       "espera. El número no es señal: una A dice que el equilibrio (c) acomoda a Sculptor con algún σ_x/m, no que lo "
       "prediga; si solo cabe en el extremo σ_x/m ≲ 0.1 no distingue la MCV de la materia oscura fría (se publica el "
       "intervalo de σ_x/m que cabe), y σ_x/m global solo se prueba fuera de la muestra (PR-S)")


def prereg() -> int:
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)}); no se sobrescribe")
    if not BANK.exists() or not CHECK.exists():
        raise SystemExit("FALLO CERRADO: faltan el banco o la comprobación del prefiltro (modos bank y check)")
    _refuse_if_bytes()
    doc_check = json.loads(CHECK.read_text(encoding="utf-8"))
    if doc_check["differences_in_domain"] != 0:
        raise SystemExit("FALLO CERRADO: el prefiltro no reproduce la búsqueda exhaustiva dentro del dominio")
    doc_bank = json.loads(BANK.read_text(encoding="utf-8"))
    if doc_bank["spec"] != SPEC:
        raise SystemExit("FALLO CERRADO: el banco no se generó con la especificación vigente")
    idx = _bank_index(doc_bank)
    repro = []
    for key in REPRO_POINTS:
        fresh = bank_point(*key, SPEC, prefilter=SPEC["prefilter"])
        if not _same_branches(fresh["branches"], idx[key]["branches"], REPRO_TOL):
            raise SystemExit(f"FALLO CERRADO: el punto {key} no se reproduce al congelar")
        repro.append({"point": list(key), "branches": idx[key]["branches"]})
    n_br = [len(p["branches"]) for p in doc_bank["points"]]
    doc = {
        "title": "La MCV (c) en Sculptor: el perfil σ_los(R) binado de Walker et al. 2009 frente al Jeans isotermo con "
                 "autointercambio empalmado con una envoltura NFW de molestia",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_mcv_sculptor.py prereg", "generator_commit_must_precede_freeze": True,
        "ordered_by": "bloque P0 del autor del 23-oct-2026 (P0-b, MCV-b): preinscripción de Sculptor con la MCV (c), "
                      "congelada antes de los bytes",
        "dataset_state_at_freeze": "walker2009 sin manifest y sin bytes: ningún byte del catálogo visto",
        "observable": {"definition": "σ_los(R) binado con el binning de PR-W: los bins, σ_obs y σ_err de "
                                     "dynamics.walker_binning con los números del anexo congelado, y sus puertas "
                                     "(esquema, centroide, N_bins ≥ 4, σ definida en todo bin)",
                       "annex_file": str(ANNEX.relative_to(ROOT)), "annex_sha256": _sha(ANNEX),
                       "local_law_prereg_sha256": _sha(LOCAL_LAW_PREREG),
                       "local_law_untouched": "la preinscripción de la ley local no se toca; las dos corren el mismo día"},
        "model": {
            "equations": ["∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E), con ρ_E = 0 en Sculptor (la ECV de fondo es ~1e-6 de ρ_M0)",
                          "Γ_x = ρ_M·(σ_x/m)·⟨v_rel⟩, ⟨v_rel⟩ = 4σ/√π; r₁: Γ_x(r₁)·t_edad = 1 con la densidad NFW",
                          "r < r₁: ρ_M = ρ_M0·exp[−(Φ − Φ₀)/σ₀²] con las estrellas de Plummer dentro de Φ",
                          "empalme en r₁ con la envoltura NFW: ρ_M y M(<r) continuos (Kaplinghat, Tulin y Yu 2016)"],
            "implementation": "dynamics/mcv_equilibrium.py (jeans_match, dm_profile) + dynamics/mcv_sculptor.py",
            "stars": "trazador de Plummer del montaje congelado del 5E: a = R_half = 260 pc, L_V = 2.6e6 L☉, M⋆ = Υ⋆·L_V, "
                     "Υ⋆ ∈ {1, 2, 3} (molestia discreta), también en el potencial",
            "jeans": "Jeans esférico con β constante del brazo de la ley local (−0.5, 0, +0.3; molestia discreta), "
                     "dynamics/jeans.py (malla de 800 radios hasta 120a; proyección hasta 120a)",
            "branches": "todas las ramas del empalme: 'con núcleo' la de menor ρ₀, 'densa' las demás; se descartan las "
                        "raíces fuera del dominio de la rejilla del empalme (σ₀ ∈ [0.5, 300] km/s, u = ρ₀/ρ_env(r₁) ≤ 1e10) "
                        "y las que tienen r₁ ≥ r200",
            "t_age_gyr": SPEC["t_age_gyr"], "H0": SPEC["H0"],
        },
        "nuisance": {
            "envelope": "c = c200_DM14(M200)·10^(0.11·k): mediana de Dutton y Macciò 2014 (c200, z = 0, log10 c = 0.905 − "
                        "0.101·log10[M200/(10¹² h⁻¹ M☉)], h = 0.674), extrapolada por debajo de su rango simulado, y k la "
                        "desviación en unidades de su dispersión (0.11 dex)",
            "prior": "log10 M200 ∈ [8.5, 10.5] y |k| ≤ 2 (dos dispersiones): incluye las envolturas del dato previo "
                     "(10⁹ con c = 20, k ≈ +0.7; 3×10⁹ con c = 15, k ≈ 0)",
            "box": "log10 M200 ∈ [7.5, 11.5] y |k| ≤ 5: la caja en que B y C se evalúan",
            "grid": "fina en las previas (0.125 dex en M200, 0.5 en k) y gruesa en la caja (0.25 dex, 1 en k): 295 envolturas; "
                    "el mínimo de χ² se toma sobre la rejilla congelada, sin refinamiento",
        },
        "sigma_over_m": {"grid_cm2_g": SPEC["sigma_over_m_grid"], "global": "un solo valor para el sistema; cada valor de la "
                         "rejilla se ajusta por separado (σ_x/m no se ajusta en continuo)"},
        "fit": {"chi2": "χ² = Σ_bins [(σ_mod(R_bin) − σ_obs)/σ_err]², minimizado sobre (M200, c) de la región, Υ⋆ y β",
                "nu": "ν = N_bins − 2 (las dos coordenadas de la envoltura)",
                "fits": "«cabe» con un σ_x/m ⇔ p(χ² ≥ χ²_mín | ν) ≥ 0.05 (prueba χ² al 5 %)"},
        "rules": {
            "A": "existe σ_x/m ≤ 10 cm²/g de la rejilla con el que cabe la rama con núcleo con la envoltura dentro de las previas",
            "B": "no A, pero existe σ_x/m ≤ 50 cm²/g con el que cabe alguna rama con alguna envoltura de la caja: solo en la "
                 "rama densa, solo con la envoltura fuera de las previas o solo con 10 < σ_x/m ≤ 50 (este último caso lo "
                 "añade el ejecutor: el texto del autor no lo nombra y es «cabe forzando», no A ni C)",
            "C": "no cabe con ningún σ_x/m ≤ 50 cm²/g, ninguna rama ni envoltura de la caja",
            "INDETERMINADO": "puertas: walker2009 no AVAILABLE; las del binning de PR-W (esquema, centroide, N_bins < 4, σ sin "
                             "definir); ν < 2; sha256 del banco distinto del congelado; los puntos de reproducción no se "
                             "reproducen (ramas o σ_los a 1e-6); la búsqueda exhaustiva del empalme en el punto decisivo no "
                             "da las mismas ramas que el banco; C con el mejor ajuste en el borde de la caja",
            "order": "puertas → A → B → C (mutuamente excluyentes por construcción)",
            "verdict": "una letra; ninguna es afirmación sobre el tratado (E8): sitúa el equilibrio (c) frente a la forma y la "
                       "amplitud del perfil de un sistema",
        },
        "spec": SPEC,
        "bank": {"file": str(BANK.relative_to(ROOT)), "sha256": _sha(BANK), "code_commit": doc_bank["code_commit"],
                 "n_points": doc_bank["n_points"], "points_without_branch": sum(1 for x in n_br if x == 0),
                 "points_with_dense_branches": sum(1 for x in n_br if x > 1),
                 "dropped_outside_domain": sum(p["dropped_outside_domain"] for p in doc_bank["points"]),
                 "dropped_r1_beyond_r200": sum(p["dropped_r1_beyond_r200"] for p in doc_bank["points"])},
        "prefilter_check": {"file": str(CHECK.relative_to(ROOT)), "sha256": _sha(CHECK), "n": doc_check["n"],
                            "seed": doc_check["seed"], "differences_in_domain": doc_check["differences_in_domain"],
                            "outside_domain_roots_only_exhaustive": doc_check["outside_domain_roots_only_exhaustive"],
                            "reading": "el banco usa el prefiltro bilineal del empalme; dentro del dominio da las mismas "
                                       "ramas que la búsqueda exhaustiva; las raíces fuera del dominio (u > 1e10) que la "
                                       "exhaustiva encuentra por azar se descartan por regla en ambos casos"},
        "reproduction_gate": {"points": repro, "sigma_los_reference": _sigma_reference(idx), "R_pc": REPRO_R_PC,
                              "tolerance_rel": REPRO_TOL},
        "prior_datum_without_letter": dato_previo(),
        "expectation_E13": E13,
        "without_letter": ["intervalos de σ_x/m que caben (A y cualquiera)", "mejor ajuste por σ_x/m con su Υ⋆, β, envoltura y "
                           "rama", "t_c gravotérmico (β = 0.45 y 0.75) de los mejores ajustes frente a t_edad",
                           "χ²_ν y p de cada categoría"],
        "what_this_cannot_decide": ["σ_x/m global: un sistema no lo calibra (se prueba fuera de la muestra en PR-S)",
                                    "el término de la ECV (ρ_E = 0 aquí) y el intercambio con ella",
                                    "la anisotropía más allá de β constante, la elipticidad, las binarias, las mareas",
                                    "la evolución gravotérmica (el equilibrio se supone; t_c se publica sin letra)"],
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    dp = doc["prior_datum_without_letter"]
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). sha256 de "
          f"`preregistration.json`: `{sha}`.", "",
          f"Banco de modelos `{doc['bank']['file']}` (sha256 `{doc['bank']['sha256']}`, {doc['bank']['n_points']} puntos, "
          f"commit `{doc['bank']['code_commit'][:9]}`). Observable: el perfil binado de PR-W (anexo sha256 "
          f"`{doc['observable']['annex_sha256'][:12]}…`). Estado del dataset: {doc['dataset_state_at_freeze']}.", "",
          "## Modelo", ""] + [f"- {e}" for e in doc["model"]["equations"]] + [
          f"- Estrellas: {doc['model']['stars']}.", f"- Jeans: {doc['model']['jeans']}.", f"- Ramas: {doc['model']['branches']}.", "",
          "## Molestias y ajuste", ""] + [f"- **{k}**: {v}" for k, v in doc["nuisance"].items()] + [
          f"- **σ_x/m**: {SPEC['sigma_over_m_grid']} cm²/g; {doc['sigma_over_m']['global']}."] + [
          f"- **{k}**: {v}" for k, v in doc["fit"].items()] + ["", "## Reglas (congeladas)", ""] + [
          f"- **{k}**: {v}" for k, v in doc["rules"].items()] + [
          "", "## Expectativa (E13, escrita antes de congelar)", "", E13 + ".", "",
          "## Dato previo, sin letra", "", f"Del autor: «{dp['author_statement']}».", "",
          f"Reproducción con el módulo: M(<r_½ = {1000 * dp['r_half_3d_kpc']:.0f} pc) de la rama con núcleo, frente a "
          f"M_half = 4σ²R_half/G = {dp['M_half_wolf_msun']:.3g} M☉ (en 10⁷ M☉; materia de la MCV / total con estrellas).", "",
          "| envoltura | σ_x/m | sin bariones | Υ⋆ = 1 | Υ⋆ = 2 | Υ⋆ = 3 |", "|---|---|---|---|---|---|"]
    for M200, c in ((1e9, 20.0), (3e9, 15.0)):
        for som in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
            cells = []
            for ups in (None, 1.0, 2.0, 3.0):
                r = next(x for x in dp["rows"] if x["M200"] == M200 and x["c"] == c and x["som"] == som and x["ups"] == ups)
                cells.append("—" if r["M_dm"] is None else f"{r['M_dm'] / 1e7:.2f} / {r['M_tot'] / 1e7:.2f}")
            md.append(f"| {M200:.0e} M☉, c = {c:g} | {som:g} | " + " | ".join(cells) + " |")
    md += ["", "## Lo que no puede decidir", ""] + [f"- {x}" for x in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _binned(pre: dict, raw: Path) -> dict:
    """El perfil binado de PR-W con su anexo congelado y sus puertas (las mismas que en la ley local)."""
    ax = json.loads(ANNEX.read_text(encoding="utf-8"))
    N = ax["numbers"]
    stars = stars_from_catalogue((raw / "ReadMe").read_text(encoding="latin-1"), raw / ax["table_file"], ax["table_file"],
                                 N["merge_radius_arcsec"])
    prof = binned_profile(stars, N["centre_ra_deg"], N["centre_dec_deg"], N["distance_kpc"], N["pm_min"], N["R_max_pc"],
                          N["target_per_bin"])
    if prof["members"] and not prof["member_centroid_offset_arcmin"] <= N["centroid_gate_arcmin"]:
        raise SchemaError(f"centroide de los miembros a {prof['member_centroid_offset_arcmin']:.3g}′ del centro fijado")
    if len(prof["bins"]) < N["min_bins"]:
        raise SchemaError(f"N_bins = {len(prof['bins'])} < {N['min_bins']}")
    if not all(math.isfinite(b["sigma_kms"]) for b in prof["bins"]):
        raise SchemaError("algún bin con s² ≤ ⟨e²⟩: σ del bin sin definir")
    return prof


def analyze_bins(bins: list[dict], pre: dict, bank_doc: dict) -> dict:
    """Letra sobre unos bins dados con el banco congelado: puertas de reproducción, χ², letra y comprobación exhaustiva
    en el punto decisivo; después, sin letra, t_c de los mejores ajustes."""
    spec = pre["spec"]
    idx = _bank_index(bank_doc)
    for ref in pre["reproduction_gate"]["points"]:
        fresh = bank_point(*ref["point"], spec, prefilter=spec["prefilter"])
        if not _same_branches(fresh["branches"], ref["branches"], pre["reproduction_gate"]["tolerance_rel"]) or \
                not _same_branches(idx[tuple(ref["point"])]["branches"], ref["branches"], pre["reproduction_gate"]["tolerance_rel"]):
            return {"verdict": "INDETERMINADO", "reason": f"el punto de reproducción {ref['point']} no se reproduce"}
    for ref in pre["reproduction_gate"]["sigma_los_reference"]:
        p = idx[tuple(ref["point"])]
        br = next(b for b in p["branches"] if b[4] == ref["branch"] and b[2] == ref["rho0"])
        s = sigma_los_model(p, br, spec["beta_arm"], pre["reproduction_gate"]["R_pc"], spec)
        for b, v in s.items():
            if max(abs(x / y - 1.0) for x, y in zip(v, ref["sigma_los_kms"][str(b)])) > pre["reproduction_gate"]["tolerance_rel"]:
                return {"verdict": "INDETERMINADO", "reason": "σ_los de referencia no reproducida por el código"}
    rows = chi2_rows(bank_doc["points"], bins, spec)
    L = letters(rows, len(bins), spec)
    out = {"verdict": L["letter"], "reason": L["reason"], "letter_detail": L}
    if "per_sigma_over_m" not in L:
        return out
    per = L["per_sigma_over_m"]
    if L["letter"] == "A":
        pool = [v["best_A"] for s, v in per.items() if float(s) in L["fits_A_sigma_over_m"]]
    else:
        pool = [v["best_any"] for s, v in per.items() if float(s) <= spec["sigma_over_m_C_max"] and v["best_any"]]
    decisive = min(pool, key=lambda r: r["chi2"])
    key = (decisive["som"], decisive["ups"], decisive["logM"], decisive["k"])
    exhaustive = bank_point(*key, spec, prefilter=False)
    out["exhaustive_check"] = {"point": list(key), "same_branches": _same_branches(exhaustive["branches"], idx[key]["branches"],
                                                                                  pre["reproduction_gate"]["tolerance_rel"])}
    if not out["exhaustive_check"]["same_branches"]:
        return {**out, "verdict": "INDETERMINADO", "reason": f"la búsqueda exhaustiva en el punto decisivo {list(key)} no da las "
                                                             "ramas del banco"}
    out["without_letter_collapse_time"] = {s: gravothermal_reference(v["best_any"], spec) for s, v in per.items() if v["best_any"]}
    return out


def analyze() -> int:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    try:
        raw = require_available("walker2009")
    except (FileNotFoundError, RuntimeError) as exc:
        res = {"preregistration_sha256": _sha(PREREG), "code_commit_analysis": _git_head(),
               "analyzed_utc": datetime.now(timezone.utc).isoformat(), "verdict": "INDETERMINADO",
               "reason": f"fallo cerrado: {exc}", "next_step": "ingerir walker2009 (PR-W) y ejecutar este analizador sin tocar la regla"}
        (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (OUTDIR / "report.md").write_text(f"# La MCV (c) en Sculptor — {res['verdict']}\n\n{res['reason']}.\n", encoding="utf-8")
        print(f"INDETERMINADO — {exc}")
        return 0
    if _sha(ANNEX) != pre["observable"]["annex_sha256"] or _sha(BANK) != pre["bank"]["sha256"]:
        raise SystemExit("FALLO CERRADO: el anexo de PR-W o el banco no son los congelados")
    try:
        prof = _binned(pre, raw)
        res = analyze_bins(prof["bins"], pre, json.loads(BANK.read_text(encoding="utf-8")))
        res["bins"] = prof["bins"]
    except SchemaError as exc:
        res = {"verdict": "INDETERMINADO", "reason": f"puerta del binning de PR-W: {exc}"}
    res = {"preregistration_sha256": _sha(PREREG), "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), **res}
    (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "report.md").write_text(f"# La MCV (c) en Sculptor — {res['verdict']}\n\n{res['reason']}.\n\nNinguna letra es "
                                      "afirmación sobre el tratado (E8); el número no es señal (E13).\n", encoding="utf-8")
    print(f"{res['verdict']} — {res['reason']}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"bank": bank, "check": check, "prereg": prereg, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
