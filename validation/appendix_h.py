"""La suite espejo del apéndice H (v35, H.1/H.3 y §13.5).

Cada verificación de signo de §13.5 se ejecuta acompañada de su CONTROL
NEGATIVO — el caso que debe fallar y falla («las filas de control son
fallos intencionados que confirman la rigidez de la construcción»,
Tabla H.1). Los resultados condicionales NO se verifican como si
estuvieran resueltos: se listan expuestos con su condición, como hace
H.3 (lo demostrado, lo condicional, lo refutable).

Uso:  from validation.appendix_h import run_all;  run_all(verbose=True)
"""

from __future__ import annotations

import numpy as np


def _checks() -> list[dict]:
    """Construye y ejecuta las verificaciones. Devuelve la lista de filas."""
    rows: list[dict] = []

    def add(name: str, cite: str, ok: bool, kind: str = "verificación"):
        rows.append({"name": name, "cite": cite, "ok": bool(ok),
                     "kind": kind})

    # ---- 1. Tensión Primordial (Prop. 3.4 / Axioma 3) -------------------
    from core.basal import T0_numeric, quasi_cancellation_ok
    add("T0 ≥ 0 con igualdad sii δ0 = 0",
        "Prop. 3.4 / Def. 3.3", T0_numeric(0.0) == 0.0
        and T0_numeric(0.01) > 0.0)
    add("sin casi-cancelación (3.3) no hay vacío más profundo",
        "ec. (3.3)", not quasi_cancellation_ok(C0=2.5), "control negativo")

    # ---- 2. Monotonía del Camino (Teo. 4.5) -----------------------------
    from core.path_flow import flow, grad_V
    res = flow(np.array([0.3, 0.1]), delta0=0.05, d_sigma=0.05,
               n_steps=400)
    add("dV/dσ ≤ 0 y producción entrópica ≥ 0",
        "Teo. 4.5", bool(np.all(np.diff(res["V"]) <= 1e-12)
                         and np.all(res["S_production_rate"] >= 0.0)))
    # control: el flujo invertido (ascenso) viola la monotonía
    phi = np.array([0.3, 0.1])
    from core.dual_plane import to_dual
    from core.basal import V0
    c0 = to_dual(phi[0], phi[1])
    v_before = float(V0(c0["rho"], c0["chi"], 0.05))
    phi_up = phi + 1e-3 * grad_V(phi, 0.05)      # ascenso deliberado
    c1 = to_dual(phi_up[0], phi_up[1])
    v_after = float(V0(c1["rho"], c1["chi"], 0.05))
    add("el flujo invertido viola la monotonía (detectado)",
        "Teo. 4.5", v_after > v_before, "control negativo")

    # ---- 3. Elipticidad euclidiana (Lema 5.2) ---------------------------
    from core.florencia import euclidean_symbol
    rng = np.random.default_rng(1)
    ks, kv = rng.normal(size=8), rng.normal(size=(8, 3))
    add("símbolo euclidiano definido positivo",
        "Lema 5.2", bool(np.all(euclidean_symbol(ks, kv) > 0.0)))
    lorentz = ks ** 2 - np.sum(kv ** 2, axis=1)
    add("el símbolo lorentziano NO es definido (detectado)",
        "Lema 5.2", bool(lorentz.min() < 0.0 < lorentz.max()
                         or not np.all(lorentz > 0)), "control negativo")

    # ---- 4. Rotación de Florencia (ec. 6.2) -----------------------------
    from core.florencia import chain_generators, florencia_rotation, signature
    rot = florencia_rotation(chain_generators(3))
    add("(γ⁰)² = −𝟙 con un solo generador girado → firma (−,+,+,+)",
        "ec. (6.2) / §13.5", signature(rot) == [-1, 1, 1, 1])
    two = florencia_rotation(chain_generators(3), n_rotations=2)
    add("girar dos generadores no da un cono de luz (detectado)",
        "§13.5", signature(two).count(-1) == 2, "control negativo")

    # ---- 5. Positividad por reflexión (Teo. 7.1, juguete) ---------------
    from core.reflection_positivity import rp_holds, rp_min_eigenvalue
    phi_g = np.linspace(-2.0, 2.0, 31)
    V_g = np.array([V0(abs(p), 0.0, 0.05) for p in phi_g])
    add("⟨ϑ(F̄)·F⟩ ≥ 0 en el retículo de juguete",
        "Teo. 7.1", rp_holds(phi_g, V_g, J=1.0))
    add("el acoplo temporal invertido rompe la RP (detectado)",
        "Teo. 7.1", rp_min_eigenvalue(phi_g, V_g, J=-1.0) < -1e-6,
        "control negativo")

    # ---- 6. Ley de Cronos débil (Def. 11.1) -----------------------------
    from cronos.cronos_v3 import lapse, gate_friction
    rho_dense = np.array([100.0])
    N_ok = lapse(np.zeros(1), rho_dense, 1e-7, 1.0)
    add("N < 1 en regiones densas (signo corregido)",
        "Def. 11.1 / Obs. 11.4", float(N_ok[0]) < 1.0)
    # control: el signo del corpus antiguo daría N > 1 en denso
    N_old = 1.0 + 0.0 + 1e-7 * (100.0) ** 1.5   # +ε_c: el error de signo
    add("el signo antiguo (+ε_c) da N > 1 en denso (detectado)",
        "Obs. 11.4", N_old > 1.0, "control negativo")

    # ---- 7. Fricción con compuerta (Cor. 11.3) --------------------------
    g_on = gate_friction(np.array([10.0]), np.array([1.0]), 1e-7, 1.0)
    g_off = gate_friction(np.array([10.0]), np.array([-1.0]), 1e-7, 1.0)
    add("Γ > 0 solo durante el colapso (Θ(ρ̇))",
        "Cor. 11.3a", float(g_on[0]) > 0.0 == float(g_off[0]))
    from cronos.kronos_kick import friction_acceleration
    a_v32 = friction_acceleration(np.ones((1, 3)), np.array([10.0]))
    add("la fricción v32 actúa también sin colapso (detectado)",
        "Obs. 11.4", bool(np.any(a_v32 != 0.0)), "control negativo")

    # ---- Límites de recuperación (§13.5) --------------------------------
    from core.path_flow import flow as _flow
    r0 = _flow(np.array([0.0, 0.0]), delta0=0.0, n_steps=50)
    add("δ0 → 0 devuelve la inercia eterna del estado perfecto",
        "§13.5", float(np.linalg.norm(r0["final"])) < 1e-15,
        "recuperación")
    from core.gea import newton_seal_ratio
    add("(λK, ξ) → (1,1) devuelve la relatividad general",
        "Prop. 9.3 / §13.5", abs(newton_seal_ratio(1.0, 1.0) - 1.0) < 1e-15,
        "recuperación")
    from cronos.cronos_v3 import kdk_step_v3
    n = 4
    rng = np.random.default_rng(2)
    x0, u0 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    g = rng.normal(size=(n, 3))
    dt = 1e-3
    x1, u1 = kdk_step_v3(x0.copy(), u0.copy(), dt, Phi_N=np.zeros(n),
                         grad_Phi_N=-g, rho=np.full(n, 10.0),
                         rho_dot=np.full(n, 1.0),
                         grad_eps_c=np.zeros((n, 3)), alpha0_inv=0.0,
                         rho_c=1.0)
    u_half = u0 + g * dt / 2.0
    ok_frw = (np.allclose(x1, x0 + u_half * dt, atol=1e-12)
              and np.allclose(u1, u_half + g * dt / 2.0, atol=1e-12))
    add("ε_c → 0 devuelve la geodésica FRW estándar",
        "§13.5 / Prop. A.1", ok_frw, "recuperación")

    return rows


# Lo condicional se EXPONE, no se verifica como resuelto (H.3):
CONDITIONALS = [
    ("Signo del Exponente de Lydia (ν > 0 ⟺ γR·ē > m̄²)",
     "Def. 10.4 / H.2.3 — frente abierto nº 4"),
    ("Estabilidad del modo de Atlas (λK ↓ 1⁺, Término de Cronos)",
     "Prop. 9.5 / §13.4 — frente abierto nº 3"),
    ("Conjetura de los Residuos (G_cosmo/G_N − 1 ≈ −1.8%)",
     "Conj. 9.6 — frente abierto nº 6"),
    ("RP del sector espinorial de Wilson con acoplos no estacionarios",
     "Teo. 7.4 — frente abierto nº 1"),
    ("λ = 10: ¿física o convención de calibre?",
     "Obs. 8.7 — frente abierto nº 2"),
    ("Pesos c_in del WKB y derivación independiente de β3",
     "Obs. 12.2 / 12.5 — frente abierto nº 7"),
    ("Cronos v3 en simulaciones de producción",
     "cap. 14.5 — frente abierto nº 5"),
]


def run_all(verbose: bool = True) -> dict:
    """Ejecuta la suite espejo. Devuelve {rows, n_ok, n_total, all_ok}."""
    rows = _checks()
    n_ok = sum(r["ok"] for r in rows)
    if verbose:
        print("--- Suite espejo del apéndice H (verificaciones + controles"
              " negativos) ---")
        for r in rows:
            tag = {"verificación": "OK  ", "control negativo": "CTRL",
                   "recuperación": "REC "}[r["kind"]]
            mark = tag if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['name']}  ({r['cite']})")
        print(f"  {n_ok}/{len(rows)} superadas "
              "(los controles negativos son fallos intencionados que "
              "confirman la rigidez — Tabla H.1)")
        print("  Condicional — EXPUESTO, no resuelto (H.3):")
        for name, cite in CONDITIONALS:
            print(f"  [COND] {name}  ({cite})")
        print("  Refutable: los tres frentes empíricos de §13.6 — "
              "G_cosmo/G_N, la compuerta en halos virializados, "
              "α0⁻¹ ≲ 1e-6.")
    return {"rows": rows, "n_ok": n_ok, "n_total": len(rows),
            "all_ok": n_ok == len(rows)}
