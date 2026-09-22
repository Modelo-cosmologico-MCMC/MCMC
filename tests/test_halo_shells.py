"""Código de capas esféricas (cronos/halo_shells.py): capas desde las ICs
3D, campo esférico exacto en M(<r), fuerzas (gravedad de Hénon + término
centrífugo + Cronos), energía K + W + (2/5)U_C + W_fric y el ejecutor de
la ronda 2 con fallo cerrado. Nada es física: instrumento (E8)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cronos.halo_shells import (
    A_SCULPTOR,
    KernelCronosField,
    ShellRun,
    equilibrium_shells,
    shells_from_3d,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-22_halo_shells"


def test_shells_from_3d_conserve_speed_and_angular_momentum():
    rng = np.random.default_rng(0)
    pos, vel = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    r, vr, L2 = shells_from_3d(pos, vel)
    v2 = np.sum(vel ** 2, axis=1)
    assert np.allclose(vr ** 2 + L2 / r ** 2, v2)          # |v|² = v_r² + L²/r²


def test_kernel_field_is_exact_for_uniform_density_and_conservative_identity_holds():
    """Depósito B-spline con volumen de núcleo: ρ_b exacta para ρ uniforme; ε̃ reproduce Aρ_NFW^{3/2}
    fuera de ε_soft; dε̃/dr → 0 en el centro (ds/dr → 0); identidad U_self(capas) = U_self(celdas)
    (la fuerza es el gradiente exacto del funcional 2/5); el refinamiento de masa no altera M(<r)."""
    from cronos.halo_nbody import KPC3_TO_PC3, nfw_structural
    f = KernelCronosField(A_SCULPTOR, eps_soft=0.1, ds=0.04)
    r_fine = np.geomspace(1e-6, 420.0, 200_000)
    dV = np.gradient(4.0 / 3.0 * np.pi * r_fine ** 3)
    f.update(r_fine, 1.0 * dV, 0.0)
    ok = np.isfinite(f.V_b) & (np.exp(f.s_b) < 300.0)
    assert np.allclose(f.rho_b[ok], 1.0, rtol=1e-3)
    p = nfw_structural(1e11, 10.0, 67.86705532886631)
    x = r_fine / p["r_s"]
    rho_nfw = p["rho_s"] / (x * (1.0 + x) ** 2)
    f.update(r_fine, rho_nfw * dV, 0.0)
    for r0 in (0.1, 0.4, 1.0):
        eps_an = A_SCULPTOR * (np.interp(r0, r_fine, rho_nfw) * KPC3_TO_PC3) ** 1.5
        assert abs(f.eps_c(np.array([r0]))[0] / eps_an - 1.0) < 0.02
    d = f.deps_c_dr(np.array([1e-4, 1e-3, 0.03]))
    assert d[2] < 0.0 and abs(d[0]) < 0.2 * abs(d[1])              # dε̃/dr → 0 linealmente en el centro (ds/dr = r/(r² + ε²))
    ic = equilibrium_shells(1e11, 10.0, 67.86705532886631, 50_000, 2, refine_r_kpc=2.0, refine_beta=1.5)
    run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], A=0.05 * A_SCULPTOR, cronos=True, dt_myr=0.2)
    for x0 in (0.4, 1.0, 5.0):
        M_an = 4 * np.pi * p["rho_s"] * p["r_s"] ** 3 * (np.log(1 + x0 / p["r_s"]) - (x0 / p["r_s"]) / (1 + x0 / p["r_s"]))
        assert abs(float(ic["m"][ic["r"] < x0].sum()) / M_an - 1.0) < 0.05
    assert run.events[0]["U_self_identity_rel"] < 1e-10
    assert run.events[0]["ds"] == 0.04 and run.events[0]["n_bins"] == run.field.nb


def test_newtonian_run_conserves_energy_and_cronos_publishes_balance():
    ic = equilibrium_shells(1e11, 10.0, 67.86705532886631, 20_000, 1, refine_r_kpc=2.0, refine_beta=1.5)
    assert ic["refined"] and ic["n_shells"] > 20_000 and ic["m"].min() < 0.1 * ic["m0"]
    assert abs(ic["m"].sum() / ic["M_tot"] - 1.0) < 1e-12
    run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], cronos=False, dt_myr=0.2)
    out = run.run(0.05, [0.0, 0.05])
    e0, e1 = out["snapshots"][0]["energy"], out["snapshots"][-1]["energy"]
    assert abs(e1["E_grav_only"] - e0["E_grav_only"]) / abs(e0["E_grav_only"]) < 1e-3
    assert e0["W_fric"] == 0.0 and e0["U_self"] == 0.0
    run_c = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], A=0.05 * A_SCULPTOR, cronos=True, dt_myr=0.2)
    out_c = run_c.run(0.02, [0.0, 0.02])
    e = out_c["snapshots"][-1]["energy"]
    assert e["U_self"] == pytest.approx(0.4 * e["U_ext"]) and e["E_self"] == pytest.approx(e["K"] + e["W"] + e["U_self"] + e["W_fric"])
    assert out_c["events"][0]["weak_regime_ok"] and out_c["events"][0]["D_F_at_soft"] > 0.0
    assert e["W_fric"] >= 0.0                                   # la fricción con compuerta solo drena


def test_runner_fails_closed_without_prereg(tmp_path):
    body = (REPO / "scripts" / "run_halo_shells.py").read_text(encoding="utf-8").split("def cmd_analyze(")[1]
    assert 'G["tol_E_cronos"]' in body and 'G["tol_E_newton"]' in body and 'G["newton_stationarity_dex"]' in body and 'G["N_control_factor"]' in body and 'r_cj[prefix]' in body
    assert re.search(r">\s*1e-4\b", body) is None and re.search(r">\s*0\.1\b", body) is None and "2000" not in body
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(REPO / "scripts" / "run_halo_shells.py"), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


@pytest.mark.skipif(not (OUT / "preregistration.json").exists(), reason="preinscripción de capas no congelada")
def test_prereg_frozen_fields():
    d = json.loads((OUT / "preregistration.json").read_text(encoding="utf-8"))
    import hashlib
    assert hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest() in (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert d["gates"]["tol_E_newton"] == 1e-4 and d["gates"]["tol_E_cronos"] == 3e-2 and d["gates"]["n_min_shells_within_0p4_N1e6"] == 2000 and d["gates"]["newton_stationarity_dex"] == 0.1
    assert d["gates"]["N_control_factor"] == 2.0 and d["gates"]["stop_rules"]["weak_regime_eps_max"] == 1e-3 and d["t_end_gyr"] == 0.1
    assert set(d["rules"]["letters"]) == {"A", "B", "C", "INDETERMINADO"} and "UV" in d["rules"]
    assert {a["ds"] for a in d["arms"].values() if a["N"] == 1_000_000 and a["cronos"]} == {0.02, 0.04, 0.08}
    assert d["system"]["refine_r_kpc"] > 0 and d["system"]["refine_beta"] > 0 and d["instrument"]["eta_dt"] > 0 and d["instrument"]["dt_min_myr"] > 0
    assert d["prediction_from_criterion"]["expected"] == "A"
    runs = OUT / "runs"
    if runs.exists():
        sha = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
        for p in runs.glob("*.json"):
            assert json.loads(p.read_text(encoding="utf-8"))["preregistration_sha256"] == sha, p.name
