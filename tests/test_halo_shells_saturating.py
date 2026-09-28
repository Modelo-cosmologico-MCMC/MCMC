"""Forma saturante de ε_c(ρ) en el código de capas (brazo de CONTROL de la ronda 3, PR-5): G(ρ) en forma cerrada
coincide con la cuadratura; la fuerza c²dε̃/dr es el gradiente exacto de U = −c² Σ_b V_b G(ρ_b) también para la
forma saturante; la ley débil no cambia (U_self = (2/5)U_ext, misma trayectoria que antes); validación declarada."""

import numpy as np
import pytest
from scipy.integrate import quad

from cronos.halo_nbody import C_KMS, KPC3_TO_PC3
from cronos.halo_shells import KernelCronosField, ShellRun, equilibrium_shells
from dynamics.epsilon_c_saturating import deps_c_drho, eps_c


def test_saturating_G_matches_quadrature_and_form_functions():
    f = KernelCronosField(1.0, eps_soft=0.1, ds=0.04, eps_form="saturating", eps_max=1e-3, rho_star=0.05)
    for rho in (1e-6, 1e-3, 0.05, 0.4, 7.0):
        G_num = quad(lambda x: eps_c(x, 1e-3, 0.05), 0.0, rho, epsabs=0.0, epsrel=1e-12)[0]
        assert f._G_of_rho(rho) == pytest.approx(G_num, rel=1e-8, abs=1e-30)
        assert f._eps_of_rho(rho) == pytest.approx(eps_c(rho, 1e-3, 0.05)) and f._deps_of_rho(rho) == pytest.approx(deps_c_drho(rho, 1e-3, 0.05))
    assert f.A == pytest.approx(1e-3 / 0.05 ** 1.5)                       # A de la ley débil equivalente
    w = KernelCronosField(2.0, eps_soft=0.1, ds=0.04)
    assert w._G_of_rho(0.3) == pytest.approx(0.4 * 2.0 * 0.3 ** 2.5) and w.eps_form == "weak"
    with pytest.raises(ValueError):
        KernelCronosField(1.0, 0.1, 0.04, eps_form="saturating")
    with pytest.raises(ValueError):
        KernelCronosField(1.0, 0.1, 0.04, eps_form="otra")


def _force_is_gradient(form_kwargs):
    rng = np.random.default_rng(5)
    r = np.sort(rng.uniform(0.05, 3.0, 400)); m = np.full(400, 2.5e8)      # M☉, kpc: ρ ~ 1e-2–1 M☉/pc³ en el centro
    f = KernelCronosField(1.0, eps_soft=0.1, ds=0.04, **form_kwargs)
    f.update(r, m, 0.0)
    U0 = f.U_self_bins()
    i, h = 7, 1e-6
    a_i = C_KMS ** 2 * f.deps_c_dr(np.array([r[i]]))[0]                   # aceleración de Cronos en la capa i (km/s)²/kpc
    rp = r.copy(); rp[i] += h; f.update(rp, m, 0.0); Up = f.U_self_bins()
    rm = r.copy(); rm[i] -= h; f.update(rm, m, 0.0); Um = f.U_self_bins()
    dU_dr = (Up - Um) / (2.0 * h)
    return a_i, -dU_dr / m[i], U0


def test_force_is_exact_gradient_of_U_for_both_forms():
    a_w, g_w, _ = _force_is_gradient({})
    assert a_w == pytest.approx(g_w, rel=1e-5) and a_w != 0.0
    a_s, g_s, U_s = _force_is_gradient({"eps_form": "saturating", "eps_max": 5e-4, "rho_star": 0.2})
    assert a_s == pytest.approx(g_s, rel=1e-5) and a_s != 0.0 and U_s < 0.0


def test_weak_form_unchanged_and_saturating_runs_publish_form():
    ic = equilibrium_shells(1e11, 10.0, 67.86705532886631, 5_000, 1, refine_r_kpc=2.0, refine_beta=1.5)
    kw = {"A": 6.201589e-7 * 0.05, "ds": 0.08, "dt_myr": 0.2}
    w = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], **kw).run(0.004, [0.0, 0.004])
    e = w["snapshots"][0]["energy"]
    assert e["U_self"] == pytest.approx(0.4 * e["U_ext"]) and w["eps_form"] == "weak" and w["eps_max_declared"] is None
    # forma saturante con la MISMA amplitud débil que A_Sculptor (ε_max = A_S·ρ*^{3/2}) y ρ* en la cúspide
    A_S, rho_star = 6.201589e-7, 0.01
    s = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], ds=0.08, dt_myr=0.2, eps_form="saturating",
                 eps_max=A_S * rho_star ** 1.5, rho_star=rho_star).run(0.004, [0.0, 0.004])
    assert s["eps_form"] == "saturating" and s["rho_star_msun_pc3"] == rho_star and s["A_weak_equivalent"] == pytest.approx(A_S)
    e0, e1 = s["snapshots"][0]["energy"], s["snapshots"][-1]["energy"]
    assert e0["U_self"] < 0.0 and abs(e1["E_self"] - e0["E_self"]) / abs(e0["E_self"]) < 1e-3
    assert s["snapshots"][-1]["eps_c_max"] <= A_S * rho_star ** 1.5 * (1 + 1e-12)   # la forma saturante nunca supera ε_max
    assert abs(e0["U_self"] / (0.4 * e0["U_ext"]) - 1.0) > 0.05           # el funcional ya no es (2/5)·U_ext (saturación en la cúspide)
    assert KPC3_TO_PC3 == pytest.approx(1e-9)
