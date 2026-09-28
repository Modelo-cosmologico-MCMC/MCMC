"""Extensiones de los instrumentos del frente 5 (orden del 28-sep, PR-3): las láminas publican la banda
ultravioleta de la densidad depositada y aceptan un filtro espectral declarado k_c; las capas esféricas
aceptan el rango M(<r) actualizado en los subpasos. Ninguna opción cambia los valores por defecto."""

import numpy as np
import pytest

from cronos.cronos_jeans_1d import run_sheets
from cronos.halo_shells import ShellRun, equilibrium_shells


def test_sheets_filter_and_uv_band_are_published():
    r0 = run_sheets(1.2, N=64 * 128 * 2, ng=128, T=0.02, dt=2e-3, quiet_start=True, n_beams=64, nmodes=2, sample_every=5)
    r1 = run_sheets(1.2, N=64 * 128 * 2, ng=128, T=0.02, dt=2e-3, quiet_start=True, n_beams=64, nmodes=2, sample_every=5, k_cut_frac=0.5)
    assert r0["k_cut"] is None and r1["k_cut"] == pytest.approx(0.5 * r1["k_nyquist"])
    assert all("uv_rms" in s for s in r1["samples"]) and r1["samples"][0]["uv_rms"] == 0.0     # retículo exacto: UV nula en t = 0
    assert r0["lattice_exact"] and r1["lattice_exact"]
    assert r0["k_cut_frac"] is None and r1["k_cut_frac"] == 0.5 and r1["uv_band_frac"] == 0.5


def test_shells_rank_update_matches_frozen_in_newtonian_short_run():
    ic = equilibrium_shells(1e11, 10.0, 67.86705532886631, 20_000, 1, refine_r_kpc=2.0, refine_beta=1.5)
    outs = {}
    for ru in (False, True):
        run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], cronos=False, dt_myr=0.2, rank_update=ru)
        out = run.run(0.01, [0.0, 0.01])
        e0, e1 = out["snapshots"][0]["energy"]["E_grav_only"], out["snapshots"][-1]["energy"]["E_grav_only"]
        outs[ru] = (abs(e1 - e0) / abs(e0), out["snapshots"][-1]["M_within"]["0.4"], out["rank_update"], out["dt_adaptive"]["n_full_sorts"])
    assert outs[False][2] is False and outs[True][2] is True and outs[True][3] > 0
    assert outs[True][0] < 1e-3 and outs[False][0] < 1e-3
    assert np.isclose(outs[True][1], outs[False][1], rtol=0.05)            # misma física a 10 Myr
