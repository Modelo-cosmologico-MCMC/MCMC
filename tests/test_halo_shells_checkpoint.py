"""Reanudación del código de capas: una corrida interrumpida y reanudada desde el punto de control reproduce la
corrida ininterrumpida (mismas posiciones, misma energía, mismas instantáneas), la pared se acumula y los
metadatos se publican. El entorno de cómputo se reinicia a intervalos de minutos: sin esto las corridas de N = 1e6
de la ronda 3 no pueden terminar."""

import numpy as np

from cronos.halo_shells import ShellRun, equilibrium_shells


def _ic():
    return equilibrium_shells(1e11, 10.0, 67.86705532886631, 3_000, 1, refine_r_kpc=2.0, refine_beta=1.5)


def _run(ic, t_end, snaps, **kw):
    return ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], cronos=False, ds=0.08, dt_myr=0.2).run(t_end, snaps, **kw)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    ic = _ic()
    snaps = [0.0, 0.002, 0.004, 0.006]
    ref = _run(ic, 0.006, snaps)
    ck = tmp_path / "arm.ckpt.npz"
    # tramo 1: la mitad del recorrido (15 pasos exactos de 0.2 Myr) guardando un punto de control en cada paso — emula la
    # interrupción del entorno (el estado guardado es el del final de la última iteración completa)
    p1 = _run(ic, 0.003, snaps, checkpoint=ck, checkpoint_every_s=0.0)
    assert not p1["stopped_early"] and ck.exists() and p1["n_checkpoints"] >= 1 and p1["n_steps"] == 15
    # tramo 2: reanuda desde el fichero con el t_end completo y termina
    p2 = _run(ic, 0.006, snaps, checkpoint=ck, checkpoint_every_s=0.0)
    assert p2["resumed_from_checkpoint"] and not p2["stopped_early"] and p2["t_final_gyr"] == ref["t_final_gyr"]
    assert p2["n_steps"] == ref["n_steps"] and p2["n_checkpoints"] > p1["n_checkpoints"]
    e_ref, e_res = ref["snapshots"][-1]["energy"]["E_grav_only"], p2["snapshots"][-1]["energy"]["E_grav_only"]
    assert e_res == e_ref                                                   # mismas operaciones ⟹ mismo resultado
    assert [s["t_gyr"] for s in p2["snapshots"]] == [s["t_gyr"] for s in ref["snapshots"]]
    assert p2["wall_s"] >= p1["wall_s"]                                    # la pared se acumula entre tramos
    assert ref["resumed_from_checkpoint"] is False and ref["n_checkpoints"] == 0


def test_checkpoint_stores_full_state(tmp_path):
    ic = _ic()
    ck = tmp_path / "c.npz"
    out = _run(ic, 0.002, [0.0, 0.002], checkpoint=ck, checkpoint_every_s=0.0)
    z = np.load(ck, allow_pickle=False)
    assert set(z.files) == {"r", "vr", "L2", "m", "scalars", "meta_json"} and len(z["r"]) == out["N"]
    assert float(z["scalars"][5]) == out["n_steps"]
