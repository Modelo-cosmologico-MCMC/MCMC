"""Perfil emergente en la malla PM: forma cored vs NFW (frente nº 5, medio paso).

Opción G (ronda 5). Sin Gadget-4-Cronos no hay valores absolutos — el
núcleo de 2.3 kpc del corpus queda para producción —, pero la malla PM
propia sí permite dos medidas honestas:

(i) el perfil radial EMERGENTE del colapso — Cronos v3 frente al control
    newtoniano de semilla idéntica (B.5/B.6) — comparado EN FORMA con

        cored (pseudo-isoterma):  ρ(r) = ρ0 / [1 + (r/r_c)²]
        NFW (referencia ΛCDM):    ρ(r) = ρ_s / [(r/r_s)·(1 + r/r_s)²]

    Comparación de formas, no de valores absolutos. ADVERTENCIA DE
    RESOLUCIÓN DECLARADA: a 16³–32³ la zona interna del perfil está
    limitada por la celda; el veredicto del núcleo es de producción.

(ii) la caída de la compuerta al virializar (H.2.5): «la fricción
    corregida cae de ≈4×10⁻⁴ a ≈7×10⁻⁹ (compuerta cerrada), mientras
    la antigua persiste». Esos valores son de la simulación del
    tratado; aquí se mide la CAÍDA EN ÓRDENES DE MAGNITUD en esta
    malla, no esos números absolutos.

El ajuste es de forma en log-espacio: para cada radio de escala de una
malla, la amplitud óptima es analítica; se comparan los RMSE
logarítmicos de ambas formas.
"""

from __future__ import annotations

import numpy as np

from .cronos_v3 import gate_friction, gate_friction_freefall
from .simulation import cic_deposit


def spherical_clump_ic(seed: int, n_p: int, box: float = 10.0,
                       scale: float = 1.0,
                       clump_frac: float = 1.0) -> tuple:
    """Condición inicial reproducible (B.5): sobredensidad esférica fría
    gaussiana en el centro + fondo uniforme opcional (1 − clump_frac).
    Misma semilla ⟹ mismo par A/B (B.5/B.6)."""
    rng = np.random.default_rng(seed)
    n_clump = int(n_p * clump_frac)
    pos = rng.uniform(0.0, box, size=(n_p, 3))
    pos[:n_clump] = box / 2.0 + rng.normal(scale=scale, size=(n_clump, 3))
    return pos % box, np.zeros((n_p, 3)), np.ones(n_p)


def track_region_density(sim, steps: int, dt: float,
                         r_region: float) -> np.ndarray:
    """Avanza la simulación `steps` pasos y devuelve la historia de la
    densidad media de la región esférica de radio `r_region` centrada
    en el pico de densidad instantáneo — el diagnóstico de H.2.5:
    «una región que colapsa y luego viriliza»."""
    vol = 4.0 * np.pi / 3.0 * r_region ** 3
    hist = []
    for _ in range(steps):
        sim.step(dt=dt)
        c = halo_center(sim.pos, sim.mass, sim.box, grid_n=16)
        d = sim.pos - c
        d -= sim.box * np.round(d / sim.box)
        M = sim.mass[np.linalg.norm(d, axis=1) < r_region].sum()
        hist.append(float(M / vol))
    return np.array(hist)


def gate_histories(rho_region: np.ndarray, dt: float, alpha0_inv: float,
                   rho_c: float, smooth: int = 21) -> dict:
    """Γ(t) de la región: con compuerta (Cor. 11.3a) y sin ella (la
    forma de caída libre 11.3b evaluada siempre — el comportamiento
    «antiguo» que H.2.5 dice que persiste). ρ(t) se suaviza con media
    móvil de `smooth` pasos (ruido de cáscara declarado)."""
    kern = np.ones(smooth) / smooth
    rho_s = np.convolve(np.asarray(rho_region, dtype=float), kern,
                        mode="valid")
    rho_dot = np.gradient(rho_s, dt)
    gated = gate_friction(rho_s, rho_dot, alpha0_inv, rho_c)
    ungated = gate_friction_freefall(rho_s, alpha0_inv, rho_c)
    i_peak = int(np.argmax(rho_s[:max(len(rho_s) // 2, 1)]))
    return {"rho_smooth": rho_s, "gated": gated, "ungated": ungated,
            "i_peak": i_peak, "smooth": smooth}


def halo_center(pos: np.ndarray, mass: np.ndarray, box: float,
                grid_n: int = 32) -> np.ndarray:
    """Centro del halo: celda de densidad CIC máxima, refinada con el
    centro de masas (imagen mínima) de las partículas a < box/8."""
    grid = cic_deposit(pos, mass, grid_n, box)
    idx = np.unravel_index(int(np.argmax(grid)), grid.shape)
    h = box / grid_n
    center = (np.array(idx, dtype=float) + 0.5) * h
    d = pos - center
    d -= box * np.round(d / box)
    near = np.linalg.norm(d, axis=1) < box / 8.0
    if near.any():
        w = mass[near] / mass[near].sum()
        center = (center + (w[:, None] * d[near]).sum(axis=0)) % box
    return center


def radial_profile(pos: np.ndarray, mass: np.ndarray, center: np.ndarray,
                   box: float, n_bins: int = 10, r_min: float | None = None,
                   r_max: float | None = None) -> dict:
    """Perfil ρ(r) en cáscaras esféricas logarítmicas (imagen mínima).

    Devuelve solo las cáscaras con partículas (las vacías se descartan
    y se declaran en `n_empty`).
    """
    d = pos - center
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    r_lo = box / 64.0 if r_min is None else r_min
    r_hi = box / 4.0 if r_max is None else r_max
    edges = np.geomspace(r_lo, r_hi, n_bins + 1)
    rho, r_mid, count = [], [], []
    n_empty = 0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (r >= a) & (r < b)
        if not sel.any():
            n_empty += 1
            continue
        vol = 4.0 * np.pi / 3.0 * (b ** 3 - a ** 3)
        rho.append(float(mass[sel].sum() / vol))
        r_mid.append(float(np.sqrt(a * b)))
        count.append(int(sel.sum()))
    return {"r": np.array(r_mid), "rho": np.array(rho),
            "count": np.array(count), "n_empty": n_empty}


def _shape_log(r: np.ndarray, scale: float, kind: str) -> np.ndarray:
    """log10 de la forma (sin amplitud) en los radios r."""
    x = r / scale
    if kind == "cored":
        return -np.log10(1.0 + x ** 2)
    if kind == "nfw":
        return -np.log10(x * (1.0 + x) ** 2)
    raise ValueError("kind: 'cored' o 'nfw'")


def fit_shape(r: np.ndarray, rho: np.ndarray, kind: str,
              n_grid: int = 200) -> dict:
    """Ajuste de forma en log-espacio (dos pasadas de malla sobre la
    escala; amplitud analítica por escala). Devuelve escala, amplitud
    y RMSE logarítmico (dex)."""
    lr = np.log10(rho)

    def scan(scales: np.ndarray) -> tuple[float, float, float]:
        best = (np.inf, np.nan, np.nan)
        for s in scales:
            model = _shape_log(r, s, kind)
            la = float(np.mean(lr - model))
            rmse = float(np.sqrt(np.mean((lr - la - model) ** 2)))
            if rmse < best[0]:
                best = (rmse, s, la)
        return best

    coarse = np.geomspace(r.min() / 4.0, r.max() * 4.0, n_grid)
    rmse, s0, _ = scan(coarse)
    fine = np.geomspace(s0 / 1.6, s0 * 1.6, n_grid)
    rmse, s_best, la_best = scan(fine)
    return {"scale": float(s_best), "amplitude": float(10.0 ** la_best),
            "rmse_log": float(rmse), "kind": kind}


def shape_comparison(r: np.ndarray, rho: np.ndarray) -> dict:
    """Las dos formas sobre el mismo perfil: RMSE log de cada una y la
    preferida (menor RMSE). Δrmse pequeño = la malla no distingue —
    se declara, no se decide."""
    cored = fit_shape(r, rho, "cored")
    nfw = fit_shape(r, rho, "nfw")
    return {
        "cored": cored,
        "nfw": nfw,
        "preferred": "cored" if cored["rmse_log"] < nfw["rmse_log"] else "nfw",
        "delta_rmse_log": float(nfw["rmse_log"] - cored["rmse_log"]),
    }


def friction_drop(gamma: np.ndarray, i_peak: int,
                  tail_frac: float = 0.25) -> dict:
    """La caída de la compuerta (H.2.5) medida sobre la historia Γ(t)
    del halo: Γ_colapso = máx en la fase de colapso (hasta i_peak);
    Γ_vir = máx en el tramo final (último `tail_frac`) — el criterio
    MÁS conservador; se acompaña de la mediana del tramo y de la
    fracción de pasos con Γ = 0 exacto (el Θ(ρ̇) del Cor. 11.3a
    cerrando la compuerta). `orders_drop` usa el máx; si el máx del
    tramo es 0, la compuerta cerró exactamente todo el tramo."""
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    g_col = float(gamma[:max(i_peak, 1)].max())
    tail = gamma[n - max(1, int(tail_frac * n)):]
    g_vir = float(tail.max())
    closed = g_vir == 0.0
    orders = float(np.inf) if closed else float(np.log10(g_col / g_vir)) \
        if g_col > 0.0 else 0.0
    return {"Gamma_collapse": g_col, "Gamma_virial": g_vir,
            "Gamma_virial_median": float(np.median(tail)),
            "tail_zero_frac": float((tail == 0.0).mean()),
            "orders_drop": orders, "gate_closed_exactly": closed}
