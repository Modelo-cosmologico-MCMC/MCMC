"""Ciclo Cronos-KDK completo en malla PM (v35, Apéndice B, B.1-B.3).

Implementación mínima y determinista del ciclo del tratado:

    1. Kick 1:   v += (Δt/2)·a,  con a = −∇Φ_N − ∇Φ_id  (B.3)
                 y la fricción de compuerta del cap. 11 aplicada como
                 v ← v·exp(−Γ·Δt/2), Γ = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇) (Cor. 11.3a)
    2. Drift:    x += Δt·(1 − ζ(x))·v,  ζ = ζ0·ρ_lat/(ρ_lat+ρ*)  (B.3)
    3. Refresco: ρ_lat, ρ_id según channels.refresh_channels  (B.3)
    4. Poisson:  ∇²Φ_N = 4πG·a²·δρ_m, ∇²Φ_id = 4πG·a²·δρ_id  (B.2)
                 y Kick 2 espejo.

El paso Δt es el entrópico de la ec. (B.3) (timestep.entropic_timestep).

ALCANCE DECLARADO: esto NO es Gadget-4-Cronos (la variante de
producción del tratado, con árbol octal, MPI, FFTW y cajas 512³-1024³,
frente abierto nº 5). Es una malla PM (CIC + FFT) suficiente para
demostrar la firma falsable de la compuerta (fig. 11.1) en colapsos
pequeños reproducibles con pares de semillas idénticas (B.5/B.6).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C

from .channels import local_dilation, refresh_channels
from .cronos_v3 import gate_friction
from .poisson import gradient, solve_poisson
from .timestep import entropic_timestep


def cic_deposit(pos: np.ndarray, mass: np.ndarray, n: int,
                box: float) -> np.ndarray:
    """Depósito Cloud-In-Cell de partículas a malla de densidad (n,n,n)."""
    grid = np.zeros((n, n, n))
    h = box / n
    u = (pos / h) - 0.5
    i0 = np.floor(u).astype(int)
    f = u - i0
    for dx in (0, 1):
        wx = np.where(dx == 0, 1.0 - f[:, 0], f[:, 0])
        ix = (i0[:, 0] + dx) % n
        for dy in (0, 1):
            wy = np.where(dy == 0, 1.0 - f[:, 1], f[:, 1])
            iy = (i0[:, 1] + dy) % n
            for dz in (0, 1):
                wz = np.where(dz == 0, 1.0 - f[:, 2], f[:, 2])
                iz = (i0[:, 2] + dz) % n
                np.add.at(grid, (ix, iy, iz), mass * wx * wy * wz)
    return grid / h ** 3  # densidad = masa / volumen de celda


def cic_gather(field: np.ndarray, pos: np.ndarray, box: float) -> np.ndarray:
    """Interpolación CIC de un campo de malla a las posiciones."""
    n = field.shape[0]
    h = box / n
    u = (pos / h) - 0.5
    i0 = np.floor(u).astype(int)
    f = u - i0
    out = np.zeros(len(pos))
    for dx in (0, 1):
        wx = np.where(dx == 0, 1.0 - f[:, 0], f[:, 0])
        ix = (i0[:, 0] + dx) % n
        for dy in (0, 1):
            wy = np.where(dy == 0, 1.0 - f[:, 1], f[:, 1])
            iy = (i0[:, 1] + dy) % n
            for dz in (0, 1):
                wz = np.where(dz == 0, 1.0 - f[:, 2], f[:, 2])
                iz = (i0[:, 2] + dz) % n
                out += field[ix, iy, iz] * wx * wy * wz
    return out


class CronosPM:
    """Simulación PM mínima con el ciclo B.3 y la compuerta del cap. 11."""

    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                 grid_n: int, box: float, *, a_scale: float = 1.0,
                 G: float = 1.0, alpha_cr: float = C.ALPHA_CRONOS,
                 rho_c: float | None = None, alpha0_inv: float = 0.0,
                 zeta0: float = 0.0, rho_star: float = 1.0,
                 kappa_lat: float = 0.0, Gamma_lat: float = 0.0,
                 eta_dir: float = 0.0, Gamma_act: float = 0.0):
        self.pos = np.asarray(pos, dtype=float) % box
        self.vel = np.asarray(vel, dtype=float)
        self.mass = np.asarray(mass, dtype=float)
        self.n = grid_n
        self.box = box
        self.a = a_scale
        self.G = G
        self.alpha_cr = alpha_cr
        self.alpha0_inv = alpha0_inv
        self.zeta0 = zeta0
        self.rho_star = rho_star
        self.rates = {"kappa_lat": kappa_lat, "Gamma_lat": Gamma_lat,
                      "eta_dir": eta_dir, "Gamma_act": Gamma_act}
        self.rho_id = np.zeros((grid_n, grid_n, grid_n))
        self.rho_lat = np.zeros((grid_n, grid_n, grid_n))
        self.rho_m = cic_deposit(self.pos, self.mass, grid_n, box)
        # ρ_c de referencia: por defecto, la densidad media de la caja
        self.rho_c = rho_c if rho_c is not None else float(self.rho_m.mean())
        self._rho_local_prev = cic_gather(self.rho_m, self.pos, box)

    # -- campos ------------------------------------------------------
    def _accel(self) -> np.ndarray:
        """a = −∇(Φ_N + Φ_id) interpolada a las partículas (B.2/B.3)."""
        delta_m = self.rho_m - self.rho_m.mean()
        phi_N = solve_poisson(delta_m, self.box, a_scale=self.a, G=self.G)
        phi = phi_N
        if self.rho_id.any():
            phi = phi + solve_poisson(self.rho_id - self.rho_id.mean(),
                                      self.box, a_scale=self.a, G=self.G)
        grad = gradient(phi, self.box)
        acc = np.stack([cic_gather(-grad[i], self.pos, self.box)
                        for i in range(3)], axis=1)
        return acc

    # -- ciclo -------------------------------------------------------
    def step(self, dt: float | None = None) -> dict:
        """Un ciclo completo B.3 (KDK). Devuelve diagnósticos del paso."""
        acc = self._accel()
        rho_loc = cic_gather(self.rho_m, self.pos, self.box)
        if dt is None:
            dt = float(np.min(entropic_timestep(
                np.linalg.norm(acc, axis=1), self.a, rho_loc,
                self.rho_c, self.alpha_cr)))
        rho_dot = (rho_loc - self._rho_local_prev) / dt

        # Kick 1 con compuerta (Cor. 11.3a; ε_c y cota de cronos_v3)
        Gamma = gate_friction(rho_loc, rho_dot, self.alpha0_inv, self.rho_c)
        self.vel *= np.exp(-Gamma * dt / 2.0)[:, None]
        self.vel += acc * dt / 2.0

        # Drift con dilatación local (B.3 paso 2)
        zeta_p = cic_gather(local_dilation(self.rho_lat, self.zeta0,
                                           self.rho_star),
                            self.pos, self.box)
        self.pos = (self.pos + dt * (1.0 - zeta_p)[:, None] * self.vel) % self.box

        # Refresco de canales (B.3 paso 3) y redepósito
        self.rho_lat, self.rho_id = refresh_channels(
            self.rho_lat, self.rho_id, self.rho_m, dt, **self.rates)
        self.rho_m = cic_deposit(self.pos, self.mass, self.n, self.box)

        # Poisson y Kick 2 (B.3 paso 4), con la compuerta espejo
        acc2 = self._accel()
        rho_loc2 = cic_gather(self.rho_m, self.pos, self.box)
        rho_dot2 = (rho_loc2 - rho_loc) / dt
        Gamma2 = gate_friction(rho_loc2, rho_dot2, self.alpha0_inv, self.rho_c)
        self.vel += acc2 * dt / 2.0
        self.vel *= np.exp(-Gamma2 * dt / 2.0)[:, None]

        self._rho_local_prev = rho_loc2
        return {
            "dt": dt,
            "rho_max": float(self.rho_m.max()),
            "rho_local_max": float(rho_loc2.max()),
            "Gamma_mean": float(np.mean(0.5 * (Gamma + Gamma2))),
            "Gamma_max": float(np.max(0.5 * (Gamma + Gamma2))),
            "v_rms": float(np.sqrt(np.mean(self.vel ** 2))),
        }
