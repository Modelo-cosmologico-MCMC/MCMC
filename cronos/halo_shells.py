"""Código de CAPAS ESFÉRICAS para el halo con la Ley de Cronos débil
(frente 5 (b), ronda 2 — orden del autor del 22-sep, §7).

Por qué capas y no más partículas: el problema es esféricamente simétrico
por construcción (la ley usa ρ̄(r)), y el diagnóstico de la ronda 1 fue
que con 18–87 partículas dentro de 0.4 kpc la ε_c ∝ ρ^{3/2} reconstruida
fluctuaba en O(1) entre actualizaciones — un pozo que fluctúa más deprisa
que el periodo orbital interior calienta y expulsa (el trinquete del
Nivel A en otra forma). Un código de capas lleva (r, v_r, L²) por capa,
gravedad M(<r) EXACTA (rango de la capa), ε_c del perfil de capas con el
MISMO estimador esférico del N-cuerpos (spline suavizante de ln M sobre
una malla logarítmica fina, radio suavizado r_s = √(r² + ε_soft²)) y
energía K + W + (2/5)U_C + W_fric con miles de capas dentro de 0.4 kpc
(refinamiento de masa), de modo que el ruido de conteo del campo, que
dominó la ronda 1, desaparece. Captura exactamente los modos
RADIALES del criterio de Cronos–Jeans (los inestables) y deja fuera los
no radiales: declarado.

Ley (Cor. 11.3, v3 completa) sobre el perfil esférico ρ(r):
    ε_c = A·ρ^{3/2}          (ρ en M☉/pc³; A ≡ A_Sculptor o fracción)
    a_C = +c²·dε_c/dr         (radial, hacia dentro)
    Γ  = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇)  (fricción con compuerta; ρ̇ lagrangiana)
    N  = 1 + Φ_N/c² − ε_c     (lapso; kick y drift lo llevan)

Balance de energía (frente 5b): E_self = K + W + U_self + W_fric con
U_self = (2/5)·Σm(−c²ε_c) (la fuerza deriva de −(2/5)c²A∫ρ^{5/2}dV) y
W_fric el trabajo acumulado de la fricción. Con el campo dinámico eso es
lo que se conserva; K + W + Σm(−c²ε_c) solo con el campo congelado.

Integrador: paso GLOBAL dt_max declarado con el campo CONGELADO dentro del
paso (rangos M(<r) de Hénon, ε̃, lapso, fricción; orden O(N log N) por
paso) y SUBPASOS por capa 2^k (k por la escala orbital propia: r/|v|,
√(r³/GM), pericentro L/v²) con salto de rana KDK: el término centrífugo
L²/r³ se integra EXACTO — suavizarlo con ε_soft vacía las barreras de
momento angular y contrae el núcleo newtoniano (piloto: M(<0.2 kpc)
+75 %), y un paso global fijo no resuelve los pericentros de las capas
casi radiales del interior refinado (piloto: |ΔE/E| ~ 1e2). La gravedad
es de Plummer suavizada con ε_soft (la escala del N-cuerpos). Con Cronos
el paso global se acorta para que ε_c máx cambie ≤ η_field por paso
(colapso de la cúspide). Con REFINAMIENTO DE MASA en el centro (capas
más ligeras muestreadas más densamente, sin tocar la distribución de masa
ni la de velocidades) el interior queda resuelto con miles de capas; lo que
ocurre por debajo de ε_soft no es decidible por este instrumento
(declarado). La energía de un código de capas está limitada por los cruces
(la fuerza salta cuando dos capas se cruzan) y por la congelación del campo
en el paso global: la puerta preinscrita |ΔE_self/E| ≤ tol_E juzga si el
paso declarado basta; si no, es un error de instrumento, no física.

Unidades: kpc, M☉, km/s (1 kpc/(km/s) = 0.977792 Gyr); G = 4.30091e-6.
"""

from __future__ import annotations

import time

import numpy as np

from cronos.halo_nbody import (
    A_SCULPTOR,
    C_KMS,
    G_KPC,
    GYR_PER_TIME_UNIT,
    KPC3_TO_PC3,
    sample_equilibrium_nfw,
)

__all__ = ["A_SCULPTOR", "KernelCronosField", "ShellRun", "equilibrium_shells", "shells_from_3d"]


def shells_from_3d(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, v_r, L²) de posiciones y velocidades 3D (Eddington isótropo)."""
    r = np.sqrt(np.sum(pos ** 2, axis=1))
    vr = np.sum(pos * vel, axis=1) / r
    L2 = np.sum(np.cross(pos, vel) ** 2, axis=1)
    return r, vr, L2


def _bspline3(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B-spline cúbico W(u) (soporte |u| < 2, Σ_b W = 1 sobre una malla unitaria) y dW/du."""
    a = np.abs(u)
    W = np.where(a < 1.0, (4.0 - 6.0 * a ** 2 + 3.0 * a ** 3) / 6.0, np.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0))
    dW = np.where(a < 1.0, -2.0 * a + 1.5 * a ** 2, np.where(a < 2.0, -0.5 * (2.0 - a) ** 2, 0.0)) * np.sign(u)
    return W, dW


class KernelCronosField:
    """Campo de Cronos CONSERVATIVO por construcción para el código de capas.

    La densidad se deposita sobre una malla fija en s = ln √(r² + ε²) (radio
    logarítmico suavizado) con el núcleo B-spline cúbico W_b, ρ_b = Σ_i m_i W_b(s_i)/V_b
    (V_b = ∫W_b(s(r)) 4πr² dr: la deposición es exacta para ρ uniforme), ε_b = Aρ_b^{3/2},
    y ε_c en una capa es la interpolación CON EL MISMO NÚCLEO, ε̃(r) = Σ_b ε_b W_b(s(r)).
    Entonces la fuerza a_i = c² dε̃/dr(r_i) es EXACTAMENTE −(1/m_i)∂U/∂r_i con
    U = −(2/5)c²A Σ_b V_b ρ_b^{5/2} = (2/5)Σ_i m_i(−c²ε̃(r_i)) (frente 5b): el balance
    K + W + U_self + W_fric se conserva salvo el error del paso de tiempo, y la puerta
    |ΔE_self/E| ≤ tol_E juzga al integrador, no al estimador. El estimador spline
    suavizante del N-cuerpos NO tiene esta propiedad (piloto: |ΔE/E| ≈ 1e-2 a A_Sculptor,
    independiente de dt y de la malla) y por eso no se usa aquí. La RESOLUCIÓN del campo
    es ds (= Δln r lejos del centro); la serie la varía ×½/×1/×2. Bajo r ≲ ε el radio
    suavizado hace dε̃/dr → 0 (declarado: no decidible por el instrumento)."""

    def __init__(self, A: float, eps_soft: float, ds: float, r_max: float = 400.0):
        self.A, self.eps, self.ds = float(A), float(eps_soft), float(ds)
        s_min, s_max = np.log(self.eps), np.log(np.sqrt(r_max ** 2 + self.eps ** 2))
        self.s0 = s_min - 2.0 * self.ds                        # margen del núcleo (soporte 2 celdas)
        self.nb = int(np.ceil((s_max - self.s0) / self.ds)) + 3
        self.s_b = self.s0 + self.ds * np.arange(self.nb)
        # volumen del núcleo de cada celda: V_b = ∫ W_b(s(r)) 4πr² dr (cuadratura fina en r)
        r_fine = np.geomspace(1e-5 * self.eps, r_max * 1.05, 40000)
        s_fine = self._s(r_fine)
        W_all = self._weights(s_fine)                          # (n_fine, 4) pesos y (n_fine, 4) índices
        dV = np.gradient(4.0 / 3.0 * np.pi * r_fine ** 3)
        vol = np.bincount(W_all[1].ravel(), weights=(W_all[0] * dV[:, None]).ravel(), minlength=self.nb)
        # la celda del margen s_b ≤ ln ε − 2ds no tiene soporte físico (W = 0 para todo r): sin densidad; las demás
        # celdas del margen tienen volumen pequeño pero finito y su ρ_b es la media ponderada por el núcleo (sin sesgo)
        self.V_b = np.where(vol > 0.0, vol, np.inf)
        self.rho_b = np.zeros(self.nb)
        self.rho_b_prev, self.t, self.t_prev = None, None, None
        self.eps_b = np.zeros(self.nb)
        self._cache_r, self._cache_w = None, None                # pesos de las capas del último update (reutilizados)

    def _s(self, r):
        return 0.5 * np.log(np.asarray(r, float) ** 2 + self.eps ** 2)

    def _weights(self, s):
        """Pesos del B-spline cúbico y las 4 celdas de soporte para cada s."""
        t = (s - self.s0) / self.ds
        k = np.floor(t).astype(int)
        idx = np.clip(k[:, None] + np.array([-1, 0, 1, 2])[None, :], 0, self.nb - 1)
        u = t[:, None] - (k[:, None] + np.array([-1, 0, 1, 2])[None, :])
        W, dW = _bspline3(u)
        return W, idx, dW

    # ------------------------------------------------------------- depósito
    def update(self, r: np.ndarray, m: np.ndarray, t: float) -> None:
        W, idx, dW = self._weights(self._s(r))
        self._cache_r, self._cache_w = r, (W, idx, dW)
        M_b = np.bincount(idx.ravel(), weights=(W * m[:, None]).ravel(), minlength=self.nb)
        self.rho_b_prev, self.t_prev = self.rho_b, self.t
        self.rho_b, self.t = M_b / self.V_b, float(t)
        self.eps_b = self.A * (self.rho_b * KPC3_TO_PC3) ** 1.5
        self.M_tot = float(m.sum())

    # --------------------------------------------------------- interpolación
    def _interp(self, r, y_b, deriv: bool = False):
        r = np.asarray(r, float)
        if self._cache_r is not None and r is self._cache_r:
            W, idx, dW = self._cache_w
        else:
            W, idx, dW = self._weights(self._s(r))
        if not deriv:
            return np.sum(W * y_b[idx], axis=1)
        ds_dr = r / (r ** 2 + self.eps ** 2)                   # ds/dr → 0 en el centro
        return np.sum(dW * y_b[idx], axis=1) / self.ds * ds_dr

    def rho(self, r):
        return self._interp(r, self.rho_b)

    def drho_dr(self, r):
        return self._interp(r, self.rho_b, deriv=True)

    def drho_dt(self, r):
        if self.t_prev is None or self.t == self.t_prev:
            return np.zeros_like(np.asarray(r, float))
        return self._interp(r, (self.rho_b - self.rho_b_prev) / (self.t - self.t_prev))

    def eps_c(self, r):
        return self._interp(r, self.eps_b)

    def deps_c_dr(self, r):
        return self._interp(r, self.eps_b, deriv=True)

    def U_self_bins(self) -> float:
        """−(2/5)c²A Σ_b V_b ρ_b^{5/2}: debe coincidir con (2/5)Σ_i m_i(−c²ε̃(r_i)) (identidad del núcleo)."""
        ok = np.isfinite(self.V_b)
        return -0.4 * C_KMS ** 2 * self.A * float(np.sum(self.V_b[ok] * (self.rho_b[ok] * KPC3_TO_PC3) ** 2.5 / KPC3_TO_PC3))

    def eps_max(self) -> float:
        return float(self.eps_b.max())


class ShellRun:
    """Halo de capas con y sin Cronos v3; salto de rana KDK global."""

    def __init__(self, r, vr, L2, m, *, A: float = A_SCULPTOR, cronos: bool = True, ds: float = 0.04,
                 eps_soft: float = 0.1, dt_myr: float = 0.15, weak_max: float = 1e-3,
                 stop_speed_kms: float | None = 1000.0, r_stop_kpc: float | None = None, max_wall_s: float = 3600.0 * 3,
                 eta_dt: float = 0.05, eta_field: float = 0.02, k_max: int = 10, dt_min_myr: float = 1e-4,
                 eps_L_factor: float = 0.05):
        self.r, self.vr, self.L2 = np.asarray(r, float).copy(), np.asarray(vr, float).copy(), np.asarray(L2, float).copy()
        self.N = len(self.r)
        self.m = np.broadcast_to(np.asarray(m, float), (self.N,)).copy()     # masas de capa (pueden variar: refinamiento)
        self.A, self.cronos = float(A), bool(cronos)
        self.dt = dt_myr / 1000.0 / GYR_PER_TIME_UNIT          # kpc/(km/s): paso GLOBAL máximo
        # subpasos por capa (η_dt·escala orbital propia, 2^k con k ≤ k_max) dentro del paso global con el campo
        # congelado; el paso global se acorta si ε_c máx cambia más de η_field por paso (colapso de la cúspide);
        # suelo dt_min declarado (alcanzarlo es parada declarada)
        self.eta_dt, self.eta_field, self.k_max = float(eta_dt), float(eta_field), int(k_max)
        self.dt_min = dt_min_myr / 1000.0 / GYR_PER_TIME_UNIT
        self.dt_hist = {"min_myr": dt_myr, "n_reduced": 0, "n_clamped": 0, "k_max_seen": 0}
        self.eps_soft, self.weak_max = float(eps_soft), float(weak_max)
        # barrera centrífuga L²/(2(r² + ε_L²)) con ε_L = ε_soft·eps_L_factor (declarado): SOLO regulariza el paso por
        # pericentros r_p < ε_L (fracción ~1e-3 de las capas) — con ε_L = ε_soft la barrera desaparece para
        # r_p < 0.2 kpc y el núcleo newtoniano se contrae (piloto: M(<0.2 kpc) +75 %); sin regularizar, el
        # sobrepaso del pericentro en un subpaso finito da patadas L²/r³ no acotadas (piloto: |ΔE/E| ~ 1e8)
        self.eps_L = self.eps_soft * float(eps_L_factor)
        self.stop_speed, self.r_stop, self.max_wall_s = stop_speed_kms, r_stop_kpc, float(max_wall_s)
        # campo conservativo por núcleo (KernelCronosField): resolución ds; malla fija en s = ln √(r² + ε²)
        self.field = KernelCronosField(A, eps_soft=eps_soft, ds=ds)
        self.t, self.W_fric = 0.0, 0.0
        self.events, self.stopped_early, self.stop_reason = [], False, None
        self._update_field()
        eps_max = self._eps_max()
        self.events.append({"t_gyr": 0.0, "kind": "inicio", "eps_c_max": eps_max, "weak_regime_ok": bool(eps_max <= self.weak_max),
                            "D_F_at_soft": self._force_dominance(self.eps_soft) if self.cronos else 0.0,
                            "ds": ds, "n_bins": self.field.nb, "N": self.N, "dt_myr": dt_myr,
                            "U_self_identity_rel": self._u_self_identity()})

    def _eps_max(self) -> float:
        return self.field.eps_max() if self.cronos else 0.0

    def _r_eps_max(self) -> float:
        """Radio (kpc) de la celda donde ε_c es máxima (r_b = √(e^{2s_b} − ε²), 0 bajo el centro suavizado)."""
        if not self.cronos:
            return 0.0
        b = int(np.argmax(self.field.eps_b))
        return float(np.sqrt(max(np.exp(2.0 * self.field.s_b[b]) - self.eps_soft ** 2, 0.0)))

    def _force_dominance(self, r_kpc: float) -> float:
        """D_F(r) = |c² dε̃/dr| / (G M(<r)/r²) en r (cociente de fuerzas)."""
        r = np.array([r_kpc])
        M_in = float(self.m[self.r < r_kpc].sum())
        g_N = G_KPC * M_in / r_kpc ** 2
        return float(C_KMS ** 2 * abs(self.field.deps_c_dr(r)[0]) / max(g_N, 1e-300))

    def _u_self_identity(self) -> float:
        """|U_self(capas) − U_self(celdas)| / |U_self|: la identidad del núcleo (≈ 1e-15 si el depósito y la
        interpolación usan el mismo W); 0 si no hay Cronos."""
        if not self.cronos:
            return 0.0
        u_i = 0.4 * (-C_KMS ** 2) * float(np.sum(self.m * self.field.eps_c(self.r)))
        u_b = self.field.U_self_bins()
        return abs(u_i - u_b) / max(abs(u_b), 1e-300)

    # ------------------------------------------------------------ campo y fuerzas
    def _update_field(self) -> None:
        self._order = np.argsort(self.r)
        m_sorted = self.m[self._order]
        m_cum = np.cumsum(m_sorted)
        # masa estrictamente interior a cada capa y masa de Hénon (interior + ½ propia)
        self._M_in = np.empty(self.N)
        self._M_in[self._order] = m_cum - m_sorted
        self._M_henon = self._M_in + 0.5 * self.m
        # Φ_N exacto por rangos con el radio suavizado: −G M_Hénon/r_s − G Σ_{j exterior} m_j/r_s,j (para el lapso)
        r_s_sorted = np.sqrt(self.r[self._order] ** 2 + self.eps_soft ** 2)
        outer = np.cumsum((m_sorted / r_s_sorted)[::-1])[::-1] - m_sorted / r_s_sorted
        self._Phi = np.empty(self.N)
        self._Phi[self._order] = -G_KPC * (self._M_henon[self._order] / r_s_sorted + outer)
        self.field.update(self.r, self.m, self.t)

    def _accel(self) -> tuple[np.ndarray, np.ndarray]:
        """Aceleración radial y lapso N por capa (campo ya actualizado): gravedad de Plummer suavizada con ε_soft
        (la escala del N-cuerpos), término centrífugo L²/r³ EXACTO (suavizarlo vacía las barreras de momento
        angular y contrae el núcleo newtoniano: piloto declarado, M(<0.2 kpc) +75 %) y fuerza de Cronos c²dε̃/dr."""
        a = self._frozen_accel(self.r, self.L2, self._M_henon)
        lapse = np.ones(self.N)
        if self.cronos:
            lapse = 1.0 + self._Phi / C_KMS ** 2 - self.field.eps_c(self.r)
        return a, lapse

    def _frozen_accel(self, r: np.ndarray, L2: np.ndarray, M_h: np.ndarray) -> np.ndarray:
        a = -G_KPC * M_h * r / (r ** 2 + self.eps_soft ** 2) ** 1.5 + L2 * r / (r ** 2 + self.eps_L ** 2) ** 2
        if self.cronos:
            a = a + C_KMS ** 2 * self.field.deps_c_dr(r)
        return a

    def _friction(self) -> np.ndarray:
        """Γ = (3/2)(ρ̇/ρ)ε_c Θ(ρ̇) con ρ̇ lagrangiana = ∂_tρ + v_r ∂_rρ."""
        if not self.cronos:
            return np.zeros(self.N)
        rho = self.field.rho(self.r)
        rdot = self.field.drho_dt(self.r) + self.vr * self.field.drho_dr(self.r)     # ρ̇ lagrangiana sobre el perfil
        return np.where(rdot > 0.0, 1.5 * rdot / rho * self.field.eps_c(self.r), 0.0)

    # ----------------------------------------------------------------- energía
    def _kinetic(self) -> float:
        return 0.5 * float(np.sum(self.m * (self.vr ** 2 + self.L2 / (self.r ** 2 + self.eps_L ** 2))))

    def energies(self) -> dict:
        K = self._kinetic()
        W = -G_KPC * float(np.sum(self.m * self._M_in / np.sqrt(self.r ** 2 + self.eps_soft ** 2)))   # −Σ_{i<j} G m_i m_j /√(r_j² + ε²)
        U_ext = -C_KMS ** 2 * float(np.sum(self.m * self.field.eps_c(self.r))) if self.cronos else 0.0
        U_self = 0.4 * U_ext
        return {"K": K, "W": W, "U_ext": U_ext, "U_self": U_self, "W_fric": self.W_fric,
                "E_grav_only": K + W, "E": K + W + U_ext, "E_self": K + W + U_self + self.W_fric}

    def snapshot(self, radii_kpc=(0.1, 0.2, 0.4, 1.0, 2.3, 5.0)) -> dict:
        rs, ms = self.r[self._order], self.m[self._order]
        m_cum = np.cumsum(ms)
        edges = np.geomspace(0.05, 50.0, 31)
        mass_in_bins = np.histogram(rs, bins=edges, weights=ms)[0]
        vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        idx = {str(x): int(np.searchsorted(rs, x, side="right")) for x in radii_kpc}
        return {"t_gyr": self.t * GYR_PER_TIME_UNIT, "energy": self.energies(),
                "M_within": {k: float(m_cum[i - 1]) if i > 0 else 0.0 for k, i in idx.items()},
                "N_within": idx,
                "r_mid_kpc": np.sqrt(edges[1:] * edges[:-1]).tolist(), "rho": (mass_in_bins / vol).tolist(),
                "r_min_kpc": float(rs[0]), "eps_c_max": self._eps_max(), "r_eps_max_kpc": self._r_eps_max(),
                "U_self_identity_rel": self._u_self_identity(),
                "half_mass_r_kpc": float(rs[np.searchsorted(m_cum, 0.5 * m_cum[-1])])}

    # -------------------------------------------------------------- integración
    def run(self, t_end_gyr: float, snapshot_gyr, log=None) -> dict:
        """Paso GLOBAL dt (≤ dt_max; con Cronos, limitado a que ε_c máx cambie ≤ η_f por paso) en el que el campo
        (rangos M(<r), ε̃, lapso, fricción) se CONGELA, y dentro de él cada capa se integra con salto de rana KDK
        en 2^k subpasos propios (k por su escala orbital: r/|v|, √(r³/GM), pericentro L/v²) — así el paso por el
        pericentro de las capas casi radiales (L²/r³) queda resuelto sin frenar a todas las demás. El error de la
        congelación lo juzga la puerta de energía preinscrita."""
        t_end = t_end_gyr / GYR_PER_TIME_UNIT
        snaps_due = sorted(float(s) / GYR_PER_TIME_UNIT for s in snapshot_gyr)
        snapshots = [self.snapshot()] if snaps_due and snaps_due[0] <= 0.0 else []
        snaps_due = [s for s in snaps_due if s > 0.0]
        wall0, n_steps, n_sub_total = time.time(), 0, 0
        dt_glob = self.dt
        eps_prev = self._eps_max()
        while self.t < t_end - 1e-12:
            dt = min(dt_glob, t_end - self.t)
            if dt < self.dt_min:
                self.stopped_early, self.stop_reason = True, f"paso global por debajo del suelo dt_min = {self.dt_min * GYR_PER_TIME_UNIT * 1000.0:.1e} Myr"
                snapshots.append(self.snapshot())
                break
            # --- campo congelado en el inicio del paso -----------------------------------------------
            r_sorted = self.r[self._order]
            M_h_sorted = self._M_henon[self._order]
            gam = self._friction()
            lapse = (1.0 + self._Phi / C_KMS ** 2 - self.field.eps_c(self.r)) if self.cronos else np.ones(self.N)
            # --- subpasos por capa: 2^k con k por la escala orbital propia -----------------------------
            r_L = np.sqrt(self.r ** 2 + self.eps_L ** 2)
            speed = np.sqrt(self.vr ** 2 + self.L2 / r_L ** 2) + 1e-30
            a0 = np.abs(self._frozen_accel(self.r, self.L2, self._M_henon)) + 1e-30
            tau = np.minimum(np.minimum(r_L / speed, np.sqrt(r_L / a0)), np.sqrt(self.r ** 3 / (G_KPC * np.clip(self._M_henon, 1e-30, None))))
            k = np.clip(np.ceil(np.log2(np.clip(dt / (self.eta_dt * tau), 1.0, None))), 0, self.k_max).astype(int)
            self.dt_hist["n_clamped"] += int(np.sum(dt / (self.eta_dt * tau) > 2.0 ** self.k_max))
            self.dt_hist["k_max_seen"] = max(self.dt_hist["k_max_seen"], int(k.max()))
            for kk in np.unique(k):
                sel = np.nonzero(k == kk)[0]
                n_sub = 2 ** int(kk)
                h = dt / n_sub
                r_g, vr_g, L2_g = self.r[sel], self.vr[sel], self.L2[sel]
                lap_g, damp_g = lapse[sel], np.exp(-0.5 * h * gam[sel])
                m_g = self.m[sel]
                a_g = self._frozen_accel(r_g, L2_g, np.interp(r_g, r_sorted, M_h_sorted))
                for _ in range(n_sub):
                    vr_g = vr_g + 0.5 * h * lap_g * a_g
                    K0 = 0.5 * float(np.sum(m_g * (vr_g ** 2 + L2_g / (r_g ** 2 + self.eps_L ** 2))))
                    vr_g, L2_g = vr_g * damp_g, L2_g * damp_g ** 2
                    self.W_fric += K0 - 0.5 * float(np.sum(m_g * (vr_g ** 2 + L2_g / (r_g ** 2 + self.eps_L ** 2))))
                    r_new = r_g + h * lap_g * vr_g
                    through = r_new < 0.0
                    r_g = np.abs(r_new)
                    vr_g = np.where(through, -vr_g, vr_g)
                    a_g = self._frozen_accel(r_g, L2_g, np.interp(r_g, r_sorted, M_h_sorted))
                    vr_g = vr_g + 0.5 * h * lap_g * a_g
                    K0 = 0.5 * float(np.sum(m_g * (vr_g ** 2 + L2_g / (r_g ** 2 + self.eps_L ** 2))))
                    vr_g, L2_g = vr_g * damp_g, L2_g * damp_g ** 2
                    self.W_fric += K0 - 0.5 * float(np.sum(m_g * (vr_g ** 2 + L2_g / (r_g ** 2 + self.eps_L ** 2))))
                self.r[sel], self.vr[sel], self.L2[sel] = r_g, vr_g, L2_g
                n_sub_total += n_sub * len(sel)
            self.t += dt
            self._update_field()
            n_steps += 1
            # --- paso global siguiente: el campo de Cronos no debe cambiar más de η_f por paso -----------
            eps_now = self._eps_max()
            if self.cronos and eps_prev > 0.0 and eps_now > 0.0:
                frac = abs(np.log(eps_now / eps_prev))
                dt_glob = float(np.clip(dt * self.eta_field / max(frac, 1e-12), 0.0, min(2.0 * dt, self.dt)))
                if dt_glob < self.dt:
                    self.dt_hist["n_reduced"] += 1
                    self.dt_hist["min_myr"] = min(self.dt_hist["min_myr"], dt_glob * GYR_PER_TIME_UNIT * 1000.0)
            eps_prev = eps_now
            # reglas de parada declaradas
            if self.stop_speed is not None and self.cronos and float(np.max(np.abs(self.vr))) > self.stop_speed:
                self.stopped_early, self.stop_reason = True, f"velocidad radial > {self.stop_speed} km/s (pozo desbocado)"
            if self.r_stop is not None and float(np.min(self.r)) < self.r_stop:
                self.stopped_early, self.stop_reason = True, f"capa por debajo de r_stop = {self.r_stop} kpc"
            if time.time() - wall0 > self.max_wall_s:
                self.stopped_early, self.stop_reason = True, "presupuesto de tiempo de pared agotado"
            if self.cronos and eps_now > self.weak_max:
                self.events.append({"t_gyr": self.t * GYR_PER_TIME_UNIT, "kind": "régimen débil violado", "eps_c_max": eps_now,
                                    "weak_regime_ok": False})
                self.stopped_early, self.stop_reason = True, f"ε_c máx = {eps_now:.2e} > {self.weak_max} (fuera del régimen débil)"
            while snaps_due and self.t >= snaps_due[0] - 1e-12:
                snaps_due.pop(0)
                snapshots.append(self.snapshot())
                if log:
                    e = snapshots[-1]["energy"]
                    log(f"    t = {snapshots[-1]['t_gyr']:.4f} Gyr: M(<0.4) = {snapshots[-1]['M_within']['0.4']:.3e}, M(<0.1) = {snapshots[-1]['M_within']['0.1']:.3e}, "
                        f"N(<0.4) = {snapshots[-1]['N_within']['0.4']}, E_self = {e['E_self']:.6e}, W_fric = {e['W_fric']:.3e}, ε_máx = {snapshots[-1]['eps_c_max']:.2e}")
            if self.stopped_early:
                snapshots.append(self.snapshot())
                break
        self.dt_hist["n_substeps_total"] = n_sub_total
        return {"snapshots": snapshots, "events": self.events, "t_final_gyr": self.t * GYR_PER_TIME_UNIT,
                "stopped_early": self.stopped_early, "stop_reason": self.stop_reason, "n_steps": n_steps,
                "wall_s": round(time.time() - wall0, 1), "N": self.N, "m_min": float(self.m.min()), "m_max": float(self.m.max()),
                "A": self.A, "cronos": self.cronos,
                "ds": self.field.ds, "n_bins": self.field.nb, "eps_soft": self.eps_soft,
                "dt_max_myr": self.dt * GYR_PER_TIME_UNIT * 1000.0, "dt_adaptive": self.dt_hist, "eta_dt": self.eta_dt, "eta_field": self.eta_field, "k_max": self.k_max,
                "eps_L_kpc": self.eps_L}


def _speeds_inverse_cdf(Psi: np.ndarray, E_t: np.ndarray, f_t: np.ndarray, rng, n_v: int = 96, chunk: int = 20000) -> np.ndarray:
    """|v| de la distribución v² f(Ψ − v²/2) en cada Ψ por CDF inversa sobre n_v velocidades
    (interpolación log-log de f como en cronos.halo_nbody.sample_equilibrium_nfw), vectorizada."""
    logE, logf = np.log(E_t), np.log(np.clip(f_t, 1e-300, None))
    u_grid = np.arange(1, n_v + 1) / (n_v + 1.0)
    v = np.empty(len(Psi))
    for i0 in range(0, len(Psi), chunk):
        Ps = Psi[i0:i0 + chunk][:, None]
        vgrid = np.sqrt(2.0 * Ps) * u_grid[None, :]
        E = np.clip(Ps - 0.5 * vgrid ** 2, 1e-300, None)
        lf = np.interp(np.log(E), logE, logf, left=-700.0, right=-700.0)
        pdf = vgrid ** 2 * np.exp(lf)
        cdfv = np.cumsum(pdf, axis=1)
        cdfv /= cdfv[:, -1:]
        uu = rng.random(len(Ps))
        idx = np.clip(np.sum(cdfv < uu[:, None], axis=1), 1, n_v - 1)
        rows = np.arange(len(Ps))
        c0, c1 = cdfv[rows, idx - 1], cdfv[rows, idx]
        w = np.where(c1 > c0, (uu - c0) / np.maximum(c1 - c0, 1e-300), 0.0)
        v[i0:i0 + chunk] = vgrid[rows, idx - 1] * (1.0 - w) + vgrid[rows, idx] * w
    return v


def equilibrium_shells(M200: float, c: float, H0: float, N: int, seed: int, r_decay_factor: float = 0.3,
                       refine_r_kpc: float = 0.0, refine_beta: float = 1.0, r_min_factor: float = 1e-6) -> dict:
    """Capas de la distribución isótropa de Eddington del NFW truncado.

    Sin refinamiento (refine_r_kpc = 0): las mismas capas que el N-cuerpos
    (halo_nbody.sample_equilibrium_nfw, misma semilla ⟹ mismas ICs),
    masas iguales m = M_tot/N.
    Con REFINAMIENTO DE MASA (refine_r_kpc > 0): las capas con r < r_ref
    llevan masa m(r) = m₀·(r/r_ref)^β y se muestrean más densamente en
    proporción (muestreo por importancia sobre la masa: n(r) ∝ ρr²/m(r)),
    de modo que el interior queda resuelto con muchas capas ligeras sin
    tocar la distribución de masa ni la de velocidades (v de f(E) en cada
    r, isótropa). N es el número de capas de masa m₀ que tendría el halo
    sin refinar (fija m₀ = M_tot/N); el número real de capas es mayor.
    Para el NFW de la ronda (1e11, c = 10) con r_ref = 2 kpc y β = 1,
    N = 1e6 da ≈ 10·M(<0.4)/m₀ ≈ 5e3 capas dentro de 0.4 kpc."""
    if refine_r_kpc <= 0.0:
        ic = sample_equilibrium_nfw(M200, c, H0, N, seed, r_decay_factor=r_decay_factor)
        r, vr, L2 = shells_from_3d(ic["pos"], ic["vel"])
        return {"r": r, "vr": vr, "L2": L2, "m": ic["mass"].copy(), "M_tot": ic["M_tot"], "params": ic.get("params"),
                "f_negative_fraction": ic.get("f_negative_fraction"), "refined": False}
    from cronos.halo_nbody import _eddington_tables, nfw_structural
    rng = np.random.default_rng(seed)
    p = nfw_structural(M200, c, H0)
    # tablas con borde interior r_min_factor·r_s (1e-6 ≈ 1e-5 kpc), muy por debajo de la capa más interna
    # muestreada (~1e-3 kpc): el pico numérico de f(E) en el último nodo queda fuera del rango muestreado
    T = _eddington_tables(p, r_decay_factor, n_E=1000, r_min_factor=r_min_factor, n_u=2000, edge_dense=True)
    r_t, M_t, Psi_t, E_t, f_t = T["r"], T["M"], T["Psi"], T["E"], T["f"]
    m0 = T["M_tot"] / N
    mass_of_r = lambda r: m0 * np.minimum(1.0, (r / refine_r_kpc) ** refine_beta)  # noqa: E731
    # número de capas por unidad de masa 1/m(r): CDF del conteo sobre la malla de las tablas
    dM = np.diff(M_t)
    r_mid = 0.5 * (r_t[1:] + r_t[:-1])
    dN = dM / mass_of_r(r_mid)
    N_cum = np.concatenate([[0.0], np.cumsum(dN)])
    n_shells = int(round(N_cum[-1]))
    u = (rng.random(n_shells) + np.arange(n_shells)) / n_shells               # estratificado
    r = np.interp(u * N_cum[-1], N_cum, r_t)
    m = mass_of_r(r)
    m *= T["M_tot"] / m.sum()                                                # masa total exacta
    Psi = np.interp(r, r_t, Psi_t)
    # velocidad: v² f(Ψ − v²/2) por CDF inversa sobre una malla de n_v velocidades por capa — el MISMO
    # método que sample_equilibrium_nfw (N-cuerpos). El rechazo con envolvente 2Ψ·f(Ψ) que usaba el piloto
    # tenía aceptación ~1e-7 en las capas más ligadas (f(E) sube ×100 en el último nodo de la tabla) y
    # dejaba el muestreo de N = 3e5 sin terminar en 15 min (piloto declarado)
    v = _speeds_inverse_cdf(Psi, E_t, f_t, rng)
    cos_t = 2.0 * rng.random(n_shells) - 1.0
    vr = v * cos_t
    L2 = r ** 2 * v ** 2 * (1.0 - cos_t ** 2)
    return {"r": r, "vr": vr, "L2": L2, "m": m, "M_tot": float(T["M_tot"]), "params": p,
            "f_negative_fraction": T["f_negative_fraction"], "refined": True, "m0": m0, "n_shells": n_shells,
            "refine_r_kpc": refine_r_kpc, "refine_beta": refine_beta}
