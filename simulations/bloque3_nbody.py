"""
Bloque 3 - Simulaciones N-body con Fricción Entrópica
======================================================

Implementa simulaciones de formación de estructura cósmica
con la fricción entrópica del modelo MCMC.

Elementos clave:
    - Fricción entrópica: η(ρ) = α × (ρ/ρc)^1.5
    - Perfil de Burkert: ρ(r) = ρ0 / [(1 + r/rc)(1 + (r/rc)²)]
    - Relación núcleo-masa: r_core(M) = 1.8 × (M/10^11 M☉)^0.35 kpc

El modelo MCMC predice que la fricción entrópica modifica la
dinámica de materia oscura, produciendo núcleos tipo Burkert
en lugar de cúspides tipo NFW.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    - Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
    - Burkert, A. (1995). ApJ, 447, L25
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


# =============================================================================
# Constantes
# =============================================================================

# Parámetro de fricción entrópica
ALPHA_FRICCION: float = 0.1

# Densidad crítica del universo (M☉/kpc³)
RHO_CRITICA: float = 1.4e11  # Aproximadamente 10^-26 kg/m³

# Constante gravitacional en unidades astrofísicas (kpc³/(M☉ Gyr²))
G_ASTRO: float = 4.498e-6

# Masa solar característica
M_CHAR: float = 1e11  # M☉


# =============================================================================
# Funciones de Fricción Entrópica
# =============================================================================

def friccion_entropica(
    rho: float,
    alpha: float = ALPHA_FRICCION,
    rho_c: float = RHO_CRITICA
) -> float:
    """
    Calcula el coeficiente de fricción entrópica.

    η(ρ) = α × (ρ/ρc)^1.5

    Esta fricción emerge de la entropía del campo tensional
    y actúa sobre la materia oscura, suavizando las cúspides
    centrales de los halos.

    Args:
        rho: Densidad local (M☉/kpc³)
        alpha: Parámetro de acoplamiento
        rho_c: Densidad crítica

    Returns:
        η: Coeficiente de fricción (1/Gyr)
    """
    if rho <= 0:
        return 0.0
    return alpha * (rho / rho_c) ** 1.5


def fuerza_friccion(
    velocidad: NDArray[np.float64],
    rho: float,
    alpha: float = ALPHA_FRICCION
) -> NDArray[np.float64]:
    """
    Calcula la fuerza de fricción entrópica.

    F_fric = -η(ρ) × v

    Args:
        velocidad: Vector velocidad (kpc/Gyr)
        rho: Densidad local (M☉/kpc³)
        alpha: Parámetro de fricción

    Returns:
        Vector fuerza de fricción
    """
    eta = friccion_entropica(rho, alpha)
    return -eta * velocidad


# =============================================================================
# Perfiles de Densidad
# =============================================================================

def perfil_burkert(
    r: float,
    rho_0: float,
    r_c: float
) -> float:
    """
    Perfil de densidad de Burkert para halos de materia oscura.

    ρ(r) = ρ0 / [(1 + r/rc)(1 + (r/rc)²)]

    Este perfil tiene un núcleo plano central (a diferencia de NFW)
    y es consistente con observaciones de galaxias enanas.

    Args:
        r: Radio (kpc)
        rho_0: Densidad central (M☉/kpc³)
        r_c: Radio del núcleo (kpc)

    Returns:
        ρ(r): Densidad en r (M☉/kpc³)
    """
    if r <= 0:
        return rho_0

    x = r / r_c
    return rho_0 / ((1 + x) * (1 + x**2))


def perfil_nfw(
    r: float,
    rho_s: float,
    r_s: float
) -> float:
    """
    Perfil de Navarro-Frenk-White (para comparación).

    ρ(r) = ρs / [(r/rs)(1 + r/rs)²]

    Tiene cúspide central (ρ ∝ 1/r) que no se observa.

    Args:
        r: Radio (kpc)
        rho_s: Densidad de escala
        r_s: Radio de escala

    Returns:
        ρ(r)
    """
    if r <= 0:
        return np.inf  # Cúspide

    x = r / r_s
    return rho_s / (x * (1 + x)**2)


def radio_core(M_halo: float) -> float:
    """
    Calcula el radio del núcleo según la relación masa-núcleo MCMC.

    r_core(M) = 1.8 × (M/10^11 M☉)^0.35 kpc

    Esta relación emerge de la fricción entrópica y explica
    la diversidad de perfiles de rotación en galaxias.

    Args:
        M_halo: Masa del halo (M☉)

    Returns:
        r_core en kpc
    """
    return 1.8 * (M_halo / M_CHAR) ** 0.35


def masa_encerrada_burkert(r: float, rho_0: float, r_c: float) -> float:
    """
    Masa encerrada dentro de radio r para perfil de Burkert.

    M(<r) = π ρ0 r_c³ × [ln(1+(r/rc)²) + 2ln(1+r/rc) - 2arctan(r/rc)]

    Args:
        r: Radio (kpc)
        rho_0: Densidad central
        r_c: Radio del núcleo

    Returns:
        M(<r) en M☉
    """
    x = r / r_c
    factor = np.pi * rho_0 * r_c**3
    return factor * (np.log(1 + x**2) + 2*np.log(1 + x) - 2*np.arctan(x))


def velocidad_circular(r: float, rho_0: float, r_c: float) -> float:
    """
    Velocidad circular para perfil de Burkert.

    v_c(r) = √(G × M(<r) / r)

    Args:
        r: Radio (kpc)
        rho_0: Densidad central
        r_c: Radio del núcleo

    Returns:
        v_c en km/s
    """
    if r <= 0:
        return 0.0

    M_enc = masa_encerrada_burkert(r, rho_0, r_c)
    # Convertir a km/s (factor de conversión kpc/Gyr → km/s ≈ 978)
    v_c_kpc_gyr = np.sqrt(G_ASTRO * M_enc / r)
    return v_c_kpc_gyr * 978


# =============================================================================
# Clases para Simulación N-body
# =============================================================================

@dataclass
class Particula:
    """
    Partícula en la simulación N-body.

    Attributes:
        posicion: Vector posición (kpc)
        velocidad: Vector velocidad (kpc/Gyr)
        masa: Masa de la partícula (M☉)
    """
    posicion: NDArray[np.float64]
    velocidad: NDArray[np.float64]
    masa: float = 1.0

    @classmethod
    def crear_aleatoria(
        cls,
        r_max: float,
        v_max: float,
        masa: float = 1.0
    ) -> Particula:
        """Crea partícula con posición y velocidad aleatorias."""
        pos = np.random.randn(3)
        pos = pos / np.linalg.norm(pos) * np.random.uniform(0, r_max)

        vel = np.random.randn(3)
        vel = vel / np.linalg.norm(vel) * np.random.uniform(0, v_max)

        return cls(posicion=pos, velocidad=vel, masa=masa)

    @property
    def r(self) -> float:
        """Radio (distancia al origen)."""
        return float(np.linalg.norm(self.posicion))

    @property
    def v(self) -> float:
        """Rapidez."""
        return float(np.linalg.norm(self.velocidad))


@dataclass
class SimulacionNBody:
    """
    Simulación N-body con fricción entrópica MCMC.

    Simula la dinámica de partículas bajo gravedad más
    la fricción entrópica del modelo MCMC.

    Attributes:
        particulas: Lista de partículas
        rho_0: Densidad central del halo (M☉/kpc³)
        r_c: Radio del núcleo (kpc)
        alpha: Parámetro de fricción
        historial: Lista de estados guardados
    """
    particulas: List[Particula] = field(default_factory=list)
    rho_0: float = 1e8  # M☉/kpc³
    r_c: float = 2.0    # kpc
    alpha: float = ALPHA_FRICCION
    dt: float = 0.001   # Gyr
    historial: List[Dict] = field(default_factory=list)

    @classmethod
    def crear_halo_inicial(
        cls,
        n_particulas: int,
        M_total: float,
        r_max: float = 50.0,
        alpha: float = ALPHA_FRICCION
    ) -> SimulacionNBody:
        """
        Crea simulación con distribución inicial de halo.

        Args:
            n_particulas: Número de partículas
            M_total: Masa total del halo (M☉)
            r_max: Radio máximo inicial (kpc)
            alpha: Parámetro de fricción

        Returns:
            Simulación inicializada
        """
        # Calcular parámetros del perfil
        r_c = radio_core(M_total)
        rho_0 = M_total / (4 * np.pi * r_c**3)  # Aproximación

        # Crear partículas
        masa_particula = M_total / n_particulas
        particulas = []

        for _ in range(n_particulas):
            # Posición según perfil de densidad (simplificado)
            r = np.random.uniform(0, r_max)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)

            pos = r * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])

            # Velocidad circular más dispersión
            v_circ = velocidad_circular(r, rho_0, r_c) / 978  # Convertir a kpc/Gyr
            v_disp = v_circ * 0.2 * np.random.randn()

            # Dirección tangencial
            if r > 0:
                r_hat = pos / r
                z_hat = np.array([0, 0, 1])
                phi_hat = np.cross(z_hat, r_hat)
                if np.linalg.norm(phi_hat) > 0:
                    phi_hat = phi_hat / np.linalg.norm(phi_hat)
                else:
                    phi_hat = np.array([1, 0, 0])
            else:
                phi_hat = np.array([1, 0, 0])

            vel = (v_circ + v_disp) * phi_hat + 0.1 * v_circ * np.random.randn(3)

            particulas.append(Particula(
                posicion=pos,
                velocidad=vel,
                masa=masa_particula
            ))

        return cls(
            particulas=particulas,
            rho_0=rho_0,
            r_c=r_c,
            alpha=alpha
        )

    @property
    def n_particulas(self) -> int:
        """Número de partículas."""
        return len(self.particulas)

    def densidad_local(self, r: float) -> float:
        """Calcula densidad local del halo en radio r."""
        return perfil_burkert(r, self.rho_0, self.r_c)

    def aceleracion_gravitatoria(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcula aceleración gravitatoria en una posición.

        Usa el potencial del halo de fondo.

        Args:
            pos: Vector posición

        Returns:
            Vector aceleración (kpc/Gyr²)
        """
        r = np.linalg.norm(pos)
        if r < 0.01:  # Regularización central
            return np.zeros(3)

        M_enc = masa_encerrada_burkert(r, self.rho_0, self.r_c)
        a_mag = G_ASTRO * M_enc / r**2

        return -a_mag * pos / r

    def paso_temporal(self) -> None:
        """Avanza la simulación un paso dt."""
        for p in self.particulas:
            r = p.r
            rho = self.densidad_local(r)

            # Aceleración gravitatoria
            a_grav = self.aceleracion_gravitatoria(p.posicion)

            # Fricción entrópica
            f_fric = fuerza_friccion(p.velocidad, rho, self.alpha)
            a_fric = f_fric / p.masa if p.masa > 0 else np.zeros(3)

            # Aceleración total
            a_total = a_grav + a_fric

            # Integración leapfrog (simplificada)
            p.velocidad = p.velocidad + a_total * self.dt
            p.posicion = p.posicion + p.velocidad * self.dt

    def simular(
        self,
        t_total: float,
        guardar_cada: int = 100
    ) -> None:
        """
        Ejecuta la simulación.

        Args:
            t_total: Tiempo total (Gyr)
            guardar_cada: Guardar estado cada N pasos
        """
        n_pasos = int(t_total / self.dt)

        for paso in range(n_pasos):
            self.paso_temporal()

            if paso % guardar_cada == 0:
                self._guardar_estado(paso * self.dt)

                if paso % (guardar_cada * 10) == 0:
                    print(f"Paso {paso}/{n_pasos}, t = {paso*self.dt:.3f} Gyr")

    def _guardar_estado(self, t: float) -> None:
        """Guarda el estado actual."""
        radios = [p.r for p in self.particulas]
        velocidades = [p.v for p in self.particulas]

        self.historial.append({
            "t": t,
            "r_medio": np.mean(radios),
            "r_std": np.std(radios),
            "v_medio": np.mean(velocidades),
            "v_std": np.std(velocidades),
        })

    def perfil_densidad_simulado(self, n_bins: int = 20) -> Tuple[NDArray, NDArray]:
        """
        Calcula perfil de densidad de las partículas.

        Args:
            n_bins: Número de bins radiales

        Returns:
            (radios, densidades)
        """
        radios = np.array([p.r for p in self.particulas])
        r_max = np.max(radios)

        bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = (bins[:-1] + bins[1:]) / 2

        densidades = []
        masa_total = sum(p.masa for p in self.particulas)

        for i in range(n_bins):
            r_in, r_out = bins[i], bins[i+1]
            vol = 4/3 * np.pi * (r_out**3 - r_in**3)

            masa_bin = sum(
                p.masa for p in self.particulas
                if r_in <= p.r < r_out
            )

            densidades.append(masa_bin / vol if vol > 0 else 0)

        return r_centers, np.array(densidades)

    def curva_rotacion(self, n_puntos: int = 20) -> Tuple[NDArray, NDArray]:
        """
        Calcula curva de rotación.

        Args:
            n_puntos: Número de puntos

        Returns:
            (radios, velocidades_circulares en km/s)
        """
        r_max = max(p.r for p in self.particulas)
        radios = np.linspace(0.1, r_max, n_puntos)
        v_circ = [velocidad_circular(r, self.rho_0, self.r_c) for r in radios]

        return radios, np.array(v_circ)


# =============================================================================
# Tests
# =============================================================================

def _test_nbody():
    """Verifica la implementación N-body."""

    # Test 1: Fricción entrópica positiva
    eta = friccion_entropica(RHO_CRITICA)
    assert eta >= 0, f"Fricción debe ser >= 0"
    assert eta == ALPHA_FRICCION, f"η(ρc) debe ser α"

    # Test 2: Perfil Burkert finito en r=0
    rho_0 = 1e8
    r_c = 2.0
    rho_center = perfil_burkert(0, rho_0, r_c)
    assert np.isfinite(rho_center), "Burkert finito en centro"
    assert rho_center == rho_0, "ρ(0) = ρ0"

    # Test 3: Radio core crece con masa
    r_c_1 = radio_core(1e10)
    r_c_2 = radio_core(1e12)
    assert r_c_2 > r_c_1, "r_core debe crecer con masa"

    # Test 4: Velocidad circular finita
    v_c = velocidad_circular(1.0, rho_0, r_c)
    assert np.isfinite(v_c) and v_c > 0, "v_c debe ser finita y positiva"

    # Test 5: Simulación corre sin errores
    sim = SimulacionNBody.crear_halo_inicial(
        n_particulas=10,
        M_total=1e11,
        alpha=ALPHA_FRICCION
    )
    sim.simular(t_total=0.01, guardar_cada=5)
    assert len(sim.historial) > 0, "Debe guardar historial"

    print("✓ Todos los tests del Bloque 3 pasaron")
    return True


if __name__ == "__main__":
    _test_nbody()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Simulación N-body MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    # Crear simulación pequeña
    print("Creando halo con 100 partículas...")
    sim = SimulacionNBody.crear_halo_inicial(
        n_particulas=100,
        M_total=1e11,  # 10^11 M☉
        alpha=ALPHA_FRICCION
    )

    print(f"  Partículas: {sim.n_particulas}")
    print(f"  ρ0 = {sim.rho_0:.2e} M☉/kpc³")
    print(f"  r_c = {sim.r_c:.2f} kpc")
    print(f"  α = {sim.alpha}")

    # Curva de rotación teórica
    print("\nCurva de rotación (teórica):")
    radios, v_circ = sim.curva_rotacion()
    for r, v in zip(radios[::4], v_circ[::4]):
        print(f"  r = {r:.1f} kpc: v_c = {v:.1f} km/s")

    # Simular brevemente
    print("\nSimulando 0.1 Gyr...")
    sim.simular(t_total=0.1, guardar_cada=50)

    print(f"\nEstados guardados: {len(sim.historial)}")
    if sim.historial:
        ultimo = sim.historial[-1]
        print(f"Estado final:")
        print(f"  r_medio = {ultimo['r_medio']:.2f} kpc")
        print(f"  v_medio = {ultimo['v_medio']*978:.1f} km/s")
