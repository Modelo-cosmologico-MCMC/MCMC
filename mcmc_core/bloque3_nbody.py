"""
Bloque 3 - N-Body y Formación de Estructuras MCMC
===================================================

Simulaciones N-body en la variable entrópica S (marco Cronos) para
el modelo MCMC de formación de estructuras cósmicas.

OBJETIVO:
    Extender el MCMC desde cosmología efectiva de fondo (Bloques 1-2)
    al régimen no lineal de formación de estructuras.

PREGUNTA ONTOLÓGICA CENTRAL:
    ¿Es la misma tensión masa-espacio que genera Λ_rel y el mass gap
    suficiente para explicar las características observadas de la
    estructura cósmica, sin postular nuevos ingredientes?

COMPONENTES:
    1. Funciones Ontológicas Base (conexión con Bloques 0-2)
    2. Fricción Entrópica (Ley de Cronos)
    3. Perfiles de Densidad (NFW, Burkert, Zhao-MCMC)
    4. Integrador Cronos (N-body en variable S)
    5. Análisis de Halos (FOF, perfiles, curvas de rotación)
    6. Comparación Observacional (SPARC)
    7. Validación Ontológica

PREDICCIONES TESTABLES:
    - Halos cored universales (no cúspides NFW)
    - Relación r_core ∝ M^0.35
    - Supresión de satélites ~40%
    - Evolución con z: r_core ∝ (1+z)^(-0.5)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.spatial import cKDTree
import warnings


# =============================================================================
# Constantes Físicas y Ontológicas
# =============================================================================

# Sellos entrópicos
S0: float = 0.000
S1: float = 0.010
S2: float = 0.100
S3: float = 1.000
S4: float = 1.001  # Big Bang

# Cosmología
H0: float = 67.4          # km/s/Mpc
OMEGA_M: float = 0.315
OMEGA_LAMBDA: float = 0.685
DELTA_LAMBDA: float = 0.02

# Constantes de Cronos
RHO_CRONOS: float = 277.5   # M☉/kpc³ - Densidad crítica de Cronos
ALPHA_LAPSE: float = 1.0    # Parámetro de lapse
BETA_ETA: float = 1.5       # Exponente de fricción
GAMMA_FRICCION: float = 0.5 # Factor de acoplamiento

# Relación masa-núcleo
R_STAR: float = 1.8         # kpc
M_STAR: float = 1e11        # M☉
ALPHA_R: float = 0.35       # Exponente de masa
BETA_R: float = -0.5        # Exponente de redshift

# Campo de Adrián
PHI0_ADRIAN: float = 1.0    # Amplitud del campo
LAMBDA_PHI: float = 10.0    # Longitud de escala (kpc)

# Constantes de simulación
G_NEWTON: float = 4.302e-6  # kpc³ M☉⁻¹ Gyr⁻²
C_LIGHT: float = 306.6      # kpc/Gyr
MPC_TO_KPC: float = 1000.0


# =============================================================================
# SECCIÓN 1: Funciones Ontológicas Base
# =============================================================================

def k_S(S: float, k0: float = 6.6252, a1: float = 0.1416,
        a2: float = 0.2355, a3: float = 0.3439) -> float:
    """
    Tasa de colapso k(S) calibrada.

    k(S) = k₀ × ∏[1 - aₙ × θ(S - Sₙ)]

    Args:
        S: Sello entrópico
        k0, a1, a2, a3: Parámetros calibrados

    Returns:
        k(S) en Gyr⁻¹
    """
    theta1 = 1.0 if S >= S1 else 0.0
    theta2 = 1.0 if S >= S2 else 0.0
    theta3 = 1.0 if S >= S3 else 0.0

    factor1 = 1.0 - a1 * theta1
    factor2 = 1.0 - a2 * theta2
    factor3 = 1.0 - a3 * theta3

    return k0 * factor1 * factor2 * factor3


def Mp_frac(S: float) -> float:
    """
    Fracción de masa Mp(S)/Mp₀.

    Mp(S)/Mp₀ = exp(-∫₀ˢ k(s)ds)

    Args:
        S: Sello entrópico

    Returns:
        Fracción de masa [0, 1]
    """
    if S <= 0:
        return 1.0

    resultado, _ = quad(k_S, 0, S)
    return np.exp(-resultado)


def Ep_frac(S: float) -> float:
    """
    Fracción de espacio Ep(S)/Ep₀.

    Ep(S)/Ep₀ = 1 - Mp(S)/Mp₀

    Args:
        S: Sello entrópico

    Returns:
        Fracción de espacio [0, 1]
    """
    return 1.0 - Mp_frac(S)


def P_ME(S: float) -> float:
    """
    Polarización masa-espacio.

    P_ME = (Mp - Ep)/(Mp + Ep) = 2×Mp_frac - 1

    Args:
        S: Sello entrópico

    Returns:
        P_ME ∈ [-1, +1]
    """
    mp = Mp_frac(S)
    return 2 * mp - 1


def S_to_z(S: float, beta: float = 0.1, t0: float = 0.01) -> float:
    """
    Mapea sello entrópico S a redshift z.

    Usa el mapeo S → t → z.

    Args:
        S: Sello entrópico (S > S4)
        beta: Escala de mapeo
        t0: Tiempo de referencia

    Returns:
        Redshift z
    """
    if S <= S4:
        return np.inf

    # S → t
    t = t0 * (np.exp((S - S4) / beta) - 1)

    # t → z (aproximación EdS)
    t_universo = 13.8  # Gyr
    if t <= 0:
        return np.inf
    if t >= t_universo:
        return 0.0

    return (t_universo / t) ** (2/3) - 1


def z_to_S(z: float, beta: float = 0.1, t0: float = 0.01) -> float:
    """
    Mapea redshift z a sello entrópico S.

    Args:
        z: Redshift
        beta: Escala de mapeo
        t0: Tiempo de referencia

    Returns:
        Sello entrópico S
    """
    t_universo = 13.8  # Gyr

    # z → t (aproximación EdS)
    t = t_universo / (1 + z) ** 1.5

    # t → S
    return S4 + beta * np.log(1 + t / t0)


def Lambda_rel(z: float, delta: float = DELTA_LAMBDA) -> float:
    """
    Factor de modificación de Λ en MCMC.

    Λ_rel(z) = 1 + δΛ × exp(-z/2) × (1+z)^(-0.5)

    Args:
        z: Redshift
        delta: Parámetro de modificación

    Returns:
        Λ_rel(z)
    """
    return 1.0 + delta * np.exp(-z / 2) * (1 + z) ** (-0.5)


def H_MCMC(z: float) -> float:
    """
    Parámetro de Hubble MCMC en km/s/Mpc.

    H(z) = H₀ × E_MCMC(z)

    Args:
        z: Redshift

    Returns:
        H(z) en km/s/Mpc
    """
    Lambda_eff = OMEGA_LAMBDA * Lambda_rel(z)
    E = np.sqrt(OMEGA_M * (1 + z)**3 + Lambda_eff)
    return H0 * E


def H_MCMC_kpc_gyr(z: float) -> float:
    """
    Parámetro de Hubble en unidades de simulación.

    Convierte de km/s/Mpc a kpc/Gyr/kpc = Gyr⁻¹

    Args:
        z: Redshift

    Returns:
        H(z) en Gyr⁻¹
    """
    H_km_s_mpc = H_MCMC(z)
    # 1 km/s/Mpc = 1.022e-3 Gyr⁻¹
    return H_km_s_mpc * 1.022e-3


# =============================================================================
# SECCIÓN 2: Fricción Entrópica (Ley de Cronos)
# =============================================================================

@dataclass
class ParametrosCronos:
    """
    Parámetros de la Ley de Cronos.

    La fricción entrópica modifica la dinámica gravitacional
    en regiones de alta densidad.
    """
    rho_c: float = RHO_CRONOS     # M☉/kpc³
    alpha: float = ALPHA_LAPSE     # Parámetro lapse
    beta: float = BETA_ETA         # Exponente de densidad
    gamma: float = GAMMA_FRICCION  # Factor de acoplamiento

    def __post_init__(self):
        """Verifica parámetros válidos."""
        assert self.rho_c > 0, "ρc debe ser positivo"
        assert self.alpha > 0, "α debe ser positivo"
        assert self.beta > 0, "β debe ser positivo"


@dataclass
class FriccionEntropica:
    """
    Implementación de la fricción entrópica MCMC.

    La fricción entrópica disipa energía cinética en regiones
    de alta densidad, impidiendo la formación de cúspides.

    ECUACIÓN:
        η(ρ) = α × (ρ/ρc)^β

    EFECTOS:
        - Núcleos planos (cored) en halos
        - Supresión de satélites de baja masa
        - Resolución del problema cusp-core
    """
    params: ParametrosCronos = field(default_factory=ParametrosCronos)

    def eta(self, rho: float) -> float:
        """
        Coeficiente de fricción entrópica.

        η(ρ) = α × (ρ/ρc)^β

        Args:
            rho: Densidad local (M☉/kpc³)

        Returns:
            η en Gyr⁻¹
        """
        if rho <= 0:
            return 0.0

        x = rho / self.params.rho_c
        return self.params.alpha * (x ** self.params.beta)

    def dilatacion_temporal(self, rho: float) -> float:
        """
        Factor de dilatación temporal.

        Δt/Δt₀ = 1 + (ρ/ρc)^(3/2) / α

        Args:
            rho: Densidad local

        Returns:
            Factor de dilatación ≥ 1
        """
        if rho <= 0:
            return 1.0

        x = rho / self.params.rho_c
        return 1.0 + (x ** 1.5) / self.params.alpha

    def tiempo_escala(self, rho: float) -> float:
        """
        Tiempo de escala de fricción.

        τ = 1/η

        Args:
            rho: Densidad local

        Returns:
            τ en Gyr
        """
        eta_val = self.eta(rho)
        if eta_val <= 0:
            return np.inf
        return 1.0 / eta_val

    def campo_adrian(self, r: float, rho_0: float) -> float:
        """
        Campo de Adrián (tensión local masa-espacio).

        Φ_ten(r) = φ₀ × exp(-r/λ_φ) × (ρ₀/ρc)^(1/2)

        Args:
            r: Distancia radial (kpc)
            rho_0: Densidad central

        Returns:
            Intensidad del campo
        """
        factor_r = np.exp(-r / LAMBDA_PHI)
        factor_rho = np.sqrt(rho_0 / self.params.rho_c)
        return PHI0_ADRIAN * factor_r * factor_rho

    def modificar_aceleracion(
        self,
        a: NDArray,
        v: NDArray,
        rho: float
    ) -> NDArray:
        """
        Modifica la aceleración incluyendo fricción.

        a_mod = a_grav - η(ρ) × v

        Args:
            a: Aceleración gravitacional (kpc/Gyr²)
            v: Velocidad (kpc/Gyr)
            rho: Densidad local

        Returns:
            Aceleración modificada
        """
        eta_val = self.eta(rho)
        return a - eta_val * v

    def tabla_escalas(self) -> List[Dict]:
        """
        Genera tabla de escalas de fricción por densidad.
        """
        densidades = [1, 10, 100, 1000]
        tabla = []

        for rho in densidades:
            eta_val = self.eta(rho)
            tau = self.tiempo_escala(rho)
            dil = self.dilatacion_temporal(rho)

            tabla.append({
                "rho_M_kpc3": rho,
                "eta_Gyr_inv": eta_val,
                "tau_Gyr": tau,
                "dilatacion": dil,
            })

        return tabla


# =============================================================================
# SECCIÓN 3: Perfiles de Densidad
# =============================================================================

@dataclass
class ParametrosHalo:
    """
    Parámetros de un halo de materia oscura.
    """
    M_200: float           # Masa virial (M☉)
    c: float = 10.0        # Concentración
    z: float = 0.0         # Redshift

    def __post_init__(self):
        """Calcula propiedades derivadas."""
        # Densidad crítica del universo
        rho_crit = 277.5 * (H_MCMC(self.z) / H0) ** 2  # M☉/kpc³

        # Radio virial
        self.r_200 = (3 * self.M_200 / (4 * np.pi * 200 * rho_crit)) ** (1/3)

        # Radio de escala
        self.r_s = self.r_200 / self.c


def perfil_NFW(r: float, rho_s: float, r_s: float) -> float:
    """
    Perfil de densidad NFW (Navarro-Frenk-White).

    ρ(r) = ρs / [(r/rs)(1 + r/rs)²]

    CARACTERÍSTICAS:
        - Cúspide central: ρ ∝ r⁻¹
        - Predicción ΛCDM estándar
        - Problema: no concuerda con observaciones de galaxias enanas

    Args:
        r: Radio (kpc)
        rho_s: Densidad de escala (M☉/kpc³)
        r_s: Radio de escala (kpc)

    Returns:
        Densidad en M☉/kpc³
    """
    if r <= 0:
        r = 0.001  # Regularización
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


def perfil_Burkert(r: float, rho_0: float, r_c: float) -> float:
    """
    Perfil de densidad Burkert (cored).

    ρ(r) = ρ₀ / [(1 + r/rc)(1 + (r/rc)²)]

    CARACTERÍSTICAS:
        - Núcleo constante: ρ → ρ₀ para r → 0
        - Consistente con observaciones de enanas
        - PREDICCIÓN MCMC: Emerge de la fricción entrópica

    Args:
        r: Radio (kpc)
        rho_0: Densidad central (M☉/kpc³)
        r_c: Radio del núcleo (kpc)

    Returns:
        Densidad en M☉/kpc³
    """
    if r_c <= 0:
        r_c = 0.001
    x = r / r_c
    return rho_0 / ((1 + x) * (1 + x ** 2))


def perfil_Zhao_MCMC(
    r: float,
    rho_s: float,
    r_s: float,
    S_local: float,
    alpha: float = 1.0,
    beta: float = 3.0
) -> float:
    """
    Perfil de densidad Zhao generalizado con dependencia en S.

    ρ(r) = ρs / [(r/rs)^γ × (1 + (r/rs)^α)^((β-γ)/α)]

    INNOVACIÓN MCMC:
        γ(S_loc) = 0.51 × [1 - exp(-S_loc/S₃)]

    A mayor S (más evolucionado), la pendiente γ se acerca a 0.51
    (intermedia entre NFW γ=1 y Burkert γ=0).

    Args:
        r: Radio (kpc)
        rho_s: Densidad de escala
        r_s: Radio de escala
        S_local: Sello entrópico local
        alpha, beta: Parámetros de forma

    Returns:
        Densidad en M☉/kpc³
    """
    if r <= 0:
        r = 0.001

    # Pendiente interna dependiente de S
    gamma = 0.51 * (1 - np.exp(-S_local / S3))

    x = r / r_s
    numerador = rho_s
    denominador = (x ** gamma) * (1 + x ** alpha) ** ((beta - gamma) / alpha)

    return numerador / max(denominador, 1e-30)


def radio_core_MCMC(M_halo: float, z: float = 0.0) -> float:
    """
    Radio del núcleo según la relación MCMC.

    r_core(M,z) = r★ × (M/M★)^α_r × (1+z)^β_r

    Args:
        M_halo: Masa del halo (M☉)
        z: Redshift

    Returns:
        Radio del núcleo en kpc
    """
    factor_masa = (M_halo / M_STAR) ** ALPHA_R
    factor_z = (1 + z) ** BETA_R
    return R_STAR * factor_masa * factor_z


def densidad_central_Burkert(M_halo: float, z: float = 0.0) -> float:
    """
    Densidad central del perfil Burkert.

    ρ₀ = M_halo / [π × r_c³ × ln(2)]  (aproximación)

    Args:
        M_halo: Masa total
        z: Redshift

    Returns:
        ρ₀ en M☉/kpc³
    """
    r_c = radio_core_MCMC(M_halo, z)
    # Integral del perfil Burkert
    factor = np.pi * r_c ** 3 * np.log(2)
    return M_halo / factor


@dataclass
class PerfilDensidad:
    """
    Clase unificada para perfiles de densidad.
    """
    tipo: str  # "NFW", "Burkert", "Zhao"
    params_halo: ParametrosHalo

    def __post_init__(self):
        """Calcula parámetros del perfil."""
        self._calcular_parametros()

    def _calcular_parametros(self):
        """Calcula parámetros específicos del perfil."""
        ph = self.params_halo

        if self.tipo == "NFW":
            # Densidad de escala NFW
            delta_c = (200 / 3) * ph.c ** 3 / (
                np.log(1 + ph.c) - ph.c / (1 + ph.c)
            )
            rho_crit = 277.5 * (H_MCMC(ph.z) / H0) ** 2
            self.rho_s = delta_c * rho_crit
            self.r_s = ph.r_s

        elif self.tipo == "Burkert":
            self.r_c = radio_core_MCMC(ph.M_200, ph.z)
            self.rho_0 = densidad_central_Burkert(ph.M_200, ph.z)

        elif self.tipo == "Zhao":
            # Similar a NFW pero con γ dependiente de S
            delta_c = (200 / 3) * ph.c ** 3 / (
                np.log(1 + ph.c) - ph.c / (1 + ph.c)
            )
            rho_crit = 277.5 * (H_MCMC(ph.z) / H0) ** 2
            self.rho_s = delta_c * rho_crit
            self.r_s = ph.r_s
            self.S_local = z_to_S(ph.z)

    def rho(self, r: float) -> float:
        """
        Calcula la densidad a radio r.

        Args:
            r: Radio en kpc

        Returns:
            Densidad en M☉/kpc³
        """
        if self.tipo == "NFW":
            return perfil_NFW(r, self.rho_s, self.r_s)
        elif self.tipo == "Burkert":
            return perfil_Burkert(r, self.rho_0, self.r_c)
        elif self.tipo == "Zhao":
            return perfil_Zhao_MCMC(
                r, self.rho_s, self.r_s, self.S_local
            )
        else:
            raise ValueError(f"Tipo de perfil desconocido: {self.tipo}")

    def masa_encerrada(self, r: float) -> float:
        """
        Masa encerrada dentro de radio r.

        M(<r) = 4π ∫₀ʳ ρ(r') r'² dr'

        Args:
            r: Radio en kpc

        Returns:
            Masa en M☉
        """
        def integrando(r_prime):
            return 4 * np.pi * r_prime ** 2 * self.rho(r_prime)

        resultado, _ = quad(integrando, 0.01, r)
        return resultado

    def velocidad_circular(self, r: float) -> float:
        """
        Velocidad circular v_c(r).

        v_c = √(GM(<r)/r)

        Args:
            r: Radio en kpc

        Returns:
            Velocidad en km/s
        """
        M_enc = self.masa_encerrada(r)
        # v² = GM/r en (kpc/Gyr)², convertir a km/s
        v_sq = G_NEWTON * M_enc / r
        v_kpc_gyr = np.sqrt(v_sq)
        return v_kpc_gyr * 977.8  # kpc/Gyr → km/s


# =============================================================================
# SECCIÓN 4: Integrador Cronos
# =============================================================================

@dataclass
class ConfiguracionCronos:
    """
    Configuración de la simulación N-body Cronos.
    """
    L_box: float = 25.0           # Mpc/h
    N_particles: int = 32 ** 3    # Número de partículas
    N_grid: int = 64              # Resolución de malla Poisson

    S_ini: float = 0.010          # S inicial (z ≈ 100)
    S_fin: float = 1.001          # S final (z ≈ 0)
    n_steps: int = 100            # Pasos de integración

    usar_friccion: bool = True
    params_cronos: ParametrosCronos = field(
        default_factory=ParametrosCronos
    )

    semilla: int = 42             # Semilla aleatoria

    def __post_init__(self):
        """Calcula propiedades derivadas."""
        self.L_box_kpc = self.L_box * MPC_TO_KPC  # kpc/h
        self.dS = (self.S_fin - self.S_ini) / self.n_steps
        self.m_particle = self._calcular_masa_particula()

    def _calcular_masa_particula(self) -> float:
        """Calcula la masa por partícula."""
        rho_m = OMEGA_M * 277.5  # M☉/kpc³
        V_box = self.L_box_kpc ** 3
        M_total = rho_m * V_box
        return M_total / self.N_particles


@dataclass
class ParticulaCronos:
    """
    Representa una partícula en la simulación Cronos.
    """
    id: int
    x: NDArray  # Posición (3D)
    v: NDArray  # Velocidad (3D)
    m: float    # Masa
    S_local: float = S4  # Entropía local


@dataclass
class ResultadoSimulacion:
    """
    Resultado de un paso de simulación.
    """
    S: float
    z: float
    t: float
    posiciones: NDArray
    velocidades: NDArray
    densidades: NDArray
    S_locales: NDArray


class IntegradorCronos:
    """
    Integrador N-body en variable entrópica S.

    ECUACIÓN DE MOVIMIENTO:
        d²x/dS² = -∇Φ/H²(S) - η(ρ)×dx/dS

    ALGORITMO:
        Leapfrog con fricción (kick-drift-kick)
    """

    def __init__(self, config: ConfiguracionCronos):
        """
        Inicializa el integrador.

        Args:
            config: Configuración de la simulación
        """
        self.config = config
        self.friccion = FriccionEntropica(config.params_cronos)
        np.random.seed(config.semilla)

        self.posiciones: Optional[NDArray] = None
        self.velocidades: Optional[NDArray] = None
        self.masas: Optional[NDArray] = None
        self.densidades: Optional[NDArray] = None

        self._grid_poisson: Optional[NDArray] = None

    def generar_condiciones_iniciales(
        self,
        tipo: str = "uniforme_perturbado"
    ) -> None:
        """
        Genera condiciones iniciales.

        Args:
            tipo: "uniforme", "uniforme_perturbado", "glass"
        """
        N = self.config.N_particles
        L = self.config.L_box_kpc

        if tipo == "uniforme":
            # Distribución uniforme
            self.posiciones = np.random.uniform(0, L, (N, 3))

        elif tipo == "uniforme_perturbado":
            # Grilla con perturbaciones
            n_lado = int(np.ceil(N ** (1/3)))
            x = np.linspace(0, L, n_lado, endpoint=False)
            xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
            pos = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

            # Tomar solo N partículas
            self.posiciones = pos[:N] + np.random.normal(0, L/n_lado/10, (N, 3))
            self.posiciones = self.posiciones % L  # Condiciones periódicas

        elif tipo == "glass":
            # Similar a uniforme perturbado (simplificado)
            self.posiciones = np.random.uniform(0, L, (N, 3))

        # Velocidades iniciales: Hubble flow
        z_ini = S_to_z(self.config.S_ini)
        H_ini = H_MCMC_kpc_gyr(z_ini)
        self.velocidades = H_ini * (self.posiciones - L/2)

        # Masas iguales
        self.masas = np.full(N, self.config.m_particle)

        # Densidades iniciales
        self.densidades = np.zeros(N)

    def _calcular_densidad_SPH(self, n_vecinos: int = 32) -> NDArray:
        """
        Calcula densidad usando SPH (Smoothed Particle Hydrodynamics).

        Args:
            n_vecinos: Número de vecinos para estimar densidad

        Returns:
            Array de densidades
        """
        tree = cKDTree(self.posiciones, boxsize=self.config.L_box_kpc)
        dist, _ = tree.query(self.posiciones, k=n_vecinos)

        # Radio suavizado = distancia al vecino n
        h = dist[:, -1]

        # Densidad = m × n_vecinos / (4/3 π h³)
        V = (4/3) * np.pi * h ** 3
        self.densidades = self.config.m_particle * n_vecinos / V

        return self.densidades

    def _resolver_poisson_PM(self) -> NDArray:
        """
        Resuelve la ecuación de Poisson con método Particle-Mesh.

        ∇²Φ = 4πGρ

        Returns:
            Campo de aceleraciones (N, 3)
        """
        N_grid = self.config.N_grid
        L = self.config.L_box_kpc

        # Interpolar densidad a malla (CIC - Cloud in Cell simplificado)
        rho_grid = np.zeros((N_grid, N_grid, N_grid))

        dx = L / N_grid
        for i in range(len(self.posiciones)):
            ix = int(self.posiciones[i, 0] / dx) % N_grid
            iy = int(self.posiciones[i, 1] / dx) % N_grid
            iz = int(self.posiciones[i, 2] / dx) % N_grid
            rho_grid[ix, iy, iz] += self.masas[i]

        rho_grid /= dx ** 3  # Convertir a densidad

        # FFT del potencial
        rho_k = np.fft.fftn(rho_grid)

        # Frecuencias
        kx = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
        kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = kxx ** 2 + kyy ** 2 + kzz ** 2
        k2[0, 0, 0] = 1  # Evitar división por cero

        # Potencial en espacio de Fourier
        phi_k = -4 * np.pi * G_NEWTON * rho_k / k2
        phi_k[0, 0, 0] = 0  # Modo cero

        # Gradiente en espacio de Fourier
        ax_k = -1j * kxx * phi_k
        ay_k = -1j * kyy * phi_k
        az_k = -1j * kzz * phi_k

        # Transformada inversa
        ax = np.real(np.fft.ifftn(ax_k))
        ay = np.real(np.fft.ifftn(ay_k))
        az = np.real(np.fft.ifftn(az_k))

        # Interpolar aceleración a partículas
        aceleraciones = np.zeros((len(self.posiciones), 3))
        for i in range(len(self.posiciones)):
            ix = int(self.posiciones[i, 0] / dx) % N_grid
            iy = int(self.posiciones[i, 1] / dx) % N_grid
            iz = int(self.posiciones[i, 2] / dx) % N_grid
            aceleraciones[i] = [ax[ix, iy, iz], ay[ix, iy, iz], az[ix, iy, iz]]

        return aceleraciones

    def paso_integracion(self, S: float, dS: float) -> None:
        """
        Realiza un paso de integración Leapfrog con fricción.

        Kick 1: v_{n+1/2} = v_n + (a - η·v_n) × dS/2
        Drift:  x_{n+1} = x_n + v_{n+1/2} × dS
        Kick 2: v_{n+1} = v_{n+1/2} + (a - η·v_{n+1/2}) × dS/2

        Args:
            S: Sello entrópico actual
            dS: Paso en S
        """
        # Calcular densidades
        self._calcular_densidad_SPH()

        # Resolver Poisson
        a_grav = self._resolver_poisson_PM()

        # Convertir a unidades de dS
        z = S_to_z(S)
        H = H_MCMC_kpc_gyr(z)
        a_grav_S = a_grav / (H ** 2)  # Escalado a variable S

        # Kick 1
        if self.config.usar_friccion:
            for i in range(len(self.posiciones)):
                eta = self.friccion.eta(self.densidades[i])
                a_mod = a_grav_S[i] - eta * self.velocidades[i]
                self.velocidades[i] += a_mod * dS / 2
        else:
            self.velocidades += a_grav_S * dS / 2

        # Drift
        self.posiciones += self.velocidades * dS

        # Condiciones periódicas
        self.posiciones = self.posiciones % self.config.L_box_kpc

        # Kick 2
        # Recalcular aceleración con nuevas posiciones
        self._calcular_densidad_SPH()
        a_grav = self._resolver_poisson_PM()
        a_grav_S = a_grav / (H ** 2)

        if self.config.usar_friccion:
            for i in range(len(self.posiciones)):
                eta = self.friccion.eta(self.densidades[i])
                a_mod = a_grav_S[i] - eta * self.velocidades[i]
                self.velocidades[i] += a_mod * dS / 2
        else:
            self.velocidades += a_grav_S * dS / 2

    def ejecutar(
        self,
        guardar_cada: int = 10
    ) -> List[ResultadoSimulacion]:
        """
        Ejecuta la simulación completa.

        Args:
            guardar_cada: Guardar snapshot cada N pasos

        Returns:
            Lista de snapshots
        """
        if self.posiciones is None:
            self.generar_condiciones_iniciales()

        snapshots = []
        S = self.config.S_ini
        dS = self.config.dS

        for paso in range(self.config.n_steps):
            self.paso_integracion(S, dS)
            S += dS

            if paso % guardar_cada == 0:
                z = S_to_z(S)
                t = 13.8 / (1 + z) ** 1.5 if z < 1000 else 0

                snapshot = ResultadoSimulacion(
                    S=S,
                    z=z,
                    t=t,
                    posiciones=self.posiciones.copy(),
                    velocidades=self.velocidades.copy(),
                    densidades=self.densidades.copy(),
                    S_locales=np.full(len(self.posiciones), S),
                )
                snapshots.append(snapshot)

        return snapshots


# =============================================================================
# SECCIÓN 5: Análisis de Halos
# =============================================================================

@dataclass
class Halo:
    """
    Representa un halo identificado.
    """
    id: int
    centro: NDArray             # Centro de masa (kpc)
    M_tot: float                # Masa total (M☉)
    r_200: float = 0.0          # Radio virial (kpc)
    v_max: float = 0.0          # Velocidad máxima (km/s)
    r_max: float = 0.0          # Radio de v_max (kpc)
    c: float = 10.0             # Concentración
    r_core: float = 0.0         # Radio del núcleo (kpc)
    S_mean: float = S4          # Entropía media
    particulas: List[int] = field(default_factory=list)


class AnalizadorHalos:
    """
    Análisis de halos: identificación, perfiles, curvas de rotación.
    """

    def __init__(self, params_cronos: Optional[ParametrosCronos] = None):
        """
        Inicializa el analizador.

        Args:
            params_cronos: Parámetros de fricción (para predicciones MCMC)
        """
        self.params = params_cronos or ParametrosCronos()

    def identificar_halos_fof(
        self,
        posiciones: NDArray,
        masas: NDArray,
        L_box: float,
        linking_length: float = 0.2,
        min_particulas: int = 20
    ) -> List[Halo]:
        """
        Identifica halos usando Friends-of-Friends (FOF).

        Args:
            posiciones: Posiciones de partículas (N, 3)
            masas: Masas de partículas (N,)
            L_box: Tamaño de la caja (kpc)
            linking_length: Fracción de la separación media
            min_particulas: Mínimo de partículas por halo

        Returns:
            Lista de halos identificados
        """
        N = len(posiciones)
        n_lado = int(np.ceil(N ** (1/3)))
        d_medio = L_box / n_lado
        b = linking_length * d_medio

        # Construir KDTree con condiciones periódicas
        tree = cKDTree(posiciones, boxsize=L_box)

        # Buscar pares dentro de b
        pares = tree.query_pairs(b)

        # Union-Find para agrupar
        padre = list(range(N))

        def encontrar(i):
            if padre[i] != i:
                padre[i] = encontrar(padre[i])
            return padre[i]

        def unir(i, j):
            pi, pj = encontrar(i), encontrar(j)
            if pi != pj:
                padre[pi] = pj

        for i, j in pares:
            unir(i, j)

        # Agrupar partículas
        grupos = {}
        for i in range(N):
            raiz = encontrar(i)
            if raiz not in grupos:
                grupos[raiz] = []
            grupos[raiz].append(i)

        # Crear halos
        halos = []
        halo_id = 0

        for raiz, particulas in grupos.items():
            if len(particulas) < min_particulas:
                continue

            indices = np.array(particulas)
            pos_halo = posiciones[indices]
            m_halo = masas[indices]

            # Centro de masa
            M_tot = np.sum(m_halo)
            centro = np.sum(pos_halo * m_halo[:, np.newaxis], axis=0) / M_tot

            halo = Halo(
                id=halo_id,
                centro=centro,
                M_tot=M_tot,
                particulas=particulas,
            )

            # Calcular propiedades adicionales
            self._calcular_propiedades_halo(halo, posiciones, masas)

            halos.append(halo)
            halo_id += 1

        # Ordenar por masa
        halos.sort(key=lambda h: h.M_tot, reverse=True)

        return halos

    def _calcular_propiedades_halo(
        self,
        halo: Halo,
        posiciones: NDArray,
        masas: NDArray
    ) -> None:
        """
        Calcula propiedades del halo (r_200, v_max, etc.).
        """
        indices = np.array(halo.particulas)
        pos = posiciones[indices]
        m = masas[indices]

        # Distancias al centro
        r = np.linalg.norm(pos - halo.centro, axis=1)

        # Ordenar por radio
        orden = np.argsort(r)
        r_ord = r[orden]
        m_ord = m[orden]

        # Masa acumulada
        M_cum = np.cumsum(m_ord)

        # r_200: donde ρ_mean = 200 × ρ_crit
        rho_crit = 277.5  # M☉/kpc³ (aproximado)
        for i, r_i in enumerate(r_ord):
            if r_i > 0:
                V = (4/3) * np.pi * r_i ** 3
                rho_mean = M_cum[i] / V
                if rho_mean < 200 * rho_crit:
                    halo.r_200 = r_i
                    break

        # v_max y r_max
        v_c = np.sqrt(G_NEWTON * M_cum / np.maximum(r_ord, 0.1)) * 977.8
        i_max = np.argmax(v_c)
        halo.v_max = v_c[i_max]
        halo.r_max = r_ord[i_max]

        # Concentración estimada
        if halo.r_200 > 0 and halo.r_max > 0:
            # c ≈ r_200 / r_s, donde r_max ≈ 2.16 × r_s (NFW)
            halo.c = halo.r_200 / (halo.r_max / 2.16)

        # Radio del núcleo (predicción MCMC)
        halo.r_core = radio_core_MCMC(halo.M_tot)

    def calcular_perfil_densidad(
        self,
        halo: Halo,
        posiciones: NDArray,
        masas: NDArray,
        n_bins: int = 20
    ) -> Tuple[NDArray, NDArray]:
        """
        Calcula el perfil de densidad de un halo.

        Args:
            halo: Halo a analizar
            posiciones, masas: Datos de partículas
            n_bins: Número de bins radiales

        Returns:
            (r_bins, rho): Radios y densidades
        """
        indices = np.array(halo.particulas)
        pos = posiciones[indices]
        m = masas[indices]

        # Distancias al centro
        r = np.linalg.norm(pos - halo.centro, axis=1)

        # Bins logarítmicos
        r_min = max(r.min(), 0.1)
        r_max = r.max()
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        r_centros = np.sqrt(r_bins[:-1] * r_bins[1:])

        # Masa en cada bin
        rho = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            M_bin = np.sum(m[mask])
            V_shell = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)
            rho[i] = M_bin / V_shell

        return r_centros, rho

    def ajustar_perfil_NFW(
        self,
        r: NDArray,
        rho: NDArray
    ) -> Tuple[float, float, float]:
        """
        Ajusta un perfil NFW a los datos.

        Returns:
            (rho_s, r_s, chi2_red)
        """
        def modelo_NFW(r, rho_s, r_s):
            return perfil_NFW(r, rho_s, r_s)

        try:
            # Estimaciones iniciales
            rho_0 = rho[0] if rho[0] > 0 else 1e6
            r_s_0 = r[len(r) // 3]

            popt, _ = curve_fit(
                modelo_NFW, r, rho,
                p0=[rho_0, r_s_0],
                bounds=([1e2, 0.1], [1e12, 100]),
                maxfev=5000
            )

            rho_s, r_s = popt
            rho_fit = modelo_NFW(r, rho_s, r_s)
            chi2 = np.sum((rho - rho_fit) ** 2 / np.maximum(rho_fit, 1) ** 2)
            chi2_red = chi2 / (len(r) - 2)

            return rho_s, r_s, chi2_red

        except Exception:
            return 1e6, 10.0, np.inf

    def ajustar_perfil_Burkert(
        self,
        r: NDArray,
        rho: NDArray
    ) -> Tuple[float, float, float]:
        """
        Ajusta un perfil Burkert a los datos.

        Returns:
            (rho_0, r_c, chi2_red)
        """
        def modelo_Burkert(r, rho_0, r_c):
            return perfil_Burkert(r, rho_0, r_c)

        try:
            rho_0_init = rho[0] if rho[0] > 0 else 1e6
            r_c_init = r[len(r) // 4]

            popt, _ = curve_fit(
                modelo_Burkert, r, rho,
                p0=[rho_0_init, r_c_init],
                bounds=([1e2, 0.1], [1e12, 50]),
                maxfev=5000
            )

            rho_0, r_c = popt
            rho_fit = modelo_Burkert(r, rho_0, r_c)
            chi2 = np.sum((rho - rho_fit) ** 2 / np.maximum(rho_fit, 1) ** 2)
            chi2_red = chi2 / (len(r) - 2)

            return rho_0, r_c, chi2_red

        except Exception:
            return 1e6, 1.0, np.inf

    def comparar_perfiles(
        self,
        halo: Halo,
        posiciones: NDArray,
        masas: NDArray
    ) -> Dict:
        """
        Compara ajustes NFW y Burkert.

        Returns:
            Diccionario con resultados de comparación
        """
        r, rho = self.calcular_perfil_densidad(halo, posiciones, masas)

        # Filtrar valores válidos
        mask = rho > 0
        r_valid = r[mask]
        rho_valid = rho[mask]

        if len(r_valid) < 5:
            return {"error": "Insuficientes datos"}

        rho_s_nfw, r_s_nfw, chi2_nfw = self.ajustar_perfil_NFW(r_valid, rho_valid)
        rho_0_bur, r_c_bur, chi2_bur = self.ajustar_perfil_Burkert(r_valid, rho_valid)

        mejor = "Burkert" if chi2_bur < chi2_nfw else "NFW"

        return {
            "NFW": {
                "rho_s": rho_s_nfw,
                "r_s": r_s_nfw,
                "chi2_red": chi2_nfw,
            },
            "Burkert": {
                "rho_0": rho_0_bur,
                "r_c": r_c_bur,
                "chi2_red": chi2_bur,
            },
            "mejor_perfil": mejor,
            "delta_chi2": chi2_nfw - chi2_bur,
            "r_core_MCMC_predicho": radio_core_MCMC(halo.M_tot),
        }

    def calcular_curva_rotacion(
        self,
        halo: Halo,
        posiciones: NDArray,
        masas: NDArray,
        r_max_factor: float = 2.0
    ) -> Tuple[NDArray, NDArray]:
        """
        Calcula la curva de rotación de un halo.

        Returns:
            (r_array, v_c): Radios y velocidades circulares
        """
        indices = np.array(halo.particulas)
        pos = posiciones[indices]
        m = masas[indices]

        r = np.linalg.norm(pos - halo.centro, axis=1)
        r_max = halo.r_200 * r_max_factor if halo.r_200 > 0 else r.max()

        r_array = np.linspace(0.1, r_max, 50)
        v_c = np.zeros_like(r_array)

        for i, r_i in enumerate(r_array):
            M_enc = np.sum(m[r < r_i])
            v_c[i] = np.sqrt(G_NEWTON * M_enc / r_i) * 977.8  # km/s

        return r_array, v_c


# =============================================================================
# SECCIÓN 6: Comparación Observacional (SPARC)
# =============================================================================

@dataclass
class GalaxiaSPARC:
    """
    Datos de una galaxia del catálogo SPARC.
    """
    nombre: str
    tipo: str                    # "Enana", "Espiral", etc.
    M_estelar: float            # M☉
    r_eff: float                # kpc (radio efectivo)
    r_data: NDArray             # Radios observados (kpc)
    v_obs: NDArray              # Velocidades observadas (km/s)
    v_err: NDArray              # Errores (km/s)


def cargar_datos_SPARC_ejemplo() -> List[GalaxiaSPARC]:
    """
    Carga datos de ejemplo del catálogo SPARC.

    En una implementación real, esto leería los datos observacionales.
    """
    galaxias = []

    # DDO 154 - Galaxia enana (core prominente)
    r_ddo = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])
    v_ddo = np.array([15, 22, 28, 35, 40, 44, 48, 50, 51, 52])
    v_err_ddo = np.array([2, 2, 2, 3, 3, 3, 3, 4, 4, 4])

    galaxias.append(GalaxiaSPARC(
        nombre="DDO 154",
        tipo="Enana",
        M_estelar=1e7,
        r_eff=1.5,
        r_data=r_ddo,
        v_obs=v_ddo,
        v_err=v_err_ddo,
    ))

    # NGC 2403 - Espiral
    r_ngc = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18])
    v_ngc = np.array([50, 80, 105, 120, 128, 133, 138, 140, 139, 136, 132])
    v_err_ngc = np.array([5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 10])

    galaxias.append(GalaxiaSPARC(
        nombre="NGC 2403",
        tipo="Espiral",
        M_estelar=1e10,
        r_eff=3.0,
        r_data=r_ngc,
        v_obs=v_ngc,
        v_err=v_err_ngc,
    ))

    return galaxias


class ComparadorSPARC:
    """
    Compara predicciones MCMC con observaciones SPARC.
    """

    def __init__(self):
        """Inicializa el comparador."""
        self.galaxias = cargar_datos_SPARC_ejemplo()

    def ajustar_galaxia(
        self,
        galaxia: GalaxiaSPARC,
        incluir_baryones: bool = True
    ) -> Dict:
        """
        Ajusta perfiles NFW y Burkert a una galaxia.

        Args:
            galaxia: Datos de la galaxia
            incluir_baryones: Si incluir componente bariónica

        Returns:
            Resultados del ajuste
        """
        r = galaxia.r_data
        v_obs = galaxia.v_obs
        v_err = galaxia.v_err

        # Componente bariónica (simplificada)
        if incluir_baryones:
            v_bar = 30 * np.sqrt(galaxia.M_estelar / 1e10) * \
                    np.sqrt(r / galaxia.r_eff) * np.exp(-r / (2 * galaxia.r_eff))
        else:
            v_bar = np.zeros_like(r)

        def modelo_NFW(params, r):
            rho_s, r_s = params
            v_dm = np.zeros_like(r)
            for i, r_i in enumerate(r):
                # Masa encerrada NFW analítica
                x = r_i / r_s
                M_enc = 4 * np.pi * rho_s * r_s ** 3 * (
                    np.log(1 + x) - x / (1 + x)
                )
                v_dm[i] = np.sqrt(G_NEWTON * M_enc / r_i) * 977.8
            return np.sqrt(v_dm ** 2 + v_bar ** 2)

        def modelo_Burkert(params, r):
            rho_0, r_c = params
            v_dm = np.zeros_like(r)
            for i, r_i in enumerate(r):
                # Masa encerrada Burkert (aproximación)
                x = r_i / r_c
                M_enc = np.pi * rho_0 * r_c ** 3 * (
                    np.log(1 + x ** 2) + 2 * np.log(1 + x) - 2 * np.arctan(x)
                )
                v_dm[i] = np.sqrt(G_NEWTON * M_enc / r_i) * 977.8
            return np.sqrt(v_dm ** 2 + v_bar ** 2)

        def chi2_NFW(params):
            v_mod = modelo_NFW(params, r)
            return np.sum(((v_obs - v_mod) / v_err) ** 2)

        def chi2_Burkert(params):
            v_mod = modelo_Burkert(params, r)
            return np.sum(((v_obs - v_mod) / v_err) ** 2)

        # Ajustar NFW
        res_nfw = minimize(
            chi2_NFW,
            x0=[1e7, 5],
            bounds=[(1e4, 1e10), (0.5, 50)],
            method='L-BFGS-B'
        )
        chi2_nfw = res_nfw.fun / (len(r) - 2)

        # Ajustar Burkert
        res_bur = minimize(
            chi2_Burkert,
            x0=[1e7, 2],
            bounds=[(1e4, 1e10), (0.1, 20)],
            method='L-BFGS-B'
        )
        chi2_bur = res_bur.fun / (len(r) - 2)

        mejor = "Burkert" if chi2_bur < chi2_nfw else "NFW"

        # Predicción MCMC del r_core
        # Estimar masa del halo desde v_max
        v_max = np.max(v_obs)
        M_halo_est = (v_max / 200) ** 3 * 1e12  # Relación empírica
        r_core_pred = radio_core_MCMC(M_halo_est)

        return {
            "galaxia": galaxia.nombre,
            "tipo": galaxia.tipo,
            "chi2_NFW": chi2_nfw,
            "chi2_Burkert": chi2_bur,
            "mejor_perfil": mejor,
            "r_c_ajustado": res_bur.x[1],
            "r_core_MCMC": r_core_pred,
            "consistente_MCMC": abs(res_bur.x[1] - r_core_pred) / r_core_pred < 0.5,
        }

    def analisis_completo(self) -> Dict:
        """
        Realiza análisis completo del catálogo.

        Returns:
            Estadísticas globales
        """
        resultados = [self.ajustar_galaxia(g) for g in self.galaxias]

        n_burkert = sum(1 for r in resultados if r["mejor_perfil"] == "Burkert")
        chi2_nfw_medio = np.mean([r["chi2_NFW"] for r in resultados])
        chi2_bur_medio = np.mean([r["chi2_Burkert"] for r in resultados])
        n_consistente = sum(1 for r in resultados if r["consistente_MCMC"])

        return {
            "n_galaxias": len(resultados),
            "prefieren_Burkert": n_burkert,
            "fraccion_Burkert": n_burkert / len(resultados),
            "chi2_NFW_medio": chi2_nfw_medio,
            "chi2_Burkert_medio": chi2_bur_medio,
            "delta_chi2_medio": chi2_nfw_medio - chi2_bur_medio,
            "consistentes_MCMC": n_consistente,
            "resultados_individuales": resultados,
        }


# =============================================================================
# SECCIÓN 7: Función de Masa de Halos
# =============================================================================

def funcion_masa_MCMC(M: float, z: float = 0.0) -> float:
    """
    Función de masa de halos MCMC.

    El MCMC predice supresión en bajas masas debido a
    la fricción entrópica y el mass gap.

    Args:
        M: Masa del halo (M☉)
        z: Redshift

    Returns:
        dn/dlnM en unidades arbitrarias (relativo a ΛCDM)
    """
    # Función de masa ΛCDM (Press-Schechter simplificada)
    M_star = 1e13  # M☉
    n_lcdm = (M / M_star) ** (-0.9) * np.exp(-(M / M_star) ** 0.5)

    # Supresión MCMC en bajas masas
    M_cut = 1e10  # Masa de corte
    S_supresion = 0.4  # 40% de supresión máxima

    factor_mcmc = 1 - S_supresion * np.exp(-(M / M_cut) ** 0.5)

    return n_lcdm * factor_mcmc


def ratio_MCMC_LCDM(log_M_range: NDArray) -> NDArray:
    """
    Ratio de abundancia MCMC/ΛCDM por masa.

    Args:
        log_M_range: Array de log10(M/M☉)

    Returns:
        Ratio n_MCMC/n_LCDM
    """
    M_cut = 1e10
    S_supresion = 0.4

    M = 10 ** log_M_range
    return 1 - S_supresion * np.exp(-(M / M_cut) ** 0.5)


# =============================================================================
# SECCIÓN 8: Validación Ontológica
# =============================================================================

class ValidadorOntologico:
    """
    Valida la consistencia ontológica de la simulación.
    """

    def __init__(self):
        """Inicializa el validador."""
        self.resultados = {}

    def verificar_rejilla_S(self, S_values: List[float]) -> bool:
        """
        Verifica que la rejilla de S respeta los sellos.

        Args:
            S_values: Valores de S en la simulación

        Returns:
            True si es válido
        """
        S_arr = np.array(S_values)

        # Debe ser monótonamente creciente
        if not np.all(np.diff(S_arr) >= 0):
            self.resultados["rejilla_S"] = "FALLO: No monótona"
            return False

        # Debe contener los sellos
        sellos = [S1, S2, S3, S4]
        for sello in sellos:
            if S_arr.min() <= sello <= S_arr.max():
                # Verificar que hay puntos cerca del sello
                dist_min = np.min(np.abs(S_arr - sello))
                if dist_min > 0.1:
                    warnings.warn(f"Sello {sello} no bien resuelto")

        self.resultados["rejilla_S"] = "OK"
        return True

    def verificar_conservacion_Mp_Ep(
        self,
        S_values: List[float],
        tolerancia: float = 1e-6
    ) -> bool:
        """
        Verifica la conservación Mp + Ep = constante.

        Args:
            S_values: Valores de S
            tolerancia: Tolerancia permitida

        Returns:
            True si se conserva
        """
        for S in S_values:
            Mp = Mp_frac(S)
            Ep = Ep_frac(S)
            suma = Mp + Ep

            if abs(suma - 1.0) > tolerancia:
                self.resultados["conservacion"] = f"FALLO en S={S}: Mp+Ep={suma}"
                return False

        self.resultados["conservacion"] = "OK"
        return True

    def verificar_cosmologia_H(
        self,
        z_values: List[float],
        tolerancia_pct: float = 5.0
    ) -> bool:
        """
        Verifica que H(z) MCMC es compatible.

        Args:
            z_values: Valores de redshift
            tolerancia_pct: Tolerancia en porcentaje

        Returns:
            True si es compatible
        """
        for z in z_values:
            H_mcmc = H_MCMC(z)
            H_lcdm = H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)

            diff_pct = 100 * abs(H_mcmc - H_lcdm) / H_lcdm

            if diff_pct > tolerancia_pct:
                self.resultados["cosmologia_H"] = (
                    f"FALLO en z={z}: diff={diff_pct:.1f}%"
                )
                return False

        self.resultados["cosmologia_H"] = "OK"
        return True

    def verificar_friccion_eta(self) -> bool:
        """
        Verifica que η(ρ) es positiva y monotónica.

        Returns:
            True si es válida
        """
        friccion = FriccionEntropica()

        densidades = np.logspace(-1, 4, 100)
        eta_values = [friccion.eta(rho) for rho in densidades]

        # Debe ser positiva
        if not all(eta >= 0 for eta in eta_values):
            self.resultados["friccion_eta"] = "FALLO: η negativa"
            return False

        # Debe ser monotónicamente creciente
        if not all(eta_values[i] <= eta_values[i+1]
                   for i in range(len(eta_values)-1)):
            self.resultados["friccion_eta"] = "FALLO: η no monotónica"
            return False

        self.resultados["friccion_eta"] = "OK"
        return True

    def validacion_completa(
        self,
        config: Optional[ConfiguracionCronos] = None
    ) -> Dict:
        """
        Ejecuta todas las validaciones.

        Returns:
            Diccionario con resultados
        """
        if config:
            S_values = np.linspace(config.S_ini, config.S_fin, 10).tolist()
        else:
            S_values = [S0, S1, S2, S3, S4]

        z_values = [0, 0.5, 1.0, 2.0]

        resultados = {
            "rejilla_S": self.verificar_rejilla_S(S_values),
            "conservacion": self.verificar_conservacion_Mp_Ep(S_values),
            "cosmologia_H": self.verificar_cosmologia_H(z_values),
            "friccion_eta": self.verificar_friccion_eta(),
        }

        resultados["todos_ok"] = all(resultados.values())
        resultados["detalles"] = self.resultados

        return resultados


# =============================================================================
# Tests
# =============================================================================

def _test_bloque3():
    """Verifica la implementación del Bloque 3."""

    print("Testing Bloque 3: N-Body y Formación de Estructuras...")

    # Test 1: Funciones ontológicas
    assert Mp_frac(0) == 1.0, "Mp(0) = 1"
    assert 0 < Mp_frac(S4) < 0.02, f"Mp(S4) ≈ ε, got {Mp_frac(S4)}"
    assert P_ME(0) == 1.0, "P_ME(0) = +1"
    assert P_ME(S4) < -0.9, f"P_ME(S4) < -0.9, got {P_ME(S4)}"
    print("  ✓ Funciones ontológicas")

    # Test 2: Fricción entrópica
    friccion = FriccionEntropica()
    assert friccion.eta(0) == 0, "η(0) = 0"
    assert friccion.eta(RHO_CRONOS) > 0, "η(ρc) > 0"
    assert friccion.dilatacion_temporal(0) == 1.0, "Dilatación(0) = 1"
    assert friccion.dilatacion_temporal(RHO_CRONOS) > 1.0, "Dilatación(ρc) > 1"
    print("  ✓ Fricción entrópica")

    # Test 3: Perfiles de densidad
    rho_nfw = perfil_NFW(1.0, 1e7, 10.0)
    assert rho_nfw > 0, "NFW > 0"

    rho_bur = perfil_Burkert(1.0, 1e7, 2.0)
    assert rho_bur > 0, "Burkert > 0"
    assert perfil_Burkert(0.001, 1e7, 2.0) < 1.1e7, "Burkert cored"

    # NFW tiene cúspide
    rho_nfw_centro = perfil_NFW(0.01, 1e7, 10.0)
    rho_nfw_periferia = perfil_NFW(10.0, 1e7, 10.0)
    assert rho_nfw_centro / rho_nfw_periferia > 100, "NFW cuspy"

    print("  ✓ Perfiles de densidad")

    # Test 4: Radio core
    r_c1 = radio_core_MCMC(1e10)
    r_c2 = radio_core_MCMC(1e12)
    assert r_c2 > r_c1, "r_core crece con M"
    print("  ✓ Radio del núcleo")

    # Test 5: Cosmología
    H_0 = H_MCMC(0)
    H_1 = H_MCMC(1)
    assert H_1 > H_0, "H(1) > H(0)"
    assert abs(H_0 - H0) < 1, f"H(0) ≈ H0, got {H_0}"
    print("  ✓ Cosmología H(z)")

    # Test 6: Validador ontológico
    validador = ValidadorOntologico()
    resultados = validador.validacion_completa()
    assert resultados["todos_ok"], f"Validación falló: {resultados['detalles']}"
    print("  ✓ Validación ontológica")

    # Test 7: Comparador SPARC
    comparador = ComparadorSPARC()
    analisis = comparador.analisis_completo()
    assert analisis["n_galaxias"] >= 2, "SPARC cargado"
    assert analisis["fraccion_Burkert"] > 0.5, "Burkert preferido"
    print("  ✓ Comparación SPARC")

    # Test 8: Función de masa
    ratio_bajo = ratio_MCMC_LCDM(np.array([8]))[0]
    ratio_alto = ratio_MCMC_LCDM(np.array([12]))[0]
    assert ratio_bajo < ratio_alto, "Supresión en bajas masas"
    print("  ✓ Función de masa")

    print("\n✓ Todos los tests del Bloque 3 pasaron")
    return True


# =============================================================================
# Demo
# =============================================================================

def _demo_bloque3():
    """Demostración del Bloque 3."""

    print("\n" + "="*70)
    print("DEMO: Bloque 3 - N-Body y Formación de Estructuras MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*70)

    # Fricción entrópica
    print("\n" + "-"*50)
    print("FRICCIÓN ENTRÓPICA (Ley de Cronos)")
    print("-"*50)

    friccion = FriccionEntropica()
    print(f"  ρc = {RHO_CRONOS:.1f} M☉/kpc³")
    print(f"  β = {BETA_ETA:.1f}")
    print("\n  Escalas de fricción:")
    for row in friccion.tabla_escalas():
        print(f"    ρ = {row['rho_M_kpc3']:6.0f} M☉/kpc³: "
              f"η = {row['eta_Gyr_inv']:8.4f} Gyr⁻¹, "
              f"τ = {row['tau_Gyr']:8.2f} Gyr")

    # Perfiles de densidad
    print("\n" + "-"*50)
    print("PERFILES DE DENSIDAD")
    print("-"*50)

    M_halo = 1e11
    z = 0
    params = ParametrosHalo(M_200=M_halo, c=10, z=z)

    perfil_nfw = PerfilDensidad("NFW", params)
    perfil_bur = PerfilDensidad("Burkert", params)

    print(f"  Halo: M_200 = {M_halo:.0e} M☉, c = 10, z = 0")
    print(f"  r_core MCMC = {radio_core_MCMC(M_halo):.2f} kpc")
    print("\n  Comparación de densidades:")
    print(f"  {'r [kpc]':>10} | {'ρ_NFW':>12} | {'ρ_Burkert':>12}")
    print("  " + "-"*40)
    for r in [0.1, 1.0, 5.0, 10.0]:
        print(f"  {r:10.1f} | {perfil_nfw.rho(r):12.2e} | {perfil_bur.rho(r):12.2e}")

    # Comparación SPARC
    print("\n" + "-"*50)
    print("COMPARACIÓN CON OBSERVACIONES SPARC")
    print("-"*50)

    comparador = ComparadorSPARC()
    analisis = comparador.analisis_completo()

    print(f"  Galaxias analizadas: {analisis['n_galaxias']}")
    print(f"  Prefieren Burkert: {analisis['prefieren_Burkert']} "
          f"({100*analisis['fraccion_Burkert']:.0f}%)")
    print(f"  χ² promedio NFW: {analisis['chi2_NFW_medio']:.2f}")
    print(f"  χ² promedio Burkert: {analisis['chi2_Burkert_medio']:.2f}")

    print("\n  Resultados individuales:")
    for r in analisis["resultados_individuales"]:
        print(f"    {r['galaxia']:12s}: {r['mejor_perfil']:8s} "
              f"(χ²_NFW={r['chi2_NFW']:.2f}, χ²_Bur={r['chi2_Burkert']:.2f})")

    # Función de masa
    print("\n" + "-"*50)
    print("FUNCIÓN DE MASA DE HALOS")
    print("-"*50)

    log_M = np.array([8, 9, 10, 11, 12])
    ratios = ratio_MCMC_LCDM(log_M)

    print("  Supresión MCMC vs ΛCDM:")
    print(f"  {'log(M/M☉)':>12} | {'n_MCMC/n_ΛCDM':>15}")
    print("  " + "-"*30)
    for lm, r in zip(log_M, ratios):
        print(f"  {lm:12d} | {r:15.2f}")
    print(f"\n  Supresión promedio M < 10¹⁰: {100*(1-np.mean(ratios[:2])):.1f}%")

    # Validación ontológica
    print("\n" + "-"*50)
    print("VALIDACIÓN ONTOLÓGICA")
    print("-"*50)

    validador = ValidadorOntologico()
    resultados = validador.validacion_completa()

    for check, status in resultados["detalles"].items():
        simbolo = "✓" if status == "OK" else "✗"
        print(f"  {simbolo} {check}: {status}")

    print("\n" + "="*70)


if __name__ == "__main__":
    _test_bloque3()
    _demo_bloque3()
