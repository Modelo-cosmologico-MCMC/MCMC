#!/usr/bin/env python3
"""
================================================================================
MAPA DE ENTROPIA 3D S(z, n̂) - MCMC
================================================================================

Implementa el mapa completo de entropia S(z, n̂) que describe la variacion
espacial de la entropia del oceano geometrico en funcion del redshift z
y la direccion angular n̂.

FUNDAMENTOS ONTOLOGICOS:
------------------------
En el MCMC, la entropia S no es uniforme en el espacio. Varia segun:
1. Redshift z: S_ext(z) decrece con z (universo mas joven = mas ordenado)
2. Ambiente local: galaxias, cumulos, voids tienen diferentes S_local
3. Estructura LSS: perturbaciones de densidad correlacionan con delta_S

ECUACIONES FUNDAMENTALES:
-------------------------
(Ec. 292) S(x) = S_ext(z) + Delta_S(x)

donde:
- S_ext(z) = S_0 * (t(z)/t_0)^p con p ~ 0.5
- Delta_S(x) = lambda_LSS * delta_m(x) + fluctuaciones

COMPONENTES DEL MAPA:
---------------------
1. S_ext(z): Entropia externa media a redshift z
2. Delta_S_env: Variacion por ambiente (galaxia, cumulo, void, filamento)
3. Delta_S_LSS: Perturbaciones de estructura a gran escala
4. Xi(x): Potencial cronologico derivado de S(x)

OUTPUTS:
--------
- Mapa HEALPix de S(z, n̂) para z dado
- Espectro de potencias C_l^SS de las fluctuaciones
- Correlacion cruzada con campo de densidad

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import spherical_jn
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FISICAS Y COSMOLOGICAS
# =============================================================================

# Cosmologia
H0 = 67.36                      # km/s/Mpc
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847
C_LIGHT = 299792.458            # km/s

# Parametros MCMC
EPSILON_ECV = 0.012             # Parametro de transicion ECV
Z_TRANS = 1.0                   # Redshift de transicion (ontologia MCMC)
DELTA_Z = 1.5                   # Anchura de transicion

# Entropia
S_0 = 90.0                      # S_ext(z=0) en unidades MCMC
S_PLANCK = 1.0                  # S en escala de Planck (normalizado)
P_ENTROPY = 0.5                 # Exponente temporal

# LSS
LAMBDA_LSS = 0.05               # Acoplamiento entropia-densidad
SIGMA_8 = 0.811                 # Normalizacion del espectro de potencias


# =============================================================================
# AMBIENTES COSMICOS
# =============================================================================

class AmbienteCosmico(Enum):
    """Tipos de ambientes cosmicos."""
    VOID = "Void"
    FILAMENTO = "Filamento"
    MURO = "Muro"
    CUMULO = "Cumulo"
    GALAXIA = "Galaxia"
    CAMPO = "Campo"


@dataclass
class PropiedadesAmbiente:
    """Propiedades de un ambiente cosmico."""
    nombre: str
    delta_rho: float        # Sobredensidad media
    Delta_S: float          # Delta_S tipico
    r_tipico_Mpc: float     # Radio tipico
    fraccion_volumen: float # Fraccion del volumen del universo


AMBIENTES = {
    AmbienteCosmico.VOID: PropiedadesAmbiente(
        "Void", -0.8, 2.0, 20.0, 0.60
    ),
    AmbienteCosmico.FILAMENTO: PropiedadesAmbiente(
        "Filamento", 0.5, -1.0, 5.0, 0.25
    ),
    AmbienteCosmico.MURO: PropiedadesAmbiente(
        "Muro/Sheet", 1.0, -2.0, 3.0, 0.10
    ),
    AmbienteCosmico.CUMULO: PropiedadesAmbiente(
        "Cumulo", 100.0, -15.0, 2.0, 0.02
    ),
    AmbienteCosmico.GALAXIA: PropiedadesAmbiente(
        "Galaxia", 1e5, -30.0, 0.05, 0.001
    ),
    AmbienteCosmico.CAMPO: PropiedadesAmbiente(
        "Campo/Field", 0.0, 0.0, 10.0, 0.03
    ),
}


# =============================================================================
# FUNCIONES COSMOLOGICAS BASE
# =============================================================================

def Lambda_rel(z: float) -> float:
    """Constante cosmologica relacional MCMC."""
    return 1.0 + EPSILON_ECV * np.tanh((Z_TRANS - z) / DELTA_Z)


def E_z(z: float) -> float:
    """E(z) = H(z)/H0."""
    return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA * Lambda_rel(z))


def H_z(z: float) -> float:
    """H(z) en km/s/Mpc."""
    return H0 * E_z(z)


def tiempo_cosmico(z: float) -> float:
    """
    Tiempo cosmico t(z) en Gyr desde el Big Bang.

    Aproximacion analitica para cosmologia plana.
    """
    if z > 1000:
        return 0.0

    # Integracion numerica
    def integrand(zp):
        return 1.0 / ((1 + zp) * E_z(zp))

    t_H = 9.78 / (H0 / 100)  # Tiempo de Hubble en Gyr
    integral, _ = quad(integrand, z, 1000)

    return t_H * integral


def distancia_comovil(z: float) -> float:
    """Distancia comovil r(z) en Mpc."""
    if z <= 0:
        return 0.0

    def integrand(zp):
        return C_LIGHT / H_z(zp)

    result, _ = quad(integrand, 0, z)
    return result


# =============================================================================
# ENTROPIA EXTERNA S_ext(z)
# =============================================================================

class EntropiaExterna:
    """
    Calcula S_ext(z) - la entropia del oceano geometrico externo.

    S_ext(z) representa el valor medio de la entropia a redshift z,
    promediando sobre todas las direcciones y ambientes.

    En el MCMC, S_ext decrece con z porque el universo joven
    estaba mas ordenado (menor entropia).

    S_ext(z) = S_0 * (t(z)/t_0)^p

    donde p ~ 0.5 captura la evolucion gradual.
    """

    def __init__(self, S_0: float = 90.0, p: float = 0.5):
        """
        Inicializa el modelo de entropia externa.

        Args:
            S_0: Entropia en z=0 (hoy)
            p: Exponente de evolucion temporal
        """
        self.S_0 = S_0
        self.p = p
        self.t_0 = tiempo_cosmico(0)  # ~13.8 Gyr

        # Cache de valores
        self._cache = {}

    def __call__(self, z: float) -> float:
        """Calcula S_ext(z)."""
        return self.S_ext(z)

    def S_ext(self, z: float) -> float:
        """
        Entropia externa a redshift z.

        Args:
            z: Redshift

        Returns:
            S_ext(z)
        """
        if z in self._cache:
            return self._cache[z]

        if z <= 0:
            return self.S_0

        # Tiempo cosmico a z
        t_z = tiempo_cosmico(z)

        # Evitar division por cero
        if t_z <= 0 or self.t_0 <= 0:
            return self.S_0 * 0.01

        # S_ext(z) = S_0 * (t(z)/t_0)^p
        ratio = t_z / self.t_0
        S = self.S_0 * ratio**self.p

        # Cache
        self._cache[z] = S

        return S

    def dS_dz(self, z: float) -> float:
        """Derivada dS_ext/dz."""
        eps = 0.01
        return (self.S_ext(z + eps) - self.S_ext(z - eps)) / (2 * eps)

    def tabla(self, z_array: np.ndarray = None) -> Dict:
        """Genera tabla de S_ext vs z."""
        if z_array is None:
            z_array = np.array([0, 0.5, 1, 2, 5, 10, 50, 100])

        S_array = np.array([self.S_ext(z) for z in z_array])
        t_array = np.array([tiempo_cosmico(z) for z in z_array])

        return {
            'z': z_array,
            'S_ext': S_array,
            't_Gyr': t_array,
            'S_ext/S_0': S_array / self.S_0
        }


# Instancia global
S_EXT = EntropiaExterna()


# =============================================================================
# VARIACION POR AMBIENTE Delta_S_env
# =============================================================================

class VariacionAmbiente:
    """
    Calcula Delta_S debido al ambiente local.

    Diferentes ambientes tienen diferentes entropias:
    - Voids: S > S_ext (baja densidad = alta entropia local)
    - Galaxias: S < S_ext (alta densidad = baja entropia local)
    - Cumulos: S << S_ext (muy alta densidad)

    La relacion basica es:
    Delta_S ~ -lambda * log(1 + delta_rho)

    donde delta_rho = (rho - rho_mean) / rho_mean
    """

    def __init__(self, lambda_env: float = 5.0):
        """
        Inicializa el modelo de variacion por ambiente.

        Args:
            lambda_env: Acoplamiento ambiente-entropia
        """
        self.lambda_env = lambda_env

    def Delta_S(self, ambiente: AmbienteCosmico) -> float:
        """
        Delta_S para un ambiente dado.

        Args:
            ambiente: Tipo de ambiente cosmico

        Returns:
            Delta_S (puede ser positivo o negativo)
        """
        props = AMBIENTES[ambiente]
        return props.Delta_S

    def Delta_S_from_delta_rho(self, delta_rho: float) -> float:
        """
        Delta_S desde sobredensidad.

        Args:
            delta_rho: (rho - rho_mean) / rho_mean

        Returns:
            Delta_S
        """
        # Evitar log de numeros negativos
        x = max(1 + delta_rho, 0.01)
        return -self.lambda_env * np.log(x)

    def S_local(self, z: float, ambiente: AmbienteCosmico) -> float:
        """
        Entropia local en un ambiente a redshift z.

        Args:
            z: Redshift
            ambiente: Tipo de ambiente

        Returns:
            S_local = S_ext(z) + Delta_S_env
        """
        S_ext_z = S_EXT(z)
        Delta_S = self.Delta_S(ambiente)
        return S_ext_z + Delta_S


# =============================================================================
# PERTURBACIONES LSS Delta_S_LSS
# =============================================================================

class PerturbacionesLSS:
    """
    Modela las perturbaciones de entropia correlacionadas con LSS.

    Las fluctuaciones de densidad delta_m(x) generan fluctuaciones
    de entropia delta_S(x) via:

    delta_S(x) = lambda_LSS * delta_m(x)

    El espectro de potencias de delta_S es:
    P_SS(k) = lambda_LSS^2 * P_mm(k)

    Y el espectro angular:
    C_l^SS = integral de P_SS * j_l^2
    """

    def __init__(self, lambda_LSS: float = 0.05, sigma_8: float = 0.811):
        """
        Inicializa el modelo de perturbaciones.

        Args:
            lambda_LSS: Acoplamiento entropia-densidad
            sigma_8: Normalizacion del espectro
        """
        self.lambda_LSS = lambda_LSS
        self.sigma_8 = sigma_8

        # Parametros del espectro
        self.n_s = 0.965      # Indice espectral
        self.k_pivot = 0.05   # Mpc^-1

    def P_mm(self, k: float, z: float = 0) -> float:
        """
        Espectro de potencias de materia P(k, z).

        Modelo simplificado Eisenstein-Hu.

        Args:
            k: Numero de onda [h/Mpc]
            z: Redshift

        Returns:
            P(k) en (Mpc/h)^3
        """
        # Normalizacion a sigma_8
        A = self.sigma_8**2 * (2 * np.pi**2) / (k**3)

        # Forma del espectro
        q = k / (OMEGA_M * H0 / 100)
        T_k = np.log(1 + 2.34 * q) / (2.34 * q) * (
            1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
        )**(-0.25)

        # Crecimiento lineal
        D_z = self._factor_crecimiento(z)

        # Espectro primordial
        P_prim = (k / self.k_pivot)**(self.n_s - 1)

        return A * T_k**2 * D_z**2 * P_prim

    def _factor_crecimiento(self, z: float) -> float:
        """Factor de crecimiento D(z) normalizado a D(0)=1."""
        a = 1 / (1 + z)
        omega_m_z = OMEGA_M * (1 + z)**3 / E_z(z)**2

        # Aproximacion de Carroll et al.
        D = a * omega_m_z**(0.55) / (
            omega_m_z**(4/7) - (1 - omega_m_z) +
            (1 + omega_m_z / 2) * (1 + (1 - omega_m_z) / 70)
        )

        # Normalizar
        D_0 = 1.0 * OMEGA_M**(0.55) / (
            OMEGA_M**(4/7) - OMEGA_LAMBDA +
            (1 + OMEGA_M / 2) * (1 + OMEGA_LAMBDA / 70)
        )

        return D / D_0

    def P_SS(self, k: float, z: float = 0) -> float:
        """
        Espectro de potencias de entropia P_SS(k).

        P_SS(k) = lambda_LSS^2 * P_mm(k)
        """
        return self.lambda_LSS**2 * self.P_mm(k, z)

    def C_l_SS(self, l: int, z: float = 0) -> float:
        """
        Espectro angular de fluctuaciones de entropia C_l^SS.

        C_l = (2/pi) * integral de k^2 * P_SS(k) * j_l(k*r)^2 dk

        Args:
            l: Multipolo
            z: Redshift

        Returns:
            C_l^SS
        """
        r_z = distancia_comovil(z) if z > 0 else 100.0  # Mpc

        def integrand(k):
            if k <= 0:
                return 0
            # Bessel esferica
            x = k * r_z
            if x < 1e-10:
                j_l = 1.0 if l == 0 else 0.0
            else:
                j_l = spherical_jn(l, x)
            return k**2 * self.P_SS(k, z) * j_l**2

        # Integracion
        k_min, k_max = 1e-4, 10.0
        result, _ = quad(integrand, k_min, k_max, limit=100)

        return 2 / np.pi * result

    def delta_S_rms(self, z: float = 0, R: float = 8.0) -> float:
        """
        RMS de fluctuaciones de entropia suavizadas a escala R.

        sigma_S = lambda_LSS * sigma_8 * D(z)
        """
        D_z = self._factor_crecimiento(z)
        return self.lambda_LSS * self.sigma_8 * D_z

    def generar_campo_delta_S(self, N: int = 64, L: float = 500.0,
                               z: float = 0, seed: int = 42) -> np.ndarray:
        """
        Genera realizacion del campo delta_S en una caja.

        Args:
            N: Numero de celdas por lado
            L: Tamano de la caja [Mpc]
            z: Redshift
            seed: Semilla aleatoria

        Returns:
            Array 3D de delta_S
        """
        np.random.seed(seed)

        # Modos de Fourier
        k_modes = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k_modes, k_modes, k_modes, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0, 0, 0] = 1  # Evitar division por cero

        # Espectro
        P_k = np.array([self.P_SS(k, z) if k > 0 else 0 for k in k_mag.flat])
        P_k = P_k.reshape(k_mag.shape)

        # Amplitudes gaussianas
        amplitude = np.sqrt(P_k * (L / N)**3 / 2)

        # Fases aleatorias
        phases = np.random.uniform(0, 2 * np.pi, k_mag.shape)

        # Campo en Fourier
        delta_k = amplitude * np.exp(1j * phases)

        # Transformar a espacio real
        delta_S = np.fft.ifftn(delta_k).real

        return delta_S


# =============================================================================
# MAPA 3D COMPLETO S(z, n̂)
# =============================================================================

@dataclass
class PuntoMapa:
    """Un punto en el mapa de entropia."""
    z: float
    theta: float  # Colatitud [rad]
    phi: float    # Azimut [rad]
    S: float      # Entropia total
    S_ext: float  # Entropia externa
    Delta_S: float  # Variacion total
    ambiente: AmbienteCosmico = AmbienteCosmico.CAMPO


class MapaEntropia3D:
    """
    Mapa completo de entropia S(z, n̂).

    Combina:
    - S_ext(z): Entropia externa media
    - Delta_S_env: Variacion por ambiente local
    - Delta_S_LSS: Perturbaciones de estructura a gran escala

    Puede generar:
    - Mapas HEALPix a z fijo
    - Perfiles radiales S(z) en direccion fija
    - Espectros de potencias angulares
    """

    def __init__(self,
                 S_ext_model: EntropiaExterna = None,
                 env_model: VariacionAmbiente = None,
                 lss_model: PerturbacionesLSS = None):
        """
        Inicializa el mapa 3D.

        Args:
            S_ext_model: Modelo de entropia externa
            env_model: Modelo de variacion por ambiente
            lss_model: Modelo de perturbaciones LSS
        """
        self.S_ext = S_ext_model if S_ext_model else EntropiaExterna()
        self.env = env_model if env_model else VariacionAmbiente()
        self.lss = lss_model if lss_model else PerturbacionesLSS()

        # Cache de campos LSS
        self._lss_cache = {}

    def S(self, z: float, theta: float = 0, phi: float = 0,
          ambiente: AmbienteCosmico = None,
          include_lss: bool = True) -> float:
        """
        Entropia total en (z, theta, phi).

        Args:
            z: Redshift
            theta: Colatitud [rad]
            phi: Azimut [rad]
            ambiente: Ambiente local (auto-detecta si None)
            include_lss: Incluir perturbaciones LSS

        Returns:
            S(z, n̂)
        """
        # Entropia externa
        S_ext_z = self.S_ext(z)

        # Variacion por ambiente
        if ambiente is None:
            ambiente = self._detectar_ambiente(z, theta, phi)
        Delta_S_env = self.env.Delta_S(ambiente)

        # Perturbaciones LSS
        Delta_S_lss = 0.0
        if include_lss:
            Delta_S_lss = self._get_delta_S_lss(z, theta, phi)

        # Total
        return S_ext_z + Delta_S_env + Delta_S_lss

    def _detectar_ambiente(self, z: float, theta: float, phi: float) -> AmbienteCosmico:
        """
        Detecta el ambiente cosmico basado en la posicion.

        Usa un modelo estadistico simple basado en las fracciones de volumen.
        """
        # Usar coordenadas como semilla pseudo-aleatoria
        seed = int((z * 1000 + theta * 100 + phi * 10) % 10000)
        np.random.seed(seed)

        r = np.random.random()

        cumsum = 0
        for amb, props in AMBIENTES.items():
            cumsum += props.fraccion_volumen
            if r < cumsum:
                return amb

        return AmbienteCosmico.CAMPO

    def _get_delta_S_lss(self, z: float, theta: float, phi: float) -> float:
        """Obtiene delta_S de LSS interpolando del campo."""
        # Calcular delta_S_rms y modular con posicion
        sigma_S = self.lss.delta_S_rms(z)

        # Fluctuacion determinista basada en posicion
        seed = int((z * 1000 + theta * 100 + phi * 10) % 10000)
        np.random.seed(seed)

        return np.random.normal(0, sigma_S)

    def generar_shell(self, z: float, nside: int = 32,
                      include_lss: bool = True) -> Dict:
        """
        Genera un shell (cascara) de entropia a z fijo.

        Simula un mapa HEALPix de S(n̂) a redshift z.

        Args:
            z: Redshift del shell
            nside: Resolucion HEALPix (npix = 12 * nside^2)
            include_lss: Incluir perturbaciones LSS

        Returns:
            Dict con el mapa y estadisticas
        """
        npix = 12 * nside**2

        # Generar coordenadas angulares
        theta = np.zeros(npix)
        phi = np.zeros(npix)
        S_map = np.zeros(npix)

        for i in range(npix):
            # Coordenadas HEALPix simplificadas
            theta[i] = np.arccos(1 - 2 * (i + 0.5) / npix)
            phi[i] = (i % int(np.sqrt(npix))) * 2 * np.pi / np.sqrt(npix)

            # Calcular S
            S_map[i] = self.S(z, theta[i], phi[i], include_lss=include_lss)

        # Estadisticas
        S_ext_z = self.S_ext(z)
        Delta_S_map = S_map - S_ext_z

        return {
            'z': z,
            'nside': nside,
            'npix': npix,
            'S_map': S_map,
            'theta': theta,
            'phi': phi,
            'S_ext': S_ext_z,
            'Delta_S': Delta_S_map,
            'S_mean': np.mean(S_map),
            'S_std': np.std(S_map),
            'Delta_S_rms': np.std(Delta_S_map),
        }

    def perfil_radial(self, theta: float = 0, phi: float = 0,
                      z_array: np.ndarray = None,
                      ambiente: AmbienteCosmico = None) -> Dict:
        """
        Genera perfil radial S(z) en direccion fija.

        Args:
            theta, phi: Direccion angular
            z_array: Array de redshifts
            ambiente: Ambiente fijo (None = variable)

        Returns:
            Dict con perfil y componentes
        """
        if z_array is None:
            z_array = np.linspace(0, 5, 100)

        S_array = np.zeros_like(z_array)
        S_ext_array = np.zeros_like(z_array)
        Delta_S_array = np.zeros_like(z_array)

        for i, z in enumerate(z_array):
            S_array[i] = self.S(z, theta, phi, ambiente)
            S_ext_array[i] = self.S_ext(z)
            Delta_S_array[i] = S_array[i] - S_ext_array[i]

        return {
            'z': z_array,
            'S': S_array,
            'S_ext': S_ext_array,
            'Delta_S': Delta_S_array,
            'theta': theta,
            'phi': phi,
        }

    def Xi(self, z: float, theta: float = 0, phi: float = 0,
           ambiente: AmbienteCosmico = None) -> float:
        """
        Potencial cronologico Xi(z, n̂).

        Xi = lambda_S * (S_ext - S_local) / S_ext

        donde lambda_S es el acoplamiento entropia-potencial.
        """
        lambda_S = 0.01  # Acoplamiento

        S_ext_z = self.S_ext(z)
        S_local = self.S(z, theta, phi, ambiente)

        if S_ext_z <= 0:
            return 0.0

        return lambda_S * (S_ext_z - S_local) / S_ext_z

    def espectro_angular(self, z: float, l_max: int = 100) -> Dict:
        """
        Calcula espectro angular C_l^SS de las fluctuaciones.

        Args:
            z: Redshift
            l_max: Multipolo maximo

        Returns:
            Dict con l y C_l
        """
        l_array = np.arange(2, l_max + 1)
        C_l = np.zeros(len(l_array))

        for i, l in enumerate(l_array):
            C_l[i] = self.lss.C_l_SS(l, z)

        return {
            'l': l_array,
            'C_l': C_l,
            'l_C_l': l_array * (l_array + 1) * C_l / (2 * np.pi),
            'z': z,
        }


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_Entropy_Map_3D() -> bool:
    """
    Test del modulo de mapa de entropia 3D.
    """
    print("\n" + "=" * 70)
    print("  TEST ENTROPY MAP 3D - MAPA S(z, n̂)")
    print("=" * 70)

    # 1. Verificar S_ext(z)
    print("\n[1] Verificacion de S_ext(z):")
    print("-" * 70)

    S_ext_model = EntropiaExterna()

    z_test = [0, 0.5, 1, 2, 5, 10]
    print(f"    {'z':>6} {'S_ext':>10} {'S_ext/S_0':>12}")

    S_prev = S_ext_model.S_0 + 1  # Iniciar con valor alto
    monotono = True
    for z in z_test:
        S = S_ext_model(z)
        ratio = S / S_ext_model.S_0
        print(f"    {z:>6.1f} {S:>10.2f} {ratio:>12.4f}")

        if S > S_prev:
            monotono = False
        S_prev = S

    sext_ok = monotono
    print(f"\n    S_ext decrece con z: {'PASS' if sext_ok else 'FAIL'}")

    # 2. Verificar variacion por ambiente
    print("\n[2] Verificacion de Delta_S por ambiente:")
    print("-" * 70)

    env_model = VariacionAmbiente()

    print(f"    {'Ambiente':>12} {'Delta_S':>10} {'S_local(z=0)':>15}")

    for amb in AmbienteCosmico:
        Delta_S = env_model.Delta_S(amb)
        S_local = env_model.S_local(0, amb)
        print(f"    {amb.value:>12} {Delta_S:>10.1f} {S_local:>15.2f}")

    # Verificar que galaxia < ext
    S_galaxia = env_model.S_local(0, AmbienteCosmico.GALAXIA)
    S_ext_0 = S_ext_model(0)
    env_ok = S_galaxia < S_ext_0
    print(f"\n    S(galaxia) < S_ext: {S_galaxia:.1f} < {S_ext_0:.1f} -> "
          f"{'PASS' if env_ok else 'FAIL'}")

    # 3. Verificar perturbaciones LSS
    print("\n[3] Verificacion de perturbaciones LSS:")
    print("-" * 70)

    lss_model = PerturbacionesLSS()

    sigma_S_0 = lss_model.delta_S_rms(0)
    sigma_S_1 = lss_model.delta_S_rms(1)

    print(f"    sigma_S(z=0) = {sigma_S_0:.4f}")
    print(f"    sigma_S(z=1) = {sigma_S_1:.4f}")

    # sigma_S debe decrecer con z (menos estructura)
    lss_ok = sigma_S_1 < sigma_S_0
    print(f"    sigma_S decrece con z: {'PASS' if lss_ok else 'FAIL'}")

    # 4. Mapa 3D completo
    print("\n[4] Mapa 3D completo:")
    print("-" * 70)

    mapa = MapaEntropia3D()

    # Shell a z=0.5
    shell = mapa.generar_shell(z=0.5, nside=8)

    print(f"    Shell a z=0.5:")
    print(f"      S_ext = {shell['S_ext']:.2f}")
    print(f"      S_mean = {shell['S_mean']:.2f}")
    print(f"      Delta_S_rms = {shell['Delta_S_rms']:.4f}")

    # Verificar que hay variacion
    map_ok = shell['Delta_S_rms'] > 0
    print(f"    Variaciones presentes: {'PASS' if map_ok else 'FAIL'}")

    # 5. Perfil radial
    print("\n[5] Perfil radial:")
    print("-" * 70)

    perfil = mapa.perfil_radial(z_array=np.array([0, 1, 2, 3, 4, 5]))

    print(f"    {'z':>6} {'S':>10} {'S_ext':>10} {'Delta_S':>10}")
    for i, z in enumerate(perfil['z']):
        print(f"    {z:>6.1f} {perfil['S'][i]:>10.2f} "
              f"{perfil['S_ext'][i]:>10.2f} {perfil['Delta_S'][i]:>10.2f}")

    # 6. Potencial cronologico Xi
    print("\n[6] Potencial cronologico Xi:")
    print("-" * 70)

    for amb in [AmbienteCosmico.VOID, AmbienteCosmico.CAMPO, AmbienteCosmico.CUMULO]:
        Xi = mapa.Xi(0.5, ambiente=amb)
        print(f"    Xi(z=0.5, {amb.value}): {Xi:>10.6f}")

    # Xi debe ser mayor en cumulos (mas sobredensidad)
    Xi_void = mapa.Xi(0.5, ambiente=AmbienteCosmico.VOID)
    Xi_cumulo = mapa.Xi(0.5, ambiente=AmbienteCosmico.CUMULO)
    xi_ok = Xi_cumulo > Xi_void
    print(f"\n    Xi(cumulo) > Xi(void): {'PASS' if xi_ok else 'FAIL'}")

    # Resultado final
    passed = sext_ok and env_ok and lss_ok and map_ok and xi_ok

    print("\n" + "=" * 70)
    print(f"  ENTROPY MAP 3D: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_Entropy_Map_3D()
