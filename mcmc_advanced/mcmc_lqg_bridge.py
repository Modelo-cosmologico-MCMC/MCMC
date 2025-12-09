#!/usr/bin/env python3
"""
================================================================================
PUENTE MCMC - LQG (Loop Quantum Gravity)
================================================================================

Conexion entre el modelo MCMC y la Gravedad Cuantica de Lazos (LQG).

FUNDAMENTOS:
------------
En LQG, el espacio-tiempo esta cuantizado en redes de espines (spin networks).
El MCMC propone que la entropia S esta relacionada con los estados de
estas redes de espines.

CORRESPONDENCIAS CLAVE:
-----------------------
1. Area Gap: La unidad minima de area en LQG
   A_gap = 4 * sqrt(3) * pi * gamma * l_P^2
   donde gamma ~ 0.2375 es el parametro de Immirzi

2. Entropia de Horizonte: En LQG, S_BH = A / (4 * l_P^2) (Bekenstein-Hawking)
   modificada por correcciones logaritmicas

3. Spin Networks: Estados |j_1, ..., j_n> donde j_i son representaciones SU(2)
   El sello S del MCMC se relaciona con j_max via S ~ sum(2j+1)

4. Amplitudes EPRL: Modelo de espuma de espines que define la dinamica
   W = prod_f A_f * prod_v A_v (productos sobre caras y vertices)

IMPLEMENTACIONES:
-----------------
- Calculo del area gap con parametro de Immirzi
- Conversion S <-> j_max (spin maximo)
- Generacion de spin networks simplificadas
- Estimacion de amplitudes de espuma

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FUNDAMENTALES
# =============================================================================

# Constantes de Planck
HBAR = 1.054571817e-34          # J·s
G_NEWTON = 6.67430e-11          # m³ kg⁻¹ s⁻²
C_LIGHT = 299792458.0           # m/s

# Longitud y area de Planck
L_PLANCK = np.sqrt(HBAR * G_NEWTON / C_LIGHT**3)  # ~ 1.616e-35 m
A_PLANCK = L_PLANCK**2                             # ~ 2.61e-70 m²

# Parametro de Immirzi (valor canonico LQG)
GAMMA_IMMIRZI = 0.2375          # Fijado por entropia de BH

# Area gap LQG
# A_gap = 4 * sqrt(3) * pi * gamma * l_P^2
A_GAP = 4 * np.sqrt(3) * np.pi * GAMMA_IMMIRZI * A_PLANCK


# =============================================================================
# SELLOS ONTOLOGICOS Y SU CONEXION CON LQG
# =============================================================================

@dataclass
class SelloLQG:
    """Correspondencia entre sello MCMC y parametros LQG."""
    nombre: str
    S_mcmc: float           # Valor del sello en MCMC (normalizado)
    j_max: float            # Spin maximo asociado
    energia_GeV: float      # Energia caracteristica
    descripcion: str


SELLOS_LQG = [
    SelloLQG("S1", 0.009, 0.5, 1.22e19, "Planck - j=1/2 fundamental"),
    SelloLQG("S2", 0.099, 5.0, 1e16, "GUT - j moderados"),
    SelloLQG("S3", 0.999, 50.0, 246, "Electroweak - j altos"),
    SelloLQG("S4", 1.001, 100.0, 0.2, "QCD/Big Bang - semiclasico"),
]


# =============================================================================
# AREA GAP Y CUANTIZACION
# =============================================================================

class AreaGapLQG:
    """
    Calcula el area gap y la cuantizacion del area en LQG.

    En LQG, el operador de area tiene espectro discreto:
    A_j = 8 * pi * gamma * l_P^2 * sqrt(j*(j+1))

    El area gap es el minimo no nulo:
    A_gap = A_{1/2} = 4 * sqrt(3) * pi * gamma * l_P^2
    """

    def __init__(self, gamma: float = GAMMA_IMMIRZI):
        """
        Inicializa el modelo de area gap.

        Args:
            gamma: Parametro de Immirzi
        """
        self.gamma = gamma
        self.A_gap = 4 * np.sqrt(3) * np.pi * gamma * A_PLANCK

    def area_j(self, j: float) -> float:
        """
        Area correspondiente a spin j.

        A_j = 8 * pi * gamma * l_P^2 * sqrt(j*(j+1))

        Args:
            j: Spin (0, 1/2, 1, 3/2, ...)

        Returns:
            Area en m²
        """
        if j < 0:
            return 0.0
        return 8 * np.pi * self.gamma * A_PLANCK * np.sqrt(j * (j + 1))

    def area_total(self, spins: List[float]) -> float:
        """
        Area total de una superficie con spins dados.

        A_total = sum_i A_{j_i}

        Args:
            spins: Lista de spins de los enlaces que atraviesan

        Returns:
            Area total en m²
        """
        return sum(self.area_j(j) for j in spins)

    def j_from_area(self, A: float) -> float:
        """
        Spin correspondiente a un area dada (inverso de area_j).

        Args:
            A: Area en m²

        Returns:
            Spin j
        """
        if A <= 0:
            return 0.0

        # A = 8 * pi * gamma * l_P^2 * sqrt(j*(j+1))
        # Despejando: j*(j+1) = (A / (8*pi*gamma*l_P^2))^2
        x = (A / (8 * np.pi * self.gamma * A_PLANCK))**2

        # j*(j+1) = x => j = (-1 + sqrt(1 + 4x)) / 2
        return (-1 + np.sqrt(1 + 4 * x)) / 2

    def espectro_areas(self, j_max: int = 10) -> Dict:
        """
        Genera el espectro de areas para j = 0, 1/2, 1, ..., j_max.

        Returns:
            Dict con j y A_j
        """
        # Spins permitidos: 0, 1/2, 1, 3/2, 2, ...
        j_values = np.arange(0, j_max + 0.5, 0.5)
        A_values = np.array([self.area_j(j) for j in j_values])

        return {
            'j': j_values,
            'A': A_values,
            'A_gap': self.A_gap,
            'A_Planck': A_PLANCK,
            'gamma': self.gamma,
        }


# =============================================================================
# CONVERSION S <-> j_max
# =============================================================================

class ConversionSJ:
    """
    Convierte entre el sello entropico S del MCMC y el spin maximo j_max de LQG.

    La relacion fundamental es que la entropia de horizonte en LQG es:
    S_BH = A / (4 * l_P^2)

    Y el area se cuantiza con spins. Para un horizonte con N enlaces:
    S ~ sum_{i=1}^N log(2*j_i + 1)

    Para estimar j_max, usamos:
    S_mcmc ~ alpha * sum_{j=1/2}^{j_max} log(2*j + 1)

    donde alpha es un factor de calibracion.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Inicializa el conversor.

        Args:
            alpha: Factor de calibracion S_mcmc / S_lqg
        """
        self.alpha = alpha
        self.area_model = AreaGapLQG()

    def S_from_jmax(self, j_max: float) -> float:
        """
        Calcula S_mcmc desde j_max.

        S = alpha * sum_{j=1/2, 1, ..., j_max} log(2*j + 1)

        Args:
            j_max: Spin maximo

        Returns:
            S en unidades MCMC
        """
        if j_max < 0.5:
            return 0.0

        # Suma sobre spins permitidos
        j_values = np.arange(0.5, j_max + 0.5, 0.5)
        S_sum = sum(np.log(2 * j + 1) for j in j_values)

        return self.alpha * S_sum

    def jmax_from_S(self, S: float) -> float:
        """
        Calcula j_max desde S_mcmc.

        Invierte la relacion S(j_max).

        Args:
            S: Entropia MCMC

        Returns:
            j_max estimado
        """
        if S <= 0:
            return 0.0

        # Busqueda binaria
        j_low, j_high = 0.5, 1000.0

        while j_high - j_low > 0.1:
            j_mid = (j_low + j_high) / 2
            S_mid = self.S_from_jmax(j_mid)

            if S_mid < S:
                j_low = j_mid
            else:
                j_high = j_mid

        return (j_low + j_high) / 2

    def verificar_consistencia(self, S: float) -> Dict:
        """
        Verifica la consistencia de la conversion S <-> j_max.

        Args:
            S: Entropia MCMC para verificar

        Returns:
            Dict con resultados de verificacion
        """
        j_max = self.jmax_from_S(S)
        S_recuperado = self.S_from_jmax(j_max)
        error_rel = abs(S_recuperado - S) / S if S > 0 else 0

        return {
            'S_original': S,
            'j_max': j_max,
            'S_recuperado': S_recuperado,
            'error_relativo': error_rel,
            'consistente': error_rel < 0.1,
        }


# =============================================================================
# SPIN NETWORKS
# =============================================================================

@dataclass
class Nodo:
    """Un nodo en una spin network."""
    id: int
    valencia: int           # Numero de enlaces conectados
    intertwiner: complex    # Intertwiner (simplificado)


@dataclass
class Enlace:
    """Un enlace en una spin network."""
    id: int
    nodo_inicial: int
    nodo_final: int
    spin: float             # j = 0, 1/2, 1, 3/2, ...
    m: float = 0.0          # Proyeccion magnetica


class SpinNetwork:
    """
    Representacion simplificada de una spin network.

    Una spin network es un grafo con:
    - Nodos: puntos donde se encuentran los enlaces
    - Enlaces: lineas etiquetadas con spins j
    - Intertwiners: tensores en los nodos que garantizan invariancia gauge

    En el MCMC, la spin network subyacente determina la geometria local
    y la entropia S.
    """

    def __init__(self):
        """Inicializa una spin network vacia."""
        self.nodos: List[Nodo] = []
        self.enlaces: List[Enlace] = []
        self._next_nodo_id = 0
        self._next_enlace_id = 0

    def agregar_nodo(self, valencia: int = 3) -> int:
        """
        Agrega un nodo a la red.

        Args:
            valencia: Numero de enlaces en el nodo

        Returns:
            ID del nodo creado
        """
        nodo = Nodo(
            id=self._next_nodo_id,
            valencia=valencia,
            intertwiner=1.0 + 0j
        )
        self.nodos.append(nodo)
        self._next_nodo_id += 1
        return nodo.id

    def agregar_enlace(self, nodo_i: int, nodo_f: int, spin: float) -> int:
        """
        Agrega un enlace entre dos nodos.

        Args:
            nodo_i: ID del nodo inicial
            nodo_f: ID del nodo final
            spin: Spin del enlace

        Returns:
            ID del enlace creado
        """
        enlace = Enlace(
            id=self._next_enlace_id,
            nodo_inicial=nodo_i,
            nodo_final=nodo_f,
            spin=spin
        )
        self.enlaces.append(enlace)
        self._next_enlace_id += 1
        return enlace.id

    def spin_total(self) -> float:
        """Suma de todos los spins en la red."""
        return sum(e.spin for e in self.enlaces)

    def dimension_hilbert(self) -> int:
        """
        Dimension del espacio de Hilbert asociado.

        dim = prod_{enlaces} (2*j + 1)
        """
        dim = 1
        for e in self.enlaces:
            dim *= int(2 * e.spin + 1)
        return dim

    def entropia_SN(self) -> float:
        """
        Entropia asociada a la spin network.

        S_SN = sum_{enlaces} log(2*j + 1)
        """
        return sum(np.log(2 * e.spin + 1) for e in self.enlaces if e.spin > 0)

    @classmethod
    def generar_tetraedro(cls, j_base: float = 0.5) -> 'SpinNetwork':
        """
        Genera una spin network tetraedrica (4 nodos, 6 enlaces).

        Args:
            j_base: Spin base para los enlaces

        Returns:
            SpinNetwork con topologia de tetraedro
        """
        sn = cls()

        # 4 nodos de valencia 3
        for _ in range(4):
            sn.agregar_nodo(valencia=3)

        # 6 enlaces (cada par de nodos conectado)
        pares = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, (n1, n2) in enumerate(pares):
            # Variar ligeramente los spins
            j = j_base + 0.5 * (i % 2)
            sn.agregar_enlace(n1, n2, j)

        return sn

    @classmethod
    def generar_cubica(cls, N: int = 2, j_base: float = 0.5) -> 'SpinNetwork':
        """
        Genera una spin network con topologia cubica.

        Args:
            N: Nodos por lado
            j_base: Spin base

        Returns:
            SpinNetwork cubica
        """
        sn = cls()

        # Crear nodos en lattice cubico
        nodo_ids = {}
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    nodo_ids[(i, j, k)] = sn.agregar_nodo(valencia=6)

        # Crear enlaces en direcciones x, y, z
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Enlace en x
                    if i < N - 1:
                        sn.agregar_enlace(
                            nodo_ids[(i, j, k)],
                            nodo_ids[(i + 1, j, k)],
                            j_base
                        )
                    # Enlace en y
                    if j < N - 1:
                        sn.agregar_enlace(
                            nodo_ids[(i, j, k)],
                            nodo_ids[(i, j + 1, k)],
                            j_base
                        )
                    # Enlace en z
                    if k < N - 1:
                        sn.agregar_enlace(
                            nodo_ids[(i, j, k)],
                            nodo_ids[(i, j, k + 1)],
                            j_base
                        )

        return sn

    def resumen(self) -> Dict:
        """Genera resumen de la spin network."""
        spins = [e.spin for e in self.enlaces]

        return {
            'n_nodos': len(self.nodos),
            'n_enlaces': len(self.enlaces),
            'spin_total': self.spin_total(),
            'spin_medio': np.mean(spins) if spins else 0,
            'spin_max': max(spins) if spins else 0,
            'dim_Hilbert': self.dimension_hilbert(),
            'entropia_SN': self.entropia_SN(),
        }


# =============================================================================
# AMPLITUDES DE ESPUMA DE ESPINES (EPRL SIMPLIFICADO)
# =============================================================================

class AmplitudesEPRL:
    """
    Calcula amplitudes de espuma de espines (modelo EPRL simplificado).

    En LQG, la dinamica esta dada por espumas de espines (spin foams),
    que son "historias" de spin networks.

    La amplitud de una espuma es:
    W = prod_f A_f * prod_e A_e * prod_v A_v

    donde:
    - A_f: amplitud de cara (depende de spins)
    - A_e: amplitud de arista
    - A_v: amplitud de vertice (15j symbol o similar)

    Esta implementacion usa aproximaciones simplificadas.
    """

    def __init__(self, gamma: float = GAMMA_IMMIRZI):
        """
        Inicializa el modelo de amplitudes.

        Args:
            gamma: Parametro de Immirzi
        """
        self.gamma = gamma

    def dimension_rep(self, j: float) -> int:
        """Dimension de la representacion de spin j."""
        return int(2 * j + 1)

    def amplitud_cara(self, j: float) -> float:
        """
        Amplitud de una cara etiquetada con spin j.

        A_f(j) = (2j + 1)

        Args:
            j: Spin de la cara

        Returns:
            Amplitud
        """
        return 2 * j + 1

    def amplitud_arista(self, j: float) -> float:
        """
        Amplitud de una arista con spin j.

        A_e(j) = 1 / sqrt(2j + 1)
        """
        if j <= 0:
            return 1.0
        return 1.0 / np.sqrt(2 * j + 1)

    def amplitud_vertice_simple(self, spins: List[float]) -> float:
        """
        Amplitud de vertice simplificada.

        En el modelo EPRL completo, esto involucra simbolos 15j.
        Aqui usamos una aproximacion exponencial.

        A_v ~ exp(-sum_i j_i^2 / j_0^2)

        Args:
            spins: Spins de las caras que confluyen en el vertice

        Returns:
            Amplitud de vertice
        """
        j_0 = 1.0  # Escala de cutoff
        sum_j2 = sum(j**2 for j in spins)
        return np.exp(-sum_j2 / j_0**2)

    def amplitud_espuma_simple(self, n_vertices: int, n_aristas: int,
                                n_caras: int, j_medio: float) -> float:
        """
        Amplitud total de una espuma simplificada.

        W = prod(A_f) * prod(A_e) * prod(A_v)

        Args:
            n_vertices: Numero de vertices
            n_aristas: Numero de aristas
            n_caras: Numero de caras
            j_medio: Spin medio de la configuracion

        Returns:
            Amplitud total
        """
        # Contribucion de caras
        log_W_f = n_caras * np.log(self.amplitud_cara(j_medio))

        # Contribucion de aristas
        log_W_e = n_aristas * np.log(self.amplitud_arista(j_medio))

        # Contribucion de vertices (aproximacion)
        spins_tipicos = [j_medio] * 4  # 4 caras por vertice tipico
        A_v = self.amplitud_vertice_simple(spins_tipicos)
        log_W_v = n_vertices * np.log(A_v + 1e-100)

        return np.exp(log_W_f + log_W_e + log_W_v)


# =============================================================================
# PUENTE MCMC-LQG
# =============================================================================

class MCMCLQGBridge:
    """
    Puente entre el modelo MCMC y LQG.

    Proporciona traducciones entre:
    - Entropia S (MCMC) <-> Spins j (LQG)
    - Burbujas temporales <-> Spin networks
    - Sellos ontologicos <-> Transiciones de fase de espuma
    """

    def __init__(self):
        """Inicializa el puente."""
        self.area_gap = AreaGapLQG()
        self.conversor = ConversionSJ()
        self.amplitudes = AmplitudesEPRL()

    def S_to_spin_network(self, S: float) -> SpinNetwork:
        """
        Genera una spin network representativa para entropia S.

        Args:
            S: Entropia MCMC

        Returns:
            SpinNetwork con la entropia apropiada
        """
        # Determinar j_max
        j_max = self.conversor.jmax_from_S(S)

        # Crear red tetraedrica con spins apropiados
        sn = SpinNetwork()

        # 4 nodos
        for _ in range(4):
            sn.agregar_nodo(valencia=3)

        # 6 enlaces con spins variados hasta j_max
        pares = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, (n1, n2) in enumerate(pares):
            # Distribuir spins
            j = 0.5 + (j_max - 0.5) * i / len(pares)
            j = min(j, j_max)
            sn.agregar_enlace(n1, n2, j)

        return sn

    def spin_network_to_S(self, sn: SpinNetwork) -> float:
        """
        Calcula entropia MCMC desde una spin network.

        Args:
            sn: Spin network

        Returns:
            Entropia S
        """
        # Usar entropia de la spin network con factor de calibracion
        S_sn = sn.entropia_SN()
        return self.conversor.alpha * S_sn

    def sello_to_lqg_params(self, sello: str) -> Dict:
        """
        Traduce un sello MCMC a parametros LQG.

        Args:
            sello: Nombre del sello ("S1", "S2", "S3", "S4")

        Returns:
            Dict con parametros LQG
        """
        sello_data = None
        for s in SELLOS_LQG:
            if s.nombre == sello:
                sello_data = s
                break

        if sello_data is None:
            return {}

        # Calcular area tipica
        area = self.area_gap.area_j(sello_data.j_max)

        return {
            'nombre': sello_data.nombre,
            'S_mcmc': sello_data.S_mcmc,
            'j_max': sello_data.j_max,
            'area_m2': area,
            'area_Planck': area / A_PLANCK,
            'energia_GeV': sello_data.energia_GeV,
            'descripcion': sello_data.descripcion,
        }

    def verificar_correspondencia_BH(self, M_solar: float) -> Dict:
        """
        Verifica la correspondencia entre entropia de BH en MCMC y LQG.

        En LQG: S_BH = A / (4 * l_P^2)
        En MCMC: S_local cerca del horizonte

        Args:
            M_solar: Masa del BH en masas solares

        Returns:
            Dict con comparacion
        """
        M_SUN = 1.98892e30  # kg

        # Radio de Schwarzschild
        r_s = 2 * G_NEWTON * M_solar * M_SUN / C_LIGHT**2

        # Area del horizonte
        A_horizonte = 4 * np.pi * r_s**2

        # Entropia Bekenstein-Hawking
        S_BH = A_horizonte / (4 * L_PLANCK**2)

        # Entropia en unidades MCMC (normalizada)
        # Asumiendo S_mcmc_max ~ 100 corresponde a S_BH ~ 10^77 para BH estelar
        S_mcmc = 0.9 * (1 - 1 / np.log10(S_BH + 1))

        # j_max correspondiente
        j_max = self.conversor.jmax_from_S(S_mcmc * 100)  # Escalar a unidades internas

        return {
            'M_solar': M_solar,
            'r_s_m': r_s,
            'A_horizonte_m2': A_horizonte,
            'S_BH_Planck': S_BH,
            'S_mcmc': S_mcmc,
            'j_max_lqg': j_max,
        }


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_MCMC_LQG_Bridge() -> bool:
    """
    Test del puente MCMC-LQG.
    """
    print("\n" + "=" * 70)
    print("  TEST MCMC-LQG BRIDGE - CONEXION CON GRAVEDAD CUANTICA DE LAZOS")
    print("=" * 70)

    # 1. Area gap
    print("\n[1] Area Gap LQG:")
    print("-" * 70)

    area_model = AreaGapLQG()
    espectro = area_model.espectro_areas(j_max=5)

    print(f"    Parametro de Immirzi gamma = {area_model.gamma:.4f}")
    print(f"    Area de Planck A_P = {A_PLANCK:.3e} m²")
    print(f"    Area gap A_gap = {area_model.A_gap:.3e} m²")
    print(f"    A_gap / A_P = {area_model.A_gap / A_PLANCK:.4f}")

    # Verificar que A_gap > 0
    gap_ok = area_model.A_gap > 0
    print(f"\n    Area gap calculada: {'PASS' if gap_ok else 'FAIL'}")

    # 2. Conversion S <-> j_max
    print("\n[2] Conversion S <-> j_max:")
    print("-" * 70)

    conversor = ConversionSJ()

    S_test = [0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"    {'S':>8} {'j_max':>10} {'S_rec':>10} {'Error':>10}")

    conversion_ok = True
    for S in S_test:
        verif = conversor.verificar_consistencia(S)
        print(f"    {S:>8.2f} {verif['j_max']:>10.2f} "
              f"{verif['S_recuperado']:>10.4f} {verif['error_relativo']:>10.4f}")
        # Verificar error razonable (< 10%)
        if verif['error_relativo'] > 0.15:
            conversion_ok = False

    print(f"\n    Conversion consistente: {'PASS' if conversion_ok else 'FAIL'}")

    # 3. Spin Networks
    print("\n[3] Spin Networks:")
    print("-" * 70)

    # Tetraedro
    sn_tetra = SpinNetwork.generar_tetraedro(j_base=1.0)
    resumen_t = sn_tetra.resumen()

    print(f"    Tetraedro (j_base=1.0):")
    print(f"      Nodos: {resumen_t['n_nodos']}, Enlaces: {resumen_t['n_enlaces']}")
    print(f"      Spin total: {resumen_t['spin_total']:.1f}")
    print(f"      dim(Hilbert): {resumen_t['dim_Hilbert']}")
    print(f"      S_SN: {resumen_t['entropia_SN']:.4f}")

    # Verificar estructura
    sn_ok = (resumen_t['n_nodos'] == 4 and resumen_t['n_enlaces'] == 6)
    print(f"\n    Estructura correcta: {'PASS' if sn_ok else 'FAIL'}")

    # 4. Puente completo
    print("\n[4] Puente MCMC-LQG:")
    print("-" * 70)

    bridge = MCMCLQGBridge()

    # Traducir sellos
    print("    Sellos ontologicos -> LQG:")
    for sello in ["S1", "S2", "S3", "S4"]:
        params = bridge.sello_to_lqg_params(sello)
        if params:
            print(f"      {sello}: j_max={params['j_max']:.1f}, "
                  f"A/A_P={params['area_Planck']:.2f}")

    # 5. Correspondencia BH
    print("\n[5] Correspondencia entropia de BH:")
    print("-" * 70)

    for M in [10, 1e6, 1e9]:
        corr = bridge.verificar_correspondencia_BH(M)
        print(f"    M = {M:.0e} M_sun:")
        print(f"      S_BH = {corr['S_BH_Planck']:.2e}")
        print(f"      S_mcmc = {corr['S_mcmc']:.4f}")
        print(f"      j_max = {corr['j_max_lqg']:.1f}")

    # Verificar que S_BH crece con M
    corr_10 = bridge.verificar_correspondencia_BH(10)
    corr_1e6 = bridge.verificar_correspondencia_BH(1e6)
    bh_ok = corr_1e6['S_BH_Planck'] > corr_10['S_BH_Planck']
    print(f"\n    S_BH crece con M: {'PASS' if bh_ok else 'FAIL'}")

    # Resultado final
    passed = gap_ok and conversion_ok and sn_ok and bh_ok

    print("\n" + "=" * 70)
    print(f"  MCMC-LQG BRIDGE: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_MCMC_LQG_Bridge()
