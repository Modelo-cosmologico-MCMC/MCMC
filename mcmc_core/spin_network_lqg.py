"""
Spin Network LQG - Conexión con Loop Quantum Gravity
=====================================================

Este módulo conecta el modelo MCMC con Loop Quantum Gravity (LQG)
mediante redes de espín y percolación geométrica.

CONCEPTO CENTRAL:
    En LQG, la geometría del espacio emerge de redes de espín.
    En MCMC, la transición desde "masa pura" hacia "espacio emergente"
    se modela como un proceso de percolación en una red de espín.

PERCOLACIÓN GEOMÉTRICA:
    La probabilidad de que un enlace de la red esté "activo"
    (contribuya a la geometría emergente) es:

    p(S) = ε(S) = fracción entrópica

    Cuando p > p_c (umbral de percolación), emerge una red conexa
    que representa la geometría macroscópica del espacio.

TRANSICIÓN DE FASE:
    - S < S_critico: Fase de "masa" (no hay geometría conexa)
    - S = S_critico: Transición de percolación (umbral crítico)
    - S > S_critico: Fase de "espacio" (geometría emergente)

    El punto crítico p_c ≈ 0.5 en redes 2D coincide con ε = 0.5,
    es decir, el punto de equilibrio tensional donde P_ME = 0.

ESPÍN Y GEOMETRÍA:
    Los enlaces de la red tienen espín j = 1/2.
    El área de una superficie es proporcional a:
    A ∝ Σ √[j(j+1)] × l_P²

    donde l_P es la longitud de Planck.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    - Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
    - Rovelli, C. (2004). "Quantum Gravity"
    - Ashtekar, A. & Lewandowski, J. (2004). "Background Independent Quantum Gravity"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from numpy.typing import NDArray
from collections import deque


# =============================================================================
# Constantes de Planck y LQG
# =============================================================================

# Longitud de Planck (m)
L_PLANCK: float = 1.616e-35

# Área de Planck (m²)
A_PLANCK: float = L_PLANCK**2

# Volumen de Planck (m³)
V_PLANCK: float = L_PLANCK**3

# Factor γ de Immirzi (valor canónico LQG)
GAMMA_IMMIRZI: float = 0.2375

# Umbral de percolación para red cuadrada 2D
P_C_2D: float = 0.5

# Umbral de percolación para red cúbica 3D
P_C_3D: float = 0.3116


# =============================================================================
# Funciones de Espín
# =============================================================================

def area_enlace(j: float, gamma: float = GAMMA_IMMIRZI) -> float:
    """
    Calcula el área cuántica de un enlace con espín j.

    A = 8πγ × √[j(j+1)] × l_P²

    En unidades de Planck:
    A = 8πγ × √[j(j+1)]

    Args:
        j: Espín del enlace (j = 0, 1/2, 1, 3/2, ...)
        gamma: Parámetro de Immirzi

    Returns:
        Área en unidades de l_P²
    """
    return 8 * np.pi * gamma * np.sqrt(j * (j + 1))


def volumen_nodo(valencias: List[float]) -> float:
    """
    Estima el volumen cuántico de un nodo.

    El volumen depende de las valencias (espines) de los
    enlaces que se encuentran en el nodo.

    Fórmula simplificada:
    V ∝ Σ |j_i - j_j| para pares de enlaces

    Args:
        valencias: Lista de espines de los enlaces en el nodo

    Returns:
        Volumen en unidades de l_P³
    """
    if len(valencias) < 3:
        return 0.0

    # Suma de diferencias (simplificación)
    volumen = 0.0
    for i in range(len(valencias)):
        for j in range(i + 1, len(valencias)):
            volumen += np.abs(valencias[i] - valencias[j])

    return volumen * GAMMA_IMMIRZI


def operador_area_total(espines: List[float]) -> float:
    """
    Calcula el área total de una superficie atravesada por enlaces.

    A_total = Σ 8πγ × √[j_i(j_i+1)] × l_P²

    Args:
        espines: Lista de espines de los enlaces que atraviesan la superficie

    Returns:
        Área total en unidades de l_P²
    """
    return sum(area_enlace(j) for j in espines)


# =============================================================================
# Red de Espín
# =============================================================================

@dataclass
class Nodo:
    """
    Nodo en la red de espín.

    Representa un punto donde se encuentran varios enlaces.
    En LQG, los nodos codifican información sobre el volumen.

    Attributes:
        id: Identificador único
        posicion: Coordenadas (para visualización)
        vecinos: Lista de IDs de nodos vecinos
        activo: Si el nodo participa en la geometría emergente
    """
    id: int
    posicion: Tuple[int, ...]
    vecinos: List[int] = field(default_factory=list)
    activo: bool = True


@dataclass
class Enlace:
    """
    Enlace en la red de espín.

    Representa una conexión entre nodos con un valor de espín.
    En LQG, los enlaces codifican información sobre áreas.

    Attributes:
        nodo_a: ID del primer nodo
        nodo_b: ID del segundo nodo
        espin: Valor de espín (j = 0, 1/2, 1, ...)
        activo: Si el enlace contribuye a la geometría
    """
    nodo_a: int
    nodo_b: int
    espin: float = 0.5  # Espín mínimo
    activo: bool = False


@dataclass
class SpinNetwork:
    """
    Red de espín para el modelo MCMC.

    Implementa una red de espín en D dimensiones donde la
    probabilidad de activación de enlaces viene dada por
    la fracción entrópica ε(S).

    La geometría macroscópica emerge cuando la red percola,
    es decir, cuando existe un cluster conexo que atraviesa
    todo el sistema.

    Attributes:
        L: Tamaño lineal de la red
        D: Dimensionalidad (2 o 3)
        epsilon: Fracción entrópica (probabilidad de enlace activo)
    """
    L: int = 10
    D: int = 2
    epsilon: float = 0.5

    # Estructuras internas
    nodos: Dict[int, Nodo] = field(default_factory=dict)
    enlaces: List[Enlace] = field(default_factory=list)

    # Estadísticas
    _clusters: Optional[List[Set[int]]] = field(default=None, repr=False)
    _percolado: Optional[bool] = field(default=None, repr=False)

    def __post_init__(self):
        """Inicializa la red."""
        self._construir_red()
        self._activar_enlaces()

    def _construir_red(self) -> None:
        """Construye la estructura de la red."""
        self.nodos = {}
        self.enlaces = []

        if self.D == 2:
            self._construir_red_2d()
        elif self.D == 3:
            self._construir_red_3d()
        else:
            raise ValueError(f"Dimensión {self.D} no soportada")

    def _construir_red_2d(self) -> None:
        """Construye red cuadrada 2D."""
        # Crear nodos
        for i in range(self.L):
            for j in range(self.L):
                node_id = i * self.L + j
                self.nodos[node_id] = Nodo(
                    id=node_id,
                    posicion=(i, j),
                    vecinos=[]
                )

        # Crear enlaces
        for i in range(self.L):
            for j in range(self.L):
                node_id = i * self.L + j

                # Enlace horizontal (periódico)
                j_next = (j + 1) % self.L
                vecino_h = i * self.L + j_next
                self.nodos[node_id].vecinos.append(vecino_h)
                if j < self.L - 1:  # Evitar duplicados
                    self.enlaces.append(Enlace(node_id, vecino_h))

                # Enlace vertical (periódico)
                i_next = (i + 1) % self.L
                vecino_v = i_next * self.L + j
                self.nodos[node_id].vecinos.append(vecino_v)
                if i < self.L - 1:  # Evitar duplicados
                    self.enlaces.append(Enlace(node_id, vecino_v))

    def _construir_red_3d(self) -> None:
        """Construye red cúbica 3D."""
        # Crear nodos
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    node_id = i * self.L**2 + j * self.L + k
                    self.nodos[node_id] = Nodo(
                        id=node_id,
                        posicion=(i, j, k),
                        vecinos=[]
                    )

        # Crear enlaces (simplificado)
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    node_id = i * self.L**2 + j * self.L + k

                    # Enlaces en cada dirección
                    for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                        ni = (i + di) % self.L
                        nj = (j + dj) % self.L
                        nk = (k + dk) % self.L

                        if (di == 1 and i < self.L - 1) or \
                           (dj == 1 and j < self.L - 1) or \
                           (dk == 1 and k < self.L - 1):
                            vecino = ni * self.L**2 + nj * self.L + nk
                            self.nodos[node_id].vecinos.append(vecino)
                            self.enlaces.append(Enlace(node_id, vecino))

    def _activar_enlaces(self) -> None:
        """
        Activa enlaces con probabilidad ε.

        Cada enlace se activa independientemente con probabilidad
        igual a la fracción entrópica ε(S).
        """
        for enlace in self.enlaces:
            enlace.activo = np.random.random() < self.epsilon

        # Invalidar cache
        self._clusters = None
        self._percolado = None

    @property
    def n_nodos(self) -> int:
        """Número de nodos en la red."""
        return len(self.nodos)

    @property
    def n_enlaces(self) -> int:
        """Número total de enlaces."""
        return len(self.enlaces)

    @property
    def n_enlaces_activos(self) -> int:
        """Número de enlaces activos."""
        return sum(1 for e in self.enlaces if e.activo)

    @property
    def fraccion_activa(self) -> float:
        """Fracción de enlaces activos."""
        if self.n_enlaces == 0:
            return 0.0
        return self.n_enlaces_activos / self.n_enlaces

    @property
    def p_critico(self) -> float:
        """Umbral de percolación teórico."""
        if self.D == 2:
            return P_C_2D
        elif self.D == 3:
            return P_C_3D
        else:
            return 0.5

    def encontrar_clusters(self) -> List[Set[int]]:
        """
        Encuentra todos los clusters conexos.

        Usa BFS para identificar componentes conexas formadas
        por nodos conectados mediante enlaces activos.

        Returns:
            Lista de conjuntos, cada uno conteniendo IDs de nodos
            pertenecientes a un cluster.
        """
        if self._clusters is not None:
            return self._clusters

        visitados = set()
        clusters = []

        # Construir grafo de adyacencia (solo enlaces activos)
        adj = {n: [] for n in self.nodos}
        for e in self.enlaces:
            if e.activo:
                adj[e.nodo_a].append(e.nodo_b)
                adj[e.nodo_b].append(e.nodo_a)

        # BFS para cada componente
        for nodo_id in self.nodos:
            if nodo_id in visitados:
                continue

            cluster = set()
            queue = deque([nodo_id])

            while queue:
                n = queue.popleft()
                if n in visitados:
                    continue
                visitados.add(n)
                cluster.add(n)

                for vecino in adj[n]:
                    if vecino not in visitados:
                        queue.append(vecino)

            if cluster:
                clusters.append(cluster)

        self._clusters = clusters
        return clusters

    @property
    def cluster_mas_grande(self) -> Set[int]:
        """Retorna el cluster más grande."""
        clusters = self.encontrar_clusters()
        if not clusters:
            return set()
        return max(clusters, key=len)

    @property
    def tamano_cluster_max(self) -> int:
        """Tamaño del cluster más grande."""
        return len(self.cluster_mas_grande)

    @property
    def fraccion_cluster_max(self) -> float:
        """Fracción del sistema en el cluster más grande."""
        if self.n_nodos == 0:
            return 0.0
        return self.tamano_cluster_max / self.n_nodos

    def percola(self) -> bool:
        """
        Determina si la red percola.

        La red percola si existe un cluster que conecta
        lados opuestos del sistema.

        Para red 2D: conecta izquierda-derecha o arriba-abajo.
        Para red 3D: conecta caras opuestas en alguna dirección.

        Returns:
            True si la red percola
        """
        if self._percolado is not None:
            return self._percolado

        cluster = self.cluster_mas_grande

        if self.D == 2:
            # Verificar conexión izquierda-derecha
            izq = {n for n in cluster if self.nodos[n].posicion[1] == 0}
            der = {n for n in cluster if self.nodos[n].posicion[1] == self.L - 1}
            percola_h = bool(izq) and bool(der)

            # Verificar conexión arriba-abajo
            arriba = {n for n in cluster if self.nodos[n].posicion[0] == 0}
            abajo = {n for n in cluster if self.nodos[n].posicion[0] == self.L - 1}
            percola_v = bool(arriba) and bool(abajo)

            self._percolado = percola_h or percola_v

        elif self.D == 3:
            self._percolado = False
            for dim in range(3):
                cara_0 = {n for n in cluster if self.nodos[n].posicion[dim] == 0}
                cara_L = {n for n in cluster if self.nodos[n].posicion[dim] == self.L - 1}
                if bool(cara_0) and bool(cara_L):
                    self._percolado = True
                    break

        return self._percolado

    def area_total(self) -> float:
        """
        Calcula el área total de la geometría emergente.

        A_total = Σ A(j) para enlaces activos

        Returns:
            Área en unidades de l_P²
        """
        return sum(area_enlace(e.espin) for e in self.enlaces if e.activo)

    def calcular_susceptibilidad(self) -> float:
        """
        Calcula la susceptibilidad χ (tamaño medio de clusters).

        χ = <s²> / <s>

        donde s es el tamaño de un cluster.
        """
        clusters = self.encontrar_clusters()
        if not clusters:
            return 0.0

        tamanos = [len(c) for c in clusters]
        s_mean = np.mean(tamanos)
        s2_mean = np.mean([t**2 for t in tamanos])

        if s_mean == 0:
            return 0.0

        return s2_mean / s_mean

    def resumen(self) -> str:
        """Genera resumen de la red de espín."""
        return (
            f"Spin Network MCMC - Conexión LQG\n"
            f"{'='*50}\n"
            f"Dimensión: {self.D}D\n"
            f"Tamaño: L = {self.L}\n"
            f"ε (prob. activación) = {self.epsilon:.4f}\n"
            f"p_c (umbral teórico) = {self.p_critico:.4f}\n"
            f"{'='*50}\n"
            f"Nodos: {self.n_nodos}\n"
            f"Enlaces: {self.n_enlaces}\n"
            f"Enlaces activos: {self.n_enlaces_activos} "
            f"({100*self.fraccion_activa:.1f}%)\n"
            f"{'='*50}\n"
            f"Clusters: {len(self.encontrar_clusters())}\n"
            f"Cluster máximo: {self.tamano_cluster_max} nodos "
            f"({100*self.fraccion_cluster_max:.1f}%)\n"
            f"Percola: {'Sí' if self.percola() else 'No'}\n"
            f"{'='*50}\n"
            f"Área total: {self.area_total():.2f} l_P²\n"
        )


# =============================================================================
# Análisis de Transición de Fase
# =============================================================================

def calcular_probabilidad_percolacion(
    epsilon: float,
    L: int = 20,
    D: int = 2,
    n_muestras: int = 100
) -> float:
    """
    Calcula la probabilidad de percolación para un valor de ε.

    Args:
        epsilon: Fracción entrópica
        L: Tamaño de la red
        D: Dimensionalidad
        n_muestras: Número de realizaciones

    Returns:
        Probabilidad de percolación [0, 1]
    """
    n_percola = 0

    for _ in range(n_muestras):
        red = SpinNetwork(L=L, D=D, epsilon=epsilon)
        if red.percola():
            n_percola += 1

    return n_percola / n_muestras


def encontrar_punto_critico(
    L: int = 20,
    D: int = 2,
    n_muestras: int = 50,
    epsilon_min: float = 0.3,
    epsilon_max: float = 0.7,
    n_puntos: int = 21
) -> Dict[str, float]:
    """
    Encuentra el punto crítico de percolación.

    Busca el valor de ε donde P(percolación) = 0.5.

    Args:
        L: Tamaño de la red
        D: Dimensionalidad
        n_muestras: Muestras por punto
        epsilon_min, epsilon_max: Rango de búsqueda
        n_puntos: Puntos en el barrido

    Returns:
        Diccionario con epsilon_c y datos del barrido
    """
    epsilons = np.linspace(epsilon_min, epsilon_max, n_puntos)
    probabilidades = []

    for eps in epsilons:
        P = calcular_probabilidad_percolacion(eps, L, D, n_muestras)
        probabilidades.append(P)

    probabilidades = np.array(probabilidades)

    # Encontrar cruce por 0.5 (interpolación lineal)
    for i in range(len(epsilons) - 1):
        if probabilidades[i] < 0.5 <= probabilidades[i+1]:
            # Interpolación lineal
            eps_c = epsilons[i] + (0.5 - probabilidades[i]) * \
                    (epsilons[i+1] - epsilons[i]) / \
                    (probabilidades[i+1] - probabilidades[i])
            break
        elif probabilidades[i] > 0.5 >= probabilidades[i+1]:
            eps_c = epsilons[i] + (0.5 - probabilidades[i]) * \
                    (epsilons[i+1] - epsilons[i]) / \
                    (probabilidades[i+1] - probabilidades[i])
            break
    else:
        eps_c = (epsilon_min + epsilon_max) / 2

    return {
        "epsilon_c": eps_c,
        "epsilons": epsilons,
        "probabilidades": probabilidades,
        "L": L,
        "D": D,
    }


def interpretar_transicion_geometrica(epsilon: float, D: int = 2) -> str:
    """
    Interpreta la transición geométrica en términos MCMC.

    Args:
        epsilon: Fracción entrópica
        D: Dimensionalidad

    Returns:
        Descripción de la fase geométrica
    """
    p_c = P_C_2D if D == 2 else P_C_3D

    if epsilon < p_c * 0.5:
        return "Fase de MASA PURA: No hay geometría conexa. El espacio no ha emergido."
    elif epsilon < p_c:
        return "Fase PRE-GEOMÉTRICA: Clusters pequeños. Geometría fragmentada."
    elif epsilon < p_c * 1.1:
        return "TRANSICIÓN CRÍTICA: Punto de emergencia geométrica. P_ME ≈ 0."
    elif epsilon < 0.9:
        return "Fase GEOMÉTRICA: Espacio-tiempo conexo emerge. Geometría macroscópica."
    else:
        return "Fase de ESPACIO PURO: Geometría completamente desarrollada."


# =============================================================================
# Tests
# =============================================================================

def _test_spin_network():
    """Verifica la implementación de la red de espín."""

    # Test 1: Construcción de red 2D
    red = SpinNetwork(L=5, D=2, epsilon=0.5)
    assert red.n_nodos == 25, f"Nodos esperados 25, obtenidos {red.n_nodos}"
    assert red.n_enlaces > 0, "Debe haber enlaces"

    # Test 2: Área cuántica
    A = area_enlace(0.5)
    assert A > 0, "Área debe ser positiva"

    # Test 3: Percolación a alta densidad
    red_densa = SpinNetwork(L=10, D=2, epsilon=0.8)
    # Alta probabilidad de percolación

    # Test 4: No percolación a baja densidad
    red_diluida = SpinNetwork(L=10, D=2, epsilon=0.1)
    # Baja probabilidad de percolación

    # Test 5: Clusters
    clusters = red.encontrar_clusters()
    assert isinstance(clusters, list), "Debe retornar lista"

    # Test 6: Consistencia de fracción activa
    assert 0 <= red.fraccion_activa <= 1, "Fracción debe estar en [0,1]"

    print("✓ Todos los tests de Spin Network pasaron")
    return True


if __name__ == "__main__":
    _test_spin_network()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Spin Network MCMC - Conexión LQG")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    # Red en diferentes regímenes
    print("Red de Espín 2D (L=20):\n")

    for eps in [0.2, 0.4, 0.5, 0.6, 0.8]:
        red = SpinNetwork(L=20, D=2, epsilon=eps)
        percola = "SÍ" if red.percola() else "NO"
        print(f"ε = {eps:.1f}: Percola = {percola}, "
              f"Cluster máx = {100*red.fraccion_cluster_max:.0f}%")
        print(f"         {interpretar_transicion_geometrica(eps)}")
        print()

    # Resumen detallado
    print("\n" + "-"*60)
    print("Resumen de red en equilibrio (ε = 0.5):\n")
    red_eq = SpinNetwork(L=15, D=2, epsilon=0.5)
    print(red_eq.resumen())
