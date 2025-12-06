"""
Bloque 0 - Estado Primordial S₀
================================

Define el estado inicial del universo antes del primer colapso,
fundamentado en la DUALIDAD TENSIONAL MASA-ESPACIO.

En el estado primordial S₀:
    - Mp0 = 1.0 (masa potencial normalizada)
    - Ep0 = 1e-10 (energía primordial mínima)
    - Φ_ten = Mp0/Ep0 = 10¹⁰ (Campo de Adrián - tensión máxima)
    - P_ME(S0) ≈ +1.0 (la masa domina completamente)
    - k_pre = 6.307 (tasa de colapso pregeométrica)

ONTOLOGÍA TENSIONAL:
    La dualidad MASA-ESPACIO constituye el sustrato ontológico primordial.
    El Campo de Adrián (Φ_ten) captura la tensión entre estos dos aspectos
    complementarios de la realidad física.

    La evolución cósmica se describe como el flujo desde:
    - Estado inicial: Φ_ten → ∞ (masa pura, S=S₀)
    - Estado final: Φ_ten → 0 (espacio puro, S=S₄)

CAMPO DE ADRIÁN:
    Φ_ten(S) = Mp(S)/Ep(S)

    Este campo tensorial describe cómo la masa potencial primordial
    se transforma progresivamente en energía-espacio a través de
    los colapsos entrópicos sucesivos.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray


# =============================================================================
# Constantes Fundamentales del Modelo MCMC
# =============================================================================

# Masa potencial primordial (normalizada a 1)
Mp0: float = 1.0

# Energía primordial (prácticamente cero, pero no exactamente)
Ep0: float = 1e-10

# Tasa de colapso pregeométrica
K_PRE: float = 6.307

# Campo de Adrián inicial (tensión máxima)
PHI_TEN_0: float = Mp0 / Ep0  # = 10^10

# Sellos entrópicos: puntos de transición en la evolución
# Representan los "momentos" ontológicos de la evolución cósmica
SELLOS: Dict[str, float] = {
    "S0": 0.000,  # Estado primordial - masa pura
    "S1": 0.010,  # Primera transición - inicio de colapsos
    "S2": 0.100,  # Segunda transición - estructura emergente
    "S3": 1.000,  # Tercera transición (era actual aproximada)
    "S4": 1.001,  # Estado final - energía domina
}

# Lista ordenada de sellos para iteración
SELLOS_ORDEN: List[str] = ["S0", "S1", "S2", "S3", "S4"]

# Nombres descriptivos de los sellos
NOMBRES_SELLOS: Dict[str, str] = {
    "S0": "Estado Primordial",
    "S1": "Primera Transición",
    "S2": "Segunda Transición",
    "S3": "Era Actual",
    "S4": "Estado Final",
}


# =============================================================================
# Funciones Fundamentales del Campo Tensional
# =============================================================================

def calcular_Phi_ten(Mp: float, Ep: float) -> float:
    """
    Calcula el Campo de Adrián (Φ_ten).

    Φ_ten = Mp / Ep

    Este campo tensorial es la cantidad fundamental que describe
    la dualidad MASA-ESPACIO en el modelo MCMC. Representa la
    "tensión ontológica" entre los dos aspectos primordiales.

    Propiedades:
        - Φ_ten → ∞ cuando Ep → 0 (masa pura, estado primordial)
        - Φ_ten = 1 cuando Mp = Ep (equilibrio tensional)
        - Φ_ten → 0 cuando Mp → 0 (espacio puro, estado final)

    El Campo de Adrián es equivalente a la tensión masa-energía
    pero enfatiza su carácter de campo ontológico fundamental.

    Args:
        Mp: Masa potencial
        Ep: Energía (espacio)

    Returns:
        Φ_ten: Campo de Adrián (adimensional)
    """
    if Ep <= 0:
        raise ValueError("La energía debe ser positiva")
    return Mp / Ep


def calcular_tension(Mp: float, Ep: float) -> float:
    """
    Calcula la tensión masa-energía (alias del Campo de Adrián).

    Tensión = Φ_ten = Mp / Ep

    La tensión representa el desequilibrio entre masa potencial
    y energía liberada. En el estado primordial es máxima.

    Args:
        Mp: Masa potencial
        Ep: Energía

    Returns:
        Tensión (adimensional)
    """
    return calcular_Phi_ten(Mp, Ep)


def calcular_P_ME(Mp: float, Ep: float) -> float:
    """
    Calcula la polarización masa-espacio P_ME.

    P_ME = (Mp - Ep) / (Mp + Ep)

    Esta es la forma normalizada del Campo de Adrián, acotada en [-1, +1].
    Representa la "polarización" del campo tensional.

    Propiedades:
        - P_ME → +1 cuando Mp >> Ep (masa domina, S → S₀)
        - P_ME → -1 cuando Ep >> Mp (espacio domina, S → S₄)
        - P_ME = 0 cuando Mp = Ep (equilibrio tensional)

    Relación con Φ_ten:
        P_ME = (Φ_ten - 1) / (Φ_ten + 1)
        Φ_ten = (1 + P_ME) / (1 - P_ME)

    Esta es la variable fundamental que describe la evolución
    del universo desde +1 (primordial) hacia -1 (futuro).

    Args:
        Mp: Masa potencial
        Ep: Energía (espacio)

    Returns:
        Polarización P_ME en [-1, +1]
    """
    total = Mp + Ep
    if total <= 0:
        raise ValueError("La suma Mp + Ep debe ser positiva")
    return (Mp - Ep) / total


def Phi_ten_desde_P_ME(P_ME: float) -> float:
    """
    Calcula Φ_ten a partir de P_ME.

    Φ_ten = (1 + P_ME) / (1 - P_ME)

    Args:
        P_ME: Polarización masa-espacio en [-1, +1)

    Returns:
        Campo de Adrián Φ_ten
    """
    if P_ME >= 1.0:
        return np.inf
    if P_ME <= -1.0:
        return 0.0
    return (1 + P_ME) / (1 - P_ME)


# =============================================================================
# Clase Principal
# =============================================================================

@dataclass
class EstadoPrimordial:
    """
    Representa el estado primordial del universo (Sello S₀).

    En el estado primordial:
        - La masa potencial es máxima (Mp = Mp0 = 1.0)
        - La energía es mínima (Ep = Ep0 ≈ 10⁻¹⁰)
        - El Campo de Adrián es máximo (Φ_ten ≈ 10¹⁰)
        - La polarización es máxima (P_ME ≈ +1)
        - La tasa de colapso pregeométrica es k_pre = 6.307

    DUALIDAD TENSIONAL MASA-ESPACIO:
        El estado primordial representa el polo "masa pura" de la
        dualidad ontológica fundamental. El universo evoluciona
        desde aquí hacia el polo "espacio puro" a través de los
        colapsos entrópicos sucesivos.

    CAMPO DE ADRIÁN:
        Φ_ten = Mp/Ep captura la tensión entre masa y espacio.
        En S₀: Φ_ten → ∞ (masa domina completamente)
        En S₄: Φ_ten → 0 (espacio domina completamente)

    Attributes:
        Mp: Masa potencial actual
        Ep: Energía (espacio) actual
        sello: Sello entrópico actual
        k_pre: Tasa de colapso pregeométrica
    """
    Mp: float = Mp0
    Ep: float = Ep0
    sello: str = "S0"
    k_pre: float = K_PRE

    # Historial de evolución
    historial: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        """Valida el estado inicial y registra en historial."""
        if self.Mp < 0:
            raise ValueError("Mp debe ser no negativa")
        if self.Ep <= 0:
            raise ValueError("Ep debe ser positiva")

        # Registrar estado inicial
        self._registrar_estado()

    def _registrar_estado(self) -> None:
        """Registra el estado actual en el historial."""
        self.historial.append({
            "sello": self.sello,
            "nombre": self.nombre_sello,
            "S": SELLOS.get(self.sello, 0.0),
            "Mp": self.Mp,
            "Ep": self.Ep,
            "Phi_ten": self.Phi_ten,
            "P_ME": self.P_ME,
            "k_pre": self.k_pre,
        })

    @property
    def Phi_ten(self) -> float:
        """
        Campo de Adrián (tensión ontológica).

        Φ_ten = Mp / Ep

        Este es el campo fundamental que describe la dualidad
        MASA-ESPACIO en el modelo MCMC.
        """
        return calcular_Phi_ten(self.Mp, self.Ep)

    @property
    def tension(self) -> float:
        """Tensión actual del sistema (alias de Phi_ten)."""
        return self.Phi_ten

    @property
    def P_ME(self) -> float:
        """
        Polarización masa-espacio actual.

        P_ME = (Mp - Ep) / (Mp + Ep)

        Forma normalizada del Campo de Adrián en [-1, +1].
        """
        return calcular_P_ME(self.Mp, self.Ep)

    @property
    def S(self) -> float:
        """Valor entrópico del sello actual."""
        return SELLOS.get(self.sello, 0.0)

    @property
    def energia_total(self) -> float:
        """Energía total conservada: E_total = Mp + Ep = Mp0 + Ep0."""
        return self.Mp + self.Ep

    @property
    def nombre_sello(self) -> str:
        """Nombre descriptivo del sello actual."""
        return NOMBRES_SELLOS.get(self.sello, "Desconocido")

    def evolucionar_a_sello(self, nuevo_sello: str, epsilon: float) -> None:
        """
        Evoluciona el sistema a un nuevo sello entrópico.

        Durante la evolución:
            - Mp disminuye: Mp(S) = Mp0 × (1 - ε)
            - Ep aumenta: Ep(S) = Mp0 × ε + Ep0
            - La energía total se conserva (aproximadamente)

        Args:
            nuevo_sello: Sello destino ("S1", "S2", "S3", "S4")
            epsilon: Fracción de energía convertida ε ∈ [0, 1]
        """
        if nuevo_sello not in SELLOS:
            raise ValueError(f"Sello {nuevo_sello} no válido")

        if epsilon < 0 or epsilon > 1:
            raise ValueError(f"epsilon debe estar en [0, 1], recibido: {epsilon}")

        # Calcular nuevos valores
        self.Mp = Mp0 * (1 - epsilon)
        self.Ep = Mp0 * epsilon + Ep0
        self.sello = nuevo_sello

        # Registrar
        self._registrar_estado()

    def trayectoria_completa(self, epsilons: Dict[str, float]) -> List[Dict]:
        """
        Calcula la trayectoria completa S0 → S4.

        Args:
            epsilons: Diccionario {sello: epsilon} con los valores de ε
                      para cada transición

        Returns:
            Lista de estados en cada sello
        """
        # Reiniciar al estado primordial
        self.Mp = Mp0
        self.Ep = Ep0
        self.sello = "S0"
        self.historial = []
        self._registrar_estado()

        # Evolucionar a través de los sellos
        for sello in SELLOS_ORDEN[1:]:  # S1, S2, S3, S4
            if sello in epsilons:
                self.evolucionar_a_sello(sello, epsilons[sello])

        return self.historial.copy()

    @classmethod
    def crear_primordial(cls) -> EstadoPrimordial:
        """Crea un estado primordial estándar (S0)."""
        return cls(Mp=Mp0, Ep=Ep0, sello="S0")

    def resumen(self) -> str:
        """Genera un resumen del estado actual con el Campo de Adrián."""
        return (
            f"Estado Primordial MCMC - Dualidad Tensional MASA-ESPACIO\n"
            f"{'='*55}\n"
            f"Sello: {self.sello} - {self.nombre_sello} (S = {self.S:.4f})\n"
            f"{'='*55}\n"
            f"  Mp (Masa potencial)    = {self.Mp:.6f}\n"
            f"  Ep (Energía/Espacio)   = {self.Ep:.6e}\n"
            f"  Φ_ten (Campo de Adrián) = {self.Phi_ten:.2e}\n"
            f"  P_ME (Polarización)    = {self.P_ME:+.6f}\n"
            f"  k_pre (Tasa colapso)   = {self.k_pre:.3f}\n"
            f"{'='*55}\n"
            f"  Energía total conservada = {self.energia_total:.6f}\n"
        )


# =============================================================================
# Funciones de Verificación
# =============================================================================

def verificar_conservacion(historial: List[Dict]) -> Tuple[bool, float]:
    """
    Verifica la conservación de energía total a lo largo de la evolución.

    Args:
        historial: Lista de estados

    Returns:
        (conservada, error_maximo): Tupla con resultado y error
    """
    if not historial:
        return True, 0.0

    E0 = historial[0]["Mp"] + historial[0]["Ep"]
    max_error = 0.0

    for estado in historial:
        E = estado["Mp"] + estado["Ep"]
        error = abs(E - E0) / E0
        max_error = max(max_error, error)

    # Conservación si error < 1%
    return max_error < 0.01, max_error


def verificar_P_ME_monotonico(historial: List[Dict]) -> bool:
    """
    Verifica que P_ME decrece monótonamente (de +1 hacia -1).

    Args:
        historial: Lista de estados

    Returns:
        True si P_ME es decreciente
    """
    if len(historial) < 2:
        return True

    for i in range(1, len(historial)):
        if historial[i]["P_ME"] > historial[i-1]["P_ME"]:
            return False

    return True


# =============================================================================
# Tests
# =============================================================================

def _test_estado_primordial():
    """Verifica la implementación del estado primordial."""

    # Test 1: Estado inicial correcto
    estado = EstadoPrimordial.crear_primordial()
    assert estado.Mp == Mp0, f"Mp debe ser {Mp0}"
    assert estado.Ep == Ep0, f"Ep debe ser {Ep0}"
    assert estado.sello == "S0", "Sello inicial debe ser S0"

    # Test 2: P_ME inicial ≈ +1
    assert estado.P_ME > 0.99, f"P_ME inicial debe ser ≈ +1, es {estado.P_ME}"

    # Test 3: Tensión inicial alta
    assert estado.tension > 1e9, f"Tensión inicial debe ser > 10^9"

    # Test 4: Evolución conserva energía
    epsilons = {"S1": 0.01, "S2": 0.1, "S3": 0.5, "S4": 0.99}
    historial = estado.trayectoria_completa(epsilons)
    conservada, error = verificar_conservacion(historial)
    assert conservada, f"Energía no conservada, error = {error}"

    # Test 5: P_ME decrece
    assert verificar_P_ME_monotonico(historial), "P_ME debe decrecer"

    # Test 6: P_ME final < 0 (energía domina)
    assert historial[-1]["P_ME"] < 0, "P_ME final debe ser negativo"

    print("✓ Todos los tests del Bloque 0 pasaron")
    return True


if __name__ == "__main__":
    _test_estado_primordial()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Estado Primordial MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    # Crear estado primordial
    estado = EstadoPrimordial.crear_primordial()
    print(estado.resumen())

    # Definir epsilons para cada transición
    # Estos valores vienen del Bloque 1 (pregeometría)
    epsilons = {
        "S1": 0.01,   # 1% convertido en S1
        "S2": 0.10,   # 10% convertido en S2
        "S3": 0.50,   # 50% convertido en S3
        "S4": 0.99,   # 99% convertido en S4
    }

    print("\nTrayectoria S0 → S4:")
    print("-"*60)
    historial = estado.trayectoria_completa(epsilons)

    for h in historial:
        print(f"Sello {h['sello']}: S={h['S']:.3f}, "
              f"Mp={h['Mp']:.4f}, Ep={h['Ep']:.4e}, "
              f"P_ME={h['P_ME']:+.4f}")

    print("-"*60)
    conservada, error = verificar_conservacion(historial)
    print(f"Conservación de energía: {'✓' if conservada else '✗'} "
          f"(error = {error:.2e})")
    print(f"P_ME decreciente: {'✓' if verificar_P_ME_monotonico(historial) else '✗'}")
