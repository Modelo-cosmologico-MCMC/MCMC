# Modelo Cosmológico de Múltiples Colapsos (MCMC)

**Autor:** Adrián Martínez Estellés
**Copyright:** (c) 2024. Todos los derechos reservados.
**Contacto:** adrianmartinezestelles92@gmail.com

---

## Descripción

El Modelo Cosmológico de Múltiples Colapsos (MCMC) es un marco teórico que describe la evolución del universo desde un estado primordial de máxima tensión masa-energía hasta el estado actual.

El modelo se estructura en **5 bloques**:

| Bloque | Nombre | Descripción |
|--------|--------|-------------|
| 0 | Estado Primordial | Mp0, Ep0, tensión máxima, P_ME ≈ +1 |
| 1 | Pregeometría | Tasa de colapso k(S), integral entrópica |
| 2 | Cosmología | Ecuaciones de Friedmann modificadas |
| 3 | N-body | Fricción entrópica, perfiles Burkert |
| 4 | Lattice-Gauge | Yang-Mills, mass gap E_min(S) |

---

## Ontología MCMC

### Sellos Entrópicos
```
S0 = 0.000  (Estado primordial)
S1 = 0.010  (Primera transición)
S2 = 0.100  (Segunda transición)
S3 = 1.000  (Tercera Transición)
S4 = 1.001  (Big bang Observable)
```

### Ecuaciones Fundamentales

**Tasa de colapso:**
```
k(S) = k0 × [1 + a1·sin(2πS) + a2·sin(4πS) + a3·sin(6πS)]
k0 = 6.307, a1 = 0.15, a2 = 0.25, a3 = 0.35
```

**Integral entrópica:**
```
ε(S) = ∫k(s)ds / ∫k(s)ds|_{0→S4}
```

**Masa y espacio:**
```
Mp(S) = Mp0 × (1 - ε)
Ep(S) = Mp0 × ε
```

**Polarización masa-espacio:**
```
P_ME = (Mp - Ep) / (Mp + Ep)
```

---

## Estructura del Repositorio

```
mcmc-cosmology/
├── mcmc_core/
│   ├── __init__.py
│   ├── bloque0_estado_primordial.py
│   ├── bloque1_pregeometria.py
│   └── bloque2_cosmologia.py
├── lattice/
│   ├── __init__.py
│   └── bloque4_ym_lattice.py
├── simulations/
│   ├── __init__.py
│   └── bloque3_nbody.py
├── examples/
│   └── ejemplo_5_bloques.py
├── tests/
│   └── test_mcmc.py
├── README.md
├── setup.py
├── requirements.txt
└── LICENSE
```

---

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Modelo-cosmologico-MCMC/MCMC.git
cd MCMC

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

---

## Uso

### Ejemplo Rápido

```python
from mcmc_core import Pregeometria, CosmologiaMCMC
from simulations import friccion_entropica, radio_core
from lattice import beta_MCMC, mass_gap

# Bloque 1: Pregeometría
preg = Pregeometria()
print(f"k(S=1.0) = {preg.k(1.0):.4f}")
print(f"P_ME(S=1.0) = {preg.P_ME(1.0):+.4f}")

# Bloque 2: Cosmología
cosmo = CosmologiaMCMC()
print(f"Edad del universo: {cosmo.edad():.2f} Gyr")

# Bloque 3: N-body
r_c = radio_core(1e11)  # M = 10^11 M☉
print(f"Radio del núcleo: {r_c:.2f} kpc")

# Bloque 4: Lattice-Gauge
E_min = mass_gap(S=1.0)
print(f"Mass gap: {E_min:.4f} GeV")
```

### Demo Completa

```bash
python examples/ejemplo_5_bloques.py
```

### Tests

```bash
python -m pytest tests/
# o
python tests/test_mcmc.py
```

---

## Bloques en Detalle

### Bloque 0: Estado Primordial
- **Mp0 = 1.0**: Masa potencial normalizada
- **Ep0 = 10⁻¹⁰**: Espacio primordial mínimo
- **Tensión = 10¹⁰**: Máxima tensión inicial
- **P_ME ≈ +1**: Masa domina completamente

### Bloque 1: Pregeometría
- Tasa de colapso k(S) con armónicos sinusoidales
- Integral entrópica para calcular ε(S)
- Trayectoria P_ME: de +1 a -1

### Bloque 2: Cosmología
```
E_LCDM(z) = √[Ωm(1+z)³ + ΩΛ]
Λ_rel(z) = 1 + δΛ × exp(-z/2) × (1+z)^(-0.5)
E_MCMC(z) = √[Ωm(1+z)³ + ΩΛ × Λ_rel(z)]
```
Parámetros: H0=67.4, Ωm=0.315, ΩΛ=0.685, δΛ=0.02

### Bloque 3: N-body
- **Fricción entrópica:** η(ρ) = α × (ρ/ρc)^1.5
- **Perfil Burkert:** ρ = ρ0/[(1+r/rc)(1+(r/rc)²)]
- **Relación núcleo-masa:** r_core = 1.8 × (M/10¹¹)^0.35 kpc

### Bloque 4: Lattice-Gauge
```
β(S) = β0 + β1 × exp[-bS × (S - S3)]
E_min = αH × ΛQCD
```
Parámetros: β0=6.0, β1=2.0, bS=10.0, αH=0.1, ΛQCD=0.2 GeV

---

## Licencia

**Propietaria - Todos los derechos reservados**

Este modelo está protegido por derechos de autor. Ver archivo `LICENSE` para términos completos.

**Restricciones principales:**
- Prohibido uso comercial sin autorización
- Prohibida modificación del código
- Requiere cita apropiada para uso académico

**Para solicitudes:** adrianmartinezestelles92@gmail.com

---

## Cita

Si utiliza este modelo en su investigación, por favor cite:

```
Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos (MCMC)".
10 de octubre de 2024.
```

---

## Contacto

**Adrián Martínez Estellés**
Email: adrianmartinezestelles92@gmail.com
