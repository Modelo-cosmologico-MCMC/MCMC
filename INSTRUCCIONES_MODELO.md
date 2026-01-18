# MCMC - Modelo Cosmológico de Masa-Espacio
## Instrucciones de Uso Completas

**Autor:** Adrián Martínez Estellés
**Versión:** 2.6.0
**Fecha:** Diciembre 2024

---

## 1. ARQUITECTURA DEL MODELO

El modelo MCMC (Modelo Cosmológico de Materia-Curvatura) propone una alternativa a ΛCDM basada en:
- **ECV** (Energía Cuántica Virtual): Reemplaza la energía oscura
- **MCV** (Materia Cuántica Virtual): Reemplaza la materia oscura
- **Ley de Cronos**: Dilatación temporal en regiones densas

### 1.1 Estructura de Módulos

```
mcmc_core/
├── ONTOLOGÍA FUNDAMENTAL
│   ├── bloque0_estado_primordial.py   # Estado inicial S₀
│   ├── bloque1_pregeometria.py        # Evolución pre-geométrica
│   └── fase_pregeometrica.py          # Fase completa S₀ → S₄
│
├── COSMOLOGÍA
│   ├── bloque2_cosmologia.py          # Ecuaciones cosmológicas
│   ├── ontologia_ecv_mcv.py           # Framework ECV/MCV
│   └── class_mcmc.py                  # CMB C_ℓ y P(k)
│
├── FORMACIÓN DE ESTRUCTURA
│   ├── bloque3_nbody.py               # N-body básico
│   ├── nbody_cronos.py                # N-body con Ley de Cronos
│   └── ley_cronos.py                  # Implementación Ley de Cronos
│
├── VALIDACIÓN CUÁNTICA
│   ├── circuito_cuantico.py           # Circuitos cuánticos
│   ├── qubit_tensorial.py             # Estados entrelazados
│   └── spin_network_lqg.py            # Redes de espín (LQG)
│
├── VALIDACIÓN OBSERVACIONAL
│   ├── datos_observacionales.py       # Base de datos observacional
│   ├── sparc_zhao.py                  # Curvas de rotación SPARC
│   ├── lensing_mcv.py                 # Weak lensing
│   └── desi_y3.py                     # DESI Year 3 BAO
│
└── __init__.py                        # Exportaciones
```

---

## 2. FLUJO DE EVOLUCIÓN DEL MODELO

### 2.1 Evolución Cósmica Completa

```
S₀ (Singularidad Ontológica)
 │  Mp = 1.0, Ep ≈ 0, P_ME ≈ 1
 │  Estado: |Φ⟩ = |00⟩ (masa pura)
 ↓
S₁ (Primera Transición) ─────── V₀D → V₁D
 │  Primeros colapsos, emerge dimensionalidad
 ↓
S₂ (Segunda Transición) ─────── V₁D → V₂D
 │  Conversión acelerada Mp → Ep
 ↓
S₃ (Tercera Transición) ────── V₂D → V₃D
 │  Geometría 3D emergente
 ↓
S₄ (Big Bang) ──────────────── V₃D → V₃₊₁D
 │  ε ≈ 0.0112, P_ME ≈ -0.978
 │  Estado: |Φ⟩ = √ε|00⟩ + √(1-ε)|11⟩
 │  Activa: Ley de Cronos
 ↓
z > 0 (Universo Observable)
 │  ECV: Λ_rel(z) = 1 + ε·tanh((z_trans-z)/Δz)
 │  MCV: Perfiles cored con fricción entrópica
 ↓
z = 0 (Hoy)
    H₀ = 67.64 km/s/Mpc
    Ω_m = 0.315, Ω_Λ = 0.685
```

---

## 3. INSTALACIÓN Y CONFIGURACIÓN

### 3.1 Requisitos

```bash
# Dependencias Python
pip install numpy scipy matplotlib

# Opcional para circuitos cuánticos
pip install qiskit  # IBM Quantum
```

### 3.2 Importación Básica

```python
# Importar todo el módulo
import mcmc_core

# O importar componentes específicos
from mcmc_core import (
    # Cosmología
    E_MCMC_ECV, H_MCMC_ECV, Lambda_rel,
    distancia_comovil_ECV, distancia_luminosidad_ECV,

    # Perfiles de densidad
    PerfilZhaoMCMC, ParametrosZhaoMCMC,

    # CLASS-MCMC
    calcular_sigma8_MCMC, P_k_MCMC, C_l_TT_approx,

    # DESI validation
    calcular_chi2_DESI, ajustar_epsilon_z_trans,
)
```

---

## 4. USO DE CADA MÓDULO

### 4.1 Ontología Fundamental

#### Estado Primordial (bloque0)

```python
from mcmc_core.bloque0_estado_primordial import (
    EstadoPrimordial, calcular_Phi_ten, calcular_P_ME,
    SELLOS, Mp0, Ep0
)

# Crear estado inicial
estado = EstadoPrimordial(Mp=Mp0, Ep=Ep0)
print(f"Tensión primordial Φ_ten = {calcular_Phi_ten(Mp0, Ep0):.2e}")
print(f"Polarización P_ME = {calcular_P_ME(Mp0, Ep0):.6f}")

# Evolucionar a través de sellos
from mcmc_core.bloque0_estado_primordial import trayectoria_completa
historial = trayectoria_completa([0.001, 0.01, 0.1, 1.0])
```

#### Pre-geometría (bloque1)

```python
from mcmc_core.bloque1_pregeometria import (
    Pregeometria, tasa_colapso_k, calcular_epsilon
)

# Tasa de colapso en función del sello entrópico
import numpy as np
S_array = np.linspace(0, 1.001, 100)
k_array = [tasa_colapso_k(S) for S in S_array]

# Clase completa
pre = Pregeometria()
trayectoria = pre.trayectoria()
```

#### Fase Pre-geométrica Completa (fase_pregeometrica)

```python
from mcmc_core.fase_pregeometrica import (
    FasePregeometrica, k_calibrado, EPSILON_RESIDUAL
)

fase = FasePregeometrica()
trayectoria = fase.trayectoria()
colapsos = fase.secuencia_colapsos()

# Verificar calibración
from mcmc_core.fase_pregeometrica import verificar_calibracion
verificar_calibracion()  # Debe dar ∫k(S)dS ≈ ln(1/ε)
```

---

### 4.2 Cosmología ECV/MCV

#### Framework Ontológico (ontologia_ecv_mcv)

```python
from mcmc_core.ontologia_ecv_mcv import (
    # Constantes fundamentales
    H0_MCMC, OMEGA_M_MCMC, OMEGA_LAMBDA_MCMC,
    EPSILON_ECV, Z_TRANS, DELTA_Z,

    # ECV (reemplaza energía oscura)
    Lambda_rel, rho_ECV, E_MCMC_ECV, H_MCMC_ECV,

    # Distancias
    distancia_comovil_ECV, distancia_luminosidad_ECV,
    modulo_distancia_ECV,

    # MCV (reemplaza materia oscura)
    rho_0_MCV, r_core_MCV,
    perfil_MCV_Burkert, perfil_MCV_isotermico,

    # Fricción entrópica (Ley de Cronos)
    FriccionEntropicaMCV, velocidad_circular_MCV,
)

# Ejemplo: calcular Λ_rel(z)
z = np.linspace(0, 10, 100)
Lambda_z = [Lambda_rel(zi) for zi in z]

# Distancia comóvil
z_test = 1.0
D_c = distancia_comovil_ECV(z_test)
print(f"D_c(z=1) = {D_c:.2f} Mpc")
```

#### Cosmología Básica (bloque2)

```python
from mcmc_core.bloque2_cosmologia import (
    E_LCDM, E_MCMC, Lambda_relativo,
    distancia_comovil, distancia_luminosidad,
    modulo_distancia, edad_universo,
    CosmologiaMCMC,
)

# Comparar MCMC vs ΛCDM
cosmo = CosmologiaMCMC()
comparacion = cosmo.comparar_con_LCDM(np.linspace(0, 2, 50))
```

---

### 4.3 CMB y Estructura (CLASS-MCMC)

```python
from mcmc_core.class_mcmc import (
    ParametrosCLASS, PARAMS_CLASS,

    # Crecimiento de estructura
    calcular_D_MCMC,      # Factor de crecimiento D(z)
    calcular_sigma8_MCMC, # σ₈ normalizado
    calcular_S8_MCMC,     # S₈ = σ₈√(Ω_m/0.3)

    # Espectro de potencias
    P_k_MCMC,             # P(k,z) materia
    C_l_TT_approx,        # C_ℓ CMB aproximado

    # Escalas acústicas
    theta_star_MCMC,      # θ_* escala angular
    l_acoustic_MCMC,      # ℓ_A multipolo acústico
    horizonte_sonido_MCMC,# r_s horizonte de sonido

    # Comparación
    comparar_con_LCDM,
    test_CLASS_MCMC,
)

# Ejemplo: calcular espectro de potencias
k = np.logspace(-3, 1, 100)  # h/Mpc
P_k = P_k_MCMC(k, z=0)

# Espectro CMB
ell = np.arange(2, 2500)
C_l = C_l_TT_approx(ell)

# Parámetros derivados
print(f"σ₈ = {calcular_sigma8_MCMC():.4f}")
print(f"S₈ = {calcular_S8_MCMC():.4f}")
print(f"ℓ_A = {l_acoustic_MCMC():.1f}")
```

---

### 4.4 Ley de Cronos y N-body

#### Ley de Cronos (ley_cronos)

```python
from mcmc_core.ley_cronos import (
    LeyCronos,
    dilatacion_temporal,  # Δt/Δt₀ = 1 + (ρ/ρc)^(3/2)/α
    tiempo_propio,
    radio_core,           # r_core ∝ M^0.35 × (1+z)^(-0.5)
)

cronos = LeyCronos()

# Dilatación temporal en región densa
rho = 1e8  # M☉/kpc³
dt_ratio = dilatacion_temporal(rho)
print(f"Dilatación temporal: {dt_ratio:.2f}x")

# Radio core predicho para un halo
M_halo = 1e12  # M☉
r_c = radio_core(M_halo, z=0)
print(f"Radio core predicho: {r_c:.2f} kpc")
```

#### N-body con Cronos (nbody_cronos)

```python
from mcmc_core.nbody_cronos import (
    ParametrosCronosNBody, PARAMS_CRONOS,
    lapse_function,       # α(ρ) función lapse
    friccion_entropica,   # F = -η(ρ)·v
    radio_core_cronos,
    IntegradorCronos,
    analizar_halo_cronos,
    test_NBody_Cronos,
)

# Función lapse (dilatación temporal)
rho = 1e8
alpha = lapse_function(rho)
print(f"α(ρ=1e8) = {alpha:.4f}")

# Fricción entrópica
v = 100  # km/s
F = friccion_entropica(v, rho)
print(f"F_friction = {F:.2f} km/s²")

# Simular halo
integrador = IntegradorCronos()
resultado = analizar_halo_cronos(M_halo=1e12)
```

---

### 4.5 Validación Observacional

#### Curvas de Rotación SPARC (sparc_zhao)

```python
from mcmc_core.sparc_zhao import (
    ParametrosZhaoMCMC, PARAMS_ZHAO,
    PerfilZhaoMCMC, PerfilNFW,
    AjustadorSPARC, AjustadorGAIA,
    test_SPARC_Zhao_MCMC, test_GAIA_Zhao_MCMC,
)

# Perfil Zhao con γ=0.51 (cored)
params = PARAMS_ZHAO
perfil = PerfilZhaoMCMC(params)

# Velocidad circular
r = np.linspace(0.1, 30, 100)  # kpc
rho_0 = 1e7  # M☉/kpc³
r_s = 5.0    # kpc
v_c = [perfil.velocidad_circular(ri, rho_0, r_s) for ri in r]

# Ajustar galaxia SPARC
ajustador = AjustadorSPARC()
resultado = ajustador.ajustar_galaxia("NGC2403")

# Test completo
test_SPARC_Zhao_MCMC()  # ~69% mejora sobre NFW
test_GAIA_Zhao_MCMC()   # ~60% mejora en Vía Láctea
```

#### Weak Lensing (lensing_mcv)

```python
from mcmc_core.lensing_mcv import (
    ParametrosLensing, PARAMS_LENSING,
    Sigma_crit,           # Densidad crítica de superficie
    kappa_NFW, kappa_Zhao,# Convergencia κ
    gamma_tangencial,     # Shear tangencial
    calcular_S8_lensing,  # S₈ desde lensing
    test_Lensing_MCV,
)

# Densidad crítica
z_l, z_s = 0.3, 1.0
Sigma_c = Sigma_crit(z_l, z_s)
print(f"Σ_crit = {Sigma_c:.2e} M☉/kpc²")

# Convergencia κ
R = np.array([100, 500, 1000])  # kpc
kappa = kappa_NFW(R, z_l, z_s)
print(f"κ_NFW(100 kpc) = {kappa[0]:.4f}")

# S₈ desde lensing
S8_result = calcular_S8_lensing()
print(f"S₈_MCV = {S8_result['S8_MCV']:.3f}")
```

#### DESI Year 3 BAO (desi_y3)

```python
from mcmc_core.desi_y3 import (
    DESI_Y3_DATA,           # 13 puntos BAO
    calcular_chi2_DESI,     # χ² para MCMC
    calcular_chi2_LCDM_DESI,# χ² para ΛCDM
    ajustar_epsilon_z_trans,# Optimizar parámetros
    comparar_DESI_detallado,
    analizar_tensiones_DESI,
    test_DESI_Y3,
)

# χ² con parámetros actuales
chi2_mcmc = calcular_chi2_DESI(epsilon=0.012, z_trans=8.9)
chi2_lcdm = calcular_chi2_LCDM_DESI()
print(f"χ²_MCMC = {chi2_mcmc:.2f}")
print(f"χ²_ΛCDM = {chi2_lcdm:.2f}")
print(f"Mejora: {(chi2_lcdm-chi2_mcmc)/chi2_lcdm*100:.1f}%")

# Optimizar parámetros
resultado = ajustar_epsilon_z_trans()
print(f"ε_opt = {resultado['epsilon_opt']:.4f}")
print(f"z_trans_opt = {resultado['z_trans_opt']:.2f}")

# Análisis de tensiones
tensiones = analizar_tensiones_DESI()
```

#### Base de Datos Observacional (datos_observacionales)

```python
from mcmc_core.datos_observacionales import (
    # Planck 2018
    DatosPlanck2018,

    # BAO compilations
    BAO_BOSS_DR12, BAO_EBOSS_DR16, BAO_DESI_2024,

    # SPARC galaxies
    SPARC_CATALOG,

    # GAIA
    DatosGAIA,

    # Supernovae
    PANTHEON_PLUS_SAMPLE,

    # H0 measurements
    MEDICIONES_H0,

    # Analysis functions
    calcular_chi2_BAO, calcular_chi2_SN,
    tension_H0, resumen_datos,
)

# Resumen de datos disponibles
resumen = resumen_datos()
print(resumen)
```

---

### 4.6 Validación Cuántica

#### Circuito Cuántico (circuito_cuantico)

```python
from mcmc_core.circuito_cuantico import (
    CircuitoTensorial,
    preparar_estado,
    medir_P_ME, medir_ZZ,
    testigo_bell, violacion_CHSH,
    generar_codigo_qiskit,
)

# Crear circuito para ε = 0.0112
circuito = CircuitoTensorial(epsilon=0.0112)

# Preparar estado |Φ⟩ = √ε|00⟩ + √(1-ε)|11⟩
psi = preparar_estado(0.0112)

# Mediciones
P_ME = medir_P_ME(psi)
ZZ = medir_ZZ(psi)
bell = testigo_bell(psi)
CHSH = violacion_CHSH(psi)

print(f"P_ME = {P_ME:.4f}")
print(f"⟨ZZ⟩ = {ZZ:.4f}")
print(f"Testigo Bell = {bell:.4f}")
print(f"CHSH S = {CHSH:.4f} (violación si > 2)")

# Generar código para IBM Quantum
codigo = generar_codigo_qiskit(0.0112)
print(codigo)
```

#### Qubit Tensorial (qubit_tensorial)

```python
from mcmc_core.qubit_tensorial import (
    QubitTensorial,
    estado_tensorial,
    concurrencia,
    entropia_entrelazamiento,
    verificar_consistencia_cuantica_clasica,
)

qubit = QubitTensorial(epsilon=0.0112)

# Estado y métricas
psi = estado_tensorial(0.0112)
C = concurrencia(psi)
S_ent = entropia_entrelazamiento(psi)

print(f"Concurrencia C = {C:.4f}")
print(f"Entropía S_ent = {S_ent:.4f}")

# Verificar consistencia
verificar_consistencia_cuantica_clasica()
```

#### Redes de Espín LQG (spin_network_lqg)

```python
from mcmc_core.spin_network_lqg import (
    SpinNetwork,
    area_enlace, volumen_nodo,
    GAMMA_IMMIRZI,
)

# Crear red de espín
red = SpinNetwork(N_nodos=100)
red.generar_enlaces_aleatorios(p=0.3)

# Encontrar punto crítico de percolación
p_c, chi_max = red.encontrar_punto_critico()
print(f"p_c = {p_c:.4f}")

# Interpretar transición geométrica
red.interpretar_transicion_geometrica()
```

---

## 5. PARÁMETROS FUNDAMENTALES

### 5.1 Constantes Cosmológicas

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| H₀ | 67.64 km/s/Mpc | Constante de Hubble |
| Ω_m | 0.315 | Densidad de materia |
| Ω_Λ | 0.685 | Densidad de energía oscura |
| Ω_b | 0.0493 | Densidad bariónica |
| Ω_r | 9.24×10⁻⁵ | Densidad de radiación |

### 5.2 Parámetros ECV

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| ε | 0.012 | Amplitud de modificación Λ |
| z_trans | 8.9 | Redshift de transición |
| Δz | 1.5 | Anchura de transición |

### 5.3 Parámetros MCV/Cronos

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| γ | 0.51 | Pendiente interna Zhao |
| α | 2.0 | Parámetro Zhao |
| β | 3.0 | Parámetro Zhao |
| ρ_star | 10⁹ M☉/kpc³ | Densidad característica |
| r_star | 2.0 kpc | Radio característico |

### 5.4 Parámetros Cuánticos

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| ε_residual | 0.0112 | Fracción masa residual |
| γ_Immirzi | 0.274 | Parámetro LQG (Bekenstein-Hawking) |
| θ_RY | 2.93 rad | Ángulo rotación circuito |

---

## 6. VALIDACIÓN Y TESTS

### 6.1 Ejecutar Todos los Tests

```python
# Test completo de todos los módulos
from mcmc_core.class_mcmc import test_CLASS_MCMC
from mcmc_core.nbody_cronos import test_NBody_Cronos
from mcmc_core.lensing_mcv import test_Lensing_MCV
from mcmc_core.desi_y3 import test_DESI_Y3
from mcmc_core.sparc_zhao import test_SPARC_Zhao_MCMC, test_GAIA_Zhao_MCMC

# Ejecutar
test_CLASS_MCMC()     # CMB y P(k)
test_NBody_Cronos()   # N-body con Cronos
test_Lensing_MCV()    # Weak lensing
test_DESI_Y3()        # DESI BAO
test_SPARC_Zhao_MCMC() # SPARC rotation curves
test_GAIA_Zhao_MCMC()  # Milky Way
```

### 6.2 Validación Rápida

```bash
python3 -c "
from mcmc_core import (
    calcular_sigma8_MCMC, l_acoustic_MCMC,
    calcular_chi2_DESI, EPSILON_ECV, Z_TRANS
)
print(f'σ₈ = {calcular_sigma8_MCMC():.4f}')
print(f'ℓ_A = {l_acoustic_MCMC():.1f}')
print(f'χ²_DESI = {calcular_chi2_DESI(EPSILON_ECV, Z_TRANS):.2f}')
"
```

---

## 7. RESULTADOS PRINCIPALES

### 7.1 Predicciones CMB (CLASS-MCMC)
- r_s(z_*) = 141.57 Mpc (Planck: ~147 Mpc)
- ℓ_acoustic = 309 (Planck: ~302)
- σ₈ = 0.8111 (Planck: 0.811)
- Primer pico acústico: ℓ ≈ 222 (Planck: ~220)

### 7.2 Validación DESI Y3
- χ²_MCMC = 12.19 vs χ²_ΛCDM = 12.68
- **Mejora: 3.9%** sobre ΛCDM

### 7.3 Curvas de Rotación SPARC
- Perfil Zhao (γ=0.51) vs NFW
- **Mejora: 69%** en χ² promedio

### 7.4 Vía Láctea (GAIA)
- **Mejora: 60%** sobre NFW en ajuste

---

## 8. REFERENCIAS

1. Planck Collaboration (2018). Planck 2018 results. VI. Cosmological parameters.
2. DESI Collaboration (2024). DESI 2024 VI: Cosmological constraints from BAO.
3. Lelli et al. (2016). SPARC: Mass Models for 175 Disk Galaxies.
4. GAIA Collaboration (2023). GAIA DR3: Milky Way kinematics.

---

## 9. SOPORTE

Para preguntas o contribuciones:
- GitHub: [Modelo-cosmologico-MCMC/MCMC](https://github.com/Modelo-cosmologico-MCMC/MCMC)
- Autor: Adrián Martínez Estellés

---

*Este documento describe la arquitectura completa del modelo MCMC v2.6.0*
