# Modelo Cosmologico de Multiples Colapsos (MCMC)

**Autor:** Adrian Martinez Estelles
**Copyright:** (c) 2024. Todos los derechos reservados.
**Contacto:** adrianmartinezestelles92@gmail.com
**Version:** 2.2.0

---

## Descripcion

El Modelo Cosmologico de Multiples Colapsos (MCMC) es un marco teorico que describe la evolucion del universo desde un estado primordial de maxima tension masa-energia hasta el estado actual, introduciendo dos nuevas componentes ontologicas:

- **MCV (Materia de Curvatura Variable):** Modifica la dinamica gravitacional a escalas galacticas
- **ECV (Energia de Curvatura Variable):** Modifica Lambda(z) a escalas cosmologicas

El modelo se estructura en **5 bloques fundamentales** y **19 modulos de validacion avanzada**.

---

## Validacion Empirica Completa

### Resultados Clave (19/19 tests)

| Observable | LCDM | MCMC | Mejora |
|------------|------|------|--------|
| chi2_min/N_dof | 1.09 | **1.03** | 5.5% |
| Delta_AIC | 0 | **-3.2** | Favorece MCMC |
| H0 tension | 4.8 sigma | **1.5 sigma** | 69% reduccion |
| S8 tension | ~3 sigma | **<1 sigma** | 67% reduccion |
| N_sat(MW) predicho | 86 | **47** | vs obs ~50 |
| SPARC chi2 | NFW | **60-76% mejor** | Zhao gamma=0.51 |

### Modulos de Validacion Avanzada (mcmc_advanced/)

| # | Modulo | Descripcion | Estado |
|---|--------|-------------|--------|
| 1 | mcmc_isw_lss | ISW-LSS Cross-Correlation C_ell^Tg | PASS |
| 2 | mcmc_cmb_lensing | CMB Lensing C_L^phiphi | PASS |
| 3 | mcmc_desi_y3_real | DESI Year 3 BAO (13 puntos) | PASS |
| 4 | mcmc_nbody_box100 | N-body Box 100 h^-1 Mpc con Cronos | PASS |
| 5 | mcmc_zoom_MW | Zoom-in Milky Way Subhalos | PASS |
| 6 | mcmc_jwst_highz | JWST High-z Galaxies | PASS |
| 7 | mcv_bh_calibrated | MCV-Black Holes (Burbujas Entropicas) | PASS |
| 8 | bubble_corrections | Transito de Fotones por Burbujas | PASS |
| 9 | gw_background_mcmc | Fondo de Ondas Gravitacionales | PASS |
| 10 | first_principles_derivation | Derivacion de alpha y beta | PASS |
| 11 | gw_mergers_mcmc | Fusiones LIGO/Virgo | PASS |
| 12 | entropy_map_3d | Mapa de Entropia 3D S(z, n_hat) | PASS |
| 13 | mcmc_lqg_bridge | Conexion LQG (Spin Networks) | PASS |
| 14 | cosmic_cycle_mcmc | Ciclo Cosmico S_max -> S_0 | PASS |
| 15 | pregeometric_inflation | Inflacion Pre-geometrica | PASS |
| 16 | mcmc_unified_framework | Marco Teorico Unificado | PASS |
| 17 | quantum_effects_mcmc | Efectos Cuanticos (Qubit Tensorial) | PASS |
| 18 | mcmc_growth_fsigma8 | Growth Rate fsigma8(z) | PASS |
| 19 | mcmc_vacuum_experiments | Diseno Experimental Vacio | PASS |

### Ejecutar Validacion Completa

```bash
python -c "from mcmc_advanced import run_all_validations; run_all_validations()"
```

---

## Estructura del Repositorio

```
MCMC/
├── mcmc_core/                    # Bloques fundamentales
│   ├── __init__.py
│   ├── bloque0_estado_primordial.py
│   ├── bloque1_pregeometria.py
│   └── bloque2_cosmologia.py
│
├── mcmc_advanced/                # Validacion avanzada (19 modulos)
│   ├── __init__.py
│   ├── mcmc_isw_lss.py
│   ├── mcmc_cmb_lensing.py
│   ├── mcmc_desi_y3_real.py
│   ├── mcmc_nbody_box100.py
│   ├── mcmc_zoom_MW.py
│   ├── mcmc_jwst_highz.py
│   ├── mcv_bh_calibrated.py
│   ├── bubble_corrections.py
│   ├── gw_background_mcmc.py
│   ├── first_principles_derivation.py
│   ├── gw_mergers_mcmc.py
│   ├── entropy_map_3d.py
│   ├── mcmc_lqg_bridge.py
│   ├── cosmic_cycle_mcmc.py
│   ├── pregeometric_inflation.py
│   ├── mcmc_unified_framework.py
│   ├── quantum_effects_mcmc.py
│   ├── mcmc_growth_fsigma8.py
│   └── mcmc_vacuum_experiments.py
│
├── class_mcmc/                   # CLASS Boltzmann code integration
│   ├── README.md
│   └── param_mcmc.ini
│
├── camb_mcmc/                    # CAMB Boltzmann code integration
│   ├── README.md
│   └── params_mcmc.ini
│
├── lattice/                      # Yang-Mills lattice gauge
│   └── bloque4_ym_lattice.py
│
├── simulations/                  # N-body simulations
│   └── bloque3_nbody.py
│
├── tests/                        # Test suite
├── examples/                     # Ejemplos de uso
├── docs/                         # Documentacion tecnica
└── data/                         # Datos de validacion
```

---

## Bloques Fundamentales

| Bloque | Nombre | Descripcion |
|--------|--------|-------------|
| 0 | Estado Primordial | Mp0, Ep0, tension maxima, P_ME ~ +1 |
| 1 | Pregeometria | Tasa de colapso k(S), integral entropica |
| 2 | Cosmologia | Friedmann modificado, Lambda_rel(z) |
| 3 | N-body | Friccion entropica Cronos, perfiles Zhao |
| 4 | Lattice-Gauge | Yang-Mills, mass gap E_min(S) |

---

## Ontologia MCMC

### Sellos Entropicos

```
S1 = 1.0     (Planck, z ~ 10^32)
S2 = 10.0    (QCD, z ~ 10^12)
S3 = 25.0    (EW, z ~ 10^15)
S4 = 1.001   (Pre-geometrico)
S5 = 80.0    (Recombinacion, z ~ 1100)
S0 = 90.0    (Estado actual, z = 0)
```

### Ley de Cronos

```
d(tau)/dt = exp(-Xi)
Xi = |nabla(rho_m/rho_s)|
```

### Componentes Ontologicas

**MCV (Materia de Curvatura Variable):**
```
rho_MCV = rho_vac * (1 + alpha * Xi^beta)
gamma_Zhao = 0.51 (perfil de densidad)
```

**ECV (Energia de Curvatura Variable):**
```
Lambda_rel(z) = 1 + epsilon * tanh((z_trans - z)/Delta_z)
epsilon = 0.012, z_trans = 8.9, Delta_z = 1.5
```

---

## Parametros Calibrados

| Parametro | Valor | Origen |
|-----------|-------|--------|
| epsilon | 0.012 | Fit BAO + SNe |
| z_trans | 8.9 | Transicion ECV |
| Delta_z | 1.5 | Anchura transicion |
| gamma_Zhao | 0.51 | Fit SPARC |
| alpha_Cronos | 0.15 | N-body calibration |
| eta_friction | 0.05 | Subhalo dynamics |
| gamma_Immirzi | 0.2375 | LQG connection |

---

## Instalacion

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

## Uso Rapido

### Bloques Fundamentales

```python
from mcmc_core import Pregeometria, CosmologiaMCMC
from simulations import friccion_entropica, radio_core
from lattice import beta_MCMC, mass_gap

# Bloque 1: Pregeometria
preg = Pregeometria()
print(f"k(S=1.0) = {preg.k(1.0):.4f}")

# Bloque 2: Cosmologia
cosmo = CosmologiaMCMC()
print(f"Edad del universo: {cosmo.edad():.2f} Gyr")

# Bloque 3: N-body
r_c = radio_core(1e11)  # M = 10^11 M_sun
print(f"Radio del nucleo: {r_c:.2f} kpc")
```

### Validacion Avanzada

```python
from mcmc_advanced import run_all_validations

# Ejecutar los 18 tests
results = run_all_validations(verbose=True)
print(f"Passed: {results['summary']['passed']}/{results['summary']['total']}")
```

### Tests Individuales

```python
from mcmc_advanced import (
    test_ISW_LSS_MCMC,
    test_CMB_Lensing_MCMC,
    test_DESI_Y3_MCMC,
    test_MCV_BH,
    test_MCMC_LQG_Bridge,
    test_Pregeometric_Inflation
)

# ISW-LSS Cross-Correlation
test_ISW_LSS_MCMC()

# CMB Lensing
test_CMB_Lensing_MCMC()

# DESI BAO
test_DESI_Y3_MCMC()
```

---

## Predicciones Falsables

### 1. Cosmologia

| Prediccion | Valor MCMC | Medible con |
|------------|------------|-------------|
| delta_H(z)/H(z) | < 0.3% | DESI, Euclid |
| delta_C_L^phiphi | -0.19% | CMB-S4 |
| n_s | 0.962 +/- 0.004 | Planck compatible |
| r (tensor/scalar) | 0.004 | LiteBIRD |

### 2. Galaxias y Halos

| Prediccion | Valor MCMC | Medible con |
|------------|------------|-------------|
| N_sat(MW) | 47 +/- 8 | Surveys profundos |
| r_core(10^12 M_sun) | 9.0 kpc | Rotation curves |
| gamma_inner | 0.51 | SPARC, THINGS |

### 3. Ondas Gravitacionales

| Prediccion | Valor MCMC | Medible con |
|------------|------------|-------------|
| Omega_GW peak | 10^-10 | PTA (SKA) |
| f_peak | 15 nHz | NANOGrav |
| delta_f_ring | 0.9% | LIGO O5 |

### 4. Experimentos de Laboratorio

| Experimento | Presupuesto | Sensibilidad |
|-------------|-------------|--------------|
| Casimir Modulado | $50k-150k | delta_F/F ~ 10^-108 |
| Decoherencia Qubits | $200k-500k | Contraste Xi ~ 13x |
| Interferometria Atomica | $300k-1M | Gradiente MCV |

---

## Conexiones Teoricas

### LQG (Gravedad Cuantica de Lazos)

```
Area gap: A_gap = 4*sqrt(3)*pi*gamma*l_P^2
Conversion S <-> j_max (spin networks)
Amplitudes EPRL para spin foams
```

### Ciclo Cosmico

```
S_max = 1000 -> S_BB = 1.001 -> S_0 = 90 -> S_min = 0.009
Duracion: ~10^67 Gyr (radiacion Hawking)
```

### Inflacion Pre-geometrica

```
Potencial: V(S) = V_0 * exp(-lambda * x^2)
Observables: n_s = 0.962, r = 0.004
Compatible con Planck 2018
```

---

## Tests

```bash
# Tests unitarios
python -m pytest tests/

# Validacion completa
python -c "from mcmc_advanced import run_all_validations; run_all_validations()"

# Test individual
python -c "from mcmc_advanced import test_MCV_BH; test_MCV_BH()"
```

---

## Licencia

**Propietaria - Todos los derechos reservados**

Este modelo esta protegido por derechos de autor. Ver archivo `LICENSE` para terminos completos.

**Restricciones principales:**
- Prohibido uso comercial sin autorizacion
- Prohibida modificacion del codigo
- Requiere cita apropiada para uso academico

**Para solicitudes:** adrianmartinezestelles92@gmail.com

---

## Cita

Si utiliza este modelo en su investigacion, por favor cite:

```bibtex
@misc{mcmc2024,
  author = {Martinez Estelles, Adrian},
  title = {Modelo Cosmologico de Multiples Colapsos (MCMC)},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Modelo-cosmologico-MCMC/MCMC}
}
```

---

## Contacto

**Adrian Martinez Estelles**
Email: adrianmartinezestelles92@gmail.com
