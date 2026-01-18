# Predicciones Falsables del Modelo MCMC

## Principio Fundamental

El modelo MCMC es cientifico porque hace predicciones especificas que pueden ser falsadas experimentalmente. Este documento lista las predicciones cuantitativas que, de no cumplirse, refutarian el modelo.

---

## 1. Predicciones Cosmologicas

### 1.1 Expansion del Universo H(z)

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| delta_H(z<3)/H | < 0.3% | > 1% | DESI, Euclid |
| H(z=2)/H_LCDM(z=2) | 0.997 | < 0.99 o > 1.01 | DESI BAO |
| Transicion z_trans | 1.0 | < 0.5 o > 2 | SNe + BAO |

**Criterio de falsacion:** Si observaciones futuras muestran delta_H/H > 1% para z < 3, el modelo ECV queda refutado.

### 1.2 CMB Lensing

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| delta_C_L^phiphi | -0.19% | > 0.5% | CMB-S4 |
| A_lens | 0.998 | < 0.95 o > 1.05 | Simons Observatory |

**Criterio de falsacion:** Si A_lens difiere de 1.0 por mas de 5% en direccion opuesta a MCMC.

### 1.3 Inflacion

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| n_s | 0.962 +/- 0.004 | < 0.95 o > 0.98 | Planck, LiteBIRD |
| r | 0.004 | > 0.01 | LiteBIRD |
| alpha_s | -7.4e-5 | > 10^-3 | CMB futuro |

**Criterio de falsacion:** Si r > 0.01 o n_s < 0.95, la inflacion pre-geometrica queda refutada.

---

## 2. Predicciones Galacticas

### 2.1 Perfiles de Densidad

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| gamma_inner | 0.51 +/- 0.05 | < 0.3 o > 0.7 | SPARC, THINGS |
| r_core(10^12 M_sun) | 9.0 kpc | < 5 o > 15 kpc | Rotation curves |

**Criterio de falsacion:** Si galaxias masivas muestran gamma_inner ~ 1.0 (cusps NFW puras).

### 2.2 Satelites de la Via Lactea

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| N_sat(M>10^8) | 47 +/- 8 | < 35 o > 60 | LSST, Rubin |
| Supresion vs CDM | 45% | < 30% o > 60% | Surveys profundos |

**Criterio de falsacion:** Si se descubren mas de 70 satelites masivos (>10^8 M_sun).

### 2.3 Too-Big-To-Fail

| Prediccion | Valor MCMC | Rango falsador | Criterio |
|------------|------------|----------------|----------|
| V_max reduccion | 15-25% | < 5% | Dinamica satelites |
| Tension TBTF | 0.8 sigma | > 2 sigma | Comparacion CDM |

**Criterio de falsacion:** Si la tension TBTF persiste > 2 sigma con MCMC.

---

## 3. Predicciones de Ondas Gravitacionales

### 3.1 Fondo Estocastico

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| f_peak | 15 nHz | < 5 o > 50 nHz | NANOGrav, SKA |
| Omega_peak | 10^-10 | > 10^-8 | PTA |

**Criterio de falsacion:** Si NANOGrav confirma Omega_GW > 10^-8 a 15 nHz con indice gamma = 4.3.

### 3.2 Ringdown Modificado

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| delta_f_ring | 0.9% | > 3% | LIGO O5, Einstein Telescope |
| delta_tau_ring | < 1% | > 3% | LISA |

**Criterio de falsacion:** Si se mide delta_f_ring > 3% o en direccion opuesta.

---

## 4. Predicciones de Laboratorio

### 4.1 Efecto Casimir Modulado

| Prediccion | Valor MCMC | Rango falsador | Setup |
|------------|------------|----------------|-------|
| delta_F/F | ~10^-108 | No detectable en 100h | Placas + masa modulada |
| Dependencia | ~1/r^4 | != 1/r^4 | Variacion distancia |

**Criterio de falsacion:** Si se detecta senal que NO escala como 1/r^4.

### 4.2 Decoherencia de Qubits

| Prediccion | Valor MCMC | Rango falsador | Setup |
|------------|------------|----------------|-------|
| Contraste Xi | 13x | < 2x | Qubit + masa cercana/lejana |
| Correlacion T2 | Positiva | Negativa o nula | Variacion posicion masa |

**Criterio de falsacion:** Si decoherencia NO correlaciona con gradiente de masa.

### 4.3 Interferometria Atomica

| Prediccion | Valor MCMC | Rango falsador | Setup |
|------------|------------|----------------|-------|
| Fase adicional | Proporcional a Xi | Independiente de Xi | Atom interferometer |

**Criterio de falsacion:** Si fase medida no depende del gradiente ontologico.

---

## 5. Predicciones sobre Agujeros Negros

### 5.1 Ley de Cronos cerca del Horizonte

| Prediccion | Valor MCMC | Rango falsador | Criterio |
|------------|------------|----------------|----------|
| Xi(horizonte) | ~1-10 | < 0.1 o > 100 | EHT, ondas GW |
| dt/dt0 | exp(Xi) | != exp(Xi) | Relojes en orbita |

**Criterio de falsacion:** Si dilatacion temporal NO sigue exp(Xi).

### 5.2 Mass Gap Ontologico

| Prediccion | Valor MCMC | Rango falsador | Instrumento |
|------------|------------|----------------|-------------|
| Supresion 2.5-5 M_sun | 35-90% | < 10% | LIGO/Virgo |

**Criterio de falsacion:** Si se detectan multiples objetos en el gap con frecuencia normal.

---

## 6. Predicciones Teoricas

### 6.1 Conexion LQG

| Prediccion | Valor MCMC | Rango falsador | Criterio |
|------------|------------|----------------|----------|
| gamma_Immirzi | 0.274 | != valor LQG | Consistencia teorica |
| Area gap ratio | 5.17 | < 4 o > 7 | Calculo LQG |

**Criterio de falsacion:** Si calculo riguroso de LQG da gamma incompatible.

### 6.2 Ciclo Cosmico

| Prediccion | Valor MCMC | Rango falsador | Criterio |
|------------|------------|----------------|----------|
| t_ciclo | ~10^67 Gyr | < 10^50 Gyr | Radiacion Hawking |

**Criterio de falsacion:** Si se demuestra que reconversion es mas rapida.

---

## 7. Tabla Resumen de Falsabilidad

| Area | Prediccion clave | Valor | Timeline |
|------|------------------|-------|----------|
| Cosmologia | delta_H/H < 0.3% | 0.3% | DESI 2024-2027 |
| CMB | delta_C_L < 0.2% | -0.19% | CMB-S4 2027+ |
| Inflacion | r < 0.01 | 0.004 | LiteBIRD 2028+ |
| Galaxias | N_sat ~ 47 | 47+/-8 | LSST 2025+ |
| Perfiles | gamma ~ 0.51 | 0.51 | SPARC actualizado |
| GW | f_peak ~ 15 nHz | 15 nHz | SKA 2030+ |
| Lab | Correlacion Xi-T2 | Positiva | Experimento propuesto |

---

## 8. Conclusiones

El modelo MCMC es falsable porque:

1. **Especificidad:** Hace predicciones numericas precisas
2. **Accesibilidad:** Las predicciones son medibles con tecnologia actual o proxima
3. **Discriminacion:** Las predicciones difieren de LCDM de forma medible
4. **Consistencia interna:** 18/18 tests pasan con datos actuales

**Estado:** El modelo NO ha sido falsado hasta la fecha (Diciembre 2024).
