# Ontologia del Modelo MCMC

## 1. Componentes Fundamentales

### 1.1 MCV - Materia de Curvatura Variable

La MCV es una componente ontologica que emerge de los gradientes de densidad materia/espacio:

```
rho_MCV = rho_vac * (1 + alpha * Xi^beta)
```

Donde:
- `rho_vac = 9.47e-27 kg/m^3` (densidad de vacio)
- `alpha ~ 10^27 m^3/kg` (coeficiente de acoplamiento)
- `beta = 10` (exponente tensorial)
- `Xi = |nabla(rho_m/rho_s)|` (gradiente ontologico)

**Efectos observables:**
- Modifica perfiles de densidad galáctica (gamma_Zhao = 0.51)
- Genera cores en halos de materia oscura
- Suprime subhalos de baja masa

### 1.2 ECV - Energia de Curvatura Variable

La ECV modifica la constante cosmológica efectiva:

```
Lambda_rel(z) = 1 + epsilon * tanh((z_trans - z)/Delta_z)
```

Parametros calibrados:
- `epsilon = 0.012` (amplitud de variacion)
- `z_trans = 8.9` (redshift de transicion)
- `Delta_z = 1.5` (anchura de transicion)

**Efectos observables:**
- Modifica H(z) a alto redshift (< 0.3%)
- Reduce tension H0 de 4.8 sigma a 1.5 sigma
- Reduce tension S8 de ~3 sigma a < 1 sigma

---

## 2. Sellos Entropicos

Los sellos marcan transiciones ontologicas fundamentales:

| Sello | Valor S | Redshift | Energia | Descripcion |
|-------|---------|----------|---------|-------------|
| S1 | 1.0 | 10^32 | 1.22e19 GeV | Escala de Planck |
| S2 | 10.0 | 10^12 | 0.2 GeV | Transicion QCD |
| S3 | 25.0 | 10^15 | 100 GeV | Electrodebil |
| S4 | 1.001 | 10^30 | 10^16 GeV | Pre-geometrico |
| S5 | 80.0 | 1100 | 0.3 eV | Recombinacion |
| S0 | 90.0 | 0 | - | Estado actual |

---

## 3. Ley de Cronos

La Ley de Cronos relaciona el flujo temporal local con el gradiente ontologico:

```
d(tau)/dt = exp(-Xi)
```

**Implicaciones:**
- El tiempo fluye mas lento donde Xi es grande (cerca de masas)
- Recupera dilatación temporal relativista como limite
- Predice efectos cronologicos medibles en laboratorio

### 3.1 Gradiente Ontologico Xi

```
Xi^(M/S) = |nabla(rho_m/rho_s)|
```

Interpretacion:
- `rho_m`: densidad de materia local
- `rho_s`: densidad de "espacio tensorial" (referencia)
- El gradiente mide el "estres" ontologico local

---

## 4. Potencial Ontologico Psi

El potencial ontologico adimensional:

```
Psi = (rho_m * c^2) / (rho_Pl * c^2) = rho_m / rho_Pl
```

Donde `rho_Pl = c^5 / (hbar * G^2) = 5.16e96 kg/m^3`

**Valores tipicos:**
- Esfera de tungsteno (100g, 1cm): Psi ~ 10^14
- Agujero negro estelar: Psi ~ 10^16
- Escala de Planck: Psi ~ 1

---

## 5. Conexion con LQG

### 5.1 Area Gap

```
A_gap = 4 * sqrt(3) * pi * gamma * l_P^2
```

Con parametro de Immirzi `gamma = 0.2375`:
- `A_gap = 1.35e-69 m^2`
- `A_gap/A_P = 5.17`

### 5.2 Conversion S <-> j_max

```
S = gamma * sqrt(j_max * (j_max + 1))
j_max = (1/2) * (-1 + sqrt(1 + 4*S^2/gamma^2))
```

Permite mapear entropia MCMC a spin networks de LQG.

### 5.3 Amplitudes EPRL

```
A_EPRL = sum_j (2j+1) * {6j}^2 * exp(-j(j+1)*hbar/(8*pi*gamma^2*A_P))
```

---

## 6. Ciclo Cosmico

### 6.1 Fases del Ciclo

```
S_max (1000) -> Big Bang -> Expansion -> S_0 (90) -> S_min (0.009) -> Reconversion -> S_max
```

### 6.2 Duracion del Ciclo

Dominado por radiacion Hawking:
```
t_ciclo ~ 10^67 Gyr
```

### 6.3 Canales de Reconversion

| Canal | Eficiencia | Escala temporal |
|-------|------------|-----------------|
| ECV -> MCV | 10^-2 | 0.1 Gyr |
| Radiacion Hawking | 10^-10 | 10^67 Gyr |
| Decaimiento vacio | 10^-1 | 10^10 Gyr |
| Aniquilacion | 1.0 | 10^-6 Gyr |

---

## 7. Inflacion Pre-geometrica

### 7.1 Estado S4

El estado S4 es pre-geometrico (dimension efectiva = 0):
```
S_S4 = 1.001
```

### 7.2 Potencial Entropico

```
V(S) = V_0 * exp(-lambda * ((S - S_S4)/S_scale)^2)
```

Con:
- `V_0 = 6.54e24 GeV^4`
- `lambda = 0.02`
- `S_scale = 3.0`

### 7.3 Parametros Slow-Roll

```
epsilon = (M_Pl^2 / 2) * (V'/V)^2
eta = M_Pl^2 * (V''/V)
```

Resultados:
- `epsilon ~ 0.0002` (durante inflacion)
- `eta ~ -0.017`

### 7.4 Observables CMB

| Observable | MCMC | Planck 2018 |
|------------|------|-------------|
| n_s | 0.962 +/- 0.004 | 0.9649 +/- 0.0042 |
| r | 0.004 | < 0.064 |
| alpha_s | -7.4e-5 | - |

---

## 8. Efectos Cuanticos (Qubit Tensorial)

### 8.1 Qubit Cosmico

Estado cuantico fundamental:
```
|psi> = alpha|0> + beta|1>
```

Con entropía de von Neumann:
```
S_vN = -Tr(rho * log(rho))
```

### 8.2 Entrelazamiento Cosmico

Para estado de Bell:
```
|Phi+> = (|00> + |11>)/sqrt(2)
S_entrelazamiento = ln(2) = 0.693
Concurrencia = 1.0
```

### 8.3 Decoherencia Cosmica

Tasa de decoherencia:
```
Gamma_dec = kappa_MS * Xi^2 * T / hbar
```

Con `kappa_MS ~ 10^-50 m^2/s`.

---

## 9. Criterios de Falsacion

### 9.1 Para MCV

1. **Perfiles galacticos:** Si gamma_inner != 0.51 +/- 0.05
2. **Subhalos MW:** Si N_sat < 35 o > 60
3. **Cores:** Si r_core(10^12 M_sun) < 5 kpc o > 15 kpc

### 9.2 Para ECV

1. **H(z):** Si delta_H/H > 1% para z < 3
2. **CMB Lensing:** Si delta_C_L > 0.5%
3. **BAO:** Si chi^2_MCMC > chi^2_LCDM significativamente

### 9.3 Para Cronos

1. **Laboratorio:** Si no hay correlacion senal-modulacion tras >100h
2. **Dependencia:** Si efecto no escala como 1/r^4
3. **Independencia:** Si efecto depende del material (no solo densidad)
