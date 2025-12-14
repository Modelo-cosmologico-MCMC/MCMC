# Roadmap de Desarrollo MCMC

## Estado Actual: v2.1.0

- 18/18 validaciones pasando
- Modulos teoricos completos
- Diseno experimental de laboratorio definido

---

## Fase 1: Consolidacion (Completado)

### 1.1 Bloques Fundamentales
- [x] Bloque 0: Estado Primordial
- [x] Bloque 1: Pregeometria
- [x] Bloque 2: Cosmologia
- [x] Bloque 3: N-body
- [x] Bloque 4: Lattice-Gauge

### 1.2 Modulos Avanzados
- [x] ISW-LSS Cross-Correlation
- [x] CMB Lensing
- [x] DESI Y3 BAO
- [x] N-body Box 100
- [x] Zoom-in Milky Way
- [x] JWST High-z
- [x] MCV Black Holes
- [x] Bubble Corrections
- [x] GW Background
- [x] First Principles Derivation
- [x] GW Mergers
- [x] Entropy Map 3D
- [x] MCMC-LQG Bridge
- [x] Cosmic Cycle
- [x] Pre-geometric Inflation
- [x] Unified Framework
- [x] Quantum Effects
- [x] Vacuum Experiments

---

## Fase 2: Integracion CLASS/CAMB (En progreso)

### 2.1 Modificaciones CLASS
- [ ] background_Lrel.c - Lambda_rel(z)
- [ ] id_fluid.c - Fluido MCV/ECV
- [ ] param_mcmc.ini - Parametros MCMC
- [ ] Compilacion y tests

### 2.2 Modificaciones CAMB
- [ ] equations_mcmc.f90
- [ ] params_mcmc.ini
- [ ] Compilacion y tests

### 2.3 Validacion Cruzada
- [ ] Comparacion CLASS vs CAMB
- [ ] Reproducir resultados Python con Boltzmann

---

## Fase 3: Ajuste Bayesiano

### 3.1 Cobaya
- [ ] Configuracion cobaya_mcmc.yaml
- [ ] Likelihood Planck TT+TE+EE
- [ ] Likelihood BAO compilada
- [ ] Cadenas MCMC
- [ ] Analisis posteriors

### 3.2 MontePython
- [ ] montepython_mcmc.param
- [ ] Likelihoods configuradas
- [ ] Cadenas paralelas
- [ ] Comparacion con Cobaya

### 3.3 Resultados Esperados
- [ ] epsilon = 0.012 +/- ?
- [ ] z_trans = 8.9 +/- ?
- [ ] gamma_Zhao = 0.51 +/- ?

---

## Fase 4: Simulaciones N-body Completas

### 4.1 N-body Cronos v2
- [ ] forces.cpp con termino Cronos
- [ ] integrator_cronos.cpp
- [ ] Condiciones iniciales MUSIC modificadas
- [ ] Runs de prueba L=50 Mpc/h

### 4.2 Simulaciones de Produccion
- [ ] Box L=100 Mpc/h, N=1024^3
- [ ] Zoom-in MW con alta resolucion
- [ ] Analisis de perfiles

### 4.3 Comparacion Observacional
- [ ] SPARC rotation curves
- [ ] MW satellites catalog
- [ ] High-z galaxy counts

---

## Fase 5: Ondas Gravitacionales

### 5.1 Predicciones PTA
- [ ] Espectro Omega_GW(f) detallado
- [ ] Comparacion NANOGrav 15yr
- [ ] Predicciones SKA

### 5.2 Eventos LIGO
- [ ] Analisis ringdown modificado
- [ ] Distancias luminosity corregidas
- [ ] Mass gap ontologico

---

## Fase 6: Experimentos de Laboratorio

### 6.1 Casimir Modulado
- [ ] Diseno experimental detallado
- [ ] Simulacion de senal esperada
- [ ] Propuesta de financiacion

### 6.2 Decoherencia Qubits
- [ ] Protocolo experimental
- [ ] Colaboracion con laboratorio
- [ ] Ejecucion piloto

### 6.3 Interferometria Atomica
- [ ] Diseno conceptual
- [ ] Estimacion de sensibilidad
- [ ] Propuesta tecnica

---

## Fase 7: Publicacion

### 7.1 Preprint
- [ ] Manuscrito LaTeX completo
- [ ] Figuras de alta calidad
- [ ] Envio a arXiv

### 7.2 Peer Review
- [ ] Revision PRD/JCAP
- [ ] Respuesta a referees
- [ ] Publicacion

### 7.3 Presentaciones
- [ ] Seminarios institucionales
- [ ] Conferencias internacionales
- [ ] Divulgacion

---

## Timeline Estimado

| Fase | Duracion | Inicio |
|------|----------|--------|
| 1. Consolidacion | Completado | - |
| 2. CLASS/CAMB | 2-3 meses | Inmediato |
| 3. Bayesiano | 2-3 meses | Mes 2 |
| 4. N-body | 3-6 meses | Mes 3 |
| 5. GW | 1-2 meses | Mes 4 |
| 6. Laboratorio | 6-12 meses | Mes 6 |
| 7. Publicacion | 3-6 meses | Mes 8 |

---

## Recursos Necesarios

### Computacionales
- HPC para N-body (>1000 core-hours)
- Storage para simulaciones (~10 TB)
- GPU para analisis ML (opcional)

### Colaboraciones
- Cosmologos observacionales (BAO, CMB)
- Fisicos experimentales (laboratorio)
- Grupos N-body existentes

### Financiacion
- Experimento Casimir: $50k-150k
- Experimento Qubits: $200k-500k
- Experimento Atomico: $300k-1M

---

## Metricas de Exito

1. **Codigo:** Modulos publicados y funcionando
2. **Validacion:** Chi^2 < LCDM en datasets nuevos
3. **Publicacion:** Paper aceptado en revista Q1
4. **Experimento:** Al menos un test de laboratorio propuesto
5. **Comunidad:** Feedback y citas de otros grupos

---

## Contacto

**Adrian Martinez Estelles**
Email: adrianmartinezestelles92@gmail.com
