# Preinscripción — Frente 5 (b): serie de resolución del campo de Cronos (k_inner 64/128/256) a A_Sculptor y a 0.05·A_Sculptor con campo instantáneo y balance K + W + (2/5)U_C + W_fric

Congelada 2026-09-21T14:59:20.085772+00:00 en el commit `82d61483c` (que contiene el generador). sha256 de `preregistration.json`: `18c5344d9d0b64f1078b645adeaa3de86a86857164dfd895396c7f1ad64190d1`.

**Pregunta**: ¿Converge el interior del halo al refinar la resolución del campo de Cronos? El criterio de Cronos–Jeans predice que NO a A_Sculptor (q > 1 dentro de r_CJ) y que SÍ a 0.05·A_Sculptor.

**Predicción del criterio**: r_CJ = 0.709 kpc a A_Sculptor (M(<r_CJ) = 1.61e+08 M☉), r_CJ = 0.18018067110148347 kpc a 0.05·A_Sculptor ⟹ letra esperada A.

## Declarado

- Sistema: {'M200_msun': 100000000000.0, 'c': 10.0, 'H0': 67.86705532886631, 'N': 200000, 'seed': 1, 'soft_plummer_kpc': 0.1, 'r_decay_factor': 0.3, 'structural': {'M200': 100000000000.0, 'c': 10.0, 'H0': 67.86705532886631, 'rho_s': np.float64(5723373.75271517), 'r_s': 9.774625607060127, 'r200': 97.74625607060126, 'rho_crit': 127.81475727256635}, 'ics': 'NFW truncado (Kazantzidis+2004), Eddington isótropo; la MISMA semilla en todos los brazos'}
- Integrador: {'theta': 0.7, 'eta_acc': 0.025, 'eta_dyn': 0.02, 'eta_cross': 0.1, 'dt_max_gyr': 0.0512, 'n_levels': 12, 'brute_max': 500, 'profile_every_ticks': 8, 'field_tau_avg_myr': 0.0, 'field_follow_particles': True, 'field_smooth': 1.0, 'scheme': 'como el Nivel A, con el campo de Cronos INSTANTÁNEO: malla logarítmica reconstruida desde la partícula k_inner-ésima actual en cada actualización (sin malla fija ni media móvil)'}
- Balance de energía: K + W + U_self + W_fric con U_self = (2/5)·Σm(−c²ε_c) (la fuerza +c²∇ε_c deriva del funcional −(2/5)c²A∫ρ^{5/2}dV) y W_fric = trabajo acumulado de la fricción con compuerta; el Nivel A usaba K + W + Σm(−c²ε_c), correcto solo con el campo congelado
- t_end = 1.0 Gyr; instantáneas [0.0, 0.25, 0.5, 0.75, 1.0].
- Brazos: ['a', 'bAS_k64', 'bAS_k128', 'bAS_k256', 'b005_k64', 'b005_k128', 'b005_k256'].

## Reglas (congeladas)

- **convergence_metric**: |log10(M_k'(<0.4 kpc)/M_k(<0.4 kpc))| en el instante final común para los saltos 64→128 y 128→256
- **tol_conv**: 0.15
- **series_converges**: ambos saltos ≤ tol_conv
- **letters**: {'A': 'A_Sculptor NO converge y 0.05·A_Sculptor SÍ (predicción del criterio)', 'B': 'ambas convergen', 'C': 'ninguna converge (límite del instrumento)', 'D': 'solo converge A_Sculptor (contrario a la predicción: sospecha de error primero)', 'INDETERMINADO': 'puerta violada (energía, régimen) o corridas ausentes'}
- **publish**: ['cociente b/a por bandas [0.1,0.4], [0.4,1], [1,2.3], [2.3,5], [5,20] kpc', 'M(<0.4 kpc)(t) por brazo', 'W_fric/|E| por brazo', '|ΔE_self/E| y |ΔE_naive/E| por brazo']
- **order**: puertas → letra; ningún umbral se toca tras ver los números
- **puertas**: {'tol_E_newton': 0.005, 'tol_E_self': 0.02, 'weak_regime_eps_max': 0.001, 'stop_rules': {'runaway_well_speed_kms': 1000.0, 'max_wall_hours_per_run': 2.5}}

## Piloto declarado

N = 5e4, 0.25 Gyr, antes de congelar: con el campo instantáneo |ΔE_self/E| = 0.6 % a A_Sculptor y 0.04 % a 0.05·A_Sculptor; U_C/|E| ~ 1e-3; W_fric/|E| < 1e-4; el estimador del Nivel A (malla fija + media móvil) da M(<0.4 kpc) ×7.6 frente a ×2.1 del instantáneo en 0.25 Gyr a la misma A: el estimador pesa. El piloto fija tol_E_self = 0.02 a priori (4× la tolerancia newtoniana, por la aproximación de spline del campo dinámico) y nada más; no incluyó brazo newtoniano ni serie k_inner, así que no anticipa la letra.

## Lo que no puede decidir

- si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): diccionario)
- la amplitud (A_Sculptor es hipótesis del 5E; 0.05 es brazo)
- el interior por debajo de ε_soft = 0.1 kpc
