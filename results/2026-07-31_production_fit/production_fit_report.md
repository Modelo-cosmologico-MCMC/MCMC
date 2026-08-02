# Ajuste de producción MCMC vs ΛCDM

Datos: 31 H(z) de cronómetros + 6 BAO + 1371 SNe (cov completa) (n = 1408). Semilla 42, 32 walkers × 4000 pasos, descarte 2000, thin 4. Priors: apéndice F (H0 ~ N(67.4, 5²)).

ΛCDM ajustado sobre LOS MISMOS datos con la MISMA maquinaria (ε = 0 exacto, Prop. A.1); M_B de SNe marginalizada analíticamente en ambos modelos por igual (no cuenta en k).

## Posteriores (mediana ± 1σ)

### MCMC (k = 4)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 66.0900 | −1.6179 / +1.6828 |
| Omega_m | 0.3436 | −0.0245 / +0.0267 |
| epsilon | 0.0143 | −0.0379 / +0.0415 |
| z_trans | 9.1880 | −4.5512 / +4.7844 |

Aceptación 0.57; τ: 57, 54, 56, 51; convergencia OK

### ΛCDM (k = 2)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 66.5538 | −1.2826 / +1.3156 |
| Omega_m | 0.3399 | −0.0238 / +0.0247 |

Aceptación 0.72; τ: 31, 32; convergencia OK

## χ² por dataset (en el máximo a posteriori de cada modelo)

| Dataset | χ² MCMC | χ² ΛCDM |
|---|---|---|
| Hz | 14.42 | 14.42 |
| SNe | 1210.09 | 1210.05 |
| BAO | 1.55 | 1.56 |
| total | 1226.07 | 1226.03 |

## Criterios de información

Fórmulas: AIC = 2k − 2·ln L_max; BIC = k·ln(n) − 2·ln L_max. k_MCMC = 4 (H0, Ωm, ε, z_trans); k_ΛCDM = 2 (H0, Ωm); n = 1408.

| Modelo | k | ln L_max | AIC | BIC |
|---|---|---|---|---|
| MCMC | 4 | -613.03 | 1234.07 | 1255.07 |
| ΛCDM | 2 | -613.02 | 1230.03 | 1240.53 |

**ΔAIC (MCMC − ΛCDM) = +4.03; ΔBIC (MCMC − ΛCDM) = +14.53** (negativo favorece al MCMC; positivo, a ΛCDM). El resultado se publica sea cual sea.

Cadenas: `chains_mcmc.npz`, `chains_lcdm.npz` (en output/).
