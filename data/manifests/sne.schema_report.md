# schema_report — union3 / union3_1 / des_dovekie

Generado de los ficheros ingeridos (nunca de memoria).

## union3 — `mu_mat_union3_cosmo=2_mu.fits`
- HDU 0: PrimaryHDU, shape = (23, 23)
- Matriz (23, 23): fila/col 0 = [0|z_i] y [μ_i|cov] — N_bins = 22
- z (3 primeros): [0.05, 0.1, 0.15]; μ (3 primeros): [36.63, 38.235, 39.148]

## union3_1 (UNITY1.7) — `mu_mat_union3.1_UNITY1.7_template_cosmo=2_0_mu.fits`
- HDU 0: PrimaryHDU, shape = (23, 23)
- Matriz (23, 23): fila/col 0 = [0|z_i] y [μ_i|cov] — N_bins = 22
- z (3 primeros): [0.05, 0.1, 0.15]; μ (3 primeros): [36.763, 38.368, 39.28]

## union3_1 (UNITY1.8) — `mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits`
- HDU 0: PrimaryHDU, shape = (23, 23)
- Matriz (23, 23): fila/col 0 = [0|z_i] y [μ_i|cov] — N_bins = 22
- z (3 primeros): [0.05, 0.1, 0.15]; μ (3 primeros): [36.765, 38.371, 39.267]

## des_dovekie — DES-Dovekie_HD.csv
- Cabecera: `# zHD       = redshift in CMB frame with VPEC correction`
- Filas de datos: 1828

## des_dovekie — STAT+SYS.npz / STATONLY.npz
- STAT+SYS: clave `nsn`, shape (1,), dtype int64
- STAT+SYS: clave `cov`, shape (1657110,), dtype float32
- STAT+SYS: clave `allow_pickle`, shape (), dtype bool
- STATONLY: clave `nsn`, shape (1,), dtype int64
- STATONLY: clave `cov`, shape (1657110,), dtype float32
- STATONLY: clave `allow_pickle`, shape (), dtype bool

El consumidor de referencia es el script OFICIAL del release (DES-Dovekie-SN_Likelihood.py, ingerido junto a los datos): la construcción exacta de la covarianza se toma de ahí, no se asume.
