# schema_report — desi_dr2_bao (generado de los ficheros reales)

Fuente: github.com/CobayaSampler/bao_data @ bb0c1c9009dc76d1391300e169e8df38fd1096db, subdir desi_bao_dr2/

## Estructura observada

- `*_mean.txt`: cabecera literal `# [z] [value at z] [quantity]`;
  filas = `z  valor  cantidad` con cantidad ∈ {DV_over_rs,
  DM_over_rs, DH_over_rs} — ADIMENSIONALES (divididos por r_s = r_d).
- `*_cov.txt`: matriz N×N en texto plano, mismo orden de filas que
  el mean correspondiente (verificado: N_cov = N_mean en los 8 pares).

## Inventario (fichero: filas mean / dimensión cov, cantidades)

- `desi_gaussian_bao_ALL_GCcomb_mean.txt`: 13 filas / cov 13×13; z_eff = [0.295, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33]; cantidades = ['DH_over_rs', 'DM_over_rs', 'DV_over_rs']
- `desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_mean.txt`: 1 filas / cov 1×1; z_eff = [0.295]; cantidades = ['DV_over_rs']
- `desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt`: 2 filas / cov 2×2; z_eff = [1.321]; cantidades = ['DH_over_rs', 'DM_over_rs']
- `desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_mean.txt`: 2 filas / cov 2×2; z_eff = [0.934]; cantidades = ['DH_over_rs', 'DM_over_rs']
- `desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt`: 2 filas / cov 2×2; z_eff = [0.51]; cantidades = ['DH_over_rs', 'DM_over_rs']
- `desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt`: 2 filas / cov 2×2; z_eff = [0.706]; cantidades = ['DH_over_rs', 'DM_over_rs']
- `desi_gaussian_bao_Lya_GCcomb_mean.txt`: 2 filas / cov 2×2; z_eff = [2.33]; cantidades = ['DH_over_rs', 'DM_over_rs']
- `desi_gaussian_bao_QSO_GCcomb_mean.txt`: 2 filas / cov 2×2; z_eff = [1.484]; cantidades = ['DH_over_rs', 'DM_over_rs']

## Mapa de bins oficiales (ALL_GCcomb, 13 componentes)

- índice 0: z_eff = 0.295 (bgs) — DV_over_rs = 7.94167639
- índice 1: z_eff = 0.51 (lrg-z0) — DM_over_rs = 13.58758434
- índice 2: z_eff = 0.51 (lrg-z0) — DH_over_rs = 21.86294686
- índice 3: z_eff = 0.706 (lrg-z1) — DM_over_rs = 17.35069094
- índice 4: z_eff = 0.706 (lrg-z1) — DH_over_rs = 19.45534918
- índice 5: z_eff = 0.934 (lrgpluselg) — DM_over_rs = 21.57563956
- índice 6: z_eff = 0.934 (lrgpluselg) — DH_over_rs = 17.64149464
- índice 7: z_eff = 1.321 (elg) — DM_over_rs = 27.60085612
- índice 8: z_eff = 1.321 (elg) — DH_over_rs = 14.17602155
- índice 9: z_eff = 1.484 (qso) — DM_over_rs = 30.51190063
- índice 10: z_eff = 1.484 (qso) — DH_over_rs = 12.81699964
- índice 11: z_eff = 2.33 (lya) — DH_over_rs = 8.631545674846294
- índice 12: z_eff = 2.33 (lya) — DM_over_rs = 38.988973961958784

## Columnas directas vs derivadas

- DIRECTAS del release: z, valor, cantidad, covarianza.
- DERIVADAS por nuestro código: identificador de bin oficial
  (mapa z_eff→bin de la tabla anterior; LRG1 ≡ lrg-z0,
  LRG2 ≡ lrg-z1 — el identificador oficial se guarda siempre),
  índices de leave-one-bin-out.

Con esta confirmación el estado pasa de SCHEMA_UNVERIFIED a AVAILABLE.
