# Anexo de binning de la preinscripción «perfil σ_los(R) de Sculptor»: lo que la regla congelada no fijaba, congelado antes de los bytes

Congelado 2026-10-23T16:21:48.177941+00:00 en el commit `b3872bd39` (que contiene el generador). sha256 de `binning_annex.json`: `2a06731525a4f3b254ead5d997880847515d3041c036e7682ba47ea1c4d66073`.

De la preinscripción `preregistration.json` (sha256 `604f9c254c7e8283949137d23bb75fb4200b5341a369b1b89f06622ce53af002`), que no se toca.

Estado del dataset al congelar: walker2009 sin manifest y sin bytes en data/raw/walker2009: ningún byte del catálogo visto.

## Lo que fija

- **fichero**: table4.dat (Sculptor; 1818 registros según el ReadMe oficial) con su ReadMe, desde data/raw/walker2009 vía require_available (manifest AVAILABLE, esquema confirmado, sha256 re-verificados)
- **esquema**: manda la descripción byte a byte del ReadMe; papeles por las reglas de dynamics.walker_binning.assign_roles (posición sexagesimal o en grados; velocidad = la única columna en km/s que no es error, la heliocéntrica si hay varias; error = e_<velocidad>; pertenencia = la única columna real cuya explicación habla de pertenencia o etiquetada PM/Pmem/Prob, con valores en [0, 1]; identificador opcional). El número de registros debe coincidir con el File Summary. Cualquier ambigüedad: INDETERMINADO («esquema no confirmado», regla congelada)
- **unidad_estrella**: dos registros son la misma estrella si comparten identificador o distan < 1″ (cierre transitivo); las líneas idénticas byte a byte cuentan una vez; se descartan los registros sin velocidad, sin error positivo o sin posición
- **epocas**: velocidad de la estrella = media pesada por inverso de varianza de sus épocas, error (Σ e⁻²)^−½; sin recorte de binarias ni de variables (la regla congelada no lo incluye)
- **pertenencia**: P_M de la estrella = mínimo de los P_M presentes de sus épocas (conservador frente a contaminación); miembro si P_M ≥ 0.9; sin P_M presente no es miembro; se publica cuántas estrellas tienen épocas a ambos lados de 0.9
- **centro**: (α, δ) J2000 = (01h00m09.4s, −33°42′33″) = (15.039167°, −33.709167°), McConnachie 2012 (AJ 144, 4). Transcrito por el ejecutor sin acceso a la fuente desde el entorno (ADS y CDS denegados por el proxy): verificación del autor pendiente ANTES de los bytes; si lo corrige, se congela un anexo v2 antes de los bytes, nunca después
- **distancia**: D = 79 kpc: la distancia con la que el montaje congelado fija R_half = 260 pc (r_h ≈ 11.3′; tabla 1 de Walker et al. 2009, ApJ 704, 1274: D = 79 ± 4 kpc). θ → R usa la MISMA distancia que fijó la escala del modelo: otra D cambiaría la forma comparada sin cambiar el modelo. Misma advertencia de verificación
- **radio**: R = D·θ con θ la separación angular exacta al centro; radio CIRCULAR (el montaje congelado es esférico)
- **corte**: R ≤ 1000 pc (43.5′ a 79 kpc)
- **bins**: n_bins = ⌊N/150 + ½⌋ (mínimo 1 si N > 0) sobre los N miembros con R ≤ 1 kpc; orden estable por R (empates por orden de aparición en el catálogo); np.array_split: los N mod n_bins bins internos llevan una estrella más
- **R_bin**: mediana de R de las estrellas del bin
- **sigma_bin**: σ² = s² − ⟨e²⟩ con s² la varianza muestral (ddof = 1) alrededor de la media PROPIA del bin y ⟨e²⟩ la media de los errores combinados al cuadrado; σ_err = σ/√(2N); si s² ≤ ⟨e²⟩ en algún bin: INDETERMINADO (puerta)
- **prediccion**: σ_pred(R_bin; β) = columna newton_cronos (la predicción del 5E) evaluada con el código congelado (dynamics.cronos_jeans.sculptor_profile_table(beta=β, R_pc=R_bin)); puerta: ese código reproduce la tabla de la preinscripción en su malla a ≤ 1e-6 km/s con la misma A_Sculptor, o INDETERMINADO
- **puerta_centro**: el centroide de las posiciones de los miembros dista ≤ 10′ del centro fijado (protege frente a un centro mal transcrito o una tabla equivocada), o INDETERMINADO
- **letras**: orden puertas → C → B → A → INDETERMINADO (convención de la casa): C precede a A y B si sus condiciones coinciden; si ninguna letra aplica (χ²_ν(mejor β) > 3 sin la exclusión del pico de C), INDETERMINADO con las cifras publicadas, sin reasignar a B ni a C
- **sin_letra**: χ²_ν de la columna newtoniana por β; sensibilidad a la distancia (D = 75 y 83 kpc, ±1σ): N_bins, χ²_ν(mejor β) y la condición de C; diagnósticos de lectura (registros, épocas, fusiones, centroide)
- **sin_correcciones**: ni rotación interna, ni rotación de perspectiva, ni recorte sigma, ni recorte de binarias; velocidades heliocéntricas tal cual (la dispersión no depende de un desplazamiento constante)

## Números (los lee el analizador; los umbrales de las letras se transcriben de las reglas congeladas)

| clave | valor |
|---|---|
| `pm_min` | 0.9 |
| `target_per_bin` | 150 |
| `R_max_pc` | 1000.0 |
| `distance_kpc` | 79.0 |
| `centre_ra_deg` | 15.039166666666667 |
| `centre_dec_deg` | -33.70916666666667 |
| `merge_radius_arcsec` | 1.0 |
| `centroid_gate_arcmin` | 10.0 |
| `prediction_reproduction_tol_kms` | 1e-06 |
| `chi2_A_max` | 1.5 |
| `chi2_B_max` | 3.0 |
| `C_z_min` | 3.0 |
| `C_inner_bins` | 2 |
| `min_bins` | 4 |
| `distance_sensitivity_kpc` | [75.0, 83.0] |

## Expectativa (E13)

C, ya declarada en la fila sculptor-perfil-sigma-los: el perfil de la literatura es plano ≈ 9–10 km/s frente al pico predicho de 14–25 km/s. A, B, C e INDETERMINADO se publican con el mismo peso.

## Lo que no puede decidir

- la elipticidad de Sculptor (≈ 0.3) frente al montaje esférico
- la rotación interna o de perspectiva (no se corrigen)
- las binarias (no se recortan: inflan σ ligeramente en todos los bins)
- la exactitud del centro y de la distancia transcritos (verificación del autor pendiente)
