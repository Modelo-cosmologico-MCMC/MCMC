# Preinscripción — La MCV (c) en Sculptor: el perfil σ_los(R) binado de Walker et al. 2009 frente al Jeans isotermo con autointercambio empalmado con una envoltura NFW de molestia

Congelada 2026-10-23T20:42:49.575387+00:00 en el commit `1e0615b02` (que contiene el generador). sha256 de `preregistration.json`: `2ed76f7f0bf4e8c5364d5bc9811a67e7712bdac95bc915ccf22093f7c4aee006`.

Banco de modelos `results/2026-10-23_mcv_sculptor/model_bank.json` (sha256 `311f919057f1fbe8dad29fd45fe7dd03c124ca0338c0993139ba3087fc9c3de8`, 13275 puntos, commit `1e0615b02`). Observable: el perfil binado de PR-W (anexo sha256 `2a06731525a4…`). Estado del dataset: walker2009 sin manifest y sin bytes: ningún byte del catálogo visto.

## Modelo

- ∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E), con ρ_E = 0 en Sculptor (la ECV de fondo es ~1e-6 de ρ_M0)
- Γ_x = ρ_M·(σ_x/m)·⟨v_rel⟩, ⟨v_rel⟩ = 4σ/√π; r₁: Γ_x(r₁)·t_edad = 1 con la densidad NFW
- r < r₁: ρ_M = ρ_M0·exp[−(Φ − Φ₀)/σ₀²] con las estrellas de Plummer dentro de Φ
- empalme en r₁ con la envoltura NFW: ρ_M y M(<r) continuos (Kaplinghat, Tulin y Yu 2016)
- Estrellas: trazador de Plummer del montaje congelado del 5E: a = R_half = 260 pc, L_V = 2.6e6 L☉, M⋆ = Υ⋆·L_V, Υ⋆ ∈ {1, 2, 3} (molestia discreta), también en el potencial.
- Jeans: Jeans esférico con β constante del brazo de la ley local (−0.5, 0, +0.3; molestia discreta), dynamics/jeans.py (malla de 800 radios hasta 120a; proyección hasta 120a).
- Ramas: todas las ramas del empalme: 'con núcleo' la de menor ρ₀, 'densa' las demás; se descartan las raíces fuera del dominio de la rejilla del empalme (σ₀ ∈ [0.5, 300] km/s, u = ρ₀/ρ_env(r₁) ≤ 1e10) y las que tienen r₁ ≥ r200.

## Molestias y ajuste

- **envelope**: c = c200_DM14(M200)·10^(0.11·k): mediana de Dutton y Macciò 2014 (c200, z = 0, log10 c = 0.905 − 0.101·log10[M200/(10¹² h⁻¹ M☉)], h = 0.674), extrapolada por debajo de su rango simulado, y k la desviación en unidades de su dispersión (0.11 dex)
- **prior**: log10 M200 ∈ [8.5, 10.5] y |k| ≤ 2 (dos dispersiones): incluye las envolturas del dato previo (10⁹ con c = 20, k ≈ +0.7; 3×10⁹ con c = 15, k ≈ 0)
- **box**: log10 M200 ∈ [7.5, 11.5] y |k| ≤ 5: la caja en que B y C se evalúan
- **grid**: fina en las previas (0.125 dex en M200, 0.5 en k) y gruesa en la caja (0.25 dex, 1 en k): 295 envolturas; el mínimo de χ² se toma sobre la rejilla congelada, sin refinamiento
- **σ_x/m**: [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0] cm²/g; un solo valor para el sistema; cada valor de la rejilla se ajusta por separado (σ_x/m no se ajusta en continuo).
- **chi2**: χ² = Σ_bins [(σ_mod(R_bin) − σ_obs)/σ_err]², minimizado sobre (M200, c) de la región, Υ⋆ y β
- **nu**: ν = N_bins − 2 (las dos coordenadas de la envoltura)
- **fits**: «cabe» con un σ_x/m ⇔ p(χ² ≥ χ²_mín | ν) ≥ 0.05 (prueba χ² al 5 %)

## Reglas (congeladas)

- **A**: existe σ_x/m ≤ 10 cm²/g de la rejilla con el que cabe la rama con núcleo con la envoltura dentro de las previas
- **B**: no A, pero existe σ_x/m ≤ 50 cm²/g con el que cabe alguna rama con alguna envoltura de la caja: solo en la rama densa, solo con la envoltura fuera de las previas o solo con 10 < σ_x/m ≤ 50 (este último caso lo añade el ejecutor: el texto del autor no lo nombra y es «cabe forzando», no A ni C)
- **C**: no cabe con ningún σ_x/m ≤ 50 cm²/g, ninguna rama ni envoltura de la caja
- **INDETERMINADO**: puertas: walker2009 no AVAILABLE; las del binning de PR-W (esquema, centroide, N_bins < 4, σ sin definir); ν < 2; sha256 del banco distinto del congelado; los puntos de reproducción no se reproducen (ramas o σ_los a 1e-6); la búsqueda exhaustiva del empalme en el punto decisivo no da las mismas ramas que el banco; C con el mejor ajuste en el borde de la caja
- **order**: puertas → A → B → C (mutuamente excluyentes por construcción)
- **verdict**: una letra; ninguna es afirmación sobre el tratado (E8): sitúa el equilibrio (c) frente a la forma y la amplitud del perfil de un sistema

## Expectativa (E13, escrita antes de congelar)

A, con σ_x/m pequeño. El dato previo del autor (reproducido aquí sin letra) dice que la rama con núcleo alcanza la masa de media luz de Sculptor (M(<r_½) ≈ 2.05×10⁷ M☉, el estimador 4σ²R_half/G con σ = 9.2 km/s) con σ_x/m ≈ 0.3–0.5 cm²/g y envolturas de 10⁹–3×10⁹ M☉ dentro de las previas, y que con σ_x/m ≥ 2 se queda corta; un perfil casi plano de ≈ 9–10 km/s hasta 1 kpc es lo que da una esfera isoterma con una envoltura así. Si el perfil binado exige más masa central que esas envolturas, la expectativa pasa a B (rama densa o envoltura fuera de las previas); C no se espera. El número no es señal: una A dice que el equilibrio (c) acomoda a Sculptor con algún σ_x/m, no que lo prediga; si solo cabe en el extremo σ_x/m ≲ 0.1 no distingue la MCV de la materia oscura fría (se publica el intervalo de σ_x/m que cabe), y σ_x/m global solo se prueba fuera de la muestra (PR-S).

## Dato previo, sin letra

Del autor: «la rama con núcleo llega a 2.05×10⁷ M☉ con σ_x/m ≈ 0.3–0.5 cm²/g y envolturas de 10⁹ (c = 20) a 3×10⁹ M☉ (c = 15); con σ_x/m ≥ 2 se queda en 0.7–1.4×10⁷».

Reproducción con el módulo: M(<r_½ = 347 pc) de la rama con núcleo, frente a M_half = 4σ²R_half/G = 2.05e+07 M☉ (en 10⁷ M☉; materia de la MCV / total con estrellas).

| envoltura | σ_x/m | sin bariones | Υ⋆ = 1 | Υ⋆ = 2 | Υ⋆ = 3 |
|---|---|---|---|---|---|
| 1e+09 M☉, c = 20 | 0.3 | 1.75 / 1.75 | 1.75 / 1.88 | 1.75 / 2.02 | 1.75 / 2.15 |
| 1e+09 M☉, c = 20 | 0.5 | 1.70 / 1.70 | 1.70 / 1.83 | 1.69 / 1.96 | 1.69 / 2.09 |
| 1e+09 M☉, c = 20 | 1 | 1.45 / 1.45 | 1.47 / 1.60 | 1.49 / 1.75 | 1.51 / 1.91 |
| 1e+09 M☉, c = 20 | 2 | 1.15 / 1.15 | 1.22 / 1.35 | 1.27 / 1.54 | 1.33 / 1.73 |
| 1e+09 M☉, c = 20 | 5 | 0.86 / 0.86 | 0.96 / 1.09 | 1.06 / 1.33 | 1.17 / 1.57 |
| 1e+09 M☉, c = 20 | 10 | 0.71 / 0.71 | 0.84 / 0.97 | 0.99 / 1.25 | 1.16 / 1.55 |
| 3e+09 M☉, c = 15 | 0.3 | 1.91 / 1.91 | 1.91 / 2.04 | 1.90 / 2.17 | 1.89 / 2.29 |
| 3e+09 M☉, c = 15 | 0.5 | 1.67 / 1.67 | 1.67 / 1.80 | 1.67 / 1.94 | 1.68 / 2.08 |
| 3e+09 M☉, c = 15 | 1 | 1.21 / 1.21 | 1.25 / 1.38 | 1.28 / 1.55 | 1.32 / 1.72 |
| 3e+09 M☉, c = 15 | 2 | 0.86 / 0.86 | 0.91 / 1.04 | 0.96 / 1.23 | 1.02 / 1.41 |
| 3e+09 M☉, c = 15 | 5 | 0.57 / 0.57 | 0.62 / 0.75 | 0.68 / 0.94 | 0.74 / 1.14 |
| 3e+09 M☉, c = 15 | 10 | 0.44 / 0.44 | 0.49 / 0.62 | 0.55 / 0.81 | 0.61 / 1.01 |

## Lo que no puede decidir

- σ_x/m global: un sistema no lo calibra (se prueba fuera de la muestra en PR-S)
- el término de la ECV (ρ_E = 0 aquí) y el intercambio con ella
- la anisotropía más allá de β constante, la elipticidad, las binarias, las mareas
- la evolución gravotérmica (el equilibrio se supone; t_c se publica sin letra)
