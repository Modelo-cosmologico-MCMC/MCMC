# DESI DR2 BAO: benchmark ΛCDM y contraste con el fondo MCMC corregido (6A)

Commit del contraste: `465ab7a2090e`; datos:
`data/manifests/desi_dr2_bao.json` (ingesta `bb0c1c9`, cargada con
`require_available` — manifest + estado + esquema + sha256 en tiempo
de ejecución). Contrato de la ronda: **datos ingeridos ≠ likelihood
validada ≠ modelo validado** — cada transición con artefacto y test.

## 1. Likelihood validada (6A.2)

χ² propio ≡ Cobaya 3.6.2 (`bao.desi_dr2.desi_bao_all`, referencia
externa fijada): **máx |Δχ²| = 1.85e-13** sobre
16 vectores sintéticos (tolerancia predeclarada
1e-10); logpdf de Cobaya en m = d: 0 exacto.
Identidad de datos: bao_data tag v2.6 (pin de Cobaya) git-idéntico a la ingesta bb0c1c9 (diff desi_bao_dr2/ = 0 líneas).

## 2. Benchmark ΛCDM (6A.4) — la puerta

| | propio | oficial (arXiv:2503.14738) |
|---|---|---|
| Ω_m | 0.2968 −0.0118/+0.0123 | 0.2975 ± 0.0086 |
| χ²_min (n=13, k=2) | 10.284 | — |
| H0·r_d [km/s] | 10167 | — |

Desviación |ΔΩ_m| = 0.0007 =
**0.09σ oficial** ⟹
**PUERTA PASS** (regla predeclarada: < 1σ). Solo tras este
PASS corre el contraste.

## 3. Contraste MCMC vs ΛCDM (6A.5) — todas las configuraciones

Convenciones idénticas (sampler, semilla, priors, H0·r_d COMÚN — el
MCMC no deriva r_d; likelihood validada). Se publican TODAS las
configuraciones preinscritas — sin selección posterior:

| config | n | χ²_Λ | χ²_M | Δχ² | ΔAIC | ΔBIC | ε_Λ | ΔΩ_m vs ALL |
|---|---|---|---|---|---|---|---|---|
| DESI_ALL | 13 | 10.284 | 10.224 | -0.060 | +3.940 | +5.070 | +0.0190 −0.0399/+0.0420 | +0.0000 |
| DESI_MINUS_BGS | 12 | 9.913 | 9.904 | -0.008 | +3.992 | +4.962 | +0.0173 −0.0399/+0.0433 | -0.0030 |
| DESI_MINUS_LRG_Z0 | 11 | 5.707 | 5.706 | -0.001 | +3.999 | +4.795 | +0.0197 −0.0404/+0.0415 | -0.0060 |
| DESI_MINUS_LRG_Z1 | 11 | 5.863 | 5.861 | -0.003 | +3.997 | +4.793 | +0.0187 −0.0399/+0.0424 | +0.0036 |
| DESI_MINUS_LRGPLUSELG | 11 | 9.396 | 9.236 | -0.160 | +3.840 | +4.636 | +0.0197 −0.0401/+0.0408 | +0.0042 |
| DESI_MINUS_ELG | 11 | 9.845 | 9.735 | -0.110 | +3.890 | +4.686 | +0.0151 −0.0377/+0.0441 | +0.0020 |
| DESI_MINUS_QSO | 11 | 9.605 | 9.520 | -0.086 | +3.914 | +4.710 | +0.0189 −0.0388/+0.0420 | +0.0013 |
| DESI_MINUS_LYA | 11 | 10.186 | 10.149 | -0.037 | +3.963 | +4.759 | +0.0120 −0.0367/+0.0437 | +0.0019 |

(Δ = MCMC − ΛCDM; positivo favorece a ΛCDM.)

**Lectura.** (i) El veredicto es UNIFORME: Δχ² ∈
[-0.160, -0.001] — los dos parámetros extra del
MCMC no compran ajuste en ninguna configuración — y ΔBIC ∈
[+4.64, +5.07] pro-ΛCDM en las 8; el mismo
veredicto que los ajustes de producción v1/v2 (ΔBIC = +14.5 con
n = 1408), ahora sobre DESI DR2. (ii) Ningún bin es decisivo:
quitar cualquiera (incluido lrg-z1 ≡ el antiguo LRG2) mueve Ω_m
menos de 0.0060 y
no cambia el signo de nada. (iii) ε_Λ queda DOMINADO POR EL PRIOR
(posterior ≈ N(0.012, 0.05) en todas las configuraciones): DESI DR2
BAO no mide ε con esta parametrización — razón ESTRUCTURAL,
computada: con z_trans = 8.9 y Δz = 1.5, la transición normalizada
hoy altera E(z ≤ 2.33) en ≤ 1.5e-03 relativo incluso con ε = 0.05
(el rango BAO está entero en la meseta post-transición). BAO DR2
constriñe la FORMA de la expansión, y en ese rango el fondo MCMC del
corpus es casi degenerado con ΛCDM; lo que el dato castiga es la
parsimonia. (iv) ε = +0.0190
−0.0399/+0.0420 en DESI_ALL —
compatible con 0 y con el 0.012 del corpus (sin necesidad, no
excluida).

## Estatuto

condicional y parcial: (a) BAO-only con r_d como calibración común —
sin sector pre-recombinación, el MCMC no reclama r_d; (b) la
degeneración estructural (iii) significa que este frente NO es el
test sensible de ε_Λ — los sensibles son los que sondean z ≳ z_trans
o la física de r_d (fuera del alcance actual); el resultado se
publica igualmente; (c) benchmark contra el valor puntual oficial
publicado — el cruce con cadenas/best-fits oficiales completos queda
pendiente del host bloqueado (data.desi.lbl.gov); (d) próximos pasos
del frente: SNe (mocks Dovekie primero) y combinaciones
DESI + {Dovekie, Pantheon+, Union3} con el mismo contrato.
