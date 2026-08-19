# DESI DR2 BAO: benchmark ΛCDM y contraste con el fondo MCMC corregido (6A)

Commit del contraste: `436f2eb1430f`; datos:
`data/manifests/desi_dr2_bao.json` (ingesta `bb0c1c9`, cargada con
`require_available` — manifest + estado + esquema + sha256 en tiempo
de ejecución). Contrato de la ronda: **datos ingeridos ≠ likelihood
validada ≠ modelo validado** — cada transición con artefacto y test.
Esta versión sustituye a la primera tras la revisión adversarial de la
ronda: fondo normalizado realmente fusionado (H(0) = H0, clausura
plana por llamada — `tests/test_physical_invariants.py`), χ²_min
acotado al soporte del prior, semilla real del sampler, identidad de
datos computada y números estructurales sobre el rango completo.

## 1. Likelihood validada (6A.2)

χ² propio ≡ Cobaya 3.6.2 (`bao.desi_dr2.desi_bao_all`, referencia
externa fijada): **máx |Δχ²| = 1.85e-13** sobre
16 vectores sintéticos (tolerancia predeclarada
1e-10); |logpdf de Cobaya en m = d| =
3.7e-29. Identidad de datos COMPUTADA (sha256
por fichero, offline): 16/16
ficheros del manifest idénticos a los que Cobaya lee en su packages
path (`bao_data` versión `v2.6`) —
PASS. El packages path se recrea con
`scripts/setup_cobaya_packages.py` (única pieza con red;
`COBAYA_PACKAGES_PATH` lo relocaliza).

## 2. Benchmark ΛCDM (6A.4) — la puerta

| | propio | oficial (arXiv:2503.14738) |
|---|---|---|
| Ω_m | 0.2971 −0.0085/+0.0086 | 0.2975 ± 0.0086 |
| χ²_min (n=13, k=2) | 10.284 | — |
| H0·r_d [km/s] | 10155 | — |

Con el fondo normalizado (clausura plana por llamada), Ω_m ES la
fracción de materia del ΛCDM plano — la comparación con el oficial es
entre cantidades homogéneas. Desviación |ΔΩ_m| =
0.0004 =
**0.04σ oficial** ⟹ **PUERTA PASS** (regla
predeclarada: < 1σ). χ²_min por multistart acotado al soporte del
prior, argmin publicado en el artefacto. Solo tras este PASS corre el
contraste.

## 3. Contraste MCMC vs ΛCDM (6A.5) — todas las configuraciones

Convenciones idénticas (emcee 32×4000 en ambos modelos, semilla
global 42 — cadenas bit-reproducibles —, priors comunes, H0·r_d COMÚN:
el MCMC no deriva r_d; likelihood validada; mismo minimizador
acotado). Se publican TODAS las configuraciones preinscritas — sin
selección posterior:

| config | n | χ²_Λ | χ²_M | Δχ² | ΔAIC | ΔBIC | ε_Λ | ΔΩ_m vs ALL |
|---|---|---|---|---|---|---|---|---|
| DESI_ALL | 13 | 10.284 | 10.058† | -0.226 | +3.774 | +4.904 | +0.0154 −0.0380/+0.0433 | +0.0000 |
| DESI_MINUS_BGS | 12 | 9.913 | 9.771† | -0.142 | +3.858 | +4.828 | +0.0160 −0.0377/+0.0426 | -0.0031 |
| DESI_MINUS_LRG_Z0 | 11 | 5.707 | 5.642† | -0.065 | +3.935 | +4.731 | +0.0165 −0.0392/+0.0445 | -0.0044 |
| DESI_MINUS_LRG_Z1 | 11 | 5.863 | 5.799† | -0.064 | +3.936 | +4.732 | +0.0175 −0.0393/+0.0420 | +0.0024 |
| DESI_MINUS_LRGPLUSELG | 11 | 9.396 | 9.065† | -0.330 | +3.670 | +4.465 | +0.0176 −0.0398/+0.0422 | +0.0031 |
| DESI_MINUS_ELG | 11 | 9.845 | 9.591† | -0.254 | +3.746 | +4.542 | +0.0163 −0.0387/+0.0419 | +0.0012 |
| DESI_MINUS_QSO | 11 | 9.605 | 9.362† | -0.243 | +3.757 | +4.552 | +0.0154 −0.0400/+0.0435 | +0.0003 |
| DESI_MINUS_LYA | 11 | 10.186 | 9.986† | -0.200 | +3.800 | +4.596 | +0.0186 −0.0389/+0.0421 | +0.0022 |

(Δ = MCMC − ΛCDM; positivo favorece a ΛCDM. † argmin del χ²_M en la
frontera del soporte del prior (ε = −0.05 y/o z_trans = 1) — ocurre en
8 de 8 configuraciones y se publica en `contrast.json`, no
se oculta: el mínimo es del CIERRE del soporte declarado.)

**Lectura.** (i) El veredicto es UNIFORME: Δχ² ∈
[-0.330, -0.064] — los dos parámetros extra del
MCMC no compran ajuste que compense su coste — y ΔBIC ∈
[+4.47, +4.90] pro-ΛCDM en las 8; el mismo
veredicto que los ajustes de producción v1/v2 repetidos con el fondo
corregido (ΔBIC = +14.50/+14.60 con n = 1408/1422), ahora sobre DESI
DR2. (ii) Ningún bin es decisivo: quitar cualquiera (incluido
lrg-z1 ≡ el antiguo LRG2) mueve la mediana de Ω_m menos de
0.0047 en AMBOS modelos y no cambia el signo de nada.
(iii) ε_Λ queda DOMINADO POR EL PRIOR, y la comparación es contra la
referencia correcta: el prior efectivamente muestreado es
N(0.012, 0.05²) TRUNCADO a (−0.05, 0.10), con mediana
+0.0163 −0.0390/+0.0423 — el cociente
anchura posterior / anchura prior es ≥ 0.99 en las 8
configuraciones. La razón es ESTRUCTURAL y está computada en el
artefacto (`contrast.json → structural`, con z_trans = 8.9
del corpus): con la transición normalizada hoy, ε = 0.05
altera E(z) en ≤ 4.3e-07 relativo sobre TODO el
rango BAO z ≤ 2.33 (máximo en z = 2.33; en z = 2.33:
4.3e-07); la distancia χ² entre los vectores DESI
con y sin ε a parámetros fijos es 4.46e-09,
y tras reabsorber (Ω_m, H0·r_d) el residuo cae a
7.6e-11. BAO DR2 constriñe la
FORMA de la expansión, y en esta parametrización el fondo MCMC es casi
degenerado con ΛCDM en ese rango; lo que el dato castiga es la
parsimonia. (iv) ε = +0.0154
−0.0380/+0.0433 en DESI_ALL —
compatible con 0 y con el 0.012 del corpus (sin necesidad, no
excluida).

## Estatuto

condicional y parcial: (a) BAO-only con r_d como calibración común —
sin sector pre-recombinación, el MCMC no reclama r_d; (b) la
degeneración estructural (iii) significa que este frente NO es el
test sensible de ε_Λ en esta parametrización — los sensibles son los
que sondean z ≳ z_trans o la física de r_d (fuera del alcance
actual); el resultado se publica igualmente; (c) benchmark contra el
valor puntual oficial publicado — el cruce con cadenas/best-fits
oficiales completos queda pendiente del host bloqueado
(data.desi.lbl.gov); (d) próximos pasos del frente: SNe (mocks
Dovekie primero) y combinaciones DESI + {Dovekie, Pantheon+, Union3}
con el mismo contrato.
