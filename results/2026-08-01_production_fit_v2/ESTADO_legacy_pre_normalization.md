# ESTADO: legacy_pre_normalization (ago-2026)

Estos resultados se generaron ANTES de la corrección de normalización
del fondo (rama fix/background-normalization-desi): la forma anterior
de Λ_rel anclaba Λ0 en el punto medio de la transición (F(z_trans) = 1,
sin dividir por F(0)) y Ω_Λ0 quedaba fijada con Ω_m = 0.300 a nivel de
módulo, de modo que H(0) = H0·√(1 + Ω_Λ0·(F(0) − 1)) ≈ 1.0042·H0 en el
punto fiducial y la clausura no era plana al variar Ω_m (en las
medianas de este posterior el sesgo combinado llega a ~1.1 %).

Qué compartían los dos brazos y qué no: el sesgo de CLAUSURA (Ω_Λ0
congelado mientras Ω_m variaba) afectaba por igual a MCMC y ΛCDM; el
sesgo de NORMALIZACIÓN F(0) era SOLO del brazo MCMC — el brazo ΛCDM
corría con ε = 0 exacto, y con ε = 0 la forma legacy es F ≡ 1 —, de
modo que en estos ajustes ε hacía doble papel (amplitud de la
transición + reescalado ~0.42 %·(ε/0.012) de H(0) disponible solo para
el MCMC). La comparación diferencial ΔAIC/ΔBIC usó la misma maquinaria
en ambos brazos, pero ese sistemático asimétrico podía moverla: el
veredicto diferencial quedaba pendiente de la repetición. Los
POSTERIORES ABSOLUTOS (H0, Ω_m, ε, σ8) llevan el sesgo en cualquier
caso.

REPETICIÓN EJECUTADA (10-ago-2026, fondo corregido, misma semilla 42 y
misma configuración 24×8000): `results/2026-08-10_production_fit_v2/`.
El veredicto diferencial SE CONFIRMA — ΔAIC = +4.08, ΔBIC = +14.60
pro-ΛCDM (legacy: +4.01/+14.53), ε = 0.018 −0.041/+0.043 compatible
con 0 — ahora con posteriores absolutos sin el sesgo de normalización.

Estos ficheros se conservan íntegros como registro histórico y no
deben citarse como estado vigente: la referencia vigente es la
repetición.
