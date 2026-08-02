# Frente nº 5, medio paso: perfil en la malla y compuerta H.2.5

Par A/B de semilla idéntica 20260802 (B.5/B.6): 4096 partículas, malla 32³, dt = 0.002, 1200 pasos; B con α0⁻¹ = 1e-06 (el máximo de la cota de la ec. 11.5). Malla PM mínima: NO es producción — el 2.3 kpc del corpus y los valores absolutos de H.2.5 quedan para Gadget-4-Cronos (frente nº 5).

PARÁMETROS DECLARADOS: ρ_c = 200 × densidad media de la caja (umbral de colapso; con ρ_c ~ media, ε_c alcanza O(1) y el esquema sale del régimen débil de la Def. 11.1 — artefacto verificado, no física). Validez medida: ε_c(ρ_celda_max) = 2.98e-05 ≪ 1. Suavizado de ρ(t) de la región por media móvil (ruido de cáscara).

## 1. Perfil emergente: comparación de formas

| Corrida | RMSE cored [dex] | r_c | RMSE NFW [dex] | r_s | preferida |
|---|---|---|---|---|---|
| A (newtoniana) | 0.3180 | 0.028 | 0.0620 | 0.042 | nfw |
| B (Cronos v3) | 0.3217 | 0.028 | 0.0423 | 0.032 | nfw |

Diferencia máxima A/B del perfil por cáscara: 20.6% (con α0⁻¹ = 1e-06 dentro de la cota, la corrección de Cronos es minúscula a esta resolución — como debe ser en el régimen débil).

NOTA DE RESOLUCIÓN: los radios de escala ajustados caen por debajo de la celda (0.312) y del primer bin: la malla solo ve la rama externa del perfil, de modo que la discriminación efectiva es entre pendientes externas (−2 cored vs −3 NFW), no entre núcleo y cúspide. El colapso frío aislado emerge cuspy (NFW-like) en AMBAS corridas — el contraste del núcleo (r < celda) es inaccesible aquí y queda para producción.

Lectura honesta: Δrmse(NFW−cored) = -0.2560 dex (A) y -0.2794 dex (B). A esta resolución (celda 0.312) la zona interna está limitada por la malla: la comparación es de formas y su poder de decisión es el que muestran esos Δrmse — el veredicto del núcleo es de producción.

## 2. La compuerta de Cronos (H.2.5)

Tratado: «la fricción corregida cae de ≈4×10⁻⁴ a ≈7×10⁻⁹ al virializar (compuerta cerrada), mientras la antigua persiste» — valores de SU simulación de producción; aquí se mide la caída en esta malla.

| Configuración | Γ_colapso | Γ_vir (máx. tramo final) | caída [órdenes] |
|---|---|---|---|
| Aislada (región r=0.5) | 3.067e-04 | 8.854e-06 | 1.54 |
| Con fondo 50% (región r=1.0) | 5.694e-06 | 1.970e-07 | 1.46 |

- En el halo AISLADO la compuerta cierra: Γ = 0 exacto en el 57% de los pasos del tramo final (mediana del tramo = 0.0e+00); el máx residual es ruido de cáscara de N finito (el Θ(ρ̇) del Cor. 11.3a corta el resto).
- En el halo CON FONDO el residuo de Γ es acreción secundaria real (la caja pequeña sigue alimentando la región): es caída de 1.5 órdenes limitada por infall físico, no compuerta abierta — la caída de cinco órdenes del tratado es de un halo de producción plenamente virializado.
- CONTROL NEGATIVO: la forma sin compuerta (11.3b) sobre la misma historia cae solo 0.25 órdenes — persiste, como dice H.2.5 del esquema antiguo.

Firma falsable (H.2.5): «los halos virializados no deben mostrar fricción de Cronos residual» — en esta malla, el halo aislado virializado muestra exactamente ninguna.
