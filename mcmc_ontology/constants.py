"""Parámetros globales del MCMC, organizados por estatuto epistémico.

FUENTE: Tratado de Fundamentos (v35, junio 2026), DOI 10.5281/zenodo.20765373.
Las referencias a ecuaciones/tablas del corpus anterior se etiquetan
«Tratado Unificado v32». Realiza el axioma 5 (sellos) del v35 §1.2.

La organización sigue el balance del Apéndice F del tratado, que distingue
cuatro estatutos: INPUTS genuinos, SELLADOS por consistencia interna,
DERIVADOS y CALIBRADOS contra observación. F.4: «No hay un mar de
parámetros libres: hay una imperfección, unas pocas formas, y un edificio
que se sostiene sobre ellas. Cada constante sellada o derivada es una
predicción que la consistencia o la observación pueden refutar.»

NO modificar valores sin justificación física documentada.
"""

import math

# =====================================================================
# INPUTS genuinos (lo único libre — v35 F.4)
# =====================================================================
# δ₀ — la imperfección primordial, input de ciclo. La v35 (F.2) la lista
# como «input de ciclo; atractor δ₀* (Teo. 10.6)» SIN asignarle valor
# numérico. REGLA CANÓNICA (ronda 5): no identificar δ₀ con ε_Λ en
# ningún material — la identificación δ₀ ≡ ε = 0.012 era herencia
# operativa del v32. La v35 separa cuatro símbolos: ε (residuo del
# colapso, rango [1e-8, 1e-3]), ε_Λ (amplitud de la transición de Λ_rel,
# 0.012 ± 0.003, A.3), ε_K (residuo gravitatorio ≈ 0.012, ec. 9.5) y δ₀
# (input de ciclo). Este módulo NO define constante numérica para δ₀:
# el empalme C¹ (H.2.4, mass_program/B7) lo MIDE — δ₀* ≈ 0.0581 con
# formas fiduciales — y core/delta0_circle.py analiza su condición de
# cierre. Las constantes de forma O(1) (m̄, b̄, ē, C0) viven en
# core/basal.py (v35 F.2).

EPSILON_0 = 0.012      # ε_Λ — amplitud de la transición de Λ_rel:
                       # 0.012 ± 0.003 (v35 A.3/F.3), consumida por
                       # cosmology/ como `eps`. CALIBRADA contra
                       # observación (ajuste v2 propio: ε = 0.015
                       # −0.039/+0.043). NO identificar con δ₀ (regla
                       # canónica; el nombre EPSILON_0 se conserva por
                       # los consumidores).

# =====================================================================
# SELLADOS por consistencia interna (no ajustables — v35 F.4)
# =====================================================================
DELTA_S = 1e-3         # Cuanto entrópico (Calibre de Cronos). Es ELECCIÓN DE
                       # UNIDADES, no constante física: «el cuanto es real,
                       # su número es convencional» (v35 F.4).

LAMBDA_DECADE = 10.0   # Razón de la Década (calibrado; su derivación es el
                       # frente abierto nº 2 — v35 §13.6)


def decade_thresholds() -> list[float]:
    """Umbrales de la Ley de la Década (v35, Prop. 8.1).

        S_k^col = λ^(k-2) − ΔS   (k = 0,1,2 → colapsos 1D, 2D, 3D)
        S_Florencia = λ^0 + ΔS

    → [0.009, 0.099, 0.999, 1.001]. El −ΔS es el cuanto de anticipación;
    el +ΔS, el cuanto de confirmación.
    """
    return [LAMBDA_DECADE ** (k - 2) - DELTA_S for k in range(3)] + [1.0 + DELTA_S]


# Sellos ontológicos (índice entrópico S) — cronología v35 (Tabla F.1).
# Claves C0..C4/V1D..V3D conservadas del Unificado (v32) porque los módulos
# las consumen; los comentarios siguen la Tabla F.1:
S_SEALS = {
    "C0":  0.001,   # Emergencia Mp/Ep, V0D, proto-gravedad (S_0,001)
    "C1":  0.009,   # Primer colapso de la Década (λ^(-2) − ΔS)
    "V1D": 0.010,   # Nace la línea (S_0,010): partícula/antipartícula
    "C2":  0.099,   # SELLO DE c — Cota de Delivery (v35 Tabla F.1; no es
                    # una emergencia de familia fermiónica)
    "V2D": 0.100,   # Nace el plano (S_0,100): giro y rotación
    "C3":  0.999,   # SELLO DE c² — preparación volumétrica (v35 Tabla F.1;
                    # la escala EW NO está asociada a este sello en la v35)
    "V3D": 1.000,   # Colapso V3D: nace el volumen, gravedad como curvatura;
                    # aquí emerge la escala QCD (v35 D.1, S3 = 1.000)
    "C4":  1.001,   # Umbral de Florencia: V3+1D, mass gap mínimo, Ley de
                    # Cronos, ΦAd → ΦH; v3 = 246 GeV sellada por V3+1D
}

GAMMA_LQG = 0.274      # Parámetro de Immirzi γ* — sellado por
                       # Bekenstein-Hawking + conteo Kaul-Majumdar (v35 F.4,
                       # Prop. D.1)

V3_GEV = 246.0         # Escala electrodébil v3 [GeV] — sellada por V3+1D
                       # (v35 F.4; asociada a S_1,001, no a S_0,999)
V_EW = V3_GEV          # Alias histórico (consumido por mass_program/B4)

# =====================================================================
# DERIVADOS (v35 F.4)
# =====================================================================
S0_VICTORIA = math.pi / math.log(10.0)   # s0 = π/ln(10) ≈ 1.3644 — exponente
                                         # de Victoria, derivado PARA λ = 10
M_HIGGS_MCMC = 125.3    # GeV — valor publicado por el tratado (v35, Prop. 12.1)
                        # vía β3; el cómputo exacto sqrt(2·0.13)·246 da 125.436.
                        # Ver la auditoría de circularidad (Obs. 12.2) en
                        # mass_program/B5_higgs.py: β3 es calibrado mientras no
                        # exista su derivación independiente (frente nº 7).

# =====================================================================
# CALIBRADOS contra observación o contra el corpus (v35 F.3/F.4)
# =====================================================================
Z_TRANS = 8.9          # Redshift de transición de Λ_rel: 8.9 ± 0.4 (v35 A.3)
DZ_TRANS = 1.5         # Ancho Δz de la transición tanh (v35 A.3)

LAMBDA_ONT = 0.01      # Grosor ontológico del sello [S-units] (v32)

# Perfil m_P(S) (v32):
MP_0  = 0.99   # m_P en C1 (S = 0.009)
MP_EQ = 0.50   # m_P en C4 (S = 1.001)

# Transmisiones WKB |T_n^(i)| (individuales, NO acumuladas).
# ENTRADAS calibradas (Tabla 3, Tratado Unificado v32), no derivaciones:
# el cálculo WKB de primeros principios es el frente abierto nº 7 (v35 §13.6).
# Filas = familia (sello de emergencia), columnas = sellos C1..C4
T_UNIVERSAL = {
    "F1": [1.000,  0.998,  0.950,  0.900],   # tau, t, b, nu_tau   — emerge en C1
    "F2": [4.3e-4, 1.000,  0.970,  0.920],   # mu,  c, s, nu_mu    — emerge en C2
    "F3": [1.2e-6, 2.1e-3, 1.000,  0.950],   # e,   u, d, nu_e     — emerge en C3
}

# Parámetros WKB calibrados (v32; frente abierto nº 7):
KAPPA_GAP_12 = 0.8613   # Barrera C1→C2
KAPPA_GAP_23 = 0.0685   # Barrera C2→C3
E_F2 = 0.4881           # Eigenvalor Dirac familia F2
E_F3 = 0.9431           # Eigenvalor Dirac familia F3
K_NORM = 679.14         # Normalización Δm_eff [GeV] (calibrada en C4)

M_HIGGS_PDG = 125.25    # GeV (dato externo, PDG)

# Running QCD (Tabla 41, Tratado Unificado v32):
ALPHA3_INV_TABLE = [
    (0.009, 27.0),
    (0.099, 25.3),
    (0.150, 24.5),
    (0.500, 17.4),
    (0.999,  9.3),
    (1.001,  9.0),
    (90.0,   8.5),
]

# Parámetros Cronos N-body del ESQUEMA ANTERIOR (v32). La v35 (cap. 11,
# Obs. 11.4) lo declara superado: dos errores de signo compensados. El
# esquema vigente (Cronos v3, cap. 11) es el frente abierto nº 5, con la
# cota dura α0⁻¹ ≲ 1e-6 (ec. 11.5). ALPHA_CRONOS NO es α0⁻¹.
ALPHA_CRONOS = 0.030
RHO_C0       = 200.0
ZETA_CRONOS  = 0.015

# Yukawa GUT y_ij^(0) — Ec. 844-845, Tratado Unificado v32.
# Filas: tipo fermiónico; columnas: F1, F2, F3 (sellos de emergencia)
YUKAWA_GUT = {
    "lepton": {"F1": 7.22e-3, "F2": 4.30e-4, "F3": 2.08e-6},
    "up":     {"F1": 7.02e-1, "F2": 5.16e-3, "F3": 8.78e-6},
    "down":   {"F1": 1.70e-2, "F2": 3.80e-4, "F3": 1.90e-5},
}

# =====================================================================
# DATOS EXTERNOS (PDG 2024)
# =====================================================================
CKM_SQ = {
    ("u", "d"): 0.9482, ("u", "s"): 0.0508, ("u", "b"): 1.5e-5,
    ("c", "d"): 0.0507, ("c", "s"): 0.9477, ("c", "b"): 1.75e-3,
    ("t", "d"): 7.5e-5, ("t", "s"): 1.69e-3, ("t", "b"): 0.9983,
}

PDG_MASSES_GEV = {
    "tau":   1.77686,  "mu":    0.10566,  "e":     0.000511,
    "t":   172.8,      "b":     4.18,     "c":     1.27,
    "s":     0.0934,   "d":     0.00467,  "u":     0.00216,
    "H":   125.25,
}

# =====================================================================
# ESTRUCTURALES (v35)
# =====================================================================
N_GEN = 3   # Ngen = 3: dimensión del módulo espinorial de la Cadena de
            # Álgebras en V3+1D (v35, Prop. 12.4). Álgebra, no ajuste.

# Mapa familia → fermiones (presentación v32; en v35 la supervivencia por
# sello la codifican los pesos c_in del Funcional del Camino, Def. 12.3):
FAMILY_FERMIONS = {
    "F1": {"lepton": "tau", "up": "t", "down": "b", "nu": "nu_tau"},
    "F2": {"lepton": "mu",  "up": "c", "down": "s", "nu": "nu_mu"},
    "F3": {"lepton": "e",   "up": "u", "down": "d", "nu": "nu_e"},
}

# =====================================================================
# VALORES DE REFERENCIA DEL CORPUS (NO salidas de este código)
# =====================================================================
# Proceden de los ajustes documentados en el corpus (v32). Se conservan
# como valores fiduciales (defaults de H_of_z, wrappers, B6) y como
# referencia histórica. NOTA (jul-2026): el primer ajuste de producción
# de este repositorio sobre datos reales (CC+BAO+SNe; ver README y
# results/2026-07-31_production_fit/) obtuvo H0 = 66.1 ± 1.7 y ΔBIC =
# +14.5 A FAVOR de ΛCDM con esa metodología mínima; la reconciliación
# con estos valores del corpus (que usan likelihoods más ricos: CMB,
# fσ8, lentes) es trabajo abierto.
CORPUS_REFERENCE = {
    "H0": 69.8,          # km/s/Mpc (corpus; tratado v35 A.6: "~69-70")
    "H0_err": 1.1,
    "sigma8": 0.805,     # (corpus)
    "S8": 0.795,         # (corpus)
    "delta_BIC": -6.1,   # (corpus; pendiente de reproducción con datos reales)
    "delta_chi2": -12.3, # (corpus; ídem)
}
# Alias planos consumidos por cosmology/, mass_program/B6 y tests.
H0_MCMC     = CORPUS_REFERENCE["H0"]
H0_ERR      = CORPUS_REFERENCE["H0_err"]
SIGMA8_MCMC = CORPUS_REFERENCE["sigma8"]
S8_MCMC     = CORPUS_REFERENCE["S8"]
DELTA_BIC   = CORPUS_REFERENCE["delta_BIC"]
DELTA_CHI2  = CORPUS_REFERENCE["delta_chi2"]

# =====================================================================
# LEGACY_V32 — parametrización del Tratado Unificado retirada en la v35
# =====================================================================
# Verificado contra el texto completo de la v35: estos valores NO existen
# en el Tratado de Fundamentos. Se conservan porque módulos del esquema
# v32 aún los consumen; no usar en código nuevo.
LEGACY_V32 = {
    # Índice entrópico "actual" y máximo de la formulación anterior
    # (retirados también del sitio web). Consumidores: cosmology/background
    # (rho_id/rho_lat), mcmc_ontology/S_map (s_to_a), cronos/rho_id_table.
    "S_actual": 95.0,   # LEGACY_V32
    "S_max":   150.0,   # LEGACY_V32
}

# VEVs por sello [GeV] — asignación v32. La v35 NO asigna Planck/GUT a
# umbrales: v3 = 246 GeV está sellada por V3+1D (S=1.001, no 0.999) y la
# escala QCD emerge en S3 = 1.000 (v35 D.1). Lo consume la maquinaria del
# potencial escalonado v32 (potential.py, seals.py, B1_vevs, P4).
V_GEV = {
    "C1": 1.22e19,  # Escala Planck (v32; sin umbral asignado en v35)
    "C2": 1.00e16,  # Escala GUT (v32; sin umbral asignado en v35)
    "C3": 246.0,    # = V3_GEV; en v35 sellada por V3+1D, no por S_0,999
    "C4": 0.20,     # Escala QCD (v32; en v35 emerge en S3 = 1.000, D.1)
}

# Parámetros cuárticos beta_n del potencial escalonado (v32):
BETA = {
    "C1": 1e-43,
    "C2": 1e-35,
    "C3": 0.13,     # λ_H del empalme C¹ (v35 F.3); la derivación independiente
                    # de m_H está pendiente — frente abierto nº 7, Obs. 12.2
    "C4": 1e7,
}
