"""Parámetros globales del MCMC.

FUENTE: Tratado Unificado (versión 8 Abril 2026).
Todos los valores numéricos provienen de la referencia técnica completa
del modelo. NO modificar sin justificación física documentada.
"""

# --- Sellos ontológicos (índice entrópico S) ---
S_SEALS = {
    "C0":  0.001,   # Colapso puntual
    "C1":  0.009,   # Sello V0D, emerge F1 (tau,t,b,nu_tau)
    "V1D": 0.010,   # Nace la línea
    "C2":  0.099,   # Sello c, emerge F2 (mu,c,s,nu_mu); GUT
    "V2D": 0.100,   # Nace el plano
    "C3":  0.999,   # Sello c^2, emerge F3 (e,u,d,nu_e); EW
    "V3D": 1.000,   # Nace el volumen
    "C4":  1.001,   # Big Bang, nace el tiempo, Higgs
    "S_actual": 95.0,
    "S_max":   150.0,
}

# --- VEVs por sello [GeV] ---
V_GEV = {
    "C1": 1.22e19,  # Escala Planck
    "C2": 1.00e16,  # Escala GUT
    "C3": 246.0,    # Escala EW
    "C4": 0.20,     # Escala QCD
}

# --- Parámetros cuárticos beta_n ---
BETA = {
    "C1": 1e-43,
    "C2": 1e-35,
    "C3": 0.130,    # = 1/2 (m_H / v_EW)^2  — NO libre
    "C4": 1e7,
}

# --- Parámetros universales ---
LAMBDA_ONT = 0.01      # Grosor ontológico del sello [S-units]
DELTA_S    = 1e-3      # Granularidad LQG: ΔS ↔ Δj=1
V_EW       = 246.0     # VEV electrodébil [GeV]
GAMMA_LQG  = 0.274     # Parámetro de Immirzi
EPSILON_0  = 0.012     # δ₀ ≡ ε (imperfección primordial, ajuste global)
Z_TRANS    = 8.9       # Redshift de transición cosmológica

# --- Tramo pre-geométrico n=0 (Ecs. 20-26) ---
# δ₀ ≡ v₀ en unidades adimensionales (v₀_norm).
# En GeV: V0_GEV = δ₀ · v₁ (anclaje a la escala Planck).
S_PRE      = 0.001     # S_{0.001}: nacimiento V₀D, sello C₀
V0_NORM    = EPSILON_0 # v₀ adimensional ≡ δ₀
V0_GEV     = EPSILON_0 * 1.22e19  # v₀ en GeV
LAMBDA_PRE = 1e-4      # λ_pre ∈ [1e-5, 5e-4]; valor central
# β₀, γ₀ se calculan dinámicamente en potential.py (Ecs. 24-25):
#   β₀ = 2 α_S · S_{0.001} / v₀²
#   γ₀ = -α_S · S_{0.001} · v₀²

# --- Tensión primordial T₀ y umbrales T_crit^(n) (Ecs. 440-445) ---
# T₀ = M_p·c²·δ₀² = E_p·δ₀² (Ec. 440)
# Valor numérico (Ec. 441 corregida):
#   T₀ = E_p · δ₀² = 1.22×10¹⁹ × (0.012)² ≈ 1.76×10¹⁵ GeV
# NOTA: el spec original "MCMC_Referencia_Claude_Code.md" daba 1.22×10¹⁵
# (factor 1.44 erróneo, omitía δ₀² aplicado a la energía completa). El
# valor correcto del Tratado (Ec. 441) es 1.76×10¹⁵ GeV.
M_PL_GEV       = 1.22e19
T0_GEV         = M_PL_GEV * EPSILON_0 ** 2     # ≈ 1.76e15 GeV (canónico)
T0_GEV_APPROX  = 1.76e15                       # alias documentado del valor canónico

# T_crit^(n) tabulados directamente desde la Tabla 67 del Tratado (p.156).
# La fórmula T_crit^(n) = (T₀/4)·(S_n/ΔS)·(v_n/v_n,ref)² requiere v_n,ref
# como escala de referencia POR SELLO (no v_1 = M_Pl global). Los valores
# tabulados son los canónicos del modelo:
T_CRIT_TABLE_67 = {
    "C1": {"ratio": 5.6e-4, "GeV": 6.9e11},
    "C2": {"ratio": 6.2e-3, "GeV": 7.6e12},
    "C3": {"ratio": 6.2e-2, "GeV": 7.6e13},
    "C4": {"ratio": 6.2e-2, "GeV": 7.6e13},
}

# --- Perfil m_P(S) ---
MP_0  = 0.99   # m_P en C1 (S = 0.009)
MP_EQ = 0.50   # m_P en C4 (S = 1.001)

# --- Transmisiones WKB |T_n^(i)| (individuales, NO acumuladas) ---
# Filas = familia (sello de emergencia), columnas = sellos C1..C4
T_UNIVERSAL = {
    "F1": [1.000,  0.998,  0.950,  0.900],   # tau, t, b, nu_tau   — emerge en C1
    "F2": [4.3e-4, 1.000,  0.970,  0.920],   # mu,  c, s, nu_mu    — emerge en C2
    "F3": [1.2e-6, 2.1e-3, 1.000,  0.950],   # e,   u, d, nu_e     — emerge en C3
}

# --- Parámetros WKB calibrados ---
KAPPA_GAP_12 = 0.8613   # Barrera C1→C2
KAPPA_GAP_23 = 0.0685   # Barrera C2→C3
E_F2 = 0.4881           # Eigenvalor Dirac familia F2
E_F3 = 0.9431           # Eigenvalor Dirac familia F3
K_NORM = 679.14         # Normalización Δm_eff [GeV] (calibrada en C4)

# --- Δm_eff calibrados por sello (Tratado, Tabla P3) ---
# Estos valores son los CANÓNICOS del modelo. La fórmula numérica
# Δm_eff = sqrt(V''_total)/(m_P · K_norm) sólo reproduce el ancla C4
# porque K_norm se calibra ahí (donde β_4=1e7 hace despreciable I_dD).
# En C1 y C2 las VEVs en GeV están a escala Planck/GUT y la fórmula
# requiere normalización S-space — los valores calibrados son los
# que entran en el cálculo final de |T_n^(i)| y de la fórmula maestra.
DELTA_M_EFF_CAL = {
    "C1": 0.0100,   # estimación: ΔS × m_P(C1); F1 emerge en C1, sin
                    # pre-emergencia anterior — no entra en la fórmula maestra.
    "C2": 0.1001,   # escala GUT (β_2 cuártico despreciable, kinético calibrado)
    "C3": 2.5647,   # escala EW (β_3 ~ kinético comparables)
    "C4": 5.2680,   # escala QCD (β_4 cuártico domina; ancla de K_norm)
}

# --- Higgs y mass gap ---
M_HIGGS_MCMC = 125.44   # GeV (predicción)
M_HIGGS_PDG  = 125.25   # GeV

# --- Running QCD (Tabla 41 del Tratado) ---
ALPHA3_INV_TABLE = [
    (0.009, 27.0),
    (0.099, 25.3),
    (0.150, 24.5),
    (0.500, 17.4),
    (0.999,  9.3),
    (1.001,  9.0),
    (90.0,   8.5),
]

# --- Parámetros Cronos (N-body) ---
ALPHA_CRONOS = 0.030
RHO_C0       = 200.0
ZETA_CRONOS  = 0.015

# --- Resultados ajuste global ---
H0_MCMC     = 69.8      # km/s/Mpc
H0_ERR      = 1.1
SIGMA8_MCMC = 0.805
S8_MCMC     = 0.795
DELTA_BIC   = -6.1
DELTA_CHI2  = -12.3

# --- Yukawa GUT y_ij^(0) — Ec. 844-845 ---
# Filas: tipo fermiónico; columnas: F1, F2, F3 (sellos de emergencia)
YUKAWA_GUT = {
    "lepton": {"F1": 7.22e-3, "F2": 4.30e-4, "F3": 2.08e-6},
    "up":     {"F1": 7.02e-1, "F2": 5.16e-3, "F3": 8.78e-6},
    "down":   {"F1": 1.70e-2, "F2": 3.80e-4, "F3": 1.90e-5},
}

# --- Elementos CKM al cuadrado (PDG 2024) ---
CKM_SQ = {
    ("u", "d"): 0.9482, ("u", "s"): 0.0508, ("u", "b"): 1.5e-5,
    ("c", "d"): 0.0507, ("c", "s"): 0.9477, ("c", "b"): 1.75e-3,
    ("t", "d"): 7.5e-5, ("t", "s"): 1.69e-3, ("t", "b"): 0.9983,
}

# --- PDG 2024 [GeV] ---
PDG_MASSES_GEV = {
    "tau":   1.77686,  "mu":    0.10566,  "e":     0.000511,
    "t":   172.8,      "b":     4.18,     "c":     1.27,
    "s":     0.0934,   "d":     0.00467,  "u":     0.00216,
    "H":   125.25,
}

# --- Mapa familia → fermiones (sello de emergencia) ---
FAMILY_FERMIONS = {
    "F1": {"lepton": "tau", "up": "t", "down": "b", "nu": "nu_tau"},
    "F2": {"lepton": "mu",  "up": "c", "down": "s", "nu": "nu_mu"},
    "F3": {"lepton": "e",   "up": "u", "down": "d", "nu": "nu_e"},
}

# --- N_gen algebraico (Cl(3,0) → 3 familias) ---
N_GEN = 3
