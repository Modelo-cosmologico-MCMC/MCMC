# Matriz canónica manuscrito ↔ código ↔ test ↔ estatuto

Trazabilidad explícita del programa (propuesta del análisis v36, punto
8; el Apéndice H exige que cada pieza resista una prueba objetiva y que
cada frente declare qué le falta). Los estatutos siguen la v35.1:
«interna demostrada» = comprobación interna de la implementación
superada (E8), nunca demostración física. Las celdas «—» declaran un
hueco real, no lo esconden. `tests/test_traceability.py` verifica que
cada ruta citada existe.

| Elemento | Manuscrito | Código | Test | Estatuto |
|---|---|---|---|---|
| Ocho axiomas | §1.2 | `mcmc_ontology/axioms.py` | `tests/test_appendix_h.py` | ancla conceptual |
| Constantes por estatuto | Ap. F | `mcmc_ontology/constants.py` | `tests/test_potential.py` | F.2-F.4 + C6 |
| Plano Dual | Cap. 2 | `core/dual_plane.py` | `tests/test_core_foundation.py` | interna demostrada |
| Potencial Basal, T₀ = c̄δ₀³ | Cap. 3 | `core/basal.py` | `tests/test_core_foundation.py` | interna demostrada (exponente medido) |
| Monotonía y Exclusión | Cap. 4 | `core/path_flow.py` | `tests/test_core_foundation.py` | interna demostrada |
| Cadena de Álgebras y Florencia | Caps. 5-6 | `core/florencia.py` | `tests/test_core_florencia_decade.py` | interna demostrada (firma −+++) |
| RP escalar | Cap. 7 (Teo. 7.1) | `core/reflection_positivity.py` | `tests/test_core_reflection.py` | juguete demostrado |
| RP no estacionaria | §7.5 (Prop. 7.4, E3) | `core/rp_nonstationary.py` | `tests/test_rp_nonstationary.py` | juguete (especularidad suficiente, necesidad no demostrada); Wilson abierto (frente 1) |
| Década y Discriminante | Cap. 8 | `core/decade.py` | `tests/test_core_florencia_decade.py` | interna demostrada; λ=10 calibrado |
| Flujo KLS integrado | Cap. 8 (frente E) | `core/kls_flow.py` | `tests/test_kls_flow.py` | resultado numérico (−1/2; −1/3 condicional) |
| Matriz de estabilidad | §14.2 (frente 2) | `core/victoria_exponent.py` | `tests/test_victoria_exponent.py` | instrumentado; β de Fokker-Planck pendientes |
| Gea, Sello de Newton, Atlas | Cap. 9 | `core/gea.py` | `tests/test_core_gea_victoria.py` | sello exacto; Atlas condicional |
| Contraste de los Residuos | Conj. 9.6 (frente 6) | `cosmology/residues_test.py` | `tests/test_residues.py` | consistencia superada (E13); refutable vivo |
| Ciclo de Victoria | Cap. 10 | `core/victoria.py` | `tests/test_core_gea_victoria.py` | condicional (signo de ν, frente 4) |
| Círculo de δ₀ | Cap. 10 + H.8 | `core/delta0_circle.py` | `tests/test_delta0_circle.py` | condicional (W_max, frente 4) |
| Fertilidad | H.2.3 (frente 4) | `core/fertility_map.py` | `tests/test_fertility.py` | cartografía; γ_max declarado |
| Naturalidad y priors | Nota I §3.2 | `core/landscape_priors.py` | `tests/test_landscape_priors.py` | robusto (cota) / fiducial (mediana) |
| Suite espejo del Ap. H | Ap. H | `validation/appendix_h.py` | `tests/test_appendix_h.py` | comprobaciones internas (E8) |
| Cronos v3 | Cap. 11 | `cronos/cronos_v3.py` | `tests/test_cronos_v3.py` | implementado; producción abierta (frente 5) |
| Ciclo PM del Ap. B | Ap. B | `cronos/simulation.py` | `tests/test_simulation.py` | prototipo (no producción) |
| Perfil y compuerta | H.2.5 (frente 5) | `cronos/profile_fit.py` | `tests/test_profile_fit.py` | compuerta reproducida; núcleo en producción |
| Fondo cosmológico | Ap. A | `cosmology/background.py` | `tests/test_recovery_limit.py` | límite de recuperación exacto (Prop. A.1) |
| Ajustes de producción v1/v2 | A.6 → E5 | `cosmology/bayesian_fit.py` | `tests/test_bayesian_fit.py` | no favorecido (ΔBIC +14.5), publicado |
| CMB comprimido y fσ8 | Ap. A (opción B) | `cosmology/extended_likelihoods.py` | `tests/test_extended_likelihoods.py` | salvedades declaradas (HS96, diagonal) |
| Canales oscuros | Ap. A.1 | `cosmology/dark_channels.py` | `tests/test_dark_channels.py` | esquema de dos canales |
| Funcional del Camino (masas) | Cap. 12 | `mass_program/B4_masses.py` | `tests/test_masses.py` | calibrado; carácter predictivo por establecer (12.5) |
| Higgs y auditoría | Prop. 12.1 / Obs. 12.2 | `mass_program/B5_higgs.py` | `tests/test_masses.py` | identidad SM calibrada, no predicción |
| Ngen = 3 | Prop. 12.4 → E4 | `mass_program/B2_clifford.py` | — | identificación estructural, sin test dedicado (hueco declarado; sub-frente del 7) |
| Empalme C¹ y WKB ab initio | H.2.4 (frente 7) | `mass_program/B7_empalme.py` | `tests/test_empalme_wkb.py` | no circular; mide δ_H (≠ δ₀*, C6); valor de β₃ condicional |
| WKB calibrado (v32) | Tabla 3 v32 | `mass_program/B3_wkb.py` | `tests/test_wkb.py` | entradas calibradas, no derivación |
| Mass gap en retículo | Ap. D | `lattice/mass_gap.py` | `tests/test_lattice.py` | piso espectral efectivo (E10), no Yang-Mills |
| Qudit y decoherencia | Ap. C | `quantum/qutip_simulation.py` | `tests/test_quantum_v35.py` | firma de consistencia del canal (E10) |
| Qudit d=5 | Ap. C.1 | `quantum/qudit.py` | `tests/test_qudit.py` | base implementada |
