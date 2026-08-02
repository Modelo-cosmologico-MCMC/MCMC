# Notebooks

- **`tour_deductive_chain_en.ipynb`** — *A Tour of the Executable
  Deductive Chain* (in English, with executed outputs): axioms, Basal
  potential, Path flow, Florencia signature, Decade, Gea/Residues,
  Victoria and the δ₀ circle, the C¹ splice and WKB ladder, cosmology
  with the production-fit results (v1/v2, published as-is), Cronos v3
  gate (H.2.5 half-step), and the Appendix-H balance. Runs offline in
  about a minute; production fits are quoted from `results/`, not re-run.

Regenerar/ejecutar localmente:

```bash
pip install nbclient ipykernel
jupyter execute notebooks/tour_deductive_chain_en.ipynb
```

Otros notebooks recomendados (a generar localmente; cualquier script de
`scripts/` puede convertirse con `jupytext`):

- `01_ontology_overview.ipynb` — Sellos, S-map, V(Φ; S), curvaturas.
- `02_mass_program.ipynb`      — WKB, M1, M2, espectro 12/12 fermiones.
- `03_cosmological_fit.ipynb`  — H(z), Λ_rel(z), ajuste bayesiano emcee.
- `04_wkb_analysis.ipynb`      — |T_n^(F_i)|, túnel secuencial, calibración.
- `05_quantum_simulation.ipynb` — Qudit d=5, fidelidades, Hamiltoniano.
