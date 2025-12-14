# Datos de Validacion MCMC

Este directorio contiene los datasets utilizados para validar el modelo MCMC.

## Archivos

### bao_desi_y3.csv

Datos de BAO (Baryon Acoustic Oscillations) de DESI Year 3.

**Columnas:**
- `tracer`: Trazador (BGS, LRG, ELG, QSO, Lya)
- `z_eff`: Redshift efectivo
- `observable`: D_M/r_d o D_H/r_d
- `value`: Valor medido
- `error`: Error (1 sigma)

**Referencia:**
DESI Collaboration (2024). "DESI Year 3 BAO Measurements"

### Uso

```python
import pandas as pd

# Cargar datos
data = pd.read_csv('data/bao_desi_y3.csv', comment='#')

# Filtrar por trazador
lrg_data = data[data['tracer'] == 'LRG']
```

## Datasets Pendientes

- [ ] Pantheon+ SNe Ia
- [ ] SPARC rotation curves
- [ ] Planck C_l spectra
- [ ] MW satellite catalog
- [ ] JWST high-z galaxies

## Fuentes

1. **DESI:** https://data.desi.lbl.gov/
2. **Pantheon+:** https://pantheonplussh0es.github.io/
3. **SPARC:** http://astroweb.cwru.edu/SPARC/
4. **Planck:** https://pla.esac.esa.int/

## Licencia

Los datos son propiedad de sus respectivas colaboraciones.
Su uso en este repositorio es exclusivamente para validacion cientifica.
