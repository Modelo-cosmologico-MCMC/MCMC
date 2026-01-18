# CAMB-MCMC Integration

## Overview

This directory contains configuration files and instructions for integrating
the MCMC cosmological model into CAMB (Code for Anisotropies in the Microwave Background).

## Key Modifications

### 1. Lambda_rel(z) in Fortran

Modify `equations.f90`:

```fortran
! In camb/equations.f90

! Add MCMC parameters module
module MCMC_params
    implicit none
    real(dl) :: epsilon_mcmc = 0.012_dl
    real(dl) :: z_trans_mcmc = 1.0_dl  ! Ontologia MCMC
    real(dl) :: delta_z_mcmc = 1.5_dl
end module MCMC_params

! Lambda_rel function
function Lambda_rel(z) result(L_rel)
    use MCMC_params
    real(dl), intent(in) :: z
    real(dl) :: L_rel
    L_rel = 1.0_dl + epsilon_mcmc * tanh((z_trans_mcmc - z) / delta_z_mcmc)
end function Lambda_rel

! Modify dark energy density
! Original: grhov = grhov_today
! Modified: grhov = grhov_today * Lambda_rel(z)
```

### 2. Required Files to Modify

1. `camb/equations.f90` - Add Lambda_rel function
2. `camb/inidriver.F90` - Add parameter reading
3. `camb/modules.f90` - Add parameter declarations

### 3. Python Interface

For CAMB Python:

```python
import camb

def get_results_mcmc(epsilon=0.012, z_trans=1.0, delta_z=1.5):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)

    # Note: Requires modified CAMB for Lambda_rel
    # pars.set_mcmc_params(epsilon=epsilon, z_trans=z_trans, delta_z=delta_z)

    results = camb.get_results(pars)
    return results
```

## Configuration Files

- `params_mcmc.ini` - CAMB configuration with MCMC parameters

## Usage

```bash
# Clone CAMB
git clone --recursive https://github.com/cmbant/CAMB.git
cd CAMB

# Apply MCMC modifications
# ... see modification guide

# Compile Fortran version
python setup.py build_cluster

# Or use Python
pip install camb
# Then modify camb/equations.f90 and reinstall
```

## Validation

After modification, compare with CLASS-MCMC:

1. H(z) should match within 0.01%
2. CMB Cl should match within 0.1%
3. Matter power spectrum P(k) should match

## References

- CAMB code: https://github.com/cmbant/CAMB
- CAMB documentation: https://camb.readthedocs.io/
- MCMC model: This repository
