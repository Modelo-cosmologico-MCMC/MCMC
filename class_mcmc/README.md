# CLASS-MCMC Integration

## Overview

This directory contains configuration files and instructions for integrating
the MCMC cosmological model into CLASS (Cosmic Linear Anisotropy Solving System).

## Key Modifications

### 1. Lambda_rel(z) Implementation

The main modification is in `background.c`:

```c
// In source/background.c

// Add MCMC parameters
double epsilon_mcmc = 0.012;
double z_trans_mcmc = 1.0;  // Ontologia MCMC
double delta_z_mcmc = 1.5;

// Lambda_rel function
double Lambda_rel(double z, double epsilon, double z_trans, double dz) {
    return 1.0 + epsilon * tanh((z_trans - z) / dz);
}

// Modify rho_lambda calculation
// Original: rho_lambda = pba->Omega0_lambda * pow(pba->H0, 2);
// Modified:
double L_rel = Lambda_rel(z, epsilon_mcmc, z_trans_mcmc, delta_z_mcmc);
rho_lambda = pba->Omega0_lambda * pow(pba->H0, 2) * L_rel;
```

### 2. Required Files to Modify

1. `source/background.c` - Add Lambda_rel function
2. `source/background.h` - Add parameter declarations
3. `source/input.c` - Add parameter reading
4. `include/background.h` - Add to structure

### 3. New Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| epsilon_mcmc | ECV amplitude | 0.012 |
| z_trans_mcmc | Transition redshift | 1.0 |
| delta_z_mcmc | Transition width | 1.5 |

## Configuration Files

- `param_mcmc.ini` - Standard precision MCMC configuration
- `param_mcmc_highprecision.ini` - High precision for CMB calculations

## Usage

```bash
# Clone CLASS
git clone https://github.com/lesgourg/class_public.git
cd class_public

# Apply MCMC modifications (manual or patch)
# ... see modification guide

# Compile
make clean
make

# Run with MCMC parameters
./class class_mcmc/param_mcmc.ini
```

## Validation

After modification, verify:

1. H(z) matches Python implementation within 0.1%
2. CMB Cl spectra are physical
3. BAO predictions match DESI Y3 data

## References

- CLASS code: https://github.com/lesgourg/class_public
- CLASS documentation: https://lesgourg.github.io/class_public/
- MCMC model: This repository
