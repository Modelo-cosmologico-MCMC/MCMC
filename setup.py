#!/usr/bin/env python3
"""
Setup para el Modelo Cosmológico de Múltiples Colapsos (MCMC)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcmc-cosmology",
    version="1.0.0",
    author="Adrián Martínez Estellés",
    author_email="adrianmartinezestelles92@gmail.com",
    description="Modelo Cosmológico de Múltiples Colapsos (MCMC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Modelo-cosmologico-MCMC/MCMC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.4",
        ],
        "visualization": [
            "matplotlib>=3.4",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "mcmc-demo=examples.ejemplo_5_bloques:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
