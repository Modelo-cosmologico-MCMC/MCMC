"""Lector de las configuraciones YAML de Cronos (configs/cronos_*.yaml).

Las tres cajas recomendadas por el tratado (v35, B.4): Local
(25-50 h⁻¹Mpc, 512³), Meso (100-200 h⁻¹Mpc, hasta 1024³) y LSS
(500-1000 h⁻¹Mpc, ~1024³). Los YAML documentan esas corridas de
producción (frente abierto nº 5); la malla PM de este repositorio
(cronos.simulation) las usa como referencia de parámetros.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Carga configs/cronos_<name>.yaml (name: 'local', 'meso', 'lss')."""
    path = CONFIG_DIR / f"cronos_{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("cronos_*.yaml"))
        raise FileNotFoundError(
            f"No existe {path.name}; disponibles: {available}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
