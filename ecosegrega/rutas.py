"""Rutas del proyecto, lectura del YAML del estudio y huellas SHA-256.

Regla del capítulo 13: toda decisión (tasa, semilla, umbral, ruta) vive en un
YAML, nunca enterrada en el código. Este módulo solo lee; no decide nada.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

# Raíz del proyecto: la carpeta que contiene ecosegrega/, config/, datos/ ...
RAIZ = Path(__file__).resolve().parents[1]


def ruta(relativa: str | Path) -> Path:
    """Convierte una ruta del YAML (relativa a la raíz) en absoluta."""
    p = Path(relativa)
    return p if p.is_absolute() else (RAIZ / p)


def relativa(absoluta: str | Path) -> str:
    """Ruta relativa a la raíz, con '/' (igual en Windows y Linux)."""
    p = Path(absoluta).resolve()
    try:
        return p.relative_to(RAIZ).as_posix()
    except ValueError:
        return p.as_posix()


def cargar_config(archivo: str | Path = "config/estudio.yaml") -> dict:
    with open(ruta(archivo), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_archivo"] = relativa(ruta(archivo))
    return cfg


def guardar_yaml(datos: dict, destino: str | Path) -> None:
    d = copy.deepcopy(datos)
    with open(destino, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)


def huella_archivo(archivo: str | Path, bloque: int = 1 << 20) -> str:
    """SHA-256 del contenido (cap. 13.3: sin huella de datos la corrida no es repetible)."""
    h = hashlib.sha256()
    with open(archivo, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()
