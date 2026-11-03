"""Bitácora de corridas en formato JSON Lines (cap. 13.3 y 13.4.3).

Cada línea de resultados/bitacora.jsonl es una corrida con sus cuatro tipos de
información: parámetros, métricas, etiquetas (huella de datos, commit, entorno)
y la ruta de sus artefactos. Se lee con pandas:  pd.read_json(..., lines=True).
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone

from .rutas import RAIZ, ruta


def _git(*a) -> str:
    try:
        r = subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def entorno(equipo=None) -> dict:
    import torch
    return {"python": platform.python_version(), "sistema": f"{platform.system()} {platform.release()}",
            "torch": torch.__version__, "cuda_compilado": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna",
            "equipo": str(equipo) if equipo is not None else "",
            "commit": _git("rev-parse", "--short", "HEAD") or "sin-git",
            "arbol_sucio": bool(_git("status", "--porcelain"))}


def anotar(cfg: dict, tipo: str, nombre: str, parametros: dict, metricas: dict, huella: str,
           artefactos: str = "", equipo=None) -> None:
    destino = ruta(cfg["salidas"]["carpeta"]) / "bitacora.jsonl"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fila = {"momento": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tipo": tipo,
            "nombre": nombre, "huella_datos": huella, "parametros": parametros, "metricas": metricas,
            "entorno": entorno(equipo), "artefactos": artefactos}
    with open(destino, "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
