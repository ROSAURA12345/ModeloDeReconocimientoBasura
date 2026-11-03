"""Catálogo de clases y color de tacho según la NTP 900.058:2019."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import yaml

from .rutas import ruta


def normalizar_etiqueta(texto: str) -> str:
    """'  Plastic-Bottle ' -> 'plastic_bottle';  'Orgánico' -> 'organico'."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    t = t.strip().lower()
    for ch in (" ", "-", ".", "/"):
        t = t.replace(ch, "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_")


_NO_ENCONTRADA = object()


@dataclass
class Catalogo:
    clases: list[str]
    info: dict
    colores: dict
    equivalencias: dict
    norma: str

    @classmethod
    def desde_yaml(cls, archivo) -> "Catalogo":
        with open(ruta(archivo), encoding="utf-8") as f:
            d = yaml.safe_load(f)
        clases = list(d["clases"].keys())
        eq = {normalizar_etiqueta(k): v for k, v in (d.get("equivalencias") or {}).items()}
        # Cada clase siempre es equivalente a sí misma
        for c in clases:
            eq.setdefault(normalizar_etiqueta(c), c)
        for origen, destino in eq.items():
            if destino is not None and destino not in clases:
                raise ValueError(f"La equivalencia '{origen}' -> '{destino}' apunta a una clase "
                                 f"que no existe en el catálogo: {clases}")
        return cls(clases, d["clases"], d.get("colores", {}), eq, d.get("norma", ""))

    def mapear(self, etiqueta: str):
        """Devuelve la clase canónica, None si se descarta a propósito, o _NO_ENCONTRADA."""
        return self.equivalencias.get(normalizar_etiqueta(etiqueta), _NO_ENCONTRADA)

    @staticmethod
    def es_desconocida(valor) -> bool:
        return valor is _NO_ENCONTRADA

    def nombre(self, clase: str) -> str:
        return self.info[clase].get("nombre", clase)

    def tacho(self, clase: str, esquema: str = "municipal") -> str:
        return self.info[clase]["tacho"][esquema]

    def color_hex(self, color: str) -> str:
        return self.colores.get(color, {}).get("hex", "#888888")

    def tachos(self, esquema: str = "municipal") -> list[str]:
        return [self.tacho(c, esquema) for c in self.clases]
