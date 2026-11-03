"""Lectura, validación y huellas de imágenes (sin torch).

- abrir_rgb: aplica la orientación EXIF. Las fotos de celular vienen giradas
  "de mentira" (la orientación está en los metadatos); sin esto el modelo
  entrena con fotos de costado.
- dhash: huella perceptual de 64 bits. Dos fotos casi iguales (otra
  compresión, otro tamaño, una aumentación leve de Roboflow) quedan a pocos
  bits de distancia. Es la herramienta del cap. 8.5.3 contra la fuga por
  duplicados.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

try:  # Fotos de iPhone (.heic). Opcional: pip install pillow-heif
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    HEIC_OK = True
except Exception:  # noqa: BLE001
    HEIC_OK = False


def es_imagen(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTENSIONES


def abrir_rgb(ruta: str | Path, orientar: bool = True) -> Image.Image:
    with Image.open(ruta) as im:
        im.load()
        if orientar:
            im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def dhash(im: Image.Image, lado: int = 8) -> int:
    """Huella por diferencias horizontales sobre una miniatura en gris de 9x8."""
    g = im.convert("L").resize((lado + 1, lado), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def sha256(ruta: str | Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def motivos_rechazo(im: Image.Image, reglas: dict) -> list[str]:
    """Validación estructural y 'física' de una imagen (cap. 7.2.4)."""
    motivos = []
    w, h = im.size
    if min(w, h) < reglas.get("lado_min", 48):
        motivos.append(f"muy_pequena_{w}x{h}")
    if max(w, h) / max(1, min(w, h)) > reglas.get("aspecto_max", 6.0):
        motivos.append("aspecto_extremo")
    gris = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
    if gris.std() < reglas.get("desviacion_min", 6.0):
        motivos.append("imagen_plana_o_velada")
    return motivos


def distancias_hamming_bajas(huellas: np.ndarray, umbral: int, trozo: int = 512):
    """Pares (i, j) con i < j cuya distancia de Hamming es <= umbral.

    Vectorizado por bloques: con 20 000 imágenes son 2*10^8 comparaciones y
    tarda segundos, no horas.
    """
    h = np.asarray(huellas, dtype=np.uint64)
    n = len(h)
    contar = getattr(np, "bitwise_count", None)
    if contar is None:  # numpy < 2.0
        tabla = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        def contar(x):  # noqa: E306
            return tabla[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)

    pares = []
    for ini in range(0, n, trozo):
        fin = min(n, ini + trozo)
        bloque = h[ini:fin, None] ^ h[None, ini:]
        d = contar(bloque)
        ii, jj = np.nonzero(d <= umbral)
        jj = jj + ini
        sel = jj > (ii + ini)
        pares.extend(zip((ii[sel] + ini).tolist(), jj[sel].tolist()))
    return pares
