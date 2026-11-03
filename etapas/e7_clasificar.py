"""Etapa 7 — Clasificar fotos nuevas y generar una galería HTML con el tacho de cada una.

    python etapas/e7_clasificar.py foto.jpg
    python etapas/e7_clasificar.py carpeta_con_fotos/ --esquema municipal

Escribe resultados/clasificacion/galeria.html (un solo archivo, se abre con doble
clic y se puede enviar) y clasificacion.csv. Si la confianza no llega al umbral
calibrado, la foto queda marcada "revisar" en lugar de mandarla a un tacho.
"""
import argparse
from pathlib import Path

import pandas as pd

import _base  # noqa: F401
from ecosegrega.clasificador import Clasificador
from ecosegrega.galeria import galeria, miniatura
from ecosegrega.fotos import abrir_rgb, es_imagen
from ecosegrega.rutas import cargar_config, ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entradas", nargs="+")
    ap.add_argument("--modelo", default="resultados/final/modelo_final.pt")
    ap.add_argument("--equipo", default="auto")
    ap.add_argument("--esquema", default=None, choices=["no_municipal", "municipal"])
    ap.add_argument("--config", default="config/estudio.yaml")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    esquema = args.esquema or cfg["decision"]["esquema"]
    clf = Clasificador(ruta(args.modelo), args.equipo, cfg["datos"]["norma"])
    archivos = []
    for e in args.entradas:
        p = Path(e)
        archivos += sorted(q for q in p.rglob("*") if es_imagen(q)) if p.is_dir() else [p]
    filas = []
    for a in archivos:
        im = abrir_rgb(a)
        r = clf(im, esquema)
        print(f"{a.name:38s} -> {r['nombre']:16s} {100 * r['confianza']:5.1f}%  tacho {r['tacho'].upper()}"
              + ("" if r["seguro"] else "   (revisar)"))
        filas.append({"archivo": a.name, "mini": miniatura(im), **r})
    salida = ruta(cfg["salidas"]["carpeta"]) / "clasificacion"
    salida.mkdir(parents=True, exist_ok=True)
    galeria(filas, salida / "galeria.html", esquema, clf.umbral)
    pd.DataFrame([{k: v for k, v in f.items() if k not in ("mini", "top3")} for f in filas]).to_csv(
        salida / "clasificacion.csv", index=False)
    print(f"\nGalería: {salida / 'galeria.html'}")


if __name__ == "__main__":
    main()
