"""Etapa 1 — Obtener datos de Roboflow Universe con su procedencia (cap. 7).

La descarga es un script versionado, no "bajé un zip y lo junté": así queda una
descripción ejecutable del origen de los datos.

Opción A (este script, necesita  pip install roboflow):
    set ROBOFLOW_API_KEY=su_clave          (Windows cmd)
    $env:ROBOFLOW_API_KEY="su_clave"       (PowerShell)
    python etapas/e1_roboflow.py --workspace wiegon --proyecto garbage-classification-3-ssg5n --version 2 --formato yolov8

    En la página del dataset en universe.roboflow.com: el workspace y el proyecto
    salen de la URL (universe.roboflow.com/<workspace>/<proyecto>), y la versión
    del botón "Download Dataset". La clave está en Settings > API Keys de su cuenta.

Opción B (manual): descargue el ZIP desde la web (formato "Folder Structure"
para clasificación o "YOLOv8"/"COCO" para detección) y descomprímalo en
    datos/originales/roboflow/<nombre_que_quiera>/
y cree a mano el archivo PROCEDENCIA.json (copie el de otra descarga).

Formatos admitidos por la etapa 2: folder, yolov5/yolov8/yolov11 (cajas o
polígonos), coco, y el CSV multietiqueta de clasificación.
"""
import argparse
import getpass
import json
import os
from datetime import datetime, timezone

import _base  # noqa: F401
from ecosegrega.rutas import cargar_config, ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--proyecto", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--formato", default="folder",
                    help="folder (clasificación) | yolov8 | yolov11 | coco (detección)")
    ap.add_argument("--licencia", default="",
                    help="Anote la licencia que muestra la página (p. ej. 'CC BY 4.0')")
    ap.add_argument("--config", default="config/estudio.yaml")
    args = ap.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Falta el paquete 'roboflow':  pip install roboflow   (o use la opción B manual)")

    # La clave NUNCA se escribe en el código ni en el YAML (se subiría a git).
    clave = os.environ.get("ROBOFLOW_API_KEY") or getpass.getpass("Clave API de Roboflow: ")
    cfg = cargar_config(args.config)
    destino = ruta(cfg["datos"]["roboflow"]) / f"{args.proyecto}-v{args.version}"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and any(destino.iterdir()):
        raise SystemExit(f"Ya existe {destino}. Los datos crudos no se sobrescriben (cap. 7.2.5). "
                         "Bórrelo usted si de verdad quiere volver a descargar.")

    rf = Roboflow(api_key=clave)
    version = rf.workspace(args.workspace).project(args.proyecto).version(args.version)
    version.download(args.formato, location=str(destino))

    procedencia = {
        "fuente": "Roboflow Universe",
        "url": f"https://universe.roboflow.com/{args.workspace}/{args.proyecto}/dataset/{args.version}",
        "workspace": args.workspace, "proyecto": args.proyecto, "version": args.version,
        "formato": args.formato, "licencia": args.licencia or "COMPLETAR A MANO (ver la página del dataset)",
        "descargado_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(destino / "PROCEDENCIA.json", "w", encoding="utf-8") as f:
        json.dump(procedencia, f, ensure_ascii=False, indent=2)
    print(f"Listo: {destino}\nSiguiente:  python etapas/e2_inventario.py")


if __name__ == "__main__":
    main()
