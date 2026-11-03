"""Etapa 2 — Inventario: Roboflow + fotos propias, validado y deduplicado (caps. 7 y 8).

    python etapas/e2_inventario.py

Traduce etiquetas a las clases de config/ntp_900058.yaml, recorta cajas de los
datasets de detección, valida cada imagen (lo que falla se APARTA con su motivo,
nunca se borra) y une en un mismo grupo las fotos duplicadas o casi idénticas.
"""
import argparse

import _base  # noqa: F401
from ecosegrega import inventario
from ecosegrega.rutas import cargar_config, ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/estudio.yaml")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    inv, apart, info = inventario.construir(cfg)
    texto = inventario.resumen(inv, apart, info)
    print(texto)
    meta = inventario.guardar(inv, apart, info, cfg)
    (ruta(cfg["datos"]["inventario"]).parent / "resumen_inventario.txt").write_text(texto, encoding="utf-8")
    n_prop = int((inv["fuente"] == "propias").sum()) if len(inv) else 0
    if cfg["factorial"]["factor_b"]["nombre"] == "fotos_propias" and n_prop == 0:
        print("\n[AVISO] El factor B es 'fotos_propias' pero no hay fotos propias. Agréguelas en "
              "datos/originales/propias/<clase>/<objeto>/ o cambie el factor B a 'aumento'.")
    print(f"\nHuella del inventario: {meta['huella_inventario'][:16]}...")
    print("Siguiente:  python etapas/e3_pliegues.py")


if __name__ == "__main__":
    main()
