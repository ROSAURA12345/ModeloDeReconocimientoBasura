"""Etapa 3 — Apartar la prueba y armar K pliegues agrupados y estratificados (caps. 9 y 10).

    python etapas/e3_pliegues.py            # primera vez
    python etapas/e3_pliegues.py --forzar   # rehacer (invalida el estudio ya corrido)

Se hace UNA sola vez. Todos los tratamientos del factorial usarán los mismos
pliegues: eso convierte la comparación en un diseño aparejado (cap. 10.5).
"""
import argparse
import json

import pandas as pd

import _base  # noqa: F401
from ecosegrega import inventario, pliegues
from ecosegrega.rutas import cargar_config, huella_archivo, ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/estudio.yaml")
    ap.add_argument("--forzar", action="store_true")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    pc = cfg["pliegues"]
    destino = ruta(pc["archivo"])
    if destino.exists() and not args.forzar:
        raise SystemExit(f"Ya existen los pliegues ({destino}). Si los rehace con --forzar, borre "
                         "también resultados/estudio/ y vuelva a correr la etapa 4.")
    inv = pd.read_csv(ruta(cfg["datos"]["inventario"]))
    if inv.empty:
        raise SystemExit("Inventario vacío: ejecute antes la etapa 2.")
    inv = inventario.submuestrear(inv, cfg["datos"].get("submuestra_grupos_por_clase"), int(cfg["semilla"]))
    df = pliegues.construir(inv, float(pc["p_prueba"]), int(pc["k"]), pc.get("estratos", ["clase"]),
                            int(cfg["semilla"]))
    destino.parent.mkdir(parents=True, exist_ok=True)
    df[["ruta", "clase", "fuente", "grupo", "bloque"]].to_csv(destino, index=False, encoding="utf-8")
    texto = pliegues.informe(df)
    print(texto)
    destino.with_name(f"informe_{destino.stem}.txt").write_text(texto, encoding="utf-8")
    meta = {"huella_pliegues": huella_archivo(destino), "k": int(pc["k"]), "p_prueba": float(pc["p_prueba"]),
            "semilla": int(cfg["semilla"])}
    destino.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Huella de los pliegues: {meta['huella_pliegues'][:16]}...")
    print("Siguiente:  python etapas/e4_estudio_factorial.py --equipo gpu")


if __name__ == "__main__":
    main()
