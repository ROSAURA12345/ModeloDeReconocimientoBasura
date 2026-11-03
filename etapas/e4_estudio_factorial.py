"""Etapa 4 — El estudio factorial con validación cruzada (caps. 10 y 22).

    python etapas/e4_estudio_factorial.py --equipo gpu
    python etapas/e4_estudio_factorial.py --equipo cpu
    python etapas/e4_estudio_factorial.py --equipo gpu --tratamiento "efficientnet_b0+con"

Para cada tratamiento (combinación de niveles de A y B) y cada pliegue k:
entrena con los otros K-1 pliegues, valida en el pliegue k con parada temprana
y guarda una fila en resultados/estudio/resultados_pliegues.csv.

* Mismos pliegues y misma semilla por pliegue para todos los tratamientos:
  diseño aparejado (cap. 10.5). Lo único que cambia entre tratamientos es el
  factor que se estudia.
* Factor B = fotos_propias: con "sin" se quitan las fotos propias SOLO del
  entrenamiento; el pliegue de validación es idéntico en ambos niveles.
* Se puede interrumpir (Ctrl+C, apagón, sesión de Colab) y volver a lanzar: lo
  ya hecho se salta (cap. 1.10, "escriba los guiones de modo que puedan reanudar").
* Corra UN proceso a la vez sobre la misma carpeta de resultados (dos en paralelo
  pueden mezclar las filas del CSV).
* Guarda las predicciones FUERA DE PLIEGUE de cada tratamiento: la etapa 6 las
  usa para calibrar sin tocar la prueba (cap. 12.3.1).
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import _base  # noqa: F401
from ecosegrega import bitacora, hardware, pliegues
from ecosegrega.entrenador import Entrenador, f1_macro
from ecosegrega.norma import Catalogo
from ecosegrega.rutas import cargar_config, guardar_yaml, huella_archivo, ruta


def tratamientos(cfg):
    fa, fb = cfg["factorial"]["factor_a"], cfg["factorial"]["factor_b"]
    return [(a, b, f"{a}+{b}") for a in fa["niveles"] for b in fb["niveles"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/estudio.yaml")
    ap.add_argument("--equipo", default=None, help="auto | gpu | cpu")
    ap.add_argument("--tratamiento", default=None, help="correr solo este, p. ej. 'efficientnet_b0+con'")
    ap.add_argument("--pliegue", type=int, default=None, help="correr solo este pliegue")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    equipo = hardware.elegir_equipo(args.equipo or cfg["equipo"])
    fb = cfg["factorial"]["factor_b"]
    if cfg["factorial"]["factor_a"]["nombre"] != "arquitectura":
        raise SystemExit("En esta versión el factor A debe ser 'arquitectura'.")
    if fb["nombre"] not in ("fotos_propias", "aumento"):
        raise SystemExit("El factor B debe ser 'fotos_propias' o 'aumento'.")
    cat = Catalogo.desde_yaml(cfg["datos"]["norma"])
    archivo_pl = ruta(cfg["pliegues"]["archivo"])
    if not archivo_pl.exists():
        raise SystemExit("Faltan los pliegues: ejecute las etapas 2 y 3.")
    df = pd.read_csv(archivo_pl, dtype={"bloque": str})
    huella = huella_archivo(archivo_pl)
    K = int(cfg["pliegues"]["k"])
    if fb["nombre"] == "fotos_propias" and not (df["fuente"] == "propias").any():
        raise SystemExit("El factor B es 'fotos_propias' pero el inventario no tiene fotos propias. "
                         "Agréguelas (etapas 2 y 3) o cambie factor_b a 'aumento' en config/estudio.yaml.")

    salida = ruta(cfg["salidas"]["carpeta"]) / "estudio"
    (salida / "oof").mkdir(parents=True, exist_ok=True)
    (salida / "historiales").mkdir(exist_ok=True)
    guardar_yaml({k: v for k, v in cfg.items() if not k.startswith("_")}, salida / "config_usada.yaml")
    tabla = salida / "resultados_pliegues.csv"
    previos = pd.read_csv(tabla) if tabla.exists() else pd.DataFrame()
    if len(previos) and (previos["huella_pliegues"] != huella).any():
        raise SystemExit("resultados_pliegues.csv viene de OTROS pliegues. Muévalo o bórrelo antes de seguir.")
    # Huella de la receta de entrenamiento: si cambia el YAML a mitad del estudio, no se mezclan resultados
    receta = json.dumps({"entrenamiento": cfg["entrenamiento"], "aumentos": cfg["aumentos"],
                         "factorial": cfg["factorial"], "semilla": cfg["semilla"]}, sort_keys=True, default=str)
    huella_receta = hashlib.sha256(receta.encode()).hexdigest()[:16]
    if len(previos) and "huella_receta" in previos and (previos["huella_receta"] != huella_receta).any():
        raise SystemExit("Cambió la configuración de entrenamiento desde que empezó el estudio. Termine con la "
                         "configuración original, o mueva resultados/estudio/ y empiece de nuevo (cap. 16.2).")
    hechos = set(zip(previos.get("tratamiento", []), previos.get("pliegue", [])))

    print("=" * 74)
    print(f"Estudio factorial en: {hardware.describir(equipo)}")
    print(f"Tratamientos: {[t for *_, t in tratamientos(cfg)]} | K = {K} | ya hechos: {len(hechos)}")
    print("=" * 74)
    for a, b, nombre in tratamientos(cfg):
        if args.tratamiento and nombre != args.tratamiento:
            continue
        for i in range(1, K + 1):
            if args.pliegue and i != args.pliegue:
                continue
            if (nombre, i) in hechos:
                print(f"[saltado] {nombre} pliegue {i} (ya está)")
                continue
            tr, va = pliegues.pliegue(df, i)
            if fb["nombre"] == "fotos_propias" and b == "sin":
                tr = tr[tr["fuente"] != "propias"].reset_index(drop=True)
            aumento = cfg["aumentos"][b] if fb["nombre"] == "aumento" else cfg["aumentos"][cfg["aumentos"]["por_defecto"]]
            hardware.sembrar(int(cfg["semilla"]) + i, bool(cfg.get("determinista", False)))
            print(f"\n>>> {nombre} | pliegue {i}/{K} | train {len(tr)} img | val {len(va)} img")
            ent = Entrenador(cfg, cat.clases, equipo, a, aumento)

            def mostrar(f):
                print(f"   época {f['epoca']:2d} | tr {f['perdida_tr']:.3f}/{f['exactitud_tr']:.3f} | "
                      f"va {f.get('perdida_va', float('nan')):.3f} F1m {f.get('f1_macro_va', float('nan')):.3f} | "
                      f"lr {f['lr_cabeza']:.1e} | {f['segundos']:.0f}s ({f['img_s']:.0f} img/s)")

            res = ent.ajustar(tr, va, al_terminar_epoca=mostrar)
            z, y = ent.predecir(va)
            p = z.argmax(1)
            prop = (va["fuente"] == "propias").to_numpy()
            fila = {"tratamiento": nombre, "a": a, "b": b, "pliegue": i,
                    "f1_macro": f1_macro(y, p, len(cat.clases)), "exactitud": float((p == y).mean()),
                    "f1_macro_propias": f1_macro(y[prop], p[prop], len(cat.clases)) if prop.any() else np.nan,
                    "exactitud_propias": float((p[prop] == y[prop]).mean()) if prop.any() else np.nan,
                    "n_train": len(tr), "n_val": len(va), "mejor_epoca": res["mejor_epoca"],
                    "duracion_s": round(res["duracion_s"], 1),
                    "img_s": round(float(res["historial"]["img_s"].mean()), 1),
                    "entrenables": res["entrenables"], "equipo": equipo.type, "huella_pliegues": huella,
                    "huella_receta": huella_receta}
            np.savez_compressed(salida / "oof" / f"{nombre}__p{i}.npz", logits=z, y=y,
                                rutas=va["ruta"].to_numpy().astype(str), huella=np.array(huella))
            res["historial"].to_csv(salida / "historiales" / f"{nombre}__p{i}.csv", index=False)
            pd.DataFrame([fila]).to_csv(tabla, mode="a", header=not tabla.exists(), index=False)
            bitacora.anotar(cfg, "pliegue", f"{nombre}__p{i}",
                            {"red": a, "factor_b": f"{fb['nombre']}={b}", "pliegue": i,
                             "estrategia": cfg["entrenamiento"]["estrategia"]},
                            {k: fila[k] for k in ("f1_macro", "exactitud", "f1_macro_propias", "mejor_epoca",
                                                  "duracion_s", "img_s")},
                            huella, str(salida), equipo)
            print(f"<<< {nombre} pliegue {i}: F1 macro {fila['f1_macro']:.4f} | exactitud {fila['exactitud']:.4f}"
                  + (f" | F1 en fotos propias {fila['f1_macro_propias']:.4f}" if prop.any() else ""))
    final = pd.read_csv(tabla) if tabla.exists() else pd.DataFrame()
    if len(final):
        print("\nResumen (media ± desv. por tratamiento):")
        print(final.groupby("tratamiento")[["f1_macro", "exactitud", "duracion_s"]].agg(["mean", "std"]).round(4))
        faltan = len(tratamientos(cfg)) * K - len(final)
        print(f"\nCeldas completas: {len(final)} | faltan: {faltan}")
        (salida / "estado.json").write_text(json.dumps({"completas": int(len(final)), "faltan": int(faltan)}),
                                            encoding="utf-8")
    print("Siguiente:  python etapas/e5_analisis_estadistico.py")


if __name__ == "__main__":
    main()
