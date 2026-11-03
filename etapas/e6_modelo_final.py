"""Etapa 6 — Modelo final y ÚNICA evaluación sobre la prueba apartada (caps. 9.5, 12, 15).

    python etapas/e6_modelo_final.py --equipo gpu
    python etapas/e6_modelo_final.py --equipo gpu --tratamiento "mobilenet_v3_large+con"

1. Toma el tratamiento ganador de la etapa 5 (o el que usted indique: p. ej. uno
   más barato si la diferencia no fue significativa).
2. Lo reentrena con TODOS los pliegues durante un número FIJO de épocas: la
   mediana de la mejor época en la validación cruzada. Así no se necesita mirar
   ningún conjunto para decidir cuándo parar.
3. Calibra temperatura y umbral "no estoy seguro" con las predicciones FUERA DE
   PLIEGUE de ese tratamiento (cap. 12.3.1: tres particiones, no dos).
4. Evalúa UNA vez en la prueba apartada y escribe resultados/final/informe_prueba.md.
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import _base  # noqa: F401
from ecosegrega import bitacora, hardware
from ecosegrega import contrastes as C
from ecosegrega import medidas as M
from ecosegrega.entrenador import Entrenador, ajustar_temperatura, softmax
from ecosegrega.norma import Catalogo
from ecosegrega.rutas import cargar_config, huella_archivo, ruta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/estudio.yaml")
    ap.add_argument("--equipo", default=None)
    ap.add_argument("--tratamiento", default=None)
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    equipo = hardware.elegir_equipo(args.equipo or cfg["equipo"])
    cat = Catalogo.desde_yaml(cfg["datos"]["norma"])
    estudio = ruta(cfg["salidas"]["carpeta"]) / "estudio"
    res = pd.read_csv(estudio / "resultados_pliegues.csv")
    huella_actual = huella_archivo(ruta(cfg["pliegues"]["archivo"]))
    if (res["huella_pliegues"] != huella_actual).any():
        raise SystemExit("resultados_pliegues.csv viene de pliegues distintos a los actuales: el estudio y la "
                         "prueba ya no corresponden. Vuelva a correr la etapa 4 (y la 5) con los pliegues actuales.")
    if args.tratamiento:
        sub = res[res.tratamiento == args.tratamiento]
        if sub.empty:
            raise SystemExit(f"'{args.tratamiento}' no está en resultados_pliegues.csv")
        g = {"tratamiento": args.tratamiento, "a": sub.a.iloc[0], "b": sub.b.iloc[0],
             "epocas_final": max(1, int(np.median(sub.mejor_epoca)))}
    else:
        archivo_g = estudio / "ganador.json"
        if not archivo_g.exists():
            raise SystemExit("Ejecute antes la etapa 5 (o indique --tratamiento).")
        g = json.loads(archivo_g.read_text(encoding="utf-8"))
    fb = cfg["factorial"]["factor_b"]
    archivo_pl = ruta(cfg["pliegues"]["archivo"])
    df = pd.read_csv(archivo_pl, dtype={"bloque": str})
    huella = huella_archivo(archivo_pl)
    pool = df[df.bloque != "prueba"].reset_index(drop=True)
    te = df[df.bloque == "prueba"].reset_index(drop=True)
    if fb["nombre"] == "fotos_propias" and g["b"] == "sin":
        pool = pool[pool.fuente != "propias"].reset_index(drop=True)
    aumento = cfg["aumentos"][g["b"]] if fb["nombre"] == "aumento" else cfg["aumentos"][cfg["aumentos"]["por_defecto"]]

    print("=" * 74)
    print(f"Modelo final: {g['tratamiento']} | {g['epocas_final']} épocas fijas | {hardware.describir(equipo)}")
    print(f"Entrenamiento: {len(pool)} imágenes (todos los pliegues) | prueba apartada: {len(te)} imágenes")
    print("=" * 74)
    hardware.sembrar(int(cfg["semilla"]), bool(cfg.get("determinista", False)))
    ent = Entrenador(cfg, cat.clases, equipo, g["a"], aumento)
    r = ent.ajustar(pool, None, epocas_fijas=int(g["epocas_final"]),
                    al_terminar_epoca=lambda f: print(f"   época {f['epoca']:2d} | pérdida {f['perdida_tr']:.4f} | "
                                                      f"exactitud {f['exactitud_tr']:.3f} | {f['segundos']:.0f}s"))

    # --- Calibración con predicciones fuera de pliegue (nunca con la prueba) ---
    K = int(cfg["pliegues"]["k"])
    oof = []
    for i in range(1, K + 1):
        f = estudio / "oof" / f"{g['tratamiento']}__p{i}.npz"
        if f.exists():
            with np.load(f) as d:
                if "huella" in d.files and str(d["huella"]) == huella:   # solo de ESTOS pliegues
                    oof.append({"logits": d["logits"], "y": d["y"]})
    if oof:
        Z = np.concatenate([d["logits"] for d in oof])
        Y = np.concatenate([d["y"] for d in oof])
        T = ajustar_temperatura(Z, Y)
        umbral = M.umbral_confianza(softmax(Z, T), Y, float(cfg["decision"]["precision_objetivo"]))
    else:
        T, umbral = 1.0, 0.0
        print("[AVISO] No hay predicciones fuera de pliegue: sin calibración.")
    print(f"Temperatura (fuera de pliegue): {T:.3f} | umbral 'no estoy seguro': {umbral:.3f}")

    # --- La única consulta a la prueba ---
    final = ruta(cfg["salidas"]["carpeta"]) / "final"
    final.mkdir(parents=True, exist_ok=True)
    log = final / "consultas_prueba.log"
    if log.exists() and log.read_text(encoding="utf-8").strip():
        print("[AVISO] La prueba ya fue consultada antes (ver consultas_prueba.log). Si cambió algo del modelo "
              "por lo que vio ahí, la prueba dejó de ser ciega (cap. 9.5). Declárelo en el informe.")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{g['tratamiento']}\t{huella[:16]}\n")
    z, y = ent.predecir(te)
    pr_c, pr = softmax(z), softmax(z, T)
    p = pr.argmax(1)
    n = len(cat.clases)
    grupos = te["grupo"].to_numpy()
    resumen = M.resumen(y, p, n)
    f1m = C.intervalo_bootstrap(M.f1_macro_fn(n), y, p, grupos)
    exa = C.intervalo_bootstrap(lambda a, b: float((a == b).mean()), y, p, grupos)
    tabla = M.tabla_por_clase(y, p, cat.clases)
    esquema = cfg["decision"]["esquema"]
    tachos = {e: float((np.array(cat.tachos(e))[y] == np.array(cat.tachos(e))[p]).mean())
              for e in ("no_municipal", "municipal")}
    seguro = pr.max(1) >= umbral
    por_fuente = pd.DataFrame([{"fuente": fu, "imagenes": int(m.sum()),
                                "exactitud": round(float((p[m] == y[m]).mean()), 4),
                                "f1_macro": round(M.resumen(y[m], p[m], n)["f1_macro"], 4)}
                               for fu in sorted(te.fuente.unique()) for m in [(te.fuente == fu).to_numpy()]])
    M.figura_confusion(M.matriz(y, p, n), cat.clases, final / "confusion_clases.png", f"Prueba · {g['tratamiento']}")
    colores = sorted(set(cat.tachos(esquema)))
    mt = np.array(cat.tachos(esquema))
    M.figura_confusion(M.matriz(np.array([colores.index(c) for c in mt[y]]),
                                np.array([colores.index(c) for c in mt[p]]), len(colores)),
                       colores, final / "confusion_tachos.png", f"Prueba por tacho ({esquema})")
    preds = te[["ruta", "grupo", "fuente"]].copy()
    preds["real"] = [cat.clases[i] for i in y]
    preds["predicho"] = [cat.clases[i] for i in p]
    preds["confianza"] = pr.max(1).round(5)
    preds["acierto"] = (p == y).astype(int)
    preds["seguro"] = seguro.astype(int)
    preds.to_csv(final / "predicciones_prueba.csv", index=False)
    ent.red.guardar(final / "modelo_final.pt", cat.clases, {
        "temperatura": T, "umbral_confianza": umbral, "tam_imagen": int(cfg["entrenamiento"]["tam_imagen"]),
        "tratamiento": g["tratamiento"], "epocas": int(g["epocas_final"]), "huella_pliegues": huella,
        "tachos": {c: cat.info[c]["tacho"] for c in cat.clases}})

    md = [f"# Evaluación final en la prueba apartada — {g['tratamiento']}", "",
          f"- Red `{g['a']}` (pesos `{ent.red.pesos}`), factor B = {fb['nombre']}:{g['b']}, "
          f"{g['epocas_final']} épocas fijas, entrenada con {len(pool)} imágenes en {r['duracion_s'] / 60:.1f} min",
          f"- Prueba: {len(te)} imágenes de {te.grupo.nunique()} grupos; nunca usada antes de esta etapa", "",
          "| Métrica | Valor | IC 95 % (bootstrap por grupo) |", "|---|---|---|",
          f"| F1 macro | {resumen['f1_macro']:.4f} | [{f1m[1]:.4f}, {f1m[2]:.4f}] |",
          f"| Exactitud | {resumen['exactitud']:.4f} | [{exa[1]:.4f}, {exa[2]:.4f}] |",
          f"| Exactitud balanceada | {resumen['exactitud_balanceada']:.4f} | |",
          f"| Exactitud de tacho (no municipal, 7 colores) | {tachos['no_municipal']:.4f} | |",
          f"| Exactitud de tacho (municipal, 4 colores) | {tachos['municipal']:.4f} | |",
          f"| Cobertura del umbral {umbral:.2f} | {100 * seguro.mean():.1f} % | |",
          f"| Exactitud en casos 'seguros' | {float((p[seguro] == y[seguro]).mean()) if seguro.any() else float('nan'):.4f} | |",
          f"| ECE sin calibrar → calibrado | {M.ece(pr_c, y):.4f} → {M.ece(pr, y):.4f} | |", "",
          "## Por clase", "", M.a_markdown(tabla), "",
          "## Por fuente (¿funciona con fotos tomadas en Perú?)", "", M.a_markdown(por_fuente), "",
          f"Media en validación cruzada para este tratamiento: "
          f"{res[res.tratamiento == g['tratamiento']]['f1_macro'].mean():.4f} (F1 macro). Esa cifra es algo "
          "optimista: la parada temprana eligió la época mirando el mismo pliegue que se reporta. La cifra que "
          "se puede prometer es la de esta prueba (cap. 5.3 y 10.4). Si la prueba sale mucho peor, sospeche "
          "además sobreajuste a los pliegues o diferencia de dominio (cap. 5.4).", "",
          "![](confusion_clases.png)", "", "![](confusion_tachos.png)", ""]
    (final / "informe_prueba.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:16]))
    print(tabla.to_string(index=False))
    bitacora.anotar(cfg, "final", g["tratamiento"], {"epocas": g["epocas_final"], "red": g["a"], "b": g["b"]},
                    {**resumen, "f1_macro_ic95": f1m[1:], "temperatura": T, "umbral": umbral,
                     "tacho_no_municipal": tachos["no_municipal"]}, huella, str(final), equipo)
    print(f"\nModelo: {final / 'modelo_final.pt'}\nSiguiente:  python etapas/e7_clasificar.py fotos/")


if __name__ == "__main__":
    main()
