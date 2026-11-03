"""Etapa 8 — ¿En qué se fija el modelo? Grad-CAM++ sobre aciertos Y fallos (cap. 21).

    python etapas/e8_mapas_gradcampp.py --n 5

Usa las predicciones de la prueba (etapa 6). Arriba: los aciertos más seguros;
abajo: los fallos más seguros, que es donde está el diagnóstico útil (cap. 21.4).
También corre la prueba de aleatorización del modelo (cap. 21.2): se destruye la
cabeza y luego cada etapa, y la correlación de Spearman con el mapa original
debe caer hacia cero. Si no cae, el mapa no está explicando al modelo.
Recuerde: el mapa no es un contorno ni sirve para medir el objeto (cap. 21.1).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _base  # noqa: F401
from ecosegrega import hardware, lectura
from ecosegrega.fotos import abrir_rgb
from ecosegrega.redes import Red, gradcam_pp, superponer
from ecosegrega.rutas import cargar_config, ruta


def reiniciar(modulos):
    for mod in modulos:
        for m in mod.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--equipo", default="auto")
    ap.add_argument("--config", default="config/estudio.yaml")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cargar_config(args.config)
    final = ruta(cfg["salidas"]["carpeta"]) / "final"
    preds = pd.read_csv(final / "predicciones_prueba.csv")
    equipo = hardware.elegir_equipo(args.equipo)
    red, ck = Red.cargar(final / "modelo_final.pt", equipo)
    clases = ck["clases"]
    _, t = lectura.recetas(red.receta, int(ck["tam_imagen"]), None)
    vista = lambda im: t.transforms[1](t.transforms[0](im))  # noqa: E731  (lo mismo que ve la red)
    aciertos = preds[preds.acierto == 1].nlargest(args.n, "confianza")
    fallos = preds[preds.acierto == 0].nlargest(args.n, "confianza")
    cols = max(1, args.n)
    fig, ejes = plt.subplots(2, cols, figsize=(3.2 * cols, 7.2), squeeze=False)
    for fila, (titulo, d) in enumerate([("Aciertos", aciertos), ("Fallos", fallos)]):
        for c in range(cols):
            ax = ejes[fila, c]
            ax.axis("off")
            if c >= len(d):
                continue
            r = d.iloc[c]
            im = abrir_rgb(ruta(r["ruta"]))
            mapa, k, pr = gradcam_pp(red.modelo, red.capa_mapa, t(im).unsqueeze(0).to(equipo))
            ax.imshow(superponer(vista(im), mapa))
            ax.set_title(f"dice {clases[k]} ({pr:.2f})\nes {r['real']}", fontsize=9,
                         color="black" if r["acierto"] else "crimson")
        ejes[fila, 0].text(-0.12, 0.5, titulo, transform=ejes[fila, 0].transAxes, rotation=90,
                           ha="right", va="center", fontsize=12)
    fig.suptitle("Grad-CAM++ sobre la prueba (el mapa no es un contorno, cap. 21.1)")
    fig.tight_layout()
    fig.savefig(final / "gradcampp_aciertos_fallos.png", dpi=130)
    plt.close(fig)

    filas = []
    for _, r in pd.concat([aciertos.head(2), fallos.head(1)]).iterrows():
        im = abrir_rgb(ruta(r["ruta"]))
        x = t(im).unsqueeze(0).to(equipo)
        copia, _ = Red.cargar(final / "modelo_final.pt", equipo)
        original, k, _ = gradcam_pp(copia.modelo, copia.capa_mapa, x)
        pasos = [("cabeza", [copia.cabeza])] + [(f"etapa -{i + 1}", e) for i, e in enumerate(reversed(copia.etapas))]
        for nombre, mods in pasos:
            reiniciar(mods)
            m, _, _ = gradcam_pp(copia.modelo, copia.capa_mapa, x, clase=k)
            rho = float(np.nan_to_num(spearmanr(original.ravel(), m.ravel()).statistic)) if m.std() > 0 else 0.0
            filas.append({"imagen": r["ruta"], "aleatorizado_hasta": nombre, "spearman": round(rho, 3)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(final / "aleatorizacion_gradcampp.csv", index=False)
    print("Correlación media con el mapa original al destruir pesos (debe caer hacia 0):")
    print(tabla.groupby("aleatorizado_hasta", sort=False)["spearman"].mean().round(3).to_string())
    print(f"\nFiguras en {final}")


if __name__ == "__main__":
    main()
