"""Etapa 5 — Análisis estadístico del estudio factorial (caps. 17, 18 y 22). No usa torch.

    python etapas/e5_analisis_estadistico.py
    python etapas/e5_analisis_estadistico.py --respuesta f1_macro_propias

Con la métrica de cada tratamiento en cada pliegue:
  1. Descriptivos y costo (cap. 16.4: comparar con el costo delante)
  2. Supuestos: Shapiro-Wilk y Levene (cap. 17.6)
  3. ANOVA factorial A x B con el pliegue como bloque + eta^2 parcial (cap. 22.2)
     y efectos principales / interacción calculados como en el libro
  4. ANOVA de una vía sobre los tratamientos + HSD de Tukey con IC (cap. 18.3-18.4)
  5. Ruta no paramétrica: Friedman + Wilcoxon con Holm (cap. 18.5)
  6. Gráfico de interacción, cajas y "líneas por pliegue" (muestra el apareamiento)
Escribe resultados/estudio/informe_estadistico.md y ganador.json.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _base  # noqa: F401
from ecosegrega import contrastes as C
from ecosegrega.medidas import a_markdown
from ecosegrega.rutas import cargar_config, ruta


def figuras(largo, cfg, carpeta, respuesta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa, fb = cfg["factorial"]["factor_a"]["nombre"], cfg["factorial"]["factor_b"]["nombre"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    niveles_b = sorted(largo.b.unique(), key=C._orden_b)
    for a_niv, g in largo.groupby("a"):
        medias = g.groupby("b")["y"].mean().reindex(niveles_b)
        err = g.groupby("b")["y"].std().reindex(niveles_b)
        ax[0].errorbar(niveles_b, medias.values, yerr=err.values, marker="o", capsize=4, label=a_niv)
    ax[0].set_title(f"Interacción {fa} x {fb}\n(rectas no paralelas = interacción, cap. 22.2)")
    ax[0].set_xlabel(fb)
    ax[0].set_ylabel(respuesta)
    ax[0].legend()
    orden = largo.groupby("tratamiento")["y"].mean().sort_values().index
    ax[1].boxplot([largo[largo.tratamiento == t]["y"] for t in orden], showmeans=True)
    ax[1].set_xticks(range(1, len(orden) + 1), orden, rotation=20, ha="right", fontsize=8)
    ax[1].set_title("Distribución por tratamiento (K pliegues)")
    ancho = largo.pivot(index="pliegue", columns="tratamiento", values="y")[orden]
    for pl, fila in ancho.iterrows():
        ax[2].plot(range(len(orden)), fila.values, marker="o", alpha=0.6, label=f"pliegue {pl}")
    ax[2].set_xticks(range(len(orden)), orden, rotation=20, ha="right", fontsize=8)
    ax[2].set_title("Mismo pliegue = misma línea (diseño aparejado, cap. 17.2)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(carpeta / "figuras_estudio.png", dpi=130)
    plt.close(fig)


def fmt_p(p):
    return "< 0,001" if p < 0.001 else f"= {p:.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/estudio.yaml")
    ap.add_argument("--respuesta", default=None, help="f1_macro | exactitud | f1_macro_propias")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    carpeta = ruta(cfg["salidas"]["carpeta"]) / "estudio"
    tabla = carpeta / "resultados_pliegues.csv"
    if not tabla.exists():
        raise SystemExit("No hay resultados: ejecute la etapa 4.")
    res = pd.read_csv(tabla)
    respuesta = args.respuesta or cfg["factorial"]["respuesta"]
    largo = res.rename(columns={respuesta: "y"})[["tratamiento", "a", "b", "pliegue", "y"]].dropna()
    n_trat = largo.tratamiento.nunique()
    completos = largo.groupby("pliegue")["tratamiento"].nunique()
    if (completos < n_trat).any() or n_trat < 2:
        print(f"[AVISO] Hay pliegues incompletos: {completos.to_dict()}. Se analizan solo los completos.")
        largo = largo[largo.pliegue.isin(completos[completos == n_trat].index)]
    if largo.pliegue.nunique() < 2:
        raise SystemExit("Se necesitan al menos 2 pliegues completos para el análisis.")
    fa, fb = cfg["factorial"]["factor_a"]["nombre"], cfg["factorial"]["factor_b"]["nombre"]
    K = largo.pliegue.nunique()

    desc = C.descriptivos(largo)
    costo = res.groupby("tratamiento")[["duracion_s", "img_s", "mejor_epoca", "entrenables"]].mean().round(1)
    sup = C.supuestos(largo)
    fact = C.factorial_bloques(largo)
    una = C.anova_una_via(largo)
    tk = C.tukey(largo)
    npar = C.friedman_wilcoxon(largo)
    figuras(largo, cfg, carpeta, respuesta)

    md = [f"# Informe estadístico — {cfg['estudio']}", "",
          f"Respuesta: **{respuesta}** · {n_trat} tratamientos · K = {K} pliegues completos · "
          f"factor A = {fa}, factor B = {fb}.", "",
          "## 1. Descriptivos y costo", "",
          a_markdown(desc.round(4).reset_index()), "", a_markdown(costo.reset_index()), "",
          "## 2. Supuestos", ""]
    for t, (w, p) in sup["shapiro"].items():
        md.append(f"- Shapiro-Wilk {t}: W = {w:.3f}, p {fmt_p(p) if np.isfinite(p) else '= n/d'}"
                  + ("" if not np.isfinite(p) or p > 0.05 else "  ← normalidad dudosa"))
    lw, lp = sup["levene"]
    md.append(f"- Levene: W = {lw:.3f}, p {fmt_p(lp) if np.isfinite(lp) else '= n/d'}")
    normal = all((not np.isfinite(p)) or p > 0.05 for _, p in sup["shapiro"].values())
    md += ["", f"Con K = {K} las pruebas de supuestos tienen poca potencia: que no rechacen no "
           "prueba que se cumplan. Por eso se reporta también la ruta no paramétrica.", "",
           "## 3. ANOVA factorial con bloques (pliegue)", "",
           a_markdown(fact.assign(SC=fact.SC.round(6), CM=fact.CM.round(6), F=fact.F.round(3),
                                  p=fact.p.map(lambda v: "" if pd.isna(v) else f"{v:.4g}"),
                                  eta2_parcial=fact.eta2_parcial.round(3))), ""]
    ef = None
    if largo.a.nunique() == 2 and largo.b.nunique() == 2:
        ef = C.efectos_2x2(largo)
        md += [f"- Efecto principal de A ({ef['a1']} → {ef['a2']}): {ef['efecto_A']:+.4f}",
               f"- Efecto principal de B ({ef['b1']} → {ef['b2']}): {ef['efecto_B']:+.4f}",
               f"- Interacción (ganancia de B con {ef['a2']} menos ganancia de B con {ef['a1']}): "
               f"{ef['interaccion']:+.4f}", ""]
    p_inter = float(fact.loc[fact.termino == "AxB", "p"].iloc[0])
    if p_inter < 0.05:
        md.append("**La interacción es significativa**: los efectos principales no se interpretan solos; "
                  "describa el efecto de cada factor condicionado al nivel del otro (cap. 22.2).")
    md += ["", "## 4. ANOVA de una vía sobre los tratamientos y Tukey HSD", "",
           f"F({una['gl'][0]}, {una['gl'][1]}) = {una['F']:.3f}, p {fmt_p(una['p'])}, η² = {una['eta2']:.3f}", "",
           a_markdown(tk.assign(p_ajust=tk.p_ajust.map(lambda v: f"{v:.4g}"))), "",
           "Los intervalos que no contienen el cero son diferencias significativas (cap. 18.4).", "",
           "## 5. Ruta no paramétrica (medidas repetidas)", ""]
    chi, pf = npar["friedman"]
    if np.isfinite(chi):
        md.append(f"Friedman: χ² = {chi:.3f}, p {fmt_p(pf)}")
    if len(npar["pares"]):
        md += ["", a_markdown(npar["pares"].assign(p=npar["pares"].p.round(4), p_holm=npar["pares"].p_holm.round(4)))]
        if K <= 5:
            md.append(f"\nCon K = {K} el menor p posible de Wilcoxon es {2 / 2 ** K:.4f}: esta ruta casi no "
                      "puede rechazar. Considere K = 10 (cap. 10.2.1).")

    ganador = desc.index[0]
    fila_g = res[res.tratamiento == ganador].iloc[0]
    epocas_final = int(np.median(res.loc[res.tratamiento == ganador, "mejor_epoca"]))
    p_a = float(fact.loc[fact.termino == "A", "p"].iloc[0])
    p_b = float(fact.loc[fact.termino == "B", "p"].iloc[0])
    eta_a = float(fact.loc[fact.termino == "A", "eta2_parcial"].iloc[0])
    eta_b = float(fact.loc[fact.termino == "B", "eta2_parcial"].iloc[0])
    gle = int(fact.loc[fact.termino == "error", "gl"].iloc[0])
    fa_v = float(fact.loc[fact.termino == "A", "F"].iloc[0])
    fb_v = float(fact.loc[fact.termino == "B", "F"].iloc[0])
    fi_v = float(fact.loc[fact.termino == "AxB", "F"].iloc[0])
    gl_a = int(fact.loc[fact.termino == "A", "gl"].iloc[0])
    gl_b = int(fact.loc[fact.termino == "B", "gl"].iloc[0])
    gl_i = int(fact.loc[fact.termino == "AxB", "gl"].iloc[0])
    md += ["", "## 6. Redacción sugerida (formato del cap. 18.3.1)", "",
           f"> Se evaluó un diseño factorial {largo.a.nunique()} × {largo.b.nunique()} ({fa} × {fb}) con "
           f"validación cruzada agrupada y estratificada de {K} pliegues, empleando el pliegue como bloque. "
           f"El efecto de {fa} sobre el {respuesta} fue F({gl_a}, {gle}) = {fa_v:.2f}, p {fmt_p(p_a)}, η²p = {eta_a:.3f}; "
           f"el de {fb}, F({gl_b}, {gle}) = {fb_v:.2f}, p {fmt_p(p_b)}, η²p = {eta_b:.3f}; y la interacción, "
           f"F({gl_i}, {gle}) = {fi_v:.2f}, p {fmt_p(p_inter)}. "
           f"El tratamiento con mayor media fue {ganador} ({desc.loc[ganador, 'mean']:.4f} ± "
           f"{desc.loc[ganador, 'std']:.4f}).", "",
           "Limitación a declarar (cap. 17.6): los pliegues comparten datos de entrenamiento, por lo que los "
           "valores p pueden ser algo optimistas. Además, la parada temprana elige la época con el mismo "
           "pliegue que se reporta, de modo que las medias absolutas están sesgadas al alza (el sesgo es igual "
           "para todos los tratamientos, así que la comparación sigue siendo justa). Las conclusiones se apoyan "
           "también en los tamaños del efecto y en la evaluación final sobre la prueba apartada (etapa 6).", "",
           f"Supuesto de normalidad: {'sin evidencia en contra' if normal else 'dudoso; prefiera la ruta no paramétrica'}.",
           "", "![](figuras_estudio.png)", ""]
    (carpeta / "informe_estadistico.md").write_text("\n".join(md), encoding="utf-8")
    json.dump({"tratamiento": ganador, "a": fila_g["a"], "b": fila_g["b"], "epocas_final": max(1, epocas_final),
               "media": float(desc.loc[ganador, "mean"]), "respuesta": respuesta},
              open(carpeta / "ganador.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(md))
    print(f"\nGanador por media: {ganador} -> resultados/estudio/ganador.json")
    print("Siguiente:  python etapas/e6_modelo_final.py --equipo gpu")


if __name__ == "__main__":
    main()
