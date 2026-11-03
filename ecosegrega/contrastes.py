"""Inferencia estadística sobre las métricas por pliegue (caps. 17, 18 y 22). Solo scipy.

Entrada típica: una tabla larga con columnas  a, b, pliegue, y  (un valor de la
métrica por tratamiento y pliegue). Como TODOS los tratamientos se evaluaron en
los MISMOS pliegues, el pliegue es un BLOQUE (medidas repetidas, cap. 17.2): la
dificultad propia de cada pliegue se resta antes de comparar tratamientos.

Advertencia del cap. 17.6 que hay que declarar en el informe: los pliegues
comparten datos de entrenamiento, así que no son del todo independientes y los
valores p tienden a salir algo pequeños. Por eso se reportan también tamaños
del efecto e intervalos, no solo p.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalores) -> np.ndarray:
    p = np.asarray(pvalores, float)
    m = len(p)
    ajust, acum = np.empty(m), 0.0
    for r, i in enumerate(np.argsort(p)):
        acum = max(acum, min((m - r) * p[i], 1.0))
        ajust[i] = acum
    return ajust


def descriptivos(largo: pd.DataFrame) -> pd.DataFrame:
    return (largo.groupby("tratamiento")["y"].agg(["count", "mean", "std", "min", "max"])
            .sort_values("mean", ascending=False))


def supuestos(largo: pd.DataFrame) -> dict:
    """Shapiro-Wilk por tratamiento y Levene entre tratamientos (cap. 18.2.2 / 17.6)."""
    grupos = {t: g["y"].to_numpy() for t, g in largo.groupby("tratamiento")}
    shapiro = {}
    for t, v in grupos.items():
        if len(v) >= 3 and np.ptp(v) > 0:
            w, p = stats.shapiro(v)
            shapiro[t] = (float(w), float(p))
        else:
            shapiro[t] = (float("nan"), float("nan"))
    vals = [v for v in grupos.values() if np.ptp(v) > 0]
    lev = stats.levene(*grupos.values()) if len(vals) >= 2 else None
    return {"shapiro": shapiro, "levene": (float(lev.statistic), float(lev.pvalue)) if lev else (np.nan, np.nan)}


def anova_una_via(largo: pd.DataFrame) -> dict:
    """ANOVA de una vía sobre los tratamientos + eta^2 (código 18.1 del libro)."""
    grupos = [g["y"].to_numpy() for _, g in largo.groupby("tratamiento")]
    F, p = stats.f_oneway(*grupos)
    todos = np.concatenate(grupos)
    m = todos.mean()
    sc_entre = sum(len(g) * (g.mean() - m) ** 2 for g in grupos)
    sc_total = ((todos - m) ** 2).sum()
    return {"F": float(F), "p": float(p), "gl": (len(grupos) - 1, len(todos) - len(grupos)),
            "eta2": float(sc_entre / sc_total) if sc_total > 0 else 0.0}


def tukey(largo: pd.DataFrame) -> pd.DataFrame:
    """HSD de Tukey (cap. 18.4) con scipy.stats.tukey_hsd: diferencias, IC simultáneo y p."""
    nombres = sorted(largo["tratamiento"].unique())
    grupos = [largo.loc[largo.tratamiento == n, "y"].to_numpy() for n in nombres]
    r = stats.tukey_hsd(*grupos)
    ic = r.confidence_interval(0.95)
    filas = []
    for i, j in itertools.combinations(range(len(nombres)), 2):
        filas.append({"grupo_1": nombres[i], "grupo_2": nombres[j],
                      "dif_medias(2-1)": round(float(grupos[j].mean() - grupos[i].mean()), 4),
                      "ic_inf": round(float(-ic.high[i, j]), 4), "ic_sup": round(float(-ic.low[i, j]), 4),
                      "p_ajust": float(r.pvalue[i, j]), "rechaza_H0": bool(r.pvalue[i, j] < 0.05)})
    return pd.DataFrame(filas)


def friedman_wilcoxon(largo: pd.DataFrame) -> dict:
    """Ruta no paramétrica para medidas repetidas (cap. 18.5): Friedman global y
    Wilcoxon por pares con Holm. Con K=5 pliegues el menor p posible de Wilcoxon
    (dos colas) es 0,0625: con tan pocos pliegues esta ruta casi nunca rechaza."""
    ancho = largo.pivot(index="pliegue", columns="tratamiento", values="y").dropna()
    res = {"friedman": (np.nan, np.nan), "pares": pd.DataFrame()}
    if ancho.shape[1] >= 3 and ancho.shape[0] >= 2:
        chi, p = stats.friedmanchisquare(*[ancho[c].to_numpy() for c in ancho.columns])
        res["friedman"] = (float(chi), float(p))
    filas = []
    for a, b in itertools.combinations(ancho.columns, 2):
        d = ancho[a] - ancho[b]
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(ancho[a], ancho[b]).pvalue)
        filas.append({"par": f"{a} vs {b}", "dif_mediana": round(float(d.median()), 4), "p": p})
    if filas:
        t = pd.DataFrame(filas)
        t["p_holm"] = holm(t["p"])
        res["pares"] = t
    return res


def factorial_bloques(largo: pd.DataFrame) -> pd.DataFrame:
    """ANOVA factorial a x b en bloques completos al azar (bloque = pliegue).

        y = media + bloque + A + B + AxB + error

    Fórmulas cerradas para diseño balanceado (todas las celdas con los mismos K
    pliegues). Devuelve la tabla con SC, gl, CM, F, p y eta^2 parcial (cap. 22.2).
    """
    d = largo.copy()
    completo = d.groupby("pliegue")["tratamiento"].nunique()
    n_trat = d["tratamiento"].nunique()
    d = d[d["pliegue"].isin(completo[completo == n_trat].index)]
    a_niv, b_niv, r_niv = sorted(d.a.unique()), sorted(d.b.unique()), sorted(d.pliegue.unique())
    a, b, r = len(a_niv), len(b_niv), len(r_niv)
    if a < 2 or b < 2 or r < 2:
        raise ValueError("Se necesitan al menos 2 niveles por factor y 2 pliegues completos")
    m = d["y"].mean()
    ma = d.groupby("a")["y"].mean()
    mb = d.groupby("b")["y"].mean()
    mab = d.groupby(["a", "b"])["y"].mean()
    mr = d.groupby("pliegue")["y"].mean()
    sc_a = b * r * ((ma - m) ** 2).sum()
    sc_b = a * r * ((mb - m) ** 2).sum()
    sc_ab = r * sum((mab[(i, j)] - ma[i] - mb[j] + m) ** 2 for i in a_niv for j in b_niv)
    sc_r = a * b * ((mr - m) ** 2).sum()
    sc_t = ((d["y"] - m) ** 2).sum()
    sc_e = max(sc_t - sc_a - sc_b - sc_ab - sc_r, 0.0)
    gl = {"A": a - 1, "B": b - 1, "AxB": (a - 1) * (b - 1), "bloque (pliegue)": r - 1,
          "error": (a * b - 1) * (r - 1)}
    sc = {"A": sc_a, "B": sc_b, "AxB": sc_ab, "bloque (pliegue)": sc_r, "error": sc_e}
    cme = sc_e / gl["error"] if gl["error"] > 0 else np.nan
    filas = []
    for t in ["A", "B", "AxB", "bloque (pliegue)", "error"]:
        cm = sc[t] / gl[t]
        if t == "error":
            filas.append({"termino": t, "SC": sc[t], "gl": gl[t], "CM": cm, "F": np.nan, "p": np.nan,
                          "eta2_parcial": np.nan})
            continue
        F = cm / cme if cme and cme > 0 else np.inf
        p = float(stats.f.sf(F, gl[t], gl["error"])) if np.isfinite(F) else 0.0
        filas.append({"termino": t, "SC": sc[t], "gl": gl[t], "CM": cm, "F": F, "p": p,
                      "eta2_parcial": sc[t] / (sc[t] + sc_e) if (sc[t] + sc_e) > 0 else 0.0})
    return pd.DataFrame(filas)


def efectos_2x2(largo: pd.DataFrame) -> dict:
    """Efectos principales y de interacción como en el cap. 22.2 (solo para 2 x 2)."""
    mab = largo.groupby(["a", "b"])["y"].mean()
    a1, a2 = sorted(largo.a.unique())
    b1, b2 = sorted(largo.b.unique(), key=_orden_b)
    ef_a = (mab[(a2, b1)] + mab[(a2, b2)]) / 2 - (mab[(a1, b1)] + mab[(a1, b2)]) / 2
    ef_b = (mab[(a1, b2)] + mab[(a2, b2)]) / 2 - (mab[(a1, b1)] + mab[(a2, b1)]) / 2
    inter = (mab[(a2, b2)] - mab[(a2, b1)]) - (mab[(a1, b2)] - mab[(a1, b1)])
    return {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "efecto_A": float(ef_a), "efecto_B": float(ef_b),
            "interaccion": float(inter)}


def _orden_b(v: str):
    """'sin' antes que 'con' y 'basico' antes que 'fuerte'; el resto alfabético."""
    return {"sin": 0, "con": 1, "basico": 0, "fuerte": 1}.get(str(v), 2), str(v)


def intervalo_bootstrap(fn, y_real, y_pred, grupos, n: int = 1000, confianza: float = 0.95,
                        semilla: int = 0):
    """IC remuestreando GRUPOS (fotos de un mismo objeto no son evidencias independientes)."""
    y_real, y_pred, grupos = map(np.asarray, (y_real, y_pred, grupos))
    est = fn(y_real, y_pred)
    unicos, inv = np.unique(grupos, return_inverse=True)
    idx = [np.nonzero(inv == k)[0] for k in range(len(unicos))]
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[k] for k in rng.integers(0, len(unicos), len(unicos))])
        vals.append(fn(y_real[sel], y_pred[sel]))
    a = (1 - confianza) / 2
    return float(est), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
